"""
create_hls_test_data.py
-----------------------
Generates test data tensors and metadata for hls4ml and C-simulation verification
of the V5-Spatial (ComponentCNNSpatial) model.

Outputs generated under fpga_export/v5_spatial/test_data/:
  - X_test.npy: preprocessed FP32 inputs with shape (346, 3, 64, 64), dtype float32
  - y_test.npy: ground-truth class indices with shape (346,), dtype int64
  - y_test_logits_fp32.npy: original PyTorch model's raw FP32 logits with shape (346, 10), dtype float32
  - y_test_preds.npy: predicted class indices with shape (346,), dtype int64 (excluding class 9)
  - test_data_metadata.json: complete metadata manifest

Requirements satisfied:
  1. Exact deterministic preprocessing from evaluate_v5_quantization.py.
  2. Uses existing V5-Spatial checkpoint without retraining or modifying weights.
  3. Excludes transistor class (9 permitted classes: 0..8).
  4. Exact file names, shapes, and dtypes.
  5. Preserves deterministic sample ordering.
  6. Saves metadata JSON with class mapping, paths, shapes, dtypes, and normalization.
  7. Validates generated files by reloading with NumPy (NaN/Inf checks, shapes, labels, accuracy).
  8. Prints final output paths, shapes, dtypes, and test accuracy.
  9. Does NOT overwrite existing files without explicit confirmation.
  10. Does NOT modify model, checkpoint, dataset, or existing evaluation scripts.
"""

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root setup
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from model_v5_spatial import ComponentCNNSpatial, NUM_CLASSES, INPUT_SIZE, PERMITTED_CLASSES
from evaluate_v5_quantization import (
    NonTransistorDataset,
    get_eval_transforms,
    build_test_loader,
    load_model,
    compute_metrics,
    CHECKPOINT_PATH,
    TEST_DIR,
    NORM_MEAN,
    NORM_STD,
    TRANSISTOR_INDEX,
    NUM_PERMITTED,
)

OUTPUT_DIR = PROJECT_DIR / "fpga_export" / "v5_spatial" / "test_data"
X_TEST_PATH = OUTPUT_DIR / "X_test.npy"
Y_TEST_PATH = OUTPUT_DIR / "y_test.npy"
LOGITS_PATH = OUTPUT_DIR / "y_test_logits_fp32.npy"
PREDS_PATH = OUTPUT_DIR / "y_test_preds.npy"
METADATA_PATH = OUTPUT_DIR / "test_data_metadata.json"

TARGET_FILES = [X_TEST_PATH, Y_TEST_PATH, LOGITS_PATH, PREDS_PATH, METADATA_PATH]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("  Creating HLS Test Data for V5-Spatial (ComponentCNNSpatial)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Check for Existing Files (Requirement 9: Do not overwrite)
    # -------------------------------------------------------------------------
    existing_files = [p for p in TARGET_FILES if p.exists()]
    if existing_files:
        print("\n[ERROR] The following target file(s) already exist:")
        for p in existing_files:
            print(f"  - {p}")
        print("\nAborting execution to prevent overwriting existing files without explicit confirmation.")
        print("To proceed, remove or back up the existing files before running this script.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nTarget directory: {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # 2. Check Prerequisites
    # -------------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        sys.exit(f"ERROR: Checkpoint file not found: {CHECKPOINT_PATH}")
    if not TEST_DIR.exists():
        sys.exit(f"ERROR: Test dataset directory not found: {TEST_DIR}")

    device = torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model, ckpt = load_model(device)

    # -------------------------------------------------------------------------
    # 3. Load Test Dataset in Deterministic Order
    # -------------------------------------------------------------------------
    print(f"Loading test images from: {TEST_DIR}")
    loader, ds = build_test_loader()
    num_samples = len(ds)
    print(f"Loaded {num_samples} test samples across {len(PERMITTED_CLASSES)} permitted classes.")
    print(f"Permitted classes: {PERMITTED_CLASSES}")
    print(f"Excluded class: transistor (index {TRANSISTOR_INDEX})")

    # -------------------------------------------------------------------------
    # 4. Generate Preprocessed Inputs, Ground-Truth, Logits, and Predictions
    # -------------------------------------------------------------------------
    print("\nRunning inference on test dataset...")
    all_inputs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_raw_logits: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            # images: (B, 3, 64, 64) float32 normalized
            # labels: (B,) int64 in [0, 8]
            images_cpu = images.to(device)
            raw_logits = model(images_cpu)  # (B, 10) raw FP32 logits

            # Mask transistor (index 9) for prediction
            masked_logits = raw_logits[:, :TRANSISTOR_INDEX]  # columns 0..8
            preds = masked_logits.argmax(dim=1)  # class index in [0, 8]

            all_inputs.append(images.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_raw_logits.append(raw_logits.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    inference_duration = time.time() - start_time
    print(f"Inference complete in {inference_duration:.2f}s.")

    # Concatenate into full arrays
    X_test = np.concatenate(all_inputs, axis=0).astype(np.float32)
    y_test = np.concatenate(all_labels, axis=0).astype(np.int64)
    y_test_logits_fp32 = np.concatenate(all_raw_logits, axis=0).astype(np.float32)
    y_test_preds = np.concatenate(all_preds, axis=0).astype(np.int64)

    # -------------------------------------------------------------------------
    # 5. Sanity Checks before Saving
    # -------------------------------------------------------------------------
    assert X_test.shape == (346, 3, 64, 64), f"Unexpected X_test shape: {X_test.shape}"
    assert X_test.dtype == np.float32, f"Unexpected X_test dtype: {X_test.dtype}"
    assert y_test.shape == (346,), f"Unexpected y_test shape: {y_test.shape}"
    assert y_test.dtype == np.int64, f"Unexpected y_test dtype: {y_test.dtype}"
    assert y_test_logits_fp32.shape == (346, 10), f"Unexpected logits shape: {y_test_logits_fp32.shape}"
    assert y_test_logits_fp32.dtype == np.float32, f"Unexpected logits dtype: {y_test_logits_fp32.dtype}"
    assert y_test_preds.shape == (346,), f"Unexpected preds shape: {y_test_preds.shape}"
    assert y_test_preds.dtype == np.int64, f"Unexpected preds dtype: {y_test_preds.dtype}"

    metrics = compute_metrics(y_test_preds.tolist(), y_test.tolist())
    accuracy = metrics["accuracy"]
    correct_count = metrics["correct"]
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}% ({correct_count}/{num_samples})")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 6. Save Target Numpy Arrays
    # -------------------------------------------------------------------------
    print("\nSaving target numpy files...")
    np.save(X_TEST_PATH, X_test)
    print(f"  Saved: {X_TEST_PATH} (shape: {X_test.shape}, dtype: {X_test.dtype})")

    np.save(Y_TEST_PATH, y_test)
    print(f"  Saved: {Y_TEST_PATH} (shape: {y_test.shape}, dtype: {y_test.dtype})")

    np.save(LOGITS_PATH, y_test_logits_fp32)
    print(f"  Saved: {LOGITS_PATH} (shape: {y_test_logits_fp32.shape}, dtype: {y_test_logits_fp32.dtype})")

    np.save(PREDS_PATH, y_test_preds)
    print(f"  Saved: {PREDS_PATH} (shape: {y_test_preds.shape}, dtype: {y_test_preds.dtype})")

    # -------------------------------------------------------------------------
    # 7. Build and Save Metadata Manifest
    # -------------------------------------------------------------------------
    print("\nBuilding metadata manifest...")
    sample_records = []
    for idx, (path_str, label) in enumerate(ds.samples):
        img_path = Path(path_str)
        rel_path = img_path.relative_to(TEST_DIR).as_posix()
        pred = int(y_test_preds[idx])
        is_corr = bool(pred == label)
        sample_records.append({
            "index": idx,
            "filename": img_path.name,
            "relative_path": rel_path,
            "class_name": PERMITTED_CLASSES[label],
            "ground_truth_label": int(label),
            "predicted_label": pred,
            "is_correct": is_corr,
        })

    class_mapping = {cls: idx for idx, cls in enumerate(PERMITTED_CLASSES)}

    metadata = {
        "title": "V5-Spatial HLS Test Data Manifest",
        "generated_at": datetime.now().isoformat(),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_DIR)),
        "dataset_root": str(TEST_DIR.relative_to(PROJECT_DIR)),
        "sample_count": num_samples,
        "class_mapping": class_mapping,
        "permitted_classes": PERMITTED_CLASSES,
        "excluded_class": "transistor",
        "excluded_class_index": TRANSISTOR_INDEX,
        "preprocessing": {
            "resize": [INPUT_SIZE, INPUT_SIZE],
            "interpolation": "bilinear",
            "color_mode": "RGB",
            "normalize_mean": list(NORM_MEAN),
            "normalize_std": list(NORM_STD),
            "pixel_value_range": [-1.0, 1.0],
            "layout": "NCHW (3, 64, 64)",
        },
        "evaluation_metrics": {
            "total_samples": num_samples,
            "correct_predictions": correct_count,
            "accuracy_pct": accuracy,
            "macro_f1": metrics["macro_f1"],
        },
        "files": {
            "X_test.npy": {
                "path": str(X_TEST_PATH.relative_to(PROJECT_DIR)),
                "description": "Preprocessed normalized FP32 input images (NCHW)",
                "shape": list(X_test.shape),
                "dtype": str(X_test.dtype),
                "min": float(X_test.min()),
                "max": float(X_test.max()),
                "size_bytes": os.path.getsize(X_TEST_PATH),
                "sha256": compute_sha256(X_TEST_PATH),
            },
            "y_test.npy": {
                "path": str(Y_TEST_PATH.relative_to(PROJECT_DIR)),
                "description": "Ground-truth class indices (0..8)",
                "shape": list(y_test.shape),
                "dtype": str(y_test.dtype),
                "min": int(y_test.min()),
                "max": int(y_test.max()),
                "size_bytes": os.path.getsize(Y_TEST_PATH),
                "sha256": compute_sha256(Y_TEST_PATH),
            },
            "y_test_logits_fp32.npy": {
                "path": str(LOGITS_PATH.relative_to(PROJECT_DIR)),
                "description": "Unmasked raw FP32 logits output from PyTorch model (10 classes)",
                "shape": list(y_test_logits_fp32.shape),
                "dtype": str(y_test_logits_fp32.dtype),
                "min": float(y_test_logits_fp32.min()),
                "max": float(y_test_logits_fp32.max()),
                "size_bytes": os.path.getsize(LOGITS_PATH),
                "sha256": compute_sha256(LOGITS_PATH),
            },
            "y_test_preds.npy": {
                "path": str(PREDS_PATH.relative_to(PROJECT_DIR)),
                "description": "Predicted class indices (0..8) excluding transistor column",
                "shape": list(y_test_preds.shape),
                "dtype": str(y_test_preds.dtype),
                "min": int(y_test_preds.min()),
                "max": int(y_test_preds.max()),
                "size_bytes": os.path.getsize(PREDS_PATH),
                "sha256": compute_sha256(PREDS_PATH),
            },
        },
        "samples": sample_records,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata manifest: {METADATA_PATH} ({os.path.getsize(METADATA_PATH)} bytes)")

    # -------------------------------------------------------------------------
    # 8. Reload and Validate Generated Files (Requirement 7)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  Validating Saved Files (NumPy Reload & Integrity Check)")
    print("=" * 80)

    # 8a. Reload X_test
    X_reloaded = np.load(X_TEST_PATH)
    assert X_reloaded.shape == (346, 3, 64, 64), f"Shape mismatch in reloaded X_test: {X_reloaded.shape}"
    assert X_reloaded.dtype == np.float32, f"Dtype mismatch in reloaded X_test: {X_reloaded.dtype}"
    assert not np.isnan(X_reloaded).any(), "NaN found in X_test!"
    assert not np.isinf(X_reloaded).any(), "Inf found in X_test!"
    assert np.array_equal(X_reloaded, X_test), "Content mismatch in reloaded X_test!"
    print(f"  [OK] X_test.npy: shape={X_reloaded.shape}, dtype={X_reloaded.dtype}, NaN/Inf=None, range=[{X_reloaded.min():.4f}, {X_reloaded.max():.4f}]")

    # 8b. Reload y_test
    y_reloaded = np.load(Y_TEST_PATH)
    assert y_reloaded.shape == (346,), f"Shape mismatch in reloaded y_test: {y_reloaded.shape}"
    assert y_reloaded.dtype == np.int64, f"Dtype mismatch in reloaded y_test: {y_reloaded.dtype}"
    assert not np.isnan(y_reloaded).any(), "NaN found in y_test!"
    assert set(np.unique(y_reloaded)).issubset(set(range(NUM_PERMITTED))), f"Invalid labels in y_test: {np.unique(y_reloaded)}"
    assert np.array_equal(y_reloaded, y_test), "Content mismatch in reloaded y_test!"
    print(f"  [OK] y_test.npy: shape={y_reloaded.shape}, dtype={y_reloaded.dtype}, valid classes=[{y_reloaded.min()}..{y_reloaded.max()}]")

    # 8c. Reload y_test_logits_fp32
    logits_reloaded = np.load(LOGITS_PATH)
    assert logits_reloaded.shape == (346, 10), f"Shape mismatch in reloaded logits: {logits_reloaded.shape}"
    assert logits_reloaded.dtype == np.float32, f"Dtype mismatch in reloaded logits: {logits_reloaded.dtype}"
    assert not np.isnan(logits_reloaded).any(), "NaN found in logits!"
    assert not np.isinf(logits_reloaded).any(), "Inf found in logits!"
    assert np.array_equal(logits_reloaded, y_test_logits_fp32), "Content mismatch in reloaded logits!"
    print(f"  [OK] y_test_logits_fp32.npy: shape={logits_reloaded.shape}, dtype={logits_reloaded.dtype}, NaN/Inf=None, range=[{logits_reloaded.min():.4f}, {logits_reloaded.max():.4f}]")

    # 8d. Reload y_test_preds
    preds_reloaded = np.load(PREDS_PATH)
    assert preds_reloaded.shape == (346,), f"Shape mismatch in reloaded preds: {preds_reloaded.shape}"
    assert preds_reloaded.dtype == np.int64, f"Dtype mismatch in reloaded preds: {preds_reloaded.dtype}"
    assert not np.isnan(preds_reloaded).any(), "NaN found in preds!"
    assert set(np.unique(preds_reloaded)).issubset(set(range(NUM_PERMITTED))), f"Invalid classes in preds: {np.unique(preds_reloaded)}"
    assert np.array_equal(preds_reloaded, y_test_preds), "Content mismatch in reloaded preds!"
    print(f"  [OK] y_test_preds.npy: shape={preds_reloaded.shape}, dtype={preds_reloaded.dtype}, valid classes=[{preds_reloaded.min()}..{preds_reloaded.max()}]")

    # 8e. Verify prediction derivation from logits
    derived_preds = logits_reloaded[:, :TRANSISTOR_INDEX].argmax(axis=1)
    assert np.array_equal(derived_preds, preds_reloaded), "Derived predictions from logits do not match y_test_preds!"
    print("  [OK] Prediction agreement: 100.0% match between derived logit argmax and y_test_preds.npy")

    # 8f. Verify accuracy reproduction
    reloaded_correct = int((preds_reloaded == y_reloaded).sum())
    reloaded_acc = 100.0 * reloaded_correct / len(y_reloaded)
    assert reloaded_correct == 238, f"Expected 238 correct predictions, got {reloaded_correct}"
    print(f"  [OK] Ground-truth match: {reloaded_correct}/{len(y_reloaded)} correct ({reloaded_acc:.4f}%)")

    # 8g. Verify metadata JSON
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta_reloaded = json.load(f)
    assert meta_reloaded["sample_count"] == 346
    assert len(meta_reloaded["samples"]) == 346
    assert meta_reloaded["evaluation_metrics"]["correct_predictions"] == 238
    print("  [OK] test_data_metadata.json: valid JSON structure and sample records verified.")

    # -------------------------------------------------------------------------
    # 9. Final Summary Table (Requirement 8)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY OF GENERATED HLS TEST DATA")
    print("=" * 80)
    print(f"{'File':<26} | {'Shape':<16} | {'Dtype':<10} | {'Size (KB)':<10} | {'SHA256 (first 10)':<18}")
    print("-" * 88)
    for name, info in metadata["files"].items():
        shape_str = str(tuple(info["shape"]))
        size_kb = info["size_bytes"] / 1024.0
        sha_prefix = info["sha256"][:10] + "..."
        print(f"{name:<26} | {shape_str:<16} | {info['dtype']:<10} | {size_kb:<10.1f} | {sha_prefix:<18}")
    print("-" * 88)
    print(f"Metadata Manifest: {METADATA_PATH}")
    print(f"Total Test Samples: {num_samples}")
    print(f"Test Set Accuracy:  {accuracy:.4f}% ({correct_count}/{num_samples})")
    print(f"Macro F1 Score:     {metrics['macro_f1']:.4f}")
    print("=" * 80)
    print("All validation checks passed successfully.")


if __name__ == "__main__":
    main()
