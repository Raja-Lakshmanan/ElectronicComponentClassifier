"""
export_v5_weights.py
--------------------
Phase 2, Step 2.2 — Export trained V5-Spatial model parameters for FPGA/HLS.

Reads the checkpoint at checkpoints/v5_spatial_best_model.pth, extracts every
trained weight and bias tensor, and writes them into fpga_export/v5_spatial/ as:
  - NumPy .npy files  (preserving original shape and float32 values)
  - Human-readable .txt files (shape header + flat values for inspection)
  - manifest.json (complete metadata: architecture, checkpoint provenance,
    normalization, layer inventory, class mapping, deployment notes)

Post-export verification: each saved .npy file is reloaded and compared against
the source checkpoint tensor for exact shape and numerical equality.

Usage:
  python export_v5_weights.py
"""

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

# ── Project paths ─────────────────────────────────────────────────────────
PROJECT_DIR    = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "v5_spatial_best_model.pth"
EXPORT_DIR     = PROJECT_DIR / "fpga_export" / "v5_spatial"
MANIFEST_PATH  = EXPORT_DIR / "manifest.json"

# ── Architecture import (read-only) ──────────────────────────────────────
sys.path.insert(0, str(PROJECT_DIR))
from model_v5_spatial import ComponentCNNSpatial, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE

SEP  = "=" * 82
SEP2 = "-" * 82

# Human-friendly names for each parameter tensor in the state dict.
# Order matches model.state_dict() iteration order.
LAYER_DESCRIPTORS = {
    "features.0.weight": {"layer": "Conv Block 1", "desc": "Conv2d(3->16, k=3, p=1) weight"},
    "features.0.bias":   {"layer": "Conv Block 1", "desc": "Conv2d(3->16, k=3, p=1) bias"},
    "features.3.weight": {"layer": "Conv Block 2", "desc": "Conv2d(16->32, k=3, p=1) weight"},
    "features.3.bias":   {"layer": "Conv Block 2", "desc": "Conv2d(16->32, k=3, p=1) bias"},
    "features.6.weight": {"layer": "Conv Block 3", "desc": "Conv2d(32->64, k=3, p=1) weight"},
    "features.6.bias":   {"layer": "Conv Block 3", "desc": "Conv2d(32->64, k=3, p=1) bias"},
    "classifier.0.weight": {"layer": "FC Hidden",   "desc": "Linear(256->64) weight"},
    "classifier.0.bias":   {"layer": "FC Hidden",   "desc": "Linear(256->64) bias"},
    "classifier.2.weight": {"layer": "FC Output",   "desc": "Linear(64->10) weight"},
    "classifier.2.bias":   {"layer": "FC Output",   "desc": "Linear(64->10) bias"},
}


def load_checkpoint() -> dict:
    """Load the V5-Spatial checkpoint and return the full dictionary."""
    if not CHECKPOINT_PATH.exists():
        print(f"\nERROR: Checkpoint not found at: {CHECKPOINT_PATH}")
        sys.exit(1)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    # Sanity checks on checkpoint metadata
    assert "model_state_dict" in checkpoint, "Checkpoint missing 'model_state_dict' key"
    assert checkpoint.get("num_classes") == NUM_CLASSES, (
        f"num_classes mismatch: checkpoint={checkpoint.get('num_classes')}, model={NUM_CLASSES}"
    )
    assert checkpoint.get("class_names") == CLASS_NAMES, "class_names mismatch"

    return checkpoint


def load_and_verify_model(state_dict: dict) -> ComponentCNNSpatial:
    """Instantiate ComponentCNNSpatial and load the state dict."""
    model = ComponentCNNSpatial(num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_tensor_npy(tensor: np.ndarray, path: Path) -> None:
    """Save a numpy array to .npy format."""
    np.save(path, tensor)


def save_tensor_txt(
    tensor: np.ndarray,
    path: Path,
    param_key: str,
    layer_desc: str,
) -> None:
    """Save a human-readable text file with shape header and flattened values."""
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Parameter : {param_key}\n")
        f.write(f"# Layer     : {layer_desc}\n")
        f.write(f"# Shape     : {list(tensor.shape)}\n")
        f.write(f"# Dtype     : {tensor.dtype}\n")
        f.write(f"# Elements  : {tensor.size}\n")
        f.write(f"# Min       : {tensor.min():.8f}\n")
        f.write(f"# Max       : {tensor.max():.8f}\n")
        f.write(f"# Mean      : {tensor.mean():.8f}\n")
        f.write("#\n")
        f.write("# Flat values (row-major / C-order):\n")
        flat = tensor.flatten()
        for i, val in enumerate(flat):
            f.write(f"{val:.8f}\n")


def verify_exported_tensor(
    original: np.ndarray, npy_path: Path, param_key: str
) -> bool:
    """Reload the .npy file and verify exact shape and numerical equality."""
    reloaded = np.load(npy_path)

    # Shape check
    if reloaded.shape != original.shape:
        print(f"  FAIL [{param_key}]: Shape mismatch — "
              f"original={original.shape}, reloaded={reloaded.shape}")
        return False

    # Numerical equality (exact, not approximate — no quantization yet)
    if not np.array_equal(reloaded, original):
        max_diff = np.abs(reloaded - original).max()
        print(f"  FAIL [{param_key}]: Values differ — max absolute diff = {max_diff}")
        return False

    return True


def build_manifest(checkpoint: dict, layer_entries: list) -> dict:
    """Build the manifest.json dictionary with full metadata."""
    return {
        "model": {
            "architecture": checkpoint.get("architecture", "ComponentCNNSpatial"),
            "spatial_pool_size": list(checkpoint.get("spatial_pool_size", (2, 2))),
            "num_classes": checkpoint.get("num_classes", NUM_CLASSES),
            "total_parameters": 40682,
        },
        "checkpoint": {
            "filename": CHECKPOINT_PATH.name,
            "best_epoch": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "val_macro_f1": checkpoint.get("val_macro_f1"),
        },
        "input": {
            "shape": [1, 3, INPUT_SIZE, INPUT_SIZE],
            "channels": "RGB",
            "height": INPUT_SIZE,
            "width": INPUT_SIZE,
            "normalization": {
                "mean": list(checkpoint.get("norm_mean", (0.5, 0.5, 0.5))),
                "std": list(checkpoint.get("norm_std", (0.5, 0.5, 0.5))),
                "pixel_range_after_normalize": [-1.0, 1.0],
            },
        },
        "layers": layer_entries,
        "class_mapping": {
            str(i): name for i, name in enumerate(CLASS_NAMES)
        },
        "deployment_notes": {
            "transistor_index": 9,
            "transistor_handling": (
                "The transistor output at index 9 is masked to negative infinity "
                "at inference time during application-level prediction. It is NOT "
                "removed from the trained network architecture. All 10 output "
                "logits remain structurally present in the exported weights."
            ),
            "permitted_classes": CLASS_NAMES[:9],
        },
        "export": {
            "format": "float32 NumPy (.npy) + human-readable (.txt)",
            "quantization": "none (raw float32, pre-quantization export)",
            "export_script": "export_v5_weights.py",
        },
    }


def main() -> None:
    print(SEP)
    print("  Phase 2, Step 2.2 — Export V5-Spatial Model Parameters")
    print(SEP)

    # ── Step 1: Load checkpoint ───────────────────────────────────────────
    print("\n[1/5] Loading checkpoint...")
    checkpoint = load_checkpoint()
    state_dict = checkpoint["model_state_dict"]
    print(f"  Checkpoint : {CHECKPOINT_PATH.name}")
    print(f"  Epoch      : {checkpoint.get('epoch')}")
    print(f"  Val Acc    : {checkpoint.get('val_accuracy', 0):.2f}%")
    print(f"  Val F1     : {checkpoint.get('val_macro_f1', 0):.4f}")
    print(f"  Tensors    : {len(state_dict)}")

    # ── Step 2: Verify model loads correctly ──────────────────────────────
    print("\n[2/5] Verifying model instantiation...")
    model = load_and_verify_model(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model      : ComponentCNNSpatial")
    print(f"  Parameters : {total_params:,}")

    # Quick forward-pass sanity check
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, NUM_CLASSES), f"Forward pass shape error: {out.shape}"
    print(f"  Forward OK : input (1,3,64,64) -> output {tuple(out.shape)}")

    # ── Step 3: Create export directory and save tensors ──────────────────
    print(f"\n[3/5] Exporting tensors to: {EXPORT_DIR}")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    layer_entries = []
    exported_count = 0
    total_elements = 0
    total_bytes = 0

    print(f"\n  {'Parameter Key':<28} {'Shape':<20} {'Elements':>10} {'Bytes':>10}")
    print(f"  {SEP2}")

    for param_key, tensor_pt in state_dict.items():
        tensor_np = tensor_pt.cpu().numpy()
        descriptor = LAYER_DESCRIPTORS.get(param_key, {})
        layer_name = descriptor.get("layer", param_key)
        layer_desc = descriptor.get("desc", param_key)

        # Filenames: replace dots with underscores for filesystem safety
        safe_name = param_key.replace(".", "_")
        npy_filename = f"{safe_name}.npy"
        txt_filename = f"{safe_name}.txt"
        npy_path = EXPORT_DIR / npy_filename
        txt_path = EXPORT_DIR / txt_filename

        # Save .npy
        save_tensor_npy(tensor_np, npy_path)

        # Save .txt
        save_tensor_txt(tensor_np, txt_path, param_key, layer_desc)

        # Track statistics
        n_elements = tensor_np.size
        n_bytes = n_elements * 4  # float32 = 4 bytes
        total_elements += n_elements
        total_bytes += n_bytes
        exported_count += 1

        print(f"  {param_key:<28} {str(list(tensor_np.shape)):<20} {n_elements:>10,} {n_bytes:>10,}")

        # Build layer manifest entry
        layer_entries.append({
            "param_key": param_key,
            "layer": layer_name,
            "description": layer_desc,
            "shape": list(tensor_np.shape),
            "dtype": str(tensor_np.dtype),
            "elements": n_elements,
            "bytes_float32": n_bytes,
            "npy_file": npy_filename,
            "txt_file": txt_filename,
        })

    print(f"  {SEP2}")
    print(f"  {'TOTAL':<28} {'':<20} {total_elements:>10,} {total_bytes:>10,}")

    # ── Step 4: Write manifest.json ───────────────────────────────────────
    print(f"\n[4/5] Writing manifest.json...")
    manifest = build_manifest(checkpoint, layer_entries)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {MANIFEST_PATH.name}")

    # ── Step 5: Verification pass ─────────────────────────────────────────
    print(f"\n[5/5] Verifying exported tensors against checkpoint...")
    all_passed = True
    for param_key, tensor_pt in state_dict.items():
        original_np = tensor_pt.cpu().numpy()
        safe_name = param_key.replace(".", "_")
        npy_path = EXPORT_DIR / f"{safe_name}.npy"

        passed = verify_exported_tensor(original_np, npy_path, param_key)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {param_key:<28} shape={list(original_np.shape)}")

        if not passed:
            all_passed = False

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Export Summary")
    print(f"{SEP}")
    print(f"  Output directory     : {EXPORT_DIR.resolve()}")
    print(f"  Exported tensors     : {exported_count}")
    print(f"  Total parameters     : {total_elements:,}")
    print(f"  Float32 payload      : {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    print(f"  Manifest             : {MANIFEST_PATH.name}")
    print(f"  Files created        : {exported_count * 2 + 1} ({exported_count} .npy + {exported_count} .txt + 1 .json)")
    print(f"  Verification         : {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    print(f"  Checkpoint modified  : No")
    print(f"  Quantization applied : No (raw float32)")
    print(f"{SEP}")

    if not all_passed:
        print("\nERROR: One or more verification checks failed. See details above.")
        sys.exit(1)
    else:
        print("\nPhase 2, Step 2.2 complete. All tensors exported and verified.")


if __name__ == "__main__":
    main()
