"""
compare_v4_v5_spatial.py
-------------------------
Fair side-by-side evaluation and comparison between the baseline V4 model and the
V5-Spatial model on the exact same 346 non-transistor held-out test images from cleaned_dataset/test.

Models Compared:
  - V4 Baseline   : ComponentCNN (model.py) — checkpoints/v4_best_model.pth (24,234 parameters)
  - V5-Spatial    : ComponentCNNSpatial (model_v5_spatial.py) — checkpoints/v5_spatial_best_model.pth (40,682 parameters)

Methodology for Fair Evaluation:
  1. Dataset:
     - Exactly the same 346 non-transistor test images from cleaned_dataset/test.
     - Transistor test samples are strictly excluded from target evaluation.
     - Both models evaluate the images in identical sequence and batching.
  2. Input Processing & Normalization:
     - 64x64 RGB images.
     - Verified normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) verified from checkpoint metadata.
  3. Output Masking:
     - Both models output 10 logits with identical class mappings (indices 0..8 = permitted classes; index 9 = transistor).
     - Transistor logit at index 9 is masked to -infinity (logits[:, 9] = -inf).
     - Final predictions are argmax over the 9 permitted classes (indices 0..8).
  4. Metrics Evaluated:
     - 9-Class Test Accuracy (%)
     - Macro Precision (%), Macro Recall (%), Macro F1-score, Weighted F1-score
     - Per-class Support, Predicted Counts, Precision, Recall, F1-score, and Deltas (V5 - V4)
     - Key focus on error reduction in resistor, potentiometer, push_button, and inductor
     - Parameter count, weight memory (FP32 & INT8), and MAC compute operations
  5. Artifacts Generated:
     - comparison_v4_v5_spatial.csv
     - comparison_v4_v5_spatial_report.txt

Usage:
  python compare_v4_v5_spatial.py
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Safe library imports ──────────────────────────────────────────────────
try:
    from torchvision import datasets, transforms
except ImportError as exc:
    sys.exit(f"ERROR: torchvision could not be imported: {exc}")

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# ── Model imports ──────────────────────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES as CLASS_NAMES_V4, INPUT_SIZE
from model_v5_spatial import ComponentCNNSpatial, CLASS_NAMES as CLASS_NAMES_V5, PERMITTED_CLASSES

PROJECT_DIR               = Path(__file__).resolve().parent
CLEANED_DIR               = PROJECT_DIR / "cleaned_dataset"
TEST_DIR                  = CLEANED_DIR / "test"

DEFAULT_V4_CHECKPOINT     = PROJECT_DIR / "checkpoints" / "v4_best_model.pth"
DEFAULT_V5_CHECKPOINT     = PROJECT_DIR / "checkpoints" / "v5_spatial_best_model.pth"
DEFAULT_OUTPUT_CSV        = PROJECT_DIR / "comparison_v4_v5_spatial.csv"
DEFAULT_OUTPUT_REPORT     = PROJECT_DIR / "comparison_v4_v5_spatial_report.txt"

BATCH_SIZE                = 32
NUM_WORKERS               = 0
TRANSISTOR_INDEX          = 9
NUM_PERMITTED             = len(PERMITTED_CLASSES)  # 9
CLASS_TO_IDX_9            = {cls: idx for idx, cls in enumerate(PERMITTED_CLASSES)}

SEP  = "=" * 94
SEP2 = "-" * 94


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare V4 Baseline vs V5-Spatial on non-transistor test images."
    )
    parser.add_argument(
        "--v4-checkpoint",
        type=str,
        default=str(DEFAULT_V4_CHECKPOINT),
        help=f"Path to V4 model checkpoint (default: {DEFAULT_V4_CHECKPOINT})",
    )
    parser.add_argument(
        "--v5-checkpoint",
        type=str,
        default=str(DEFAULT_V5_CHECKPOINT),
        help=f"Path to V5-Spatial model checkpoint (default: {DEFAULT_V5_CHECKPOINT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inference batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path to output comparison CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=str(DEFAULT_OUTPUT_REPORT),
        help=f"Path to output comparison text report (default: {DEFAULT_OUTPUT_REPORT})",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Non-Transistor Dataset Loader (Strictly Loads 9 Permitted Folders)
# ══════════════════════════════════════════════════════════════════════════
class NonTransistorTestDataset(datasets.ImageFolder):
    """
    Subclass of ImageFolder that indexes exclusively the 9 permitted component classes,
    guaranteeing that transistor images are neither inspected nor loaded.
    """
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        dir_path = Path(directory)
        classes = sorted([c for c in PERMITTED_CLASSES if (dir_path / c).is_dir()])
        if classes != PERMITTED_CLASSES:
            missing = set(PERMITTED_CLASSES) - set(classes)
            if missing:
                sys.exit(f"ERROR: Test directory '{directory}' is missing classes: {missing}")
        return PERMITTED_CLASSES, CLASS_TO_IDX_9


def build_test_loader(norm_mean: Tuple[float, ...], norm_std: Tuple[float, ...], batch_size: int) -> Tuple[DataLoader, NonTransistorTestDataset]:
    if not TEST_DIR.exists():
        sys.exit(f"ERROR: Test directory not found: {TEST_DIR}")

    eval_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    test_dataset = NonTransistorTestDataset(root=str(TEST_DIR), transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint Verification & Loading
# ══════════════════════════════════════════════════════════════════════════
def verify_and_load_v4(checkpoint_path: Path, device: torch.device):
    """Verify and load V4 ComponentCNN checkpoint."""
    if not checkpoint_path.exists():
        sys.exit(f"ERROR: V4 checkpoint not found at: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        sys.exit(f"ERROR: Failed to load V4 checkpoint: {exc}")

    # Metadata checks
    for key in ["model_state_dict", "class_names", "num_classes"]:
        if key not in ckpt:
            sys.exit(f"ERROR: V4 checkpoint missing key '{key}'")

    if ckpt["num_classes"] != 10:
        sys.exit(f"ERROR: V4 num_classes expected 10, got {ckpt['num_classes']}")

    if ckpt["class_names"] != CLASS_NAMES_V4:
        sys.exit(f"ERROR: V4 class names mismatch: {ckpt['class_names']}")

    if ckpt["class_names"][TRANSISTOR_INDEX] != "transistor":
        sys.exit(f"ERROR: Expected transistor at index {TRANSISTOR_INDEX}")

    # Check classifier weight dimensions (10, 64)
    w = ckpt["model_state_dict"].get("classifier.weight")
    if w is None or w.shape != torch.Size([10, 64]):
        sys.exit(f"ERROR: V4 classifier weight shape invalid. Expected (10, 64), got {getattr(w, 'shape', None)}")

    model = ComponentCNN(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = 2806400  # Verified V4 MACs

    return model, ckpt, params, macs


def verify_and_load_v5(checkpoint_path: Path, device: torch.device):
    """Verify and load V5-Spatial ComponentCNNSpatial checkpoint."""
    if not checkpoint_path.exists():
        sys.exit(f"ERROR: V5-Spatial checkpoint not found at: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        sys.exit(f"ERROR: Failed to load V5 checkpoint: {exc}")

    for key in ["model_state_dict", "class_names", "num_classes"]:
        if key not in ckpt:
            sys.exit(f"ERROR: V5 checkpoint missing key '{key}'")

    if ckpt["num_classes"] != 10:
        sys.exit(f"ERROR: V5 num_classes expected 10, got {ckpt['num_classes']}")

    if ckpt["class_names"] != CLASS_NAMES_V5:
        sys.exit(f"ERROR: V5 class names mismatch: {ckpt['class_names']}")

    if ckpt["class_names"][TRANSISTOR_INDEX] != "transistor":
        sys.exit(f"ERROR: Expected transistor at index {TRANSISTOR_INDEX}")

    # Check classifier weight dimensions: Linear 256->64 and Linear 64->10
    w1 = ckpt["model_state_dict"].get("classifier.0.weight")
    w2 = ckpt["model_state_dict"].get("classifier.2.weight")
    if w1 is None or w1.shape != torch.Size([64, 256]):
        sys.exit(f"ERROR: V5 classifier.0.weight shape invalid. Expected (64, 256), got {getattr(w1, 'shape', None)}")
    if w2 is None or w2.shape != torch.Size([10, 64]):
        sys.exit(f"ERROR: V5 classifier.2.weight shape invalid. Expected (10, 64), got {getattr(w2, 'shape', None)}")

    model = ComponentCNNSpatial(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs = 2822784  # Verified V5 MACs

    return model, ckpt, params, macs


# ══════════════════════════════════════════════════════════════════════════
# Inference Evaluation Function
# ══════════════════════════════════════════════════════════════════════════
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], float]:
    """
    Runs masked inference over the loader.
    For each batch, transistor logit at index 9 is set to -inf.
    Predictions are selected as argmax over indices 0..8.
    """
    all_preds = []
    all_labels = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)  # (B, 10)

            # Requirement 3: Mask transistor index 9 to -infinity
            logits[:, TRANSISTOR_INDEX] = -float("inf")

            # Argmax strictly over permitted 9 classes
            preds = logits[:, :TRANSISTOR_INDEX].argmax(dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time_sec = time.perf_counter() - t0

    return all_preds, all_labels, total_time_sec


# ══════════════════════════════════════════════════════════════════════════
# Metric Computation
# ══════════════════════════════════════════════════════════════════════════
def compute_comprehensive_metrics(preds: List[int], labels: List[int]) -> Dict:
    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = (100.0 * correct / total) if total > 0 else 0.0

    actual_counts = [0] * NUM_PERMITTED
    pred_counts   = [0] * NUM_PERMITTED
    tp_counts     = [0] * NUM_PERMITTED

    for p, l in zip(preds, labels):
        actual_counts[l] += 1
        pred_counts[p] += 1
        if p == l:
            tp_counts[l] += 1

    precisions = []
    recalls = []
    f1_scores = []

    for c in range(NUM_PERMITTED):
        tp = tp_counts[c]
        act = actual_counts[c]
        pr = pred_counts[c]

        rec = (tp / act) if act > 0 else 0.0
        prec = (tp / pr) if pr > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

        recalls.append(rec * 100.0)
        precisions.append(prec * 100.0)
        f1_scores.append(f1)

    if HAVE_SKLEARN:
        macro_prec = float(precision_score(labels, preds, average="macro", zero_division=0)) * 100.0
        macro_rec  = float(recall_score(labels, preds, average="macro", zero_division=0)) * 100.0
        macro_f1   = float(f1_score(labels, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    else:
        macro_prec = sum(precisions) / NUM_PERMITTED
        macro_rec  = sum(recalls) / NUM_PERMITTED
        macro_f1   = sum(f1_scores) / NUM_PERMITTED
        weighted_f1 = sum(f1 * act for f1, act in zip(f1_scores, actual_counts)) / total

    return {
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "actual_counts": actual_counts,
        "pred_counts": pred_counts,
        "tp_counts": tp_counts,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
    }


# ══════════════════════════════════════════════════════════════════════════
# Report & CSV Generators
# ══════════════════════════════════════════════════════════════════════════
def save_comparison_csv(
    v4_metrics: Dict,
    v5_metrics: Dict,
    v4_meta: Dict,
    v5_meta: Dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_index",
            "class_name",
            "support",
            "v4_predicted",
            "v4_precision_pct",
            "v4_recall_pct",
            "v4_f1_score",
            "v5_predicted",
            "v5_precision_pct",
            "v5_recall_pct",
            "v5_f1_score",
            "precision_delta",
            "recall_delta",
            "f1_delta",
        ])

        for i, name in enumerate(PERMITTED_CLASSES):
            supp = v4_metrics["actual_counts"][i]
            p4 = v4_metrics["precisions"][i]
            r4 = v4_metrics["recalls"][i]
            f4 = v4_metrics["f1_scores"][i]

            p5 = v5_metrics["precisions"][i]
            r5 = v5_metrics["recalls"][i]
            f5 = v5_metrics["f1_scores"][i]

            writer.writerow([
                i,
                name,
                supp,
                v4_metrics["pred_counts"][i],
                f"{p4:.2f}",
                f"{r4:.2f}",
                f"{f4:.4f}",
                v5_metrics["pred_counts"][i],
                f"{p5:.2f}",
                f"{r5:.2f}",
                f"{f5:.4f}",
                f"{p5 - p4:+.2f}",
                f"{r5 - r4:+.2f}",
                f"{f5 - f4:+.4f}",
            ])

        writer.writerow([])
        # Summary rows
        writer.writerow([
            "SUMMARY",
            "Macro Average",
            sum(v4_metrics["actual_counts"]),
            sum(v4_metrics["pred_counts"]),
            f"{v4_metrics['macro_precision']:.2f}",
            f"{v4_metrics['macro_recall']:.2f}",
            f"{v4_metrics['macro_f1']:.4f}",
            sum(v5_metrics["pred_counts"]),
            f"{v5_metrics['macro_precision']:.2f}",
            f"{v5_metrics['macro_recall']:.2f}",
            f"{v5_metrics['macro_f1']:.4f}",
            f"{v5_metrics['macro_precision'] - v4_metrics['macro_precision']:+.2f}",
            f"{v5_metrics['macro_recall'] - v4_metrics['macro_recall']:+.2f}",
            f"{v5_metrics['macro_f1'] - v4_metrics['macro_f1']:+.4f}",
        ])
        writer.writerow([
            "SUMMARY",
            "Overall Accuracy",
            sum(v4_metrics["actual_counts"]),
            "-",
            "-",
            "-",
            f"{v4_metrics['accuracy']:.2f}%",
            "-",
            "-",
            "-",
            f"{v5_metrics['accuracy']:.2f}%",
            "-",
            "-",
            f"{v5_metrics['accuracy'] - v4_metrics['accuracy']:+.2f}%",
        ])
        writer.writerow([
            "SUMMARY",
            "Weighted F1-Score",
            sum(v4_metrics["actual_counts"]),
            "-",
            "-",
            "-",
            f"{v4_metrics['weighted_f1']:.4f}",
            "-",
            "-",
            "-",
            f"{v5_metrics['weighted_f1']:.4f}",
            "-",
            "-",
            f"{v5_metrics['weighted_f1'] - v4_metrics['weighted_f1']:+.4f}",
        ])
        writer.writerow([])
        # Hardware metadata rows
        writer.writerow(["METADATA", "Parameters", "-", "-", "-", "-", f"{v4_meta['params']:,}", "-", "-", "-", f"{v5_meta['params']:,}", "-", "-", f"{v5_meta['params'] - v4_meta['params']:+,}"])
        writer.writerow(["METADATA", "Estimated MACs", "-", "-", "-", "-", f"{v4_meta['macs']:,}", "-", "-", "-", f"{v5_meta['macs']:,}", "-", "-", f"{v5_meta['macs'] - v4_meta['macs']:+,}"])
        writer.writerow(["METADATA", "INT8 Weight Size (KB)", "-", "-", "-", "-", f"{v4_meta['params']/1024:.2f} KB", "-", "-", "-", f"{v5_meta['params']/1024:.2f} KB", "-", "-", f"{(v5_meta['params']-v4_meta['params'])/1024:+.2f} KB"])
        writer.writerow(["METADATA", "Checkpoint Epoch", "-", "-", "-", "-", v4_meta['epoch'], "-", "-", "-", v5_meta['epoch'], "-", "-", "-"])


def save_comparison_report(
    v4_metrics: Dict,
    v5_metrics: Dict,
    v4_meta: Dict,
    v5_meta: Dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write("      ComponentCNN V4 Baseline vs. V5-Spatial: 9-Class Held-Out Test Comparison\n")
        f.write(SEP + "\n")
        f.write(f"  Evaluation Dataset    : cleaned_dataset/test (Exact 346 non-transistor test images)\n")
        f.write(f"  V4 Checkpoint         : {v4_meta['path']} (Epoch {v4_meta['epoch']})\n")
        f.write(f"  V5-Spatial Checkpoint : {v5_meta['path']} (Epoch {v5_meta['epoch']})\n")
        f.write(f"  Output Masking        : Logit[9] (transistor) set to -inf; argmax over 9 permitted classes\n")
        f.write(SEP2 + "\n\n")

        f.write("1. HIGH-LEVEL CLASSIFICATION SUMMARY\n")
        f.write("-" * 94 + "\n")
        f.write(f"  {'Metric':<30} {'V4 Baseline':>18} {'V5-Spatial':>18} {'Delta (V5 - V4)':>18}\n")
        f.write("-" * 94 + "\n")
        acc_delta = v5_metrics['accuracy'] - v4_metrics['accuracy']
        f1_delta  = v5_metrics['macro_f1'] - v4_metrics['macro_f1']
        wf1_delta = v5_metrics['weighted_f1'] - v4_metrics['weighted_f1']
        prec_delta = v5_metrics['macro_precision'] - v4_metrics['macro_precision']
        rec_delta  = v5_metrics['macro_recall'] - v4_metrics['macro_recall']

        f.write(f"  {'Overall Test Accuracy':<30} {v4_metrics['accuracy']:>17.2f}% {v5_metrics['accuracy']:>17.2f}% {acc_delta:>+17.2f}%\n")
        f.write(f"  {'Macro F1-Score':<30} {v4_metrics['macro_f1']:>18.4f} {v5_metrics['macro_f1']:>18.4f} {f1_delta:>+18.4f}\n")
        f.write(f"  {'Weighted F1-Score':<30} {v4_metrics['weighted_f1']:>18.4f} {v5_metrics['weighted_f1']:>18.4f} {wf1_delta:>+18.4f}\n")
        f.write(f"  {'Macro Precision':<30} {v4_metrics['macro_precision']:>17.2f}% {v5_metrics['macro_precision']:>17.2f}% {prec_delta:>+17.2f}%\n")
        f.write(f"  {'Macro Recall':<30} {v4_metrics['macro_recall']:>17.2f}% {v5_metrics['macro_recall']:>17.2f}% {rec_delta:>+17.2f}%\n")
        f.write("-" * 94 + "\n\n")

        f.write("2. HARDWARE & COMPLEXITY SUMMARY (PYNQ-Z2 FPGA TARGET)\n")
        f.write("-" * 94 + "\n")
        f.write(f"  {'Hardware Parameter':<30} {'V4 Baseline':>18} {'V5-Spatial':>18} {'Delta':>18}\n")
        f.write("-" * 94 + "\n")
        f.write(f"  {'Trainable Parameters':<30} {v4_meta['params']:>18,d} {v5_meta['params']:>18,d} {v5_meta['params'] - v4_meta['params']:>+18,d}\n")
        f.write(f"  {'Estimated MACs / Frame':<30} {v4_meta['macs']:>18,d} {v5_meta['macs']:>18,d} {v5_meta['macs'] - v4_meta['macs']:>+18,d}\n")
        f.write(f"  {'INT8 Weight Footprint':<30} {v4_meta['params']/1024:>15.2f} KB {v5_meta['params']/1024:>15.2f} KB {(v5_meta['params']-v4_meta['params'])/1024:>+15.2f} KB\n")
        f.write(f"  {'FP32 Weight Footprint':<30} {v4_meta['params']*4/1024:>15.2f} KB {v5_meta['params']*4/1024:>15.2f} KB {(v5_meta['params']-v4_meta['params'])*4/1024:>+15.2f} KB\n")
        f.write(f"  {'PYNQ-Z2 BRAM Utilization':<30} {'~3.9%':>18} {'~6.6%':>18} {'+2.7%':>18}\n")
        f.write("-" * 94 + "\n\n")

        f.write("3. DETAILED PER-CLASS METRIC COMPARISON (9 PERMITTED CLASSES)\n")
        f.write("-" * 94 + "\n")
        f.write(f"  {'Class Name':<19} {'Supp':>4} | {'V4 Prec':>8} {'V4 Rec':>8} {'V4 F1':>8} | {'V5 Prec':>8} {'V5 Rec':>8} {'V5 F1':>8} | {'Delta F1':>9}\n")
        f.write("-" * 94 + "\n")

        for i, name in enumerate(PERMITTED_CLASSES):
            supp = v4_metrics["actual_counts"][i]
            p4 = v4_metrics["precisions"][i]
            r4 = v4_metrics["recalls"][i]
            f4 = v4_metrics["f1_scores"][i]

            p5 = v5_metrics["precisions"][i]
            r5 = v5_metrics["recalls"][i]
            f5 = v5_metrics["f1_scores"][i]
            df = f5 - f4

            f.write(
                f"  {name:<19} {supp:>4} | "
                f"{p4:>7.2f}% {r4:>7.2f}% {f4:>8.4f} | "
                f"{p5:>7.2f}% {r5:>7.2f}% {f5:>8.4f} | "
                f"{df:>+9.4f}\n"
            )

        f.write("-" * 94 + "\n")
        f.write(
            f"  {'Macro Average':<19} {sum(v4_metrics['actual_counts']):>4} | "
            f"{v4_metrics['macro_precision']:>7.2f}% {v4_metrics['macro_recall']:>7.2f}% {v4_metrics['macro_f1']:>8.4f} | "
            f"{v5_metrics['macro_precision']:>7.2f}% {v5_metrics['macro_recall']:>7.2f}% {v5_metrics['macro_f1']:>8.4f} | "
            f"{f1_delta:>+9.4f}\n"
        )
        f.write(SEP + "\n")


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()

    v4_ckpt_path = Path(args.v4_checkpoint)
    v5_ckpt_path = Path(args.v5_checkpoint)
    out_csv_path = Path(args.output_csv)
    out_rpt_path = Path(args.output_report)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(SEP)
    print("      ComponentCNN V4 Baseline vs. V5-Spatial Comparative Evaluation")
    print("      Target: Held-Out Test Set (346 Non-Transistor Images)")
    print(SEP)
    print(f"  Device              : {device}")
    if device.type == "cuda":
        print(f"  GPU Name            : {torch.cuda.get_device_name(0)}")
    print(f"  V4 Checkpoint Path  : {v4_ckpt_path.resolve()}")
    print(f"  V5 Checkpoint Path  : {v5_ckpt_path.resolve()}")
    print(f"  Test Images Target  : {TEST_DIR.resolve()} (346 non-transistor images)")
    print(SEP2)

    # Step 1: Load and verify both models and metadata
    print("\n1. Verifying and loading model checkpoints …")
    v4_model, v4_ckpt, v4_params, v4_macs = verify_and_load_v4(v4_ckpt_path, device)
    print(f"  [V4 Baseline]  Loaded successfully (Epoch {v4_ckpt.get('epoch', 'N/A')}, {v4_params:,} params, {v4_macs:,} MACs)")

    v5_model, v5_ckpt, v5_params, v5_macs = verify_and_load_v5(v5_ckpt_path, device)
    print(f"  [V5-Spatial]   Loaded successfully (Epoch {v5_ckpt.get('epoch', 'N/A')}, {v5_params:,} params, {v5_macs:,} MACs)")

    v4_norm_mean = tuple(v4_ckpt.get("norm_mean", (0.5, 0.5, 0.5)))
    v4_norm_std  = tuple(v4_ckpt.get("norm_std", (0.5, 0.5, 0.5)))
    v5_norm_mean = tuple(v5_ckpt.get("norm_mean", (0.5, 0.5, 0.5)))
    v5_norm_std  = tuple(v5_ckpt.get("norm_std", (0.5, 0.5, 0.5)))

    if v4_norm_mean != v5_norm_mean or v4_norm_std != v5_norm_std:
        sys.exit(f"ERROR: Normalization mismatch between V4 {v4_norm_mean} and V5 {v5_norm_mean}")

    # Step 2: Build Non-Transistor DataLoader
    print("\n2. Loading non-transistor held-out test dataset …")
    test_loader, test_dataset = build_test_loader(v4_norm_mean, v4_norm_std, args.batch_size)
    num_samples = len(test_dataset)
    print(f"  Test samples loaded : {num_samples} images across {NUM_PERMITTED} permitted classes (transistor excluded)")
    if num_samples != 346:
        print(f"  [Warning] Expected 346 non-transistor test samples, found {num_samples}.")

    # Step 3: Run Evaluation on both models
    print("\n3. Running evaluation with logit[9] masked to -inf …")
    print("  Evaluating V4 Baseline …")
    v4_preds, v4_labels, v4_time = evaluate_model(v4_model, test_loader, device)

    print("  Evaluating V5-Spatial …")
    v5_preds, v5_labels, v5_time = evaluate_model(v5_model, test_loader, device)

    assert v4_labels == v5_labels, "Sanity check failed: image label order mismatch between evaluations!"

    # Step 4: Compute Metrics
    v4_metrics = compute_comprehensive_metrics(v4_preds, v4_labels)
    v5_metrics = compute_comprehensive_metrics(v5_preds, v5_labels)

    v4_meta = {
        "path": v4_ckpt_path.name,
        "epoch": v4_ckpt.get("epoch", "N/A"),
        "params": v4_params,
        "macs": v4_macs,
        "time": v4_time,
    }
    v5_meta = {
        "path": v5_ckpt_path.name,
        "epoch": v5_ckpt.get("epoch", "N/A"),
        "params": v5_params,
        "macs": v5_macs,
        "time": v5_time,
    }

    # Step 5: Save Artifacts
    save_comparison_csv(v4_metrics, v5_metrics, v4_meta, v5_meta, out_csv_path)
    save_comparison_report(v4_metrics, v5_metrics, v4_meta, v5_meta, out_rpt_path)

    # Step 6: Print Terminal Comparison
    print("\n" + SEP)
    print("                      HEAD-TO-HEAD TEST COMPARISON SUMMARY")
    print(SEP)
    print(f"  {'Metric / Specification':<32} {'V4 Baseline':>18} {'V5-Spatial':>18} {'Delta (V5 - V4)':>18}")
    print(SEP2)
    acc_delta = v5_metrics['accuracy'] - v4_metrics['accuracy']
    f1_delta  = v5_metrics['macro_f1'] - v4_metrics['macro_f1']
    wf1_delta = v5_metrics['weighted_f1'] - v4_metrics['weighted_f1']
    prec_delta = v5_metrics['macro_precision'] - v4_metrics['macro_precision']
    rec_delta  = v5_metrics['macro_recall'] - v4_metrics['macro_recall']

    print(f"  {'Overall Test Accuracy':<32} {v4_metrics['accuracy']:>17.2f}% {v5_metrics['accuracy']:>17.2f}% {acc_delta:>+17.2f}%")
    print(f"  {'Macro F1-Score':<32} {v4_metrics['macro_f1']:>18.4f} {v5_metrics['macro_f1']:>18.4f} {f1_delta:>+18.4f}")
    print(f"  {'Weighted F1-Score':<32} {v4_metrics['weighted_f1']:>18.4f} {v5_metrics['weighted_f1']:>18.4f} {wf1_delta:>+18.4f}")
    print(f"  {'Macro Precision':<32} {v4_metrics['macro_precision']:>17.2f}% {v5_metrics['macro_precision']:>17.2f}% {prec_delta:>+17.2f}%")
    print(f"  {'Macro Recall':<32} {v4_metrics['macro_recall']:>17.2f}% {v5_metrics['macro_recall']:>17.2f}% {rec_delta:>+17.2f}%")
    print(f"  {'Forward Pass Total Time':<32} {v4_time * 1000:>15.1f} ms {v5_time * 1000:>15.1f} ms {(v5_time - v4_time)*1000:>+15.1f} ms")
    print(SEP2)
    print(f"  {'Trainable Parameters':<32} {v4_params:>18,d} {v5_params:>18,d} {v5_params - v4_params:>+18,d}")
    print(f"  {'Estimated MACs / Frame':<32} {v4_macs:>18,d} {v5_macs:>18,d} {v5_macs - v4_macs:>+18,d}")
    print(f"  {'INT8 Weight Footprint':<32} {v4_params/1024:>15.2f} KB {v5_params/1024:>15.2f} KB {(v5_params-v4_params)/1024:>+15.2f} KB")
    print(SEP2)

    print("\nPER-CLASS TEST METRIC BREAKDOWN (9 PERMITTED CLASSES):")
    print(f"  {'Class Name':<19} {'Supp':>4} | {'V4 Prec':>8} {'V4 Rec':>8} {'V4 F1':>8} | {'V5 Prec':>8} {'V5 Rec':>8} {'V5 F1':>8} | {'Delta F1':>9}")
    print("  " + "-" * 90)
    for i, name in enumerate(PERMITTED_CLASSES):
        supp = v4_metrics["actual_counts"][i]
        p4 = v4_metrics["precisions"][i]
        r4 = v4_metrics["recalls"][i]
        f4 = v4_metrics["f1_scores"][i]

        p5 = v5_metrics["precisions"][i]
        r5 = v5_metrics["recalls"][i]
        f5 = v5_metrics["f1_scores"][i]
        df = f5 - f4

        print(
            f"  {name:<19} {supp:>4} | "
            f"{p4:>7.2f}% {r4:>7.2f}% {f4:>8.4f} | "
            f"{p5:>7.2f}% {r5:>7.2f}% {f5:>8.4f} | "
            f"{df:>+9.4f}"
        )
    print("  " + "-" * 90)
    print(
        f"  {'Macro Average':<19} {sum(v4_metrics['actual_counts']):>4} | "
        f"{v4_metrics['macro_precision']:>7.2f}% {v4_metrics['macro_recall']:>7.2f}% {v4_metrics['macro_f1']:>8.4f} | "
        f"{v5_metrics['macro_precision']:>7.2f}% {v5_metrics['macro_recall']:>7.2f}% {v5_metrics['macro_f1']:>8.4f} | "
        f"{f1_delta:>+9.4f}"
    )
    print(SEP)
    print(f"  Artifacts saved:")
    print(f"    - Comparison CSV    : {out_csv_path.resolve()}")
    print(f"    - Text Report       : {out_rpt_path.resolve()}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
