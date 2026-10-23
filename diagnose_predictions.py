"""
diagnose_predictions.py
-----------------------
Diagnostic tool to compare predictions from:
  1. checkpoints/best_model.pth    (Original baseline)
  2. checkpoints/v2_best_model.pth (Improved v2 with augmentation & class weights)

Purpose
-------
Analyze prediction frequency bias, per-class recall, and macro F1 on the
validation set (cleaned_dataset/val/) to evaluate how class weighting and
data augmentation affected model predictions on minority and problem classes.

Requirements Satisfied
----------------------
1. Uses cleaned_dataset/val/ ONLY (never touches the test set).
2. Preserves exact 64x64 RGB resize, [-1, 1] normalization, class order,
   and ComponentCNN architecture from evaluate.py.
3. For each checkpoint, reports:
   - Validation accuracy
   - Macro-averaged F1-score
   - Per-class recall
   - Number of predictions assigned to each class
4. Compares predicted class distribution against actual ground-truth class counts.
5. Saves a two-panel visual comparison bar chart to prediction_distribution.png:
   - Panel 1: Prediction frequency vs actual ground-truth counts per class.
   - Panel 2: Per-class recall % comparison (baseline vs v2).
6. Read-only operation: does not train, change checkpoints, or modify data.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Safe package imports ──────────────────────────────────────────────────
try:
    from torchvision import datasets, transforms
except ImportError as exc:
    print(f"ERROR: torchvision could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    print(f"ERROR: matplotlib could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)

try:
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# ── Architecture & Class definitions ──────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE


# ══════════════════════════════════════════════════════════════════════════
# Paths and Defaults
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR         = Path(__file__).resolve().parent
CLEANED_DIR         = PROJECT_DIR / "cleaned_dataset"
VAL_DIR             = CLEANED_DIR / "val"

DEFAULT_BASELINE    = PROJECT_DIR / "checkpoints" / "best_model.pth"
DEFAULT_V2          = PROJECT_DIR / "checkpoints" / "v2_best_model.pth"
DEFAULT_OUTPUT_PNG  = PROJECT_DIR / "prediction_distribution.png"

NORM_MEAN           = (0.5, 0.5, 0.5)
NORM_STD            = (0.5, 0.5, 0.5)
BATCH_SIZE          = 32
NUM_WORKERS         = 0

SEP  = "=" * 94
SEP2 = "-" * 94


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose and compare prediction distributions of baseline vs v2 models."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(DEFAULT_BASELINE),
        help=f"Path to baseline checkpoint (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--v2",
        type=str,
        default=str(DEFAULT_V2),
        help=f"Path to v2 checkpoint (default: {DEFAULT_V2})",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=str(DEFAULT_OUTPUT_PNG),
        help=f"Output bar chart filename (default: {DEFAULT_OUTPUT_PNG.name})",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Data Pipeline (Validation set ONLY)
# ══════════════════════════════════════════════════════════════════════════
def build_val_loader(val_dir: Path) -> tuple[DataLoader, datasets.ImageFolder]:
    """
    Build DataLoader for cleaned_dataset/val/ using the exact deterministic
    preprocessing pipeline from evaluate.py.
    """
    if not val_dir.exists():
        sys.exit(f"ERROR: Validation directory not found at: {val_dir}")

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=transform)

    # Verify class ordering matches model.CLASS_NAMES
    expected = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if val_dataset.class_to_idx != expected:
        sys.exit(
            f"FATAL: Class ordering mismatch in validation folder!\n"
            f"  Expected (model.py): {expected}\n"
            f"  Got (ImageFolder)  : {val_dataset.class_to_idx}"
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return val_loader, val_dataset


# ══════════════════════════════════════════════════════════════════════════
# Inference and Checkpoint Loading
# ══════════════════════════════════════════════════════════════════════════
def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        sys.exit(
            f"ERROR: Checkpoint file not found: {path}\n"
            "Please ensure the model training script has finished and saved the checkpoint."
        )
    return torch.load(path, map_location=device, weights_only=False)


@torch.no_grad()
def run_model_inference(
    checkpoint_path: Path,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict, list[int], list[int]]:
    """
    Load a checkpoint and run evaluation forward passes.
    Returns (checkpoint_dict, all_predictions, all_ground_truth_labels).
    """
    ckpt = load_checkpoint(checkpoint_path, device)

    num_classes = ckpt.get("num_classes", NUM_CLASSES)
    model = ComponentCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return ckpt, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════
# Metric Calculations
# ══════════════════════════════════════════════════════════════════════════
def compute_detailed_metrics(
    preds: list[int],
    labels: list[int],
    num_classes: int = NUM_CLASSES,
) -> dict:
    """
    Calculate accuracy, macro F1, per-class recall, and prediction counts.
    """
    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    actual_counts = [0] * num_classes
    pred_counts = [0] * num_classes
    tp_counts = [0] * num_classes

    for p, l in zip(preds, labels):
        actual_counts[l] += 1
        pred_counts[p] += 1
        if p == l:
            tp_counts[l] += 1

    per_class_recall = []
    per_class_precision = []
    per_class_f1 = []

    for c in range(num_classes):
        tp = tp_counts[c]
        actual = actual_counts[c]
        pred = pred_counts[c]

        rec = (tp / actual) if actual > 0 else 0.0
        prec = (tp / pred) if pred > 0 else 0.0

        if (prec + rec) > 0.0:
            f1 = 2.0 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0

        per_class_recall.append(rec * 100.0)
        per_class_precision.append(prec * 100.0)
        per_class_f1.append(f1)

    if HAVE_SKLEARN:
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    else:
        macro_f1 = sum(per_class_f1) / num_classes

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "actual_counts": actual_counts,
        "pred_counts": pred_counts,
        "tp_counts": tp_counts,
        "recall": per_class_recall,
        "precision": per_class_precision,
        "f1": per_class_f1,
    }


# ══════════════════════════════════════════════════════════════════════════
# Visualization: Two-Panel Diagnostic Chart
# ══════════════════════════════════════════════════════════════════════════
def plot_prediction_distribution(
    class_names: list[str],
    actual_counts: list[int],
    base_pred_counts: list[int],
    v2_pred_counts: list[int],
    base_recalls: list[float],
    v2_recalls: list[float],
    output_png: Path,
) -> None:
    """
    Generate and save a clear comparison chart:
      Panel 1: Actual Ground-Truth Counts vs Baseline vs v2 Predicted Counts.
      Panel 2: Per-Class Recall % (Baseline vs v2).
    """
    num_classes = len(class_names)
    indices = np.arange(num_classes)
    bar_width = 0.26

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 11), dpi=150)

    # ── Panel 1: Prediction Frequency vs Actual Ground Truth ──────────────
    b_actual = ax1.bar(
        indices - bar_width,
        actual_counts,
        bar_width,
        label="Actual Val Ground Truth",
        color="#546e7a",
        edgecolor="#263238",
        alpha=0.9,
    )
    b_base = ax1.bar(
        indices,
        base_pred_counts,
        bar_width,
        label="Baseline (v1) Predictions",
        color="#e65100",
        edgecolor="#bf360c",
        alpha=0.9,
    )
    b_v2 = ax1.bar(
        indices + bar_width,
        v2_pred_counts,
        bar_width,
        label="Improved (v2) Predictions",
        color="#1e88e5",
        edgecolor="#0d47a1",
        alpha=0.9,
    )

    ax1.set_title(
        "Validation Set: Prediction Frequency Distribution vs Actual Ground-Truth Counts",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylabel("Number of Images", fontsize=11, fontweight="bold")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9.5)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.9)

    # Add numeric labels on top of bars
    for bars in (b_actual, b_base, b_v2):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f"{int(height)}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    # ── Panel 2: Per-Class Recall Comparison ──────────────────────────────
    width_recall = 0.35
    b_rec_base = ax2.bar(
        indices - width_recall / 2,
        base_recalls,
        width_recall,
        label="Baseline (v1) Recall %",
        color="#ff7043",
        edgecolor="#d84315",
        alpha=0.9,
    )
    b_rec_v2 = ax2.bar(
        indices + width_recall / 2,
        v2_recalls,
        width_recall,
        label="Improved (v2) Recall %",
        color="#42a5f5",
        edgecolor="#1565c0",
        alpha=0.9,
    )

    ax2.set_title(
        "Validation Set: Per-Class Recall Comparison (Baseline v1 vs Improved v2)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylabel("Recall (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(class_names, rotation=25, ha="right", fontsize=9.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.axhline(50, color="#78909c", linestyle=":", alpha=0.8, label="50% Recall Reference")
    ax2.legend(fontsize=10, loc="upper right", framealpha=0.9)

    # Add percentage labels on top of bars
    for bar in b_rec_base:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    for bar in b_rec_v2:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
        )

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()

    baseline_path = Path(args.baseline)
    v2_path       = Path(args.v2)
    output_png    = Path(args.output_png)

    # Resolve relative paths against PROJECT_DIR if needed
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_DIR / baseline_path
    if not v2_path.is_absolute():
        v2_path = PROJECT_DIR / v2_path
    if not output_png.is_absolute():
        output_png = PROJECT_DIR / output_png

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print(SEP)
    print("  Electronic Component Classifier — Prediction & Recall Diagnosis")
    print(SEP)
    print(f"  Device           : {device}")
    if device.type == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Dataset split    : {VAL_DIR} (Validation Set ONLY — Test set untouched)")
    print(f"  Baseline Checkpt : {baseline_path}")
    print(f"  v2 Checkpoint    : {v2_path}")
    print(SEP2)

    # 1. Load validation DataLoader
    print("\n[1/3] Loading validation dataset …")
    val_loader, val_dataset = build_val_loader(VAL_DIR)
    total_samples = len(val_dataset)
    print(f"  Validation images: {total_samples}")
    print(f"  Classes          : {NUM_CLASSES}")

    # 2. Evaluate Baseline Model
    print("\n[2/3] Evaluating Baseline Model (v1) on validation set …")
    base_ckpt, base_preds, base_labels = run_model_inference(baseline_path, val_loader, device)
    base_metrics = compute_detailed_metrics(base_preds, base_labels, NUM_CLASSES)
    base_epoch = base_ckpt.get("epoch", "N/A")
    base_saved_val_acc = base_ckpt.get("val_accuracy", base_metrics["accuracy"])
    print(f"  Baseline Epoch   : {base_epoch}")
    print(f"  Baseline Val Acc : {base_metrics['accuracy']:.2f}%")
    print(f"  Baseline Macro F1: {base_metrics['macro_f1']:.4f}")

    # 3. Evaluate Improved Model (v2)
    print("\n[3/3] Evaluating Improved Model (v2) on validation set …")
    v2_ckpt, v2_preds, v2_labels = run_model_inference(v2_path, val_loader, device)
    v2_metrics = compute_detailed_metrics(v2_preds, v2_labels, NUM_CLASSES)
    v2_epoch = v2_ckpt.get("epoch", "N/A")
    v2_saved_macro_f1 = v2_ckpt.get("val_macro_f1", v2_metrics["macro_f1"])
    print(f"  v2 Epoch         : {v2_epoch}")
    print(f"  v2 Val Acc       : {v2_metrics['accuracy']:.2f}%")
    print(f"  v2 Macro F1      : {v2_metrics['macro_f1']:.4f}")

    # 4. Print Overall Summary Table
    acc_delta = v2_metrics["accuracy"] - base_metrics["accuracy"]
    f1_delta  = v2_metrics["macro_f1"] - base_metrics["macro_f1"]

    print("\n" + SEP)
    print("  OVERALL MODEL COMPARISON (Validation Set: 342 images)")
    print(SEP)
    print(f"  {'Metric':<24} {'Baseline (v1)':>18} {'Improved (v2)':>18} {'Delta (v2 - v1)':>18}")
    print("  " + "-" * 82)
    print(f"  {'Checkpoint Path':<24} {baseline_path.name:>18} {v2_path.name:>18} {'':>18}")
    print(f"  {'Checkpoint Epoch':<24} {str(base_epoch):>18} {str(v2_epoch):>18} {'':>18}")
    print(f"  {'Validation Accuracy':<24} {base_metrics['accuracy']:>17.2f}% {v2_metrics['accuracy']:>17.2f}% {acc_delta:>+17.2f}%")
    print(f"  {'Validation Macro F1':<24} {base_metrics['macro_f1']:>18.4f} {v2_metrics['macro_f1']:>18.4f} {f1_delta:>+18.4f}")
    print(SEP2)

    # 5. Print Per-Class Detailed Diagnostic Table
    print("\n" + SEP)
    print("  PER-CLASS PREDICTION FREQUENCY & RECALL COMPARISON")
    print(SEP)
    header = (
        f"  {'Class Name':<20} "
        f"{'Actual':>7} "
        f"{'Base Pred':>10} "
        f"{'Base Rec':>10} "
        f"{'v2 Pred':>9} "
        f"{'v2 Rec':>9} "
        f"{'Recall Delta':>14}"
    )
    print(header)
    print("  " + "-" * 85)

    actual_counts = base_metrics["actual_counts"]
    for idx, cls in enumerate(CLASS_NAMES):
        act       = actual_counts[idx]
        b_pred    = base_metrics["pred_counts"][idx]
        b_rec     = base_metrics["recall"][idx]
        v_pred    = v2_metrics["pred_counts"][idx]
        v_rec     = v2_metrics["recall"][idx]
        rec_diff  = v_rec - b_rec

        print(
            f"  {cls:<20} "
            f"{act:>7} "
            f"{b_pred:>10} "
            f"{b_rec:>9.2f}% "
            f"{v_pred:>9} "
            f"{v_rec:>8.2f}% "
            f"{rec_diff:>+13.2f}%"
        )

    print("  " + "-" * 85)
    mean_base_rec = sum(base_metrics["recall"]) / NUM_CLASSES
    mean_v2_rec   = sum(v2_metrics["recall"]) / NUM_CLASSES
    mean_diff     = mean_v2_rec - mean_base_rec
    print(
        f"  {'Macro Average':<20} "
        f"{sum(actual_counts):>7} "
        f"{sum(base_metrics['pred_counts']):>10} "
        f"{mean_base_rec:>9.2f}% "
        f"{sum(v2_metrics['pred_counts']):>9} "
        f"{mean_v2_rec:>8.2f}% "
        f"{mean_diff:>+13.2f}%"
    )
    print(SEP)

    # 6. Generate and Save the Visual Chart
    print("\nGenerating visual comparison chart …")
    plot_prediction_distribution(
        class_names=CLASS_NAMES,
        actual_counts=actual_counts,
        base_pred_counts=base_metrics["pred_counts"],
        v2_pred_counts=v2_metrics["pred_counts"],
        base_recalls=base_metrics["recall"],
        v2_recalls=v2_metrics["recall"],
        output_png=output_png,
    )
    print(f"  -> Chart successfully saved to: {output_png}")
    print(SEP)
    print("Diagnosis complete. Output chart ready for visual inspection.\n")


if __name__ == "__main__":
    main()
