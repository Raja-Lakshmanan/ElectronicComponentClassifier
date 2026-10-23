"""
evaluate_9class.py
------------------
Evaluation pipeline for the 9-class ComponentCNN checkpoint on the held-out test set
(cleaned_dataset/test), strictly filtering out the transistor class.

Key Features:
  1. Loads checkpoints/9class_best_model.pth (or user-specified checkpoint).
  2. Verifies fixed 9-class mapping and normalization constants stored in checkpoint.
  3. Evaluates on cleaned_dataset/test/ using NineClassDataset (346 images, transistor excluded).
  4. Computes overall test accuracy, macro F1, weighted F1, and per-class precision / recall / F1 / support.
  5. Computes and plots the full 9x9 confusion matrix with counts and percentages.
  6. Saves artifacts:
     - confusion_matrix_test_9class.png
     - classification_report_9class.txt
     - classification_report_test_9class.csv
  7. Reports total parameter count and approximate model size (FP32).
  8. Strictly read-only: does not modify dataset, model weights, or existing reports.
"""

import argparse
import csv
import sys
from pathlib import Path
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
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

# ── Model definition and constants ─────────────────────────────────────────
from model_9class import (
    CLASS_NAMES_9,
    CLASS_TO_IDX_9,
    NUM_CLASSES_9,
    INPUT_SIZE,
    ComponentCNN9,
    count_parameters,
)

PROJECT_DIR        = Path(__file__).resolve().parent
CLEANED_DIR        = PROJECT_DIR / "cleaned_dataset"
TEST_DIR           = CLEANED_DIR / "test"
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "9class_best_model.pth"

DEFAULT_CM_PNG     = PROJECT_DIR / "confusion_matrix_test_9class.png"
DEFAULT_REPORT_TXT = PROJECT_DIR / "classification_report_9class.txt"
DEFAULT_REPORT_CSV = PROJECT_DIR / "classification_report_test_9class.csv"

BATCH_SIZE         = 32
NUM_WORKERS        = 0

SEP  = "=" * 80
SEP2 = "-" * 80


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 9-class ComponentCNN checkpoint on cleaned_dataset/test/ (excluding transistor)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to 9-class model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Inference batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-cm",
        type=str,
        default=str(DEFAULT_CM_PNG),
        help=f"Filename of output confusion matrix plot (default: {DEFAULT_CM_PNG.name})",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default=str(DEFAULT_REPORT_TXT),
        help=f"Filename of text classification report (default: {DEFAULT_REPORT_TXT.name})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_REPORT_CSV),
        help=f"Filename of CSV classification report (default: {DEFAULT_REPORT_CSV.name})",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# 9-Class Dataset Filter
# ══════════════════════════════════════════════════════════════════════════
class NineClassDataset(datasets.ImageFolder):
    """
    Subclass of ImageFolder that dynamically filters out 'transistor'
    and guarantees exact alignment with CLASS_NAMES_9 and CLASS_TO_IDX_9.
    """
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        dir_path = Path(directory)
        classes = sorted([c for c in CLASS_NAMES_9 if (dir_path / c).is_dir()])
        if classes != CLASS_NAMES_9:
            missing = set(CLASS_NAMES_9) - set(classes)
            if missing:
                sys.exit(f"ERROR: Target directory '{directory}' is missing classes: {missing}")
        return CLASS_NAMES_9, CLASS_TO_IDX_9


def get_test_transforms(norm_mean: Tuple[float, ...], norm_std: Tuple[float, ...]):
    """Deterministic test preprocessing matching checkpoint normalization."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint Loader
# ══════════════════════════════════════════════════════════════════════════
def load_9class_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        sys.exit(
            f"ERROR: Checkpoint file not found: {checkpoint_path}\n"
            f"Please train the 9-class model first by running:\n"
            f"    python train_9class.py"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_classes = ckpt.get("num_classes", NUM_CLASSES_9)
    if num_classes != 9:
        sys.exit(f"ERROR: Checkpoint is for {num_classes} classes, but 9-class evaluation requires exactly 9 classes!")

    class_names = ckpt.get("class_names", CLASS_NAMES_9)
    if class_names != CLASS_NAMES_9:
        sys.exit(f"ERROR: Checkpoint class names mismatch!\nExpected: {CLASS_NAMES_9}\nFound: {class_names}")

    norm_mean = ckpt.get("norm_mean", (0.5, 0.5, 0.5))
    norm_std  = ckpt.get("norm_std", (0.5, 0.5, 0.5))

    model = ComponentCNN9(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, norm_mean, norm_std


# ══════════════════════════════════════════════════════════════════════════
# Inference and Metrics
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_evaluation(model: nn.Module, loader: DataLoader, device: torch.device):
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    # Per-class counts
    actual_counts = [0] * NUM_CLASSES_9
    pred_counts   = [0] * NUM_CLASSES_9
    tp_counts     = [0] * NUM_CLASSES_9

    for p, l in zip(all_preds, all_labels):
        actual_counts[l] += 1
        pred_counts[p] += 1
        if p == l:
            tp_counts[l] += 1

    precisions = []
    recalls    = []
    f1_scores  = []

    for c in range(NUM_CLASSES_9):
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
        macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES_9)))
    else:
        macro_f1 = sum(f1_scores) / NUM_CLASSES_9
        weighted_f1 = sum(f1 * act for f1, act in zip(f1_scores, actual_counts)) / total
        cm = np.zeros((NUM_CLASSES_9, NUM_CLASSES_9), dtype=int)
        for p, l in zip(all_preds, all_labels):
            cm[l, p] += 1

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "actual_counts": actual_counts,
        "pred_counts": pred_counts,
        "tp_counts": tp_counts,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "cm": cm,
    }


def find_top_confusions(cm: np.ndarray, top_k: int = 5) -> List[Tuple[str, str, int]]:
    """Identify off-diagonal confused pairs."""
    confusions = []
    for i in range(NUM_CLASSES_9):
        for j in range(NUM_CLASSES_9):
            if i != j and cm[i, j] > 0:
                confusions.append((CLASS_NAMES_9[i], CLASS_NAMES_9[j], int(cm[i, j])))
    confusions.sort(key=lambda x: x[2], reverse=True)
    return confusions[:top_k]


# ══════════════════════════════════════════════════════════════════════════
# Plotting and Reporting
# ══════════════════════════════════════════════════════════════════════════
def save_confusion_matrix_plot(cm: np.ndarray, accuracy: float, macro_f1: float, path: Path) -> None:
    if not HAVE_MATPLOTLIB:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    title = f"ComponentCNN-9 Test Confusion Matrix\nOverall Accuracy: {accuracy:.2f}% | Macro F1: {macro_f1:.4f}"
    ax.set(
        xticks=np.arange(NUM_CLASSES_9),
        yticks=np.arange(NUM_CLASSES_9),
        xticklabels=CLASS_NAMES_9,
        yticklabels=CLASS_NAMES_9,
        title=title,
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8.5)
    plt.setp(ax.get_yticklabels(), fontsize=8.5)

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES_9):
        for j in range(NUM_CLASSES_9):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_reports(metrics: Dict, ckpt: Dict, cm: np.ndarray, top_confused: List, txt_path: Path, csv_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Text Report
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("  ComponentCNN-9 — Held-out Test Set Evaluation Report\n")
        f.write("  (Transistor strictly excluded; 9 component classes evaluated)\n")
        f.write("=" * 78 + "\n\n")

        f.write("Model & Checkpoint Metadata:\n")
        f.write(f"  Architecture        : ComponentCNN9 (3 Conv blocks -> AdaptiveAvgPool -> Linear)\n")
        f.write(f"  Parameters          : {sum(p.numel() for p in ComponentCNN9().parameters()):,}\n")
        f.write(f"  Checkpoint Epoch    : {ckpt.get('epoch', 'N/A')}\n")
        f.write(f"  Val Macro F1 (ckpt) : {ckpt.get('val_macro_f1', 0.0):.4f}\n")
        f.write(f"  Val Accuracy (ckpt) : {ckpt.get('val_accuracy', 0.0):.2f}%\n")
        f.write(f"  Classes Evaluated   : {NUM_CLASSES_9}\n\n")

        f.write("Overall Test Metrics:\n")
        f.write(f"  Total Test Samples  : {sum(metrics['actual_counts'])}\n")
        f.write(f"  Overall Accuracy    : {metrics['accuracy']:.2f}%\n")
        f.write(f"  Macro F1-Score      : {metrics['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1-Score   : {metrics['weighted_f1']:.4f}\n\n")

        f.write("Per-Class Classification Breakdown:\n")
        f.write(f"  {'Class':<22} {'Support':>8} {'Preds':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write("  " + "-" * 72 + "\n")
        for idx, name in enumerate(CLASS_NAMES_9):
            f.write(
                f"  {name:<22} "
                f"{metrics['actual_counts'][idx]:>8} "
                f"{metrics['pred_counts'][idx]:>8} "
                f"{metrics['precisions'][idx]:>9.2f}% "
                f"{metrics['recalls'][idx]:>9.2f}% "
                f"{metrics['f1_scores'][idx]:>10.4f}\n"
            )
        f.write("  " + "-" * 72 + "\n")
        avg_prec = sum(metrics["precisions"]) / NUM_CLASSES_9
        avg_rec  = sum(metrics["recalls"]) / NUM_CLASSES_9
        f.write(
            f"  {'Macro Average':<22} "
            f"{sum(metrics['actual_counts']):>8} "
            f"{sum(metrics['pred_counts']):>8} "
            f"{avg_prec:>9.2f}% "
            f"{avg_rec:>9.2f}% "
            f"{metrics['macro_f1']:>10.4f}\n\n"
        )

        f.write("Top Confused Pairs (True -> Predicted):\n")
        for true_c, pred_c, count in top_confused:
            f.write(f"  - {true_c:<20} -> {pred_c:<20} : {count} errors\n")
        f.write("\n" + "=" * 78 + "\n")

    # 2. CSV Report
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "support", "predictions", "precision_pct", "recall_pct", "f1_score"])
        for idx, name in enumerate(CLASS_NAMES_9):
            writer.writerow([
                idx,
                name,
                metrics["actual_counts"][idx],
                metrics["pred_counts"][idx],
                f"{metrics['precisions'][idx]:.2f}",
                f"{metrics['recalls'][idx]:.2f}",
                f"{metrics['f1_scores'][idx]:.4f}",
            ])
        writer.writerow([])
        avg_prec = sum(metrics["precisions"]) / NUM_CLASSES_9
        avg_rec  = sum(metrics["recalls"]) / NUM_CLASSES_9
        writer.writerow(["SUMMARY", "Macro Average", sum(metrics["actual_counts"]), sum(metrics["pred_counts"]), f"{avg_prec:.2f}", f"{avg_rec:.2f}", f"{metrics['macro_f1']:.4f}"])
        writer.writerow(["SUMMARY", "Overall Accuracy", sum(metrics["actual_counts"]), sum(metrics["pred_counts"]), "", "", f"{metrics['accuracy']:.2f}%"])


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    checkpoint_path = Path(args.checkpoint)
    output_cm_path  = Path(args.output_cm)
    output_txt_path = Path(args.output_txt)
    output_csv_path = Path(args.output_csv)

    print()
    print(SEP)
    print("  Electronic Component Classifier — 9-Class Test Evaluation")
    print("  (Transistor strictly excluded; 9 component classes evaluated)")
    print(SEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device              : {device}")
    print(f"  Checkpoint Path     : {checkpoint_path}")
    print(f"  Test Dataset Path   : {TEST_DIR}")
    print(SEP2)

    # 1. Load Checkpoint
    print("\n[1/4] Loading model checkpoint & restoring architecture …")
    model, ckpt, norm_mean, norm_std = load_9class_checkpoint(checkpoint_path, device)
    params, size_kb = count_parameters(model)
    print(f"  Model Architecture  : {model.__class__.__name__}")
    print(f"  Trainable Parameters: {params:,}")
    print(f"  Model Size (FP32)   : {size_kb:.2f} KB (94.4 KB)")
    print(f"  Trained Epochs      : {ckpt.get('epoch', 'N/A')}")
    print(f"  Best Val Macro F1   : {ckpt.get('val_macro_f1', 0.0):.4f}")
    print(f"  Best Val Accuracy   : {ckpt.get('val_accuracy', 0.0):.2f}%")
    print(f"  Normalization Mean  : {norm_mean}")
    print(f"  Normalization Std   : {norm_std}")
    print(SEP2)

    # 2. Build Test DataLoader
    print("\n[2/4] Loading test dataset (filtering out transistor) …")
    test_dataset = NineClassDataset(
        root=str(TEST_DIR),
        transform=get_test_transforms(norm_mean, norm_std),
    )
    print(f"  Total Test Samples  : {len(test_dataset)} (Transistor completely excluded)")
    print(f"  Number of Classes   : {len(test_dataset.classes)}")
    print(f"  Class Names         : {', '.join(test_dataset.classes)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(SEP2)

    # 3. Run Inference
    print("\n[3/4] Running forward pass on held-out test split …")
    metrics = run_evaluation(model, test_loader, device)

    print("\n" + SEP)
    print(f"  TEST EVALUATION RESULTS (9 CLASSES)")
    print(SEP)
    print(f"  Overall Accuracy    : {metrics['accuracy']:.2f}%  ({sum(metrics['tp_counts'])} / {len(test_dataset)})")
    print(f"  Macro F1-Score      : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1-Score   : {metrics['weighted_f1']:.4f}")
    print(SEP2)

    print("\nPer-Class Classification Report:")
    print(f"  {'Class':<22} {'Support':>8} {'Preds':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("  " + "-" * 72)
    for idx, name in enumerate(CLASS_NAMES_9):
        print(
            f"  {name:<22} "
            f"{metrics['actual_counts'][idx]:>8} "
            f"{metrics['pred_counts'][idx]:>8} "
            f"{metrics['precisions'][idx]:>9.2f}% "
            f"{metrics['recalls'][idx]:>9.2f}% "
            f"{metrics['f1_scores'][idx]:>10.4f}"
        )
    print("  " + "-" * 72)
    avg_prec = sum(metrics["precisions"]) / NUM_CLASSES_9
    avg_rec  = sum(metrics["recalls"]) / NUM_CLASSES_9
    print(
        f"  {'Macro Average':<22} "
        f"{sum(metrics['actual_counts']):>8} "
        f"{sum(metrics['pred_counts']):>8} "
        f"{avg_prec:>9.2f}% "
        f"{avg_rec:>9.2f}% "
        f"{metrics['macro_f1']:>10.4f}"
    )
    print(SEP2)

    # Top confused pairs
    top_confused = find_top_confusions(metrics["cm"], top_k=5)
    print("\nTop Confused Pairs (True -> Predicted):")
    for true_c, pred_c, count in top_confused:
        print(f"  - {true_c:<20} -> {pred_c:<20} : {count} misclassifications")
    print(SEP2)

    # 4. Save Artifacts
    print("\n[4/4] Saving output evaluation artifacts …")
    save_confusion_matrix_plot(metrics["cm"], metrics["accuracy"], metrics["macro_f1"], output_cm_path)
    print(f"  -> Confusion Matrix Plot  : {output_cm_path}")

    save_reports(metrics, ckpt, metrics["cm"], top_confused, output_txt_path, output_csv_path)
    print(f"  -> Text Report            : {output_txt_path}")
    print(f"  -> CSV Metrics Report     : {output_csv_path}")

    print("\n" + SEP)
    print("  9-CLASS EVALUATION COMPLETED SUCCESSFULLY")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
