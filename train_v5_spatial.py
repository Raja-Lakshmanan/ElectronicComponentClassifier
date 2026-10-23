"""
train_v5_spatial.py
-------------------
Training pipeline for ComponentCNNSpatial (V5-Spatial) on cleaned_dataset/.

Experiment Motivation (V5-Spatial):
------------------------------------
1. Diagnostic Findings from V4:
   - Training loss (1.258) and accuracy (58.10%) plateaued early, matching validation loss (1.212).
   - This proved architectural UNDERFITTING (high bias), caused by the 1x1 Global Average
     Pooling bottleneck which collapsed the 8x8x64 feature map into just 64 scalar averages.
   - Severe confusion between push_button (recall 47.8%) and potentiometer, as well as resistor
     (recall 57.8%) and potentiometer, because 1x1 average pooling completely lost spatial
     symmetry, lead orientation, and multi-pin geometric layout.
2. V5-Spatial Architecture:
   - Preserves all 3 convolutional blocks identical to V4 (23,584 conv parameters).
   - Replaces AdaptiveAvgPool2d((1, 1)) with AdaptiveAvgPool2d((2, 2)), preserving 4 spatial quadrants.
   - Replaces the 1-layer Linear(64, 10) with a 2-stage MLP: Linear(256, 64) -> ReLU -> Linear(64, 10).
   - Parameters: exactly 40,682 (only +16,448 params over V4, weight memory ~40.7 KB in INT8).
   - Compute load: 2,822,784 MACs (<0.6% change over V4).
   - 100% fits within on-chip BRAM (<7% of PYNQ-Z2's 612.5 KB BRAM) with zero off-chip DDR bandwidth.

Training Configuration:
  - Input: 64x64 RGB
  - Training augmentation: RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter
  - Normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
  - Loss: CrossEntropyLoss with square-root inverse class frequency weights (mean normalized to 1)
  - Optimizer: Adam (lr=0.001)
  - Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-5) monitored on val loss
  - Early stopping: 10 epochs patience monitored on validation macro F1
  - Max epochs: up to 50
  - Random seed: 42
  - Strictly uses cleaned_dataset/train and cleaned_dataset/val. Test split is untouched!
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Safe imports ──────────────────────────────────────────────────────────
try:
    from sklearn.metrics import f1_score, confusion_matrix
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Headless safe
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

# ── Model definition ──────────────────────────────────────────────────────
from model_v5_spatial import (
    ComponentCNNSpatial,
    CLASS_NAMES,
    NUM_CLASSES,
    INPUT_SIZE,
    PERMITTED_CLASSES,
)

# ══════════════════════════════════════════════════════════════════════════
# Defaults and Paths
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR    = Path(__file__).resolve().parent
CLEANED_DIR    = PROJECT_DIR / "cleaned_dataset"
TRAIN_DIR      = CLEANED_DIR / "train"
VAL_DIR        = CLEANED_DIR / "val"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

HISTORY_CSV_PATH     = PROJECT_DIR / "training_history_v5_spatial.csv"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "v5_spatial_best_model.pth"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "v5_spatial_last_model.pth"
VAL_REPORT_PATH      = PROJECT_DIR / "classification_report_val_v5_spatial.txt"
VAL_REPORT_CSV_PATH  = PROJECT_DIR / "classification_report_val_v5_spatial.csv"
VAL_CM_PNG_PATH      = PROJECT_DIR / "confusion_matrix_val_v5_spatial.png"

# Normalization constants (matching V4)
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)

# Hyperparameter defaults
DEFAULT_LR         = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE   = 10
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 3
NUM_WORKERS        = 0
SEED               = 42
TRANSISTOR_INDEX   = 9

# Baseline V4 metrics for comparison
V4_BASELINE_VAL_ACC = 64.91
V4_BASELINE_VAL_F1  = 0.5866
V4_BASELINE_EPOCH   = 45

SEP  = "=" * 82
SEP2 = "-" * 82


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ComponentCNNSpatial (V5-Spatial) on cleaned_dataset/."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"Maximum training epochs (default: {DEFAULT_MAX_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Initial learning rate (default: {DEFAULT_LR})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience on val macro F1 (default: {DEFAULT_PATIENCE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
# Transforms Pipeline (Identical to V4)
# ══════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    """Modest training augmentation matching train_v4.py."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def get_val_transforms():
    """Strictly deterministic validation preprocessing (NO random augmentation)."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════
# Square-Root Class Weight Computation
# ══════════════════════════════════════════════════════════════════════════
def compute_sqrt_class_weights(train_dataset: datasets.ImageFolder) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    """
    Compute square-root inverse class frequency weights matching train_v4.py:
        raw_w_c = sqrt( N / (C * N_c) )
        w_c = raw_w_c / mean(raw_w)
    """
    counts = Counter(train_dataset.targets)
    class_counts = [counts.get(i, 0) for i in range(NUM_CLASSES)]
    total_samples = len(train_dataset.targets)

    sqrt_raw = []
    linear_raw = []
    for count in class_counts:
        if count > 0:
            lin = total_samples / (NUM_CLASSES * count)
            sqrt_val = lin ** 0.5
        else:
            lin = 1.0
            sqrt_val = 1.0
        linear_raw.append(lin)
        sqrt_raw.append(sqrt_val)

    sqrt_weights = torch.tensor(sqrt_raw, dtype=torch.float32)
    sqrt_weights = sqrt_weights / sqrt_weights.mean()

    linear_weights = torch.tensor(linear_raw, dtype=torch.float32)
    linear_weights = linear_weights / linear_weights.mean()

    return sqrt_weights, class_counts, linear_weights


def verify_class_order(dataset_class_to_idx: dict) -> None:
    expected = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if dataset_class_to_idx != expected:
        sys.exit(
            f"FATAL: Class ordering mismatch!\n"
            f"  Expected: {expected}\n"
            f"  Received: {dataset_class_to_idx}"
        )


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    if not TRAIN_DIR.exists():
        sys.exit(f"ERROR: Training directory not found at: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        sys.exit(f"ERROR: Validation directory not found at: {VAL_DIR}")

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=get_train_transforms())
    val_dataset   = datasets.ImageFolder(root=str(VAL_DIR),   transform=get_val_transforms())

    verify_class_order(train_dataset.class_to_idx)
    verify_class_order(val_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, train_dataset, val_dataset


# ══════════════════════════════════════════════════════════════════════════
# Metrics and Evaluation
# ══════════════════════════════════════════════════════════════════════════
def compute_detailed_metrics(
    preds: list[int],
    labels: list[int],
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Compute overall accuracy, macro F1, and per-class precision/recall/F1."""
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

    recalls = []
    precisions = []
    f1_scores = []

    for c in range(num_classes):
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
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    else:
        macro_f1 = sum(f1_scores) / num_classes

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "actual_counts": actual_counts,
        "pred_counts": pred_counts,
        "tp_counts": tp_counts,
        "recalls": recalls,
        "precisions": precisions,
        "f1_scores": f1_scores,
    }


def compute_masked_9class_metrics(
    all_logits: torch.Tensor,
    all_labels: list[int],
) -> dict:
    """
    Computes validation metrics for the 9 permitted classes with transistor masked.
    Filters out true transistor samples from the evaluation set.
    """
    masked_logits = all_logits.clone()
    masked_logits[:, TRANSISTOR_INDEX] = -float("inf")
    masked_preds = masked_logits[:, :TRANSISTOR_INDEX].argmax(dim=1).cpu().tolist()

    # Filter to non-transistor instances
    non_trans_indices = [i for i, l in enumerate(all_labels) if l != TRANSISTOR_INDEX]
    filtered_preds  = [masked_preds[i] for i in non_trans_indices]
    filtered_labels = [all_labels[i] for i in non_trans_indices]

    metrics_9 = compute_detailed_metrics(filtered_preds, filtered_labels, num_classes=9)
    return metrics_9, filtered_preds, filtered_labels


# ══════════════════════════════════════════════════════════════════════════
# Training and Validation Loops
# ══════════════════════════════════════════════════════════════════════════
def handle_cuda_oom(batch_size: int, phase: str, err: Exception) -> None:
    print("\n" + SEP)
    print(f"ERROR: CUDA Out-Of-Memory (OOM) encountered during {phase}!")
    print(SEP)
    print(f"Detail: {err}")
    print(f"Current batch size: {batch_size}")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Allocated memory  : {allocated:.1f} MB")
        print(f"Reserved memory   : {reserved:.1f} MB")
    print(f"Please reduce batch size (e.g. python train_v5_spatial.py --batch-size {max(8, batch_size // 2)}).")
    print(SEP + "\n")
    sys.exit(1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" in str(err).lower():
                handle_cuda_oom(batch_size, "training step", err)
            raise

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return (running_loss / total), (100.0 * correct / total)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float, dict, torch.Tensor, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    logits_list = []

    for images, labels in loader:
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" in str(err).lower():
                handle_cuda_oom(batch_size, "validation step", err)
            raise

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        logits_list.append(logits.cpu())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / len(all_labels)
    metrics = compute_detailed_metrics(all_preds, all_labels, NUM_CLASSES)
    stacked_logits = torch.cat(logits_list, dim=0)

    return val_loss, metrics["accuracy"], metrics["macro_f1"], metrics, stacked_logits, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint and Logging Helpers
# ══════════════════════════════════════════════════════════════════════════
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    val_macro_f1: float,
    best_val_macro_f1: float,
    class_weights: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":                epoch,
        "val_loss":             val_loss,
        "val_accuracy":         val_accuracy,
        "val_macro_f1":         val_macro_f1,
        "best_val_macro_f1":    best_val_macro_f1,
        "class_names":          CLASS_NAMES,
        "num_classes":          NUM_CLASSES,
        "input_size":           INPUT_SIZE,
        "norm_mean":            NORM_MEAN,
        "norm_std":             NORM_STD,
        "class_weights":        class_weights.cpu().tolist(),
        "architecture":         "ComponentCNNSpatial",
        "spatial_pool_size":    (2, 2),
    }
    torch.save(checkpoint, path)


def init_history_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_9class_acc",
            "val_9class_macro_f1",
            "learning_rate",
        ])


def append_history_csv(
    path: Path,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_macro_f1: float,
    val_9class_acc: float,
    val_9class_macro_f1: float,
    learning_rate: float,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_loss:.4f}",
            f"{train_acc:.2f}",
            f"{val_loss:.4f}",
            f"{val_acc:.2f}",
            f"{val_macro_f1:.4f}",
            f"{val_9class_acc:.2f}",
            f"{val_9class_macro_f1:.4f}",
            f"{learning_rate:.6f}",
        ])


def save_validation_report(
    epoch: int,
    metrics_10: dict,
    metrics_9: dict,
    txt_path: Path,
    csv_path: Path,
) -> None:
    """Saves comprehensive validation performance reports in TXT and CSV formats."""
    # Write TXT report
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(SEP + "\n")
        f.write("  ComponentCNNSpatial (V5-Spatial) — Validation Performance Report\n")
        f.write(SEP + "\n")
        f.write(f"  Checkpoint Epoch           : {epoch}\n")
        f.write(f"  Overall Val Accuracy (10c) : {metrics_10['accuracy']:.2f}%\n")
        f.write(f"  Overall Val Macro F1 (10c) : {metrics_10['macro_f1']:.4f}\n")
        f.write(f"  Permitted Val Acc (9c mask): {metrics_9['accuracy']:.2f}%\n")
        f.write(f"  Permitted Val F1 (9c mask) : {metrics_9['macro_f1']:.4f}\n")
        f.write(f"  V4 Baseline Reference      : Val Acc={V4_BASELINE_VAL_ACC:.2f}%, Val F1={V4_BASELINE_VAL_F1:.4f} (Epoch {V4_BASELINE_EPOCH})\n")
        f.write(SEP2 + "\n\n")

        f.write("Per-Class Performance (Permitted 9 Classes):\n")
        f.write(f"  {'Class Name':<20} {'Support':>8} {'Pred':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
        f.write("  " + "-" * 72 + "\n")
        for i, name in enumerate(PERMITTED_CLASSES):
            supp = metrics_9["actual_counts"][i]
            pred = metrics_9["pred_counts"][i]
            prec = metrics_9["precisions"][i]
            rec  = metrics_9["recalls"][i]
            f1   = metrics_9["f1_scores"][i]
            f.write(f"  {name:<20} {supp:>8} {pred:>8} {prec:>9.2f}% {rec:>9.2f}% {f1:>10.4f}\n")
        f.write("  " + "-" * 72 + "\n")
        avg_prec = sum(metrics_9["precisions"]) / len(PERMITTED_CLASSES)
        avg_rec  = sum(metrics_9["recalls"]) / len(PERMITTED_CLASSES)
        f.write(f"  {'Macro Average':<20} {sum(metrics_9['actual_counts']):>8} {sum(metrics_9['pred_counts']):>8} {avg_prec:>9.2f}% {avg_rec:>9.2f}% {metrics_9['macro_f1']:>10.4f}\n")
        f.write(SEP + "\n")

    # Write CSV report
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "support", "predicted", "precision_pct", "recall_pct", "f1_score"])
        for i, name in enumerate(PERMITTED_CLASSES):
            writer.writerow([
                i,
                name,
                metrics_9["actual_counts"][i],
                metrics_9["pred_counts"][i],
                f"{metrics_9['precisions'][i]:.2f}",
                f"{metrics_9['recalls'][i]:.2f}",
                f"{metrics_9['f1_scores'][i]:.4f}",
            ])
        writer.writerow([])
        writer.writerow(["SUMMARY", "Permitted 9-Class Macro", sum(metrics_9["actual_counts"]), sum(metrics_9["pred_counts"]), f"{avg_prec:.2f}", f"{avg_rec:.2f}", f"{metrics_9['macro_f1']:.4f}"])
        writer.writerow(["BASELINE", "V4 Baseline Val Macro", "-", "-", "-", f"{V4_BASELINE_VAL_ACC:.2f}%", f"{V4_BASELINE_VAL_F1:.4f}"])


def save_confusion_matrix_plot(
    preds_9: list[int],
    labels_9: list[int],
    save_path: Path,
    epoch: int,
) -> None:
    """Renders and saves a 9-class validation confusion matrix heatmap."""
    if not (HAVE_SKLEARN and HAVE_MATPLOTLIB):
        return

    cm = confusion_matrix(labels_9, preds_9, labels=list(range(len(PERMITTED_CLASSES))))
    n = len(PERMITTED_CLASSES)

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PERMITTED_CLASSES, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(PERMITTED_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=10, labelpad=10)
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_title(f"ComponentCNNSpatial (V5) Validation Confusion Matrix (Epoch {epoch})", fontsize=11, pad=15)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    seed_everything(args.seed)

    print(SEP)
    print("  Electronic Component Classifier — V5-Spatial Controlled Experiment")
    print("  (ComponentCNNSpatial: 2x2 Spatial Pooling + 2-Stage MLP Classifier)")
    print(SEP)

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device              : {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Model           : {gpu_name}")
        print(f"  GPU Memory          : {gpu_mem:.1f} GB")
    print(f"  Input Resolution    : {INPUT_SIZE}x{INPUT_SIZE} RGB")
    print(f"  Batch Size          : {args.batch_size}")
    print(f"  Max Epochs          : {args.epochs}")
    print(f"  Initial LR          : {args.lr}")
    print(f"  Early Stopping      : {args.patience} epochs patience on Val Macro F1")
    print(f"  LR Scheduler        : ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")
    print(f"  Random Seed         : {args.seed}")
    print(SEP2)

    # 2. Data Loaders (Strictly Train & Val Splits — Test is Untouched)
    print("\nLoading datasets and configuring data augmentation …")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(args.batch_size)
    print(f"  Training samples    : {len(train_dataset)}")
    print(f"  Validation samples  : {len(val_dataset)} (Validation set ONLY — Test set untouched)")
    print(f"  Number of classes   : {NUM_CLASSES}")
    print("  Augmentation        : RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter")
    print("  Validation          : Deterministic (no random transforms)")
    print(SEP2)

    # 3. Square-Root Class Weights
    sqrt_weights, class_counts, lin_weights = compute_sqrt_class_weights(train_dataset)

    # 4. Model Initialization (From Scratch — No Partial Classifier Loading)
    model = ComponentCNNSpatial(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=sqrt_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-5,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model architecture  : ComponentCNNSpatial (model_v5_spatial.py)")
    print(f"  Model parameters    : {param_count:,} (vs 24,234 in V4 baseline, +{param_count - 24234:,} / +{(param_count - 24234)/24234*100:.1f}%)")
    print(f"  Classifier structure: Linear(256, 64) -> ReLU -> Linear(64, 10)")
    print(f"  Spatial pooling     : AdaptiveAvgPool2d((2, 2)) -> 256 features")
    print(SEP2)

    # 5. Checkpoints & History Setup
    init_history_csv(HISTORY_CSV_PATH)
    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_val_metrics_10 = None
    best_val_metrics_9 = None
    best_preds_9 = None
    best_labels_9 = None
    epochs_no_improve = 0

    header = (
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
        f"{'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>8}  "
        f"{'Val9 Acc':>8}  {'Val9 F1':>8}  {'LR':>9}  {'Time':>5}  Note"
    )
    print("\nStarting Training V5-Spatial:")
    print(header)
    print("  " + "-" * 102)

    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
        )

        (
            val_loss,
            val_acc,
            val_f1,
            val_metrics_10,
            val_logits,
            val_preds,
            val_labels,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
        )

        # 9-class masked validation metrics
        val_metrics_9, preds_9, labels_9 = compute_masked_9class_metrics(
            val_logits, val_labels
        )
        val_9_acc = val_metrics_9["accuracy"]
        val_9_f1  = val_metrics_9["macro_f1"]

        elapsed = time.time() - t0

        # Step LR Scheduler based on validation loss
        scheduler.step(val_loss)
        next_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint based on Validation Macro F1
        note = ""
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            best_val_metrics_10 = val_metrics_10
            best_val_metrics_9  = val_metrics_9
            best_preds_9 = preds_9
            best_labels_9 = labels_9
            epochs_no_improve = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_acc,
                val_macro_f1=val_f1,
                best_val_macro_f1=best_val_macro_f1,
                class_weights=sqrt_weights,
                path=BEST_CHECKPOINT_PATH,
            )
            note = "<- best F1"
        else:
            epochs_no_improve += 1
            note = f"no impr ({epochs_no_improve}/{args.patience})"

        # Always save last checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            best_val_macro_f1=best_val_macro_f1,
            class_weights=sqrt_weights,
            path=LAST_CHECKPOINT_PATH,
        )

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
            f"{val_loss:>9.4f}  {val_acc:>7.2f}%  {val_f1:>8.4f}  "
            f"{val_9_acc:>7.2f}%  {val_9_f1:>8.4f}  "
            f"{curr_lr:>9.2e}  {elapsed:>4.1f}s  {note}"
        )

        if next_lr < curr_lr:
            print(f"         └─> [LR Scheduler] Learning rate reduced: {curr_lr:.2e} -> {next_lr:.2e}")

        append_history_csv(
            path=HISTORY_CSV_PATH,
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            val_macro_f1=val_f1,
            val_9class_acc=val_9_acc,
            val_9class_macro_f1=val_9_f1,
            learning_rate=curr_lr,
        )

        if epochs_no_improve >= args.patience:
            print("\n" + SEP2)
            print(f"  Early stopping triggered!")
            print(f"  No improvement in validation macro F1 for {args.patience} consecutive epochs.")
            print(f"  Best validation macro F1 was {best_val_macro_f1:.4f} at epoch {best_epoch}.")
            print(SEP2)
            break

    # 7. Final Outputs and Comparative Reporting
    print("\n" + SEP)
    print("  Training V5-Spatial Complete!")
    print(SEP)
    print(f"  Best Val Macro F1  : {best_val_macro_f1:.4f}  (epoch {best_epoch})")
    print(f"  Best Checkpoint    : {BEST_CHECKPOINT_PATH.resolve()}")
    print(f"  Last Checkpoint    : {LAST_CHECKPOINT_PATH.resolve()}")
    print(f"  History CSV        : {HISTORY_CSV_PATH.resolve()}")

    if best_val_metrics_10 is not None and best_val_metrics_9 is not None:
        save_validation_report(
            epoch=best_epoch,
            metrics_10=best_val_metrics_10,
            metrics_9=best_val_metrics_9,
            txt_path=VAL_REPORT_PATH,
            csv_path=VAL_REPORT_CSV_PATH,
        )
        print(f"  Val Report (TXT)   : {VAL_REPORT_PATH.resolve()}")
        print(f"  Val Report (CSV)   : {VAL_REPORT_CSV_PATH.resolve()}")

        if best_preds_9 is not None and best_labels_9 is not None:
            save_confusion_matrix_plot(
                preds_9=best_preds_9,
                labels_9=best_labels_9,
                save_path=VAL_CM_PNG_PATH,
                epoch=best_epoch,
            )
            print(f"  Confusion Matrix   : {VAL_CM_PNG_PATH.resolve()}")

        print("\n" + SEP)
        print("  VALIDATION PERFORMANCE COMPARISON: V4 BASELINE vs. V5-SPATIAL")
        print(SEP)
        print(f"  {'Metric':<32} {'V4 Baseline':<18} {'V5-Spatial (Epoch ' + str(best_epoch) + ')':<22} {'Delta':<10}")
        print(SEP2)
        v5_acc = best_val_metrics_10['accuracy']
        v5_f1  = best_val_metrics_10['macro_f1']
        acc_delta = v5_acc - V4_BASELINE_VAL_ACC
        f1_delta  = v5_f1 - V4_BASELINE_VAL_F1
        print(f"  {'Val Accuracy (10-class)':<32} {V4_BASELINE_VAL_ACC:>8.2f}%         {v5_acc:>8.2f}%              {acc_delta:>+6.2f}%")
        print(f"  {'Val Macro F1 (10-class)':<32} {V4_BASELINE_VAL_F1:>10.4f}         {v5_f1:>10.4f}              {f1_delta:>+8.4f}")
        v5_9_acc = best_val_metrics_9['accuracy']
        v5_9_f1  = best_val_metrics_9['macro_f1']
        print(f"  {'Val Accuracy (9-class mask)':<32} {'N/A (N/A)':<18} {v5_9_acc:>8.2f}%")
        print(f"  {'Val Macro F1 (9-class mask)':<32} {'N/A (N/A)':<18} {v5_9_f1:>10.4f}")
        print(SEP)


if __name__ == "__main__":
    main()
