"""
train_9class.py
---------------
Training pipeline for 9-class ComponentCNN on cleaned_dataset/, strictly
filtering out the transistor class from training, validation, and evaluation.

Key Requirements:
  1. Only 9 target classes:
     capacitor, diode, fuse, inductor, integrated_circuit,
     led, potentiometer, push_button, resistor
  2. Transistor class completely excluded in data loaders (no dataset files modified).
  3. Uses ComponentCNN9 (24,169 parameters, ~94.4 KB).
  4. 64x64 RGB input with fixed normalization: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5).
  5. Modest data augmentation: RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter.
  6. Square-root inverse-frequency class weights (normalized to mean 1.0) to handle class imbalance gently.
  7. Adam optimizer (lr=0.001), ReduceLROnPlateau (factor=0.5, patience=3).
  8. Early stopping (default: 10 epochs patience) monitored on validation macro F1.
  9. Saves best checkpoint to checkpoints/9class_best_model.pth.
 10. Saves training history to training_history_9class.csv, validation confusion matrix to
     confusion_matrix_val_9class.png, and validation metrics report to classification_report_val_9class.csv.
 11. Strictly uses train and val splits only. Test split is untouched.
"""

import argparse
import csv
import random
import sys
import time
from collections import Counter
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

# ── Defaults & Hyperparameters ─────────────────────────────────────────────
SEED                 = 42
DEFAULT_MAX_EPOCHS   = 50
DEFAULT_BATCH_SIZE   = 32
DEFAULT_LR           = 0.001
DEFAULT_PATIENCE     = 10      # Early stopping patience on Val Macro F1
SCHEDULER_PATIENCE   = 3       # ReduceLROnPlateau patience on Val Loss
SCHEDULER_FACTOR     = 0.5
NUM_WORKERS          = 0       # Safe for Windows environment

# Normalization constants (matching project standard)
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)

PROJECT_DIR       = Path(__file__).resolve().parent
CLEANED_DIR       = PROJECT_DIR / "cleaned_dataset"
TRAIN_DIR         = CLEANED_DIR / "train"
VAL_DIR           = CLEANED_DIR / "val"
CHECKPOINT_DIR    = PROJECT_DIR / "checkpoints"
HISTORY_CSV_PATH  = PROJECT_DIR / "training_history_9class.csv"
VAL_REPORT_CSV    = PROJECT_DIR / "classification_report_val_9class.csv"
VAL_CM_PNG        = PROJECT_DIR / "confusion_matrix_val_9class.png"

SEP  = "=" * 86
SEP2 = "-" * 86


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 9-class ComponentCNN (excluding transistor) with balanced weighting and macro F1 checkpointing."
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
        "--weights",
        type=str,
        choices=["sqrt", "none", "linear"],
        default="sqrt",
        help="Class weight strategy: 'sqrt' (recommended, balanced), 'none', or 'linear' (default: sqrt)",
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


def get_train_transforms():
    """Modest training augmentation matching project v3/v4 guidelines."""
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


def build_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader, NineClassDataset, NineClassDataset]:
    if not TRAIN_DIR.exists():
        sys.exit(f"ERROR: Training directory not found at: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        sys.exit(f"ERROR: Validation directory not found at: {VAL_DIR}")

    train_dataset = NineClassDataset(root=str(TRAIN_DIR), transform=get_train_transforms())
    val_dataset   = NineClassDataset(root=str(VAL_DIR),   transform=get_val_transforms())

    # Verify fixed mapping
    if train_dataset.class_to_idx != CLASS_TO_IDX_9:
        sys.exit(f"FATAL: Train class mapping mismatch! Got {train_dataset.class_to_idx}")
    if val_dataset.class_to_idx != CLASS_TO_IDX_9:
        sys.exit(f"FATAL: Val class mapping mismatch! Got {val_dataset.class_to_idx}")

    # Confirm transistor is 100% excluded
    if "transistor" in train_dataset.classes or "transistor" in val_dataset.classes:
        sys.exit("FATAL: Transistor class detected in 9-class dataset!")

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
# Class Weight Calculations
# ══════════════════════════════════════════════════════════════════════════
def compute_class_weights(train_dataset: NineClassDataset, strategy: str) -> Tuple[torch.Tensor, List[int]]:
    """
    Compute class weights for the 9 classes:
      - 'none'   : all 1.0
      - 'sqrt'   : sqrt(N / (C * N_c)), normalized to mean 1.0 (gentle, recommended)
      - 'linear' : N / (C * N_c), normalized to mean 1.0
    """
    counts = Counter(train_dataset.targets)
    class_counts = [counts.get(i, 0) for i in range(NUM_CLASSES_9)]
    total_samples = len(train_dataset.targets)

    if strategy == "none":
        weights = torch.ones(NUM_CLASSES_9, dtype=torch.float32)
        return weights, class_counts

    raw_weights = []
    for count in class_counts:
        if count > 0:
            lin = total_samples / (NUM_CLASSES_9 * count)
            val = (lin ** 0.5) if strategy == "sqrt" else lin
        else:
            val = 1.0
        raw_weights.append(val)

    w = torch.tensor(raw_weights, dtype=torch.float32)
    w = w / w.mean()  # Normalize mean to 1.0
    return w, class_counts


# ══════════════════════════════════════════════════════════════════════════
# Metrics and Evaluation
# ══════════════════════════════════════════════════════════════════════════
def compute_metrics(preds: List[int], labels: List[int]) -> Dict:
    """Compute overall accuracy, macro F1, and per-class precision/recall/F1/support."""
    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    actual_counts = [0] * NUM_CLASSES_9
    pred_counts   = [0] * NUM_CLASSES_9
    tp_counts     = [0] * NUM_CLASSES_9

    for p, l in zip(preds, labels):
        actual_counts[l] += 1
        pred_counts[p] += 1
        if p == l:
            tp_counts[l] += 1

    recalls = []
    precisions = []
    f1_scores = []

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
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    else:
        macro_f1 = sum(f1_scores) / NUM_CLASSES_9

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


def save_confusion_matrix_plot(cm: np.ndarray, path: Path, title: str) -> None:
    """Render and save confusion matrix heatmap."""
    if not HAVE_MATPLOTLIB:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    ax.set(
        xticks=np.arange(NUM_CLASSES_9),
        yticks=np.arange(NUM_CLASSES_9),
        xticklabels=CLASS_NAMES_9,
        yticklabels=CLASS_NAMES_9,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(NUM_CLASSES_9):
        for j in range(NUM_CLASSES_9):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:d}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(metrics: Dict, path: Path) -> None:
    """Save per-class precision/recall/F1/support to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
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
# Checkpointing and History CSV
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
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch":                epoch,
        "val_loss":             val_loss,
        "val_accuracy":         val_accuracy,
        "val_macro_f1":         val_macro_f1,
        "best_val_macro_f1":    best_val_macro_f1,
        "class_names":          CLASS_NAMES_9,
        "class_to_idx":         CLASS_TO_IDX_9,
        "num_classes":          NUM_CLASSES_9,
        "input_size":           INPUT_SIZE,
        "norm_mean":            NORM_MEAN,
        "norm_std":             NORM_STD,
        "class_weights":        class_weights.cpu().tolist() if class_weights is not None else None,
        "model_type":           "ComponentCNN9",
    }
    torch.save(checkpoint, path)


def init_history_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
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
    lr: float,
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
            f"{lr:.2e}",
        ])


# ══════════════════════════════════════════════════════════════════════════
# Training and Validation Epochs
# ══════════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return (running_loss / total), (100.0 * correct / total)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, Dict, List[int], List[int]]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / len(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    return val_loss, metrics["accuracy"], metrics["macro_f1"], metrics, all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════
# Main Training Routine
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    seed_everything(args.seed)

    print()
    print(SEP)
    print("  Electronic Component Classifier — 9-Class Experiment (ComponentCNN9)")
    print("  (Transistor completely excluded from training, validation, and outputs)")
    print(SEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device              : {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Model           : {gpu_name} ({gpu_mem:.1f} GB VRAM)")

    # 1. Load Data
    print("\n[1/5] Loading 9-class datasets (filtering out transistor) …")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(args.batch_size)

    print(f"  Training samples    : {len(train_dataset)}")
    print(f"  Validation samples  : {len(val_dataset)} (Validation only — Test set strictly untouched)")
    print(f"  Number of classes   : {NUM_CLASSES_9}")
    print(f"  Input resolution    : {INPUT_SIZE}x{INPUT_SIZE} RGB")
    print(f"  Normalization       : mean={NORM_MEAN}, std={NORM_STD}")
    print(f"  Batch size          : {args.batch_size}")
    print(f"  Data augmentation   : RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter")
    print(SEP2)

    # 2. Class Weight Strategy
    print(f"\n[2/5] Computing class weights (strategy: {args.weights}) …")
    weights, counts = compute_class_weights(train_dataset, args.weights)
    print(f"  {'Class':<22} {'Train Samples':>14} {'Class Weight':>14}")
    print("  " + "-" * 52)
    for idx, name in enumerate(CLASS_NAMES_9):
        print(f"  {name:<22} {counts[idx]:>14} {weights[idx].item():>14.4f}")
    print("  " + "-" * 52)
    print(f"  {'Total / Mean':<22} {sum(counts):>14} {weights.mean().item():>14.4f}")
    print(SEP2)

    # 3. Model, Loss, Optimizer, Scheduler
    print("\n[3/5] Instantiating ComponentCNN9 architecture …")
    model = ComponentCNN9(num_classes=NUM_CLASSES_9).to(device)
    param_count, model_size_kb = count_parameters(model)
    print(f"  Trainable parameters: {param_count:,}")
    print(f"  Model size (FP32)   : {model_size_kb:.2f} KB (94.4 KB)")
    print(f"  Output classes      : {NUM_CLASSES_9} (transistor completely excluded)")

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=1e-5,
    )
    print(f"  Optimizer           : Adam (lr={args.lr})")
    print(f"  LR Scheduler        : ReduceLROnPlateau (factor={SCHEDULER_FACTOR}, patience={SCHEDULER_PATIENCE})")
    print(f"  Early stopping      : {args.patience} epochs on validation macro F1")
    print(SEP2)

    # 4. Checkpoint & History paths
    best_checkpoint_path = CHECKPOINT_DIR / "9class_best_model.pth"
    last_checkpoint_path = CHECKPOINT_DIR / "9class_last_model.pth"
    init_history_csv(HISTORY_CSV_PATH)

    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_metrics = None
    best_preds = []
    best_labels = []
    epochs_no_improve = 0

    print(f"\n[4/5] Starting training loop (up to {args.epochs} epochs) …")
    header = (
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
        f"{'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>8}  "
        f"{'LR':>9}  {'Time':>5}  Note"
    )
    print(header)
    print("  " + "-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, val_metrics, preds, labels = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        next_lr = optimizer.param_groups[0]["lr"]

        note = ""
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            best_metrics = val_metrics
            best_preds = preds
            best_labels = labels
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
                class_weights=weights,
                path=best_checkpoint_path,
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
            class_weights=weights,
            path=last_checkpoint_path,
        )

        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
            f"{val_loss:>9.4f}  {val_acc:>7.2f}%  {val_f1:>8.4f}  "
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
            lr=curr_lr,
        )

        if epochs_no_improve >= args.patience:
            print(f"\n[Early Stopping] Triggered after {args.patience} epochs without validation macro F1 improvement.")
            break

    # 5. Save Final Validation Artifacts
    print("\n" + SEP)
    print("  9-CLASS TRAINING COMPLETED")
    print(SEP)
    print(f"  Best Epoch              : {best_epoch}")
    print(f"  Best Val Macro F1       : {best_val_macro_f1:.4f}")
    if best_metrics:
        print(f"  Best Val Accuracy       : {best_metrics['accuracy']:.2f}%")
    print(f"  Best Checkpoint Saved   : {best_checkpoint_path}")
    print(f"  Training History CSV    : {HISTORY_CSV_PATH}")

    # Generate and save validation confusion matrix and report
    if best_metrics and best_preds and best_labels:
        save_metrics_csv(best_metrics, VAL_REPORT_CSV)
        print(f"  Val Classification CSV  : {VAL_REPORT_CSV}")

        if HAVE_SKLEARN and HAVE_MATPLOTLIB:
            cm = confusion_matrix(best_labels, best_preds, labels=list(range(NUM_CLASSES_9)))
            cm_title = f"ComponentCNN-9 Validation Confusion Matrix (Epoch {best_epoch})\nAccuracy: {best_metrics['accuracy']:.2f}% | Macro F1: {best_val_macro_f1:.4f}"
            save_confusion_matrix_plot(cm, VAL_CM_PNG, cm_title)
            print(f"  Val Confusion Matrix PNG: {VAL_CM_PNG}")

        print("\nBest Validation Per-Class Breakdown:")
        print(f"  {'Class':<22} {'Support':>8} {'Preds':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("  " + "-" * 66)
        for idx, name in enumerate(CLASS_NAMES_9):
            print(
                f"  {name:<22} "
                f"{best_metrics['actual_counts'][idx]:>8} "
                f"{best_metrics['pred_counts'][idx]:>8} "
                f"{best_metrics['precisions'][idx]:>7.1f}% "
                f"{best_metrics['recalls'][idx]:>7.1f}% "
                f"{best_metrics['f1_scores'][idx]:>8.4f}"
            )
        print("  " + "-" * 66)
        avg_prec = sum(best_metrics["precisions"]) / NUM_CLASSES_9
        avg_rec  = sum(best_metrics["recalls"]) / NUM_CLASSES_9
        print(
            f"  {'Macro Average':<22} "
            f"{sum(best_metrics['actual_counts']):>8} "
            f"{sum(best_metrics['pred_counts']):>8} "
            f"{avg_prec:>7.1f}% "
            f"{avg_rec:>7.1f}% "
            f"{best_val_macro_f1:>8.4f}"
        )
    print(SEP + "\n")


if __name__ == "__main__":
    main()
