"""
train_v4.py
-----------
Training pipeline (v4) for ComponentCNN on cleaned_dataset/.

Experiment Motivation (v4)
--------------------------
1. Baseline (train.py):
   - Unweighted CrossEntropyLoss, LR=0.001, 20 epochs.
   - Result: 51.12% test accuracy, but 0.0% transistor recall due to sample imbalance.
2. Improved v2 (train_v2.py):
   - Linear inverse-frequency weights (transistor ~3.82x, IC ~3.49x, push_button ~0.51x), LR=0.001, 30 epochs.
   - Result: 46.35% test accuracy. Over-corrected: severe transistor overprediction and push-button underprediction.
3. Improved v3 (train_v3.py):
   - Softened square-root inverse weights (transistor ~1.68x, push_button ~0.61x), LR=0.0003, 30 epochs.
   - Result: 39.77% validation accuracy. Diagnostic audit proved the model was severely under-trained because
     LR=0.0003 slowed convergence (train loss was 1.907 at epoch 30 vs 1.521 in v2). Validation macro F1 was
     still actively increasing at epoch 30.
4. Experiment v4 (this script):
   - Best of both worlds:
     * Softer square-root inverse-frequency class weights from v3 (preventing overcorrection).
     * Faster learning rate from v2 (Adam lr=0.001) for robust convergence.
     * Extended training budget: up to 50 epochs.
     * Early stopping patience: 10 epochs on validation macro F1.
     * Retains training augmentation: horizontal flip (p=0.5), modest rotation (15°), and mild color jitter.
     * Selects best model using validation macro F1.

FPGA Deployment Safety:
- ComponentCNN architecture in model.py is completely unchanged (24,234 parameters).
- Input resolution: 64x64 RGB.
- Normalization: [-1.0, 1.0] (mean=0.5, std=0.5 per channel).
"""

import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Safe imports ──────────────────────────────────────────────────────────
try:
    from torchvision import datasets, transforms
except ImportError as exc:
    print(f"ERROR: torchvision could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)

try:
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# ── Model definition ──────────────────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE


# ══════════════════════════════════════════════════════════════════════════
# Hyperparameters and Configuration
# ══════════════════════════════════════════════════════════════════════════
SEED                     = 42
DEFAULT_MAX_EPOCHS       = 50      # Extended training budget
DEFAULT_BATCH_SIZE       = 32
DEFAULT_LR               = 0.001   # Restored faster learning rate from v2
DEFAULT_PATIENCE         = 10      # Early stopping patience on val macro F1
SCHEDULER_PATIENCE       = 3       # ReduceLROnPlateau patience on val loss
SCHEDULER_FACTOR         = 0.5     # Halve learning rate on plateau
NUM_WORKERS              = 0       # Safest on Windows PowerShell

# Fixed normalization matching baseline and FPGA fixed-point pipeline
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)

PROJECT_DIR       = Path(__file__).resolve().parent
CLEANED_DIR       = PROJECT_DIR / "cleaned_dataset"
TRAIN_DIR         = CLEANED_DIR / "train"
VAL_DIR           = CLEANED_DIR / "val"
CHECKPOINT_DIR    = PROJECT_DIR / "checkpoints"
HISTORY_CSV_PATH  = PROJECT_DIR / "training_history_v4.csv"

SEP  = "=" * 86
SEP2 = "-" * 86


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ComponentCNN v4 with square-root class weights, LR=0.001, and 50 epochs."
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
# Transforms Pipeline
# ══════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    """Modest training augmentation matching train_v3.py."""
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
    Compute square-root inverse class frequency weights matching train_v3.py:
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
            f"  Expected (model.py)  : {expected}\n"
            f"  Received (ImageFolder): {dataset_class_to_idx}"
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
    print(f"Please reduce batch size (e.g. python train_v4.py --batch-size {max(8, batch_size // 2)}).")
    print("Script will stop now rather than silently falling back to CPU.")
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
    batch_size: int,
) -> tuple[float, float, float, dict]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

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

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / len(all_labels)
    metrics = compute_detailed_metrics(all_preds, all_labels, NUM_CLASSES)

    return val_loss, metrics["accuracy"], metrics["macro_f1"], metrics


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint and CSV Logging
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
            f"{learning_rate:.6f}",
        ])


def print_per_class_table(metrics: dict, title: str = "Validation Metrics Breakdown") -> None:
    """Print clean ASCII table of per-class precision, recall, and F1."""
    print("\n" + SEP2)
    print(f"  {title}")
    print(SEP2)
    print(
        f"  {'Class Name':<20} "
        f"{'Support':>8} "
        f"{'Predicted':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1-Score':>10}"
    )
    print("  " + "-" * 74)
    for i, name in enumerate(CLASS_NAMES):
        supp = metrics["actual_counts"][i]
        pred = metrics["pred_counts"][i]
        prec = metrics["precisions"][i]
        rec  = metrics["recalls"][i]
        f1   = metrics["f1_scores"][i]
        print(
            f"  {name:<20} "
            f"{supp:>8} "
            f"{pred:>10} "
            f"{prec:>9.2f}% "
            f"{rec:>9.2f}% "
            f"{f1:>10.4f}"
        )
    print("  " + "-" * 74)
    avg_prec = sum(metrics["precisions"]) / NUM_CLASSES
    avg_rec  = sum(metrics["recalls"]) / NUM_CLASSES
    print(
        f"  {'Macro Average':<20} "
        f"{sum(metrics['actual_counts']):>8} "
        f"{sum(metrics['pred_counts']):>10} "
        f"{avg_prec:>9.2f}% "
        f"{avg_rec:>9.2f}% "
        f"{metrics['macro_f1']:>10.4f}"
    )
    print(SEP2)


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    seed_everything(args.seed)

    print(SEP)
    print("  Electronic Component Classifier — Training Pipeline v4")
    print("  (Square-Root Class Weights + LR=0.001 + Up to 50 Epochs + Macro F1 Patience=10)")
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

    # 2. Data Loaders
    print("\nLoading datasets and configuring data augmentation …")
    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(args.batch_size)
    print(f"  Training samples    : {len(train_dataset)}")
    print(f"  Validation samples  : {len(val_dataset)} (Validation set ONLY — Test set untouched)")
    print(f"  Number of classes   : {NUM_CLASSES}")
    print("  Augmentation        : RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter")
    print("  Validation          : Deterministic (no random transforms)")
    print(SEP2)

    # 3. Square-Root Class Weights Computation & Comparison
    sqrt_weights, class_counts, lin_weights = compute_sqrt_class_weights(train_dataset)
    print("\nClass Weight Comparison (v4 Square-Root vs v2 Linear Inverse Frequency):")
    print(f"  {'Class':<20} {'Train Samples':>13} {'v2 Linear Wt':>14} {'v4 Sqrt Wt':>14} {'Reduction':>14}")
    print("  " + "-" * 79)
    for idx, name in enumerate(CLASS_NAMES):
        lin_w = lin_weights[idx].item()
        sq_w  = sqrt_weights[idx].item()
        diff  = sq_w - lin_w
        print(f"  {name:<20} {class_counts[idx]:>13} {lin_w:>14.3f} {sq_w:>14.3f} {diff:>+14.3f}")
    print("  " + "-" * 79)
    print(f"  {'Mean Weight':<20} {sum(class_counts):>13} {lin_weights.mean().item():>14.3f} {sqrt_weights.mean().item():>14.3f}")
    print(SEP2)

    # 4. Model, Loss, Optimizer, Scheduler
    model = ComponentCNN(num_classes=NUM_CLASSES).to(device)
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
    print(f"\n  Model parameters    : {param_count:,} (ComponentCNN unchanged)")
    print(SEP2)

    # 5. Checkpoints & History Setup
    init_history_csv(HISTORY_CSV_PATH)
    best_model_path = CHECKPOINT_DIR / "v4_best_model.pth"
    last_model_path = CHECKPOINT_DIR / "v4_last_model.pth"

    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_val_metrics = None
    epochs_no_improve = 0

    header = (
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
        f"{'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>8}  "
        f"{'LR':>9}  {'Time':>5}  Note"
    )
    print("\nStarting Training v4:")
    print(header)
    print("  " + "-" * 80)

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

        val_loss, val_acc, val_f1, val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            batch_size=args.batch_size,
        )

        elapsed = time.time() - t0

        # Step LR Scheduler based on validation loss
        scheduler.step(val_loss)
        next_lr = optimizer.param_groups[0]["lr"]

        # Checkpoint based on Validation Macro F1
        note = ""
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            best_val_metrics = val_metrics
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
                path=best_model_path,
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
            path=last_model_path,
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
            learning_rate=curr_lr,
        )

        if epochs_no_improve >= args.patience:
            print("\n" + SEP2)
            print(f"  Early stopping triggered!")
            print(f"  No improvement in validation macro F1 for {args.patience} consecutive epochs.")
            print(f"  Best validation macro F1 was {best_val_macro_f1:.4f} at epoch {best_epoch}.")
            print(SEP2)
            break

    # 7. Final Detailed Report
    print("\n" + SEP)
    print("  Training v4 Complete!")
    print(SEP)
    print(f"  Best Val Macro F1  : {best_val_macro_f1:.4f}  (epoch {best_epoch})")
    print(f"  Best Checkpoint    : {best_model_path}")
    print(f"  Last Checkpoint    : {last_model_path}")
    print(f"  History CSV        : {HISTORY_CSV_PATH}")

    if best_val_metrics is not None:
        print_per_class_table(
            best_val_metrics,
            title=f"Best Model Validation Performance Breakdown (Epoch {best_epoch})",
        )

    print(SEP)
    print("  To evaluate this model on the held-out test set, run:")
    print("    python evaluate.py --checkpoint checkpoints/v4_best_model.pth --output-suffix _v4")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
