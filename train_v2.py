"""
train_v2.py
-----------
Improved training pipeline (v2) for ComponentCNN on cleaned_dataset/.

Key Improvements over train.py
------------------------------
1. Training-Only Data Augmentation:
   - Modest random rotation (+/- 15 degrees)
   - Random horizontal flip (p=0.5)
   - Mild color jitter (brightness, contrast, saturation=0.2, hue=0.05)
   - Validation pipeline remains strictly deterministic.
2. Class-Weighted Loss:
   - Inverse-frequency class weights computed from training set labels.
   - Normalized so mean(weights) == 1.0.
   - Directly tackles severe minority class recall (e.g. transistor: 42 images,
     integrated_circuit: 46 images vs push_button: 317 images).
3. Learning Rate Scheduling:
   - Adam with initial lr=0.001.
   - ReduceLROnPlateau monitoring validation loss (halves lr on plateau).
4. Validation Macro F1 Tracking & Early Stopping:
   - Computes overall accuracy AND macro-averaged F1-score across all 10 classes.
   - Saves best model based on validation macro F1 (checkpoints/v2_best_model.pth).
   - Early stops after 8 consecutive epochs without validation macro F1 improvement.
5. Preserves FPGA Deployment Constraints:
   - ComponentCNN architecture in model.py is unchanged (~24,234 parameters).
   - Normalization remains mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5).
   - Input remains 64x64 RGB.
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

# ── Import torchvision safely ─────────────────────────────────────────────
try:
    from torchvision import datasets, transforms
except ImportError as exc:
    print(f"ERROR: torchvision could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)

# ── Import scikit-learn for metric verification if available ───────────────
try:
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# ── Model and Class Definitions ────────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE


# ══════════════════════════════════════════════════════════════════════════
# Configuration and Hyperparameters
# ══════════════════════════════════════════════════════════════════════════
SEED                     = 42
DEFAULT_MAX_EPOCHS       = 30
DEFAULT_BATCH_SIZE       = 32
DEFAULT_LR               = 0.001
DEFAULT_PATIENCE         = 8       # Early stopping patience (macro F1)
SCHEDULER_PATIENCE       = 3       # ReduceLROnPlateau patience (val loss)
SCHEDULER_FACTOR         = 0.5     # Learning rate reduction factor
NUM_WORKERS              = 0       # 0 is safest on Windows PowerShell

# Normalization: simple [-1.0, 1.0] scaling preserved for FPGA HLS
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)

PROJECT_DIR       = Path(__file__).resolve().parent
CLEANED_DIR       = PROJECT_DIR / "cleaned_dataset"
TRAIN_DIR         = CLEANED_DIR / "train"
VAL_DIR           = CLEANED_DIR / "val"
CHECKPOINT_DIR    = PROJECT_DIR / "checkpoints"
HISTORY_CSV_PATH  = PROJECT_DIR / "training_history_v2.csv"

SEP  = "=" * 82
SEP2 = "-" * 82


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ComponentCNN v2 with augmentation, class weighting, and macro F1 tracking."
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
    """Set seeds across Python, PyTorch, and CUDA for deterministic execution."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
# Preprocessing and Augmentation Pipelines
# ══════════════════════════════════════════════════════════════════════════
def get_train_transforms():
    """
    Training pipeline with data augmentation:
      1. Resize to 64x64
      2. Ensure 3-channel RGB
      3. Random horizontal flip (p=0.5)
      4. Modest random rotation (+/- 15 degrees)
      5. Mild color jitter (brightness/contrast/saturation=0.2, hue=0.05)
      6. Convert to tensor in [0.0, 1.0]
      7. Normalization: (x - 0.5) / 0.5 -> [-1.0, 1.0]
    """
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
    """
    Deterministic validation pipeline (strictly NO random augmentation):
      1. Resize to 64x64
      2. Ensure 3-channel RGB
      3. Convert to tensor in [0.0, 1.0]
      4. Normalization: (x - 0.5) / 0.5 -> [-1.0, 1.0]
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ══════════════════════════════════════════════════════════════════════════
# Dataset Verification & Class Weights
# ══════════════════════════════════════════════════════════════════════════
def verify_class_order(dataset_class_to_idx: dict) -> None:
    """Verify that ImageFolder class_to_idx matches CLASS_NAMES from model.py."""
    expected = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if dataset_class_to_idx != expected:
        print("FATAL: Class ordering mismatch!")
        print(f"  Expected (model.py)  : {expected}")
        print(f"  Received (ImageFolder): {dataset_class_to_idx}")
        sys.exit(1)


def compute_class_weights(train_dataset: datasets.ImageFolder) -> tuple[torch.Tensor, list[int]]:
    """
    Compute balanced inverse-frequency class weights from training labels:
        w_c = N / (C * N_c)
    Normalized so that mean(weights) == 1.0.
    """
    # Count occurrences of each class index
    counts = Counter(train_dataset.targets)
    class_counts = [counts.get(i, 0) for i in range(NUM_CLASSES)]

    total_samples = len(train_dataset.targets)
    weights = []
    for count in class_counts:
        if count > 0:
            w = total_samples / (NUM_CLASSES * count)
        else:
            w = 1.0
        weights.append(w)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Sensible normalization: average weight across classes equals 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()

    return weights_tensor, class_counts


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, datasets.ImageFolder, datasets.ImageFolder]:
    """Create train and validation DataLoaders."""
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
# Metrics: Macro-Averaged F1
# ══════════════════════════════════════════════════════════════════════════
def compute_macro_f1(preds: list[int], labels: list[int], num_classes: int = NUM_CLASSES) -> float:
    """
    Compute macro-averaged F1-score across all classes:
        F1_c = 2 * (P_c * R_c) / (P_c + R_c)
        macro_F1 = mean(F1_c for c in classes)
    Gracefully handles zero-division (0.0 when TP == 0).
    Matches sklearn.metrics.f1_score(..., average='macro', zero_division=0).
    """
    if HAVE_SKLEARN:
        return float(f1_score(labels, preds, average="macro", zero_division=0))

    # Pure Python implementation fallback
    f1_list = []
    for c in range(num_classes):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_list.append(f1)

    return sum(f1_list) / num_classes


# ══════════════════════════════════════════════════════════════════════════
# Training and Validation Loops
# ══════════════════════════════════════════════════════════════════════════
def handle_cuda_oom(batch_size: int, phase: str, err: Exception) -> None:
    """Clean exit on GPU Out-of-Memory; does not silently fallback to CPU."""
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
    print("\nAction required:")
    print(f"  Please reduce the batch size (e.g. from {batch_size} to 16 or 8):")
    print(f"    python train_v2.py --batch-size {max(8, batch_size // 2)}")
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
    """Train for one epoch. Returns (average_loss, accuracy_percentage)."""
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

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float, float]:
    """
    Evaluate on validation set without gradients.
    Returns (average_loss, accuracy_percentage, macro_f1).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
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
        _, preds = logits.max(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    macro_f1 = compute_macro_f1(all_preds, all_labels, num_classes=NUM_CLASSES)

    return epoch_loss, epoch_acc, macro_f1


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint & History Helpers
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
    """Save full model and training state checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        # Core model & optimizer
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        # Training progress
        "epoch":                epoch,
        "val_loss":             val_loss,
        "val_accuracy":         val_accuracy,
        "val_macro_f1":         val_macro_f1,
        "best_val_macro_f1":    best_val_macro_f1,
        # Metadata required by evaluate.py and FPGA export
        "class_names":          CLASS_NAMES,
        "num_classes":          NUM_CLASSES,
        "input_size":           INPUT_SIZE,
        "norm_mean":            NORM_MEAN,
        "norm_std":             NORM_STD,
        "class_weights":        class_weights.cpu().tolist(),
    }
    torch.save(checkpoint, path)


def init_history_csv(path: Path) -> None:
    """Initialize history CSV with header row."""
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
    """Append one epoch's metrics to history CSV."""
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


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    seed_everything(args.seed)

    print(SEP)
    print("  Electronic Component Classifier — Training Pipeline v2")
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
    print(f"  Validation samples  : {len(val_dataset)}")
    print(f"  Number of classes   : {NUM_CLASSES}")
    print("  Data augmentation   : RandomRotation(15°), RandomHorizontalFlip(0.5), ColorJitter")
    print("  Validation pipeline : Deterministic resize + RGB + normalize (no augmentation)")
    print(SEP2)

    # 3. Class Weights Computation
    class_weights, class_counts = compute_class_weights(train_dataset)
    print("\nComputed Class Weights (Inverse Frequency Normalized to Mean = 1.0):")
    print(f"  {'Class':<20} {'Train Samples':>13} {'Computed Weight':>16}")
    print("  " + "-" * 52)
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {name:<20} {class_counts[idx]:>13} {class_weights[idx].item():>16.3f}")
    print(f"  {'Mean Weight':<20} {sum(class_counts):>13} {class_weights.mean().item():>16.3f}")
    print(SEP2)

    # 4. Model, Loss, Optimizer, Scheduler
    model = ComponentCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
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

    # 5. History and Paths
    init_history_csv(HISTORY_CSV_PATH)
    best_model_path = CHECKPOINT_DIR / "v2_best_model.pth"
    last_model_path = CHECKPOINT_DIR / "v2_last_model.pth"

    # 6. Training Loop with Macro F1 Tracking and Early Stopping
    best_val_macro_f1 = 0.0
    best_epoch = 0
    epochs_no_improve = 0

    header = (
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
        f"{'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>8}  "
        f"{'LR':>9}  {'Time':>5}  Note"
    )
    print("\nStarting Training v2:")
    print(header)
    print("  " + "-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        curr_lr = optimizer.param_groups[0]["lr"]

        # Train & Validate
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            batch_size=args.batch_size,
        )
        val_loss, val_acc, val_f1 = validate(
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

        # Checkpoint Selection based on Validation Macro F1
        note = ""
        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
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
                class_weights=class_weights,
                path=best_model_path,
            )
            note = "<- best F1"
        else:
            epochs_no_improve += 1
            note = f"no impr ({epochs_no_improve}/{args.patience})"

        # Always save last model checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_macro_f1=val_f1,
            best_val_macro_f1=best_val_macro_f1,
            class_weights=class_weights,
            path=last_model_path,
        )

        # Log row to console
        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
            f"{val_loss:>9.4f}  {val_acc:>7.2f}%  {val_f1:>8.4f}  "
            f"{curr_lr:>9.2e}  {elapsed:>4.1f}s  {note}"
        )

        if next_lr < curr_lr:
            print(f"         └─> [LR Scheduler] Learning rate reduced: {curr_lr:.2e} -> {next_lr:.2e}")

        # Append to CSV
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

        # Early Stopping Check
        if epochs_no_improve >= args.patience:
            print("\n" + SEP2)
            print(f"  Early stopping triggered!")
            print(f"  No improvement in validation macro F1 for {args.patience} consecutive epochs.")
            print(f"  Best validation macro F1 was {best_val_macro_f1:.4f} at epoch {best_epoch}.")
            print(SEP2)
            break

    # 7. Summary
    print("\n" + SEP)
    print("  Training v2 Complete!")
    print(SEP)
    print(f"  Best Val Macro F1  : {best_val_macro_f1:.4f}  (epoch {best_epoch})")
    print(f"  Best Checkpoint    : {best_model_path}")
    print(f"  Last Checkpoint    : {last_model_path}")
    print(f"  History CSV        : {HISTORY_CSV_PATH}")
    print(SEP)
    print("  Model is ready for test evaluation.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
