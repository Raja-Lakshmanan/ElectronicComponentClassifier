"""
train.py
--------
Train ComponentCNN on the cleaned electronic component dataset.

Usage
-----
    python train.py

What this script does
---------------------
1.  Loads cleaned_dataset/train and cleaned_dataset/val via ImageFolder.
2.  Verifies the class ordering matches model.CLASS_NAMES exactly.
3.  Trains for up to MAX_EPOCHS using CrossEntropyLoss + Adam.
4.  Saves best (val-accuracy) and last checkpoints to checkpoints/.
5.  Writes training_history.csv after every epoch.
6.  Does NOT touch the test set.

What this script does NOT do
-----------------------------
- Does not modify model.py, the dataset, or any package versions.
- Does not use random augmentation (deterministic pipeline for now).
- Does not evaluate on the test set.

Design notes for the ECE student
---------------------------------
- The normalization values (mean=0.5, std=0.5 per channel) are stored
  inside the checkpoint so that the same transform is applied at
  inference time and later during FPGA fixed-point quantization.
- The class_names list is saved in the checkpoint so that the FPGA
  inference output indices can be mapped back to human-readable labels.
"""

import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── Import torchvision (fail with a helpful message if missing) ────────────
try:
    from torchvision import datasets, transforms
except ImportError as exc:
    sys.exit(
        f"ERROR: torchvision could not be imported.\n"
        f"Detail: {exc}\n"
        f"Install it with:  pip install torchvision\n"
        f"Do NOT change torch or Python versions."
    )

# ── Import the model definition ───────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE

# ══════════════════════════════════════════════════════════════════════════
# Hyperparameters and paths
# ══════════════════════════════════════════════════════════════════════════
SEED         = 42
MAX_EPOCHS   = 20
BATCH_SIZE   = 32
LR           = 0.001
NUM_WORKERS  = 0           # safest on Windows; avoids multiprocessing issues

# Normalization: simple [0,1] → [-1,1] mapping (easy to replicate in HLS)
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)

PROJECT_DIR    = Path(__file__).resolve().parent
CLEANED_DIR    = PROJECT_DIR / "cleaned_dataset"
TRAIN_DIR      = CLEANED_DIR / "train"
VAL_DIR        = CLEANED_DIR / "val"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
HISTORY_CSV    = PROJECT_DIR / "training_history.csv"

SEP  = "=" * 72
SEP2 = "-" * 72


# ══════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolution algorithms (may reduce speed slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def get_transforms():
    """
    Deterministic transform pipeline (no random augmentation).

    Steps:
        1. Resize to INPUT_SIZE x INPUT_SIZE.
        2. Convert to RGB (handles any grayscale or RGBA images).
        3. To tensor (scales pixel values from [0, 255] to [0.0, 1.0]).
        4. Normalize each channel: (x - 0.5) / 0.5 → range [-1.0, 1.0].

    The same transform is used for train, val, and test.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def verify_class_order(dataset_class_to_idx: dict) -> None:
    """
    Verify that ImageFolder's class_to_idx maps class names to indices
    in the same order as CLASS_NAMES from model.py.

    ImageFolder sorts subdirectory names alphabetically by default.
    CLASS_NAMES is also alphabetical, so they should match.  If they
    don't, training labels would be wrong and the model would be useless.
    """
    expected = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    if dataset_class_to_idx != expected:
        print("FATAL: Class ordering mismatch!")
        print(f"  Expected (model.py) : {expected}")
        print(f"  Got (ImageFolder)   : {dataset_class_to_idx}")
        sys.exit(1)


def build_dataloaders():
    """Create train and val DataLoaders."""
    transform = get_transforms()

    if not TRAIN_DIR.exists():
        sys.exit(f"ERROR: Training directory not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        sys.exit(f"ERROR: Validation directory not found: {VAL_DIR}")

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=transform)
    val_dataset   = datasets.ImageFolder(root=str(VAL_DIR),   transform=transform)

    # Verify class ordering before any training happens
    verify_class_order(train_dataset.class_to_idx)
    verify_class_order(val_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════════════════
# Training and validation loops
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple:
    """
    Train for one epoch.
    Returns (average_loss, accuracy_percent).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        try:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        except torch.cuda.OutOfMemoryError:
            print(
                "\nERROR: GPU out of memory while transferring data.\n"
                f"Current batch size: {BATCH_SIZE}\n"
                "Try reducing BATCH_SIZE in train.py (e.g., 16 or 8)."
            )
            sys.exit(1)

        try:
            logits = model(images)
            loss = criterion(logits, labels)
        except torch.cuda.OutOfMemoryError:
            print(
                "\nERROR: GPU out of memory during forward pass.\n"
                f"Current batch size: {BATCH_SIZE}\n"
                "Try reducing BATCH_SIZE in train.py (e.g., 16 or 8)."
            )
            sys.exit(1)

        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print(
                "\nERROR: GPU out of memory during backward pass.\n"
                f"Current batch size: {BATCH_SIZE}\n"
                "Try reducing BATCH_SIZE in train.py (e.g., 16 or 8)."
            )
            sys.exit(1)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple:
    """
    Evaluate on validation set (no gradients).
    Returns (average_loss, accuracy_percent).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(dim=1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════
# Checkpointing
# ══════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_accuracy: float,
                    path: Path) -> None:
    """
    Save a checkpoint with everything needed to resume training
    or run standalone inference (including on the FPGA pipeline).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch":                epoch,
        "val_accuracy":         val_accuracy,
        "class_names":          CLASS_NAMES,
        "num_classes":          NUM_CLASSES,
        "input_size":           INPUT_SIZE,
        "norm_mean":            NORM_MEAN,
        "norm_std":             NORM_STD,
    }
    torch.save(checkpoint, path)


# ══════════════════════════════════════════════════════════════════════════
# Training history CSV
# ══════════════════════════════════════════════════════════════════════════

def init_history_csv(path: Path) -> None:
    """Create (or overwrite) the CSV with a header row."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy",
                         "val_loss", "val_accuracy"])


def append_history_csv(path: Path,
                       epoch: int,
                       train_loss: float,
                       train_acc: float,
                       val_loss: float,
                       val_acc: float) -> None:
    """Append one row to the training history CSV."""
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch,
                         f"{train_loss:.4f}",
                         f"{train_acc:.2f}",
                         f"{val_loss:.4f}",
                         f"{val_acc:.2f}"])


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    seed_everything(SEED)

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(SEP)
    print("  Electronic Component Classifier – Training")
    print(SEP)
    print(f"  Device       : {device}")
    if device.type == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )
        print(f"  GPU memory   : {gpu_mem:.1f} GB")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Max epochs   : {MAX_EPOCHS}")
    print(f"  Learning rate: {LR}")
    print(f"  Input size   : {INPUT_SIZE}x{INPUT_SIZE}")
    print(f"  Norm mean    : {NORM_MEAN}")
    print(f"  Norm std     : {NORM_STD}")
    print(f"  Seed         : {SEED}")
    print(SEP)

    # ── Data ───────────────────────────────────────────────────────────────
    print("\nLoading datasets …")
    train_loader, val_loader = build_dataloaders()
    print(f"  Train images : {len(train_loader.dataset)}")
    print(f"  Val images   : {len(val_loader.dataset)}")
    print(f"  Classes      : {NUM_CLASSES}")
    print(f"  Class order verified against model.CLASS_NAMES.")
    print()

    # ── Model, loss, optimizer ─────────────────────────────────────────────
    model     = ComponentCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params : {param_count:,}")
    print()

    # ── History CSV ────────────────────────────────────────────────────────
    init_history_csv(HISTORY_CSV)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0

    header = (f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
              f"{'Val Loss':>10}  {'Val Acc':>9}  {'Time':>6}  Note")
    print(header)
    print("  " + "-" * 75)

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0

        # ── Checkpoint logic ───────────────────────────────────────────────
        note = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            save_checkpoint(model, optimizer, epoch, val_acc,
                            CHECKPOINT_DIR / "best_model.pth")
            note = "<- best"

        save_checkpoint(model, optimizer, epoch, val_acc,
                        CHECKPOINT_DIR / "last_model.pth")

        # ── Log ────────────────────────────────────────────────────────────
        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
            f"{val_loss:>10.4f}  {val_acc:>8.2f}%  {elapsed:>5.1f}s  {note}"
        )

        append_history_csv(HISTORY_CSV, epoch,
                           train_loss, train_acc, val_loss, val_acc)

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"  Training complete.")
    print(f"  Best val accuracy : {best_val_acc:.2f}%  (epoch {best_epoch})")
    print(f"  Best checkpoint   : {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"  Last checkpoint   : {CHECKPOINT_DIR / 'last_model.pth'}")
    print(f"  History CSV       : {HISTORY_CSV}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
