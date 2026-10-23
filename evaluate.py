"""
evaluate.py
-----------
Evaluate the best saved ComponentCNN checkpoint on the held-out test set.

Usage
-----
    python evaluate.py                                            # Default: checkpoints/v2_best_model.pth
    python evaluate.py --checkpoint checkpoints/best_model.pth    # Original baseline checkpoint

What this script does
---------------------
1.  Loads checkpoints/best_model.pth (reads norm values and class names
    from the checkpoint itself for full traceability).
2.  Applies the same 64x64 RGB resize + normalize pipeline used during
    training.
3.  Runs inference on cleaned_dataset/test/ — one forward pass, no
    gradient computation.
4.  Reports overall test accuracy and per-class precision / recall /
    F1-score / support.
5.  Saves:
        confusion_matrix.png         – colour-coded confusion matrix
        classification_report.txt    – full text report

Metrics primer (for ECE students)
---------------------------------
  Precision : Of all images the model *predicted* as class X, what
              fraction actually *are* class X?
              → High precision = few false positives.
  Recall    : Of all images that actually *are* class X, what fraction
              did the model correctly *predict* as X?
              → High recall = few false negatives.
  F1-score  : Harmonic mean of precision and recall.  Balances both.
              F1 = 2 * (P * R) / (P + R)
  Support   : Number of true instances of that class in the test set.
  Confusion matrix (row i, col j) : number of images whose true label is
              class i that the model predicted as class j.  The diagonal
              is correct predictions; off-diagonal entries are errors.

What this script does NOT do
-----------------------------
- Does not train or fine-tune the model.
- Does not modify checkpoints, datasets, or packages.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── torchvision ────────────────────────────────────────────────────────────
try:
    from torchvision import datasets, transforms
except ImportError as exc:
    sys.exit(
        f"ERROR: torchvision could not be imported.\n"
        f"Detail: {exc}\n"
        f"Do NOT install or upgrade packages automatically."
    )

# ── scikit-learn (for classification report & confusion matrix) ────────────
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
except ImportError as exc:
    sys.exit(
        f"ERROR: scikit-learn could not be imported.\n"
        f"Detail: {exc}\n"
        f"Install it with:  pip install scikit-learn\n"
        f"Do NOT change torch or Python versions."
    )

# ── matplotlib (for confusion matrix plot) ─────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    sys.exit(
        f"ERROR: matplotlib or numpy could not be imported.\n"
        f"Detail: {exc}\n"
        f"Install with:  pip install matplotlib numpy\n"
        f"Do NOT change torch or Python versions."
    )

# ── Model definition ──────────────────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE

# ══════════════════════════════════════════════════════════════════════════
# Paths and constants
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR    = Path(__file__).resolve().parent
CLEANED_DIR    = PROJECT_DIR / "cleaned_dataset"
TEST_DIR       = CLEANED_DIR / "test"
DEFAULT_CHECKPOINT = "checkpoints/v2_best_model.pth"

BATCH_SIZE  = 32
NUM_WORKERS = 0

SEP  = "=" * 72
SEP2 = "-" * 72


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ComponentCNN checkpoint on cleaned_dataset/test/."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint to evaluate (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional suffix for outputs (default: '_v2' for v2 checkpoints, '' for baseline)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: Path, device: torch.device):
    """
    Load a checkpoint saved by train.py or train_v2.py.

    Checkpoint keys (from train.py save_checkpoint):
        model_state_dict, optimizer_state_dict, epoch,
        val_accuracy, class_names, num_classes,
        input_size, norm_mean, norm_std
    """
    if not path.exists():
        sys.exit(
            f"ERROR: Checkpoint not found: {path}\n"
            f"If evaluating the original baseline checkpoint, run:\n"
            f"    python evaluate.py --checkpoint checkpoints/best_model.pth"
        )

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Validate expected keys
    required_keys = [
        "model_state_dict", "epoch", "val_accuracy",
        "class_names", "num_classes", "input_size",
        "norm_mean", "norm_std",
    ]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        sys.exit(
            f"ERROR: Checkpoint is missing keys: {missing}\n"
            f"Expected keys saved by train.py: {required_keys}"
        )

    return ckpt


def build_test_loader(test_dir: Path,
                      input_size: int,
                      norm_mean: tuple,
                      norm_std: tuple) -> DataLoader:
    """
    Build a DataLoader for the test set using the exact same
    preprocessing pipeline as training (deterministic, no augmentation).
    """
    if not test_dir.exists():
        sys.exit(f"ERROR: Test directory not found: {test_dir}")

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader, test_dataset


def verify_class_order(dataset_class_to_idx: dict,
                       checkpoint_class_names: list) -> None:
    """
    Verify that ImageFolder's class_to_idx matches the class ordering
    saved in the checkpoint (which itself was verified against model.py
    during training).
    """
    expected = {name: idx for idx, name in enumerate(checkpoint_class_names)}
    if dataset_class_to_idx != expected:
        print("FATAL: Class ordering mismatch!")
        print(f"  Checkpoint class_names : {checkpoint_class_names}")
        print(f"  ImageFolder class_to_idx: {dataset_class_to_idx}")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model: nn.Module,
                  loader: DataLoader,
                  device: torch.device) -> tuple:
    """
    Run the model on every batch in *loader*.

    Returns
    -------
    all_preds  : list[int]   – predicted class index per image
    all_labels : list[int]   – ground-truth class index per image
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)                       # (B, num_classes)
        preds  = logits.argmax(dim=1).cpu().tolist()  # list of ints
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_preds, all_labels


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════

def compute_overall_accuracy(preds: list, labels: list) -> float:
    correct = sum(p == l for p, l in zip(preds, labels))
    return 100.0 * correct / len(labels)


def build_classification_report(preds: list,
                                labels: list,
                                class_names: list) -> str:
    """
    Build a human-readable classification report.

    Uses sklearn.metrics.classification_report which produces
    per-class precision, recall, F1, support plus macro/weighted
    averages.
    """
    return classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )


def find_confused_pairs(cm, class_names: list, top_n: int = 5) -> list:
    """
    From the confusion matrix, find the most frequently confused
    (true_class, predicted_class) pairs — excluding correct predictions
    (the diagonal).

    Returns a list of (true_cls, pred_cls, count) sorted descending.
    """
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i][j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def plot_confusion_matrix(cm, class_names: list, save_path: Path) -> None:
    """
    Plot and save a colour-coded confusion matrix.

    Rows = true labels, columns = predicted labels.
    Diagonal cells (correct) use a stronger colour.
    """
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a blue colour map — darker = higher count
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Labels
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix – Test Set",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annotate each cell with the count
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_arguments()

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        if (PROJECT_DIR / checkpoint_path).exists():
            checkpoint_path = PROJECT_DIR / checkpoint_path
        else:
            checkpoint_path = (PROJECT_DIR / checkpoint_path).resolve()

    # Distinguish output filenames for v2 vs original model
    if args.output_suffix is not None:
        suffix = args.output_suffix
    elif "v2" in checkpoint_path.name.lower():
        suffix = "_v2"
    else:
        suffix = ""

    cm_png = PROJECT_DIR / f"confusion_matrix{suffix}.png"
    report_txt = PROJECT_DIR / f"classification_report{suffix}.txt"

    # ── Device ─────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    print(SEP)
    print("  Electronic Component Classifier – Test Set Evaluation")
    print(SEP)
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoint  : {checkpoint_path}")
    print(f"  Test folder : {TEST_DIR}")
    print(SEP)

    # ── Load checkpoint ────────────────────────────────────────────────────
    print("\n[1/5] Loading checkpoint …")
    ckpt = load_checkpoint(checkpoint_path, device)

    ckpt_class_names = ckpt["class_names"]
    ckpt_num_classes = ckpt["num_classes"]
    ckpt_input_size  = ckpt["input_size"]
    ckpt_norm_mean   = ckpt["norm_mean"]
    ckpt_norm_std    = ckpt["norm_std"]
    ckpt_epoch       = ckpt["epoch"]
    ckpt_val_acc     = ckpt["val_accuracy"]

    print(f"  Checkpoint path        : {checkpoint_path}")
    print(f"  Checkpoint epoch       : {ckpt_epoch}")
    print(f"  Checkpoint val accuracy: {ckpt_val_acc:.2f}%")
    if "val_macro_f1" in ckpt:
        print(f"  Checkpoint val macro F1: {ckpt['val_macro_f1']:.4f}")
    print(f"  Input size             : {ckpt_input_size}x{ckpt_input_size}")
    print(f"  Norm mean              : {ckpt_norm_mean}")
    print(f"  Norm std               : {ckpt_norm_std}")
    print(f"  Classes                : {ckpt_num_classes}")

    # Cross-check checkpoint class names against model.py
    if ckpt_class_names != CLASS_NAMES:
        print(f"\n  WARNING: Checkpoint class_names differ from model.py!")
        print(f"    Checkpoint: {ckpt_class_names}")
        print(f"    model.py  : {CLASS_NAMES}")
        print(f"  Using checkpoint class_names for consistency.")

    # ── Build model and load weights ───────────────────────────────────────
    print("\n[2/5] Building model and loading weights …")
    model = ComponentCNN(num_classes=ckpt_num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")
    print(f"  Model set to eval mode.")

    # ── Load test set ──────────────────────────────────────────────────────
    print("\n[3/5] Loading test set …")
    test_loader, test_dataset = build_test_loader(
        TEST_DIR, ckpt_input_size, ckpt_norm_mean, ckpt_norm_std
    )
    verify_class_order(test_dataset.class_to_idx, ckpt_class_names)
    print(f"  Test images  : {len(test_dataset)}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Class order verified against checkpoint.")

    # ── Run inference ──────────────────────────────────────────────────────
    print("\n[4/5] Running inference …")
    all_preds, all_labels = run_inference(model, test_loader, device)

    overall_acc = compute_overall_accuracy(all_preds, all_labels)
    print(f"  Test accuracy: {overall_acc:.2f}%  "
          f"({sum(p == l for p, l in zip(all_preds, all_labels))}"
          f"/{len(all_labels)} correct)")

    # ── Metrics and reports ────────────────────────────────────────────────
    print("\n[5/5] Generating reports …")

    # Classification report
    report_str = build_classification_report(
        all_preds, all_labels, ckpt_class_names
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_np = np.array(cm)

    # Confused pairs
    confused_pairs = find_confused_pairs(cm_np, ckpt_class_names, top_n=10)

    # ── Save confusion matrix plot ─────────────────────────────────────────
    plot_confusion_matrix(cm_np, ckpt_class_names, cm_png)
    print(f"  Saved: {cm_png}")

    # ── Save classification report ─────────────────────────────────────────
    report_lines = []
    report_lines.append(SEP)
    report_lines.append("  Electronic Component Classifier – Test Set Report")
    report_lines.append(SEP)
    report_lines.append(f"  Checkpoint       : {checkpoint_path}")
    report_lines.append(f"  Checkpoint epoch : {ckpt_epoch}")
    report_lines.append(f"  Val accuracy     : {ckpt_val_acc:.2f}%")
    if "val_macro_f1" in ckpt:
        report_lines.append(f"  Val macro F1     : {ckpt['val_macro_f1']:.4f}")
    report_lines.append(f"  Test accuracy    : {overall_acc:.2f}%")
    report_lines.append(f"  Test images      : {len(all_labels)}")
    report_lines.append(f"  Device           : {device}")
    report_lines.append(SEP)
    report_lines.append("")
    report_lines.append("Per-class Classification Report")
    report_lines.append(SEP2)
    report_lines.append(report_str)
    report_lines.append("")
    report_lines.append("Most Frequently Confused Pairs")
    report_lines.append(SEP2)
    report_lines.append(f"  {'True Class':<24} {'Predicted As':<24} {'Count':>6}")
    report_lines.append("  " + "-" * 56)
    for true_cls, pred_cls, count in confused_pairs:
        report_lines.append(
            f"  {true_cls:<24} {pred_cls:<24} {count:>6}"
        )
    report_lines.append("")
    report_lines.append("Confusion Matrix (rows=true, cols=predicted)")
    report_lines.append(SEP2)
    # Column header
    col_header = "  " + f"{'':>20}" + "".join(
        f"{name:>6}" for name in
        [n[:5] for n in ckpt_class_names]   # abbreviated to fit
    )
    report_lines.append(col_header)
    for i, row in enumerate(cm_np):
        row_str = "".join(f"{v:>6}" for v in row)
        report_lines.append(f"  {ckpt_class_names[i]:>20}{row_str}")
    report_lines.append("")
    report_lines.append(SEP)
    report_lines.append("  No files were modified. No training was performed.")
    report_lines.append(SEP)

    report_txt.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  Saved: {report_txt}")

    # ── Console summary ────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  TEST SET RESULTS")
    print(SEP)
    print(f"  Checkpoint path                  : {checkpoint_path}")
    print(f"  Checkpoint epoch                 : {ckpt_epoch}")
    print(f"  Overall accuracy                 : {overall_acc:.2f}%")
    print(f"  Total test images                : {len(all_labels)}")
    print()
    print("  Per-class report:")
    print(report_str)
    print()
    print("  Most frequently confused pairs:")
    print(f"    {'True Class':<22} {'Predicted As':<22} {'Count':>5}")
    print("    " + "-" * 51)
    for true_cls, pred_cls, count in confused_pairs:
        print(f"    {true_cls:<22} {pred_cls:<22} {count:>5}")
    print()
    print(f"  Saved confusion matrix           : {cm_png}")
    print(f"  Saved classification report      : {report_txt}")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
