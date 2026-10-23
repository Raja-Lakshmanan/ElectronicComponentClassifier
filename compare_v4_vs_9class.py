"""
compare_v4_vs_9class.py
-----------------------
Fair side-by-side comparison between the V4 10-class model and the new 9-class model
on the exact same 346 non-transistor held-out test images from cleaned_dataset/test.

Methodology for Fair Comparison:
  1. Dataset:
     - Exactly the same 346 non-transistor test images from cleaned_dataset/test.
     - Transistor test samples are strictly excluded.
     - Both models evaluate the images in identical order.
  2. Input Processing & Normalization:
     - 64x64 RGB images.
     - Verified normalization parameters loaded directly from each model's checkpoint.
  3. Prediction Alignment:
     - 9-Class Model: Produces 9 logits corresponding to the fixed 9 classes (indices 0..8).
       Prediction is argmax over the 9 logits.
     - V4 10-Class Model: Produces 10 logits (indices 0..8 match the 9 non-transistor classes;
       index 9 is transistor). The transistor logit is masked to -infinity so the prediction
       is strictly chosen from among the 9 non-transistor classes.
  4. Metrics Computed:
     - Overall Accuracy (%)
     - Macro Precision (%), Macro Recall (%), Macro F1-score, Weighted F1-score
     - Per-class Precision, Recall, F1-score, Support, Predicted counts, and Deltas
     - Parameter count and FP32 model size comparison
  5. Artifact:
     - Saves comparison table to comparison_v4_vs_9class.csv
     - Prints a formatted terminal summary
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
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# ── Model imports ──────────────────────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES as CLASS_NAMES_10, INPUT_SIZE
from model_9class import (
    ComponentCNN9,
    CLASS_NAMES_9,
    CLASS_TO_IDX_9,
    NUM_CLASSES_9,
    count_parameters as count_params_9,
)

PROJECT_DIR           = Path(__file__).resolve().parent
CLEANED_DIR           = PROJECT_DIR / "cleaned_dataset"
TEST_DIR              = CLEANED_DIR / "test"

DEFAULT_V4_CHECKPOINT = PROJECT_DIR / "checkpoints" / "v4_best_model.pth"
DEFAULT_9C_CHECKPOINT = PROJECT_DIR / "checkpoints" / "9class_best_model.pth"
DEFAULT_OUTPUT_CSV    = PROJECT_DIR / "comparison_v4_vs_9class.csv"

BATCH_SIZE            = 32
NUM_WORKERS           = 0

SEP  = "=" * 90
SEP2 = "-" * 90


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare V4 10-class model vs 9-class model on non-transistor test images."
    )
    parser.add_argument(
        "--v4-checkpoint",
        type=str,
        default=str(DEFAULT_V4_CHECKPOINT),
        help=f"Path to V4 10-class model checkpoint (default: {DEFAULT_V4_CHECKPOINT})",
    )
    parser.add_argument(
        "--nine-checkpoint",
        type=str,
        default=str(DEFAULT_9C_CHECKPOINT),
        help=f"Path to 9-class model checkpoint (default: {DEFAULT_9C_CHECKPOINT})",
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
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Non-Transistor Dataset Loader
# ══════════════════════════════════════════════════════════════════════════
class NonTransistorTestDataset(datasets.ImageFolder):
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
                sys.exit(f"ERROR: Test directory '{directory}' is missing classes: {missing}")
        return CLASS_NAMES_9, CLASS_TO_IDX_9


def get_eval_transforms(norm_mean: Tuple[float, ...], norm_std: Tuple[float, ...]):
    """Strictly deterministic evaluation preprocessing."""
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint Verification & Loading
# ══════════════════════════════════════════════════════════════════════════
def verify_and_load_v4(checkpoint_path: Path, device: torch.device):
    """Verify and load V4 10-class model checkpoint."""
    if not checkpoint_path.exists():
        sys.exit(
            f"ERROR: V4 checkpoint not found at: {checkpoint_path}\n"
            f"Cannot proceed without verified checkpoint."
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        sys.exit(f"ERROR: Failed to load V4 checkpoint from {checkpoint_path}: {exc}")

    # Validate essential metadata keys
    required_keys = ["model_state_dict", "class_names", "num_classes"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        sys.exit(f"ERROR: V4 checkpoint missing required keys: {missing}")

    if ckpt["num_classes"] != 10:
        sys.exit(f"ERROR: V4 checkpoint num_classes expected 10, got {ckpt['num_classes']}")

    if ckpt["class_names"] != CLASS_NAMES_10:
        sys.exit(
            f"ERROR: V4 checkpoint class_names mismatch!\n"
            f"  Expected: {CLASS_NAMES_10}\n"
            f"  Received: {ckpt['class_names']}"
        )

    # Verify transistor index
    if "transistor" not in ckpt["class_names"]:
        sys.exit("ERROR: 'transistor' not found in V4 class_names!")
    transistor_idx = ckpt["class_names"].index("transistor")

    # Verify non-transistor classes align with CLASS_NAMES_9
    non_trans_v4 = [c for c in ckpt["class_names"] if c != "transistor"]
    if non_trans_v4 != CLASS_NAMES_9:
        sys.exit(
            f"ERROR: Non-transistor class ordering in V4 does not match 9-class ordering!\n"
            f"  V4 non-transistor: {non_trans_v4}\n"
            f"  9-Class target   : {CLASS_NAMES_9}"
        )

    # Verify classifier weight shape
    cls_weight = ckpt["model_state_dict"].get("classifier.weight")
    if cls_weight is None or cls_weight.shape != torch.Size([10, 64]):
        sys.exit(f"ERROR: V4 classifier.weight shape invalid. Expected (10, 64), got {getattr(cls_weight, 'shape', None)}")

    norm_mean = ckpt.get("norm_mean", (0.5, 0.5, 0.5))
    norm_std  = ckpt.get("norm_std", (0.5, 0.5, 0.5))

    model = ComponentCNN(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, ckpt, transistor_idx, norm_mean, norm_std, total_params


def verify_and_load_9class(checkpoint_path: Path, device: torch.device):
    """Verify and load 9-class model checkpoint."""
    if not checkpoint_path.exists():
        sys.exit(
            f"ERROR: 9-Class checkpoint not found at: {checkpoint_path}\n"
            f"Please ensure train_9class.py has been run and generated checkpoints/9class_best_model.pth."
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        sys.exit(f"ERROR: Failed to load 9-class checkpoint from {checkpoint_path}: {exc}")

    # Validate essential metadata keys
    required_keys = ["model_state_dict", "class_names", "num_classes"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        sys.exit(f"ERROR: 9-class checkpoint missing required keys: {missing}")

    if ckpt["num_classes"] != 9:
        sys.exit(f"ERROR: 9-class checkpoint num_classes expected 9, got {ckpt['num_classes']}")

    if ckpt["class_names"] != CLASS_NAMES_9:
        sys.exit(
            f"ERROR: 9-class checkpoint class_names mismatch!\n"
            f"  Expected: {CLASS_NAMES_9}\n"
            f"  Received: {ckpt['class_names']}"
        )

    # Verify classifier weight shape
    cls_weight = ckpt["model_state_dict"].get("classifier.weight")
    if cls_weight is None or cls_weight.shape != torch.Size([9, 64]):
        sys.exit(f"ERROR: 9-class classifier.weight shape invalid. Expected (9, 64), got {getattr(cls_weight, 'shape', None)}")

    norm_mean = ckpt.get("norm_mean", (0.5, 0.5, 0.5))
    norm_std  = ckpt.get("norm_std", (0.5, 0.5, 0.5))

    model = ComponentCNN9(num_classes=9).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, ckpt, norm_mean, norm_std, total_params


# ══════════════════════════════════════════════════════════════════════════
# Metrics Calculation
# ══════════════════════════════════════════════════════════════════════════
def compute_model_metrics(preds: List[int], labels: List[int]) -> Dict:
    """Compute accuracy, macro precision, macro recall, macro F1, and per-class stats."""
    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    actual_counts = [0] * NUM_CLASSES_9
    pred_counts   = [0] * NUM_CLASSES_9
    tp_counts     = [0] * NUM_CLASSES_9

    for p, l in zip(preds, labels):
        actual_counts[l] += 1
        pred_counts[p]   += 1
        if p == l:
            tp_counts[l] += 1

    precisions = []
    recalls    = []
    f1_scores  = []

    for c in range(NUM_CLASSES_9):
        tp = tp_counts[c]
        act = actual_counts[c]
        pr = pred_counts[c]

        rec  = (tp / act) if act > 0 else 0.0
        prec = (tp / pr) if pr > 0 else 0.0
        f1   = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0

        recalls.append(rec * 100.0)
        precisions.append(prec * 100.0)
        f1_scores.append(f1)

    if HAVE_SKLEARN:
        macro_f1    = float(f1_score(labels, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    else:
        macro_f1    = sum(f1_scores) / NUM_CLASSES_9
        weighted_f1 = sum(f1 * act for f1, act in zip(f1_scores, actual_counts)) / total

    macro_precision = sum(precisions) / NUM_CLASSES_9
    macro_recall    = sum(recalls) / NUM_CLASSES_9

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
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
# Inference Pipeline
# ══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_both_models(
    model_v4: nn.Module,
    model_9c: nn.Module,
    loader: DataLoader,
    transistor_idx: int,
    device: torch.device,
) -> Tuple[Dict, Dict, List[int]]:
    """
    Run inference for both models over the exact same DataLoader batches.
    For V4, masks the transistor logit to -inf before argmax.
    """
    v4_preds = []
    c9_preds = []
    ground_truth = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        batch_labels = labels.tolist()
        ground_truth.extend(batch_labels)

        # 1. V4 Inference (10 classes -> mask transistor out)
        v4_logits = model_v4(images)  # (B, 10)
        # Clone logits and mask transistor logit to -infinity
        masked_v4_logits = v4_logits.clone()
        masked_v4_logits[:, transistor_idx] = -float("inf")
        # Predicted index will be strictly within the 9 non-transistor classes (0..8)
        pred_v4 = masked_v4_logits.argmax(dim=1).cpu().tolist()
        v4_preds.extend(pred_v4)

        # 2. 9-Class Inference (9 classes)
        c9_logits = model_9c(images)  # (B, 9)
        pred_9c = c9_logits.argmax(dim=1).cpu().tolist()
        c9_preds.extend(pred_9c)

    metrics_v4 = compute_model_metrics(v4_preds, ground_truth)
    metrics_9c = compute_model_metrics(c9_preds, ground_truth)

    return metrics_v4, metrics_9c, ground_truth


# ══════════════════════════════════════════════════════════════════════════
# CSV Export
# ══════════════════════════════════════════════════════════════════════════
def export_comparison_csv(
    metrics_v4: Dict,
    metrics_9c: Dict,
    params_v4: int,
    params_9c: int,
    epoch_v4: int,
    epoch_9c: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "class_index",
            "class_name",
            "support",
            "v4_pred_count",
            "v4_precision_pct",
            "v4_recall_pct",
            "v4_f1_score",
            "9class_pred_count",
            "9class_precision_pct",
            "9class_recall_pct",
            "9class_f1_score",
            "f1_delta",
        ])

        # Per-class rows
        for idx, name in enumerate(CLASS_NAMES_9):
            f1_v4 = metrics_v4["f1_scores"][idx]
            f1_9c = metrics_9c["f1_scores"][idx]
            delta = f1_9c - f1_v4

            writer.writerow([
                idx,
                name,
                metrics_v4["actual_counts"][idx],
                metrics_v4["pred_counts"][idx],
                f"{metrics_v4['precisions'][idx]:.2f}",
                f"{metrics_v4['recalls'][idx]:.2f}",
                f"{f1_v4:.4f}",
                metrics_9c["pred_counts"][idx],
                f"{metrics_9c['precisions'][idx]:.2f}",
                f"{metrics_9c['recalls'][idx]:.2f}",
                f"{f1_9c:.4f}",
                f"{delta:+.4f}",
            ])

        writer.writerow([])
        # Summary Rows
        f1_delta_macro = metrics_9c["macro_f1"] - metrics_v4["macro_f1"]
        acc_delta      = metrics_9c["accuracy"] - metrics_v4["accuracy"]

        writer.writerow([
            "SUMMARY",
            "Macro Average",
            sum(metrics_v4["actual_counts"]),
            sum(metrics_v4["pred_counts"]),
            f"{metrics_v4['macro_precision']:.2f}",
            f"{metrics_v4['macro_recall']:.2f}",
            f"{metrics_v4['macro_f1']:.4f}",
            sum(metrics_9c["pred_counts"]),
            f"{metrics_9c['macro_precision']:.2f}",
            f"{metrics_9c['macro_recall']:.2f}",
            f"{metrics_9c['macro_f1']:.4f}",
            f"{f1_delta_macro:+.4f}",
        ])
        writer.writerow([
            "SUMMARY",
            "Overall Accuracy",
            sum(metrics_v4["actual_counts"]),
            "", "", "",
            f"{metrics_v4['accuracy']:.2f}%",
            "", "", "",
            f"{metrics_9c['accuracy']:.2f}%",
            f"{acc_delta:+.2f}%",
        ])
        writer.writerow([
            "METADATA",
            "Parameters",
            "", "", "", "",
            f"{params_v4:,}",
            "", "", "",
            f"{params_9c:,}",
            f"{params_9c - params_v4:+d}",
        ])
        writer.writerow([
            "METADATA",
            "Model Size (FP32 KB)",
            "", "", "", "",
            f"{(params_v4 * 4) / 1024:.2f} KB",
            "", "", "",
            f"{(params_9c * 4) / 1024:.2f} KB",
            f"{((params_9c - params_v4) * 4) / 1024:+.2f} KB",
        ])
        writer.writerow([
            "METADATA",
            "Checkpoint Epoch",
            "", "", "", "",
            epoch_v4,
            "", "", "",
            epoch_9c,
            "",
        ])


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    v4_path = Path(args.v4_checkpoint)
    c9_path = Path(args.nine_checkpoint)
    out_csv = Path(args.output_csv)

    print()
    print(SEP)
    print("  ComponentCNN Comparison: V4 (10-Class) vs 9-Class Experiment")
    print("  Evaluated on Exactly the Same 346 Non-Transistor Test Images")
    print(SEP)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device                 : {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU Hardware           : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print(SEP2)

    # 1. Verify and Load Checkpoints
    print("\n[1/4] Verifying and loading model checkpoints …")
    model_v4, ckpt_v4, trans_idx, mean_v4, std_v4, params_v4 = verify_and_load_v4(v4_path, device)
    print(f"  V4 10-Class Checkpoint : [VERIFIED]")
    print(f"    - File Path          : {v4_path}")
    print(f"    - Parameters         : {params_v4:,} ({ (params_v4*4)/1024:.2f} KB)")
    print(f"    - Trained Epoch      : {ckpt_v4.get('epoch', 'N/A')}")
    print(f"    - Transistor Index   : {trans_idx} (will be masked to -inf during inference)")
    print(f"    - Normalization      : mean={mean_v4}, std={std_v4}")

    model_9c, ckpt_9c, mean_9c, std_9c, params_9c = verify_and_load_9class(c9_path, device)
    print(f"\n  9-Class Checkpoint     : [VERIFIED]")
    print(f"    - File Path          : {c9_path}")
    print(f"    - Parameters         : {params_9c:,} ({ (params_9c*4)/1024:.2f} KB)")
    print(f"    - Trained Epoch      : {ckpt_9c.get('epoch', 'N/A')}")
    print(f"    - Output Classes     : {NUM_CLASSES_9} (transistor completely excluded)")
    print(f"    - Normalization      : mean={mean_9c}, std={std_9c}")
    print(SEP2)

    # Verify normalizations match
    if mean_v4 != mean_9c or std_v4 != std_9c:
        sys.exit(f"ERROR: Normalization mismatch between checkpoints!\nV4: {mean_v4}/{std_v4}\n9C: {mean_9c}/{std_9c}")

    # 2. Build Dataset & DataLoader
    print("\n[2/4] Building test data loader on cleaned_dataset/test/ (346 non-transistor images) …")
    test_dataset = NonTransistorTestDataset(
        root=str(TEST_DIR),
        transform=get_eval_transforms(mean_v4, std_v4),
    )
    print(f"  Non-transistor images  : {len(test_dataset)}")
    print(f"  Target classes (9)     : {', '.join(test_dataset.classes)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(SEP2)

    # 3. Run Inference on Both Models
    print("\n[3/4] Running paired inference on held-out non-transistor test split …")
    metrics_v4, metrics_9c, ground_truth = evaluate_both_models(
        model_v4=model_v4,
        model_9c=model_9c,
        loader=test_loader,
        transistor_idx=trans_idx,
        device=device,
    )
    print(SEP2)

    # 4. Save CSV and Display Results
    epoch_v4 = ckpt_v4.get("epoch", "N/A")
    epoch_9c = ckpt_9c.get("epoch", "N/A")
    export_comparison_csv(metrics_v4, metrics_9c, params_v4, params_9c, epoch_v4, epoch_9c, out_csv)

    print("\n" + SEP)
    print("  OVERALL COMPARISON RESULTS")
    print(SEP)
    acc_delta = metrics_9c["accuracy"] - metrics_v4["accuracy"]
    f1_delta  = metrics_9c["macro_f1"] - metrics_v4["macro_f1"]
    prec_delta = metrics_9c["macro_precision"] - metrics_v4["macro_precision"]
    rec_delta  = metrics_9c["macro_recall"] - metrics_v4["macro_recall"]

    print(f"  Metric                     V4 (10-Class Masked)     9-Class Model         Delta (9C - V4)")
    print("  " + "-" * 76)
    print(f"  Overall Accuracy          {metrics_v4['accuracy']:>18.2f}%    {metrics_9c['accuracy']:>16.2f}%     {acc_delta:>+16.2f}%")
    print(f"  Macro Precision           {metrics_v4['macro_precision']:>18.2f}%    {metrics_9c['macro_precision']:>16.2f}%     {prec_delta:>+16.2f}%")
    print(f"  Macro Recall              {metrics_v4['macro_recall']:>18.2f}%    {metrics_9c['macro_recall']:>16.2f}%     {rec_delta:>+16.2f}%")
    print(f"  Macro F1-Score            {metrics_v4['macro_f1']:>19.4f}     {metrics_9c['macro_f1']:>17.4f}      {f1_delta:>+16.4f}")
    print(f"  Weighted F1-Score         {metrics_v4['weighted_f1']:>19.4f}     {metrics_9c['weighted_f1']:>17.4f}      {metrics_9c['weighted_f1'] - metrics_v4['weighted_f1']:>+16.4f}")
    print(f"  Model Parameters          {params_v4:>19,d}     {params_9c:>17,d}      {params_9c - params_v4:>+16,d}")
    print(f"  Model Size (FP32)         {(params_v4*4)/1024:>16.2f} KB     {(params_9c*4)/1024:>14.2f} KB      {((params_9c - params_v4)*4)/1024:>+13.2f} KB")
    print(SEP2)

    print("\nPer-Class Breakdown (Precision / Recall / F1-Score):")
    print(f"  {'Class':<20} {'Supp':>5} | {'V4 Prec':>8} {'V4 Rec':>7} {'V4 F1':>8} | {'9C Prec':>8} {'9C Rec':>7} {'9C F1':>8} | {'F1 Delta':>9}")
    print("  " + "-" * 88)
    for idx, name in enumerate(CLASS_NAMES_9):
        f1_v4 = metrics_v4["f1_scores"][idx]
        f1_9c = metrics_9c["f1_scores"][idx]
        d_f1  = f1_9c - f1_v4
        print(
            f"  {name:<20} {metrics_v4['actual_counts'][idx]:>5} | "
            f"{metrics_v4['precisions'][idx]:>7.1f}% {metrics_v4['recalls'][idx]:>6.1f}% {f1_v4:>8.4f} | "
            f"{metrics_9c['precisions'][idx]:>7.1f}% {metrics_9c['recalls'][idx]:>6.1f}% {f1_9c:>8.4f} | "
            f"{d_f1:>+9.4f}"
        )
    print("  " + "-" * 88)
    print(
        f"  {'Macro Average':<20} {sum(metrics_v4['actual_counts']):>5} | "
        f"{metrics_v4['macro_precision']:>7.1f}% {metrics_v4['macro_recall']:>6.1f}% {metrics_v4['macro_f1']:>8.4f} | "
        f"{metrics_9c['macro_precision']:>7.1f}% {metrics_9c['macro_recall']:>6.1f}% {metrics_9c['macro_f1']:>8.4f} | "
        f"{f1_delta:>+9.4f}"
    )
    print(SEP)

    print(f"\n  [OUTPUT] Detailed comparison saved to: {out_csv.resolve()}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
