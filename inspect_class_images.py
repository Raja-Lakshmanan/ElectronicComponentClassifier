"""
inspect_class_images.py
-----------------------
Generate a visual contact sheet and CSV catalog of training images from
underperforming and critical problem classes in cleaned_dataset/train/.

Purpose and Context (ECE / FPGA Classifier)
-------------------------------------------
Our baseline ComponentCNN achieved 51.12% test accuracy on cleaned_dataset/test/,
with extreme variance across classes:
  - transistor:          0.00% recall (0 / 10 correct)
  - integrated_circuit:  8.33% recall (1 / 12 correct)
  - fuse:               20.00% recall (2 / 10 correct)
  - diode:              23.81% recall (5 / 21 correct)
  - inductor:           26.47% recall (9 / 34 correct)
  - push_button:        57.41% recall (31 / 54 correct)
  - potentiometer:      58.82% recall (20 / 34 correct)

Before redesigning the network, tuning hyperparameters, or adding data
augmentation, visual inspection of actual training samples is necessary to
identify root causes:
  1. Low intra-class consistency or extreme appearance variability.
  2. Visual overlap across classes (e.g. 3-pin TO-92 transistors vs diodes).
  3. Image artifacts: extreme cropping, background noise, or poor lighting.
  4. Class sample imbalance (transistor has only 42 training images vs 317 for push_button).

Requirements Satisfied
----------------------
1. Uses Pillow and Matplotlib (exits with exact error if missing; does not install).
2. Generates contact sheet of up to 12 randomly selected training images from:
   transistor, integrated_circuit, diode, fuse, inductor, push_button, potentiometer.
3. Displays class label and filename on each tile.
4. Preserves true aspect ratio (no stretching) and outputs high-resolution canvas.
5. Uses fixed random seed (default 42) for deterministic selection across runs.
6. Saves contact sheet to class_inspection.png in project root.
7. Saves selected image manifest to class_inspection.csv in project root.
8. Inspects cleaned_dataset/train/ ONLY (no test/val access).
9. Read-only operation on dataset; does not modify models, checkpoints, or data.
10. Prints output paths and exits without running training or evaluation.
"""

import argparse
import csv
import random
import sys
import textwrap
from pathlib import Path

# ── Pillow Import Check ──────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError as exc:
    print(f"ERROR: Pillow could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)

# ── Matplotlib Import Check ──────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (headless safe)
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    print(f"ERROR: matplotlib could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# Configuration and Constants
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR = Path(__file__).resolve().parent
CLEANED_DATASET_DIR = PROJECT_DIR / "cleaned_dataset"
TRAIN_DIR = CLEANED_DATASET_DIR / "train"

# Target problem classes in priority order (weakest recall first)
TARGET_CLASSES = [
    "transistor",
    "integrated_circuit",
    "diode",
    "fuse",
    "inductor",
    "push_button",
    "potentiometer",
]

DEFAULT_IMAGES_PER_CLASS = 12
DEFAULT_RANDOM_SEED = 42
DEFAULT_PNG_NAME = "class_inspection.png"
DEFAULT_CSV_NAME = "class_inspection.csv"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

SEP = "=" * 78
SEP2 = "-" * 78


# ══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    """Parse optional command-line arguments while providing exact defaults."""
    parser = argparse.ArgumentParser(
        description="Inspect training images for problem classes (contact sheet + CSV)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Fixed random seed for reproducible sampling (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help=f"Maximum images to sample per class (default: {DEFAULT_IMAGES_PER_CLASS})",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=DEFAULT_PNG_NAME,
        help=f"Filename of output contact sheet (default: {DEFAULT_PNG_NAME})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=DEFAULT_CSV_NAME,
        help=f"Filename of output CSV catalog (default: {DEFAULT_CSV_NAME})",
    )
    return parser.parse_args()


def collect_and_sample_images(
    train_dir: Path,
    target_classes: list[str],
    max_per_class: int,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """
    Deterministically sample up to max_per_class images for each target class.
    
    Files are sorted before sampling so selection is completely invariant
    to filesystem traversal ordering across operating systems.
    """
    rng = random.Random(seed)
    sampled_by_class: dict[str, list[Path]] = {}
    total_available_by_class: dict[str, int] = {}

    for cls in target_classes:
        class_path = train_dir / cls
        if not class_path.exists() or not class_path.is_dir():
            print(f"WARNING: Class directory not found: {class_path}")
            sampled_by_class[cls] = []
            total_available_by_class[cls] = 0
            continue

        # Sort all valid images deterministically
        candidates = sorted([
            p for p in class_path.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ])
        total_available_by_class[cls] = len(candidates)

        # Sample up to max_per_class
        sample_count = min(max_per_class, len(candidates))
        selected = rng.sample(candidates, k=sample_count)
        # Sort selected by filename for neat row layout
        selected.sort(key=lambda p: p.name)
        sampled_by_class[cls] = selected

    return sampled_by_class, total_available_by_class


def build_contact_sheet(
    sampled_by_class: dict[str, list[Path]],
    target_classes: list[str],
    max_per_class: int,
    output_png_path: Path,
) -> list[dict]:
    """
    Render a high-resolution contact sheet preserving image aspect ratios.
    Returns metadata list for all successfully processed images.
    """
    num_rows = len(target_classes)
    num_cols = max_per_class

    # Figure dimensions: ~2.8 inches wide per col, ~3.1 inches high per row
    fig_width = max(18, num_cols * 2.8)
    fig_height = max(14, num_rows * 3.1)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(fig_width, fig_height),
        dpi=150,
    )

    # Ensure axes is a 2D array even if nrows or ncols is 1
    axes = np.atleast_2d(axes)
    if num_rows == 1:
        axes = axes.reshape(1, num_cols)
    elif num_cols == 1:
        axes = axes.reshape(num_rows, 1)

    # Global title
    fig.suptitle(
        "ComponentCNN Training Set Visual Inspection — Problem Classes\n"
        "(cleaned_dataset/train — up to 12 samples per class)",
        fontsize=16,
        fontweight="bold",
        y=0.992,
    )

    metadata_records: list[dict] = []

    for row_idx, cls_name in enumerate(target_classes):
        images = sampled_by_class.get(cls_name, [])

        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])

            # Leftmost column: display class row header
            if col_idx == 0:
                ax.set_ylabel(
                    f"{cls_name}\n({len(images)} / {len(images)} shown)",
                    fontsize=10,
                    fontweight="bold",
                    color="#0d47a1",
                    labelpad=10,
                )

            if col_idx < len(images):
                img_path = images[col_idx]
                rel_path = img_path.relative_to(PROJECT_DIR).as_posix()

                try:
                    with Image.open(img_path) as pil_img:
                        orig_w, orig_h = pil_img.size
                        rgb_img = pil_img.convert("RGB")
                        # aspect='equal' is default in imshow, maintaining natural aspect ratio
                        ax.imshow(rgb_img)

                        # Formatted tile title: class + wrapped filename + dimensions
                        wrapped_filename = textwrap.fill(img_path.name, width=22)
                        tile_title = f"[{cls_name}]\n{wrapped_filename}\n({orig_w}×{orig_h})"
                        ax.set_title(tile_title, fontsize=6.8, pad=3)

                        metadata_records.append({
                            "class_name": cls_name,
                            "filename": img_path.name,
                            "relative_path": rel_path,
                            "full_path": str(img_path.resolve()),
                            "width": orig_w,
                            "height": orig_h,
                        })

                except Exception as err:
                    ax.text(
                        0.5, 0.5,
                        f"Error:\n{img_path.name}\n{err}",
                        ha="center", va="center",
                        fontsize=6.5, color="red",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"[{cls_name}]\n{img_path.name}", fontsize=6.8)
                    metadata_records.append({
                        "class_name": cls_name,
                        "filename": img_path.name,
                        "relative_path": rel_path,
                        "full_path": str(img_path.resolve()),
                        "width": 0,
                        "height": 0,
                    })

                # Subtle neat frame around each image tile
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.8)
            else:
                # Slot unused (if fewer than max_per_class images exist)
                ax.axis("off")

    # Layout spacing adjustments: ample room for titles and labels
    fig.subplots_adjust(
        top=0.955,
        bottom=0.025,
        left=0.085,
        right=0.990,
        hspace=0.45,
        wspace=0.15,
    )

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return metadata_records


def save_csv_manifest(records: list[dict], output_csv_path: Path) -> None:
    """Save selected image paths and class details to CSV."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class_name", "filename", "relative_path", "full_path", "width", "height"]
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()

    print(SEP)
    print("Electronic Component Classifier — Training Set Visual Inspection")
    print(SEP)
    print(f"Project directory : {PROJECT_DIR}")
    print(f"Target split      : {TRAIN_DIR} (cleaned_dataset/train ONLY)")
    print(f"Random seed       : {args.seed}")
    print(f"Max per class     : {args.per_class}")
    print(SEP2)

    # 1. Verification of train directory
    if not TRAIN_DIR.exists():
        sys.exit(
            f"ERROR: cleaned_dataset/train directory does not exist at:\n{TRAIN_DIR}\n"
            "Please ensure clean_dataset.py and verify_cleaned_dataset.py were executed."
        )

    # 2. Collect and sample images
    sampled_by_class, total_available_by_class = collect_and_sample_images(
        train_dir=TRAIN_DIR,
        target_classes=TARGET_CLASSES,
        max_per_class=args.per_class,
        seed=args.seed,
    )

    print("Sampled Training Images:")
    total_sampled = 0
    for cls in TARGET_CLASSES:
        available = total_available_by_class.get(cls, 0)
        selected_count = len(sampled_by_class.get(cls, []))
        total_sampled += selected_count
        print(f"  - {cls:<20}: {selected_count:>2} sampled / {available:>4} total available in train")

    print(f"\nTotal images selected across {len(TARGET_CLASSES)} classes: {total_sampled}")
    print(SEP2)

    # 3. Generate Contact Sheet
    output_png = PROJECT_DIR / args.output_png
    print(f"Generating contact sheet: {output_png.name} ...")
    metadata_records = build_contact_sheet(
        sampled_by_class=sampled_by_class,
        target_classes=TARGET_CLASSES,
        max_per_class=args.per_class,
        output_png_path=output_png,
    )
    print(f"  -> Contact sheet successfully written to: {output_png}")

    # 4. Generate CSV Catalog
    output_csv = PROJECT_DIR / args.output_csv
    print(f"Generating CSV manifest:  {output_csv.name} ...")
    save_csv_manifest(metadata_records, output_csv)
    print(f"  -> CSV manifest successfully written to:  {output_csv}")

    # 5. Summary and exit
    print(SEP)
    print("Visual Inspection Outputs Summary:")
    print(f"  1. Contact Sheet Image : {output_png}")
    print(f"  2. Selected Images CSV : {output_csv}")
    print(SEP)
    print("Done. Ready for visual inspection.")


if __name__ == "__main__":
    main()
