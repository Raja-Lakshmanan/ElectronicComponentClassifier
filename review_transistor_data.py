"""
review_transistor_data.py
-------------------------
Visual review and integrity diagnostic for all available transistor images
in cleaned_dataset/train/transistor and cleaned_dataset/val/transistor.

Background & Motivation
-----------------------
In our component classifier experiments:
  - Baseline model (train.py): 0.00% transistor recall (0 / 10 test).
  - Improved v2 (train_v2.py): Heavy class weights (3.82x) caused severe overprediction.
  - Improved v3 (train_v3.py): Softer weights (1.68x) with low LR caused underfitting.
  - Improved v4 (train_v4.py): 59.83% test accuracy, 0.5371 macro F1, but still 0.00% transistor recall!

The training set contains only 42 transistor images and validation contains only 9.
Before deciding whether to collect/augment more data or source new transistor images,
this script provides a complete visual contact sheet and hash manifest of every
available training and validation transistor image.

Requirements Satisfied
----------------------
1. Inspects cleaned_dataset/train/transistor and cleaned_dataset/val/transistor ONLY.
2. Generates a high-resolution contact sheet of all 51 transistor images, displaying
   split, filename, and original dimensions on each tile.
3. Saves contact sheet as transistor_training_review.png in the project root.
4. Saves full metadata manifest to transistor_review.csv.
5. Strictly avoids inspecting test images; does not modify or delete any dataset images.
6. Computes SHA-256 hashes and reports duplicate counts and unique file counts.
7. Uses installed libraries only (Pillow, Matplotlib, standard library).
8. Read-only operation: does not train, evaluate, or modify models/checkpoints.
"""

import argparse
import csv
import hashlib
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

# ── Safe package imports ──────────────────────────────────────────────────
try:
    from PIL import Image
except ImportError as exc:
    print(f"ERROR: Pillow could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")  # Headless non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    print(f"ERROR: matplotlib or numpy could not be imported.\nDetail: {exc}")
    print("Do NOT install or upgrade packages automatically.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# Paths and Defaults
# ══════════════════════════════════════════════════════════════════════════
PROJECT_DIR        = Path(__file__).resolve().parent
CLEANED_DIR        = PROJECT_DIR / "cleaned_dataset"
TRAIN_TRANSISTOR   = CLEANED_DIR / "train" / "transistor"
VAL_TRANSISTOR     = CLEANED_DIR / "val" / "transistor"

DEFAULT_OUTPUT_PNG = PROJECT_DIR / "transistor_training_review.png"
DEFAULT_OUTPUT_CSV = PROJECT_DIR / "transistor_review.csv"

VALID_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

SEP  = "=" * 88
SEP2 = "-" * 88


# ══════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ══════════════════════════════════════════════════════════════════════════
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and review all train and val transistor images (contact sheet + CSV)."
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=str(DEFAULT_OUTPUT_PNG),
        help=f"Filename of output contact sheet (default: {DEFAULT_OUTPUT_PNG.name})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Filename of output CSV catalog (default: {DEFAULT_OUTPUT_CSV.name})",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Hash Calculation & Data Gathering
# ══════════════════════════════════════════════════════════════════════════
def compute_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_transistor_images() -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Gather metadata for all transistor images in train and val splits.
    Returns:
        all_records: list of image metadata dicts
        split_map: {"train": [...], "val": [...]}
    """
    records = []
    split_map = {"train": [], "val": []}

    targets = [
        ("train", TRAIN_TRANSISTOR),
        ("val", VAL_TRANSISTOR),
    ]

    for split_name, split_dir in targets:
        if not split_dir.exists():
            print(f"WARNING: Directory not found: {split_dir}")
            continue

        files = sorted([
            f for f in split_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ])

        for filepath in files:
            sha256 = compute_sha256(filepath)
            file_size = filepath.stat().st_size
            rel_path = filepath.relative_to(PROJECT_DIR).as_posix()

            width, height = 0, 0
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
            except Exception as err:
                print(f"WARNING: Could not open {filepath.name}: {err}")

            aspect_ratio = (width / height) if height > 0 else 0.0

            item = {
                "filename": filepath.name,
                "split": split_name,
                "relative_path": rel_path,
                "full_path": str(filepath.resolve()),
                "width": width,
                "height": height,
                "aspect_ratio": round(aspect_ratio, 3),
                "file_size_bytes": file_size,
                "sha256_hash": sha256,
                "path_obj": filepath,
            }
            records.append(item)
            split_map[split_name].append(item)

    return records, split_map


# ══════════════════════════════════════════════════════════════════════════
# Contact Sheet Rendering
# ══════════════════════════════════════════════════════════════════════════
def render_contact_sheet(
    train_records: list[dict],
    val_records: list[dict],
    output_png: Path,
) -> None:
    """
    Render a high-resolution contact sheet of all train and val transistor images.
    Organized into dedicated sections:
      - Rows 1..7: Training images (42 images, 6 per row)
      - Rows 8..9: Validation images (9 images, 6 per row)
    """
    num_cols = 6
    train_rows = (len(train_records) + num_cols - 1) // num_cols  # 42 / 6 = 7 rows
    val_rows = (len(val_records) + num_cols - 1) // num_cols      # 9 / 6 = 2 rows
    total_rows = train_rows + val_rows                            # 9 rows total

    fig_width = max(18, num_cols * 3.8)
    fig_height = max(24, total_rows * 3.4)

    fig, axes = plt.subplots(
        nrows=total_rows,
        ncols=num_cols,
        figsize=(fig_width, fig_height),
        dpi=150,
    )

    # Global title
    fig.suptitle(
        f"Transistor Dataset Visual Review — cleaned_dataset\n"
        f"({len(train_records)} Train Images + {len(val_records)} Validation Images = {len(train_records) + len(val_records)} Total)",
        fontsize=16,
        fontweight="bold",
        y=0.992,
    )

    # 1. Render Training Images
    for idx in range(train_rows * num_cols):
        r = idx // num_cols
        c = idx % num_cols
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])

        # Section label on leftmost tile
        if c == 0 and r == 0:
            ax.set_ylabel(
                f"TRAIN SPLIT\n({len(train_records)} images)",
                fontsize=11,
                fontweight="bold",
                color="#0d47a1",
                labelpad=12,
            )

        if idx < len(train_records):
            item = train_records[idx]
            try:
                with Image.open(item["path_obj"]) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    ax.imshow(rgb_img)

                    wrapped = textwrap.fill(item["filename"], width=24)
                    ax.set_title(
                        f"[TRAIN #{idx+1}]\n{wrapped}\n({item['width']}×{item['height']})",
                        fontsize=7.2,
                        pad=4,
                        color="#0d47a1",
                    )
            except Exception as err:
                ax.text(0.5, 0.5, f"Error:\n{err}", ha="center", va="center", color="red", fontsize=7)
                ax.set_title(f"[TRAIN #{idx+1}] Error", fontsize=7.2)

            for spine in ax.spines.values():
                spine.set_color("#1565c0")
                spine.set_linewidth(1.2)
        else:
            ax.axis("off")

    # 2. Render Validation Images
    for idx in range(val_rows * num_cols):
        r = train_rows + (idx // num_cols)
        c = idx % num_cols
        ax = axes[r, c]
        ax.set_xticks([])
        ax.set_yticks([])

        # Section label on leftmost tile
        if c == 0 and idx == 0:
            ax.set_ylabel(
                f"VAL SPLIT\n({len(val_records)} images)",
                fontsize=11,
                fontweight="bold",
                color="#e65100",
                labelpad=12,
            )

        if idx < len(val_records):
            item = val_records[idx]
            try:
                with Image.open(item["path_obj"]) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    ax.imshow(rgb_img)

                    wrapped = textwrap.fill(item["filename"], width=24)
                    ax.set_title(
                        f"[VAL #{idx+1}]\n{wrapped}\n({item['width']}×{item['height']})",
                        fontsize=7.2,
                        pad=4,
                        color="#e65100",
                    )
            except Exception as err:
                ax.text(0.5, 0.5, f"Error:\n{err}", ha="center", va="center", color="red", fontsize=7)
                ax.set_title(f"[VAL #{idx+1}] Error", fontsize=7.2)

            for spine in ax.spines.values():
                spine.set_color("#e65100")
                spine.set_linewidth(1.2)
        else:
            ax.axis("off")

    fig.subplots_adjust(
        top=0.960,
        bottom=0.025,
        left=0.075,
        right=0.990,
        hspace=0.45,
        wspace=0.15,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# CSV Catalog Export
# ══════════════════════════════════════════════════════════════════════════
def save_transistor_csv(records: list[dict], output_csv: Path) -> None:
    """Save metadata records to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "split",
        "relative_path",
        "full_path",
        "width",
        "height",
        "aspect_ratio",
        "file_size_bytes",
        "sha256_hash",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()

    output_png = Path(args.output_png)
    output_csv = Path(args.output_csv)

    if not output_png.is_absolute():
        output_png = PROJECT_DIR / output_png
    if not output_csv.is_absolute():
        output_csv = PROJECT_DIR / output_csv

    print()
    print(SEP)
    print("  Electronic Component Classifier — Transistor Dataset Visual Review")
    print(SEP)
    print(f"  Target Splits      : cleaned_dataset/train/transistor & cleaned_dataset/val/transistor")
    print(f"  Test Set           : STRICTLY EXCLUDED (Untouched)")
    print(f"  Output PNG         : {output_png}")
    print(f"  Output CSV         : {output_csv}")
    print(SEP2)

    # 1. Gather Images and Compute Hashes
    print("\n[1/4] Inspecting transistor image files & computing SHA-256 hashes …")
    all_records, split_map = collect_transistor_images()

    train_recs = split_map["train"]
    val_recs   = split_map["val"]
    total_imgs = len(all_records)

    print(f"  Training transistor images  : {len(train_recs)}")
    print(f"  Validation transistor images: {len(val_recs)}")
    print(f"  Total transistor images     : {total_imgs}")

    if total_imgs == 0:
        sys.exit("ERROR: No transistor images found in train or val directories!")

    # 2. Hash & Duplicate Analysis
    print("\n[2/4] Analyzing uniqueness and duplicate hashes …")
    hash_groups = defaultdict(list)
    for r in all_records:
        hash_groups[r["sha256_hash"]].append(r)

    unique_hashes = len(hash_groups)
    duplicate_groups = {h: items for h, items in hash_groups.items() if len(items) > 1}
    duplicate_count = sum(len(items) - 1 for items in duplicate_groups.values())

    print(f"  Total images processed      : {total_imgs}")
    print(f"  Unique image hashes         : {unique_hashes}")
    print(f"  Duplicate hashes detected   : {len(duplicate_groups)} duplicate group(s) ({duplicate_count} redundant images)")

    if duplicate_groups:
        print("\n  WARNING: Duplicate images found within transistor data:")
        for h, items in duplicate_groups.items():
            file_list = ", ".join(f"{it['split']}/{it['filename']}" for it in items)
            print(f"    - Hash {h[:12]}… shared by: {file_list}")
    else:
        print("  VERDICT: [PASS] All 51 transistor images have unique SHA-256 hashes. No exact duplicates exist.")

    # Dimension statistics
    widths = [r["width"] for r in all_records]
    heights = [r["height"] for r in all_records]
    print(f"\n  Image Resolution Statistics:")
    print(f"    - Width range  : {min(widths)} px to {max(widths)} px (Average: {sum(widths)/len(widths):.1f} px)")
    print(f"    - Height range : {min(heights)} px to {max(heights)} px (Average: {sum(heights)/len(heights):.1f} px)")

    # 3. Render Contact Sheet
    print(f"\n[3/4] Rendering visual contact sheet ({output_png.name}) …")
    render_contact_sheet(train_recs, val_recs, output_png)
    print(f"  -> Contact sheet successfully written to:\n     {output_png}")

    # 4. Save CSV Manifest
    print(f"\n[4/4] Writing metadata catalog to CSV ({output_csv.name}) …")
    save_transistor_csv(all_records, output_csv)
    print(f"  -> Metadata CSV successfully written to:\n     {output_csv}")

    # Summary
    print()
    print(SEP)
    print("  TRANSISTOR DATASET REVIEW COMPLETE")
    print(SEP)
    print(f"  1. Visual Contact Sheet : {output_png}")
    print(f"  2. Image Metadata CSV   : {output_csv}")
    print(f"  3. Total Images Reviewed: {total_imgs} (Train: {len(train_recs)}, Val: {len(val_recs)})")
    print(f"  4. Unique Hashes        : {unique_hashes} / {total_imgs}")
    print(SEP)
    print("  Review transistor_training_review.png to examine package styles, leads, and visual diversity.\n")


if __name__ == "__main__":
    main()
