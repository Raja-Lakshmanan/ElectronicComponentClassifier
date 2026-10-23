"""
inspect_dataset.py
------------------
Read-only audit of prepared_dataset/  (train / val / test).

What this script does
---------------------
1.  Counts images per class in each split.
2.  Verifies all 10 expected classes exist in every split.
3.  Opens every image with Pillow to catch corrupted files.
4.  Reports image dimensions (W x H) and colour modes.
5.  Detects exact duplicate files via MD5 hashing.

What this script does NOT do
-----------------------------
- Does NOT move, rename, copy, or delete any file.
- Does NOT install, upgrade, or downgrade any Python package.
- Does NOT train a model.
"""

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Pillow is part of the standard PyTorch environment
try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    sys.exit(
        "ERROR: Pillow is not installed.\n"
        "It is bundled with most PyTorch environments. "
        "If missing, install it with:  pip install pillow"
    )

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
PREPARED_DIR = PROJECT_DIR / "prepared_dataset"

SPLITS = ["train", "val", "test"]
EXPECTED_CLASSES = [
    "capacitor", "diode", "fuse", "inductor", "integrated_circuit",
    "led", "potentiometer", "push_button", "resistor", "transistor",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

SEP  = "=" * 70
SEP2 = "-" * 70


# ── Helpers ────────────────────────────────────────────────────────────────

def collect_images(split_dir: Path) -> dict:
    """Return {class_name: [image_paths]} for one split directory."""
    result = {}
    if not split_dir.exists():
        return result
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        result[class_dir.name] = images
    return result


def md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex MD5 digest of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def inspect_image(path: Path):
    """
    Open *path* with Pillow.
    Returns (size, mode, error_message).
    size and mode are None when an error occurs.
    error_message is None when the file opens successfully.
    """
    try:
        with Image.open(path) as img:
            img.verify()          # catches truncated / corrupted data
        # Re-open after verify() (verify leaves the file in an unusable state)
        with Image.open(path) as img:
            img.load()            # fully decode the pixel data
            return img.size, img.mode, None
    except (UnidentifiedImageError, Exception) as exc:
        return None, None, str(exc)


# ── Section 1 & 2 – Class counts + missing-class check ───────────────────

def report_counts(split_data: dict) -> None:
    print(SEP)
    print("SECTION 1 & 2 - Image counts per class and split")
    print(SEP)

    # Header
    header = f"{'Class':<22}" + "".join(f"{s:>10}" for s in SPLITS) + f"{'TOTAL':>10}"
    print(header)
    print(SEP2)

    class_totals = defaultdict(int)
    split_totals = defaultdict(int)
    missing = defaultdict(list)   # class -> missing splits

    all_classes = sorted(
        {cls for split_imgs in split_data.values() for cls in split_imgs}
        | set(EXPECTED_CLASSES)
    )

    for cls in all_classes:
        counts = []
        row_total = 0
        for split in SPLITS:
            n = len(split_data.get(split, {}).get(cls, []))
            counts.append(n)
            row_total += n
            split_totals[split] += n
            class_totals[cls] += n
            if n == 0:
                missing[cls].append(split)
        mark = "  <- MISSING IN SOME SPLIT" if cls in missing else ""
        print(f"{cls:<22}" + "".join(f"{c:>10}" for c in counts) + f"{row_total:>10}{mark}")

    print(SEP2)
    print(
        f"{'TOTAL':<22}"
        + "".join(f"{split_totals[s]:>10}" for s in SPLITS)
        + f"{sum(split_totals.values()):>10}"
    )
    print()

    # Missing-class report
    missing_expected = [c for c in EXPECTED_CLASSES if c in missing]
    extra_classes    = [c for c in all_classes if c not in EXPECTED_CLASSES]

    if not missing_expected and not extra_classes:
        print("OK  All 10 expected classes are present in every split.")
    else:
        if missing_expected:
            print("FAIL  Classes missing from one or more splits:")
            for cls in missing_expected:
                print(f"     {cls}: absent in {missing[cls]}")
        if extra_classes:
            print("WARN  Unexpected classes found (not in the 10-class list):")
            for cls in extra_classes:
                print(f"     {cls}")
    print()


# ── Section 3 & 4 – Image integrity, dimensions, modes ───────────────────

def report_integrity(split_data: dict) -> None:
    print(SEP)
    print("SECTION 3 & 4 - Image integrity, dimensions, and colour modes")
    print(SEP)

    all_sizes = []
    all_modes = []
    corrupted = []   # (split, class, filename, error)
    non_rgb   = []   # (split, class, filename, mode)

    total_checked = 0
    for split in SPLITS:
        for cls, paths in split_data.get(split, {}).items():
            for p in paths:
                total_checked += 1
                size, mode, err = inspect_image(p)
                if err:
                    corrupted.append((split, cls, p.name, err))
                else:
                    all_sizes.append(size)
                    all_modes.append(mode)
                    if mode != "RGB":
                        non_rgb.append((split, cls, p.name, mode))

    print(f"Total images checked  : {total_checked}")
    print(f"Corrupted / unreadable: {len(corrupted)}")
    print()

    # Dimension summary
    if all_sizes:
        widths  = [s[0] for s in all_sizes]
        heights = [s[1] for s in all_sizes]
        unique_sizes = Counter(all_sizes)
        print("Dimension statistics (W x H):")
        print(f"  Min  : {min(widths)} x {min(heights)}")
        print(f"  Max  : {max(widths)} x {max(heights)}")
        print(f"  Unique dimensions : {len(unique_sizes)}")
        print(f"  Top-5 most common:")
        for size, count in unique_sizes.most_common(5):
            print(f"    {size[0]:>5} x {size[1]:<5}  -> {count} images")
    print()

    # Mode summary
    if all_modes:
        mode_counter = Counter(all_modes)
        print("Colour mode distribution:")
        for mode, count in mode_counter.most_common():
            print(f"  {mode:<10}: {count}")
    print()

    # Non-RGB detail
    if non_rgb:
        print(f"WARN  Non-RGB images ({len(non_rgb)} files) - consider converting to RGB before training:")
        for split, cls, fname, mode in non_rgb[:20]:
            print(f"    [{split}/{cls}] {fname}  ->  {mode}")
        if len(non_rgb) > 20:
            print(f"    ... and {len(non_rgb) - 20} more.")
    else:
        print("OK  All readable images are in RGB mode.")
    print()

    # Corrupted detail
    if corrupted:
        print(f"FAIL  Corrupted / unreadable files ({len(corrupted)}):")
        for split, cls, fname, err in corrupted:
            print(f"    [{split}/{cls}] {fname}")
            print(f"       Error: {err}")
    else:
        print("OK  No corrupted or unreadable images found.")
    print()


# ── Section 5 – Exact duplicate detection via MD5 ─────────────────────────

def report_duplicates(split_data: dict) -> None:
    print(SEP)
    print("SECTION 5 - Exact duplicate detection (MD5 hash)")
    print(SEP)

    hash_map = defaultdict(list)
    total = 0

    for split in SPLITS:
        for cls, paths in split_data.get(split, {}).items():
            for p in paths:
                total += 1
                digest = md5_of_file(p)
                hash_map[digest].append(p)

    duplicates = {h: ps for h, ps in hash_map.items() if len(ps) > 1}

    print(f"Total files hashed  : {total}")
    print(f"Unique MD5 digests  : {len(hash_map)}")
    print(f"Duplicate groups    : {len(duplicates)}")
    print()

    if duplicates:
        print("FAIL  Duplicate groups (files with identical content):")
        for i, (digest, paths) in enumerate(duplicates.items(), 1):
            print(f"  Group {i}  [MD5: {digest}]")
            for p in paths:
                try:
                    rel = p.relative_to(PREPARED_DIR)
                except ValueError:
                    rel = p
                print(f"    {rel}")
        print()
        print("NOTE: No files were modified, moved, or deleted.")
    else:
        print("OK  No exact duplicate image files found.")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print(SEP)
    print("  Electronic Component Classifier - Dataset Inspection")
    print(f"  prepared_dataset : {PREPARED_DIR}")
    print(SEP)

    if not PREPARED_DIR.exists():
        sys.exit(
            f"ERROR: {PREPARED_DIR} does not exist.\n"
            "Run split_dataset.py first to create the splits."
        )

    # Collect all images
    split_data = {}
    for split in SPLITS:
        split_data[split] = collect_images(PREPARED_DIR / split)

    report_counts(split_data)
    report_integrity(split_data)
    report_duplicates(split_data)

    print(SEP)
    print("  Inspection complete - no files were modified.")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
