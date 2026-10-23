"""
verify_cleaned_dataset.py
-------------------------
Independent, read-only verification of cleaned_dataset/.

Checks performed
----------------
1.  Per-class image counts for train / val / test  (+ totals).
2.  All 10 expected class directories exist in each split.
3.  SHA-256 hashes – any identical hash appearing in more than one split
    (cross-split leakage) is reported as a FAIL.
4.  Any identical hash appearing under different class labels, even within
    the same split (cross-class contamination), is reported as a FAIL.
5.  Cross-class groups from cross_class_duplicates.csv are checked
    individually – every copy must be absent from cleaned_dataset/.
6.  Every retained image file can be opened and loaded with Pillow.

Output
------
  console        – concise pass/fail summary
  verify_cleaned_dataset_report.txt – full report

This script does NOT modify, move, or delete any file.
No packages are installed or changed.
No model is trained.
"""

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    sys.exit(
        "ERROR: Pillow is not installed.\n"
        "Install it with:  pip install pillow"
    )

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent
CLEANED_DIR   = PROJECT_DIR / "cleaned_dataset"
CC_CSV        = PROJECT_DIR / "cross_class_duplicates.csv"
REPORT_PATH   = PROJECT_DIR / "verify_cleaned_dataset_report.txt"

SPLITS = ["train", "val", "test"]
EXPECTED_CLASSES = [
    "capacitor", "diode", "fuse", "inductor", "integrated_circuit",
    "led", "potentiometer", "push_button", "resistor", "transistor",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}
CHUNK = 1 << 20  # 1 MiB

SEP  = "=" * 72
SEP2 = "-" * 72


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def collect_images(split_dir: Path) -> dict:
    """Return {class_name: [sorted Path list]} for one split directory."""
    result = {}
    if not split_dir.exists():
        return result
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        result[cls_dir.name] = images
    return result


def try_open_image(path: Path):
    """Return error string if the image cannot be opened/loaded, else None."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return None
    except (UnidentifiedImageError, Exception) as exc:
        return str(exc)


def load_cross_class_csv(csv_path: Path) -> dict:
    """
    Parse cross_class_duplicates.csv.
    Returns {sha256: {"groups": group_id, "classes": set, "files": [rel_path_str]}}
    keyed by hash.
    """
    cc_groups: dict[str, dict] = {}
    if not csv_path.exists():
        return cc_groups
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            digest = row["sha256"].strip()
            if digest not in cc_groups:
                cc_groups[digest] = {
                    "group_id": row["group_id"].strip(),
                    "classes": set(),
                    "files": [],
                }
            cc_groups[digest]["classes"].add(row["class"].strip())
            cc_groups[digest]["files"].append(row["full_path"].strip())
    return cc_groups


# ══════════════════════════════════════════════════════════════════════════
# Check 1 & 2 – Counts and class presence
# ══════════════════════════════════════════════════════════════════════════

def check_counts_and_classes(split_data: dict, lines: list) -> dict:
    """
    Print and record per-class/split counts.
    Returns {"missing_classes": {cls: [splits]}, "counts": {split: {cls: int}}}
    """
    lines.append(SEP)
    lines.append("CHECK 1 & 2 – Per-class image counts and class presence")
    lines.append(SEP)

    header = f"  {'Class':<24}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'TOTAL':>8}"
    lines.append(header)
    lines.append("  " + "-" * 56)

    missing_classes: dict[str, list] = {}
    counts: dict[str, dict] = {s: {} for s in SPLITS}
    split_totals = {s: 0 for s in SPLITS}

    all_classes = sorted(
        {cls for sd in split_data.values() for cls in sd} | set(EXPECTED_CLASSES)
    )

    for cls in all_classes:
        row_vals = []
        row_total = 0
        absent_in = []
        for split in SPLITS:
            n = len(split_data.get(split, {}).get(cls, []))
            counts[split][cls] = n
            split_totals[split] += n
            row_total += n
            row_vals.append(n)
            if n == 0:
                absent_in.append(split)
        if absent_in:
            missing_classes[cls] = absent_in
        mark = "  <- MISSING" if absent_in else ""
        lines.append(
            f"  {cls:<24}" + "".join(f"{v:>8}" for v in row_vals)
            + f"{row_total:>8}{mark}"
        )

    lines.append("  " + "-" * 56)
    grand = sum(split_totals.values())
    lines.append(
        f"  {'TOTAL':<24}"
        + "".join(f"{split_totals[s]:>8}" for s in SPLITS)
        + f"{grand:>8}"
    )
    lines.append("")

    if missing_classes:
        lines.append("FAIL  Classes absent from one or more splits:")
        for cls, absent in missing_classes.items():
            lines.append(f"      {cls}: missing from {absent}")
    else:
        lines.append("PASS  All 10 expected classes present in every split.")
    lines.append("")

    # Extra classes
    extra = [c for c in all_classes if c not in EXPECTED_CLASSES]
    if extra:
        lines.append(f"WARN  Unexpected class directories: {extra}")
    else:
        lines.append("PASS  No unexpected class directories found.")
    lines.append("")

    return {"missing_classes": missing_classes, "counts": counts, "grand_total": grand}


# ══════════════════════════════════════════════════════════════════════════
# Check 3 & 4 – SHA-256 cross-split and cross-class leakage
# ══════════════════════════════════════════════════════════════════════════

def check_hashes(split_data: dict, lines: list) -> dict:
    """
    Hash every file in cleaned_dataset/.
    Detect:
      - Same hash in more than one split  (cross-split leakage)
      - Same hash under different labels, any split (cross-class contamination)
    Returns {"cross_split": [...], "cross_class": [...], "hash_map": {digest: [rec]}}
    """
    lines.append(SEP)
    lines.append("CHECK 3 & 4 – SHA-256 duplicate detection (cross-split & cross-class)")
    lines.append(SEP)

    hash_map: dict[str, list] = defaultdict(list)
    total = 0

    for split in SPLITS:
        for cls, paths in split_data.get(split, {}).items():
            for p in paths:
                total += 1
                digest = sha256_of_file(p)
                hash_map[digest].append({"path": p, "split": split, "cls": cls})

    lines.append(f"  Total files hashed    : {total}")
    lines.append(f"  Unique SHA-256 hashes : {len(hash_map)}")
    lines.append("")

    cross_split_groups = []
    cross_class_groups = []

    for digest, members in hash_map.items():
        if len(members) < 2:
            continue
        splits_seen  = {m["split"] for m in members}
        classes_seen = {m["cls"]   for m in members}
        if len(splits_seen) > 1:
            cross_split_groups.append((digest, members))
        if len(classes_seen) > 1:
            cross_class_groups.append((digest, members))

    # Cross-split
    if cross_split_groups:
        lines.append(f"FAIL  Cross-split duplicates found: {len(cross_split_groups)} group(s)")
        for digest, members in cross_split_groups:
            lines.append(f"  Hash: {digest}")
            for m in members:
                try:
                    rel = m["path"].relative_to(CLEANED_DIR)
                except ValueError:
                    rel = m["path"]
                lines.append(f"    [{m['split']}/{m['cls']}] {rel.name}")
        lines.append("")
    else:
        lines.append("PASS  No cross-split duplicate hashes found.")
        lines.append("")

    # Cross-class
    if cross_class_groups:
        lines.append(f"FAIL  Cross-class duplicate hashes found: {len(cross_class_groups)} group(s)")
        for digest, members in cross_class_groups:
            labels = sorted({m["cls"] for m in members})
            lines.append(f"  Hash: {digest}  Labels: {labels}")
            for m in members:
                try:
                    rel = m["path"].relative_to(CLEANED_DIR)
                except ValueError:
                    rel = m["path"]
                lines.append(f"    [{m['split']}/{m['cls']}] {rel.name}")
        lines.append("")
    else:
        lines.append("PASS  No cross-class duplicate hashes found.")
        lines.append("")

    return {
        "cross_split": cross_split_groups,
        "cross_class": cross_class_groups,
        "hash_map":    dict(hash_map),
        "total":       total,
    }


# ══════════════════════════════════════════════════════════════════════════
# Check 5 – Cross-class CSV exclusion verification
# ══════════════════════════════════════════════════════════════════════════

def check_cross_class_exclusions(cc_groups: dict,
                                 hash_map: dict,
                                 lines: list) -> dict:
    """
    For each cross-class group from cross_class_duplicates.csv, verify that
    NONE of its SHA-256 hashes appear in the cleaned dataset.
    """
    lines.append(SEP)
    lines.append("CHECK 5 – Cross-class duplicate exclusion verification")
    lines.append(SEP)

    if not cc_groups:
        lines.append("WARN  cross_class_duplicates.csv not found or is empty.")
        lines.append("      Cannot perform exclusion check.")
        lines.append("")
        return {"status": "skipped", "leaked": []}

    lines.append(f"  Cross-class groups in CSV : {len(cc_groups)}")
    lines.append("")

    leaked = []
    for digest, info in cc_groups.items():
        gid     = info["group_id"]
        classes = sorted(info["classes"])
        present_in_cleaned = hash_map.get(digest, [])

        if present_in_cleaned:
            leaked.append((gid, digest, classes, present_in_cleaned))
            lines.append(f"FAIL  Group {gid} (labels: {classes})")
            lines.append(f"      Hash {digest}")
            lines.append(f"      Found {len(present_in_cleaned)} copy(ies) in cleaned_dataset/:")
            for m in present_in_cleaned:
                try:
                    rel = m["path"].relative_to(CLEANED_DIR)
                except ValueError:
                    rel = m["path"]
                lines.append(f"        [{m['split']}/{m['cls']}] {rel.name}")
        else:
            lines.append(
                f"PASS  Group {gid} (labels: {classes})"
                f" – correctly excluded. Hash: {digest[:20]}..."
            )
    lines.append("")

    if not leaked:
        lines.append(
            "PASS  All 4 cross-class duplicate groups are absent from "
            "cleaned_dataset/."
        )
    else:
        lines.append(
            f"FAIL  {len(leaked)} cross-class group(s) still present in "
            f"cleaned_dataset/."
        )
    lines.append("")

    return {"status": "done", "leaked": leaked}


# ══════════════════════════════════════════════════════════════════════════
# Check 6 – Pillow integrity of retained images
# ══════════════════════════════════════════════════════════════════════════

def check_image_integrity(split_data: dict, lines: list) -> dict:
    """Open and load every image in cleaned_dataset/ with Pillow."""
    lines.append(SEP)
    lines.append("CHECK 6 – Image file integrity (Pillow open + load)")
    lines.append(SEP)

    total   = 0
    bad     = []

    for split in SPLITS:
        for cls, paths in split_data.get(split, {}).items():
            for p in paths:
                total += 1
                err = try_open_image(p)
                if err:
                    bad.append((split, cls, p.name, err))

    lines.append(f"  Total images checked  : {total}")
    lines.append(f"  Unreadable / corrupted: {len(bad)}")
    lines.append("")

    if bad:
        lines.append(f"FAIL  Unreadable images ({len(bad)}):")
        for split, cls, fname, err in bad:
            lines.append(f"    [{split}/{cls}] {fname}")
            lines.append(f"       {err}")
    else:
        lines.append("PASS  All retained images opened and loaded successfully.")
    lines.append("")

    return {"total": total, "bad": bad}


# ══════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════

def print_console_summary(count_result, hash_result,
                          cc_result, integrity_result) -> None:
    print()
    print(SEP)
    print("  VERIFICATION SUMMARY  –  cleaned_dataset/")
    print(SEP)

    def status(cond): return "PASS" if cond else "FAIL"

    c1 = not count_result["missing_classes"]
    c2 = not hash_result["cross_split"]
    c3 = not hash_result["cross_class"]
    c4 = (cc_result["status"] != "skipped") and not cc_result["leaked"]
    c5 = not integrity_result["bad"]

    print(f"  [{status(c1)}] CHECK 1/2 – All 10 classes in every split")
    print(f"  [{status(c2)}] CHECK 3   – No cross-split hash leakage")
    print(f"  [{status(c3)}] CHECK 4   – No cross-class hash contamination")
    print(f"  [{status(c4)}] CHECK 5   – Cross-class groups correctly excluded")
    print(f"  [{status(c5)}] CHECK 6   – All images open with Pillow")
    print()
    print(f"  Total images in cleaned_dataset/ : {count_result['grand_total']}")
    print(f"  Full report : verify_cleaned_dataset_report.txt")
    print(SEP)
    print()

    all_pass = c1 and c2 and c3 and c4 and c5
    if all_pass:
        print("  Result: ALL CHECKS PASSED – cleaned_dataset/ is ready.")
    else:
        fails = [
            name for name, cond in [
                ("CHECK 1/2 (missing classes)", c1),
                ("CHECK 3 (cross-split leakage)", c2),
                ("CHECK 4 (cross-class contamination)", c3),
                ("CHECK 5 (cross-class exclusion)", c4),
                ("CHECK 6 (image integrity)", c5),
            ] if not cond
        ]
        print(f"  Result: {len(fails)} check(s) FAILED – see report for details.")
        for f in fails:
            print(f"    - {f}")
    print()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print(SEP)
    print("  Electronic Component Classifier – Cleaned Dataset Verification")
    print(f"  cleaned_dataset : {CLEANED_DIR}")
    print(SEP)

    if not CLEANED_DIR.exists():
        sys.exit(
            f"ERROR: {CLEANED_DIR} does not exist.\n"
            "Run clean_dataset.py first."
        )

    # Collect images
    split_data = {}
    for split in SPLITS:
        split_data[split] = collect_images(CLEANED_DIR / split)

    # Load cross-class CSV
    cc_groups = load_cross_class_csv(CC_CSV)
    if not cc_groups:
        print(f"  WARNING: {CC_CSV} not found or empty.")
        print("           Cross-class exclusion check (CHECK 5) will be skipped.")
    else:
        print(f"  Loaded {len(cc_groups)} cross-class group(s) from {CC_CSV.name}")
    print()

    report_lines: list[str] = []
    report_lines.append(SEP)
    report_lines.append("  Electronic Component Classifier – Cleaned Dataset Verification Report")
    report_lines.append(f"  cleaned_dataset : {CLEANED_DIR}")
    report_lines.append(f"  cross_class CSV : {CC_CSV}")
    report_lines.append(SEP)
    report_lines.append("")

    print("[1/4] Checking class counts …")
    count_result = check_counts_and_classes(split_data, report_lines)

    print("[2/4] Hashing all images and checking for duplicates …")
    hash_result = check_hashes(split_data, report_lines)

    print("[3/4] Verifying cross-class exclusions …")
    cc_result = check_cross_class_exclusions(cc_groups, hash_result["hash_map"], report_lines)

    print("[4/4] Checking image integrity with Pillow …")
    integrity_result = check_image_integrity(split_data, report_lines)

    # Write report
    report_lines.append(SEP)
    report_lines.append("  End of report – no files were modified.")
    report_lines.append(SEP)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  Report written to: {REPORT_PATH}")

    # Console summary
    print_console_summary(count_result, hash_result, cc_result, integrity_result)


if __name__ == "__main__":
    main()
