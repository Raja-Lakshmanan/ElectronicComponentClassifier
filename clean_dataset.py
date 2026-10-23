"""
clean_dataset.py
----------------
Audits prepared_dataset/ for exact duplicates (SHA-256), produces CSV
reports, then builds cleaned_dataset/ according to strict, documented
rules.

Pipeline
--------
Phase 1 – Hash every image (read-only, no files changed).
Phase 2 – Classify each duplicate group:
    SAME_CLASS   : all copies share the same class label
                   → keep exactly one copy per split; drop the rest.
    CROSS_CLASS  : copies appear under different class labels
                   → exclude ALL copies; flag for manual review.
    CROSS_SPLIT  : copies of the same image appear in more than one split
                   → exclude ALL copies to prevent data leakage,
                     regardless of class agreement.
    AMBIGUOUS    : any group that does not fall cleanly into the above
                   categories is excluded and logged separately.
Phase 3 – Write output files:
    duplicate_report.csv        – every duplicate group (all categories)
    cross_class_duplicates.csv  – cross-class groups only (manual review)
    cleaning_summary.txt        – before/after counts per class per split
    cleaned_dataset/            – new directory tree, original untouched

Rules
-----
* The original prepared_dataset/ is NEVER modified.
* No files are deleted anywhere.
* No guesses are made about the correct label for cross-class duplicates.
* If clean_dataset.py is run a second time, it aborts rather than
  silently overwriting cleaned_dataset/.
* No Python packages are installed or changed.
* No model is trained.
"""

import csv
import hashlib
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR  = Path(__file__).resolve().parent
PREPARED_DIR = PROJECT_DIR / "prepared_dataset"
CLEANED_DIR  = PROJECT_DIR / "cleaned_dataset"
REPORT_DIR   = PROJECT_DIR  # CSVs and summary land next to the script

SPLITS = ["train", "val", "test"]
EXPECTED_CLASSES = [
    "capacitor", "diode", "fuse", "inductor", "integrated_circuit",
    "led", "potentiometer", "push_button", "resistor", "transistor",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}

CHUNK = 1 << 20   # 1 MiB read buffer for hashing

SEP  = "=" * 72
SEP2 = "-" * 72

# ── Category constants ─────────────────────────────────────────────────────
CAT_UNIQUE      = "UNIQUE"        # no duplicate
CAT_SAME_CLASS  = "SAME_CLASS"    # duplicate, same label, same split(s)
CAT_CROSS_SPLIT = "CROSS_SPLIT"   # same image appears in > 1 split
CAT_CROSS_CLASS = "CROSS_CLASS"   # same image appears under different labels
CAT_AMBIGUOUS   = "AMBIGUOUS"     # does not fit a clean category


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 – Collect images and compute SHA-256 hashes
# ══════════════════════════════════════════════════════════════════════════

def sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def collect_and_hash(prepared_dir: Path) -> list[dict]:
    """
    Walk prepared_dir and return a list of record dicts, one per image:
        {path, split, cls, filename, sha256}
    """
    records = []
    for split in SPLITS:
        split_dir = prepared_dir / split
        if not split_dir.exists():
            print(f"  WARNING: split directory not found: {split_dir}")
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for p in sorted(cls_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    digest = sha256_of_file(p)
                    records.append(
                        dict(path=p, split=split, cls=cls,
                             filename=p.name, sha256=digest)
                    )
    return records


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 – Classify duplicate groups
# ══════════════════════════════════════════════════════════════════════════

def classify_groups(records: list[dict]) -> tuple[dict, dict]:
    """
    Group records by SHA-256 hash.

    Returns
    -------
    groups : dict  {sha256: [record, ...]}
        All groups (including singletons).
    categories : dict  {sha256: CAT_*}
        Category label for every hash.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["sha256"]].append(rec)

    categories: dict[str, str] = {}
    for digest, members in groups.items():
        if len(members) == 1:
            categories[digest] = CAT_UNIQUE
            continue

        splits_seen  = {m["split"] for m in members}
        classes_seen = {m["cls"]   for m in members}

        cross_split  = len(splits_seen)  > 1
        cross_class  = len(classes_seen) > 1

        if cross_class:
            # Cross-class always takes priority; it implies potential mislabelling.
            categories[digest] = CAT_CROSS_CLASS
        elif cross_split:
            # Same class but image leaked across splits → data leakage risk.
            categories[digest] = CAT_CROSS_SPLIT
        else:
            # Same class, same split – simple intra-split duplicate.
            categories[digest] = CAT_SAME_CLASS

    return dict(groups), categories


# ══════════════════════════════════════════════════════════════════════════
# Phase 3a – Write CSV reports
# ══════════════════════════════════════════════════════════════════════════

def write_duplicate_report(groups: dict, categories: dict,
                           out_path: Path) -> int:
    """
    Write duplicate_report.csv.
    Columns: group_id, category, sha256, split, class, filename, full_path
    Returns the number of duplicate groups written.
    """
    dup_groups = {h: g for h, g in groups.items()
                  if categories[h] != CAT_UNIQUE}

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group_id", "category", "sha256",
             "split", "class", "filename", "full_path"]
        )
        for gid, (digest, members) in enumerate(
                sorted(dup_groups.items()), start=1):
            cat = categories[digest]
            for m in members:
                writer.writerow([
                    gid, cat, digest,
                    m["split"], m["cls"], m["filename"], str(m["path"]),
                ])
    return len(dup_groups)


def write_cross_class_report(groups: dict, categories: dict,
                             out_path: Path) -> int:
    """
    Write cross_class_duplicates.csv  (manual-review file).
    Columns: group_id, sha256, split, class, filename, full_path, review_note
    Returns the number of cross-class groups written.
    """
    cc_groups = {h: g for h, g in groups.items()
                 if categories[h] == CAT_CROSS_CLASS}

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["group_id", "sha256", "split", "class",
             "filename", "full_path", "review_note"]
        )
        for gid, (digest, members) in enumerate(
                sorted(cc_groups.items()), start=1):
            classes = sorted({m["cls"] for m in members})
            note = (
                f"MANUAL REVIEW REQUIRED – image appears under "
                f"{len(classes)} different labels: {', '.join(classes)}. "
                f"All copies excluded from cleaned_dataset/ until resolved."
            )
            for m in members:
                writer.writerow([
                    gid, digest,
                    m["split"], m["cls"], m["filename"],
                    str(m["path"]), note,
                ])
    return len(cc_groups)


# ══════════════════════════════════════════════════════════════════════════
# Phase 3b – Build cleaned_dataset/
# ══════════════════════════════════════════════════════════════════════════

def decide_retained(groups: dict, categories: dict) -> tuple[set, dict]:
    """
    Determine which files are copied to cleaned_dataset/.

    Returns
    -------
    retained_paths : set of Path
        Files that will be copied.
    exclusion_log  : dict {path: reason_str}
        Files that will NOT be copied, with the reason why.
    """
    retained_paths: set[Path] = set()
    exclusion_log:  dict[Path, str] = {}

    for digest, members in groups.items():
        cat = categories[digest]

        if cat == CAT_UNIQUE:
            retained_paths.add(members[0]["path"])
            continue

        if cat == CAT_SAME_CLASS:
            # Same class, same split: keep the lexicographically first
            # filename for determinism; exclude the rest.
            by_split: dict[str, list[dict]] = defaultdict(list)
            for m in members:
                by_split[m["split"]].append(m)

            for split_members in by_split.values():
                split_members.sort(key=lambda x: x["filename"])
                keep = split_members[0]
                retained_paths.add(keep["path"])
                for m in split_members[1:]:
                    exclusion_log[m["path"]] = (
                        f"SAME_CLASS duplicate – retained copy is "
                        f"{keep['split']}/{keep['cls']}/{keep['filename']} "
                        f"[SHA-256: {digest[:16]}...]"
                    )
            continue

        if cat == CAT_CROSS_SPLIT:
            # Same class, same image in multiple splits → data leakage.
            # Exclude all copies; do not guess the canonical split.
            for m in members:
                exclusion_log[m["path"]] = (
                    f"CROSS_SPLIT duplicate – same image (SHA-256: "
                    f"{digest[:16]}...) appears in splits: "
                    f"{sorted({x['split'] for x in members})}. "
                    f"All copies excluded to prevent data leakage."
                )
            continue

        if cat == CAT_CROSS_CLASS:
            # Different labels for the same image → exclude all.
            classes = sorted({m["cls"] for m in members})
            for m in members:
                exclusion_log[m["path"]] = (
                    f"CROSS_CLASS duplicate – image appears under labels "
                    f"{classes} (SHA-256: {digest[:16]}...). "
                    f"All copies excluded pending manual review."
                )
            continue

        # Fallback: AMBIGUOUS
        for m in members:
            exclusion_log[m["path"]] = (
                f"AMBIGUOUS duplicate group – excluded for safety "
                f"(SHA-256: {digest[:16]}...)."
            )

    return retained_paths, exclusion_log


def build_cleaned_dataset(retained_paths: set[Path],
                          exclusion_log: dict[Path, str],
                          cleaned_dir: Path,
                          prepared_dir: Path) -> None:
    """
    Copy retained files into cleaned_dir, preserving split/class structure.
    Raises FileExistsError if cleaned_dir already exists (safety guard).
    """
    if cleaned_dir.exists():
        raise FileExistsError(
            f"\n  {cleaned_dir}\n  already exists.\n"
            "  Rename or delete it manually before re-running clean_dataset.py.\n"
            "  This guard prevents accidental overwriting of prior cleaning runs."
        )

    copied = 0
    for src in sorted(retained_paths):
        # Reconstruct relative path: split/class/filename
        try:
            rel = src.relative_to(prepared_dir)
        except ValueError:
            print(f"  WARNING: could not compute relative path for {src}; skipping.")
            continue
        dst = cleaned_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"  Copied {copied} files into {cleaned_dir}")


# ══════════════════════════════════════════════════════════════════════════
# Phase 3c – cleaning_summary.txt
# ══════════════════════════════════════════════════════════════════════════

def write_summary(records: list[dict],
                  retained_paths: set[Path],
                  exclusion_log: dict[Path, str],
                  groups: dict,
                  categories: dict,
                  out_path: Path) -> None:
    """Write cleaning_summary.txt with before/after counts and exclusion log."""

    # ── Before counts ──────────────────────────────────────────────────────
    before: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in SPLITS}
    for rec in records:
        before[rec["split"]][rec["cls"]] += 1

    # ── After counts ───────────────────────────────────────────────────────
    after: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in SPLITS}
    for p in retained_paths:
        parts = p.relative_to(PREPARED_DIR).parts   # split, class, filename
        if len(parts) == 3:
            after[parts[0]][parts[1]] += 1

    # ── Category counts ────────────────────────────────────────────────────
    cat_counts: dict[str, int] = defaultdict(int)
    for digest, cat in categories.items():
        if cat != CAT_UNIQUE:
            cat_counts[cat] += 1

    lines: list[str] = []
    a = lines.append

    a(SEP)
    a("  Electronic Component Classifier – Cleaning Summary")
    a(SEP)
    a("")
    a("SOURCE      : prepared_dataset/")
    a("OUTPUT      : cleaned_dataset/")
    a("HASH METHOD : SHA-256")
    a("")
    a("Duplicate group categories")
    a(SEP2)
    a(f"  SAME_CLASS   (intra-split, same label, kept one copy) : "
      f"{cat_counts[CAT_SAME_CLASS]}")
    a(f"  CROSS_SPLIT  (same image in multiple splits, excluded) : "
      f"{cat_counts[CAT_CROSS_SPLIT]}")
    a(f"  CROSS_CLASS  (different labels, excluded for review)   : "
      f"{cat_counts[CAT_CROSS_CLASS]}")
    a(f"  AMBIGUOUS    (excluded for safety)                     : "
      f"{cat_counts[CAT_AMBIGUOUS]}")
    a("")

    # ── Per-class, per-split table ─────────────────────────────────────────
    for split in SPLITS:
        a(SEP2)
        a(f"  Split: {split.upper()}")
        a(SEP2)
        header = f"  {'Class':<24}{'Before':>8}{'After':>8}{'Removed':>9}"
        a(header)
        a("  " + "-" * 49)
        split_before = split_after = split_removed = 0
        for cls in sorted(EXPECTED_CLASSES):
            b = before[split].get(cls, 0)
            af = after[split].get(cls, 0)
            rm = b - af
            split_before  += b
            split_after   += af
            split_removed += rm
            a(f"  {cls:<24}{b:>8}{af:>8}{rm:>9}")
        a("  " + "-" * 49)
        a(f"  {'TOTAL':<24}{split_before:>8}{split_after:>8}"
          f"{split_removed:>9}")
        a("")

    # ── Grand totals ───────────────────────────────────────────────────────
    grand_b  = sum(r for s in before.values() for r in s.values())
    grand_a  = sum(r for s in after.values()  for r in s.values())
    grand_rm = grand_b - grand_a
    a(SEP)
    a(f"  GRAND TOTAL   Before: {grand_b}   After: {grand_a}"
      f"   Removed: {grand_rm}")
    a(SEP)
    a("")

    # ── Exclusion log ──────────────────────────────────────────────────────
    if exclusion_log:
        a("Excluded files and reasons")
        a(SEP2)
        for p in sorted(exclusion_log.keys()):
            try:
                rel = p.relative_to(PREPARED_DIR)
            except ValueError:
                rel = p
            a(f"  EXCLUDED  {rel}")
            a(f"  REASON    {exclusion_log[p]}")
            a("")

    a(SEP)
    a("  No original files were modified, moved, or deleted.")
    a(SEP)

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════
# Console summary helper
# ══════════════════════════════════════════════════════════════════════════

def print_console_summary(records: list[dict],
                          groups: dict,
                          categories: dict,
                          retained_paths: set[Path],
                          exclusion_log: dict[Path, str]) -> None:
    total         = len(records)
    unique_hashes = sum(1 for c in categories.values() if c == CAT_UNIQUE)
    dup_groups    = sum(1 for c in categories.values() if c != CAT_UNIQUE)
    same_class_g  = sum(1 for c in categories.values() if c == CAT_SAME_CLASS)
    cross_split_g = sum(1 for c in categories.values() if c == CAT_CROSS_SPLIT)
    cross_class_g = sum(1 for c in categories.values() if c == CAT_CROSS_CLASS)
    ambiguous_g   = sum(1 for c in categories.values() if c == CAT_AMBIGUOUS)

    print(SEP)
    print("  CLEANING SUMMARY")
    print(SEP)
    print(f"  Total images scanned       : {total}")
    print(f"  Unique SHA-256 hashes      : {len(categories)}")
    print(f"  Unique images (no dupes)   : {unique_hashes}")
    print(f"  Duplicate groups total     : {dup_groups}")
    print(f"    SAME_CLASS  (kept 1/split): {same_class_g}")
    print(f"    CROSS_SPLIT (excluded all): {cross_split_g}")
    print(f"    CROSS_CLASS (excl / review): {cross_class_g}")
    print(f"    AMBIGUOUS   (excl / safety): {ambiguous_g}")
    print(f"  Files in cleaned_dataset/  : {len(retained_paths)}")
    print(f"  Files excluded             : {len(exclusion_log)}")
    print(SEP)
    print()
    print("  Output files created:")
    print(f"    duplicate_report.csv       (all duplicate groups)")
    print(f"    cross_class_duplicates.csv (manual review required)")
    print(f"    cleaning_summary.txt       (before/after per class)")
    print(f"    cleaned_dataset/           (ready-to-use split tree)")
    print(SEP)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print(SEP)
    print("  Electronic Component Classifier – Dataset Cleaner")
    print(f"  Source : {PREPARED_DIR}")
    print(f"  Output : {CLEANED_DIR}")
    print(SEP)

    # ── Guard: source must exist ───────────────────────────────────────────
    if not PREPARED_DIR.exists():
        sys.exit(
            f"ERROR: {PREPARED_DIR} does not exist.\n"
            "Run split_dataset.py first."
        )

    # ── Guard: cleaned_dataset must NOT exist yet ──────────────────────────
    if CLEANED_DIR.exists():
        sys.exit(
            f"ERROR: {CLEANED_DIR} already exists.\n"
            "Rename or remove it manually before running clean_dataset.py again.\n"
            "This prevents accidental overwriting of a previous cleaning run."
        )

    # ── Phase 1: Hash ──────────────────────────────────────────────────────
    print("\n[1/5] Hashing images with SHA-256 …")
    records = collect_and_hash(PREPARED_DIR)
    print(f"      {len(records)} images hashed.")

    # ── Phase 2: Classify ──────────────────────────────────────────────────
    print("[2/5] Classifying duplicate groups …")
    groups, categories = classify_groups(records)
    n_dup = sum(1 for c in categories.values() if c != CAT_UNIQUE)
    print(f"      {n_dup} duplicate groups identified.")

    # ── Phase 3a: CSV reports ──────────────────────────────────────────────
    print("[3/5] Writing CSV reports …")
    dup_report_path = REPORT_DIR / "duplicate_report.csv"
    cc_report_path  = REPORT_DIR / "cross_class_duplicates.csv"

    n_dup_written = write_duplicate_report(groups, categories, dup_report_path)
    n_cc_written  = write_cross_class_report(groups, categories, cc_report_path)
    print(f"      duplicate_report.csv       – {n_dup_written} groups")
    print(f"      cross_class_duplicates.csv – {n_cc_written} groups "
          f"(MANUAL REVIEW REQUIRED)")

    # ── Phase 3b: Decide what to keep ─────────────────────────────────────
    print("[4/5] Deciding which files to retain …")
    retained_paths, exclusion_log = decide_retained(groups, categories)
    print(f"      Retained : {len(retained_paths)}")
    print(f"      Excluded : {len(exclusion_log)}")

    # ── Phase 3b: Build cleaned_dataset ───────────────────────────────────
    print("[5/5] Building cleaned_dataset/ …")
    build_cleaned_dataset(retained_paths, exclusion_log,
                          CLEANED_DIR, PREPARED_DIR)

    # ── cleaning_summary.txt ───────────────────────────────────────────────
    summary_path = REPORT_DIR / "cleaning_summary.txt"
    write_summary(records, retained_paths, exclusion_log,
                  groups, categories, summary_path)
    print(f"      cleaning_summary.txt written.")

    # ── Console summary ────────────────────────────────────────────────────
    print()
    print_console_summary(records, groups, categories,
                          retained_paths, exclusion_log)

    print("  Original prepared_dataset/ was NOT modified.")
    print()


if __name__ == "__main__":
    main()
