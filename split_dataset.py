
from pathlib import Path
import random
import shutil

# Reproducible split
random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
SPLIT_DIR = PROJECT_DIR / "prepared_dataset"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Prevent accidental overwriting of an existing split.
if SPLIT_DIR.exists():
    raise FileExistsError(
        f"{SPLIT_DIR} already exists. "
        "Rename or remove it only after checking its contents."
    )

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    # Exclude the manually downloaded test image.
    images = [
        p for p in class_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.name.lower() != "test_image.jpg"
    ]

    random.shuffle(images)

    total = len(images)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_files = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, files in split_files.items():
        destination = SPLIT_DIR / split_name / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        for source in files:
            shutil.copy2(source, destination / source.name)

    print(
        f"{class_dir.name}: "
        f"train={len(split_files['train'])}, "
        f"val={len(split_files['val'])}, "
        f"test={len(split_files['test'])}"
    )

print("\nDataset split complete.")
print("Prepared dataset:", SPLIT_DIR)
print("Original dataset remains unchanged.")
