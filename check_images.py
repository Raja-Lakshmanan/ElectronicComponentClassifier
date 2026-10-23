
from pathlib import Path
from PIL import Image

root = Path("dataset")
extensions = {".jpg", ".jpeg", ".png", ".webp"}

files = [
    p for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in extensions
]

bad_images = []

print("Checking images. Please wait...")

for path in files:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as error:
        bad_images.append((path, str(error)))

print("\nImage check complete!")
print("Total image files:", len(files))
print("Corrupted images:", len(bad_images))

if bad_images:
    print("\nFirst 20 problematic files:")
    for path, error in bad_images[:20]:
        print(path, "|", error)
else:
    print("All images passed the file-integrity check.")
