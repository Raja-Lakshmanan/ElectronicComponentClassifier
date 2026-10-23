
from pathlib import Path
from io import BytesIO
import time

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "raw_dataset"
OUTPUT_DIR = PROJECT_DIR / "dataset"

CLASS_FILES = {
    "through_hole_resistors.csv": "resistor",
    "Capacitors_aluminum_electrolytic.csv": "capacitor",
    "mica_and_ptfe_capacitors.csv": "capacitor",
    "fixed_inductors.csv": "inductor",
    "single_diodes.csv": "diode",
    "single_bipolar_transistors.csv": "transistor",
    "led_character_and_numeric.csv": "led",
    "instrumentation__op_amps__buffer_amps.csv": "integrated_circuit",
    "rotary_potentiometers__rheostats.csv": "potentiometer",
    "pushbutton_switches.csv": "push_button",
    "fuses.csv": "fuse",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
    )
}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    total_saved = 0

    for csv_name, class_name in CLASS_FILES.items():
        csv_path = RAW_DIR / csv_name

        if not csv_path.exists():
            print(f"\nMISSING CSV: {csv_name}")
            continue

        df = pd.read_csv(csv_path)

        if "Image" not in df.columns:
            print(f"\nNo Image column in {csv_name}: {list(df.columns)}")
            continue

        out_dir = OUTPUT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Remove repeated URLs within this CSV.
        urls = df["Image"].dropna().astype(str).str.strip().drop_duplicates()

        saved = 0
        failed = 0
        existing = 0

        print(f"\nDownloading {class_name}: {len(urls)} unique URLs")

        for index, url in tqdm(urls.items(), desc=class_name):
            if not url.startswith(("http://", "https://")):
                failed += 1
                continue

            destination = out_dir / f"{csv_path.stem}_{index:05d}.jpg"

            if destination.exists():
                existing += 1
                continue

            try:
                response = session.get(url, timeout=20)
                response.raise_for_status()

                with Image.open(BytesIO(response.content)) as img:
                    img.load()
                    img.convert("RGB").save(
                        destination, format="JPEG", quality=95
                    )

                saved += 1
                total_saved += 1

            except Exception as exc:
                failed += 1

                # Show the first few errors for diagnosis.
                if failed <= 3:
                    print(f"\nDownload error: {url}")
                    print(f"Reason: {type(exc).__name__}: {exc}")

            time.sleep(0.05)

        print(
            f"{class_name}: saved={saved}, "
            f"already present={existing}, failed={failed}"
        )

    print(f"\nFinished. Newly saved images: {total_saved}")
    print(f"Dataset folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
