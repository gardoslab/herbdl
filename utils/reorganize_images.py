import os
import shutil
from pathlib import Path

SRC_DIR = Path("/projectnb/herbdl/data/GBIF-F25/images")
DST_DIR = Path("/projectnb/herbdl/data/GBIF-F25h")

DST_DIR.mkdir(parents=True, exist_ok=True)

patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
image_paths = []
for p in patterns:
    image_paths.extend(SRC_DIR.rglob(p))
    image_paths.extend(SRC_DIR.rglob(p.upper()))

print(f"Found {len(image_paths)} images to process from {SRC_DIR}")

moved = 0
skipped = 0

for file_path in image_paths:
    if not file_path.is_file():
        continue

    filename = file_path.name
    stem = file_path.stem

    if not stem.isdigit():
        print(f"Skipping: {filename}")
        skipped += 1
        continue

    prefix1 = stem[:3] if len(stem) >= 3 else stem
    prefix2 = stem[3:6] if len(stem) >= 6 else "000"

    dest_dir = DST_DIR / prefix1 / prefix2
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / filename
    shutil.move(str(file_path), dest_path)
    moved += 1
    print(f"Moved {filename} → {dest_path}")

print(f"Done. Moved: {moved}, Skipped: {skipped}")
