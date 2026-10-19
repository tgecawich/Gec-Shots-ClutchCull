"""Extract jersey/torso crops from a folder of game photos, for labeling.

Step 1 of training the jersey digit classifier (the recognition model in
jersey.py has perfect precision but only ~29% recall; the fix is a small
classifier trained on real jersey crops, and that needs labeled data).

Usage:
    cd api
    python tools/extract_jersey_crops.py /path/to/shoot1 /path/to/shoot2 ...

Writes torso crops to tools/crops/<shoot-name>/ as JPEGs, one per detected
person, sized for labeling (448px wide). Then open tools/label_crops.html in a
browser, point it at the crops folder, and label. Nothing here is imported by
the API; it is a local tool only.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Run from api/ so `import engine` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

import engine  # noqa: E402

OUT_ROOT = Path(__file__).parent / "crops"
CROP_WIDTH = 448          # big enough to read a number, small enough to label fast
MAX_PEOPLE_PER_FRAME = 6  # same cap as jersey.py; benches add noise past this
MIN_CROP_PX = 56          # skip tiny background detections


def torso_box(x: float, y: float, w: float, h: float, W: int, H: int, sc: float):
    """Same torso framing jersey.py uses, so training data matches inference."""
    x0 = max(0, int((x + w * 0.06) * sc))
    y0 = max(0, int((y + h * 0.12) * sc))
    x1 = min(W, int((x + w * 0.94) * sc))
    y1 = min(H, int((y + h * 0.66) * sc))
    return x0, y0, x1, y1


def extract(folder: Path) -> int:
    out = OUT_ROOT / folder.name
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    for i, path in enumerate(files):
        try:
            im = Image.open(path)
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        small = im.copy()
        small.thumbnail((1600, 1600))
        sc = W / small.size[0]
        people = engine.detect_persons(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR))
        for j, (x, y, w, h, _c) in enumerate(
                sorted(people, key=lambda p: -p[2] * p[3])[:MAX_PEOPLE_PER_FRAME]):
            x0, y0, x1, y1 = torso_box(x, y, w, h, W, H, sc)
            if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
                continue
            crop = im.crop((x0, y0, x1, y1))
            k = CROP_WIDTH / crop.width
            crop = crop.resize((CROP_WIDTH, max(1, round(crop.height * k))))
            name = f"{path.stem}_p{j}.jpg"
            crop.save(out / name, "JPEG", quality=88)
            count += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)} frames, {count} crops")
    return count


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if not engine.person_detection_enabled():
        print("Person detection unavailable (model missing?). Aborting.")
        sys.exit(1)
    total = 0
    for arg in sys.argv[1:]:
        folder = Path(arg).expanduser()
        if not folder.is_dir():
            print(f"skip (not a folder): {folder}")
            continue
        print(f"{folder.name}:")
        n = extract(folder)
        print(f"  -> {n} crops in {OUT_ROOT / folder.name}")
        total += n
    print(f"\nDone: {total} crops. Next: open tools/label_crops.html and label them.")


if __name__ == "__main__":
    main()
