"""Prepare a raw-off-the-card shoot for evaluation, exactly as the browser does.

Evaluating the engine on full-resolution originals would measure something the
engine never sees. In production the browser extracts the embedded preview from
a RAW, downscales the long edge to 1400px and encodes JPEG q72 before upload.
If the benchmark skips that, blur separation measures far higher than reality
(sharp-vs-soft is ~4x at 1200px but ~990x at native) and every conclusion drawn
from it is wrong.

So this mirrors web/lib/raw.ts and web/lib/resize.ts in Python:
  RAW -> embedded JPEG -> EXIF rotate -> 1400px long edge -> q72

Usage:
    python tools/prep_shoot.py ~/Desktop/game-raw  ~/Desktop/game-prepped

Output filenames keep the ORIGINAL stem (D91A1234.CR3 -> D91A1234.jpg) so a
list of keepers can be matched back whatever format they were exported in.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageOps

RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
JPEG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_DIM = 1400
QUALITY = 72


def embedded_jpegs(data: bytes):
    """Yield (start, end) of every embedded JPEG, largest last.

    Format-agnostic on purpose: rather than parsing TIFF IFDs for CR2/NEF/ARW
    and ISO-BMFF boxes for CR3, scan for JPEG SOI/EOI markers. One code path
    covers every RAW container, including ones that did not exist when this was
    written. The largest JPEG found is the full-size preview.
    """
    spans = []
    i = 0
    n = len(data)
    while True:
        s = data.find(b"\xff\xd8\xff", i)
        if s < 0:
            break
        e = data.find(b"\xff\xd9", s + 3)
        if e < 0:
            break
        spans.append((s, e + 2))
        i = s + 3
    spans.sort(key=lambda sp: sp[1] - sp[0])
    return spans


def open_any(path: Path) -> Image.Image | None:
    ext = path.suffix.lower()
    if ext in JPEG_EXT:
        try:
            return Image.open(path)
        except Exception:
            return None
    if ext in RAW_EXT:
        data = path.read_bytes()
        for s, e in reversed(embedded_jpegs(data)):     # largest first
            try:
                im = Image.open(__import__("io").BytesIO(data[s:e]))
                im.load()
                # Thumbnails are tiny; the preview we want is sensor-sized.
                if max(im.size) >= 800:
                    return im
            except Exception:
                continue
        return None
    return None


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()
    if not src.is_dir():
        sys.exit(f"not a folder: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.iterdir()
                   if p.suffix.lower() in RAW_EXT | JPEG_EXT)
    if not files:
        sys.exit(f"no photos found in {src}")

    done = skipped = paired = 0
    for i, p in enumerate(files, 1):
        out = dst / (p.stem + ".jpg")
        # A RAW+JPEG pair shares a stem, so the pair collapses to one prepped
        # frame. That is correct: otherwise the same photograph would enter the
        # benchmark twice and be scored as two separate frames.
        if out.exists():
            paired += 1
            continue
        im = open_any(p)
        if im is None:
            print(f"  skip (no readable preview): {p.name}")
            skipped += 1
            continue
        try:
            im = ImageOps.exif_transpose(im).convert("RGB")
            k = min(1.0, MAX_DIM / max(im.size))
            if k < 1.0:
                im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                               Image.LANCZOS)
            im.save(out, "JPEG", quality=QUALITY)
            done += 1
        except Exception as e:
            print(f"  skip ({e}): {p.name}")
            skipped += 1
        if i % 50 == 0:
            print(f"  {i}/{len(files)} ...")
    print(f"\nprepared {done} unique frames into {dst}"
          + (f", {paired} RAW+JPEG pairs collapsed" if paired else "")
          + (f", skipped {skipped}" if skipped else ""))
    print("next:  python tools/eval_cull.py accuracy", dst, "--kept-folder /path/to/your/keepers")


if __name__ == "__main__":
    main()
