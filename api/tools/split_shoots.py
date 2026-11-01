"""Split a memory card, or any folder of photos, into separate shoots.

Why this exists: I first tried to separate shoots by gaps in the frame NUMBER,
and it silently merged a baseball game and a lacrosse game into one set because
they were shot four days apart but only one frame number apart. Frame numbers
tell you the order the shutter fired, not when. Capture time does.

Clustering on EXIF DateTimeOriginal instead is both simpler and correct: frames
within one game are seconds or minutes apart, and separate shoots are hours or
days apart, so a two hour gap separates them cleanly with no tuning.

Usage:
    python tools/split_shoots.py /Volumes/Untitled/DCIM/101EOSR6
    python tools/split_shoots.py <card> --out ~/Desktop/shoots      # copy them
    python tools/split_shoots.py <card> --out ~/Desktop/shoots --min 40

Reports what it finds first. Nothing is copied unless --out is given, and
nothing is ever moved or deleted from the card.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp",
             ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
GAP_HOURS = 2.0


def capture_time(path: Path) -> datetime | None:
    """EXIF capture time, falling back to the file's modification time.

    RAW files are read the same way: Pillow can parse the EXIF block of most
    of them even when it cannot decode the image itself.
    """
    try:
        ex = Image.open(path)._getexif() or {}
        tags = {TAGS.get(k, k): v for k, v in ex.items()}
        raw = tags.get("DateTimeOriginal") or tags.get("DateTime")
        if raw:
            return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, help="copy each shoot into its own folder here")
    ap.add_argument("--min", type=int, default=25,
                    help="ignore runs smaller than this (default 25, they are usually strays)")
    ap.add_argument("--gap", type=float, default=GAP_HOURS,
                    help="hours of silence that separates two shoots (default 2)")
    a = ap.parse_args()
    if not a.folder.is_dir():
        sys.exit(f"not a folder: {a.folder}")

    files = [p for p in a.folder.rglob("*") if p.suffix.lower() in PHOTO_EXT]
    if not files:
        sys.exit(f"no photos found in {a.folder}")
    print(f"reading capture times for {len(files)} files ...")
    stamped = sorted(((capture_time(p), p) for p in files), key=lambda t: (t[0] or datetime.min))
    stamped = [(t, p) for t, p in stamped if t]

    runs, cur = [], [stamped[0]]
    for t, p in stamped[1:]:
        if (t - cur[-1][0]).total_seconds() > a.gap * 3600:
            runs.append(cur)
            cur = []
        cur.append((t, p))
    runs.append(cur)

    print(f"\n{len(runs)} shoot(s) found:\n")
    keep = []
    for r in runs:
        start, end = r[0][0], r[-1][0]
        mins = (end - start).total_seconds() / 60
        small = len(r) < a.min
        label = "  (too small, skipped)" if small else ""
        print(f"  {start:%a %b %d  %H:%M} to {end:%H:%M}   {len(r):>4} frames"
              f"   {mins:>5.0f} min{label}")
        if not small:
            keep.append(r)

    if not a.out:
        print("\nreport only. Pass --out <folder> to copy each shoot into its own folder.")
        return

    a.out.mkdir(parents=True, exist_ok=True)
    for r in keep:
        name = f"{r[0][0]:%Y-%m-%d_%H%M}_{len(r)}frames"
        dest = a.out / name
        dest.mkdir(exist_ok=True)
        for _, p in r:
            target = dest / p.name
            if not target.exists():
                shutil.copy2(p, target)
        print(f"  copied {len(r):>4} frames -> {dest}")
    print(f"\n{len(keep)} shoot(s) written to {a.out}")
    print("Next: judge one with tools/rank_pairs.html, then")
    print("      python tools/eval_cull.py learn <shoot> --comparisons comparisons.csv")


if __name__ == "__main__":
    main()
