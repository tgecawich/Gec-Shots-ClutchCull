"""Prepare several shoots off a card for judging, in one pass.

Each additional judged shoot is currently worth about +4 points of agreement
(measured: training on one shoot scores 60.9% on a held-out shoot, training on
two scores 64.9%), so the bottleneck is simply how many shoots exist as labelled
data. This removes the friction from producing more.

Two decisions worth stating, because both affect what the numbers mean.

SAMPLING IS RANDOM, with a fixed seed per shoot. Picking frames that span the
engine's score range would be more efficient, and would also bias the test
toward the engine's own opinion of what is interesting. The point is to measure
the engine, not to hand it a flattering subset.

SIZE IS 140 FRAMES, not the whole shoot. Pair COVERAGE is what makes a ranking
trustworthy, not frame count. 150 comparisons over 800 frames is 0.4 per frame
and orders nothing; over 140 frames it is 2.1 per frame, which is enough.

JUDGE ~150 PAIRS PER SHOOT, not 250. Measured by subsampling the comparisons
already collected and scoring on a held-out shoot:

      40 comparisons  61.6%        160 comparisons  63.6%
      80 comparisons  62.8%        250 comparisons  64.5%
     120 comparisons  63.2%        291 comparisons  64.9%

The curve flattens after about 80. Going from 80 to 291 costs nine more
minutes and buys 2.1 points, while an entirely NEW shoot buys 4.0. So minutes
spent deepening one shoot are worth less than half what the same minutes buy
on another shoot, and the right move is more shoots judged more shallowly.
150 captures roughly 97% of what a single shoot has to offer.

    python tools/prep_judge_batch.py /Volumes/Untitled/DCIM --out ~/Desktop
    python tools/prep_judge_batch.py <card> --out ~/Desktop --shoots 4 --n 140
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageOps  # noqa: E402

from split_shoots import PHOTO_EXT, capture_time  # noqa: E402

MAX_DIM = 1500
QUALITY = 85


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("card", type=Path)
    ap.add_argument("--out", type=Path, default=Path.home() / "Desktop")
    ap.add_argument("--shoots", type=int, default=4)
    ap.add_argument("--n", type=int, default=140, help="frames sampled per shoot")
    ap.add_argument("--min-frames", type=int, default=300,
                    help="ignore shoots smaller than this; a 30 frame run is "
                         "not a game and adds little")
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--skip", default="",
                    help="comma-separated dates already judged, e.g. 09-03")
    a = ap.parse_args()
    if not a.card.is_dir():
        sys.exit(f"not a folder: {a.card}")

    files = [p for p in a.card.rglob("*") if p.suffix.lower() in PHOTO_EXT]
    if not files:
        sys.exit(f"no photos in {a.card}")
    print(f"reading capture times for {len(files)} files ...")
    stamped = sorted(((capture_time(p), p) for p in files),
                     key=lambda t: (t[0] or 0))
    stamped = [(t, p) for t, p in stamped if t]

    runs, cur = [], [stamped[0]]
    for t, p in stamped[1:]:
        if (t - cur[-1][0]).total_seconds() > a.gap * 3600:
            runs.append(cur)
            cur = []
        cur.append((t, p))
    runs.append(cur)

    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    big = [r for r in runs if len(r) >= a.min_frames
           and f"{r[0][0]:%m-%d}" not in skip]
    # Spread the picks across the calendar rather than taking the largest
    # shoots. Different months mean different sports, light and time of day,
    # and a weight that holds across those is a weight about the photographer
    # instead of about one afternoon.
    big.sort(key=lambda r: r[0][0])
    if len(big) > a.shoots:
        step = len(big) / a.shoots
        big = [big[int(i * step)] for i in range(a.shoots)]

    if not big:
        sys.exit("no shoots large enough; lower --min-frames")
    print(f"\nselected {len(big)} shoot(s) of {len(runs)} on the card:\n")
    a.out.mkdir(parents=True, exist_ok=True)
    for r in big:
        start = r[0][0]
        tag = f"{start:%m-%d}"
        dest = a.out / f"JUDGE-{tag}"
        mins = (r[-1][0] - start).total_seconds() / 60
        print(f"  {start:%a %b %d %H:%M}  {len(r):>4} frames  {mins:>4.0f} min"
              f"  -> {dest.name}")
        if dest.exists() and any(dest.iterdir()):
            print("     already prepared, skipping")
            continue
        dest.mkdir(exist_ok=True)
        paths = [p for _, p in r]
        # Seed from the date so re-running picks the SAME frames. Without this,
        # a second run would silently produce a different sample and any
        # comparisons already collected would no longer match the folder.
        rng = random.Random(int(start.strftime("%m%d")))
        pick = sorted(rng.sample(paths, min(a.n, len(paths))), key=lambda p: p.name)
        written = 0
        for p in pick:
            try:
                im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
                im.save(dest / f"{p.stem}.jpg", "JPEG", quality=QUALITY)
                written += 1
            except Exception as e:
                print(f"     skip {p.name}: {e}")
        size = sum(f.stat().st_size for f in dest.iterdir()) / 1e6
        print(f"     wrote {written} frames, {size:.0f} MB")

    print("\nFor each folder, in tools/rank_pairs.html:")
    print("  Pick photo folder -> judge ~150 pairs -> Export")
    print("  Save as comparisons-<tag>.csv so they do not overwrite each other.")
    print("\n150, not 250: measured, the per-shoot curve flattens after ~80, and")
    print("the 9 minutes it takes to go from 80 to 290 buys 2.1 points where a")
    print("whole new shoot buys 4.0. More shoots, judged shallower, wins.")
    print("\nThen register them in tools/cross_shoot.py SHOOTS and run:")
    print("  python tools/cross_shoot.py --search")


if __name__ == "__main__":
    main()
