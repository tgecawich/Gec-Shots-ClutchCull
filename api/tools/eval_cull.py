"""Measure whether the culling engine is actually any good, and whether a
change made it better or worse.

Why this exists: "improve the algorithm" is unfalsifiable without a yardstick.
The last time scoring was changed on intuition it rejected entire shoots in
production. Nothing here changes engine behaviour; it only measures it.

Three modes, each answering a different question.

1. ACCURACY  (needs ground truth: the frames YOU kept)
       python tools/eval_cull.py accuracy /path/to/shoot --kept kept.txt
   kept.txt is one filename per line, the frames you actually used. Reports
   precision@k, recall@k and where your picks landed in the engine's ranking.
   This is the only mode that can say "more correct".

2. STABILITY (no ground truth needed)
       python tools/eval_cull.py stability /path/to/shoot
   A photo's score should be a property of the photo. Because several metrics
   are normalised against batch percentiles, it is currently partly a property
   of what else was uploaded. This measures how often ranking flips and how far
   scores drift when the batch changes.

3. ABLATION  (no ground truth needed)
       python tools/eval_cull.py ablation /path/to/shoot
   Zeroes each scoring dimension in turn and measures how much the ranking
   moves. A dimension that can be removed without changing the order is costing
   weight while contributing nothing.
"""
from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import engine  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load(folder: Path):
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        sys.exit(f"no images in {folder}")
    print(f"scoring {len(paths)} frames ...")
    metrics = engine.compute_metrics_batch(paths)
    return {m["filename"]: m for m in metrics if not m.get("unreadable")}


def order_of(by_name, names, preset="Balanced", weights=None):
    """Ranked filenames for a subset. weights overrides the preset."""
    subset = [by_name[n] for n in names]
    if weights is None:
        res = engine.rank_metrics(subset, top_n=len(subset), preset=preset)
    else:
        saved = engine.SCORING_PRESETS.get("__eval__")
        engine.SCORING_PRESETS["__eval__"] = weights
        res = engine.rank_metrics(subset, top_n=len(subset), preset="__eval__")
        if saved is None:
            engine.SCORING_PRESETS.pop("__eval__", None)
    return [k["filename"] for k in res["keepers"]], {k["filename"]: k["score"] for k in res["keepers"]}


# ── 1. accuracy ────────────────────────────────────────────────────────────
def accuracy(folder: Path, kept_file: Path, preset: str):
    by_name = load(folder)
    kept = {l.strip() for l in kept_file.read_text().splitlines() if l.strip()}
    kept = {k for k in kept if k in by_name}
    if not kept:
        sys.exit("none of the filenames in --kept matched images in the folder")
    order, _ = order_of(by_name, list(by_name), preset)
    pos = {n: i for i, n in enumerate(order)}
    n, g = len(order), len(kept)
    print(f"\nground truth: {g} keepers out of {n} frames ({100*g/n:.0f}%)\n")
    print(f"{'k':>5}{'precision@k':>14}{'recall@k':>11}   what it means")
    for k in sorted({g, g * 2, 10, 25, 35, 50, n}):
        if k < 1 or k > n:
            continue
        hits = sum(1 for f in order[:k] if f in kept)
        print(f"{k:>5}{hits/k:>14.2f}{hits/g:>11.2f}   top {k} contains {hits} of your {g} picks")
    ranks = sorted(pos[f] + 1 for f in kept)
    print(f"\nyour picks landed at ranks: {ranks[:20]}{' ...' if len(ranks) > 20 else ''}")
    print(f"median rank of a real keeper: {int(np.median(ranks))} of {n}")
    worst = ranks[-1]
    print(f"worst-placed real keeper: rank {worst}  "
          f"({'inside' if worst <= g else 'OUTSIDE'} a top-{g} cull)")
    # A random ranker would put half your picks in the top half.
    baseline = sum(1 for f in order[:g] if f in kept) / g
    print(f"\nrecall@{g} = {baseline:.2f}   (random guessing would score {g/n:.2f})")


# ── 2. stability ───────────────────────────────────────────────────────────
def stability(folder: Path, preset: str, trials: int = 40):
    by_name = load(folder)
    names = list(by_name)
    full_order, full_scores = order_of(by_name, names, preset)
    pos_full = {n: i for i, n in enumerate(full_order)}
    rng = random.Random(7)
    flips = pairs = 0
    drift = []
    for _ in range(trials):
        k = rng.choice([5, 8, 12, 20, max(5, len(names) // 2)])
        if k >= len(names):
            continue
        sub = rng.sample(names, k)
        order, scores = order_of(by_name, sub, preset)
        pos = {n: i for i, n in enumerate(order)}
        for a, b in itertools.combinations(sub, 2):
            if a in pos and b in pos:
                pairs += 1
                if (pos_full[a] < pos_full[b]) != (pos[a] < pos[b]):
                    flips += 1
        for s in sub:
            if s in scores and s in full_scores:
                drift.append(abs(scores[s] - full_scores[s]))
    drift.sort()
    print(f"\nre-ranked {trials} random subsets of the same shoot\n")
    print(f"pair order flips : {flips}/{pairs} = {100*flips/max(pairs,1):.1f}%")
    print(f"score drift      : median {np.median(drift):.1f} pts, "
          f"p90 {np.percentile(drift,90):.1f}, max {max(drift):.1f}")
    print("\nA photo's score should depend on the photo, not on what else was")
    print("uploaded with it. Anything above 0% here is batch contamination:")
    print("subject sharpness, detail and contrast are scaled to batch")
    print("percentiles in engine._normalize, so changing the batch changes them.")


# ── 3. ablation ────────────────────────────────────────────────────────────
def ablation(folder: Path, preset: str):
    by_name = load(folder)
    names = list(by_name)
    base_w = dict(engine.SCORING_PRESETS[preset])
    base_order, _ = order_of(by_name, names, weights=base_w)
    base_pos = {n: i for i, n in enumerate(base_order)}
    print(f"\nbaseline preset: {preset}  {base_w}\n")
    print(f"{'dimension':<12}{'weight':>8}{'pairs moved':>14}{'top-10 changed':>16}   verdict")
    for dim in base_w:
        w = dict(base_w)
        w[dim] = 0.0
        if sum(w.values()) == 0:
            continue
        order, _ = order_of(by_name, names, weights=w)
        pos = {n: i for i, n in enumerate(order)}
        moved = sum(1 for a, b in itertools.combinations(names, 2)
                    if (base_pos[a] < base_pos[b]) != (pos[a] < pos[b]))
        total = len(names) * (len(names) - 1) // 2
        top_change = len(set(base_order[:10]) ^ set(order[:10])) // 2
        verdict = ("CARRIES THE RANKING" if moved / total > 0.15 else
                   "contributes" if moved / total > 0.04 else
                   "NEARLY DEAD WEIGHT")
        print(f"{dim:<12}{base_w[dim]:>8.2f}{moved/total:>13.1%}{top_change:>16}   {verdict}")
    print("\n'pairs moved' is the share of photo pairs that swap order when the")
    print("dimension is removed. A dimension near 0% is being paid weight for")
    print("nothing: its value is nearly identical on every frame, so it shifts")
    print("every score by the same amount and changes no decision.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["accuracy", "stability", "ablation"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--kept", type=Path, help="file of kept filenames (accuracy mode)")
    ap.add_argument("--preset", default="Balanced", choices=list(engine.SCORING_PRESETS))
    a = ap.parse_args()
    if not a.folder.is_dir():
        sys.exit(f"not a folder: {a.folder}")
    if a.mode == "accuracy":
        if not a.kept or not a.kept.exists():
            sys.exit("accuracy mode needs --kept pointing at a file of filenames you kept")
        accuracy(a.folder, a.kept, a.preset)
    elif a.mode == "stability":
        stability(a.folder, a.preset)
    else:
        ablation(a.folder, a.preset)


if __name__ == "__main__":
    main()
