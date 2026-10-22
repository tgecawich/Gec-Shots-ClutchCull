"""Measure whether the culling engine is actually any good, and whether a
change made it better or worse.

Why this exists: "improve the algorithm" is unfalsifiable without a yardstick.
The last time scoring was changed on intuition it rejected entire shoots in
production. Nothing here changes engine behaviour; it only measures it.

Three modes, each answering a different question.

1. ACCURACY  (needs ground truth: the frames YOU kept)
       python tools/eval_cull.py accuracy /path/to/shoot --kept-folder /path/to/keepers
   Point --kept-folder at the frames you actually used (or --kept at a text
   file of filenames). Matching is by filename stem, so a CR3 roll pairs with
   JPEG exports automatically. Reports
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
import csv
import itertools
import re
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


def order_of(by_name, names, preset="Balanced", weights=None,
             blur=40.0, dupes=2):
    """Ranked filenames for a subset. weights overrides the preset.

    blur=0, dupes=0 produces a COMPLETE ordering of every frame. That matters
    for evaluation: rank_metrics only returns survivors, so a frame the engine
    rejected outright disappears from the result. Measuring recall against the
    survivors only would hide the single worst failure this engine can have,
    which is discarding a photo the photographer wanted.
    """
    subset = [by_name[n] for n in names]
    if weights is None:
        res = engine.rank_metrics(subset, top_n=len(subset), preset=preset,
                                  blur_threshold=blur, duplicate_threshold=dupes)
    else:
        saved = engine.SCORING_PRESETS.get("__eval__")
        engine.SCORING_PRESETS["__eval__"] = weights
        res = engine.rank_metrics(subset, top_n=len(subset), preset="__eval__",
                                  blur_threshold=blur, duplicate_threshold=dupes)
        if saved is None:
            engine.SCORING_PRESETS.pop("__eval__", None)
    return [k["filename"] for k in res["keepers"]], {k["filename"]: k["score"] for k in res["keepers"]}


# ── 1. accuracy ────────────────────────────────────────────────────────────
def read_kept(by_name, kept_file: Path | None, kept_folder: Path | None) -> set[str]:
    """Resolve which frames the photographer actually kept.

    Matching is on the filename STEM, case-insensitively, because the keepers
    are almost never the same files as the originals: the roll is D91A1234.CR3
    and the keeper is D91A1234.jpg exported from Lightroom. Extension matching
    would silently find zero overlap and report the engine as useless.
    """
    stems = {Path(n).stem.lower(): n for n in by_name}
    wanted: set[str] = set()
    raw: list[str] = []
    if kept_file:
        raw += [l.strip() for l in kept_file.read_text().splitlines() if l.strip()]
    if kept_folder:
        raw += [p.name for p in kept_folder.iterdir() if p.is_file() and not p.name.startswith(".")]
    unmatched = []
    for r in raw:
        s = Path(r).stem.lower()
        # ClutchCull's own export prefixes the rank (01_D91A6535.CR3), and
        # Lightroom can prefix a sequence too. Strip a leading "<digits>_" so a
        # previously exported folder still matches the original roll.
        alt = re.sub(r"^\d+[_-]", "", s)
        if s in stems:
            wanted.add(stems[s])
        elif alt in stems:
            wanted.add(stems[alt])
        else:
            unmatched.append(r)
    if unmatched:
        print(f"note: {len(unmatched)} keeper name(s) had no match in the shoot folder, "
              f"e.g. {unmatched[:3]}")
    return wanted


def accuracy(folder: Path, kept_file: Path | None, kept_folder: Path | None, preset: str):
    by_name = load(folder)
    kept = read_kept(by_name, kept_file, kept_folder)
    if not kept:
        sys.exit("no keeper filenames matched the shoot folder (checked by stem, "
                 "case-insensitive) - are these the same shoot?")
    # Complete ordering: gate and dedup off, so every frame has a rank.
    order, _ = order_of(by_name, list(by_name), preset, blur=0.0, dupes=0)
    pos = {n: i for i, n in enumerate(order)}
    n, g = len(order), len(kept)

    # Separately: what would a REAL cull (gate + dedup on) have thrown away?
    live = engine.rank_metrics([by_name[x] for x in by_name], top_n=len(by_name),
                               preset=preset)
    survived = {k["filename"] for k in live["keepers"]}
    binned = sorted(kept - survived)
    if binned:
        print(f"\n*** {len(binned)} of your keepers were DISCARDED outright "
              f"(blur gate or duplicate), not merely ranked low:")
        for f in binned[:10]:
            print(f"      {f}")
        print("    This is the worst failure mode: the photographer cannot rescue")
        print("    what they never see. Fix this before tuning any weights.")
    else:
        print("\nno keeper was discarded outright by the blur gate or dedup")
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



# ── 4. tune ────────────────────────────────────────────────────────────────
def tune(folder: Path, kept_file: Path | None, kept_folder: Path | None, preset: str):
    """Sweep the default-off signals against ground truth and report the effect.

    This is the step that decides whether eye sharpness and ball proximity are
    switched on in production. Nothing is adopted because it sounds right; it
    is adopted if recall against the photographer's own picks goes up.
    """
    by_name = load(folder)
    kept = read_kept(by_name, kept_file, kept_folder)
    if not kept:
        sys.exit("no keeper filenames matched the shoot folder")
    g = len(kept)

    def recall_at(k):
        order, _ = order_of(by_name, list(by_name), preset, blur=0.0, dupes=0)
        return sum(1 for f in order[:k] if f in kept) / g

    print(f"\nground truth: {g} keepers of {len(by_name)} frames")
    print(f"metric: recall@{g}  (share of YOUR picks the engine puts in its top {g})\n")
    base_eye, base_ball = engine.EYE_WEIGHT, engine.BALL_BONUS
    try:
        engine.EYE_WEIGHT, engine.BALL_BONUS = 0.0, 0.0
        base = recall_at(g)
        print(f"{'eye_weight':>11}{'ball_bonus':>12}{'recall':>9}{'change':>9}")
        print(f"{0.0:>11.2f}{0.0:>12.1f}{base:>9.3f}{'baseline':>9}")
        best = (base, 0.0, 0.0)
        for ew in (0.0, 0.2, 0.35, 0.5, 0.65):
            for bb in (0.0, 2.0, 5.0):
                if ew == 0.0 and bb == 0.0:
                    continue
                engine.EYE_WEIGHT, engine.BALL_BONUS = ew, bb
                r = recall_at(g)
                mark = "  <-- best" if r > best[0] else ""
                print(f"{ew:>11.2f}{bb:>12.1f}{r:>9.3f}{r-base:>+9.3f}{mark}")
                if r > best[0]:
                    best = (r, ew, bb)
    finally:
        engine.EYE_WEIGHT, engine.BALL_BONUS = base_eye, base_ball
    r, ew, bb = best
    print()
    if r > base:
        print(f"BEST: eye_weight={ew}, ball_bonus={bb}  ->  recall {base:.3f} to {r:.3f} "
              f"(+{100*(r-base):.1f} points)")
        print(f"ship with:  CLUTCHCULL_EYE_WEIGHT={ew}  CLUTCHCULL_BALL_BONUS={bb}")
    else:
        print("No setting beat the baseline on this shoot. Leave both at 0 and")
        print("keep looking: a signal that does not improve recall is not a feature.")
    print("\nOne shoot is one data point. Confirm on a second before shipping.")



# ── 5. pairs ───────────────────────────────────────────────────────────────
def _bradley_terry(items, comps, iters=300):
    """Fit a strength per photo from pairwise wins.

    Bradley-Terry rather than raw win counts: a frame that only ever met strong
    opponents and a frame that only met weak ones can have the same win rate
    while being very different photographs. The iterative fit accounts for who
    each frame actually played.
    """
    idx = {n: i for i, n in enumerate(items)}
    w = np.ones(len(items))
    wins = np.zeros(len(items))
    plays = np.zeros((len(items), len(items)))
    for a, b, winner in comps:
        if a not in idx or b not in idx:
            continue
        i, j = idx[a], idx[b]
        plays[i, j] += 1
        plays[j, i] += 1
        if winner == "left":
            wins[i] += 1
        elif winner == "right":
            wins[j] += 1
        else:                      # a tie is half a win each
            wins[i] += 0.5
            wins[j] += 0.5
    for _ in range(iters):
        denom = np.zeros(len(items))
        for i in range(len(items)):
            for j in range(len(items)):
                if plays[i, j]:
                    denom[i] += plays[i, j] / (w[i] + w[j])
        new = np.where(denom > 0, wins / np.maximum(denom, 1e-9), w)
        new = np.maximum(new, 1e-9)
        new /= new.mean()
        if np.allclose(new, w, rtol=1e-6):
            w = new
            break
        w = new
    return {n: float(w[idx[n]]) for n in items}


def pairs(folder: Path, comps_file: Path, preset: str):
    by_name = load(folder)
    stems = {Path(n).stem.lower(): n for n in by_name}

    def resolve(x):
        return stems.get(Path(x).stem.lower())

    comps = []
    with open(comps_file, newline="") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) < 3:
                continue
            a, b, winner = resolve(row[0]), resolve(row[1]), row[2].strip()
            if a and b and a != b:
                comps.append((a, b, winner))
    if not comps:
        sys.exit("no comparisons matched photos in that folder")

    items = sorted({x for c in comps for x in c[:2]})
    strength = _bradley_terry(items, comps)
    human = [n for n, _ in sorted(strength.items(), key=lambda kv: -kv[1])]
    order, _ = order_of(by_name, list(by_name), preset, blur=0.0, dupes=0)
    engine_rank = {n: i for i, n in enumerate(order)}
    human_rank = {n: i for i, n in enumerate(human)}
    shared = [n for n in human if n in engine_rank]

    conc = disc = 0
    for a, b in itertools.combinations(shared, 2):
        same = (human_rank[a] < human_rank[b]) == (engine_rank[a] < engine_rank[b])
        conc += same
        disc += not same
    tau = (conc - disc) / max(conc + disc, 1)

    print(f"\n{len(comps)} comparisons over {len(items)} frames\n")
    print(f"agreement with your ranking (Kendall tau): {tau:+.3f}")
    print("   +1.00 = identical order, 0.00 = no better than chance, -1.00 = reversed")
    half = max(1, len(shared) // 2)
    overlap = len(set(human[:half]) & set(order[:half]))
    print(f"top half overlap: {overlap}/{half} of your better half is in the engine's better half")

    print("\nbiggest disagreements (your rank vs engine rank):")
    gaps = sorted(shared, key=lambda n: -abs(human_rank[n] - engine_rank[n]))
    for n in gaps[:8]:
        hr, er = human_rank[n] + 1, engine_rank[n] + 1
        who = "you rank it HIGHER" if hr < er else "engine ranks it higher"
        print(f"  {n[:26]:<28} you #{hr:<3} engine #{er:<3}  {who}")
    print("\nThose rows are the engine's actual failures. Everything else is noise.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["accuracy", "stability", "ablation", "tune", "pairs"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--kept", type=Path, help="text file, one kept filename per line")
    ap.add_argument("--kept-folder", type=Path, help="folder containing your keepers "
                    "(easiest: point this at your exported/edited folder)")
    ap.add_argument("--comparisons", type=Path, help="comparisons.csv from rank_pairs.html")
    ap.add_argument("--preset", default="Balanced", choices=list(engine.SCORING_PRESETS))
    a = ap.parse_args()
    if not a.folder.is_dir():
        sys.exit(f"not a folder: {a.folder}")
    if a.mode == "accuracy":
        if not a.kept and not a.kept_folder:
            sys.exit("accuracy mode needs --kept-folder (your keepers) or --kept (a filename list)")
        if a.kept and not a.kept.exists():
            sys.exit(f"no such file: {a.kept}")
        if a.kept_folder and not a.kept_folder.is_dir():
            sys.exit(f"not a folder: {a.kept_folder}")
        accuracy(a.folder, a.kept, a.kept_folder, a.preset)
    elif a.mode == "tune":
        if not a.kept and not a.kept_folder:
            sys.exit("tune mode needs --kept-folder or --kept")
        tune(a.folder, a.kept, a.kept_folder, a.preset)
    elif a.mode == "pairs":
        if not a.comparisons or not a.comparisons.exists():
            sys.exit("pairs mode needs --comparisons comparisons.csv from tools/rank_pairs.html")
        pairs(a.folder, a.comparisons, a.preset)
    elif a.mode == "stability":
        stability(a.folder, a.preset)
    else:
        ablation(a.folder, a.preset)


if __name__ == "__main__":
    main()
