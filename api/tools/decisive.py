"""Is the engine wrong, or is it wrong about pairs that do not matter?

A single pairwise-agreement percentage treats two very different questions as
one. "Is this great frame better than this unusable one" and "is frame 70
better than frame 71" both count once, but only the first has a right answer a
photographer would defend. The comparisons were sampled at random, so most of
them are the second kind, and they drag the headline number toward 50% no
matter how good the ranker is.

That is not a reason to ignore the headline number. It is a reason to break it
down by how decisive the photographer's own preference was, measured by the gap
in fitted Bradley-Terry strength between the two frames. If agreement is high
where the gap is large and chance-level where the gap is small, the engine is
working and the aggregate is diluted. If agreement is flat across the range,
the engine really is guessing and there is no excuse available.

The self-consistency check at the end is the honest ceiling: among the pairs
the photographer effectively judged twice (same two frames, or near-identical
strength), how often did he agree with himself?

Usage:
    python tools/decisive.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from cross_shoot import (FEATURE_SETS, SHOOTS, build_xy, fit, read_comps,  # noqa: E402
                         zscore)
from eval_cull import _bradley_terry, load, order_of  # noqa: E402

FEATS = FEATURE_SETS["core"]
LAM = 0.03


def main() -> None:
    data = []
    for name, folder, comps_file in SHOOTS:
        if not folder.is_dir() or not comps_file.exists():
            continue
        print(f"[{name}]")
        by = load(folder)
        comps = read_comps(comps_file, by)
        if len(comps) < 50:
            continue
        items = sorted({x for c in comps for x in c[:2]})
        order, _ = order_of(by, list(by), "Balanced", blur=0.0, dupes=0)
        data.append({
            "name": name, "by": by, "comps": comps, "items": items,
            "strength": _bradley_terry(items, comps),
            "rank": {n: i for i, n in enumerate(order)},
            "z": zscore(by, list(by), FEATS),
        })

    # Leave-one-shoot-out predictions, so nothing below is fitted on its own
    # test shoot.
    for i, test in enumerate(data):
        Xtr, ytr = build_xy([d for j, d in enumerate(data) if j != i])
        test["w"] = fit(Xtr, ytr, LAM)

    print(f"\n{'=' * 74}")
    print("AGREEMENT BY HOW DECISIVE THE PHOTOGRAPHER'S PREFERENCE WAS")
    print(f"{'=' * 74}")
    print("Gap = difference in fitted strength between the two frames, in")
    print("quartiles of the pairs actually judged. Q1 is near-ties, Q4 is")
    print("'obviously this one'. Learned is leave-one-shoot-out.\n")

    rows = []
    for s in data:
        for a, b, w in s["comps"]:
            if w == "tie" or a not in s["rank"] or b not in s["rank"]:
                continue
            la, lb = np.log(s["strength"][a]), np.log(s["strength"][b])
            human_left = w == "left"
            eng_left = s["rank"][a] < s["rank"][b]
            learn_left = float((s["z"][a] - s["z"][b]) @ s["w"]) > 0
            rows.append((abs(la - lb), human_left == eng_left,
                         human_left == learn_left))
    gaps = np.array([r[0] for r in rows])
    eng_ok = np.array([r[1] for r in rows], dtype=float)
    lrn_ok = np.array([r[2] for r in rows], dtype=float)
    qs = np.percentile(gaps, [25, 50, 75])
    bins = np.digitize(gaps, qs)
    print(f"{'pairs':<10}{'n':>6}{'gap':>14}{'ships':>10}{'learned':>10}")
    labels = ["Q1 near-ties", "Q2", "Q3", "Q4 decisive"]
    for q in range(4):
        m = bins == q
        if not m.any():
            continue
        lo, hi = gaps[m].min(), gaps[m].max()
        print(f"{labels[q]:<10}{int(m.sum()):>6}{f'{lo:.2f}-{hi:.2f}':>14}"
              f"{100 * eng_ok[m].mean():>9.1f}%{100 * lrn_ok[m].mean():>9.1f}%")
    print(f"{'ALL':<10}{len(rows):>6}{'':>14}{100 * eng_ok.mean():>9.1f}%"
          f"{100 * lrn_ok.mean():>9.1f}%")

    # Top-half agreement is the production question: of the frames it would
    # hand back, how many would the photographer have chosen?
    print(f"\n{'=' * 74}")
    print("WOULD YOU HAVE KEPT WHAT IT HANDS BACK?  (precision at k)")
    print(f"{'=' * 74}")
    print("Ground truth = your own top k by fitted strength. Random is the")
    print("baseline you beat by doing nothing, and it rises with k.\n")
    print(f"{'shoot':<20}{'k':>5}{'random':>9}{'ships':>9}{'learned':>10}")
    for s in data:
        judged = [n for n in s["items"] if n in s["rank"]]
        order_h = sorted(judged, key=lambda n: -s["strength"][n])
        sc = {n: float(s["z"][n] @ s["w"]) for n in judged}
        order_l = sorted(judged, key=lambda n: -sc[n])
        order_e = sorted(judged, key=lambda n: s["rank"][n])
        for frac in (0.25, 0.5):
            k = max(1, int(len(judged) * frac))
            truth = set(order_h[:k])
            pe = 100 * len(truth & set(order_e[:k])) / k
            pl = 100 * len(truth & set(order_l[:k])) / k
            print(f"{s['name'][:19]:<20}{k:>5}{100 * k / len(judged):>8.0f}%"
                  f"{pe:>8.0f}%{pl:>9.0f}%")

    print(f"\n{'=' * 74}")
    print("THE CEILING: how often you agree with YOURSELF")
    print(f"{'=' * 74}")
    for s in data:
        seen: dict[tuple[str, str], list[str]] = {}
        for a, b, w in s["comps"]:
            if w == "tie":
                continue
            key = tuple(sorted((a, b)))
            # Normalise the answer to "did the alphabetically-first frame win",
            # so the same verdict recorded with the photos swapped still counts
            # as consistent rather than as a contradiction.
            winner = a if w == "left" else b
            seen.setdefault(key, []).append(winner)
        rep = {k: v for k, v in seen.items() if len(v) > 1}
        if not rep:
            print(f"  {s['name']:<20}no pair was shown twice")
            continue
        same = sum(1 for v in rep.values() if len(set(v)) == 1)
        print(f"  {s['name']:<20}{same}/{len(rep)} repeated pairs answered the "
              f"same way ({100 * same / len(rep):.0f}%)")
    print("\nNo ranker can beat this number. If it is 95%+, the limit is the")
    print("features, not the labels.")


if __name__ == "__main__":
    main()
