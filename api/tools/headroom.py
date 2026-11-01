"""How high can agreement possibly go, and is a non-linear model worth it?

Two questions that decide whether 75% is a target or a fantasy.

1. THE CEILING. Pairwise labels are noisy: shown the same two frames twice, a
   photographer does not always answer the same way. If his answers are a noisy
   reading of a fixed underlying preference, then even a PERFECT model cannot
   agree with the observed label more often than he agrees with himself in the
   relevant sense. Worth computing exactly, because the intuition ("he is 85%
   consistent so the cap is 85%") is wrong and far too pessimistic.

   Model it as: there is a true preference, and he reports it correctly with
   probability p. Shown a pair twice he answers the same way when he is right
   both times or wrong both times, so observed consistency is

       c = p^2 + (1 - p)^2

   A perfect model always states the true preference, so it agrees with a single
   observed answer at rate p. Solving for p:

       p = (1 + sqrt(2c - 1)) / 2

2. IS THE MODEL SHAPE THE LIMIT? The shipped ranker is linear in four features,
   which forces every feature to be monotonic: more is always better or always
   worse. That is almost certainly wrong for at least two of them. A subject can
   be too small (a distant speck) AND too large (a tight crop with limbs cut
   off), and a frame can be too dark AND blown out. Both have an interior
   optimum that a straight line cannot express.

   Tested by adding squared terms and interactions, still leave-one-shoot-out.

    python tools/headroom.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from cross_shoot import (FEATURE_SETS, SHOOTS, accuracy, build_xy, fit,
                         read_comps, zscore)
from eval_cull import _bradley_terry, load  # noqa: E402

BASE = FEATURE_SETS["bright"]


def ceiling(c: float) -> float:
    """Best possible agreement with a single noisy label, given consistency c.

    c below 0.5 means answers are anti-correlated with themselves, which is not
    a noise model, it is a broken experiment; clamped rather than returning a
    complex number.
    """
    c = max(0.5, min(1.0, c))
    return (1.0 + float(np.sqrt(max(0.0, 2.0 * c - 1.0)))) / 2.0


def main() -> None:
    print(f"\n{'=' * 74}")
    print("1. THE CEILING: what a PERFECT ranker could score")
    print(f"{'=' * 74}\n")
    print("Read this as: if you agree with yourself at the left rate, no model")
    print("can beat the right rate, because the label itself is noisy.\n")
    print(f"{'your self-consistency':<24}{'max possible agreement':>24}")
    for c in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
        print(f"{100 * c:>20.0f}%{100 * ceiling(c):>23.0f}%")
    print("\nThe cap is much higher than the consistency, because a model is")
    print("allowed to be right when you slipped. So 75% stays reachable even")
    print("if you only agree with yourself 80% of the time. That is the")
    print("reassuring part: labels are probably NOT the binding constraint.")

    data = []
    for name, folder, cf in SHOOTS:
        if not folder.is_dir() or not cf.exists():
            continue
        by = load(folder)
        comps = read_comps(cf, by)
        items = sorted({x for c in comps for x in c[:2]})
        data.append({"name": name, "by": by, "comps": comps, "items": items,
                     "strength": _bradley_terry(items, comps)})

    print(f"\n{'=' * 74}")
    print("2. IS A STRAIGHT LINE THE LIMIT?  (leave-one-shoot-out)")
    print(f"{'=' * 74}\n")

    def run(label: str, build) -> tuple[float, float]:
        c = []
        for s in data:
            z = zscore(s["by"], list(s["by"]), BASE)
            c.append({**s, "z": {n: build(v) for n, v in z.items()}})
        accs = []
        for i in range(len(c)):
            Xtr, ytr = build_xy([d for j, d in enumerate(c) if j != i])
            Xte, yte = build_xy([c[i]])
            # Tune L2 on training shoots only. Expanded feature spaces need
            # more regularisation, so holding lambda fixed would hand the
            # linear model an unfair advantage.
            best, bs = 0.03, -1.0
            for lam in (0.01, 0.03, 0.1, 0.3, 1.0):
                sc = []
                for k in range(len(c)):
                    if k == i:
                        continue
                    inner = [d for m, d in enumerate(c) if m not in (i, k)]
                    Xi, yi = build_xy(inner)
                    Xv, yv = build_xy([c[k]])
                    sc.append(accuracy(fit(Xi, yi, lam), Xv, yv))
                if np.mean(sc) > bs:
                    best, bs = lam, float(np.mean(sc))
            accs.append(accuracy(fit(Xtr, ytr, best), Xte, yte))
        print(f"{label:<30}{len(build(np.zeros(len(BASE)))):>4}"
              + "".join(f"{v:>11.1f}%" for v in accs)
              + f"{min(accs):>9.1f}%{np.mean(accs):>8.1f}%")
        return min(accs), float(np.mean(accs))

    print(f"{'model':<30}{'n':>4}" + "".join(f"{s['name'][:10]:>12}" for s in data)
          + f"{'worst':>9}{'mean':>8}")

    run("linear (shipped)", lambda v: v)
    # Squared terms give each feature an interior optimum. The difference of
    # squares is what the pairwise model sees, which is exactly "how far from
    # ideal is A versus B".
    run("+ squared terms", lambda v: np.concatenate([v, v ** 2]))
    idx = list(itertools.combinations(range(len(BASE)), 2))
    run("+ interactions", lambda v: np.concatenate(
        [v, np.array([v[i] * v[j] for i, j in idx])]))
    run("+ squared + interactions", lambda v: np.concatenate(
        [v, v ** 2, np.array([v[i] * v[j] for i, j in idx])]))

    print("\nIf squared terms do not help, the features are not merely being")
    print("read monotonically wrong, they genuinely do not contain the missing")
    print("information, and the next gain has to come from new features or")
    print("more shoots rather than from a cleverer model.")


if __name__ == "__main__":
    main()
