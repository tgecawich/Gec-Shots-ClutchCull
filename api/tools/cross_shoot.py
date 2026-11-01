"""Fit a ranker on some shoots, test it on a shoot it has never seen.

Why this exists, and why `eval_cull.py learn` is not enough: that mode picks the
best of four hand-written weightings and validates by splitting ONE shoot in
half. Both halves are the same game, the same light, the same lens, and the same
twenty minutes of a photographer's taste. A config that wins there has only been
shown to generalise from one half of an afternoon to the other half.

The number that matters to a user is different: they upload a shoot this engine
has never seen. So the only honest test is leave-one-SHOOT-out. Train on the
others, predict the held-out one, never let a frame from the test shoot touch
the fit.

The model is pairwise logistic regression:

    P(a beats b) = sigmoid(w . (z_a - z_b))

Three deliberate choices in that line:

  * Trained on DIFFERENCES, not on absolute scores. The labels are comparisons,
    so differences are what was actually observed. Fitting absolute quality
    would require inventing a target that was never measured.
  * No intercept. "a beats b" and "b beats a" must be mirror images, and any
    constant term would break that symmetry and encode a left/right bias.
  * Features z-scored WITHIN each shoot. Raw Laplacian variance depends on
    light, lens and ISO, so a weight fitted on a bright afternoon game would be
    meaningless at dusk. Per-shoot standardisation is also what the engine
    already does with its percentile normalisation, so this matches production.

Usage:
    python tools/cross_shoot.py
    python tools/cross_shoot.py --features core
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

import engine  # noqa: E402
from eval_cull import _bradley_terry, load, order_of  # noqa: E402

HOME = Path.home()

# Every shoot judged so far. Each is one photographer on one sport on one day,
# which is exactly why no single one of them can validate anything.
SHOOTS = [
    ("baseball-raw", HOME / "Desktop/JUDGE-baseball",
     HOME / "Downloads/comparisons.csv"),
    ("baseball-keepers", HOME / "Downloads/FOR csv photos machine learning",
     HOME / "Downloads/comparisons-baseball-keepers.csv"),
    ("soccer", HOME / "Desktop/JUDGE-soccer",
     HOME / "Downloads/comparisons-soccer.csv"),
    ("baseball-0415", HOME / "Desktop/JUDGE-04-15",
     HOME / "Downloads/comparisons-04-15.csv"),
    ("jun26", HOME / "Desktop/JUDGE-06-26",
     HOME / "Downloads/comparisons-06-26.csv"),
]

# EXCLUDED, deliberately, and the reason is worth more than the data was.
#
#   ("mar08", "JUDGE-03-08", "comparisons-03-08.csv")   166 comparisons
#
# Indoor gymnastics, and per the photographer "very few of the photos came out
# good before serious editing". Adding it dropped the best model from 64.9% to
# 61.4% and hurt every other shoot it was trained alongside: baseball -2.9,
# soccer -1.7, keepers -0.6.
#
# That is not thin data. Trimming soccer's test comparisons to the same 166
# changed its score by 0.2 points, so sparseness is not the explanation.
#
# The explanation is that the labels answer a different question. When nearly
# every frame is marginal, choosing between them is a judgement about which one
# would SURVIVE EDITING, and that depends on recoverable highlights, croppable
# framing and noise that cleans up. The engine scores the frame as it is. Those
# labels therefore encode information no feature here can see, so fitting to
# them moves the weights away from what works on frames that are already good.
#
# The lesson for picking future shoots: judge shoots that went WELL. A shoot
# where the photographer would keep nothing teaches the ranker nothing about
# ranking keepers. This file previously chose shoots by spreading them across
# the calendar, which is good for variety in light and sport and blind to
# whether the shoot was usable at all.

# `sharpness` and `subject_sharpness` stay in despite being weak-to-negative
# predictors: the point is to let the fit decide the sign rather than to assume
# it. `separation` and `subject_area` are the two features added after the
# photographer described his own preference in words the engine could not
# measure ("blurry background, clear subject, large in frame").
FEATURE_SETS = {
    "all": ["subject_sharpness", "sharpness", "frame_tiled", "detail_ratio",
            "contrast", "brightness_mean", "exposure_balance", "face_score",
            "ball_score", "eye_sharpness", "bg_sharpness", "subject_area",
            "separation"],
    # Only what the photographer actually articulated, plus exposure. Fewer
    # parameters generalise better when there are three shoots of data.
    "core": ["separation", "subject_area", "face_score", "exposure_balance",
             "subject_sharpness"],
    # What ships today, to show the fit is not just winning on extra inputs.
    "legacy": ["subject_sharpness", "sharpness", "detail_ratio", "contrast",
               "exposure_balance", "face_score"],
    "minimal": ["separation", "subject_area", "face_score"],
    "core+ball": ["separation", "subject_area", "face_score", "exposure_balance",
                  "subject_sharpness", "ball_score"],
    "core+eyes": ["separation", "subject_area", "face_score", "exposure_balance",
                  "subject_sharpness", "eye_sharpness"],
    "core+detail": ["separation", "subject_area", "face_score", "exposure_balance",
                    "subject_sharpness", "detail_ratio"],
    "bright": ["separation", "subject_area", "face_score", "brightness_mean"],
    # Composition features, added after pose estimation failed. Pose asked "what
    # is the body doing"; these ask "is the photograph intact" - clipped by the
    # frame, blocked by another player, badly placed. All free from boxes the
    # engine already computes.
    "bright+edge": ["separation", "subject_area", "face_score", "brightness_mean",
                    "edge_contact"],
    "bright+crowd": ["separation", "subject_area", "face_score", "brightness_mean",
                     "crowding", "person_count"],
    "bright+comp": ["separation", "subject_area", "face_score", "brightness_mean",
                    "edge_contact", "crowding", "person_count", "off_center"],
    "comp alone": ["edge_contact", "crowding", "person_count", "off_center"],
}


def read_comps(path: Path, by_name: dict) -> list[tuple[str, str, str]]:
    """Comparisons whose BOTH frames exist in the scored folder.

    Matched on filename stem so a `84_D91A3192.jpg` label pairs with whatever
    the folder calls the same frame.
    """
    stems = {Path(n).stem.lower(): n for n in by_name}
    # Labels sometimes carry a rank prefix the folder does not have, or vice
    # versa. Try the stem as-is, then with a leading `<digits>_` stripped.
    def resolve(x: str):
        s = Path(x).stem.lower()
        if s in stems:
            return stems[s]
        if "_" in s:
            return stems.get(s.split("_", 1)[1])
        return None

    out = []
    with open(path, newline="") as f:
        for row in list(csv.reader(f))[1:]:
            if len(row) < 3:
                continue
            a, b, w = resolve(row[0]), resolve(row[1]), row[2].strip()
            if a and b and a != b and w in ("left", "right", "tie"):
                out.append((a, b, w))
    return out


def zscore(by_name: dict, names: list[str], feats: list[str]) -> dict:
    """Per-shoot standardised feature vectors.

    A constant feature becomes zeros rather than NaN: within a single shoot a
    detector can fire identically on every frame, and that should contribute
    nothing instead of poisoning the whole fit.
    """
    raw = np.array([[float(by_name[n].get(f, 0.0) or 0.0) for f in feats]
                    for n in names])
    # log1p the heavy-tailed sharpness-like features first. Laplacian variance
    # spans orders of magnitude, so one tack-sharp frame otherwise drags the
    # mean past every other frame and z-scoring cannot undo that.
    #
    # Imported from the engine rather than re-listed here, deliberately. The two
    # lists were duplicated once and drifted: this one log1p'd
    # subject_sharpness while the engine's did not, so shipping any feature set
    # containing it would have mis-scaled that input with nothing reporting an
    # error. The ranking would simply have been wrong. One definition now.
    for i, f in enumerate(feats):
        if f in engine.LEARNED_LOG:
            raw[:, i] = np.log1p(np.maximum(raw[:, i], 0.0))
    mu, sd = raw.mean(0), raw.std(0)
    # A constant feature divides by 1 instead of ~0. Its numerator is already
    # ~0 (every value equals the mean), so it lands at 0 and contributes
    # nothing, which is the correct behaviour rather than a NaN.
    sd = np.where(sd < 1e-9, 1.0, sd)
    z = (raw - mu) / sd
    return {n: z[i] for i, n in enumerate(names)}


def build_xy(shoot_data: list[dict], weighted: bool = False):
    """Stack (feature difference, label) rows across shoots.

    Each comparison is emitted ONCE as (z_a - z_b, won). The model has no
    intercept, so it is antisymmetric by construction and adding the mirrored
    row would only duplicate information.

    With `weighted`, each pair is weighted by how decisive the photographer's
    preference was, taken from the fitted Bradley-Terry gap. The reasoning: a
    pair of near-identical frames is close to a coin flip even for the person
    who judged it, so its label is mostly noise, and fitting it as hard as an
    obvious call teaches the model to reproduce that noise. Weighting is applied
    to TRAINING ONLY; scoring stays unweighted, otherwise the easy pairs would
    be allowed to flatter the result.
    """
    X, y, sw = [], [], []
    for s in shoot_data:
        st = s.get("strength") or {}
        for a, b, w in s["comps"]:
            if w == "tie":
                continue
            X.append(s["z"][a] - s["z"][b])
            y.append(1.0 if w == "left" else 0.0)
            if weighted and a in st and b in st:
                sw.append(abs(float(np.log(st[a]) - np.log(st[b]))))
            else:
                sw.append(1.0)
    X, y, sw = np.array(X), np.array(y), np.array(sw)
    if weighted and len(sw):
        sw = sw / max(sw.mean(), 1e-9)   # keep the effective sample size
    return (X, y, sw) if weighted else (X, y)


def fit(X: np.ndarray, y: np.ndarray, lam: float,
        sw: np.ndarray | None = None) -> np.ndarray:
    """L2-regularised logistic fit, no intercept.

    Regularisation is not optional here. Thirteen features against a few hundred
    comparisons will happily find weights that explain the training shoots and
    mean nothing elsewhere, which is the exact failure this script exists to
    detect.
    """
    sw = np.ones(len(y)) if sw is None else sw
    tot = max(sw.sum(), 1e-9)

    def obj(w):
        t = X @ w
        # log(1+exp(-t)) written via logaddexp so a large |t| cannot overflow.
        ll = np.logaddexp(0.0, -t) * y + np.logaddexp(0.0, t) * (1.0 - y)
        p = 1.0 / (1.0 + np.exp(-np.clip(t, -500, 500)))
        g = X.T @ (sw * (p - y)) / tot + 2.0 * lam * w
        return float(sw @ ll) / tot + lam * float(w @ w), g

    r = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
    return r.x


def accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    if not len(y):
        return float("nan")
    pred = (X @ w) > 0
    return 100.0 * float(((pred == (y > 0.5)).mean()))


def engine_agreement(s: dict, preset: str = "Balanced") -> float:
    """What ships today, on this shoot's comparisons. The number to beat."""
    order, _ = order_of(s["by"], list(s["by"]), preset, blur=0.0, dupes=0)
    rank = {n: i for i, n in enumerate(order)}
    ok = tot = 0
    for a, b, w in s["comps"]:
        if w == "tie" or a not in rank or b not in rank:
            continue
        tot += 1
        ok += (rank[a] < rank[b]) == (w == "left")
    return 100.0 * ok / max(tot, 1)


def search(raw_shoots: list[tuple], lams: list[float]) -> None:
    """Honest feature-set selection: nested, so the test shoot never votes.

    Running every feature set through leave-one-shoot-out and reporting the best
    one is a trap I nearly walked into. The test shoots would then have chosen
    the configuration, and the winner's score is no longer an estimate of how it
    does on an unseen shoot: it is the maximum of several noisy numbers, which
    is biased upward by construction.

    So for each held-out shoot, the feature set is chosen using ONLY the other
    shoots, then scored once on the held-out one. That measures the procedure
    "fit this engine on the shoots you have", which is the thing a user actually
    receives.
    """
    # Score every shoot under every feature set once, then reuse.
    cache: dict[str, list[dict]] = {}
    for key, feats in FEATURE_SETS.items():
        cache[key] = [{**s, "z": zscore(s["by"], list(s["by"]), feats)}
                      for s in raw_shoots]

    print(f"\n{'=' * 72}")
    print("NESTED SELECTION  (feature set chosen WITHOUT the held-out shoot)")
    print(f"{'=' * 72}\n")
    print(f"{'held-out':<20}{'chosen set':<14}{'ships':>8}{'learned':>9}")
    rows = []
    for i, test in enumerate(raw_shoots):
        # Inner loop: rank feature sets on the training shoots only, by
        # leave-one-training-shoot-out.
        inner_score = {}
        for key in FEATURE_SETS:
            sc = []
            for k in range(len(raw_shoots)):
                if k == i:
                    continue
                tr = [d for m, d in enumerate(cache[key]) if m not in (i, k)]
                if not tr:
                    continue
                Xi, yi = build_xy(tr)
                Xv, yv = build_xy([cache[key][k]])
                if not len(yi) or not len(yv):
                    continue
                sc.append(accuracy(fit(Xi, yi, lams[len(lams) // 2]), Xv, yv))
            if sc:
                inner_score[key] = float(np.mean(sc))
        best = max(inner_score, key=inner_score.get)
        tr = [d for j, d in enumerate(cache[best]) if j != i]
        Xtr, ytr = build_xy(tr)
        Xte, yte = build_xy([cache[best][i]])
        acc = accuracy(fit(Xtr, ytr, lams[len(lams) // 2]), Xte, yte)
        eng = engine_agreement(test)
        rows.append((acc, eng))
        print(f"{test['name']:<20}{best:<14}{eng:>7.1f}%{acc:>8.1f}%")
    print(f"\n{'mean':<34}{np.mean([r[1] for r in rows]):>7.1f}%"
          f"{np.mean([r[0] for r in rows]):>8.1f}%")
    print("\nThis is the number to quote. It is lower than the best single row")
    print("of the per-feature-set table, and that difference IS the selection")
    print("bias that table hides.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="all", choices=list(FEATURE_SETS))
    ap.add_argument("--lam", type=float, default=None,
                    help="L2 strength; default sweeps and picks per fold")
    ap.add_argument("--search", action="store_true",
                    help="compare all feature sets with nested selection")
    a = ap.parse_args()
    feats = FEATURE_SETS[a.features]

    data = []
    for name, folder, comps_file in SHOOTS:
        if not folder.is_dir() or not comps_file.exists():
            print(f"  skip {name}: missing {folder if not folder.is_dir() else comps_file}")
            continue
        print(f"\n[{name}]")
        by = load(folder)
        comps = read_comps(comps_file, by)
        items = sorted({x for c in comps for x in c[:2]})
        if len(comps) < 50:
            print(f"  skip {name}: only {len(comps)} usable comparisons")
            continue
        data.append({
            "name": name, "by": by, "comps": comps, "items": items,
            "z": zscore(by, list(by), feats),
            "strength": _bradley_terry(items, comps),
        })
        print(f"  {len(comps)} comparisons over {len(items)} frames")

    if len(data) < 2:
        sys.exit("need at least two judged shoots to test generalisation")

    lams_all = [a.lam] if a.lam else [0.003, 0.01, 0.03, 0.1, 0.3]
    if a.search:
        print(f"\n{'=' * 72}")
        print("EVERY FEATURE SET, leave-one-shoot-out (selection NOT yet honest)")
        print(f"{'=' * 72}\n")
        print(f"{'feature set':<14}{'n':>4}", end="")
        for s in data:
            print(f"{s['name'][:13]:>15}", end="")
        print(f"{'mean':>9}")
        for key, fs in FEATURE_SETS.items():
            cached = [{**s, "z": zscore(s["by"], list(s["by"]), fs)} for s in data]
            accs = []
            for i in range(len(cached)):
                Xtr, ytr = build_xy([d for j, d in enumerate(cached) if j != i])
                Xte, yte = build_xy([cached[i]])
                accs.append(accuracy(fit(Xtr, ytr, lams_all[len(lams_all) // 2]),
                                     Xte, yte))
            print(f"{key:<14}{len(fs):>4}", end="")
            for v in accs:
                print(f"{v:>14.1f}%", end="")
            print(f"{np.mean(accs):>8.1f}%")
        search(data, lams_all)
        return

    print(f"\n{'=' * 72}")
    print(f"LEAVE-ONE-SHOOT-OUT  ({a.features}: {len(feats)} features)")
    print(f"{'=' * 72}")
    print("Trained on the other shoots. The test shoot never touches the fit.\n")
    print(f"{'held-out shoot':<20}{'pairs':>7}{'ships':>9}{'learned':>9}{'in-sample':>11}")

    lams = [a.lam] if a.lam else [0.003, 0.01, 0.03, 0.1, 0.3]
    held, shipped = [], []
    for i, test in enumerate(data):
        train = [d for j, d in enumerate(data) if j != i]
        Xtr, ytr = build_xy(train)
        Xte, yte = build_xy([test])
        # Pick lambda by leaving one TRAINING shoot out, never with the test
        # shoot. With two training shoots that is a single honest split.
        best_lam, best = lams[0], -1.0
        if len(lams) > 1 and len(train) > 1:
            for lam in lams:
                sc = []
                for k in range(len(train)):
                    inner = [d for m, d in enumerate(train) if m != k]
                    Xi, yi = build_xy(inner)
                    Xv, yv = build_xy([train[k]])
                    sc.append(accuracy(fit(Xi, yi, lam), Xv, yv))
                if np.mean(sc) > best:
                    best_lam, best = lam, float(np.mean(sc))
        w = fit(Xtr, ytr, best_lam)
        acc = accuracy(w, Xte, yte)
        ins = accuracy(fit(Xte, yte, best_lam), Xte, yte)
        eng = engine_agreement(test)
        held.append(acc)
        shipped.append(eng)
        print(f"{test['name']:<20}{len(yte):>7}{eng:>8.1f}%{acc:>8.1f}%{ins:>10.1f}%")

    print(f"\n{'mean':<20}{'':>7}{np.mean(shipped):>8.1f}%{np.mean(held):>8.1f}%")
    print("\n'ships' is the live engine. 'learned' is the honest number: fitted")
    print("without ever seeing this shoot. 'in-sample' is what you get by")
    print("training and testing on the same shoot, shown only to make the size")
    print("of the overfit visible.")

    # Final model on everything, for the weights themselves.
    Xa, ya = build_xy(data)
    wa = fit(Xa, ya, lams[len(lams) // 2])
    print(f"\n{'=' * 72}")
    print("WEIGHTS fitted on all shoots (z-scored units, sign is the message)")
    print(f"{'=' * 72}")
    for f, v in sorted(zip(feats, wa), key=lambda t: -abs(t[1])):
        bar = "#" * int(min(40, abs(v) * 20))
        print(f"  {f:<20}{v:>+8.3f}  {bar}")
    print("\nA NEGATIVE weight means the photographer prefers LESS of it. If a")
    print("feature the engine currently ADDS shows up negative here, it is not")
    print("merely mis-weighted, it is scored backwards.")


if __name__ == "__main__":
    main()
