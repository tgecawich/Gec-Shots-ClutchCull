"""Is the pose model actually working on these photos?

Needed before concluding anything from the negative result. "Pose features do
not predict his preference" and "pose detection failed on most frames" produce
exactly the same table, and only the first is a finding. If presence is near
zero on most frames then the features were noise and pose has not been tested
at all.

    python tools/check_pose_quality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from pose_features import CACHE, POSE_FEATURES  # noqa: E402


def main() -> None:
    files = sorted(CACHE.glob("*.json"))
    if not files:
        sys.exit(f"no cache in {CACHE}; run pose_features.py --extract first")

    print(f"\n{'=' * 76}")
    print("POSE DETECTION QUALITY  (did the model see a body at all?)")
    print(f"{'=' * 76}\n")
    for f in files:
        rows = json.loads(f.read_text())
        pres = np.array([r.get("pose_presence", 0.0) for r in rows.values()])
        visi = np.array([r.get("body_visibility", 0.0) for r in rows.values()])
        blank = sum(1 for r in rows.values()
                    if all(abs(r.get(k, 0.0)) < 1e-9 for k in POSE_FEATURES))
        print(f"{f.stem}  ({len(rows)} frames)")
        print(f"   all-zero (no crop / no landmarks) : {blank}  "
              f"({100 * blank / len(rows):.0f}%)")
        print(f"   presence       median {np.median(pres):.2f}   "
              f">0.9 on {100 * (pres > 0.9).mean():.0f}% of frames")
        print(f"   body visibility median {np.median(visi):.2f}   "
              f">0.5 on {100 * (visi > 0.5).mean():.0f}% of frames")

    print(f"\n{'=' * 76}")
    print("SPREAD  (a feature with no variance cannot predict anything)")
    print(f"{'=' * 76}\n")
    print(f"{'feature':<20}{'median':>10}{'IQR':>12}{'coef of var':>14}")
    allrows = []
    for f in files:
        allrows += list(json.loads(f.read_text()).values())
    for k in POSE_FEATURES:
        v = np.array([r.get(k, 0.0) for r in allrows])
        q1, q3 = np.percentile(v, [25, 75])
        cv = v.std() / max(abs(v.mean()), 1e-9)
        print(f"{k:<20}{np.median(v):>10.2f}{q3 - q1:>12.2f}{cv:>14.2f}")
    print("\nIf presence is high and these features have real spread, then the")
    print("pose model worked and the honest conclusion is that body position")
    print("does not predict THIS photographer's preference among frames that")
    print("already passed the sharpness gate.")


if __name__ == "__main__":
    main()
