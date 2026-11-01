"""Extract body-pose features for judged shoots, and test whether they help.

The question this answers, and nothing more: the learned ranker plateaus around
59-62% on unseen shoots and is weakest exactly where it matters most, the top
25% of a cull (26-48% precision). Every feature it has is a low-level image
statistic. None of them can tell a player at full stride from the same player
jogging, because both are equally sharp, equally separated and equally well lit.
"Peak action" is a statement about a body, so it needs a model of a body.

This is deliberately a measurement, not a feature. Pose costs about 48MB of RAM
on a box that peaks at 423MB of 512MB, so it is only worth engineering around if
it actually moves held-out accuracy. If it does not, the right outcome is to
delete it and say so.

    python tools/pose_features.py --extract       # writes a cache per shoot
    python tools/pose_features.py --evaluate      # does it help?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import engine  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / \
    "pose_estimation_mediapipe_2023mar.onnx"
CACHE = Path(__file__).resolve().parent / "_pose_cache"
SIZE = 256

# BlazePose landmark indices. The model emits 39 points; 0-32 are the standard
# set and the rest are auxiliary, which we ignore.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

POSE_FEATURES = ["pose_presence", "limb_extension", "limb_asymmetry",
                 "arm_elevation", "knee_lift", "stride_spread",
                 "body_visibility", "pose_extent"]


def _blob(rgb: np.ndarray) -> np.ndarray:
    """NHWC float input in 0..1.

    NOT cv2.dnn.blobFromImage: that emits NCHW, and this is a TensorFlow export
    that wants NHWC. Feeding it NCHW throws a channel-count error at 256 only
    because 256 is not a multiple of 3; at a size that happened to divide by 3
    it would have run and silently returned nonsense.
    """
    img = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img.astype(np.float32)[None, ...] / 255.0)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-max(-50.0, min(50.0, x)))))


def extract_one(net, rgb: np.ndarray, box) -> dict:
    """Pose features for one frame's subject crop.

    Everything is normalised by TORSO LENGTH rather than by crop size. A player
    close to the camera and the same player across the pitch should produce the
    same "arm fully extended" reading, and torso length is the only scale in the
    frame that belongs to the body rather than to the lens.
    """
    h, w = rgb.shape[:2]
    x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
    x1, y1 = int(min(w, box[2])), int(min(h, box[3]))
    blank = {f: 0.0 for f in POSE_FEATURES}
    if x1 - x0 < 16 or y1 - y0 < 16:
        return blank
    crop = rgb[y0:y1, x0:x1]
    if crop.size < 256:
        return blank

    net.setInput(_blob(crop))
    names = net.getUnconnectedOutLayersNames()
    outs = net.forward(names)
    # Pick heads by SHAPE, not by name: these are "Identity", "Identity_1" ...
    # whatever the TF export emitted, and the zoo has renamed them between
    # revisions. The landmark head is the only 195-value tensor (39 points x 5)
    # and presence is the only single scalar.
    lm = next((o for o in outs if o.size == 195), None)
    pres = next((o for o in outs if o.size == 1), None)
    if lm is None:
        return blank
    pts = lm.reshape(39, 5)
    xy = pts[:, :2].astype(float)          # pixels within the 256 crop
    # Measured with tools/probe_ranges.py, because guessing this is silent:
    # column 3 is a LOGIT (range -3.3..+20.4, only 9% of values inside 0..1) so
    # it needs a sigmoid, while the presence HEAD is already a probability
    # (100% inside 0..1). Sigmoiding the latter squashed it into 0.5..0.73 and
    # threw away most of its spread, which is the bug this comment replaces.
    vis = np.array([_sigmoid(v) for v in pts[:, 3]])
    presence = float(np.clip(pres.flatten()[0], 0.0, 1.0)) if pres is not None else 0.0

    hip = (xy[L_HIP] + xy[R_HIP]) / 2.0
    sho = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2.0
    torso = float(np.linalg.norm(sho - hip))
    if torso < 1e-3:
        return blank

    def d(i) -> float:
        return float(np.linalg.norm(xy[i] - hip)) / torso

    # How far the limbs are thrown out from the body's centre. A sprint, a
    # swing and a dive all score high; standing still scores low.
    limbs = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]
    limb_extension = float(np.mean([d(i) for i in limbs]))
    # Symmetric poses are usually static. Mid-stride and mid-throw are not.
    limb_asymmetry = float(
        abs(d(L_WRIST) - d(R_WRIST)) + abs(d(L_ANKLE) - d(R_ANKLE))) / 2.0
    # Image y grows downward, so "above" is a NEGATIVE difference; the sign is
    # flipped here so a raised arm reads positive.
    arm_elevation = float(
        ((sho[1] - xy[L_WRIST][1]) + (sho[1] - xy[R_WRIST][1])) / 2.0) / torso
    knee_lift = float(
        ((hip[1] - xy[L_KNEE][1]) + (hip[1] - xy[R_KNEE][1])) / 2.0) / torso
    stride_spread = float(abs(xy[L_ANKLE][0] - xy[R_ANKLE][0])) / torso
    # Is this actually a clean look at a whole body, or a guess through a
    # crowd? Occlusion is the main reason a pose reading should be distrusted.
    body_visibility = float(np.mean(vis[[NOSE, L_SHOULDER, R_SHOULDER, L_HIP,
                                         R_HIP, L_KNEE, R_KNEE]]))
    pose_extent = float(xy[:33, 1].max() - xy[:33, 1].min()) / torso

    return {
        "pose_presence": presence,
        "limb_extension": limb_extension,
        "limb_asymmetry": limb_asymmetry,
        "arm_elevation": arm_elevation,
        "knee_lift": knee_lift,
        "stride_spread": stride_spread,
        "body_visibility": body_visibility,
        "pose_extent": pose_extent,
    }


def extract_shoot(name: str, folder: Path) -> dict:
    CACHE.mkdir(exist_ok=True)
    out_path = CACHE / f"{name}.json"
    if out_path.exists():
        print(f"  {name}: cached ({out_path})")
        return json.loads(out_path.read_text())
    net = cv2.dnn.readNet(str(MODEL))
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    out, skipped = {}, 0
    for i, p in enumerate(paths):
        prev = engine._load_metrics_image(p)
        if prev is None:
            skipped += 1
            continue
        rgb = np.array(prev)
        hh, ww = rgb.shape[:2]
        faces = engine.detect_faces_scaled(rgb)
        objs = engine.detect_objects(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        box, _, _ = engine._pick_subject(objs.get(0, []), faces, ww, hh)
        out[p.name] = extract_one(net, rgb, box)
        if (i + 1) % 25 == 0:
            print(f"  {name}: {i + 1}/{len(paths)}")
    out_path.write_text(json.dumps(out))
    print(f"  {name}: wrote {len(out)} frames ({skipped} unreadable)")
    return out


def main() -> None:
    from cross_shoot import (FEATURE_SETS, SHOOTS, accuracy, build_xy, fit,
                             read_comps, zscore)
    from eval_cull import _bradley_terry, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if not (a.extract or a.evaluate):
        a.extract = a.evaluate = True

    pose = {}
    if a.extract:
        print("extracting pose features ...")
        for name, folder, _ in SHOOTS:
            if folder.is_dir():
                pose[name] = extract_shoot(name, folder)
    if not a.evaluate:
        return

    data = []
    for name, folder, cf in SHOOTS:
        if not folder.is_dir():
            continue
        pf = pose.get(name) or json.loads((CACHE / f"{name}.json").read_text())
        by = load(folder)
        # Merge pose onto the existing metrics so zscore() can see them as
        # ordinary features and nothing downstream needs to know the difference.
        for fn, m in by.items():
            m.update(pf.get(fn, {f: 0.0 for f in POSE_FEATURES}))
        comps = read_comps(cf, by)
        items = sorted({x for c in comps for x in c[:2]})
        data.append({"name": name, "by": by, "comps": comps,
                     "strength": _bradley_terry(items, comps), "items": items})

    print(f"\n{'=' * 74}")
    print("DO POSE FEATURES PREDICT THE PHOTOGRAPHER'S PREFERENCE?")
    print(f"{'=' * 74}\n")
    print(f"{'feature':<20}" + "".join(f"{s['name'][:12]:>15}" for s in data))
    for f in POSE_FEATURES:
        row = f"{f:<20}"
        for s in data:
            y = np.array([np.log(s["strength"][n]) for n in s["items"]])
            x = np.array([float(s["by"][n].get(f, 0.0) or 0.0) for n in s["items"]])
            r = 0.0 if x.std() < 1e-9 else float(np.corrcoef(x, y)[0, 1])
            row += f"{r:>+15.3f}"
        print(row)
    print("\nA feature that only correlates on ONE shoot is that shoot's")
    print("lighting or that team's kit, not peak action.")

    print(f"\n{'=' * 74}")
    print("LEAVE-ONE-SHOOT-OUT: does adding pose beat the shipped 4 features?")
    print(f"{'=' * 74}\n")
    base = FEATURE_SETS["bright"]
    trials = {
        "shipped (4)": base,
        "+ all pose": base + POSE_FEATURES,
        "+ action only": base + ["limb_extension", "limb_asymmetry", "knee_lift"],
        "+ visibility": base + ["body_visibility", "pose_presence"],
        "pose alone": POSE_FEATURES,
    }
    print(f"{'features':<16}{'n':>4}" + "".join(f"{s['name'][:12]:>14}" for s in data)
          + f"{'worst':>8}{'mean':>8}")
    for label, fs in trials.items():
        c = [{**s, "z": zscore(s["by"], list(s["by"]), fs)} for s in data]
        accs = []
        for i in range(len(c)):
            Xtr, ytr = build_xy([d for j, d in enumerate(c) if j != i])
            Xte, yte = build_xy([c[i]])
            accs.append(accuracy(fit(Xtr, ytr, 0.03), Xte, yte))
        print(f"{label:<16}{len(fs):>4}" + "".join(f"{v:>13.1f}%" for v in accs)
              + f"{min(accs):>7.1f}%{np.mean(accs):>7.1f}%")
    print("\nPose is worth its 48MB only if it beats 'shipped' on the WORST")
    print("fold. Beating it on the mean while losing a fold means it helps one")
    print("sport and hurts another, which is not an improvement to ship.")


if __name__ == "__main__":
    main()
