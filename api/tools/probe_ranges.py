"""Are the pose model's confidence columns logits or probabilities?

Guessing this wrong is silent. Applying a sigmoid to a value already in 0..1
squashes it into 0.5..0.73 and strips most of its discriminative power, while
NOT applying one to a logit leaves an unbounded number mixed in with bounded
ones. Either way the feature still looks plausible in a table.

    python tools/probe_ranges.py ~/Desktop/JUDGE-soccer
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import engine  # noqa: E402
from pose_features import MODEL, _blob  # noqa: E402


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1
                  else Path.home() / "Desktop/JUDGE-soccer").expanduser()
    net = cv2.dnn.readNet(str(MODEL))
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)[:12]
    pres, vis_c, pres_c = [], [], []
    for p in paths:
        prev = engine._load_metrics_image(p)
        if prev is None:
            continue
        rgb = np.array(prev)
        h, w = rgb.shape[:2]
        faces = engine.detect_faces_scaled(rgb)
        objs = engine.detect_objects(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        box, _, _ = engine._pick_subject(objs.get(0, []), faces, w, h)
        x0, y0 = int(max(0, box[0])), int(max(0, box[1]))
        x1, y1 = int(min(w, box[2])), int(min(h, box[3]))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        net.setInput(_blob(rgb[y0:y1, x0:x1]))
        names = net.getUnconnectedOutLayersNames()
        outs = net.forward(names)
        lm = next((o for o in outs if o.size == 195), None)
        ps = next((o for o in outs if o.size == 1), None)
        if lm is None:
            continue
        pts = lm.reshape(39, 5)
        pres.append(float(ps.flatten()[0]))
        vis_c.append(pts[:33, 3])
        pres_c.append(pts[:33, 4])

    def report(label: str, arr: np.ndarray) -> None:
        inside = float(((arr >= 0.0) & (arr <= 1.0)).mean())
        verdict = ("already a PROBABILITY (no sigmoid)" if inside > 0.98
                   else "a LOGIT (apply sigmoid)")
        print(f"{label:<28} min {arr.min():+8.2f}  max {arr.max():+8.2f}  "
              f"in[0,1] {100 * inside:5.1f}%   -> {verdict}")

    print(f"\n{len(pres)} frames from {folder.name}\n")
    report("presence head (scalar)", np.array(pres))
    report("landmark col 3 (visibility)", np.concatenate(vis_c))
    report("landmark col 4 (presence)", np.concatenate(pres_c))
    print("\nAlso: landmark x,y should be in 0..256 (crop pixels).")
    print("If they are far outside that, the coordinate convention is wrong")
    print("and every normalised feature built on them is wrong too.")


if __name__ == "__main__":
    main()
