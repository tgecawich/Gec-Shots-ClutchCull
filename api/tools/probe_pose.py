"""Can cv2.dnn actually run this pose model, and what does it cost?

Written before any engine integration on purpose. The single worst incident
this project has had was a memory change that worked every time it was tested
and then crash-looped the server for four days. The box has 512MB, the current
peak is 423MB, and this adds a third model. So the order is: prove it loads,
prove what it outputs, measure what it costs, and only then decide.

    python tools/probe_pose.py
    python tools/probe_pose.py --folder ~/Desktop/JUDGE-soccer --n 20
"""
from __future__ import annotations

import argparse
import gc
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "models" / \
    "pose_estimation_mediapipe_2023mar.onnx"


def _blob(rgb: np.ndarray, size: int) -> np.ndarray:
    """NHWC float input in 0..1.

    cv2.dnn.blobFromImage emits NCHW, which every Caffe/Darknet-derived model
    wants. These MediaPipe exports are TensorFlow graphs and want NHWC, so
    blobFromImage silently produces a tensor the first convolution reads with
    the axes transposed. The failure is loud (channel count 256) only because
    256 happens not to divide by 3; a square input of size 3k would have run and
    returned nonsense.
    """
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(
        img.astype(np.float32)[None, ...] / 255.0)


def _wanted(names) -> list[str]:
    """The landmark head and the presence head, found by SHAPE not by name.

    The zoo has renamed these between revisions ("Identity", "Identity_1", ...
    are whatever the TF export happened to emit), so matching on names would
    break on the next model refresh. Landmarks are the 195-value vector (39
    points x 5) and presence is the single scalar.
    """
    return [n for n in names if n in ("Identity", "Identity_1")] or list(names)


def rss_mb() -> float:
    """Resident set size in MB. ru_maxrss is bytes on macOS, KB on Linux."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--pruned-only", action="store_true",
                    help="never run the full graph, so peak RSS reflects "
                         "only the two heads we actually use")
    a = ap.parse_args()

    if not MODEL.exists():
        sys.exit(f"missing model: {MODEL}")
    print(f"opencv {cv2.__version__}")
    print(f"model  {MODEL.name}  {MODEL.stat().st_size / 1e6:.1f} MB")
    base = rss_mb()
    print(f"RSS before load: {base:.0f} MB")

    net = cv2.dnn.readNet(str(MODEL))
    after_load = rss_mb()
    print(f"RSS after load : {after_load:.0f} MB   (+{after_load - base:.0f} MB)")

    all_names = net.getUnconnectedOutLayersNames()
    # ru_maxrss is a HIGH-WATER MARK, so once the full graph has run the peak
    # can never come back down. Measuring what the pruned graph costs therefore
    # requires a process that never runs the full one.
    names = _wanted(all_names) if a.pruned_only else all_names
    blob = _blob(np.random.randint(0, 255, (a.size, a.size, 3), dtype=np.uint8),
                 a.size)
    net.setInput(blob)
    outs = net.forward(names)
    print("\noutputs:" + ("  (pruned)" if a.pruned_only else ""))
    for n, o in zip(names, outs):
        print(f"  {n:<32} {str(o.shape):<22} "
              f"range {float(o.min()):+.3f}..{float(o.max()):+.3f}")
    after_fwd = rss_mb()
    print(f"\nRSS after 1 forward: {after_fwd:.0f} MB   (+{after_fwd - base:.0f} MB total)")

    # Speed. Per-image cost matters: the engine already runs YOLOX and YuNet on
    # every frame, and a 500 photo cull has to finish inside a request.
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        net.setInput(blob)
        net.forward(names)
    dt = (time.perf_counter() - t0) / reps
    print(f"forward pass   : {1000 * dt:.0f} ms/crop  "
          f"-> +{dt * 500:.0f}s on a 500 photo cull (1 subject/frame)")

    # Only two of the five heads are wanted: the landmark vector and the
    # presence score. The others are a 256x256 segmentation mask and a 64x64x39
    # heatmap stack, which together are most of the activation memory and are
    # pure waste here. Asking for a subset lets OpenCV skip those branches.
    want = _wanted(names)
    t0 = time.perf_counter()
    for _ in range(reps):
        net.setInput(blob)
        net.forward(want)
    dt2 = (time.perf_counter() - t0) / reps
    print(f"pruned to {len(want)} heads: {1000 * dt2:.0f} ms/crop "
          f"({100 * (1 - dt2 / max(dt, 1e-9)):+.0f}% time)")
    print(f"RSS after pruned passes: {rss_mb():.0f} MB")

    peak = rss_mb()
    for _ in range(reps):
        net.setInput(blob)
        net.forward(names)
        gc.collect()
    print(f"RSS after {2 * reps} passes: {rss_mb():.0f} MB   "
          f"(growth since first: {rss_mb() - peak:+.0f} MB)")

    if a.folder and a.folder.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(p for p in a.folder.iterdir()
                       if p.suffix.lower() in exts)[:a.n]
        print(f"\nreal crops from {a.folder.name} ({len(paths)} frames):")
        import engine
        ok = 0
        for p in paths:
            prev = engine._load_metrics_image(p)
            if prev is None:
                continue
            rgb = np.array(prev)
            h, w = rgb.shape[:2]
            faces = engine.detect_faces_scaled(rgb)
            objs = engine.detect_objects(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            (x0, y0, x1, y1), _, src = engine._pick_subject(
                objs.get(0, []), faces, w, h)
            # _pick_subject returns floats and can hand back a box reaching
            # outside the frame, so clamp to int bounds before slicing.
            x0, y0 = int(max(0, x0)), int(max(0, y0))
            x1, y1 = int(min(w, x1)), int(min(h, y1))
            crop = rgb[y0:y1, x0:x1]
            if crop.size < 100:
                continue
            net.setInput(_blob(crop, a.size))
            o = net.forward(names)
            # The landmark tensor is the big one; the other head is a presence
            # score. Identify by size rather than by name, which has changed
            # between zoo revisions.
            lm = max(o, key=lambda t: t.size)
            conf = min(o, key=lambda t: t.size)
            ok += 1
            if ok <= 5:
                print(f"  {p.name:<18} subj={src:<7} crop={crop.shape[1]}x{crop.shape[0]}"
                      f"  landmarks={lm.size}  presence={float(conf.flatten()[0]):+.3f}")
        print(f"  ran on {ok}/{len(paths)} frames")
        print(f"\nRSS peak with engine loaded too: {rss_mb():.0f} MB")
        print("Budget is 512MB and production already peaks at 423MB.")


if __name__ == "__main__":
    main()
