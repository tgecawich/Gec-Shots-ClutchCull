"""ClutchCull culling engine — Streamlit-free, stateless, API-ready.

Faithful port of the proven logic from the Streamlit app: subject-aware
sharpness (YuNet face detection), near-duplicate removal (perceptual hash),
weighted quality scoring, and white-canvas export. No global state — every
function takes its inputs explicitly so it's safe behind a web API.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps

# OpenCV allocates scratch buffers per internal thread. We parallelize across
# images ourselves, so internal threading only adds memory (and contention) on a
# small box. Keep it at 1 unless explicitly overridden.
try:
    cv2.setNumThreads(int(os.getenv("CLUTCHCULL_CV_THREADS", "1")))
except Exception:
    pass

RESAMPLING = getattr(Image, "Resampling", Image)
# Blur detection is extremely resolution-sensitive: measured on the same photo,
# sharp-vs-soft separation is only ~4x at 1200px but ~28x at 2400px (downscaling
# is itself a blur filter, so it hides the very thing we're testing for).
# 1800 is the memory-safe default for a 512MB instance while still giving ~11x
# sharp-vs-soft separation (vs only 4x at the old 1200px). Raise to 2400 (~28x)
# via env if you move to a bigger box.
METRICS_MAX_WIDTH = int(os.getenv("CLUTCHCULL_METRICS_WIDTH", "1800"))
# Focus-miss guard: if the subject region is much softer than the overall
# frame, focus probably landed on the background — demote it. Tunable at runtime.
FOCUS_MIN = float(os.getenv("CLUTCHCULL_FOCUS_MIN", "0.6"))   # subject/frame sharpness at/above this = fine
FOCUS_FLOOR = float(os.getenv("CLUTCHCULL_FOCUS_FLOOR", "0.25"))  # worst-case multiplier for a clear miss
NOFACE_TRUST = float(os.getenv("CLUTCHCULL_NOFACE_TRUST", "0.85"))  # discount unverified (no-face) subjects
# Sharpness GATE: for sports, a soft subject is a delete no matter how well
# exposed/composed. So sharpness multiplies the whole score instead of merely
# adding to it — a soft shot can't buy its way to the top with light + detail.
# Judged in ABSOLUTE terms (subject sharpness vs the blur floor) so a batch of
# genuinely sharp keepers isn't penalized just for having a "least sharp" one.
SHARP_GATE_SPAN = float(os.getenv("CLUTCHCULL_SHARP_GATE_SPAN", "6.0"))    # full credit at blur_floor * this
SHARP_GATE_FLOOR = float(os.getenv("CLUTCHCULL_SHARP_GATE_FLOOR", "0.3"))  # worst-case score multiplier
SHARP_SOFT_MARK = float(os.getenv("CLUTCHCULL_SOFT_MARK", "1.8"))     # subject sharpness below blur_floor*this -> 'soft'
SUBJECT_REJECT = float(os.getenv("CLUTCHCULL_SUBJECT_REJECT", "0.9"))  # reject when subject sharpness < blur_floor*this
# Faces are large features, so detection stays accurate on a downscaled copy —
# and YuNet cost grows fast with resolution (~3x from 800px to 1200px). We
# detect small, then scale boxes back up; sharpness still uses full metrics res.
FACE_DETECT_WIDTH = int(os.getenv("CLUTCHCULL_FACE_WIDTH", "800"))
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CANVAS_MAX_LONG_EDGE = 2880

SCORING_PRESETS = {
    "Sports Action": {"sharpness": 0.50, "faces": 0.17, "detail": 0.20, "contrast": 0.08, "exposure": 0.05},
    "Portraits": {"sharpness": 0.25, "faces": 0.30, "detail": 0.10, "contrast": 0.15, "exposure": 0.20},
    "Events": {"sharpness": 0.35, "faces": 0.22, "detail": 0.15, "contrast": 0.13, "exposure": 0.15},
    "Balanced": {"sharpness": 0.45, "faces": 0.17, "detail": 0.18, "contrast": 0.10, "exposure": 0.10},
}

CANVAS_RATIOS = {"3:4": (1080, 1440), "4:5": (1080, 1350), "1:1": (1080, 1080)}

# Person detector (COCO YOLOX-tiny). Sports subjects wear helmets and turn away,
# so face detection alone can't find them — this locates the ATHLETE'S BODY, which
# is what we must measure sharpness on.
YOLOX_MODEL_PATH = Path(__file__).parent / "models" / "object_detection_yolox_2022nov.onnx"
YOLOX_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
# 416 not 640: on a 512MB box the detector's activations are the single biggest
# memory consumer (640 -> 349MB peak, 416 -> 282MB). Sports subjects are large in
# frame, so the smaller input costs little accuracy. Env-tunable if you size up.
YOLOX_SIZE = int(os.getenv("CLUTCHCULL_YOLOX_SIZE", "320"))
PERSON_CONF = float(os.getenv("CLUTCHCULL_PERSON_CONF", "0.35"))
# ONE shared detector, not thread-local: a per-thread copy would load the whole
# network per concurrent request and blow a 512MB box instantly. cv2 DNN forward
# isn't thread-safe, so calls are serialized with _infer_lock — which is fine
# because we deliberately process one image at a time on a small instance.
_person_net = None
_yolox_lock = threading.Lock()
_infer_lock = threading.Lock()
_yolox_attempted = False

YUNET_MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_face_detector_local = threading.local()
_yunet_lock = threading.Lock()
_yunet_attempted = False


@dataclass
class PhotoCandidate:
    path: Path
    sharpness: float = 0.0
    detail_ratio: float = 0.0
    contrast: float = 0.0
    brightness_mean: float = 0.0
    exposure_balance: float = 0.0
    subject_sharpness: float = 0.0
    face_score: float = 0.0
    has_face: bool = False
    perceptual_hash: imagehash.ImageHash | None = None
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    selection_reason: str = ""
    soft: bool = False


# --- face model (downloaded once at runtime) ------------------------------
def ensure_yunet_model() -> bool:
    global _yunet_attempted
    if YUNET_MODEL_PATH.exists():
        return True
    with _yunet_lock:
        if YUNET_MODEL_PATH.exists():
            return True
        if _yunet_attempted:
            return False
        _yunet_attempted = True
        try:
            import requests

            YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(YUNET_MODEL_URL, timeout=20)
            resp.raise_for_status()
            YUNET_MODEL_PATH.write_bytes(resp.content)
            return True
        except Exception:
            return False


def _get_face_detector():
    det = getattr(_face_detector_local, "detector", None)
    if det is None:
        if not ensure_yunet_model() or not hasattr(cv2, "FaceDetectorYN"):
            _face_detector_local.detector = False
            return None
        try:
            _face_detector_local.detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL_PATH), "", (320, 320), 0.6, 0.3, 5000
            )
        except Exception:
            _face_detector_local.detector = False
    return _face_detector_local.detector or None


def detect_faces(bgr):
    det = _get_face_detector()
    if det is None:
        return []
    try:
        h, w = bgr.shape[:2]
        det.setInputSize((w, h))
        _, faces = det.detect(bgr)
        if faces is None:
            return []
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in faces]
    except Exception:
        return []


def detect_faces_scaled(rgb):
    """Detect on a downscaled copy for speed, then scale boxes back to full res.
    ~2x faster than detecting at metrics resolution, with no meaningful accuracy
    loss for the primary (foreground) subject."""
    h, w = rgb.shape[:2]
    if w > FACE_DETECT_WIDTH:
        sh = max(1, int(h * FACE_DETECT_WIDTH / w))
        small = cv2.resize(rgb, (FACE_DETECT_WIDTH, sh), interpolation=cv2.INTER_AREA)
        inv = w / FACE_DETECT_WIDTH
    else:
        small, inv = rgb, 1.0
    faces = detect_faces(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if inv != 1.0:
        faces = [(x * inv, y * inv, fw * inv, fh * inv, conf) for (x, y, fw, fh, conf) in faces]
    return faces


def ensure_yolox_model() -> bool:
    global _yolox_attempted
    if YOLOX_MODEL_PATH.exists():
        return True
    with _yolox_lock:
        if YOLOX_MODEL_PATH.exists():
            return True
        if _yolox_attempted:
            return False
        _yolox_attempted = True
        try:
            import requests

            YOLOX_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            resp = requests.get(YOLOX_MODEL_URL, timeout=120)
            resp.raise_for_status()
            YOLOX_MODEL_PATH.write_bytes(resp.content)
            return True
        except Exception:
            return False


def person_detection_enabled() -> bool:
    """Kill switch: set CLUTCHCULL_PERSON=0 to disable person detection if the
    instance is memory-starved. Culling still works (face -> center fallback) and
    still benefits from the higher analysis resolution."""
    return os.getenv("CLUTCHCULL_PERSON", "1").strip().lower() not in ("0", "false", "no")


def _get_person_net():
    global _person_net
    if _person_net is None:
        with _yolox_lock:
            if _person_net is None:
                if not person_detection_enabled():
                    _person_net = False
                elif not ensure_yolox_model():
                    _person_net = False
                else:
                    try:
                        _person_net = cv2.dnn.readNet(str(YOLOX_MODEL_PATH))
                    except Exception:
                        _person_net = False
    return _person_net or None


_YOLOX_GRID = None


def _yolox_grid():
    """Grid + stride tensors for decoding YOLOX output (built once)."""
    global _YOLOX_GRID
    if _YOLOX_GRID is None:
        grids, strides = [], []
        for s in (8, 16, 32):
            g = YOLOX_SIZE // s
            xv, yv = np.meshgrid(np.arange(g), np.arange(g))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((g * g, 1), s))
        _YOLOX_GRID = (np.concatenate(grids), np.concatenate(strides))
    return _YOLOX_GRID


def detect_persons(bgr):
    """Return [(x, y, w, h, conf)] for people, in `bgr` pixel coords."""
    net = _get_person_net()
    if net is None:
        return []
    try:
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (YOLOX_SIZE, YOLOX_SIZE)), 1.0, (YOLOX_SIZE, YOLOX_SIZE), swapRB=True
        )
        with _infer_lock:  # shared net: one inference at a time
            net.setInput(blob)
            out = net.forward()[0]
        grids, strides = _yolox_grid()
        xy = (out[:, 0:2] + grids) * strides
        wh = np.exp(out[:, 2:4]) * strides
        cls_scores = out[:, 5:]
        scores = out[:, 4] * cls_scores.max(1)
        keep = (cls_scores.argmax(1) == 0) & (scores > PERSON_CONF)  # class 0 = person
        if not keep.any():
            return []
        xy, wh, sc = xy[keep], wh[keep], scores[keep]
        boxes = np.stack([xy[:, 0] - wh[:, 0] / 2, xy[:, 1] - wh[:, 1] / 2, wh[:, 0], wh[:, 1]], 1)
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), PERSON_CONF, 0.45)
        if len(idx) == 0:
            return []
        sx, sy = w / float(YOLOX_SIZE), h / float(YOLOX_SIZE)
        return [
            (float(boxes[i][0] * sx), float(boxes[i][1] * sy),
             float(boxes[i][2] * sx), float(boxes[i][3] * sy), float(sc[i]))
            for i in np.array(idx).flatten()
        ]
    except Exception:
        return []


def _region_sharpness(gray, x0, y0, x1, y1, tiles=4):
    """Sharpness of a region, robust to a few crisp background edges sneaking into
    the box. Plain variance is dominated by the single sharpest thing present, so
    we tile the region and take a high percentile — 'is most of the subject sharp?'
    rather than 'is anything in this box sharp?'."""
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(gray.shape[1], int(x1)), min(gray.shape[0], int(y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    crop = gray[y0:y1, x0:x1]
    lap = cv2.Laplacian(crop, cv2.CV_32F)  # 32F not 64F: half the memory, same ranking
    h, w = lap.shape
    th, tw = max(1, h // tiles), max(1, w // tiles)
    vals = []
    for ty in range(0, h - th + 1, th):
        for tx in range(0, w - tw + 1, tw):
            vals.append(lap[ty:ty + th, tx:tx + tw].var())
    if not vals:
        return float(lap.var())
    # 75th percentile: the subject's detailed areas, ignoring flat jersey/sky and
    # without letting one sharp background sliver carry the whole frame.
    return float(np.percentile(vals, 75))


def _pick_subject(persons, faces, img_w, img_h):
    """Choose the subject box: prefer a detected PERSON (works with helmets and
    backs turned), then a face, then fall back to a center crop."""
    cx, cy = img_w / 2.0, img_h / 2.0
    maxd = (cx ** 2 + cy ** 2) ** 0.5 or 1.0

    def score_box(x, y, bw, bh, conf):
        bcx, bcy = x + bw / 2.0, y + bh / 2.0
        centrality = 1.0 - min(1.0, ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5 / maxd)
        return (bw * bh) * (0.55 + 0.45 * centrality) * conf

    if persons:
        x, y, bw, bh, conf = max(persons, key=lambda p: score_box(*p))
        pad_x, pad_y = bw * 0.05, bh * 0.05
        subj = (x - pad_x, y - pad_y, x + bw + pad_x, y + bh + pad_y)
        # Confidence that there's a clear subject: bigger + surer = better.
        presence = min(1.0, (bw * bh) / (0.16 * img_w * img_h)) * conf
        return subj, presence, "person"
    if faces:
        x, y, fw, fh, conf = max(faces, key=lambda f: score_box(*f))
        subj = (x - fw * 0.6, y - fh * 0.5, x + fw * 1.6, y + fh + fh * 1.4)
        presence = min(1.0, (fw / (0.22 * img_w)) if img_w else 0.0) * conf
        return subj, presence, "face"
    mw, mh = img_w * 0.225, img_h * 0.225
    return (mw, mh, img_w - mw, img_h - mh), 0.0, "center"


def _subject_metrics(gray, faces, img_w, img_h):
    if faces:
        cx, cy = img_w / 2.0, img_h / 2.0
        maxd = (cx ** 2 + cy ** 2) ** 0.5 or 1.0

        def priority(f):
            x, y, fw, fh, conf = f
            fcx, fcy = x + fw / 2.0, y + fh / 2.0
            centrality = 1.0 - min(1.0, ((fcx - cx) ** 2 + (fcy - cy) ** 2) ** 0.5 / maxd)
            return (fw * fh) * (0.5 + 0.5 * centrality) * conf

        x, y, fw, fh, conf = max(faces, key=priority)
        x0, y0 = int(max(0, x - fw * 0.6)), int(max(0, y - fh * 0.5))
        x1, y1 = int(min(img_w, x + fw * 1.6)), int(min(img_h, y + fh + fh * 1.4))
        face_score = min(1.0, (fw / (0.22 * img_w)) if img_w else 0.0) * conf
    else:
        mw, mh = int(img_w * 0.225), int(img_h * 0.225)
        x0, y0, x1, y1 = mw, mh, img_w - mw, img_h - mh
        face_score = 0.0
    crop = gray[y0:y1, x0:x1]
    src = crop if crop.size else gray
    return float(cv2.Laplacian(src, cv2.CV_64F).var()), face_score


def _load_metrics_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            ow, oh = img.size
            if ow > METRICS_MAX_WIDTH:
                img.draft("RGB", (METRICS_MAX_WIDTH, max(1, int(oh * METRICS_MAX_WIDTH / ow))))
            img = img.convert("RGB")
            if img.width > METRICS_MAX_WIDTH:
                scale = METRICS_MAX_WIDTH / img.width
                img = img.resize((METRICS_MAX_WIDTH, max(1, int(img.height * scale))), RESAMPLING.BILINEAR)
            else:
                img = img.copy()
            return img
    except Exception:
        return None


def compute_metrics(path: Path) -> PhotoCandidate | None:
    preview = _load_metrics_image(path)
    if preview is None:
        return None
    rgb = np.array(preview)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    contrast = float(gray.std())
    brightness = float(gray.mean())
    edges = cv2.Canny(gray, 100, 200)
    detail = float(np.count_nonzero(edges) / edges.size)
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    h, w = gray.shape[:2]
    # Whole-frame sharpness, measured the same robust (tiled) way as the subject
    # so the two are directly comparable for the focus-miss check.
    frame_sharp = _region_sharpness(gray, 0, 0, w, h, tiles=6)
    faces = detect_faces_scaled(rgb)
    persons = detect_persons(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    (sx0, sy0, sx1, sy1), presence, src = _pick_subject(persons, faces, w, h)
    subject_sharp = _region_sharpness(gray, sx0, sy0, sx1, sy1)
    return PhotoCandidate(
        path=path, sharpness=frame_sharp, detail_ratio=detail,
        contrast=contrast, brightness_mean=brightness, exposure_balance=exposure,
        subject_sharpness=subject_sharp, face_score=presence,
        has_face=(src != "center"),  # subject was actually located, not guessed
        perceptual_hash=imagehash.phash(preview),
    )


def _normalize(values):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


_BADGE = {"sharpness": "Sharp subject", "faces": "Clear subject", "detail": "Rich detail",
          "contrast": "Clean contrast", "exposure": "Well-exposed"}


def _effective_subject_sharpness(c: PhotoCandidate) -> float:
    """Absolute subject sharpness, discounted when focus missed to background."""
    return c.subject_sharpness * _focus_factor(c)


def add_scores(cands, weights, blur_threshold=40.0):
    if not cands:
        return []
    total = sum(weights.values()) or 1.0
    w = {k: v / total for k, v in weights.items()}
    subj = _normalize([c.subject_sharpness for c in cands])
    det = _normalize([c.detail_ratio for c in cands])
    con = _normalize([c.contrast for c in cands])
    exp = [c.exposure_balance for c in cands]
    fac = [min(1.0, max(0.0, c.face_score)) for c in cands]
    for i, c in enumerate(cands):
        # Focus-miss guard: how sharp is the SUBJECT vs the whole frame? A shot
        # where focus fell on the background (soft subject, crisp background) has
        # a low ratio and gets its sharpness credit cut. Detail (whole-frame edge
        # richness) is cut the same way so a crisp background can't carry a soft
        # subject to the top.
        focus = _focus_factor(c)
        sharp_i = subj[i] * focus
        detail_i = det[i] * (0.5 + 0.5 * focus)  # background detail counts less on a focus miss
        c.score_breakdown = {"sharpness": sharp_i, "faces": fac[i], "detail": detail_i,
                             "contrast": con[i], "exposure": exp[i]}
        base = 100 * (w.get("sharpness", 0) * sharp_i + w.get("faces", 0) * fac[i]
                      + w.get("detail", 0) * detail_i + w.get("contrast", 0) * con[i]
                      + w.get("exposure", 0) * exp[i])
        # Sharpness GATE (multiplicative, ABSOLUTE): a soft subject caps the whole
        # score, so good light/contrast/background-detail can't rescue an out-of-
        # focus shot. Measured against the blur floor, not the batch, so a set of
        # genuinely sharp keepers isn't dinged for having a relatively-softer one.
        eff = _effective_subject_sharpness(c)
        lo, hi = blur_threshold, blur_threshold * SHARP_GATE_SPAN
        q = 0.0 if hi <= lo else max(0.0, min(1.0, (eff - lo) / (hi - lo)))
        gate = SHARP_GATE_FLOOR + (1.0 - SHARP_GATE_FLOOR) * q
        c.score = base * gate
        c.soft = eff < blur_threshold * SHARP_SOFT_MARK
        if c.soft:
            c.selection_reason = "Soft focus" if focus >= 0.55 else "Soft subject"
        else:
            top = max(c.score_breakdown.items(), key=lambda kv: kv[1])
            c.selection_reason = _BADGE.get(top[0], "Strong pick")
    return cands


def _focus_factor(c: PhotoCandidate) -> float:
    """1.0 when the subject is as sharp as (or sharper than) the frame; drops
    toward FOCUS_FLOOR as the subject gets softer than the background. No-face
    shots (subject is only a center guess) are trusted a little less."""
    ratio = c.subject_sharpness / (c.sharpness + 1e-6)
    factor = FOCUS_FLOOR + (1.0 - FOCUS_FLOOR) * min(1.0, ratio / FOCUS_MIN)
    if not c.has_face:
        factor *= NOFACE_TRUST
    return max(FOCUS_FLOOR * (NOFACE_TRUST if not c.has_face else 1.0), min(1.0, factor))


def remove_duplicates(cands, threshold):
    """Return (kept, dup_map) where dup_map[keeper_name] = [near-dup candidates]."""
    kept = []
    dup_map: dict = {}
    for c in sorted(cands, key=lambda x: -x.score):
        dup_of = None
        for k in kept:
            if c.perceptual_hash is not None and k.perceptual_hash is not None:
                if (c.perceptual_hash - k.perceptual_hash) <= threshold:
                    dup_of = k
                    break
        if dup_of is not None:
            dup_map.setdefault(dup_of.path.name, []).append(c)
        else:
            kept.append(c)
    return kept, dup_map


# --- metrics <-> JSON (so the browser can cache them and re-rank instantly) --
def _metrics_to_dict(c: PhotoCandidate) -> dict:
    return {
        "filename": c.path.name,
        "sharpness": c.sharpness,
        "detail_ratio": c.detail_ratio,
        "contrast": c.contrast,
        "brightness_mean": c.brightness_mean,
        "exposure_balance": c.exposure_balance,
        "subject_sharpness": c.subject_sharpness,
        "face_score": c.face_score,
        "has_face": c.has_face,
        "phash": str(c.perceptual_hash) if c.perceptual_hash is not None else None,
    }


def _dict_to_candidate(d: dict) -> PhotoCandidate:
    ph = d.get("phash")
    return PhotoCandidate(
        path=Path(d.get("filename", "")),
        sharpness=float(d.get("sharpness", 0.0)),
        detail_ratio=float(d.get("detail_ratio", 0.0)),
        contrast=float(d.get("contrast", 0.0)),
        brightness_mean=float(d.get("brightness_mean", 0.0)),
        exposure_balance=float(d.get("exposure_balance", 0.0)),
        subject_sharpness=float(d.get("subject_sharpness", 0.0)),
        face_score=float(d.get("face_score", 0.0)),
        has_face=bool(d.get("has_face", False)),
        perceptual_hash=imagehash.hex_to_hash(ph) if ph else None,
    )


def _cpu_workers(n_items: int) -> int:
    # Metrics ops (OpenCV, PIL, numpy, YuNet) release the GIL, so threads give
    # real parallelism. Capped low (and overridable) to stay within the tight
    # free-tier memory budget — each worker holds a decoded image + detector.
    # 1 by default: Render Starter is 0.5 CPU / 512MB, so extra threads buy no
    # throughput and only add memory pressure. Raise via env on a bigger box.
    cap = int(os.getenv("CLUTCHCULL_WORKERS", "1"))
    return max(1, min(cap, os.cpu_count() or 2, n_items))


def compute_metrics_batch(image_paths) -> list[dict]:
    """Compute per-image metrics in PARALLEL. Each dict is JSON-serializable and
    self-contained, so the browser can cache it and re-rank (change sliders)
    without re-uploading or recomputing. Unreadable images -> {'unreadable': True}."""
    paths = list(image_paths)
    if not paths:
        return []

    def work(p):
        m = compute_metrics(p)
        if m is None:
            return {"filename": Path(p).name, "unreadable": True}
        return _metrics_to_dict(m)

    workers = _cpu_workers(len(paths))
    if workers == 1:
        return [work(p) for p in paths]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(work, paths))


def _assemble_result(total, blurry, unreadable, selected, dup_map):
    dup_count = sum(len(v) for v in dup_map.values())

    def cand_dict(c, with_dupes=False):
        d = {"filename": c.path.name, "score": round(c.score, 2), "badge": c.selection_reason,
             "soft": bool(c.soft),
             "breakdown": {k: round(v, 3) for k, v in c.score_breakdown.items()}}
        if with_dupes:
            d["duplicates"] = [cand_dict(a) for a in dup_map.get(c.path.name, [])]
        return d

    return {
        "total": total,
        "blurry_removed": len(blurry),
        "duplicates_removed": dup_count,
        "unreadable_skipped": unreadable,
        "keepers": [cand_dict(c, with_dupes=True) for c in selected],
        "rejected": [c.path.name for c in sorted(blurry, key=lambda c: -c.sharpness)],
    }


def rank_metrics(metrics, blur_threshold=40.0, duplicate_threshold=2, top_n=35, preset="Balanced"):
    """Turn pre-computed metrics into keepers. Cheap + fast (no image work) — this
    is what runs when the user tweaks the keeper/blur/duplicate/preset controls."""
    weights = SCORING_PRESETS.get(preset, SCORING_PRESETS["Balanced"])
    total = len(metrics)
    unreadable = sum(1 for m in metrics if m.get("unreadable"))
    candidates, blurry = [], []
    for m in metrics:
        if m.get("unreadable"):
            continue
        c = _dict_to_candidate(m)
        # Reject if the whole frame is blurry (original gate) OR the SUBJECT
        # itself is below the sharpness floor — a soft subject is a delete even
        # when the background is crisp.
        if c.sharpness < blur_threshold or _effective_subject_sharpness(c) < blur_threshold * SUBJECT_REJECT:
            blurry.append(c)
        else:
            candidates.append(c)
    scored = add_scores(candidates, weights, blur_threshold)
    unique, dup_map = remove_duplicates(scored, duplicate_threshold)
    unique.sort(key=lambda c: -c.score)
    selected = unique[:top_n]
    return _assemble_result(total, blurry, unreadable, selected, dup_map)


def cull(image_paths, blur_threshold=40.0, duplicate_threshold=2, top_n=35, preset="Balanced"):
    """One-shot cull (compute + rank). Kept for the legacy /cull-upload path."""
    metrics = compute_metrics_batch(image_paths)
    return rank_metrics(metrics, blur_threshold, duplicate_threshold, top_n, preset)


def create_white_canvas(src: Path, out: Path, canvas_w: int, canvas_h: int, padding: int):
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        sw, sh = img.size
        aw, ah = max(1, canvas_w - 2 * padding), max(1, canvas_h - 2 * padding)
        base_fit = min(aw / sw, ah / sh)
        scale = 1.0 / base_fit if base_fit < 1 else 1.0
        long_edge = max(canvas_w, canvas_h) * scale
        if long_edge > CANVAS_MAX_LONG_EDGE:
            scale *= CANVAS_MAX_LONG_EDGE / long_edge
        scale = max(scale, 1.0)
        ow, oh, op = round(canvas_w * scale), round(canvas_h * scale), round(padding * scale)
        img.thumbnail((max(1, ow - 2 * op), max(1, oh - 2 * op)), RESAMPLING.LANCZOS)
        canvas = Image.new("RGB", (ow, oh), "white")
        canvas.paste(img, ((ow - img.width) // 2, (oh - img.height) // 2))
        canvas.save(out, quality=98, subsampling=0, optimize=True)
