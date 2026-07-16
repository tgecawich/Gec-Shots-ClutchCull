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

RESAMPLING = getattr(Image, "Resampling", Image)
METRICS_MAX_WIDTH = 1200
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CANVAS_MAX_LONG_EDGE = 2880

SCORING_PRESETS = {
    "Sports Action": {"sharpness": 0.50, "faces": 0.17, "detail": 0.20, "contrast": 0.08, "exposure": 0.05},
    "Portraits": {"sharpness": 0.25, "faces": 0.30, "detail": 0.10, "contrast": 0.15, "exposure": 0.20},
    "Events": {"sharpness": 0.35, "faces": 0.22, "detail": 0.15, "contrast": 0.13, "exposure": 0.15},
    "Balanced": {"sharpness": 0.45, "faces": 0.17, "detail": 0.18, "contrast": 0.10, "exposure": 0.10},
}

CANVAS_RATIOS = {"3:4": (1080, 1440), "4:5": (1080, 1350), "1:1": (1080, 1080)}

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
    perceptual_hash: imagehash.ImageHash | None = None
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    selection_reason: str = ""


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
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    edges = cv2.Canny(gray, 100, 200)
    detail = float(np.count_nonzero(edges) / edges.size)
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    faces = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    subject_sharp, face_score = _subject_metrics(gray, faces, gray.shape[1], gray.shape[0])
    return PhotoCandidate(
        path=path, sharpness=sharpness, detail_ratio=detail, contrast=contrast,
        brightness_mean=brightness, exposure_balance=exposure,
        subject_sharpness=subject_sharp, face_score=face_score,
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


def add_scores(cands, weights):
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
        c.score_breakdown = {"sharpness": subj[i], "faces": fac[i], "detail": det[i],
                             "contrast": con[i], "exposure": exp[i]}
        c.score = 100 * (w.get("sharpness", 0) * subj[i] + w.get("faces", 0) * fac[i]
                         + w.get("detail", 0) * det[i] + w.get("contrast", 0) * con[i]
                         + w.get("exposure", 0) * exp[i])
        top = max(c.score_breakdown.items(), key=lambda kv: kv[1])
        c.selection_reason = _BADGE.get(top[0], "Strong pick")
    return cands


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
        perceptual_hash=imagehash.hex_to_hash(ph) if ph else None,
    )


def _cpu_workers(n_items: int) -> int:
    # Metrics ops (OpenCV, PIL, numpy, YuNet) release the GIL, so threads give
    # real parallelism. Capped to stay within the free-tier memory budget.
    return max(1, min(4, os.cpu_count() or 2, n_items))


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
        if c.sharpness < blur_threshold:
            blurry.append(c)
        else:
            candidates.append(c)
    scored = add_scores(candidates, weights)
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
