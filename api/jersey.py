"""Find jersey numbers in a shoot, constrained to a roster the user supplies.

STATUS: groundwork, not wired into the app yet.

Measured on real game frames: precision is perfect (zero numbers returned that
were not on the roster) but recall is only about 29%, so most photos of a given
player are missed. The bottleneck is the recognition model, which handles plain
printed digits but not stylised athletic numerals. Replacing it with a small
digit classifier trained on jersey crops is the outstanding work; everything
around it (person detection, torso crop, text detection, geometry filter,
roster matching) is done and tested at 427MB peak, inside the 512MB budget.

Intended UX once recall is good enough: run it on the keepers after a cull, or
standalone on a folder without culling first, the same way the canvas tool
works.

Why constrained: open-vocabulary reading of jersey numbers is unreliable. The
recognition model is trained on signage and documents, and stylised athletic
numerals sit well outside that. Measured on real game frames it produced
confident wrong answers, which for grouping is worse than producing nothing.

Giving the model a roster turns recognition into matching against a handful of
candidates. Anything the roster cannot explain is discarded, so the failure
mode becomes "missed a photo" instead of "filed it under the wrong player".

Pipeline: person detection (shared with culling) -> torso crop -> text
detection -> geometry filter to drop wordmarks and sponsor logos -> text
recognition -> roster match.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import engine

VOCAB = list("0123456789abcdefghijklmnopqrstuvwxyz")

DET_PATH = Path(__file__).parent / "models" / "text_detection_en_ppocrv3.onnx"
REC_PATH = Path(__file__).parent / "models" / "text_recognition_CRNN_EN.onnx"
DET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
           "text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx")
REC_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
           "text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx")

# Loaded lazily: these add ~200MB on top of the person detector, which only fits
# because jersey search is opt-in and runs after culling, never during it.
_det = None
_rec = None
_lock = threading.Lock()
_attempted = False

# Geometry thresholds for telling a number apart from a team wordmark or a
# sponsor logo. Tuned against real frames; overridable if a sport differs.
MAX_ASPECT = float(os.getenv("CLUTCHCULL_JERSEY_MAX_AR", "2.6"))
MIN_H_FRAC = float(os.getenv("CLUTCHCULL_JERSEY_MIN_H", "0.07"))
EDGE_MARGIN = float(os.getenv("CLUTCHCULL_JERSEY_EDGE", "0.06"))


def _ensure_models() -> bool:
    global _attempted
    if DET_PATH.exists() and REC_PATH.exists():
        return True
    with _lock:
        if DET_PATH.exists() and REC_PATH.exists():
            return True
        if _attempted:
            return False
        _attempted = True
        try:
            import requests

            DET_PATH.parent.mkdir(parents=True, exist_ok=True)
            for url, path in ((DET_URL, DET_PATH), (REC_URL, REC_PATH)):
                if path.exists():
                    continue
                r = requests.get(url, timeout=180)
                r.raise_for_status()
                path.write_bytes(r.content)
            return True
        except Exception:
            return False


def _models():
    global _det, _rec
    if _det is None or _rec is None:
        with _lock:
            if _det is None or _rec is None:
                if not _ensure_models():
                    return None, None
                try:
                    d = cv2.dnn_TextDetectionModel_DB(str(DET_PATH))
                    d.setBinaryThreshold(0.3).setPolygonThreshold(0.5)
                    d.setMaxCandidates(200).setUnclipRatio(2.0)
                    d.setInputParams(1 / 255.0, (736, 736), (122.67, 116.66, 104.01))
                    r = cv2.dnn_TextRecognitionModel(str(REC_PATH))
                    r.setDecodeType("CTC-greedy")
                    r.setVocabulary(VOCAB)
                    # CRNN_EN takes a single grey channel; BGR fails the first conv.
                    r.setInputParams(1 / 127.5, (100, 32), (127.5,), False)
                    _det, _rec = d, r
                except Exception:
                    _det, _rec = False, False
    return (_det or None), (_rec or None)


def available() -> bool:
    d, r = _models()
    return d is not None and r is not None


def _looks_like_number(bw, bh, tw, th, cx) -> bool:
    """Numbers are chunky and central; wordmarks are wide, logos sit off-centre."""
    if bw < 8 or bh < 12:
        return False
    if bw / max(bh, 1) > MAX_ASPECT:
        return False
    if bh < th * MIN_H_FRAC:
        return False
    return EDGE_MARGIN < (cx / max(tw, 1)) < (1 - EDGE_MARGIN)


def match_roster(reading: str, roster: list[str]) -> str | None:
    """Only accept a reading the roster can account for."""
    if not reading:
        return None
    if not roster:
        return reading
    if reading in roster:
        return reading
    # Half the number is often hidden by an arm, so a partial read is common.
    cands = [n for n in roster if n.startswith(reading) or reading.startswith(n)]
    return cands[0] if len(cands) == 1 else None


def find_numbers(path: Path, roster: list[str], debug: bool = False):
    """Return the roster numbers visible in one frame."""
    det, rec = _models()
    if det is None:
        return [] if not debug else ([], [])

    try:
        im = Image.open(path)
        im = engine.ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return [] if not debug else ([], [])

    W, H = im.size
    small = im.copy()
    small.thumbnail((1600, 1600))
    sc = W / small.size[0]
    people = engine.detect_persons(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR))

    hits, trace = [], []
    for (x, y, w, h, _c) in sorted(people, key=lambda p: -p[2] * p[3])[:6]:
        x0 = max(0, int((x + w * 0.06) * sc)); y0 = max(0, int((y + h * 0.12) * sc))
        x1 = min(W, int((x + w * 0.94) * sc)); y1 = min(H, int((y + h * 0.66) * sc))
        if x1 - x0 < 40 or y1 - y0 < 40:
            continue
        bgr = cv2.cvtColor(np.array(im.crop((x0, y0, x1, y1))), cv2.COLOR_RGB2BGR)
        th, tw = bgr.shape[:2]
        try:
            boxes, _ = det.detect(bgr)
        except cv2.error:
            boxes = []
        boxes = list(boxes or [])

        # Detection often returns one box covering the whole jersey rather than
        # isolating the numeral, and recognition fails on that. Because the
        # roster rejects anything it cannot explain, we can afford to also try a
        # few fixed chest crops: extra candidates cost precision nothing.
        for fx0, fy0, fx1, fy1 in ((0.22, 0.10, 0.78, 0.52), (0.30, 0.16, 0.70, 0.46),
                                   (0.15, 0.05, 0.85, 0.40)):
            cx0, cy0 = int(tw * fx0), int(th * fy0)
            cx1, cy1 = int(tw * fx1), int(th * fy1)
            if cx1 - cx0 > 12 and cy1 - cy0 > 16:
                boxes.append(np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]]))
        for box in (boxes or []):
            p = np.array(box)
            bx0, by0 = max(0, p[:, 0].min()), max(0, p[:, 1].min())
            bx1, by1 = p[:, 0].max(), p[:, 1].max()
            bw, bh = bx1 - bx0, by1 - by0
            patch = bgr[int(by0):int(by1), int(bx0):int(bx1)]
            if patch.size == 0:
                continue
            try:
                txt = rec.recognize(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            except cv2.error:
                continue
            digits = "".join(c for c in txt if c.isdigit())
            passed = _looks_like_number(bw, bh, tw, th, (bx0 + bx1) / 2)
            if debug:
                trace.append({"raw": txt, "digits": digits, "w": int(bw), "h": int(bh),
                              "ar": round(bw / max(bh, 1), 2),
                              "h_frac": round(bh / max(th, 1), 3),
                              "cx_frac": round((bx0 + bx1) / 2 / max(tw, 1), 2),
                              "geom_ok": passed})
            if not passed or not digits:
                continue
            m = match_roster(digits, roster)
            if m:
                hits.append(m)

    uniq = sorted(set(hits))
    return (uniq, trace) if debug else uniq
