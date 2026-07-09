import gc
import os
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import cv2
import imagehash
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import requests
except ImportError:
    requests = None


INPUT_DIR = Path("input_photos")
OUTPUT_DIR = Path("output_photos")
CANVAS_DIR = Path("canvas_photos")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Browser-side preview settings for the fast uploader. Photos are downscaled
# in the user's browser to this max dimension before upload, then PUT directly
# to R2 via presigned URLs -- the app server never touches the upload bytes.
# 1800px keeps portrait-orientation frames at >= 1200px wide, so metric
# computation (METRICS_MAX_WIDTH below) still sees full-resolution inputs.
PREVIEW_MAX_DIMENSION = 1800
PREVIEW_JPEG_QUALITY = 0.82
FAST_UPLOADER_DIR = Path(__file__).parent / "components" / "fast_uploader"

ANALYSIS_MAX_WIDTH = 1200
# Metrics are computed at this width via a draft()-subsampled decode (see
# load_metrics_image). Kept at 1200 to preserve the exact keeper SET: dropping
# to 870 (a 1/8 decode) loses enough high-frequency detail to reselect a photo
# in the top-N. At 1200 the decoder subsamples to 1/4 during decode, then a
# cheap BILINEAR finishes the downscale. UI display previews are unaffected and
# still use ANALYSIS_MAX_WIDTH with LANCZOS.
METRICS_MAX_WIDTH = 1200
UI_PREVIEW_LIMIT = 20
RESAMPLING = getattr(Image, "Resampling", Image)

# Per-image analysis (JPEG decode + cv2 ops) releases the GIL, so threads scale
# well. Default to the CPU count; override via $CLUTCHCULL_WORKERS or the
# max_workers arg. None/0 -> auto.
def _default_worker_count() -> int:
    env_value = os.getenv("CLUTCHCULL_WORKERS", "").strip()
    if env_value:
        try:
            parsed = int(env_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return os.cpu_count() or 1

GOOGLE_FORM_URL = "https://docs.google.com/forms/u/0/d/e/1FAIpQLSdE_xxiIaiHwYX9LQag1kipieTojmqEfqv1fVqwtsCKo45Mlg/formResponse"
GOOGLE_FORM_FIELDS = {
    "event_type": "entry.1792514521",
    "email": "entry.1250685824",
    "photos_processed": "entry.1966720800",
    "exports": "entry.23206585",
    "minutes_saved": "entry.1192573871",
    "session_id": "entry.747824885",
}
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi3K-ggYir5zDj6mYXIMrWcG-fyTqWu6tQtQ2g97vFpzSZmKrJ0nnExHIBzA7zDCbvhabDdCG8EYSa/pub?gid=370944124&single=true&output=csv"

SCORING_PRESETS = {
    "Sports Action": {
        "sharpness": 0.6,
        "detail": 0.25,
        "contrast": 0.1,
        "exposure": 0.05,
    },
    "Portraits": {
        "sharpness": 0.25,
        "detail": 0.15,
        "contrast": 0.3,
        "exposure": 0.3,
    },
    "Events": {
        "sharpness": 0.35,
        "detail": 0.2,
        "contrast": 0.2,
        "exposure": 0.25,
    },
    "Balanced": {
        "sharpness": 0.55,
        "detail": 0.2,
        "contrast": 0.15,
        "exposure": 0.1,
    },
}

# Instagram-ready canvas sizes -> (width, height) in pixels. Order matters:
# the first entry is the default (3:4, the strongest single-image IG post size).
CANVAS_RATIOS = {
    "3:4 — 1080 × 1440 (best for IG posts)": (1080, 1440),
    "4:5 — 1080 × 1350": (1080, 1350),
    "1:1 — 1080 × 1080 (square)": (1080, 1080),
}

# Canvases render at up to this long-edge (2x Instagram's 1080/1440 display
# size). Keeps the photo near its native resolution for crisp downsampling
# while keeping file sizes reasonable. See create_white_canvas().
CANVAS_MAX_LONG_EDGE = 2880


@dataclass
class CanvasSettings:
    width: int
    height: int
    padding: int
    create_exports: bool


@dataclass
class PhotoCandidate:
    path: Path
    sharpness: float
    detail_ratio: float
    contrast: float
    brightness_mean: float
    exposure_balance: float
    perceptual_hash: imagehash.ImageHash
    r2_key: str = ""
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    selection_reason: str = ""


@dataclass
class SimilarPhotoGroup:
    keeper: PhotoCandidate
    rejected: list[PhotoCandidate] = field(default_factory=list)


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --clutch-bg: #090713;
            --clutch-panel: rgba(22, 18, 38, 0.92);
            --clutch-panel-soft: rgba(255, 255, 255, 0.055);
            --clutch-border: rgba(255, 255, 255, 0.12);
            --clutch-text: #f8f4ff;
            --clutch-muted: #b7aacb;
            --clutch-orange: #ff8a2a;
            --clutch-gold: #ffc15a;
            --clutch-purple: #7b3cff;
            --clutch-pink: #ff4ecd;
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 8%, rgba(123, 60, 255, 0.34), transparent 34rem),
                radial-gradient(circle at 92% 18%, rgba(255, 138, 42, 0.25), transparent 28rem),
                linear-gradient(135deg, #070510 0%, #111025 46%, #090713 100%);
            color: var(--clutch-text);
        }

        .block-container {
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(16, 12, 31, 0.98), rgba(8, 6, 18, 0.98));
            border-right: 1px solid var(--clutch-border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--clutch-text);
        }

        .clutch-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 28px;
            padding: clamp(1.4rem, 4vw, 3.2rem);
            margin-bottom: 1.4rem;
            background:
                linear-gradient(135deg, rgba(123, 60, 255, 0.38), rgba(255, 138, 42, 0.22)),
                linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03));
            box-shadow: 0 28px 90px rgba(0, 0, 0, 0.38);
        }

        .clutch-hero:before {
            content: "";
            position: absolute;
            inset: -35%;
            background:
                radial-gradient(circle, rgba(255, 255, 255, 0.16) 0 1px, transparent 1px);
            background-size: 28px 28px;
            transform: rotate(-8deg);
            opacity: 0.28;
        }

        .clutch-hero-content {
            position: relative;
            z-index: 1;
            max-width: 830px;
        }

        .clutch-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.42rem 0.72rem;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            background: rgba(8, 6, 18, 0.46);
            color: #ffe2bd;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .clutch-logo {
            margin-top: 1rem;
            color: var(--clutch-text);
            font-size: clamp(2.4rem, 7vw, 5.2rem);
            line-height: 0.9;
            font-weight: 950;
            letter-spacing: -0.08em;
        }

        .clutch-logo span {
            background: linear-gradient(90deg, var(--clutch-orange), var(--clutch-pink));
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .clutch-byline {
            margin-top: 0.55rem;
            color: #ffe2bd;
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }

        .clutch-headline {
            margin-top: 1rem;
            margin-bottom: 0.7rem;
            color: var(--clutch-text);
            font-size: clamp(1.55rem, 4vw, 3.3rem);
            line-height: 1.02;
            font-weight: 900;
            letter-spacing: -0.055em;
        }

        .clutch-subheadline {
            max-width: 760px;
            color: #efe9ff;
            font-size: clamp(1rem, 2vw, 1.18rem);
            line-height: 1.65;
        }

        .clutch-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.35rem;
        }

        .clutch-pill {
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            padding: 0.48rem 0.78rem;
            background: rgba(255, 255, 255, 0.08);
            color: #fff4e8;
            font-size: 0.88rem;
            font-weight: 750;
        }

        .clutch-section {
            margin: 2rem 0 0.85rem;
        }

        .clutch-section-label {
            display: inline-flex;
            padding: 0.32rem 0.62rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 138, 42, 0.26);
            background: rgba(255, 138, 42, 0.12);
            color: #ffd7b4;
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .clutch-section h2 {
            margin: 0.72rem 0 0.25rem;
            color: var(--clutch-text);
            font-size: clamp(1.45rem, 3vw, 2.2rem);
            line-height: 1.05;
            font-weight: 900;
            letter-spacing: -0.045em;
        }

        .clutch-section p {
            margin: 0;
            max-width: 760px;
            color: var(--clutch-muted);
            line-height: 1.55;
        }

        .clutch-card {
            border: 1px solid var(--clutch-border);
            border-radius: 22px;
            padding: 1.1rem 1.2rem;
            margin: 0.75rem 0 1.1rem;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.085), rgba(255, 255, 255, 0.035));
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        }

        .clutch-upload-card {
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 26px;
            padding: 1.2rem;
            background:
                linear-gradient(135deg, rgba(255, 138, 42, 0.12), rgba(123, 60, 255, 0.14)),
                rgba(255, 255, 255, 0.04);
        }

        .clutch-upload-title {
            margin: 0;
            color: var(--clutch-text);
            font-size: 1.45rem;
            font-weight: 900;
            letter-spacing: -0.035em;
        }

        .clutch-upload-copy {
            margin: 0.35rem 0 0;
            color: var(--clutch-muted);
            line-height: 1.55;
        }

        .clutch-metric-card {
            min-height: 84px;
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 16px;
            padding: 0.75rem 0.85rem;
            background:
                radial-gradient(circle at 90% 10%, rgba(255, 138, 42, 0.20), transparent 45%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.085), rgba(255, 255, 255, 0.035));
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
        }

        .clutch-metric-label {
            color: var(--clutch-muted);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .clutch-metric-value {
            margin-top: 0.3rem;
            color: var(--clutch-text);
            font-size: clamp(1.25rem, 3.2vw, 1.85rem);
            font-weight: 950;
            letter-spacing: -0.05em;
        }

        .clutch-note {
            border: 1px solid rgba(255, 193, 90, 0.22);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin: 0.55rem 0 1rem;
            background: rgba(255, 193, 90, 0.08);
            color: #ffe2bd;
            font-size: 0.94rem;
            line-height: 1.5;
        }

        .clutch-story-card {
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 24px;
            padding: clamp(1.1rem, 3vw, 1.7rem);
            background:
                radial-gradient(circle at 88% 20%, rgba(255, 138, 42, 0.18), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.085), rgba(255, 255, 255, 0.035));
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        }

        .clutch-story-card h3 {
            margin: 0 0 0.6rem;
            color: var(--clutch-text);
            font-size: clamp(1.25rem, 3vw, 1.8rem);
            font-weight: 900;
            letter-spacing: -0.04em;
        }

        .clutch-story-card p {
            margin: 0;
            color: #dfd5f4;
            font-size: 1rem;
            line-height: 1.7;
        }

        .clutch-step-card {
            min-height: 152px;
            border: 1px solid rgba(255, 255, 255, 0.13);
            border-radius: 22px;
            padding: 1.05rem;
            background:
                linear-gradient(145deg, rgba(123, 60, 255, 0.16), rgba(255, 138, 42, 0.08)),
                rgba(255, 255, 255, 0.045);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.18);
        }

        .clutch-step-number {
            width: 2.2rem;
            height: 2.2rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--clutch-orange), var(--clutch-pink));
            color: white;
            font-weight: 950;
        }

        .clutch-step-title {
            margin-top: 0.85rem;
            color: var(--clutch-text);
            font-size: 1.05rem;
            font-weight: 850;
            line-height: 1.25;
        }

        .clutch-proof-strip {
            display: inline-flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 999px;
            padding: 0.58rem 0.9rem;
            margin-bottom: 0.9rem;
            background: rgba(255, 255, 255, 0.06);
            color: #ffe2bd;
            font-size: 0.92rem;
            font-weight: 750;
        }

        .stButton > button,
        .stDownloadButton > button {
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            border-radius: 999px !important;
            background: linear-gradient(90deg, #ff8a2a, #ff4ecd, #7b3cff) !important;
            color: white !important;
            font-weight: 850 !important;
            letter-spacing: -0.015em;
            box-shadow: 0 12px 34px rgba(123, 60, 255, 0.28);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 44px rgba(255, 138, 42, 0.26);
        }

        label,
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: #f8f7ff !important;
            font-weight: 800 !important;
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span,
        small,
        .stMarkdown small {
            color: #d8d2f0 !important;
        }

        .stTextInput,
        .stFileUploader {
            color: #f8f7ff !important;
        }

        .stTextInput label,
        .stFileUploader label,
        .stTextInput [data-testid="stWidgetLabel"] p,
        .stFileUploader [data-testid="stWidgetLabel"] p {
            color: #f8f7ff !important;
            font-weight: 850 !important;
        }

        .stTextInput input,
        input,
        textarea {
            background: #f8f7ff !important;
            color: #111827 !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            border-radius: 14px !important;
            caret-color: #7b3cff !important;
            font-weight: 650 !important;
        }

        .stTextInput input:focus,
        input:focus,
        textarea:focus {
            border-color: #ff8a2a !important;
            box-shadow: 0 0 0 3px rgba(255, 138, 42, 0.22) !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed rgba(255, 255, 255, 0.28);
            border-radius: 20px;
            padding: 0.7rem;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.065));
            color: #f8f7ff !important;
        }

        [data-testid="stFileUploader"] section {
            border-color: rgba(255, 255, 255, 0.30) !important;
            background: rgba(248, 247, 255, 0.96) !important;
            color: #111827 !important;
        }

        [data-testid="stFileUploader"] section *,
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
            color: #111827 !important;
        }

        [data-testid="stFileUploader"] button {
            background: linear-gradient(90deg, #ff8a2a, #ff4ecd) !important;
            border: 0 !important;
            border-radius: 999px !important;
            color: #ffffff !important;
            font-weight: 850 !important;
        }

        [data-testid="stFileUploader"] button * {
            color: #ffffff !important;
        }

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] [data-testid="stCaptionContainer"],
        [data-testid="stFileUploader"] [data-testid="stCaptionContainer"] * {
            color: #d8d2f0 !important;
        }

        .clutch-upload-card + div label,
        .clutch-upload-card + div [data-testid="stWidgetLabel"] p {
            color: #f8f7ff !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--clutch-border);
        }

        img {
            border-radius: 18px;
        }

        .stAlert {
            border-radius: 18px;
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                padding-top: 1rem;
            }

            .clutch-hero {
                border-radius: 22px;
                padding: 1.25rem;
            }

            .clutch-pills {
                gap: 0.45rem;
            }

            .clutch-pill {
                font-size: 0.78rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Bigger, clearer buttons and controls across the whole app.
    st.markdown(
        """
        <style>
        /* Every button (actions + downloads): larger, bolder, easier to tap. */
        .stButton > button,
        .stDownloadButton > button {
            font-size: 1.2rem !important;
            font-weight: 700 !important;
            padding: 0.95rem 1.6rem !important;
            min-height: 3.3rem !important;
            border-radius: 16px !important;
            letter-spacing: 0.01em;
            line-height: 1.25 !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
        }
        /* Control labels: larger and higher-contrast so they're easy to read. */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            font-size: 1.08rem !important;
            font-weight: 600 !important;
        }
        /* Inputs and dropdowns: bigger text. */
        .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            font-size: 1.08rem !important;
            min-height: 3rem !important;
        }
        /* Sidebar headings + captions: a touch larger. */
        [data-testid="stSidebar"] h2 {
            font-size: 1.45rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            font-size: 0.98rem !important;
        }
        /* Section descriptions: more readable body size. */
        .clutch-section p {
            font-size: 1.08rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="clutch-hero">
            <div class="clutch-hero-content">
                <div class="clutch-kicker">Creator-tech culling suite</div>
                <div class="clutch-logo">Clutch<span>Cull</span></div>
                <h1 class="clutch-headline">AI photo culling for sports photographers</h1>
                <p class="clutch-subheadline">
                    Upload a full shoot, cut blurry frames, remove near-duplicates, and rank the best
                    action shots while preserving your original files for clean export.
                </p>
                <div class="clutch-pills">
                    <span class="clutch-pill">AI-powered</span>
                    <span class="clutch-pill">Built for sports</span>
                    <span class="clutch-pill">Fast workflow</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_landing_hero() -> None:
    st.markdown(
        """
        <section class="clutch-hero">
            <div class="clutch-hero-content">
                <div class="clutch-kicker">Sports photography workflow</div>
                <div class="clutch-logo">Clutch<span>Cull</span></div>
                <div class="clutch-byline">by Gec Shots</div>
                <h1 class="clutch-headline">Cull game-day shoots in minutes, not hours.</h1>
                <p class="clutch-subheadline">
                    Built by Gec Shots for sports photographers who come home with hundreds of frames,
                    near-duplicates, blurry shots, and almost-moments.
                </p>
                <div class="clutch-pills">
                    <span class="clutch-pill">AI-powered</span>
                    <span class="clutch-pill">Built from the sideline</span>
                    <span class="clutch-pill">Export-ready picks</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(label: str, title: str, description: str = "") -> None:
    description_html = f"<p>{description}</p>" if description else ""
    st.markdown(
        f"""
        <div class="clutch-section">
            <span class="clutch-section-label">{label}</span>
            <h2>{title}</h2>
            {description_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hide_sidebar_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none;
        }

        .block-container {
            padding-left: clamp(1rem, 4vw, 3rem);
            padding-right: clamp(1rem, 4vw, 3rem);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_pinned_sidebar_css() -> None:
    """Keep the settings sidebar always visible on the workspace -- it can't be
    collapsed behind the arrow, so the controls are never hidden.
    """
    st.markdown(
        """
        <style>
        /* Force the sidebar open and visible (overrides a collapsed state). */
        [data-testid="stSidebar"] {
            transform: none !important;
            visibility: visible !important;
            min-width: 300px !important;
            width: 300px !important;
            margin-left: 0 !important;
        }
        /* Remove every collapse/expand arrow so it stays pinned open. */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        button[kind="header"][aria-label*="sidebar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_live_stats_snapshot() -> tuple[int, int, int, float] | None:
    try:
        sessions, photos, exports, hours = load_live_stats()
        return int(sessions), int(photos), int(exports), float(hours)
    except Exception:
        return None


def render_workspace_proof_text() -> None:
    stats = get_live_stats_snapshot()

    if stats is None:
        st.caption("Live proof: stats temporarily unavailable")
        return

    sessions, photos, exports, _hours = stats
    st.markdown(
        f"""
        <div class="clutch-proof-strip">
            Live proof: {sessions:,} sessions • {photos:,} photos processed • {exports:,} exports
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_built_from_sideline_card() -> None:
    st.markdown(
        """
        <div class="clutch-story-card">
            <h3>Built from the sideline</h3>
            <p>
                ClutchCull started inside the Gec Shots workflow: football, basketball, baseball,
                hockey, and event galleries with hundreds of frames per shoot. The goal is simple —
                cut the sorting time, keep the strongest images, and get photographers to the edit faster.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    render_section_header(
        "Workflow",
        "How it works",
        "A focused four-step flow from raw game-day batch to export-ready selects.",
    )

    steps = [
        "Upload your shoot",
        "ClutchCull removes blur and near-duplicates",
        "Review the strongest frames",
        "Export your final picks",
    ]
    columns = st.columns(4)

    for index, (column, step) in enumerate(zip(columns, steps), start=1):
        with column:
            st.markdown(
                f"""
                <div class="clutch-step-card">
                    <div class="clutch-step-number">{index}</div>
                    <div class="clutch-step-title">{step}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_cards(metrics: list[tuple[str, str | int | float]]) -> None:
    columns = st.columns(len(metrics))

    for column, (label, value) in zip(columns, metrics):
        with column:
            st.markdown(
                f"""
                <div class="clutch-metric-card">
                    <div class="clutch-metric-label">{label}</div>
                    <div class="clutch-metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_preview_note(extra: str = "") -> None:
    message = (
        f"Showing top {UI_PREVIEW_LIMIT} previews for performance. "
        "All selected files can still be exported."
    )
    if extra:
        message = f"{message} {extra}"

    st.markdown(
        f'<div class="clutch-note">{message}</div>',
        unsafe_allow_html=True,
    )


def normalize_email(email: str) -> str:
    return email.strip().lower()


def calculate_minutes_saved(photos_processed: int, seconds_per_photo: int) -> float:
    manual_minutes = photos_processed * seconds_per_photo / 60
    ai_minutes = manual_minutes * 0.25
    return manual_minutes - ai_minutes


def get_session_id() -> str:
    if "tracking_session_id" not in st.session_state:
        st.session_state.tracking_session_id = str(uuid.uuid4())
    return st.session_state.tracking_session_id


def get_next_batch_id() -> str:
    return uuid.uuid4().hex[:12]


def get_r2_bucket_name() -> str:
    return os.getenv("R2_BUCKET_NAME", "").strip()


def r2_enabled() -> bool:
    required_values = [
        os.getenv("R2_ACCOUNT_ID"),
        os.getenv("R2_ACCESS_KEY_ID"),
        os.getenv("R2_SECRET_ACCESS_KEY"),
        os.getenv("R2_BUCKET_NAME"),
    ]
    return boto3 is not None and all(value for value in required_values)


@st.cache_resource(show_spinner=False)
def get_r2_client():
    if not r2_enabled():
        return None

    account_id = os.getenv("R2_ACCOUNT_ID", "").strip()
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    try:
        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
        )
    except Exception:
        return None


def upload_file_to_r2(local_path: Path, r2_key: str) -> bool:
    client = get_r2_client()
    bucket_name = get_r2_bucket_name()

    if client is None or not bucket_name:
        return False

    try:
        client.upload_file(str(local_path), bucket_name, r2_key)
        return True
    except Exception:
        return False


# Background pool for R2 uploads so saving/analysis never block on the network.
# Created lazily and kept for the process lifetime (survives Streamlit reruns).
_upload_executor: ThreadPoolExecutor | None = None
_upload_executor_lock = threading.Lock()


def _get_upload_executor() -> ThreadPoolExecutor:
    global _upload_executor
    if _upload_executor is None:
        with _upload_executor_lock:
            if _upload_executor is None:
                try:
                    workers = int(os.getenv("CLUTCHCULL_UPLOAD_WORKERS", "4"))
                except ValueError:
                    workers = 4
                _upload_executor = ThreadPoolExecutor(
                    max_workers=max(1, workers),
                    thread_name_prefix="r2-upload",
                )
    return _upload_executor


def _register_pending_uploads(futures: list) -> None:
    """Track in-flight upload futures so we can drain them before cleanup."""
    st.session_state.pending_upload_futures = list(futures)


def wait_for_pending_uploads(timeout: float | None = None) -> None:
    """Block until background R2 uploads for the current batch finish.

    Best-effort and resilient: a failed upload is swallowed (the helper already
    returns False / logs nothing), so a network hiccup never crashes the run.
    Called before any local-original cleanup so we never delete a file that is
    still waiting to be uploaded.
    """
    futures = st.session_state.get("pending_upload_futures", [])
    if not futures:
        return

    for future in futures:
        try:
            future.result(timeout=timeout)
        except Exception:
            pass

    st.session_state.pending_upload_futures = []


def download_file_from_r2(r2_key: str, local_path: Path) -> bool:
    client = get_r2_client()
    bucket_name = get_r2_bucket_name()

    if client is None or not bucket_name:
        return False

    try:
        local_path.parent.mkdir(exist_ok=True)
        client.download_file(bucket_name, r2_key, str(local_path))
        return True
    except Exception:
        return False


def delete_file_from_r2(r2_key: str) -> bool:
    client = get_r2_client()
    bucket_name = get_r2_bucket_name()

    if client is None or not bucket_name:
        return False

    try:
        client.delete_object(Bucket=bucket_name, Key=r2_key)
        return True
    except Exception:
        return False


def cleanup_r2_prefix(prefix: str) -> None:
    client = get_r2_client()
    bucket_name = get_r2_bucket_name()

    if client is None or not bucket_name or not prefix:
        return

    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            objects = [
                {"Key": item["Key"]}
                for item in page.get("Contents", [])
            ]
            if objects:
                client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Fast uploader: browser-side resize + direct-to-R2 presigned uploads.
#
# The old path funneled full-resolution originals (10-25MB each) through the
# app server via st.file_uploader. The fast path instead:
#   1. JS component asks Python for presigned PUT URLs for the chosen files.
#   2. Browser downscales each photo to PREVIEW_MAX_DIMENSION and PUTs the
#      ~0.5MB preview straight to R2, 5 files in parallel.
#   3. Python pulls the small previews from R2 into INPUT_DIR and analyzes
#      them exactly as before.
# Net effect: ~95% fewer bytes uploaded, and none of them pass through the
# (slow) app server. Originals never leave the user's machine; the keeper
# list export tells them which files to pull from their own disk.
# ---------------------------------------------------------------------------

_fast_uploader_component = None


def get_fast_uploader_component():
    global _fast_uploader_component
    if _fast_uploader_component is None and FAST_UPLOADER_DIR.is_dir():
        _fast_uploader_component = components.declare_component(
            "clutchcull_fast_uploader",
            path=str(FAST_UPLOADER_DIR),
        )
    return _fast_uploader_component


@st.cache_resource(show_spinner=False)
def ensure_r2_cors() -> bool:
    """Best-effort: make sure the R2 bucket allows browser PUTs.

    Presigned URLs handle auth; CORS only needs to let the PUT through. If the
    API token lacks bucket-settings permissions this fails quietly and the
    sidebar shows manual instructions instead.
    """
    client = get_r2_client()
    bucket_name = get_r2_bucket_name()
    if client is None or not bucket_name:
        return False

    desired_rule = {
        "AllowedMethods": ["PUT", "GET"],
        "AllowedOrigins": ["*"],
        "AllowedHeaders": ["*"],
        "MaxAgeSeconds": 3600,
    }

    try:
        existing = client.get_bucket_cors(Bucket=bucket_name)
        for rule in existing.get("CORSRules", []):
            if "PUT" in rule.get("AllowedMethods", []):
                return True
    except Exception:
        pass

    try:
        client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration={"CORSRules": [desired_rule]},
        )
        return True
    except Exception:
        return False


def generate_presigned_put_urls(
    file_names: list[str],
    r2_prefix: str,
    expires_in: int = 3600,
) -> dict[str, str]:
    client = get_r2_client()
    bucket_name = get_r2_bucket_name()
    if client is None or not bucket_name:
        return {}

    urls: dict[str, str] = {}
    for file_name in file_names:
        safe_name = Path(file_name).name
        try:
            urls[file_name] = client.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket_name, "Key": f"{r2_prefix}{safe_name}"},
                ExpiresIn=expires_in,
            )
        except Exception:
            continue
    return urls


def fetch_previews_from_r2(file_names: list[str], r2_prefix: str) -> dict[str, str]:
    """Download browser-uploaded previews from R2 into INPUT_DIR for analysis.

    Mirrors save_uploaded_files: clears the working folders, returns the
    name -> r2_key mapping that downstream code uses for re-download fallback.
    Previews are ~0.5MB each, so pulling a whole batch server-side takes
    seconds even on the free tier.
    """
    ensure_directories()
    clear_output_folder(INPUT_DIR)
    clear_output_folder(OUTPUT_DIR)
    clear_output_folder(CANVAS_DIR)

    r2_keys_by_name: dict[str, str] = {}

    def fetch_one(file_name: str) -> None:
        safe_name = Path(file_name).name
        r2_key = f"{r2_prefix}{safe_name}"
        if download_file_from_r2(r2_key, INPUT_DIR / safe_name):
            r2_keys_by_name[safe_name] = r2_key

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="r2-fetch") as pool:
        list(pool.map(fetch_one, file_names))

    return r2_keys_by_name


def render_fast_uploader(raw_upload: bool = False) -> dict | None:
    """Render the JS uploader and drive the presigned-URL handshakes.

    Handles two browser->server exchanges through the one component instance:
      - preview upload: browser resizes + PUTs previews, reports phase "done".
      - full-res keeper upload (at export time): browser PUTs the untouched
        originals of just the selected keepers, reports phase "fullres_done".

    When ``raw_upload`` is True the initial upload skips in-browser resizing
    and PUTs the untouched originals -- used by the canvas tool so exports are
    built from full-resolution, uncompressed source images.

    Returns the persisted preview-upload payload for the current batch (so the
    caller can proceed to analysis), or None until that first upload lands.
    The full-res result is stashed in session_state.fullres_result for the
    export section to consume.
    """
    component = get_fast_uploader_component()
    if component is None:
        return None

    state = st.session_state
    value = component(
        urls=state.get("fast_upload_urls"),
        urls_nonce=state.get("fast_upload_urls_nonce"),
        fullres=state.get("fullres_request"),
        max_dim=PREVIEW_MAX_DIMENSION,
        quality=PREVIEW_JPEG_QUALITY,
        raw_upload=raw_upload,
        key="fast_uploader",
        default=None,
    )

    if isinstance(value, dict):
        phase = value.get("phase")
        nonce = value.get("nonce")

        if phase == "need_urls" and nonce and nonce != state.get("fast_upload_urls_nonce"):
            file_names = [
                Path(name).name
                for name in value.get("files", [])
                if isinstance(name, str) and name
            ]
            if file_names:
                # Drop the previous unprocessed batch's objects (but never the
                # prefix backing currently displayed results -- its keys are
                # still used for preview re-download fallback).
                previous_prefix = state.get("fast_upload_prefix", "")
                if previous_prefix and previous_prefix != state.get("current_r2_prefix", ""):
                    cleanup_r2_prefix(previous_prefix)

                r2_prefix = f"uploads/{get_session_id()}/{get_next_batch_id()}/"
                state.fast_upload_urls = generate_presigned_put_urls(file_names, r2_prefix)
                state.fast_upload_urls_nonce = nonce
                state.fast_upload_prefix = r2_prefix
                state.fast_upload_names = file_names
                st.rerun()

        elif phase == "done" and nonce and nonce == state.get("fast_upload_urls_nonce"):
            # Persist so we keep returning it on later reruns (e.g. after a
            # full-res exchange changes the component's live value).
            state.fast_upload_done = value
            state.fast_upload_done_nonce = nonce

        elif phase == "fullres_done" and nonce:
            request = state.get("fullres_request") or {}
            if nonce == request.get("nonce"):
                state.fullres_result = value

    done = state.get("fast_upload_done")
    if done and state.get("fast_upload_done_nonce") == state.get("fast_upload_urls_nonce"):
        return done

    return None


def fetch_fullres_keepers(file_names: list[str], fullres_prefix: str) -> tuple[int, int]:
    """Pull browser-uploaded full-res keepers from R2 into INPUT_DIR.

    Overwrites the analysis previews at the same filenames, so the existing
    export pipeline (ranked copy + optional canvas) produces full-resolution
    output with no further changes. Returns (ok_count, fail_count).
    """
    if not fullres_prefix or not file_names:
        return 0, len(file_names)

    ensure_directories()

    def fetch_one(file_name: str) -> bool:
        safe_name = Path(file_name).name
        return download_file_from_r2(
            f"{fullres_prefix}{safe_name}", INPUT_DIR / safe_name
        )

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="r2-fullres") as pool:
        for success in pool.map(fetch_one, file_names):
            if success:
                ok += 1
            else:
                fail += 1

    # The overwritten files invalidate their cached metrics (mtime changed);
    # analysis is already done for this batch, but drop them to be safe in
    # case the user reprocesses after exporting.
    clear_metrics_cache()
    return ok, fail


def log_export_completed_once(email: str, total_photos: int, seconds_per_photo: int) -> None:
    current_batch_id = st.session_state.get("current_batch_id")
    counted = list(st.session_state.get("counted_export_batch_ids", []))
    if current_batch_id in counted:
        return
    log_google_form_event(
        "export_completed",
        email=email,
        photos_processed=total_photos,
        exports=1,
        minutes_saved=calculate_minutes_saved(total_photos, seconds_per_photo),
    )
    counted.append(current_batch_id)
    st.session_state.counted_export_batch_ids = counted


def post_google_form_event(
    event_type: str,
    email: str = "",
    photos_processed: int = 0,
    exports: int = 0,
    minutes_saved: float = 0.0,
) -> None:
    if requests is None:
        return

    payload = {
        GOOGLE_FORM_FIELDS["event_type"]: event_type,
        GOOGLE_FORM_FIELDS["email"]: email,
        GOOGLE_FORM_FIELDS["photos_processed"]: str(photos_processed),
        GOOGLE_FORM_FIELDS["exports"]: str(exports),
        GOOGLE_FORM_FIELDS["minutes_saved"]: f"{minutes_saved:.2f}",
        GOOGLE_FORM_FIELDS["session_id"]: get_session_id(),
    }

    try:
        requests.post(GOOGLE_FORM_URL, data=payload, timeout=3)
    except Exception:
        pass


def log_google_form_event(
    event_type: str,
    email: str = "",
    photos_processed: int = 0,
    exports: int = 0,
    minutes_saved: float = 0.0,
) -> None:
    thread = threading.Thread(
        target=post_google_form_event,
        kwargs={
            "event_type": event_type,
            "email": email,
            "photos_processed": photos_processed,
            "exports": exports,
            "minutes_saved": minutes_saved,
        },
        daemon=True,
    )
    thread.start()


def log_session_start_once(email: str) -> None:
    if st.session_state.get("session_start_logged", False):
        return

    log_google_form_event("session_start", email=email)
    st.session_state.session_start_logged = True


@st.cache_data(ttl=60)
def load_live_stats():
    if requests is None or pd is None:
        raise RuntimeError("Live stats dependencies are unavailable.")

    response = requests.get(GOOGLE_SHEET_CSV_URL, timeout=10)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    total_sessions = (df["event_type"] == "session_start").sum()
    total_photos = pd.to_numeric(df["photos_processed"], errors="coerce").fillna(0).sum()
    total_exports = pd.to_numeric(df["exports"], errors="coerce").fillna(0).sum()
    total_hours_saved = pd.to_numeric(df["minutes_saved"], errors="coerce").fillna(0).sum() / 60

    return total_sessions, total_photos, total_exports, total_hours_saved


def render_live_stats() -> None:
    try:
        sessions, photos, exports, hours = load_live_stats()

        render_section_header(
            "Live proof",
            "Impact Dashboard",
            "A real-time look at how many shoots ClutchCull has helped speed up.",
        )
        render_metric_cards(
            [
                ("Sessions", f"{int(sessions):,}"),
                ("Photos processed", f"{int(photos):,}"),
                ("Exports", f"{int(exports):,}"),
                ("Hours saved", f"{round(hours, 1):,}"),
            ]
        )
    except Exception:
        st.info("Live stats temporarily unavailable.")


def ensure_directories() -> None:
    for folder in (INPUT_DIR, OUTPUT_DIR, CANVAS_DIR):
        folder.mkdir(exist_ok=True)


def clear_output_folder(folder: Path) -> None:
    folder.mkdir(exist_ok=True)

    for item in folder.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception:
            pass


def safe_cleanup_after_download_ready() -> None:
    if st.session_state.get("cleanup_after_downloads", False):
        return

    if r2_enabled():
        # Make sure every background upload has landed before we delete the
        # local originals it was reading from.
        wait_for_pending_uploads()
        clear_output_folder(INPUT_DIR)

    st.session_state.cleanup_after_downloads = True


def remove_file_safely(file_path: Path) -> None:
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


def get_image_files(folder: Path) -> list[Path]:
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    )


def load_analysis_preview(image_path: Path) -> Image.Image | None:
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            if img.width > ANALYSIS_MAX_WIDTH:
                scale = ANALYSIS_MAX_WIDTH / img.width
                preview_size = (ANALYSIS_MAX_WIDTH, max(1, int(img.height * scale)))
                img = img.resize(preview_size, RESAMPLING.LANCZOS)
            else:
                img = img.copy()

            return img
    except Exception:
        return None


def load_display_preview(image_path: Path, r2_key: str = "") -> Image.Image | None:
    if not image_path.exists() and r2_key:
        download_file_from_r2(r2_key, image_path)
    return load_analysis_preview(image_path)


def load_metrics_image(image_path: Path) -> Image.Image | None:
    """Load a small RGB image for metric computation only (not display).

    Uses Image.draft() so the JPEG decoder subsamples natively (typically 1/8)
    *during* decode instead of fully decoding a ~32MP frame and then resizing.
    Any residual downscale to METRICS_MAX_WIDTH uses BILINEAR, not LANCZOS:
    metrics (Laplacian variance, Canny, contrast/brightness, pHash) don't need
    the sharper, slower LANCZOS kernel.
    """
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size

            # Hint the decoder toward our target so it can subsample on decode.
            # draft() only applies to JPEG; it's a no-op for other formats.
            if original_width > METRICS_MAX_WIDTH:
                scale = METRICS_MAX_WIDTH / original_width
                target_size = (
                    METRICS_MAX_WIDTH,
                    max(1, int(original_height * scale)),
                )
                img.draft("RGB", target_size)

            img = img.convert("RGB")

            # After draft, the decoded image is >= target (for the ~6960px
            # originals the decoder lands on a 1/4 subsample, ~1740px). Finish
            # the downscale to METRICS_MAX_WIDTH with a cheap BILINEAR pass.
            if img.width > METRICS_MAX_WIDTH:
                scale = METRICS_MAX_WIDTH / img.width
                resize_size = (METRICS_MAX_WIDTH, max(1, int(img.height * scale)))
                img = img.resize(resize_size, RESAMPLING.BILINEAR)
            else:
                img = img.copy()

            return img
    except Exception:
        return None


# Per-image metric cache. The computed metrics (sharpness, detail, contrast,
# brightness/exposure, pHash) are pure functions of the image pixels -- they do
# not depend on the blur/duplicate/top-N sliders or scoring weights. So we cache
# them keyed by (path, mtime). Tweaking a threshold and reprocessing then reuses
# these and only re-runs the cheap filter/score/dedup steps instead of decoding
# every JPEG again. mtime keying means a re-uploaded/overwritten file (new
# mtime) misses the cache and is recomputed.
_METRICS_CACHE: dict[tuple[str, float], dict] = {}
_METRIC_FIELDS = (
    "sharpness",
    "detail_ratio",
    "contrast",
    "brightness_mean",
    "exposure_balance",
    "perceptual_hash",
)


def clear_metrics_cache() -> None:
    _METRICS_CACHE.clear()


def _metrics_cache_key(image_path: Path) -> tuple[str, float]:
    try:
        mtime = image_path.stat().st_mtime
    except OSError:
        mtime = -1.0
    return (str(image_path), mtime)


def compute_image_metrics(image_path: Path, r2_key: str = "") -> PhotoCandidate | None:
    """Decode the image and compute its raw quality metrics (no caching)."""
    try:
        if not image_path.exists() and r2_key:
            download_file_from_r2(r2_key, image_path)

        preview = load_metrics_image(image_path)
        if preview is None:
            return None

        rgb_array = np.array(preview)
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        brightness_mean = float(gray.mean())

        edges = cv2.Canny(gray, 100, 200)
        detail_ratio = float(np.count_nonzero(edges) / edges.size)
        exposure_balance = max(0.0, 1.0 - abs(brightness_mean - 127.5) / 127.5)
        perceptual_hash = imagehash.phash(preview)

        return PhotoCandidate(
            path=image_path,
            sharpness=sharpness,
            detail_ratio=detail_ratio,
            contrast=contrast,
            brightness_mean=brightness_mean,
            exposure_balance=exposure_balance,
            perceptual_hash=perceptual_hash,
            r2_key=r2_key,
        )
    except Exception:
        return None


def load_image_metrics(image_path: Path, r2_key: str = "") -> PhotoCandidate | None:
    """Return metrics for an image, using the (path, mtime) cache when warm.

    A cache hit rebuilds a *fresh* PhotoCandidate from the cached raw metrics so
    that per-run scoring state (score, breakdown, selection reason) never leaks
    between runs.
    """
    # Make sure the file is local before we stat it for the cache key.
    if not image_path.exists() and r2_key:
        download_file_from_r2(r2_key, image_path)

    key = _metrics_cache_key(image_path)
    cached = _METRICS_CACHE.get(key)
    if cached is not None:
        return PhotoCandidate(path=image_path, r2_key=r2_key, **cached)

    candidate = compute_image_metrics(image_path, r2_key=r2_key)
    if candidate is not None:
        _METRICS_CACHE[key] = {
            field: getattr(candidate, field) for field in _METRIC_FIELDS
        }
    return candidate


def normalize_metric(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value - min_value < 1e-9:
        return [0.5 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def add_quality_scores(
    candidates: list[PhotoCandidate],
    weights: dict[str, float],
) -> list[PhotoCandidate]:
    if not candidates:
        return []

    total_weight = sum(weights.values())
    normalized_weights = {
        metric: weight / total_weight
        for metric, weight in weights.items()
    }

    sharpness_scores = normalize_metric([candidate.sharpness for candidate in candidates])
    detail_scores = normalize_metric([candidate.detail_ratio for candidate in candidates])
    contrast_scores = normalize_metric([candidate.contrast for candidate in candidates])
    exposure_scores = [candidate.exposure_balance for candidate in candidates]

    for index, candidate in enumerate(candidates):
        candidate.score_breakdown = {
            "sharpness": sharpness_scores[index],
            "detail": detail_scores[index],
            "contrast": contrast_scores[index],
            "exposure": exposure_scores[index],
        }
        candidate.score = 100 * (
            normalized_weights["sharpness"] * sharpness_scores[index]
            + normalized_weights["detail"] * detail_scores[index]
            + normalized_weights["contrast"] * contrast_scores[index]
            + normalized_weights["exposure"] * exposure_scores[index]
        )
        candidate.selection_reason = build_selection_reason(candidate)

    return candidates


def build_selection_reason(candidate: PhotoCandidate) -> str:
    component_labels = {
        "sharpness": "strong sharpness",
        "detail": "good subject detail",
        "contrast": "clean contrast",
        "exposure": "balanced exposure",
    }

    top_components = sorted(
        candidate.score_breakdown.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:2]

    highlights = [
        component_labels[name]
        for name, value in top_components
        if value >= 0.55
    ]

    if not highlights:
        highlights = ["solid overall image quality"]

    summary = " and ".join(highlights)
    return (
        f"Selected for {summary}. "
        f"Sharpness {candidate.sharpness:.1f}, "
        f"detail {candidate.detail_ratio:.1%}, "
        f"brightness {candidate.brightness_mean:.0f}."
    )


# Short, friendly badge shown on each keeper so the pick feels earned, not
# a black box. Driven by the top-scoring quality component.
_BADGE_BY_COMPONENT = {
    "sharpness": "⚡ Tack-sharp",
    "detail": "🔍 Rich detail",
    "contrast": "🌗 Clean contrast",
    "exposure": "☀️ Well-exposed",
}


def build_selection_badge(candidate: PhotoCandidate) -> str:
    if not candidate.score_breakdown:
        return "✅ Strong pick"
    top_component = max(candidate.score_breakdown.items(), key=lambda item: item[1])
    return _BADGE_BY_COMPONENT.get(top_component[0], "✅ Strong pick")


def filter_blurry_images(
    image_files: list[Path],
    blur_threshold: float,
    r2_keys_by_name: dict[str, str],
    progress_bar=None,
    progress_text=None,
    max_workers: int | None = None,
) -> tuple[list[PhotoCandidate], int, int]:
    total_images = len(image_files)
    if total_images == 0:
        return [], 0, 0

    if max_workers is None or max_workers <= 0:
        max_workers = _default_worker_count()
    # No point spinning up more threads than images.
    max_workers = max(1, min(max_workers, total_images))

    # Compute metrics in parallel. Per-image decode + cv2 work releases the GIL,
    # so threads overlap well. Results are stored by original index so the
    # downstream order is identical to the sequential version -- this keeps
    # scoring/dedup tie-breaking (stable sort on file order) deterministic.
    results: list[PhotoCandidate | None] = [None] * total_images
    processed_count = 0

    def analyze(index: int, image_path: Path):
        return index, load_image_metrics(
            image_path,
            r2_key=r2_keys_by_name.get(image_path.name, ""),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(analyze, index, image_path)
            for index, image_path in enumerate(image_files)
        ]
        # Progress reflects completion count; the main thread owns the Streamlit
        # widgets, so updating them here (not inside workers) is safe.
        for future in as_completed(futures):
            index, candidate = future.result()
            results[index] = candidate
            processed_count += 1

            if progress_text is not None:
                progress_text.text(
                    f"Analyzing image {processed_count} of {total_images}..."
                )
            if progress_bar is not None:
                progress_bar.progress(processed_count / total_images)

    # Apply the blur/unreadable filter in deterministic file order. Blurry
    # shots are RETAINED (not discarded) so the user can rescue any the tool
    # cut too aggressively -- nothing is ever deleted.
    candidates: list[PhotoCandidate] = []
    blurry_candidates: list[PhotoCandidate] = []
    unreadable_count = 0

    for candidate in results:
        if candidate is None:
            unreadable_count += 1
            continue
        if candidate.sharpness < blur_threshold:
            blurry_candidates.append(candidate)
            continue
        candidates.append(candidate)

    gc.collect()

    return candidates, blurry_candidates, unreadable_count


def remove_near_duplicates(
    candidates: list[PhotoCandidate],
    duplicate_threshold: int,
) -> tuple[list[PhotoCandidate], int, list[SimilarPhotoGroup]]:
    kept_candidates: list[PhotoCandidate] = []
    duplicate_groups_by_keeper: dict[str, SimilarPhotoGroup] = {}
    duplicate_count = 0

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate_keeper = None

        for kept in kept_candidates:
            if abs(candidate.perceptual_hash - kept.perceptual_hash) <= duplicate_threshold:
                duplicate_keeper = kept
                break

        if duplicate_keeper is not None:
            duplicate_count += 1
            keeper_name = duplicate_keeper.path.name
            duplicate_groups_by_keeper.setdefault(
                keeper_name,
                SimilarPhotoGroup(keeper=duplicate_keeper),
            ).rejected.append(candidate)
            continue

        kept_candidates.append(candidate)

    duplicate_groups = list(duplicate_groups_by_keeper.values())
    return kept_candidates, duplicate_count, duplicate_groups


def create_white_canvas(
    image_path: Path,
    output_path: Path,
    canvas_width: int,
    canvas_height: int,
    padding: int,
) -> None:
    """Center a photo on a white canvas at the given aspect ratio.

    The passed (canvas_width, canvas_height, padding) describe the *base*
    Instagram size (e.g. 1080x1440). Rendering only at 1080px would downscale
    a 4000-6000px original to roughly a third of its resolution -- visibly
    soft. Instead we scale the whole canvas UP (same aspect ratio) so the photo
    sits at close to its native resolution, capped at CANVAS_MAX_LONG_EDGE.
    Instagram accepts the larger file and downsamples it cleanly, and the
    output is genuinely high resolution for any other use too.
    """
    with Image.open(image_path) as img:
        # Respect embedded EXIF orientation so portraits aren't rotated wrong.
        img = ImageOps.exif_transpose(img).convert("RGB")
        source_width, source_height = img.size

        # Photo area at the base canvas size.
        base_area_width = max(1, canvas_width - (2 * padding))
        base_area_height = max(1, canvas_height - (2 * padding))

        # How much the photo is scaled to fit at the base size. < 1 means the
        # original is being shrunk -- so scale the canvas up by the inverse to
        # recover that lost resolution, capped by the long-edge ceiling.
        base_fit = min(
            base_area_width / source_width,
            base_area_height / source_height,
        )
        scale = 1.0 / base_fit if base_fit < 1 else 1.0
        long_edge = max(canvas_width, canvas_height) * scale
        if long_edge > CANVAS_MAX_LONG_EDGE:
            scale *= CANVAS_MAX_LONG_EDGE / long_edge
        scale = max(scale, 1.0)

        out_width = round(canvas_width * scale)
        out_height = round(canvas_height * scale)
        out_padding = round(padding * scale)

        area_width = max(1, out_width - (2 * out_padding))
        area_height = max(1, out_height - (2 * out_padding))
        # thumbnail() only ever downscales, so a smaller-than-target source is
        # never upscaled (which would just soften it).
        img.thumbnail((area_width, area_height), RESAMPLING.LANCZOS)

        canvas = Image.new("RGB", (out_width, out_height), "white")
        x = (out_width - img.width) // 2
        y = (out_height - img.height) // 2
        canvas.paste(img, (x, y))

        # Max-quality JPEG: quality 98, no chroma subsampling (4:4:4) so fine
        # edges and text stay crisp.
        canvas.save(
            output_path,
            quality=98,
            subsampling=0,
            optimize=True,
        )


def ensure_candidate_local_file(candidate: PhotoCandidate) -> bool:
    if candidate.path.exists():
        return True

    if candidate.r2_key:
        return download_file_from_r2(candidate.r2_key, candidate.path)

    return False


def export_selected_images(
    selected_candidates: list[PhotoCandidate],
    canvas_settings: CanvasSettings,
) -> tuple[list[Path], list[Path]]:
    clear_output_folder(OUTPUT_DIR)
    clear_output_folder(CANVAS_DIR)

    saved_files: list[Path] = []
    canvas_files: list[Path] = []

    for rank, candidate in enumerate(selected_candidates, start=1):
        if not ensure_candidate_local_file(candidate):
            continue

        destination = OUTPUT_DIR / f"{rank:02d}_{candidate.path.name}"
        shutil.copy2(candidate.path, destination)
        saved_files.append(destination)

        if canvas_settings.create_exports:
            canvas_destination = CANVAS_DIR / f"{rank:02d}_canvas.jpg"
            create_white_canvas(
                destination,
                canvas_destination,
                canvas_settings.width,
                canvas_settings.height,
                canvas_settings.padding,
            )
            canvas_files.append(canvas_destination)

        gc.collect()

    return saved_files, canvas_files


def process_images(
    blur_threshold: float,
    duplicate_threshold: int,
    top_n: int,
    scoring_weights: dict[str, float],
    scoring_preset: str,
    r2_keys_by_name: dict[str, str],
    progress_bar=None,
    progress_text=None,
) -> dict:
    ensure_directories()

    image_files = get_image_files(INPUT_DIR)
    candidates, blurry_candidates, unreadable_count = filter_blurry_images(
        image_files,
        blur_threshold,
        r2_keys_by_name=r2_keys_by_name,
        progress_bar=progress_bar,
        progress_text=progress_text,
    )

    if progress_text is not None:
        progress_text.text("Ranking photos and removing near-duplicates...")

    scored_candidates = add_quality_scores(candidates, scoring_weights)
    unique_candidates, duplicate_count, duplicate_groups = remove_near_duplicates(
        scored_candidates,
        duplicate_threshold,
    )
    selected_candidates = unique_candidates[:top_n]

    # Rejected shots the user can rescue, most-likely-wrongly-cut first
    # (closest to the sharpness cutoff shown at the top).
    rejected_candidates = sorted(
        blurry_candidates,
        key=lambda candidate: candidate.sharpness,
        reverse=True,
    )
    for candidate in rejected_candidates:
        candidate.selection_reason = (
            f"Removed as soft/blurry (sharpness {candidate.sharpness:.0f}). "
            "Rescue it if this frame is a keeper."
        )

    if progress_bar is not None:
        progress_bar.progress(1.0)
    if progress_text is not None:
        progress_text.text("Processing complete.")

    gc.collect()

    return {
        "total": len(image_files),
        "blurry_removed": len(blurry_candidates),
        "duplicates_removed": duplicate_count,
        "unreadable_skipped": unreadable_count,
        "selected": len(selected_candidates),
        "selected_candidates": selected_candidates,
        "rejected_candidates": rejected_candidates,
        "duplicate_groups": duplicate_groups,
        "scoring_preset": scoring_preset,
    }


def save_uploaded_files(uploaded_files: list, r2_prefix: str) -> dict[str, str]:
    ensure_directories()
    clear_output_folder(INPUT_DIR)
    clear_output_folder(OUTPUT_DIR)
    clear_output_folder(CANVAS_DIR)

    r2_keys_by_name: dict[str, str] = {}
    use_r2 = r2_enabled()

    if use_r2:
        previous_prefix = st.session_state.get("current_r2_prefix", "")
        if previous_prefix and previous_prefix != r2_prefix:
            cleanup_r2_prefix(previous_prefix)

    upload_executor = _get_upload_executor() if use_r2 else None
    upload_futures = []

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        file_path = INPUT_DIR / file_name

        # Save locally first -- this is the critical path: analysis reads these
        # local files immediately after, so we never wait on the network here.
        with open(file_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())

        if use_r2:
            # The R2 key is deterministic, so assign it now and push the actual
            # upload onto the background pool. Downstream code only uses the key
            # as a fallback to re-download a file that is missing locally, which
            # can't happen until after cleanup (and we drain uploads first).
            r2_key = f"{r2_prefix}{file_name}"
            r2_keys_by_name[file_name] = r2_key
            upload_futures.append(
                upload_executor.submit(upload_file_to_r2, file_path, r2_key)
            )

    if upload_futures:
        _register_pending_uploads(upload_futures)

    return r2_keys_by_name


def make_zip(folder: Path, zip_name: str) -> Path:
    zip_path = Path(zip_name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file in sorted(folder.iterdir()):
            if file.is_file():
                zip_file.write(file, arcname=file.name)

    return zip_path


def render_summary(results: dict) -> None:
    render_section_header(
        "Results",
        "Cull Summary",
        "A quick read on what ClutchCull kept, removed, and shortlisted from this batch.",
    )
    render_metric_cards(
        [
            ("Uploaded", f"{results['total']:,}"),
            ("Blurred removed", f"{results['blurry_removed']:,}"),
            ("Duplicates removed", f"{results['duplicates_removed']:,}"),
            ("Unreadable skipped", f"{results['unreadable_skipped']:,}"),
            ("Auto shortlist", f"{results['selected']:,}"),
        ]
    )


def render_selected_table(candidates: list[PhotoCandidate]) -> None:
    table_rows = []

    for rank, candidate in enumerate(candidates, start=1):
        table_rows.append(
            {
                "Rank": rank,
                "File": candidate.path.name,
                "Score": round(candidate.score, 1),
                "Sharpness": round(candidate.sharpness, 1),
                "Detail %": round(candidate.detail_ratio * 100, 2),
                "Contrast": round(candidate.contrast, 1),
                "Brightness": round(candidate.brightness_mean, 1),
                "Why Selected": candidate.selection_reason,
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_image_grid(candidates: list[PhotoCandidate]) -> None:
    if not candidates:
        return

    render_preview_note()

    preview_candidates = candidates[:UI_PREVIEW_LIMIT]
    columns = st.columns(3)

    for index, candidate in enumerate(preview_candidates):
        with columns[index % 3]:
            preview = load_display_preview(candidate.path, candidate.r2_key)
            st.image(
                preview if preview is not None else str(candidate.path),
                use_container_width=True,
            )
            st.markdown(
                f"**#{index + 1}** &nbsp; `{build_selection_badge(candidate)}`"
            )
            st.caption(f"{candidate.path.name}")
            st.caption(candidate.selection_reason)

    if len(candidates) > UI_PREVIEW_LIMIT:
        st.caption(
            f"{len(candidates) - UI_PREVIEW_LIMIT} more shortlisted files are hidden here "
            "to keep the app stable."
        )


def render_rescue_bin(rejected_candidates: list[PhotoCandidate]) -> None:
    """Show auto-removed (soft/blurry) shots so the user can add any back.

    Directly answers the biggest first-timer fear: "did it throw out my one
    great shot?" Nothing is ever deleted, and rescued frames flow straight
    into the final review and export.
    """
    if not rejected_candidates:
        return

    rescued = set(st.session_state.get("rescued_filenames", set()))
    with st.expander(
        f"🩹 Removed shots — rescue any you want back ({len(rejected_candidates)})"
    ):
        st.caption(
            "These were auto-removed as soft or blurry — but nothing is ever deleted. "
            "Check any frame you want back in your keepers. Closest-to-sharp shown first."
        )
        columns = st.columns(3)
        for index, candidate in enumerate(rejected_candidates[:UI_PREVIEW_LIMIT]):
            with columns[index % 3]:
                preview = load_display_preview(candidate.path, candidate.r2_key)
                st.image(
                    preview if preview is not None else str(candidate.path),
                    use_container_width=True,
                )
                st.caption(f"{candidate.path.name} · sharpness {candidate.sharpness:.0f}")
                is_checked = st.checkbox(
                    "Rescue this shot",
                    value=candidate.path.name in rescued,
                    key=f"rescue__{candidate.path.name}",
                )
                if is_checked:
                    rescued.add(candidate.path.name)
                else:
                    rescued.discard(candidate.path.name)

        if len(rejected_candidates) > UI_PREVIEW_LIMIT:
            st.caption(
                f"{len(rejected_candidates) - UI_PREVIEW_LIMIT} more removed shots are "
                "hidden here for performance."
            )

    st.session_state.rescued_filenames = rescued


def merge_rescued_candidates(
    effective_candidates: list[PhotoCandidate],
    rejected_candidates: list[PhotoCandidate],
) -> list[PhotoCandidate]:
    """Fold any rescued shots into the keeper pool and pre-check them."""
    rescued_names = set(st.session_state.get("rescued_filenames", set()))
    if not rescued_names:
        return effective_candidates

    existing_names = {candidate.path.name for candidate in effective_candidates}
    rescued_candidates = [
        candidate
        for candidate in rejected_candidates
        if candidate.path.name in rescued_names
        and candidate.path.name not in existing_names
    ]
    if not rescued_candidates:
        return effective_candidates

    selected = set(st.session_state.get("selected_filenames", set()))
    selected.update(candidate.path.name for candidate in rescued_candidates)
    st.session_state.selected_filenames = selected
    return effective_candidates + rescued_candidates


def get_duplicate_candidate_lookup(
    duplicate_groups: list[SimilarPhotoGroup],
) -> dict[str, PhotoCandidate]:
    lookup = {}

    for group in duplicate_groups:
        lookup[group.keeper.path.name] = group.keeper

        for candidate in group.rejected:
            lookup[candidate.path.name] = candidate

    return lookup


def apply_similar_photo_swaps(
    candidates: list[PhotoCandidate],
    duplicate_groups: list[SimilarPhotoGroup],
) -> list[PhotoCandidate]:
    swaps = st.session_state.get("similar_photo_swaps", {})
    duplicate_lookup = get_duplicate_candidate_lookup(duplicate_groups)
    effective_candidates = []

    for candidate in candidates:
        replacement_name = swaps.get(candidate.path.name)
        replacement = duplicate_lookup.get(replacement_name)
        effective_candidates.append(replacement or candidate)

    return effective_candidates


def swap_similar_photo(original_name: str, current_name: str, replacement_name: str) -> None:
    swaps = dict(st.session_state.get("similar_photo_swaps", {}))
    selected_filenames = set(st.session_state.get("selected_filenames", set()))

    swaps[original_name] = replacement_name

    if current_name in selected_filenames:
        selected_filenames.remove(current_name)
        selected_filenames.add(replacement_name)

    st.session_state.similar_photo_swaps = swaps
    st.session_state.selected_filenames = selected_filenames
    st.session_state.export_results = None
    st.session_state.export_signature = None


def render_compare_similar_photos(
    candidates: list[PhotoCandidate],
    duplicate_groups: list[SimilarPhotoGroup],
) -> list[PhotoCandidate]:
    preview_candidate_names = {
        candidate.path.name
        for candidate in candidates[:UI_PREVIEW_LIMIT]
    }
    relevant_groups = [
        group
        for group in duplicate_groups
        if group.keeper.path.name in preview_candidate_names
    ]

    if not relevant_groups:
        return candidates

    render_section_header(
        "Review",
        "Compare Similar Photos",
        "Swap a keeper with a near-duplicate when the rejected frame has the better moment.",
    )
    render_preview_note()

    swaps = st.session_state.get("similar_photo_swaps", {})

    for group_index, group in enumerate(relevant_groups):
        current_name = swaps.get(group.keeper.path.name, group.keeper.path.name)
        all_options = [group.keeper] + group.rejected
        current_candidate = next(
            (
                candidate
                for candidate in all_options
                if candidate.path.name == current_name
            ),
            group.keeper,
        )
        current_name = current_candidate.path.name

        st.write(f"Similar set {group_index + 1}")
        columns = st.columns(min(len(all_options), 4))

        for option_index, candidate in enumerate(all_options[:4]):
            with columns[option_index % 4]:
                preview = load_display_preview(candidate.path, candidate.r2_key)
                st.image(
                    preview if preview is not None else str(candidate.path),
                    use_container_width=True,
                )
                label = "Current pick" if candidate.path.name == current_name else "Use this instead"
                st.caption(f"{candidate.path.name} | Score {candidate.score:.1f}")

                if st.button(
                    label,
                    key=f"swap_{group.keeper.path.name}_{candidate.path.name}",
                    disabled=candidate.path.name == current_name,
                ):
                    swap_similar_photo(
                        original_name=group.keeper.path.name,
                        current_name=current_candidate.path.name,
                        replacement_name=candidate.path.name,
                    )
                    st.rerun()

        if len(all_options) > 4:
            selected_option = st.selectbox(
                "More similar rejected files",
                [candidate.path.name for candidate in all_options],
                index=[candidate.path.name for candidate in all_options].index(current_name),
                key=f"similar_select_{group.keeper.path.name}",
            )

            if selected_option != current_name:
                if st.button(
                    "Use selected similar file",
                    key=f"similar_select_button_{group.keeper.path.name}",
                ):
                    swap_similar_photo(
                        original_name=group.keeper.path.name,
                        current_name=current_candidate.path.name,
                        replacement_name=selected_option,
                    )
                    st.rerun()

    if len(candidates) > UI_PREVIEW_LIMIT:
        st.caption(
            f"Compare mode is shown for the first {UI_PREVIEW_LIMIT} shortlisted previews "
            "to keep large batches stable."
        )

    return apply_similar_photo_swaps(candidates, duplicate_groups)


def reset_manual_selection(candidates: list[PhotoCandidate]) -> None:
    for key in list(st.session_state.keys()):
        if (
            key.startswith("select_photo__")
            or key.startswith("similar_select_")
            or key.startswith("rescue__")
        ):
            del st.session_state[key]

    st.session_state.selected_filenames = {
        candidate.path.name
        for candidate in candidates
    }
    st.session_state.rescued_filenames = set()
    st.session_state.export_results = None
    st.session_state.export_signature = None
    st.session_state.similar_photo_swaps = {}


def get_selected_candidates(candidates: list[PhotoCandidate]) -> list[PhotoCandidate]:
    selected_filenames = set(st.session_state.get("selected_filenames", set()))

    return [
        candidate
        for candidate in candidates
        if candidate.path.name in selected_filenames
    ]


def get_export_signature(
    candidates: list[PhotoCandidate],
    canvas_settings: CanvasSettings,
) -> tuple:
    selected_signature = tuple(
        candidate.path.name
        for candidate in get_selected_candidates(candidates)
    )
    canvas_signature = (
        canvas_settings.width,
        canvas_settings.height,
        canvas_settings.padding,
        canvas_settings.create_exports,
    )

    return selected_signature + canvas_signature


def render_manual_selection_grid(candidates: list[PhotoCandidate]) -> list[PhotoCandidate]:
    selected_filenames = set(st.session_state.get("selected_filenames", set()))
    preview_candidates = candidates[:UI_PREVIEW_LIMIT]
    remaining_candidates = candidates[UI_PREVIEW_LIMIT:]
    columns = st.columns(3)

    render_preview_note()

    for index, candidate in enumerate(preview_candidates):
        checkbox_key = f"select_photo__{candidate.path.name}"

        with columns[index % 3]:
            preview = load_display_preview(candidate.path, candidate.r2_key)
            st.image(
                preview if preview is not None else str(candidate.path),
                use_container_width=True,
            )
            st.caption(f"#{index + 1} {candidate.path.name}")
            is_checked = st.checkbox(
                "Keep in final export",
                value=candidate.path.name in selected_filenames,
                key=checkbox_key,
            )

        if is_checked:
            selected_filenames.add(candidate.path.name)
        else:
            selected_filenames.discard(candidate.path.name)

    if remaining_candidates:
        with st.expander(f"Additional selected files ({len(remaining_candidates)})"):
            st.caption(
                "These files are still selectable and exportable. Image previews are hidden "
                "to reduce memory use."
            )

            for index, candidate in enumerate(remaining_candidates, start=UI_PREVIEW_LIMIT + 1):
                checkbox_key = f"select_photo__{candidate.path.name}"
                is_checked = st.checkbox(
                    f"#{index} {candidate.path.name}",
                    value=candidate.path.name in selected_filenames,
                    key=checkbox_key,
                )

                if is_checked:
                    selected_filenames.add(candidate.path.name)
                else:
                    selected_filenames.discard(candidate.path.name)

    st.session_state.selected_filenames = selected_filenames
    return get_selected_candidates(candidates)


def render_downloads(results: dict, create_canvas_exports: bool) -> None:
    output_zip = make_zip(OUTPUT_DIR, "selected_photos.zip")

    with open(output_zip, "rb") as output_handle:
        st.download_button(
            "Download Selected Photos ZIP (slower for large batches)",
            data=output_handle.read(),
            file_name="selected_photos.zip",
            mime="application/zip",
        )

    remove_file_safely(output_zip)

    if create_canvas_exports and results["canvas_files"]:
        canvas_zip = make_zip(CANVAS_DIR, "canvas_photos.zip")

        with open(canvas_zip, "rb") as canvas_handle:
            st.download_button(
                "Download Canvas Photos ZIP (slower for large batches)",
                data=canvas_handle.read(),
                file_name="canvas_photos.zip",
                mime="application/zip",
            )

        remove_file_safely(canvas_zip)

    safe_cleanup_after_download_ready()


def render_landing_view() -> None:
    render_hide_sidebar_css()
    render_landing_hero()
    render_live_stats()

    if st.button("Get Started — it's free", type="primary"):
        st.session_state["view"] = "choose"
        st.rerun()

    render_section_header(
        "Origin",
        "Built for real game-day volume",
        "ClutchCull is shaped around the pressure of getting hundreds of sports frames down to the real keepers.",
    )
    render_built_from_sideline_card()
    render_how_it_works()


def render_compact_brand() -> None:
    st.markdown(
        '<div class="clutch-logo" style="text-align:center;margin:0.5rem 0 0.25rem;">'
        'Clutch<span>Cull</span></div>'
        '<div class="clutch-byline" style="text-align:center;">by Gec Shots</div>',
        unsafe_allow_html=True,
    )


def render_mode_choice() -> None:
    render_hide_sidebar_css()
    render_compact_brand()
    render_section_header(
        "Choose your tool",
        "What would you like to do?",
        "Two tools in one workspace: cull a shoot down to its keepers, or turn photos "
        "into Instagram-ready canvas posts.",
    )

    cull_col, canvas_col = st.columns(2)
    with cull_col:
        st.markdown(
            """
            <div class="clutch-upload-card">
                <h3 class="clutch-upload-title">🏟️ Cull my photos</h3>
                <p class="clutch-upload-copy">
                    Upload a full shoot and let ClutchCull remove blurry frames and
                    near-duplicates, then rank the strongest shots into a keeper shortlist.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start culling", type="primary", use_container_width=True):
            st.session_state["view"] = "cull"
            st.rerun()

    with canvas_col:
        st.markdown(
            """
            <div class="clutch-upload-card">
                <h3 class="clutch-upload-title">📱 Instagram canvas posts</h3>
                <p class="clutch-upload-copy">
                    Drop in your picks and get clean, ready-to-post versions centered on a
                    white canvas — 3:4, 4:5, or 1:1, sized exactly for Instagram.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Create canvas posts", type="primary", use_container_width=True):
            st.session_state["view"] = "canvas"
            st.rerun()

    if st.button("← Back"):
        st.session_state["view"] = "landing"
        st.rerun()


def render_impact_email_capture() -> None:
    """Optional, post-export email ask. Shown only after the user has a result,
    so we never gate value behind an email. Purely for the impact dashboard.
    """
    if st.session_state.get("impact_email_saved"):
        return
    with st.container(border=True):
        st.markdown("#### 📊 Add your shoot to the Impact Dashboard (optional)")
        st.caption(
            "Enter your email to be counted in the community impact stats. "
            "That's the only use — no spam, never shared."
        )
        email_input = st.text_input(
            "Email (optional)",
            value=st.session_state.get("user_email", ""),
            placeholder="you@example.com",
            key="impact_email_input",
        )
        if st.button("Add to Impact Dashboard"):
            st.session_state.user_email = normalize_email(email_input)
            st.session_state.impact_email_saved = True
            if st.session_state.user_email:
                log_google_form_event("email_provided", email=st.session_state.user_email)
                st.success("Thanks — you're counted in the impact stats. 🙌")
            st.rerun()


def render_cull_workspace(email: str) -> None:
    if st.button("← Back to tools"):
        st.session_state["view"] = "choose"
        st.rerun()

    render_pinned_sidebar_css()
    render_compact_brand()
    log_session_start_once(email)

    st.sidebar.markdown("## Culling settings")
    scoring_preset = st.sidebar.selectbox(
        "What kind of shoot is this?",
        list(SCORING_PRESETS.keys()),
        index=list(SCORING_PRESETS.keys()).index("Balanced"),
        help="Sets smart defaults for how ClutchCull picks your best shots.",
    )
    top_n = st.sidebar.slider(
        "How many keepers do you want?",
        1,
        100,
        35,
        1,
        help="The most photos ClutchCull will shortlist for your final review.",
    )
    blur_threshold = st.sidebar.slider(
        "How strict on sharpness?",
        0.0,
        100.0,
        40.0,
        1.0,
        help="Higher = pickier. Removes more soft/blurry shots.",
    )
    duplicate_threshold = st.sidebar.slider(
        "How aggressively to remove near-duplicates?",
        0,
        10,
        2,
        1,
        help="Higher = removes more near-identical burst frames.",
    )
    st.sidebar.caption("All defaults are tuned by Gec Shots — adjust any time.")

    # Fixed metric input (drives the 'hours saved' stat only, not the cull).
    seconds_per_photo = 15
    # Canvas exports live in their own mode now; culling never builds canvases.
    create_canvas_exports = False
    canvas_width, canvas_height, padding = 1080, 1350, 80

    if r2_enabled():
        st.sidebar.success("Cloudflare R2 storage enabled.")
    else:
        st.sidebar.info("Using local temporary storage.")

    # Fast path: browser-side resize + direct-to-R2 upload. Falls back to the
    # classic st.file_uploader when R2 isn't configured or CORS can't be
    # verified (override with CLUTCHCULL_ASSUME_CORS=1 once set manually).
    use_fast_uploader = r2_enabled() and get_fast_uploader_component() is not None
    if use_fast_uploader:
        assume_cors = os.getenv("CLUTCHCULL_ASSUME_CORS", "").strip() == "1"
        if not assume_cors and not ensure_r2_cors():
            use_fast_uploader = False
            st.sidebar.warning(
                "Fast uploads disabled: couldn't verify CORS on the R2 bucket. "
                "Add a CORS rule allowing PUT from any origin in the Cloudflare "
                "dashboard, then set CLUTCHCULL_ASSUME_CORS=1."
            )

    render_section_header(
        "Start here",
        "Drop your shoot here",
        "Photos are optimized in your browser before upload, so big batches move fast. "
        "Your full-resolution originals stay on your computer."
        if use_fast_uploader
        else "Upload the full batch. ClutchCull analyzes optimized previews while preserving originals for export.",
    )
    st.success(
        "🔒 Your original photos are never changed or deleted. ClutchCull only "
        "builds a shortlist of your best shots that you choose what to export."
    )

    uploaded_files = None
    fast_upload_result = None
    upload_card = st.container(border=True)
    with upload_card:
        if use_fast_uploader:
            fast_upload_result = render_fast_uploader()
        else:
            st.markdown(
                """
                <div class="clutch-upload-card">
                    <h3 class="clutch-upload-title">Drop your shoot here</h3>
                    <p class="clutch-upload-copy">
                        ClutchCull analyzes optimized previews while preserving originals for export.
                        Best on desktop for large batches; mobile users can open the sidebar for controls.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Upload game, event, or portrait photos",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
            )

    uploaded_names: list[str] = []
    if use_fast_uploader:
        if fast_upload_result is None:
            st.info("Add a batch of photos to start culling.")
            return
        uploaded_names = [
            Path(name).name for name in fast_upload_result.get("uploaded", [])
        ]
        if not uploaded_names:
            st.warning(
                "The upload didn't complete. Check your connection and try again."
            )
            return
    else:
        if not uploaded_files:
            st.info("Add a batch of photos to start culling.")
            return
        st.write(f"{len(uploaded_files)} files uploaded.")

    if st.button("Process Photos", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)

        if use_fast_uploader:
            r2_prefix = st.session_state.get("fast_upload_prefix", "")
            # Prefix looks like uploads/{session}/{batch}/ -- reuse its batch id
            # so export logging stays deduped per batch.
            prefix_parts = [part for part in r2_prefix.split("/") if part]
            batch_id = prefix_parts[-1] if prefix_parts else get_next_batch_id()
        else:
            batch_id = get_next_batch_id()
            r2_prefix = f"uploads/{get_session_id()}/{batch_id}/"

        with st.spinner("Analyzing resized previews while preserving originals for export..."):
            if use_fast_uploader:
                r2_keys_by_name = fetch_previews_from_r2(uploaded_names, r2_prefix)
            else:
                r2_keys_by_name = save_uploaded_files(uploaded_files, r2_prefix)
            results = process_images(
                blur_threshold=blur_threshold,
                duplicate_threshold=duplicate_threshold,
                top_n=top_n,
                scoring_weights=SCORING_PRESETS[scoring_preset],
                scoring_preset=scoring_preset,
                r2_keys_by_name=r2_keys_by_name,
                progress_bar=progress_bar,
                progress_text=progress_text,
            )

        st.session_state.cull_results = results
        st.session_state.last_photos_processed = results["total"]
        st.session_state.current_batch_id = batch_id
        st.session_state.current_r2_prefix = r2_prefix
        st.session_state.cleanup_after_downloads = False

        log_google_form_event(
            "photos_processed",
            email=email,
            photos_processed=results["total"],
        )
        reset_manual_selection(results["selected_candidates"])

    results = st.session_state.get("cull_results")

    if results is None:
        return

    canvas_settings = CanvasSettings(
        width=canvas_width,
        height=canvas_height,
        padding=padding,
        create_exports=create_canvas_exports,
    )

    st.success("Processing complete. Your shortlist is ready for review.")
    render_summary(results)

    duplicate_groups = results.get("duplicate_groups", [])
    effective_candidates = apply_similar_photo_swaps(
        results["selected_candidates"],
        duplicate_groups,
    )

    if not effective_candidates:
        st.warning(
            "No photos passed the current filters. Try lowering the blur threshold "
            "or increasing the duplicate threshold."
        )
        return

    render_section_header(
        "Shortlist",
        "Your Top Picks",
        "Here are the strongest shots ClutchCull surfaced from your batch. "
        "Scroll down to fine-tune your selection and export.",
    )
    render_image_grid(effective_candidates)

    with st.expander("📊 See the scores behind these picks (optional)"):
        st.caption(
            "Higher score = a stronger keeper, sorted best-first. Ranking uses the "
            f"{results['scoring_preset']} preset after removing blurry and duplicate frames."
        )
        render_selected_table(effective_candidates)

    effective_candidates = render_compare_similar_photos(
        results["selected_candidates"],
        duplicate_groups,
    )

    render_section_header(
        "Safety net",
        "Nothing was deleted",
        "ClutchCull removed soft and duplicate frames, but every original is safe. "
        "Open the bin below to add any removed shot back to your keepers.",
    )
    render_rescue_bin(results.get("rejected_candidates", []))
    effective_candidates = merge_rescued_candidates(
        effective_candidates,
        results.get("rejected_candidates", []),
    )

    render_section_header(
        "Final pass",
        "Final Review",
        "Keep the frames that should make the delivery. Hidden extra files are still selectable and exportable.",
    )
    selected_candidates = render_manual_selection_grid(effective_candidates)
    st.write(f"{len(selected_candidates)} of {len(effective_candidates)} photos selected for export.")

    current_signature = get_export_signature(effective_candidates, canvas_settings)
    export_results = st.session_state.get("export_results")
    export_signature = st.session_state.get("export_signature")

    render_section_header(
        "Delivery",
        "Export Your Picks",
        "Export a full-resolution ZIP of your checked keepers -- only those photos upload, "
        "so it stays fast. Prefer to pull originals from your own catalog? Grab the keeper "
        "list instead.",
    )

    if selected_candidates:
        keeper_names = [candidate.path.name for candidate in selected_candidates]
        keeper_txt = "\n".join(keeper_names) + "\n"
        keeper_csv_lines = ["rank,filename,score,selection_reason"]
        for rank, candidate in enumerate(selected_candidates, start=1):
            reason = candidate.selection_reason.replace('"', "'")
            keeper_csv_lines.append(
                f'{rank},"{candidate.path.name}",{candidate.score:.4f},"{reason}"'
            )
        keeper_csv = "\n".join(keeper_csv_lines) + "\n"

        keeper_columns = st.columns(2)
        with keeper_columns[0]:
            st.download_button(
                "Download Keeper List (.txt)",
                data=keeper_txt,
                file_name="clutchcull_keepers.txt",
                mime="text/plain",
                type="primary",
            )
        with keeper_columns[1]:
            st.download_button(
                "Keeper List with Scores (.csv)",
                data=keeper_csv,
                file_name="clutchcull_keepers.csv",
                mime="text/csv",
            )
        st.caption(
            "The keeper list names every checked photo so you can select the "
            "originals in Lightroom, Finder, or your card reader instantly."
        )

    # With the fast uploader, only ~1800px previews live on the server, so the
    # ZIP would be preview-quality. Instead, on export we ask the browser to
    # upload the untouched originals of just the checked keepers, pull those
    # back, and build the ZIP from them -- full resolution, small upload.
    fullres_mode = use_fast_uploader
    total_photos = st.session_state.get("last_photos_processed", results["total"])

    export_label = (
        "Export Full-Resolution Picks" if fullres_mode else "Export Checked Photos"
    )
    if st.button(export_label, type="primary", disabled=not selected_candidates):
        if fullres_mode:
            keeper_names = [candidate.path.name for candidate in selected_candidates]
            base_prefix = st.session_state.get("current_r2_prefix", "")
            fullres_prefix = f"{base_prefix}fullres/"
            st.session_state.fullres_request = {
                "nonce": uuid.uuid4().hex[:12],
                "urls": generate_presigned_put_urls(keeper_names, fullres_prefix),
            }
            st.session_state.fullres_prefix = fullres_prefix
            st.session_state.fullres_names = keeper_names
            st.session_state.fullres_export_signature = current_signature
            st.session_state.pop("fullres_result", None)
            st.session_state.pop("export_results", None)
            st.rerun()
        else:
            with st.spinner("Exporting checked photos and building ZIP files. Large batches may take longer..."):
                saved_files, canvas_files = export_selected_images(
                    selected_candidates,
                    canvas_settings,
                )
            export_results = {"saved_files": saved_files, "canvas_files": canvas_files}
            log_export_completed_once(email, total_photos, seconds_per_photo)
            st.session_state.export_results = export_results
            st.session_state.export_signature = current_signature
            export_signature = current_signature
            st.success("Checked photos exported.")

    if fullres_mode:
        request = st.session_state.get("fullres_request")
        result = st.session_state.get("fullres_result")
        waiting = request and (not result or result.get("nonce") != request.get("nonce"))

        if waiting:
            st.info(
                "⏳ Uploading full-resolution versions of just your keepers and building the ZIP. "
                "Only the checked photos upload, so this is quick."
            )
        elif request and result and result.get("nonce") == request.get("nonce"):
            with st.spinner("Building your full-resolution ZIP..."):
                fetch_fullres_keepers(
                    st.session_state.get("fullres_names", []),
                    st.session_state.get("fullres_prefix", ""),
                )
                saved_files, canvas_files = export_selected_images(
                    selected_candidates,
                    canvas_settings,
                )
            export_results = {"saved_files": saved_files, "canvas_files": canvas_files}
            log_export_completed_once(email, total_photos, seconds_per_photo)
            export_signature = st.session_state.get("fullres_export_signature", current_signature)
            st.session_state.export_results = export_results
            st.session_state.export_signature = export_signature

            failed = result.get("failed") or []
            if failed:
                st.warning(
                    f"{len(failed)} photo(s) couldn't upload at full resolution and are "
                    "preview-quality in the ZIP. Re-export to retry, or use the keeper list "
                    "to grab those originals from your own files."
                )
            st.success("Full-resolution ZIP ready.")
            st.session_state.pop("fullres_request", None)
            st.session_state.pop("fullres_result", None)

    if export_results and export_signature == current_signature:
        if canvas_settings.create_exports and export_results["canvas_files"]:
            st.write("### Canvas Versions")
            st.caption(
                f"Showing top {UI_PREVIEW_LIMIT} previews for performance. "
                "All canvas files are included in the ZIP."
            )

            canvas_preview_files = export_results["canvas_files"][:UI_PREVIEW_LIMIT]
            canvas_columns = st.columns(3)

            for index, image_path in enumerate(canvas_preview_files):
                with canvas_columns[index % 3]:
                    preview = load_display_preview(image_path)
                    st.image(
                        preview if preview is not None else str(image_path),
                        caption=image_path.name,
                        use_container_width=True,
                    )

            if len(export_results["canvas_files"]) > UI_PREVIEW_LIMIT:
                st.caption(
                    f"{len(export_results['canvas_files']) - UI_PREVIEW_LIMIT} more canvas previews "
                    "are hidden, but still included in the ZIP."
                )

        render_downloads(export_results, canvas_settings.create_exports)
        render_impact_email_capture()

    elif export_results:
        st.info("Your manual selection changed. Export checked photos again to refresh the ZIP files.")


def render_canvas_workspace(email: str) -> None:
    if st.button("← Back to tools"):
        st.session_state["view"] = "choose"
        st.rerun()

    render_pinned_sidebar_css()
    render_compact_brand()
    log_session_start_once(email)

    st.sidebar.markdown("## Canvas Studio")
    st.sidebar.caption("Turn your picks into clean, ready-to-post Instagram canvas versions.")
    ratio_label = st.sidebar.selectbox(
        "Post size",
        list(CANVAS_RATIOS.keys()),
        index=0,
        help="3:4 fills the most feed space for a single image; 4:5 is the classic "
        "portrait post; 1:1 is a square.",
    )
    canvas_width, canvas_height = CANVAS_RATIOS[ratio_label]
    padding = st.sidebar.slider(
        "White border (padding)",
        0,
        100,
        20,
        5,
        help="How much white space to leave around each photo. "
        "Gec Shots recommends 20.",
    )
    st.sidebar.caption("✨ Gec Shots recommends a padding of 20.")

    if r2_enabled():
        st.sidebar.success("Cloudflare R2 storage enabled.")
    else:
        st.sidebar.info("Using local temporary storage.")

    use_fast_uploader = r2_enabled() and get_fast_uploader_component() is not None
    if use_fast_uploader:
        assume_cors = os.getenv("CLUTCHCULL_ASSUME_CORS", "").strip() == "1"
        if not assume_cors and not ensure_r2_cors():
            use_fast_uploader = False

    render_section_header(
        "Instagram Canvas",
        "Create ready-to-post canvas versions",
        f"Each photo is centered on a clean white {ratio_label} canvas, sized exactly "
        "for Instagram. Upload your picks, then download them all as a ZIP.",
    )

    uploaded_files = None
    fast_upload_result = None
    upload_card = st.container(border=True)
    with upload_card:
        if use_fast_uploader:
            # raw_upload: send full-resolution originals so canvases are built
            # from uncompressed source, not the 1800px analysis preview.
            fast_upload_result = render_fast_uploader(raw_upload=True)
        else:
            st.markdown(
                """
                <div class="clutch-upload-card">
                    <h3 class="clutch-upload-title">Drop your photos here</h3>
                    <p class="clutch-upload-copy">
                        These become white-canvas posts ready for Instagram.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_files = st.file_uploader(
                "Upload photos to place on a canvas",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="canvas_file_uploader",
            )

    uploaded_names: list[str] = []
    if use_fast_uploader:
        if fast_upload_result is None:
            st.info("Add photos to create Instagram canvas posts.")
            return
        uploaded_names = [
            Path(name).name for name in fast_upload_result.get("uploaded", [])
        ]
        if not uploaded_names:
            st.warning("The upload didn't complete. Check your connection and try again.")
            return
    else:
        if not uploaded_files:
            st.info("Add photos to create Instagram canvas posts.")
            return
        st.write(f"{len(uploaded_files)} photo(s) ready.")

    if st.button("Create Canvas Posts", type="primary"):
        with st.spinner(f"Building {ratio_label} canvas posts..."):
            if use_fast_uploader:
                r2_prefix = st.session_state.get("fast_upload_prefix", "")
                fetch_previews_from_r2(uploaded_names, r2_prefix)
            else:
                r2_prefix = f"uploads/{get_session_id()}/{get_next_batch_id()}/"
                save_uploaded_files(uploaded_files, r2_prefix)

            clear_output_folder(CANVAS_DIR)
            image_files = get_image_files(INPUT_DIR)
            canvas_files: list[Path] = []
            for index, image_path in enumerate(image_files, start=1):
                destination = CANVAS_DIR / f"{index:02d}_canvas.jpg"
                create_white_canvas(
                    image_path, destination, canvas_width, canvas_height, padding
                )
                canvas_files.append(destination)

        st.session_state.canvas_output_files = [str(path) for path in canvas_files]
        log_google_form_event(
            "canvas_created",
            email=email,
            photos_processed=len(canvas_files),
            exports=1,
        )
        st.success(f"Created {len(canvas_files)} canvas post(s).")

    canvas_files = [Path(path) for path in st.session_state.get("canvas_output_files", [])]
    canvas_files = [path for path in canvas_files if path.exists()]

    if not canvas_files:
        return

    render_section_header(
        "Preview",
        "Your canvas posts",
        f"Showing up to {UI_PREVIEW_LIMIT} previews. Every canvas is included in the ZIP.",
    )
    columns = st.columns(3)
    for index, image_path in enumerate(canvas_files[:UI_PREVIEW_LIMIT]):
        with columns[index % 3]:
            preview = load_display_preview(image_path)
            st.image(
                preview if preview is not None else str(image_path),
                caption=image_path.name,
                use_container_width=True,
            )
    if len(canvas_files) > UI_PREVIEW_LIMIT:
        st.caption(
            f"{len(canvas_files) - UI_PREVIEW_LIMIT} more canvas previews are hidden, "
            "but still included in the ZIP."
        )

    canvas_zip = make_zip(CANVAS_DIR, "instagram_canvas_posts.zip")
    with open(canvas_zip, "rb") as zip_handle:
        st.download_button(
            "Download All Canvas Posts (ZIP)",
            data=zip_handle.read(),
            file_name="instagram_canvas_posts.zip",
            mime="application/zip",
            type="primary",
        )
    remove_file_safely(canvas_zip)

    render_impact_email_capture()


def main() -> None:
    ensure_directories()

    st.set_page_config(
        page_title="Gec Shots ClutchCull",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    view = st.session_state.get("view", "landing")

    if view == "landing":
        render_landing_view()
    elif view == "choose":
        render_mode_choice()
    elif view == "canvas":
        render_canvas_workspace(st.session_state.get("user_email", ""))
    elif view == "cull":
        render_cull_workspace(st.session_state.get("user_email", ""))
    else:
        st.session_state["view"] = "landing"
        render_landing_view()


if __name__ == "__main__":
    main()
