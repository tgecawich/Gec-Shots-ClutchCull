"""ClutchCull API — FastAPI backend for the Next.js front-end.

Exposes the culling engine + canvas export over HTTP. Browsers upload photos
directly to Cloudflare R2 via presigned URLs; this service pulls them, runs
the engine, and returns results. Deployed as a Hugging Face Docker Space.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import engine

app = FastAPI(title="ClutchCull API", version="2.0.0")

# Frontend origins allowed to call the API. Override with CLUTCHCULL_ORIGINS
# (comma-separated) in the Space secrets once the Vercel domain is known.
_origins = os.getenv("CLUTCHCULL_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Cloudflare R2 ---------------------------------------------------------
def _r2():
    account = os.getenv("R2_ACCOUNT_ID", "").strip()
    if not account:
        return None
    try:
        return boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
    except Exception:
        return None


def _bucket() -> str:
    return os.getenv("R2_BUCKET_NAME", "").strip()


# --- request models --------------------------------------------------------
class PresignRequest(BaseModel):
    files: list[str]
    prefix: str | None = None


class CullRequest(BaseModel):
    prefix: str
    files: list[str]
    blur_threshold: float = 40.0
    duplicate_threshold: int = 2
    top_n: int = 35
    preset: str = "Balanced"


class CanvasRequest(BaseModel):
    prefix: str
    files: list[str]
    ratio: str = "3:4"
    padding: int = 20


# --- endpoints -------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "r2": _r2() is not None and bool(_bucket()), "presets": list(engine.SCORING_PRESETS)}


@app.post("/presign")
def presign(req: PresignRequest):
    client, bucket = _r2(), _bucket()
    if client is None or not bucket:
        raise HTTPException(503, "R2 storage not configured")
    prefix = req.prefix or f"uploads/{uuid.uuid4().hex[:12]}/"
    urls = {}
    for name in req.files:
        safe = Path(name).name
        try:
            urls[name] = client.generate_presigned_url(
                "put_object", Params={"Bucket": bucket, "Key": f"{prefix}{safe}"}, ExpiresIn=3600
            )
        except Exception:
            continue
    return {"prefix": prefix, "urls": urls}


def _download(prefix: str, names: list[str], dest: Path) -> list[Path]:
    client, bucket = _r2(), _bucket()
    if client is None or not bucket:
        raise HTTPException(503, "R2 storage not configured")
    dest.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> Path | None:
        safe = Path(name).name
        local = dest / safe
        try:
            client.download_file(bucket, f"{prefix}{safe}", str(local))
            return local
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        return [p for p in pool.map(one, names) if p is not None]


@app.post("/cull")
def cull(req: CullRequest):
    with tempfile.TemporaryDirectory() as tmp:
        paths = _download(req.prefix, req.files, Path(tmp))
        if not paths:
            raise HTTPException(400, "No photos could be retrieved")
        return engine.cull(
            paths,
            blur_threshold=req.blur_threshold,
            duplicate_threshold=req.duplicate_threshold,
            top_n=req.top_n,
            preset=req.preset,
        )


@app.post("/canvas")
def canvas(req: CanvasRequest):
    if req.ratio not in engine.CANVAS_RATIOS:
        raise HTTPException(400, f"Unknown ratio {req.ratio}")
    w, h = engine.CANVAS_RATIOS[req.ratio]
    client, bucket = _r2(), _bucket()
    out_prefix = f"{req.prefix}canvas/"
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        paths = _download(req.prefix, req.files, tmp_path)
        for src in paths:
            out = tmp_path / f"canvas_{src.name}"
            try:
                engine.create_white_canvas(src, out, w, h, req.padding)
                key = f"{out_prefix}{src.name}"
                client.upload_file(str(out), bucket, key)
                url = client.generate_presigned_url(
                    "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=3600
                )
                results.append({"filename": src.name, "url": url})
            except Exception:
                continue
    return {"prefix": out_prefix, "canvases": results}
