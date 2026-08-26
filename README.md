# ClutchCull

**Free photo culling for photographers. Drop in a full shoot, get your sharpest keepers back in minutes.**

🔗 **[gec-shots-clutch-cull.vercel.app](https://gec-shots-clutch-cull.vercel.app)**
No sign-up, nothing to install.

Built by [@gec.shots](https://instagram.com/gec.shots), a student-athlete sports photographer who was losing two hours after every game deciding which 35 of 500 photos were worth keeping.

> **122,000+ photos culled · 900+ photographers · ~228 hours handed back**
> The counters on the site read from the same usage log, so they never drift ahead of reality.

---

## The problem

A single game produces 400–600 frames. Most are near-identical burst shots, and a meaningful slice are soft because the moment happened faster than the autofocus did. Finding the keepers means clicking through images one at a time, and it's the part of the job nobody enjoys.

ClutchCull does that pass automatically: it locates the athlete in each frame, discards blurry frames and duplicate burst sequences, and ranks what's left with a score and a reason.

## What makes it different

Most culling tools measure sharpness across the **whole frame**. That's why a blurry player standing in front of a crisp scoreboard scores well: the background carries the score.

ClutchCull finds the **subject** first (person detection, falling back to faces, falling back to a centre estimate) and judges the shot on whether *they* are sharp. If focus landed on the background instead of the athlete, the frame is demoted no matter how good the rest of the image looks.

---

## Architecture

```
┌─ web/  Next.js 14 · TypeScript · Tailwind ─────────── Vercel ─┐
│  Browser: extract RAW previews, resize to 1800px,             │
│  upload in chunks. Exports (ZIP, XMP, canvas) are local.      │
└──────────────────────────┬────────────────────────────────────┘
                           │  HTTPS · multipart + JSON
┌──────────────────────────▼────────────────────────────────────┐
│  api/  Python · FastAPI · Docker ───────────────────── Render │
│  OpenCV metrics · YOLOX person detection · YuNet faces        │
│  · perceptual-hash duplicate detection                        │
└───────────────────────────────────────────────────────────────┘
```

**Full-resolution originals never leave the device.** Only downscaled analysis copies are uploaded, and every export is generated in the browser. A 5 GB shoot uploads about 91 MB.

---

## Engineering notes

The decisions below were the non-obvious ones.

### Blur detection is resolution-sensitive, and that nearly broke the product

A photo the photographer would have deleted instantly (a motion-blurred catcher) was ranking **#1**. Two plausible fixes (rebalancing the score weights, then the subject-vs-background logic) both failed.

Measuring instead of guessing found the real cause. Taking one real camera file, blurring it, then testing how well the algorithm could separate sharp from soft at different analysis resolutions:

| Analysis width | Sharp vs. blurred separation |
| -------------- | ---------------------------- |
| 1200 px *(what was in use)* | **4.1×** |
| 2400 px | 28× |
| Native (6712 px) | **990×** |

Downscaling is itself a low-pass filter. The pipeline was shrinking the blur away *before* measuring it. Analysis width moved to 1800 px: the point where separation is ~11× and memory still fits a 512 MB instance.

The second cause was subject location: a helmeted athlete facing away has no detectable face, so the engine fell back to "the subject is probably the middle of the frame" and measured the crisp dirt behind him. That is what motivated real person detection.

### RAW support without RAW decoding

Decoding sensor data would mean uploading 25–50 MB per frame and running libraw server-side. Instead, ClutchCull reads the **JPEG preview every camera already embeds** in its RAW file. That is the image the camera shows on its own LCD.

Extraction is format-agnostic. Rather than parsing each vendor's layout (CR2/NEF/ARW/DNG are TIFF-based; CR3 is an ISO-BMFF container), it scans for JPEG start/end markers and takes the largest valid image, skipping the ~160 px thumbnail. One code path covers every format.

Measured on 16 real Canon CR3 files (~40 MB each):

- 16/16 previews extracted, each **6960 × 4640** (full sensor resolution)
- ~60 ms to scan a 40 MB file
- On RAW+JPEG pairs, metrics from the CR3 preview matched the camera's own JPEG within **3.6%**, so RAW-only shooters need no paired JPEG

### Score once, rank instantly

Image analysis is expensive; ranking is arithmetic. The two are separate endpoints: `/score-upload` computes per-photo metrics once and the browser caches them, then `/rank` turns those into keepers in ~1 ms. Changing the keeper count or preset re-ranks immediately with no re-upload.

### Running real ML inside 512 MB

The detector was originally stored per-thread, meaning every concurrent request loaded its own copy of the network, a guaranteed OOM under normal chunked uploads. It is now a single shared instance behind an inference lock; calls serialise, which costs nothing on a half-core box. Input size, worker count and analysis width are environment-tunable, with a kill switch that degrades to face detection rather than failing.

### Testing against real files, twice over

Two bugs were invisible to synthetic tests and obvious the moment real camera files were used:

1. **EXIF orientation was never applied during analysis.** Every vertically shot frame was analysed lying on its side, and both detectors are trained on upright images. This was silently degrading subject detection on all portrait-orientation work.
2. **Min-max score normalisation collapsed under outliers.** One frame of chain-link fence or crowd texture set the maximum and pushed every other photo's score toward zero, capping a 500-photo shoot at ~9/100. Fixed with percentile-clipped normalisation.

---

## Features

- **Subject-aware culling**, ranks on whether the athlete is sharp, not the frame
- **RAW support**, CR3, CR2, NEF, ARW, DNG, RAF, ORF, RW2 and more
- **Burst grouping**, near-identical sequences collapsed, with side-by-side swapping
- **Nothing is deleted**, every rejected frame is one tap from being rescued
- **Lightroom hand-off**, export `.xmp` sidecars so keepers arrive pre-starred
- **Transparent scoring**, every pick shows why: sharp subject, rich detail, well-exposed
- **Instagram canvas**, padded, ready-to-post versions of your picks

## Exports

| Export | Contents |
| ------ | -------- |
| Full-resolution ZIP | Untouched originals, keepers only, rank-prefixed (auto-split for large sets) |
| Lightroom `.xmp` | ~560 bytes per keeper, named to pair with the original file |
| Scores `.csv` | Rank, filename, score and the full metric breakdown |
| Filenames `.txt` | Plain list for filtering elsewhere |
| Cull report | Shareable summary card |

---

## Running locally

**Requirements:** Python 3.11+, Node 18+

```bash
# API
cd api
pip install -r requirements.txt
uvicorn main:app --port 7860
curl localhost:7860/health          # reports the live culling config

# Web (separate terminal)
cd web
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:7860" > .env.local
npm run dev                         # localhost:3000
```

ML models download automatically on first run, and are baked into the Docker image for deployment.

### Configuration

All optional. Defaults are tuned for a 512 MB instance.

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `CLUTCHCULL_METRICS_WIDTH` | `1800` | Analysis resolution. Must match the browser's resize width |
| `CLUTCHCULL_YOLOX_SIZE` | `320` | Person-detector input size |
| `CLUTCHCULL_WORKERS` | `1` | Parallel image workers |
| `CLUTCHCULL_PERSON` | `1` | `0` disables person detection (falls back to faces) |
| `CLUTCHCULL_FOCUS_MIN` / `_FLOOR` | `0.6` / `0.25` | Focus-miss sensitivity |
| `CLUTCHCULL_SHARP_GATE_SPAN` | `6.0` | How far above the blur floor earns full credit |

## Project layout

```
api/         FastAPI service
  engine.py    culling engine, metrics, detection, scoring, dedup
  main.py      HTTP layer
web/         Next.js app
  app/         landing page + culling workspace
  lib/         raw.ts (RAW previews) · resize.ts · xmp.ts · api.ts · canvas.ts
app.py       original Streamlit prototype, kept for reference
```

## Tech stack

**Frontend**, TypeScript · React 18 · Next.js 14 · Tailwind CSS · Vercel
**Backend**, Python · FastAPI · Docker · Render
**Vision**, OpenCV · YOLOX (person detection) · YuNet (faces) · imagehash · NumPy · Pillow

---

## License

MIT, see [LICENSE](LICENSE).
