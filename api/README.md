---
title: ClutchCull API
emoji: 📸
colorFrom: purple
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Culling + canvas engine API for the ClutchCull web app
---

# ClutchCull API

FastAPI backend that powers the ClutchCull web front-end. Wraps the culling
engine (subject-aware sharpness via YuNet, near-duplicate removal, weighted
scoring) and white-canvas export.

## Endpoints
- `GET /health` — status + whether R2 is configured
- `POST /presign` — presigned R2 PUT URLs for browser uploads
- `POST /cull` — analyze a batch, return ranked keepers
- `POST /canvas` — build Instagram-ready white-canvas posts

## Required secrets (Space → Settings → Variables and secrets)
`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`,
and `CLUTCHCULL_ORIGINS` (comma-separated allowed frontend origins).

The YuNet face model downloads automatically at runtime.
