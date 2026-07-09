---
title: ClutchCull
emoji: 📸
colorFrom: purple
colorTo: red
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
---

# ClutchCull

AI photo culling for sports photographers, by Gec Shots. Upload a shoot, cut
blurry frames, remove near-duplicates, and rank the strongest action shots —
your full-resolution originals never leave your computer.

The block above (the `---` front matter) is required by Hugging Face Spaces:
it tells the Space to run as a Streamlit app from `app.py`. Leave it at the
very top of the file.

## Fast uploads (browser-side optimization + direct-to-R2)

When Cloudflare R2 is configured, ClutchCull skips the slow "push originals
through the app server" path entirely:

1. The uploader component (`components/fast_uploader/index.html`) downscales
   each photo **in the browser** to max 1800px JPEG (~0.5MB instead of
   10–25MB — roughly 95% fewer bytes).
2. Previews are `PUT` **directly to R2** with presigned URLs, 5 in parallel,
   so upload bytes never touch the app server.
3. The server pulls the small previews from R2 and runs the same analysis
   as before.
4. After culling, the **keeper list** (.txt/.csv) names the selected photos so
   the photographer grabs the originals from their own disk or card — no
   original ever needs uploading.

If R2 env vars are missing, the app falls back to the classic
`st.file_uploader` path automatically.

### Required setup

Environment variables (already used for R2 mirroring):
`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`.

The R2 bucket needs a CORS rule that allows browser `PUT`s. The app tries to
set this automatically at startup; if the API token lacks bucket-settings
permission, add it manually in the Cloudflare dashboard
(R2 → your bucket → Settings → CORS policy):

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["PUT", "GET"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

Then set `CLUTCHCULL_ASSUME_CORS=1` so the app skips the automatic check.
(Presigned URLs handle authentication — the wildcard origin only lets the
browser deliver an already-signed request.)
