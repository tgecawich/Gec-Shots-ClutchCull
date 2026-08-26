// Run an async op over a list with a fixed concurrency cap. Decoding 300
// images at once thrashes memory and stalls the tab; a small pool is faster.
export async function mapLimit<T, R>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      results[i] = await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

// Downscale a photo in the browser before upload (keeps big batches fast).
// Analysis only needs enough resolution to rank sharpness/faces/detail: the // final keeper EXPORT always uses the untouched original, so we can send a
// much smaller copy to the server: faster upload + faster face detection.
// 1800px, NOT smaller: blur detection is highly resolution-sensitive. Measured
// on real shoots, sharp-vs-soft separation is only ~4x at 1200px, ~11x at 1800px
// and ~28x at 2400px, downscaling is itself a blur filter, so shrinking too far
// hides the very thing we're testing for. 1800 is the sweet spot that fits a
// 512MB API box. Must match the API's CLUTCHCULL_METRICS_WIDTH.
export async function resizeImage(
  file: File,
  maxDim = 1800,
  quality = 0.72
): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file);
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    if (scale >= 1 && /jpe?g/i.test(file.type) && file.size < 900_000) {
      bitmap.close();
      return file;
    }
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const h = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();
    const blob: Blob = await new Promise((res, rej) =>
      canvas.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/jpeg", quality)
    );
    // Keep the original filename so results map cleanly back to thumbnails.
    return new File([blob], file.name, { type: "image/jpeg" });
  } catch {
    return file;
  }
}

// ── RAW handling ──────────────────────────────────────────────────────────
// RAW files can't be drawn by the browser, so before anything else we pull out
// the JPEG preview the camera already embedded. Everything downstream (analysis
// and thumbnails) then behaves exactly as it does for a JPEG shoot.

/** Resize a Blob (not just a File) to maxDim, returning a JPEG blob. */
async function resizeBlob(blob: Blob, maxDim: number, quality: number): Promise<Blob> {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
  const w = Math.max(1, Math.round(bitmap.width * scale));
  const h = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();
  return await new Promise<Blob>((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/jpeg", quality)
  );
}

/**
 * Prepare any file for analysis. RAW files yield their embedded preview,
 * resized like any other photo. The RAW itself is never uploaded.
 * Keeps the ORIGINAL filename so results map back to the user's RAW files.
 */
export async function prepareForAnalysis(file: File): Promise<File> {
  const { isRaw, extractRawPreview } = await import("./raw");
  if (!isRaw(file)) return resizeImage(file);
  const preview = await extractRawPreview(file);
  if (!preview) return file; // no preview found; server will skip it as unreadable
  const small = await resizeBlob(preview, 1800, 0.72);
  return new File([small], file.name, { type: "image/jpeg" });
}

/**
 * A small displayable thumbnail. For RAW we must extract and downscale, because
 * an object URL pointing at a .CR2 renders nothing. Kept small on purpose: the
 * embedded previews are often full-resolution and holding many is expensive.
 */
export async function makeThumbUrl(file: File, maxDim = 520): Promise<string | null> {
  const { isRaw, extractRawPreview } = await import("./raw");
  if (!isRaw(file)) {
    try { return URL.createObjectURL(file); } catch { return null; }
  }
  try {
    const preview = await extractRawPreview(file);
    if (!preview) return null;
    const thumb = await resizeBlob(preview, maxDim, 0.7);
    return URL.createObjectURL(thumb);
  } catch { return null; }
}
