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
// Analysis only needs enough resolution to rank sharpness/faces/detail — the
// final keeper EXPORT always uses the untouched original, so we can send a
// much smaller copy to the server: faster upload + faster face detection.
export async function resizeImage(
  file: File,
  maxDim = 1200, // the server analyzes at 1200px anyway — sending bigger is pure waste
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
