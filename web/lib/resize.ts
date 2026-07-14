// Downscale a photo in the browser before upload (keeps big batches fast).
export async function resizeImage(
  file: File,
  maxDim = 1800,
  quality = 0.82
): Promise<File> {
  try {
    const bitmap = await createImageBitmap(file).catch(() => createImageBitmap(file));
    const scale = Math.min(1, maxDim / Math.max(bitmap.width, bitmap.height));
    if (scale >= 1 && /jpe?g/i.test(file.type) && file.size < 2_000_000) {
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
