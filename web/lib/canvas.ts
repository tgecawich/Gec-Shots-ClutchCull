export const CANVAS_RATIOS: Record<string, [number, number]> = {
  "3:4": [1080, 1440],
  "4:5": [1080, 1350],
  "1:1": [1080, 1080],
};

// Center a full-resolution photo on a white canvas at the given IG ratio.
// Rendered at 2x for crisp output (matches the Python engine).
export async function makeCanvas(
  file: File,
  ratio: string,
  padding: number,
  scale = 2
): Promise<Blob> {
  const [baseW, baseH] = CANVAS_RATIOS[ratio] || CANVAS_RATIOS["3:4"];
  const W = baseW * scale;
  const H = baseH * scale;
  const pad = padding * scale;
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" }).catch(
    () => createImageBitmap(file)
  );
  const availW = Math.max(1, W - 2 * pad);
  const availH = Math.max(1, H - 2 * pad);
  const fit = Math.min(availW / bmp.width, availH / bmp.height);
  const dw = Math.round(bmp.width * fit);
  const dh = Math.round(bmp.height * fit);
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, (W - dw) / 2, (H - dh) / 2, dw, dh);
  bmp.close();
  return new Promise((res, rej) =>
    canvas.toBlob((b) => (b ? res(b) : rej(new Error("canvas failed"))), "image/jpeg", 0.95)
  );
}
