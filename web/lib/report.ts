// Client-side shareable "Cull Report" card (portrait 1080x1350, rendered 2x).
const GOLD = "#ffd66b";
const WHITE = "#f8f4ff";
const MUTED = "#d9cff0";
const ORANGE = "#ff8a2a";
const PINK = "#ff4ecd";

export async function makeCullReport(opts: {
  shotsIn: number;
  keepers: number;
  hoursSaved: number;
  filtered: number;
  elapsedSeconds: number;
  keeperFiles: File[];
  handle?: string;
}): Promise<Blob> {
  const S = 2;
  const W = 1080 * S;
  const H = 1350 * S;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d")!;
  try {
    await (document as any).fonts?.ready;
  } catch {}

  const bg = ctx.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, "#2a1152");
  bg.addColorStop(0.5, "#5a24c0");
  bg.addColorStop(1, "#ff8a2a");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);
  ctx.textAlign = "center";
  const cx = W / 2;
  const font = (px: number, w = 800) => `${w} ${px * S}px 'Plus Jakarta Sans', sans-serif`;

  // Wordmark
  ctx.font = font(84);
  const clutchW = ctx.measureText("Clutch").width;
  const cullW = ctx.measureText("Cull").width;
  const startX = cx - (clutchW + cullW) / 2;
  ctx.textAlign = "left";
  ctx.fillStyle = WHITE;
  ctx.fillText("Clutch", startX, 130 * S);
  const grad = ctx.createLinearGradient(startX + clutchW, 0, startX + clutchW + cullW, 0);
  grad.addColorStop(0, ORANGE);
  grad.addColorStop(1, PINK);
  ctx.fillStyle = grad;
  ctx.fillText("Cull", startX + clutchW, 130 * S);
  ctx.textAlign = "center";

  ctx.font = font(34, 700);
  ctx.fillStyle = GOLD;
  ctx.fillText(`made by ${opts.handle || "@gec.shots"}`, cx, 190 * S);

  // Hero X -> Y
  ctx.font = font(150);
  const a = String(opts.shotsIn);
  const b = String(opts.keepers);
  const arrow = "  →  ";
  const aw = ctx.measureText(a).width;
  const arw = ctx.measureText(arrow).width;
  const bw = ctx.measureText(b).width;
  const gx = cx - (aw + arw + bw) / 2;
  ctx.textAlign = "left";
  ctx.fillStyle = WHITE;
  ctx.fillText(a, gx, 380 * S);
  ctx.fillStyle = MUTED;
  ctx.fillText(arrow, gx + aw, 380 * S);
  const g2 = ctx.createLinearGradient(gx + aw + arw, 0, gx + aw + arw + bw, 0);
  g2.addColorStop(0, ORANGE);
  g2.addColorStop(1, PINK);
  ctx.fillStyle = g2;
  ctx.fillText(b, gx + aw + arw, 380 * S);
  ctx.textAlign = "center";
  ctx.font = font(30, 700);
  ctx.fillStyle = MUTED;
  ctx.fillText("SHOTS", gx + aw / 2, 430 * S);
  ctx.fillStyle = ORANGE;
  ctx.fillText("KEEPERS", gx + aw + arw + bw / 2, 430 * S);

  // Stats
  const m = Math.max(0, Math.round(opts.elapsedSeconds));
  ctx.fillStyle = WHITE;
  ctx.font = font(44);
  ctx.fillText(`${opts.hoursSaved.toFixed(1)} hours saved`, cx, 560 * S);
  ctx.fillStyle = MUTED;
  ctx.font = font(34, 600);
  ctx.fillText(`${opts.filtered} blurry & duplicates filtered`, cx, 615 * S);
  ctx.fillStyle = WHITE;
  ctx.font = font(44);
  ctx.fillText(`culled in ${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`, cx, 675 * S);

  // Keeper thumbnails
  const thumbs = opts.keeperFiles.slice(0, 3);
  if (thumbs.length) {
    ctx.font = font(28, 700);
    ctx.fillStyle = MUTED;
    ctx.fillText("T O P   K E E P E R S", cx, 790 * S);
    const margin = 80 * S;
    const gap = 26 * S;
    const cw = (W - 2 * margin - (thumbs.length - 1) * gap) / thumbs.length;
    const chh = cw * 1.15;
    const ty = 830 * S;
    let tx = margin;
    for (const f of thumbs) {
      try {
        const bmp = await createImageBitmap(f, { imageOrientation: "from-image" } as any).catch(() => createImageBitmap(f));
        const scale = Math.max(cw / bmp.width, chh / bmp.height);
        const dw = bmp.width * scale;
        const dh = bmp.height * scale;
        ctx.save();
        roundRect(ctx, tx, ty, cw, chh, 18 * S);
        ctx.clip();
        ctx.drawImage(bmp, tx + (cw - dw) / 2, ty + (chh - dh) / 2, dw, dh);
        ctx.restore();
        bmp.close();
      } catch {}
      tx += cw + gap;
    }
  }

  ctx.font = font(26, 600);
  ctx.fillStyle = MUTED;
  ctx.fillText("made with ClutchCull", cx, (H / S - 60) * S);

  return new Promise((res, rej) =>
    canvas.toBlob((bl) => (bl ? res(bl) : rej(new Error("report failed"))), "image/jpeg", 0.95)
  );
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
