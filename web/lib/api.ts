export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

export type Dupe = { filename: string; score: number; badge: string };

export type Keeper = {
  filename: string;
  score: number;
  badge: string;
  breakdown: Record<string, number>;
  duplicates?: Dupe[];
};

export type CullResult = {
  total: number;
  blurry_removed: number;
  duplicates_removed: number;
  unreadable_skipped: number;
  keepers: Keeper[];
  rejected: string[];
};

export type CullSettings = {
  preset: string;
  top_n: number;
  blur: number;
  dupes: number;
};

// Wake the Render dyno (free tier sleeps after ~15 min idle). Fire-and-forget
// so the server is warm by the time the user actually hits "Cull".
export function warmApi(): void {
  try {
    fetch(`${API_URL}/health`, { method: "GET", cache: "no-store" }).catch(() => {});
  } catch {
    /* best-effort */
  }
}

export async function cullUpload(files: File[], s: CullSettings): Promise<CullResult> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  fd.append("preset", s.preset);
  fd.append("top_n", String(s.top_n));
  fd.append("blur_threshold", String(s.blur));
  fd.append("duplicate_threshold", String(s.dupes));
  const res = await fetch(`${API_URL}/cull-upload`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Culling failed (${res.status})`);
  return res.json();
}

// Per-image metrics computed once by the server, cached in the browser so
// changing sliders re-ranks instantly (no re-upload, no recompute).
export type Metric = { filename: string; unreadable?: boolean; [k: string]: unknown };

const SCORE_CHUNK = 14; // images per upload request
const SCORE_CONCURRENCY = 2; // requests in flight (pipelines upload + analysis)

// Upload photos in concurrent chunks and collect their metrics. onProgress fires
// as each chunk lands, so the bar shows real progress and huge shoots never
// hit a single-request timeout.
export async function scoreUpload(
  files: File[],
  onProgress?: (done: number, total: number) => void
): Promise<Metric[]> {
  const chunks: File[][] = [];
  for (let i = 0; i < files.length; i += SCORE_CHUNK) chunks.push(files.slice(i, i + SCORE_CHUNK));
  const out: Metric[] = [];
  let done = 0;
  let idx = 0;
  async function worker() {
    while (idx < chunks.length) {
      const chunk = chunks[idx++];
      const fd = new FormData();
      chunk.forEach((f) => fd.append("files", f, f.name));
      const res = await fetch(`${API_URL}/score-upload`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Analysis failed (${res.status})`);
      const data = await res.json();
      out.push(...(data.metrics || []));
      done += chunk.length;
      onProgress?.(done, files.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(SCORE_CONCURRENCY, chunks.length) }, worker));
  return out;
}

export async function rankMetrics(metrics: Metric[], s: CullSettings): Promise<CullResult> {
  const res = await fetch(`${API_URL}/rank`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      metrics,
      blur_threshold: s.blur,
      duplicate_threshold: s.dupes,
      top_n: s.top_n,
      preset: s.preset,
    }),
  });
  if (!res.ok) throw new Error(`Ranking failed (${res.status})`);
  return res.json();
}
