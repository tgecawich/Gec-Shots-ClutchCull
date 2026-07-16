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

const SCORE_CHUNK = 8; // images per upload request — small so a chunk can't OOM the free tier
const SCORE_CONCURRENCY = 2; // requests in flight (pipelines upload + analysis)
const CHUNK_RETRIES = 4; // survive a free-tier restart / transient blip
const CHUNK_TIMEOUT_MS = 90_000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Score one chunk with a timeout + retries. On the free tier a big shoot can
// briefly restart the container mid-request; retrying lets the cull recover
// instead of dying with "failed to fetch".
async function scoreChunk(chunk: File[]): Promise<Metric[]> {
  let lastErr: unknown;
  for (let attempt = 0; attempt <= CHUNK_RETRIES; attempt++) {
    if (attempt > 0) await sleep(Math.min(8000, 700 * 2 ** (attempt - 1))); // 0.7s,1.4s,2.8s,5.6s
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), CHUNK_TIMEOUT_MS);
    try {
      const fd = new FormData();
      chunk.forEach((f) => fd.append("files", f, f.name));
      const res = await fetch(`${API_URL}/score-upload`, { method: "POST", body: fd, signal: ctrl.signal });
      if (res.status >= 500 || res.status === 429) throw new Error(`server ${res.status}`); // transient → retry
      if (!res.ok) throw new Error(`Analysis failed (${res.status})`); // 4xx → real, still surfaced
      const data = await res.json();
      return data.metrics || [];
    } catch (e) {
      lastErr = e;
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("Analysis failed");
}

// Upload photos in small concurrent chunks and collect their metrics. onProgress
// fires as each chunk lands, so the bar shows real progress, big shoots never
// hit a single-request timeout, and a transient failure retries automatically.
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
      out.push(...(await scoreChunk(chunk)));
      done += chunk.length;
      onProgress?.(done, files.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(SCORE_CONCURRENCY, chunks.length) }, worker));
  return out;
}

export async function rankMetrics(metrics: Metric[], s: CullSettings): Promise<CullResult> {
  const body = JSON.stringify({
    metrics,
    blur_threshold: s.blur,
    duplicate_threshold: s.dupes,
    top_n: s.top_n,
    preset: s.preset,
  });
  let lastErr: unknown;
  for (let attempt = 0; attempt <= 2; attempt++) {
    if (attempt > 0) await sleep(600 * attempt);
    try {
      const res = await fetch(`${API_URL}/rank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (res.status >= 500 || res.status === 429) throw new Error(`server ${res.status}`);
      if (!res.ok) throw new Error(`Ranking failed (${res.status})`);
      return res.json();
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("Ranking failed");
}
