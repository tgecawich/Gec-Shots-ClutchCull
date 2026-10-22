// Usage tracking.
//
// Two backends. NEXT_PUBLIC_TRACK_DB=1 sends to our own /api/track, where the
// database credential stays server-side. Otherwise it posts to the legacy
// Google Form, which is what the counters have always read.
//
// They are mutually exclusive, never both: dual-writing would double-count
// every photo on a dashboard that is labelled "updating in real time".
//
// What is never sent, under either backend: filenames, EXIF, location, or any
// image data. Decisions are the numbers the engine computed plus the
// photographer's verdict, which is what makes the ranking trainable without
// anybody's shoot leaving their machine.

const FORM =
  "https://docs.google.com/forms/u/0/d/e/1FAIpQLSdE_xxiIaiHwYX9LQag1kipieTojmqEfqv1fVqwtsCKo45Mlg/formResponse";
const F = {
  event: "entry.1792514521",
  email: "entry.1250685824",
  photos: "entry.1966720800",
  exports: "entry.23206585",
  minutes: "entry.1192573871",
  session: "entry.747824885",
};

const USE_DB = process.env.NEXT_PUBLIC_TRACK_DB === "1";

function sessionId(): string {
  if (typeof window === "undefined") return "";
  const existing = localStorage.getItem("cc_sid");
  if (existing) return existing;
  const id: string = (globalThis.crypto as any)?.randomUUID?.() || String(Date.now()) + Math.random();
  localStorage.setItem("cc_sid", id);
  return id;
}

async function postForm(params: Record<string, string>) {
  try {
    await fetch(FORM, {
      method: "POST",
      mode: "no-cors",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams(params).toString(),
    });
  } catch {
    /* best-effort */
  }
}

async function postDb(payload: Record<string, unknown>) {
  try {
    await fetch("/api/track", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, session_id: sessionId() }),
      keepalive: true, // survive the tab closing right after an export
    });
  } catch {
    /* best-effort: tracking must never break a cull */
  }
}

function event(type: string, fields: { photos?: number; exports?: number; minutes?: number }, email = "") {
  if (USE_DB) {
    void postDb({ kind: "event", event_type: type, ...fields });
    return;
  }
  void postForm({
    [F.event]: type,
    [F.session]: sessionId(),
    [F.email]: email,
    ...(fields.photos !== undefined ? { [F.photos]: String(fields.photos) } : {}),
    ...(fields.exports !== undefined ? { [F.exports]: String(fields.exports) } : {}),
    ...(fields.minutes !== undefined ? { [F.minutes]: fields.minutes.toFixed(2) } : {}),
  });
}

export function trackSessionStart() {
  if (typeof window === "undefined") return;
  if (sessionStorage.getItem("cc_started")) return;
  sessionStorage.setItem("cc_started", "1");
  event("session_start", {});
}

export function trackPhotos(n: number, email = "") {
  event("photos_processed", { photos: n }, email);
}

// One export event per download. `minutes` (time saved) should be passed once
// per shoot (0 on later downloads) so Hours Saved isn't inflated. Photos are
// counted at cull time, not here.
export function trackExport(minutes: number, email = "") {
  event("export_completed", { exports: 1, minutes }, email);
}

// Building canvas posts is real work saved too, so it credits `minutes` (once
// per batch). No `exports` here, downloading the batch logs that separately,
// so one canvas batch can't be counted as two exports.
export function trackCanvas(n: number, minutes: number, email = "") {
  event("canvas_created", { photos: n, minutes }, email);
}

export function trackEmail(email: string) {
  // Only the legacy backend stores an address. The database schema has no
  // email column on purpose: the old sheet is publicly readable and already
  // exposes 38 real addresses, and that mistake is not worth reproducing.
  if (USE_DB) return;
  void postForm({ [F.event]: "email_provided", [F.email]: email, [F.session]: sessionId() });
}

/**
 * Per-photo cull decisions: the engine's features and verdict, then the
 * photographer's. This is the training signal, and it only exists while the
 * database backend is on, because 500 rows per cull would have destroyed the
 * legacy sheet (already truncating at ~7,400 rows).
 */
export function trackDecisions(
  rows: Array<Record<string, unknown>>,
  preset: string,
  cullId: string
) {
  if (!USE_DB || !rows.length) return;
  void postDb({ kind: "decisions", rows, preset, cull_id: cullId });
}
