// Usage + decision intake.
//
// Replaces the Google Form. The database credential stays server-side, so
// unlike the form (whose entry ids shipped in the client bundle) a visitor
// cannot write straight to storage.
//
// This endpoint is still public, because the app has no accounts. A public
// endpoint can always be called by hand, so the goal here is to make abuse
// bounded and boring rather than to pretend it is impossible:
//   - every field is range-checked, so one request cannot add a million photos
//   - a per-IP rate limit caps how fast anyone can try
//   - decisions are capped per request
//   - nothing identifying is accepted: no filenames, no EXIF, no image data
//
// Returns 204 on success and never leaks database errors to the caller.

import { NextRequest, NextResponse } from "next/server";
import { dbEnabled, insertEvent, insertDecisions, type DecisionRow } from "@/lib/db";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const EVENTS = new Set([
  "session_start", "photos_processed", "export_completed", "canvas_created",
]);
const VERDICTS = new Set(["agree_keep", "false_positive", "rescued", "agree_drop"]);

// A 500 photo cull is the realistic ceiling the UI allows; leave headroom.
const MAX_DECISIONS = 1000;
const MAX_PHOTOS = 10_000;
const MAX_MINUTES = 100_000;

// Fixed-window limiter. In-memory, so it is per serverless instance rather
// than global: enough to stop a loop from one machine, not a distributed
// flood. Redis would make it global, which is worth doing only if abuse
// actually shows up.
const WINDOW_MS = 60_000;
const MAX_REQ = 30;
const hits = new Map<string, { n: number; reset: number }>();

function limited(ip: string): boolean {
  const now = Date.now();
  const h = hits.get(ip);
  if (!h || now > h.reset) {
    hits.set(ip, { n: 1, reset: now + WINDOW_MS });
    if (hits.size > 5000) for (const [k, v] of hits) if (now > v.reset) hits.delete(k);
    return false;
  }
  h.n += 1;
  return h.n > MAX_REQ;
}

const clamp = (v: unknown, lo: number, hi: number): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : 0;
};

export async function POST(req: NextRequest) {
  // Accept and discard while unconfigured, so the client can switch over
  // before the database exists without logging errors at every visitor.
  if (!dbEnabled()) return new NextResponse(null, { status: 204 });

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0].trim() || "unknown";
  if (limited(ip)) return new NextResponse(null, { status: 429 });

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return new NextResponse(null, { status: 400 });
  }

  const sessionId = String(body.session_id ?? "").trim().slice(0, 64);
  if (!sessionId) return new NextResponse(null, { status: 400 });

  try {
    const kind = String(body.kind ?? "event");

    if (kind === "event") {
      const type = String(body.event_type ?? "");
      if (!EVENTS.has(type)) return new NextResponse(null, { status: 400 });
      await insertEvent({
        session_id: sessionId,
        event_type: type,
        photos: Math.round(clamp(body.photos, 0, MAX_PHOTOS)),
        exports: Math.round(clamp(body.exports, 0, 100)),
        minutes: clamp(body.minutes, 0, MAX_MINUTES),
      });
      return new NextResponse(null, { status: 204 });
    }

    if (kind === "decisions") {
      const raw = Array.isArray(body.rows) ? body.rows : [];
      const rows = raw.slice(0, MAX_DECISIONS).filter(
        (r): r is DecisionRow =>
          !!r && typeof r === "object" && VERDICTS.has(String((r as DecisionRow).verdict))
      );
      if (!rows.length) return new NextResponse(null, { status: 400 });
      const cullId = String(body.cull_id ?? "").trim().slice(0, 64) || `${sessionId}-${Date.now()}`;
      const preset = String(body.preset ?? "").slice(0, 32);
      await insertDecisions(sessionId, cullId, preset, rows);
      return new NextResponse(null, { status: 204 });
    }

    return new NextResponse(null, { status: 400 });
  } catch {
    // Never surface a database error to a visitor, and never fail their cull
    // because logging broke. Tracking is strictly best-effort.
    return new NextResponse(null, { status: 204 });
  }
}
