// Postgres access for usage counters and cull decisions.
//
// Replaces a public Google Form writing to a publicly readable sheet. That
// pipeline had three faults, all of which this fixes:
//   1. The form entry ids ship in the client bundle, so ANYONE could write
//      rows. Counters and training data could both be poisoned.
//   2. The published sheet is readable by anyone with the link and contains
//      real email addresses.
//   3. Google serves that CSV truncated once it gets large, which silently
//      under-reported the dashboard by about 60%.
//
// Provider-agnostic on purpose: anything that speaks Postgres works (Neon,
// Supabase, Vercel Postgres, Railway). Set DATABASE_URL and it switches on.
// Leave it unset and every call here no-ops, so the site keeps running exactly
// as it does today. That is deliberate: this must be riskless to deploy before
// the database exists.

import postgres from "postgres";

let _sql: ReturnType<typeof postgres> | null = null;
let _init: Promise<void> | null = null;

// Circuit breaker. Without this, an outage means every single request pays the
// connect timeout before failing, turning a logging problem into a site-wide
// slowdown. After a failure we skip the database entirely for a cooldown and
// serve from the fallback instead.
const BREAKER_MS = 30_000;
let _openUntil = 0;

function breakerOpen(): boolean {
  return Date.now() < _openUntil;
}
function tripBreaker(): void {
  _openUntil = Date.now() + BREAKER_MS;
  _init = null;
}

export function dbEnabled(): boolean {
  return Boolean(process.env.DATABASE_URL) && !breakerOpen();
}

function client() {
  if (!process.env.DATABASE_URL) return null;
  if (!_sql) {
    _sql = postgres(process.env.DATABASE_URL, {
      // Serverless: every invocation is a short-lived process, so a big pool
      // just exhausts the database's connection limit. Use the provider's
      // pooled connection string and keep this tiny.
      max: 1,
      idle_timeout: 20,
      // 3s, not 10. Measured: with an unreachable database every request
      // stalled for the full connect timeout, so the dashboard took >10s to
      // render and the API effectively hung. A page must not be slow because
      // logging is broken.
      connect_timeout: 3,
      prepare: false, // pgbouncer in transaction mode rejects prepared statements
    });
  }
  return _sql;
}

/**
 * Create tables on first use. Cheap (IF NOT EXISTS) and runs once per process,
 * which avoids needing a separate migration step during setup.
 */
export async function ensureSchema(): Promise<void> {
  const sql = client();
  if (!sql) return;
  if (!_init) {
    _init = (async () => {
      await sql`
        CREATE TABLE IF NOT EXISTS events (
          id          BIGSERIAL PRIMARY KEY,
          ts          TIMESTAMPTZ  NOT NULL DEFAULT now(),
          session_id  TEXT         NOT NULL,
          event_type  TEXT         NOT NULL,
          photos      INTEGER      NOT NULL DEFAULT 0,
          exports     INTEGER      NOT NULL DEFAULT 0,
          minutes     NUMERIC(10,2) NOT NULL DEFAULT 0,
          source      TEXT         NOT NULL DEFAULT 'web'
        )`;
      await sql`CREATE INDEX IF NOT EXISTS events_ts_idx ON events (ts)`;
      await sql`CREATE INDEX IF NOT EXISTS events_session_idx ON events (session_id)`;
      // One row per photo per cull: the engine's features and verdict, then the
      // photographer's. No filename, no EXIF, no image data - the label is the
      // only thing worth keeping and the rest is somebody's private shoot.
      await sql`
        CREATE TABLE IF NOT EXISTS decisions (
          id               BIGSERIAL PRIMARY KEY,
          ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
          session_id       TEXT    NOT NULL,
          cull_id          TEXT    NOT NULL,
          engine_rank      INTEGER,
          engine_score     REAL,
          engine_kept      BOOLEAN NOT NULL,
          user_selected    BOOLEAN NOT NULL,
          verdict          TEXT    NOT NULL,
          preset           TEXT,
          top_n            INTEGER,
          blur             REAL,
          dupes            INTEGER,
          sharpness        REAL, subject_sharpness REAL, frame_tiled    REAL,
          detail_ratio     REAL, contrast          REAL, brightness_mean REAL,
          exposure_balance REAL, face_score        REAL, ball_score     REAL,
          eye_sharpness    REAL
        )`;
      await sql`CREATE INDEX IF NOT EXISTS decisions_cull_idx ON decisions (cull_id)`;
      await sql`CREATE INDEX IF NOT EXISTS decisions_verdict_idx ON decisions (verdict)`;
    })().catch((e) => {
      _init = null; // let a later request retry rather than wedging forever
      throw e;
    });
  }
  return _init;
}

export type EventRow = {
  session_id: string;
  event_type: string;
  photos?: number;
  exports?: number;
  minutes?: number;
};

export async function insertEvent(e: EventRow): Promise<void> {
  const sql = client();
  if (!sql) return;
  try {
    await ensureSchema();
    await sql`
      INSERT INTO events (session_id, event_type, photos, exports, minutes)
      VALUES (${e.session_id}, ${e.event_type}, ${e.photos ?? 0}, ${e.exports ?? 0}, ${e.minutes ?? 0})`;
  } catch (err) {
    tripBreaker();
    throw err;
  }
}

export type DecisionRow = Record<string, unknown> & {
  engine_kept: boolean;
  user_selected: boolean;
  verdict: string;
};

const DECISION_NUMERIC = [
  "engine_rank", "engine_score", "top_n", "blur", "dupes",
  "sharpness", "subject_sharpness", "frame_tiled", "detail_ratio", "contrast",
  "brightness_mean", "exposure_balance", "face_score", "ball_score", "eye_sharpness",
] as const;

export async function insertDecisions(
  sessionId: string, cullId: string, preset: string, rows: DecisionRow[]
): Promise<number> {
  const sql = client();
  if (!sql || !rows.length) return 0;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const payload = rows.map((r) => {
    const o: Record<string, unknown> = {
      session_id: sessionId, cull_id: cullId, preset,
      engine_kept: Boolean(r.engine_kept),
      user_selected: Boolean(r.user_selected),
      verdict: String(r.verdict).slice(0, 24),
    };
    for (const k of DECISION_NUMERIC) o[k] = num(r[k]);
    return o;
  });
  // One multi-row INSERT: a 500 photo cull is 500 rows, and 500 round trips
  // would blow the serverless function's time budget.
  try {
    await ensureSchema();
    await sql`INSERT INTO decisions ${sql(payload,
      "session_id", "cull_id", "preset", "engine_kept", "user_selected", "verdict",
      ...DECISION_NUMERIC)}`;
  } catch (err) {
    tripBreaker();
    throw err;
  }
  return payload.length;
}

export type Totals = {
  people: number; photos: number; exports: number; minutes: number;
  recentPeople: number; recentPhotos: number;
};

/**
 * Dashboard aggregate. Computed in the database rather than by shipping rows
 * to the client, so raw usage never leaves the server and the result cannot be
 * truncated the way the published CSV was.
 */
export async function getTotals(recentDays: number): Promise<Totals | null> {
  const sql = client();
  if (!sql) return null;
  try {
    await ensureSchema();
  } catch (err) {
    tripBreaker();
    throw err;
  }
  const [t] = await sql<[Totals]>`
    SELECT
      (SELECT COUNT(DISTINCT session_id) FROM events)::int AS people,
      COALESCE((SELECT SUM(photos)  FROM events), 0)::int  AS photos,
      COALESCE((SELECT SUM(exports) FROM events), 0)::int  AS exports,
      COALESCE((SELECT SUM(minutes) FROM events), 0)::float AS minutes,
      (SELECT COUNT(DISTINCT session_id) FROM events
        WHERE session_id IN (
          SELECT session_id FROM events GROUP BY session_id
          HAVING MIN(ts) >= now() - ${recentDays} * INTERVAL '1 day'))::int AS "recentPeople",
      COALESCE((SELECT SUM(photos) FROM events
        WHERE ts >= now() - ${recentDays} * INTERVAL '1 day'), 0)::int AS "recentPhotos"`
    .catch((err) => { tripBreaker(); throw err; });
  return t ?? null;
}
