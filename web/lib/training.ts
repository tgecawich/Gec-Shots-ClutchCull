// Training data export.
//
// Every cull is a labelled example: the engine proposes a ranking, then a
// photographer corrects it. Those corrections are the only ground truth this
// project has, and the browser already holds all of it, so nothing needs to be
// uploaded to capture it.
//
// Why this is a local download rather than remote logging: usage currently goes
// through a public Google Form into a publicly readable sheet. That sheet is
// already being served truncated at ~7,400 rows, which is why the dashboard
// under-reported. One 500 photo cull would add 500 rows, roughly 7% of
// everything logged since April, so per-photo logging there would break the
// live counters outright. A real endpoint can collect this later; the file
// below has the same contents either way.
//
// Nothing here identifies anybody: numbers the engine computed, the rank it
// chose, and the photographer's verdict. No images, no EXIF, no location.

import type { CullResult, Metric } from "./api";

/** What the photographer did with a frame, relative to what the engine said. */
export type Verdict =
  | "agree_keep"    // engine kept it, photographer kept it
  | "false_positive" // engine kept it, photographer threw it out
  | "rescued"       // engine rejected it, photographer wanted it  <- most valuable
  | "agree_drop";   // engine rejected it, photographer agreed

const NUMERIC_FIELDS = [
  "sharpness", "subject_sharpness", "frame_tiled", "detail_ratio",
  "contrast", "brightness_mean", "exposure_balance", "face_score",
  "ball_score", "eye_sharpness",
];

function csvCell(v: unknown): string {
  const s = v === undefined || v === null ? "" : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * One row per photo: the engine's features and verdict, then the human's.
 *
 * `rescued` rows are the valuable ones. They are frames the engine demoted
 * that a photographer went and retrieved, which is precisely where the model
 * is wrong in the direction that costs a user a photograph.
 */
export function buildTrainingRows(
  results: CullResult,
  metrics: Metric[],
  selected: Set<string>
): Array<Record<string, unknown>> {
  const byName = new Map(metrics.map((m) => [String(m.filename), m]));
  const keeperRank = new Map(results.keepers.map((k, i) => [k.filename, i + 1]));
  const keeperScore = new Map(results.keepers.map((k) => [k.filename, k.score]));
  const names = new Set<string>([
    ...results.keepers.map((k) => k.filename),
    ...results.rejected,
    ...metrics.map((m) => String(m.filename)),
  ]);
  const out: Array<Record<string, unknown>> = [];
  for (const name of names) {
    const m = byName.get(name);
    const engineKept = keeperRank.has(name);
    const userSelected = selected.has(name);
    // NOTE: no filename. The verdict and the numbers are the whole signal;
    // sending names would leak what somebody shot for no modelling benefit.
    const row: Record<string, unknown> = {
      engine_rank: keeperRank.get(name) ?? null,
      engine_score: keeperScore.get(name) ?? null,
      engine_kept: engineKept,
      user_selected: userSelected,
      verdict: engineKept
        ? (userSelected ? "agree_keep" : "false_positive")
        : (userSelected ? "rescued" : "agree_drop"),
    };
    for (const f of NUMERIC_FIELDS) {
      const v = m?.[f];
      row[f] = typeof v === "number" ? v : null;
    }
    out.push(row);
  }
  return out;
}

export function buildTrainingCsv(
  results: CullResult,
  metrics: Metric[],
  selected: Set<string>,
  preset: string,
  settings: { top_n: number; blur: number; dupes: number }
): string {
  const byName = new Map(metrics.map((m) => [String(m.filename), m]));
  const keeperRank = new Map(results.keepers.map((k, i) => [k.filename, i + 1]));
  const keeperScore = new Map(results.keepers.map((k) => [k.filename, k.score]));
  const keeperBadge = new Map(results.keepers.map((k) => [k.filename, k.badge]));
  const softSet = new Set(results.keepers.filter((k) => k.soft).map((k) => k.filename));

  // Every frame the engine saw, keepers and rejects alike. Training on keepers
  // only would teach the model to rank what it already liked.
  const names = new Set<string>([
    ...results.keepers.map((k) => k.filename),
    ...results.rejected,
    ...metrics.map((m) => String(m.filename)),
  ]);

  const header = [
    "filename", "engine_rank", "engine_score", "engine_badge", "engine_soft",
    "engine_kept", "user_selected", "verdict", "preset", "top_n", "blur", "dupes",
    ...NUMERIC_FIELDS,
  ];
  const lines = [header.join(",")];

  for (const name of names) {
    const m = byName.get(name);
    const engineKept = keeperRank.has(name);
    const userSelected = selected.has(name);
    const verdict: Verdict = engineKept
      ? (userSelected ? "agree_keep" : "false_positive")
      : (userSelected ? "rescued" : "agree_drop");
    const row = [
      name,
      keeperRank.get(name) ?? "",
      keeperScore.get(name) !== undefined ? Math.round(keeperScore.get(name)! * 100) / 100 : "",
      keeperBadge.get(name) ?? "",
      softSet.has(name) ? 1 : 0,
      engineKept ? 1 : 0,
      userSelected ? 1 : 0,
      verdict,
      preset, settings.top_n, settings.blur, settings.dupes,
      ...NUMERIC_FIELDS.map((f) => {
        const v = m?.[f];
        return typeof v === "number" ? Math.round(v * 10000) / 10000 : "";
      }),
    ];
    lines.push(row.map(csvCell).join(","));
  }
  return lines.join("\n");
}

/** Human-readable summary, so the photographer can see what they're sending. */
export function summarise(results: CullResult, selected: Set<string>) {
  const keeperNames = new Set(results.keepers.map((k) => k.filename));
  let agreeKeep = 0, falsePositive = 0, rescued = 0;
  for (const k of results.keepers) {
    if (selected.has(k.filename)) agreeKeep++;
    else falsePositive++;
  }
  for (const name of results.rejected) if (selected.has(name)) rescued++;
  const judged = agreeKeep + falsePositive + rescued;
  return {
    agreeKeep, falsePositive, rescued,
    total: results.total,
    agreementPct: judged ? Math.round((agreeKeep / judged) * 100) : 0,
    keeperCount: keeperNames.size,
  };
}
