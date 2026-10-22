// Shared data layer for the landing pages.
//
// Both the current design (/) and the preserved original (/classic) read from
// here. The counter logic lives in exactly one place on purpose: two copies
// would eventually disagree, and a dashboard labelled "updating in real time"
// cannot afford a second opinion about what the numbers are.

export const APP_URL = process.env.NEXT_PUBLIC_APP_URL || "/app";

const SHEET_CSV =
  "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi3K-ggYir5zDj6mYXIMrWcG-fyTqWu6tQtQ2g97vFpzSZmKrJ0nnExHIBzA7zDCbvhabDdCG8EYSa/pub?gid=370944124&single=true&output=csv";

// Photographers who used the original Streamlit version, before the Next.js
// rebuild. Recorded here as a constant because the shared Google Sheet only kept
// partial data for that period: Streamlit minted a NEW session_id on every visit
// (and re-logged session_start on every rerun), so its ids can't be counted as
// people. Sourced from the original app's own user count, not derived from the
// sheet, anything after LEGACY_CUTOVER is counted live below.
const LEGACY_PHOTOGRAPHERS = 270;
const LEGACY_CUTOVER = "2026-07-01";
// Rolling window for the momentum strip, so it stays current on its own.
// 30 rather than 7: a week is noisy enough that a single quiet stretch or an
// API outage dominates it, and the strip then reads as decline rather than as
// normal variance. A month is the more honest summary of current activity.
export const RECENT_WINDOW_DAYS = 30;

export type Impact = {
  people: number;
  photos: number;
  exports: number;
  hours: number;
  recentPeople: number;
  recentPhotos: number;
};

// Live Impact Dashboard, fetched server-side from the shared Google Sheet
// (same data source as the original app), so it stays visible and current.
//
// Note on the source: this endpoint has been observed serving TRUNCATED copies
// of the sheet, which silently under-reported the totals (two reads minutes
// apart summed to exact row-prefixes of the real file). Nothing here can detect
// that, because a short file is indistinguishable from a small one. The real
// fix is moving logging off the published-CSV endpoint; api/tools/audit_usage_log.py
// re-checks it on demand in the meantime.
// How many independent reads to take before trusting a number. See parseCsv's
// comment for why more than one is necessary and why taking the largest is safe.
const READS = 3;

export async function getImpact(): Promise<Impact | null> {
  // Prefer the database once DATABASE_URL is set. It is aggregated server-side,
  // so it cannot be truncated the way the published CSV was, and raw usage rows
  // never leave the server. Falls back to the sheet while unconfigured, so this
  // is safe to deploy before the database exists.
  try {
    const { dbEnabled, getTotals } = await import("./db");
    if (dbEnabled()) {
      const t = await getTotals(RECENT_WINDOW_DAYS);
      if (t && t.photos > 0) {
        return {
          people: LEGACY_PHOTOGRAPHERS + t.people,
          photos: t.photos,
          exports: t.exports,
          hours: t.minutes / 60,
          recentPeople: t.recentPeople,
          recentPhotos: t.recentPhotos,
        };
      }
    }
  } catch {
    // Fall through to the sheet rather than showing an empty dashboard.
  }
  return getImpactFromSheet();
}

async function getImpactFromSheet(): Promise<Impact | null> {
  // Observed in production: two routes calling this function in the same minute
  // rendered 207,110 and 549,141. The endpoint intermittently serves a
  // TRUNCATED copy of the sheet, and a short file is indistinguishable from a
  // small one, so a single read cannot detect it.
  //
  // Truncation can only ever DROP trailing rows, never invent them, so every
  // read is a lower bound on the truth and the largest read is the closest to
  // it. Taking the max across a few reads can therefore under-report slightly
  // but can never over-report, which is the only direction that matters for a
  // counter people are asked to believe.
  //
  // Distinct cache-buster values give each read its own Next.js cache entry
  // (one shared URL would just be deduped into a single request), so this costs
  // three fetches per five-minute revalidation window, not three per visitor.
  const reads = await Promise.all(
    Array.from({ length: READS }, (_, i) => readOnce(`${SHEET_CSV}&_r=${i}`))
  );
  const best = reads.filter(Boolean).sort((a, b) => b!.rows - a!.rows)[0];
  return best ? best.impact : null;
}

async function readOnce(url: string): Promise<{ rows: number; impact: Impact } | null> {
  try {
    const res = await fetch(url, { next: { revalidate: 300 } });
    if (!res.ok) return null;
    const rows = (await res.text()).trim().split(/\r?\n/).map((r) => r.split(","));
    const header = rows[0].map((h) => h.trim().toLowerCase().replace(/ /g, "_"));
    const ci = (n: string) => header.indexOf(n);
    const iT = ci("timestamp"), iP = ci("photos_processed"),
      iX = ci("exports"), iM = ci("minutes_saved"), iS = ci("session_id");

    // Count UNIQUE people (distinct session_id), never raw session_start rows:
    // the old Streamlit app re-ran its script on every interaction, so a single
    // browser logged up to 314 session_starts (799 "sessions" from 20 ids in April).
    const firstSeen = new Map<string, number>();
    let photos = 0, exports = 0, minutes = 0;
    let recentPhotos = 0;
    const now = Date.now();
    const windowStart = now - RECENT_WINDOW_DAYS * 86400_000;

    for (const row of rows.slice(1)) {
      const t = iT >= 0 ? Date.parse(row[iT]) : NaN;
      const sid = iS >= 0 ? (row[iS] || "").trim() : "";
      if (sid && !Number.isNaN(t)) {
        const prev = firstSeen.get(sid);
        if (prev === undefined || t < prev) firstSeen.set(sid, t);
      } else if (sid && !firstSeen.has(sid)) {
        firstSeen.set(sid, NaN);
      }
      const p = iP >= 0 ? parseFloat(row[iP]) || 0 : 0;
      photos += p;
      if (!Number.isNaN(t) && t >= windowStart) recentPhotos += p;
      if (iX >= 0) exports += parseFloat(row[iX]) || 0;
      if (iM >= 0) minutes += parseFloat(row[iM]) || 0;
    }

    // Only count ids first seen after the Streamlit -> Next.js cutover, then add
    // the recorded legacy figure. Counting all ids AND adding the legacy total
    // would double-count the Apr–Jun period.
    const cutover = Date.parse(LEGACY_CUTOVER);
    let livePeople = 0, recentPeople = 0;
    for (const t of firstSeen.values()) {
      if (Number.isNaN(t) || t >= cutover) livePeople++;
      if (!Number.isNaN(t) && t >= windowStart) recentPeople++;
    }

    return {
      rows: rows.length - 1,
      impact: {
        people: LEGACY_PHOTOGRAPHERS + livePeople,
        photos, exports, hours: minutes / 60,
        recentPeople, recentPhotos,
      },
    };
  } catch {
    return null;
  }
}

// Real frames from a real shoot, scored by the actual engine. A photography
// tool whose landing page shows no photographs asks people to take its word.
// These are the top six keepers from a 33 frame soccer shoot: the files were
// downscaled to 1400px exactly as the browser does before upload, run through
// cull(), and the badge and score below are whatever the engine returned. The
// crops are centred on the person the engine detected, not on the frame.
export const shots = [
  { src: "/shots/D91A7783.jpg", tag: "Sharp subject", score: 88, alt: "Soccer player jogging upfield under a burning sunset sky" },
  { src: "/shots/D91A7300.jpg", tag: "Clean contrast", score: 87, alt: "Two teammates leaning together on the bench between shifts" },
  { src: "/shots/D91A7770.jpg", tag: "Sharp subject", score: 82, alt: "Attacker and keeper converging on the ball at the near post" },
  { src: "/shots/D91A7758.jpg", tag: "Sharp subject", score: 69, alt: "Player walking the touchline under stadium lights at dusk" },
  { src: "/shots/D91A7763.jpg", tag: "Clear subject", score: 69, alt: "Midfielder waiting for a restart against a storm-blue sky" },
  { src: "/shots/D91A7166.jpg", tag: "Rich detail", score: 67, alt: "Two players shoulder to shoulder chasing a loose ball" },
];

// The BOTTOM of the same 33 frame shoot, same engine run. Showing positions
// 31-33 next to 1-3 is the only falsifiable way to demonstrate that the ranking
// works: anyone can display six good photos and claim an AI chose them.
export const lowShots = [
  { src: "/shots/D91A7103.jpg", tag: "Clear subject", score: 32, alt: "Player standing with the ball at his feet, mid-field" },
  { src: "/shots/D91A7175.jpg", tag: "Well-exposed", score: 23, alt: "Two players jogging, subject small in the frame" },
  { src: "/shots/D91A7330.jpg", tag: "Clear subject", score: 13, alt: "Backlit player in silhouette against a low sun" },
];
