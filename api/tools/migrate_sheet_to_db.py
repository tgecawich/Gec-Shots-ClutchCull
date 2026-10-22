"""Import the legacy Google Sheet into Postgres, once.

The sheet holds everything logged since April. Switching to the database
without importing it would reset the counters to zero, which is both wrong and
the single most visible thing on the homepage.

Usage:
    pip install psycopg2-binary
    export DATABASE_URL='postgresql://...'
    python tools/migrate_sheet_to_db.py --dry-run     # inspect first
    python tools/migrate_sheet_to_db.py

Safe to re-run: rows are tagged source='legacy' and the script refuses to
import twice unless --force is passed, so a double run cannot double the
counters.

Two things are deliberately NOT imported:
  * the email column. The sheet is publicly readable and already exposes 38
    real addresses; the new schema has no email column and reproducing that
    mistake is not worth it.
  * rows whose event_type is not one the app emits. The sheet contains two
    'test' rows and one row whose event_type is a prose sentence about an
    outage, carrying 19,800 photos that never happened. Those are reported and
    skipped. Pass --include-anomalies to import them anyway.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime

SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi3K-ggYir5zDj6mYXIMrWcG-"
    "fyTqWu6tQtQ2g97vFpzSZmKrJ0nnExHIBzA7zDCbvhabDdCG8EYSa/pub?gid=370944124&single=true&output=csv"
)
KNOWN_EVENTS = {"session_start", "photos_processed", "export_completed",
                "canvas_created", "email_provided"}
TS_FORMATS = ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y")


def fetch() -> bytes:
    """python.org builds on macOS ship without a CA bundle; fall back to curl."""
    req = urllib.request.Request(SHEET_CSV, headers={"User-Agent": "clutchcull-migrate"})
    try:
        ctx = None
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            return r.read()
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        return subprocess.run(["curl", "-sL", "--max-time", "60", SHEET_CSV],
                              capture_output=True, check=True).stdout


def parse(data: bytes, include_anomalies: bool):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8", "replace"))))
    header = [h.strip().lower().replace(" ", "_") for h in rows[0]]
    idx = {n: (header.index(n) if n in header else -1) for n in
           ("timestamp", "event_type", "photos_processed", "exports", "minutes_saved", "session_id")}

    def val(row, name):
        i = idx[name]
        return row[i].strip() if 0 <= i < len(row) else ""

    def num(row, name):
        try:
            return float(val(row, name) or 0)
        except ValueError:
            return 0.0

    good, anomalies = [], []
    for row in rows[1:]:
        if not row:
            continue
        etype = val(row, "event_type")
        sid = val(row, "session_id")
        if not sid:
            continue
        ts = None
        for fmt in TS_FORMATS:
            try:
                ts = datetime.strptime(val(row, "timestamp"), fmt)
                break
            except ValueError:
                continue
        rec = (ts, sid, etype[:64], int(num(row, "photos_processed")),
               int(num(row, "exports")), round(num(row, "minutes_saved"), 2))
        if etype in KNOWN_EVENTS:
            good.append(rec)
        else:
            anomalies.append((rec, etype))
    if include_anomalies:
        good.extend(r for r, _ in anomalies)
    return good, anomalies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="import even if legacy rows exist")
    ap.add_argument("--include-anomalies", action="store_true")
    a = ap.parse_args()

    good, anomalies = parse(fetch(), a.include_anomalies)
    photos = sum(r[3] for r in good)
    people = len({r[1] for r in good})
    print(f"sheet parsed: {len(good):,} importable rows, {people:,} distinct sessions, "
          f"{photos:,} photos")

    if anomalies:
        print(f"\n{len(anomalies)} row(s) NOT of a known event type:")
        for rec, etype in anomalies:
            print(f"   photos={rec[3]:>6}  event_type={etype[:80]!r}")
        print("   excluded by default. --include-anomalies imports them.")

    if a.dry_run:
        print("\ndry run, nothing written")
        return

    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("set DATABASE_URL first")
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        sys.exit("pip install psycopg2-binary")

    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS events (
            id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ NOT NULL DEFAULT now(),
            session_id TEXT NOT NULL, event_type TEXT NOT NULL,
            photos INTEGER NOT NULL DEFAULT 0, exports INTEGER NOT NULL DEFAULT 0,
            minutes NUMERIC(10,2) NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'web')""")
        cur.execute("SELECT COUNT(*) FROM events WHERE source = 'legacy'")
        existing = cur.fetchone()[0]
        if existing and not a.force:
            sys.exit(f"{existing:,} legacy rows already imported. Re-running would "
                     f"double the counters. Pass --force only if you cleared them.")
        execute_values(cur,
            "INSERT INTO events (ts, session_id, event_type, photos, exports, minutes, source) VALUES %s",
            [(r[0], r[1], r[2], r[3], r[4], r[5], "legacy") for r in good])
        conn.commit()
        cur.execute("SELECT COUNT(*), COALESCE(SUM(photos),0), COUNT(DISTINCT session_id) FROM events")
        n, p, s = cur.fetchone()
    print(f"\nimported. events table now holds {n:,} rows, {p:,} photos, {s:,} sessions")
    print("set DATABASE_URL on Vercel and the dashboard switches over automatically")


if __name__ == "__main__":
    main()
