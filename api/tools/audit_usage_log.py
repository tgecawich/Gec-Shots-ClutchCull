"""Audit the usage log that feeds the Impact Dashboard.

Why this exists: two reads of the published sheet taken the same day returned
totals that disagreed by tens of thousands of photos (and the photographer
count went DOWN between reads, which a cumulative counter cannot do). Either
Google's published-CSV endpoint serves truncated/stale snapshots once a sheet
gets large, or the sheet's rows changed between reads. This script settles it.

Usage:
    python tools/audit_usage_log.py            # fetch 5 times, compare, summarize
    python tools/audit_usage_log.py --once     # single fetch + summary only

It reports, per fetch: row count, byte size, content hash, and computed totals
(photos / exports / minutes / distinct sessions). If hashes differ across
fetches made seconds apart, the serving layer is unstable and the dashboard
should move off the published-CSV endpoint. It also summarizes activity by
month and flags days whose volume is a large multiple of the surrounding days,
so any anomalous batch of rows is easy to see and explain.

Local diagnostic only; not imported by the API.
"""
from __future__ import annotations

import csv
import hashlib
import io
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime

SHEET_CSV = (
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQi3K-ggYir5zDj6mYXIMrWcG-"
    "fyTqWu6tQtQ2g97vFpzSZmKrJ0nnExHIBzA7zDCbvhabDdCG8EYSa/pub?gid=370944124&single=true&output=csv"
)
FETCHES = 5
PAUSE_S = 4


def fetch() -> bytes:
    """Fetch the published CSV.

    python.org builds on macOS ship without a CA bundle, so urllib raises
    CERTIFICATE_VERIFY_FAILED on a perfectly valid Google certificate. Use
    certifi when it is installed, otherwise shell out to curl, which uses the
    system trust store. Verification is never disabled.
    """
    req = urllib.request.Request(SHEET_CSV, headers={"User-Agent": "clutchcull-audit"})
    try:
        ctx = None
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            pass
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            return r.read()
    except urllib.error.URLError as e:
        if not isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise
        out = subprocess.run(
            ["curl", "-sL", "--max-time", "30", SHEET_CSV],
            capture_output=True, check=True,
        ).stdout
        if not out:
            raise RuntimeError("curl fallback returned no data") from e
        return out


def parse(data: bytes):
    rows = list(csv.reader(io.StringIO(data.decode("utf-8", "replace"))))
    if not rows:
        return None
    header = [h.strip().lower().replace(" ", "_") for h in rows[0]]
    idx = {n: (header.index(n) if n in header else -1)
           for n in ("timestamp", "event_type", "photos_processed", "exports",
                     "minutes_saved", "session_id")}

    def val(row, name):
        i = idx[name]
        return row[i].strip() if 0 <= i < len(row) else ""

    photos = exports = minutes = 0.0
    sessions = set()
    by_day = defaultdict(float)
    by_month = defaultdict(lambda: [0.0, 0])  # photos, rows
    for row in rows[1:]:
        try:
            photos_v = float(val(row, "photos_processed") or 0)
        except ValueError:
            photos_v = 0.0
        photos += photos_v
        try:
            exports += float(val(row, "exports") or 0)
        except ValueError:
            pass
        try:
            minutes += float(val(row, "minutes_saved") or 0)
        except ValueError:
            pass
        sid = val(row, "session_id")
        if sid:
            sessions.add(sid)
        ts = val(row, "timestamp")
        for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y"):
            try:
                d = datetime.strptime(ts, fmt)
                by_day[d.strftime("%Y-%m-%d")] += photos_v
                m = by_month[d.strftime("%Y-%m")]
                m[0] += photos_v
                m[1] += 1
                break
            except ValueError:
                continue
    return {
        "rows": len(rows) - 1, "photos": photos, "exports": exports,
        "minutes": minutes, "sessions": len(sessions),
        "by_day": dict(by_day), "by_month": {k: tuple(v) for k, v in by_month.items()},
    }


def main() -> None:
    once = "--once" in sys.argv
    n = 1 if once else FETCHES
    snaps = []
    for i in range(n):
        if i:
            time.sleep(PAUSE_S)
        data = fetch()
        h = hashlib.sha256(data).hexdigest()[:12]
        s = parse(data)
        snaps.append((h, len(data), s))
        print(f"fetch {i + 1}: {s['rows']:>7} rows  {len(data):>9} bytes  sha {h}  "
              f"photos {s['photos']:>10,.0f}  sessions {s['sessions']:>5}")

    hashes = {h for h, _, _ in snaps}
    print()
    if len(hashes) == 1:
        print("STABLE: every fetch returned byte-identical data. The serving layer is")
        print("fine; any discrepancy between dashboard reads means the SHEET changed")
        print("between those reads (rows added, edited, or deleted).")
    else:
        print(f"UNSTABLE: {len(hashes)} distinct payloads across {n} fetches seconds apart.")
        print("Google's published-CSV endpoint is not serving this sheet consistently.")
        print("The dashboard numbers will wobble until logging moves off this pipeline.")

    s = snaps[-1][2]
    print(f"\nTotals (last fetch): {s['photos']:,.0f} photos, {s['exports']:,.0f} exports, "
          f"{s['minutes'] / 60:,.1f} hours, {s['sessions']:,} distinct sessions")
    print("\nBy month (photos / rows):")
    for m in sorted(s["by_month"]):
        p, r = s["by_month"][m]
        print(f"  {m}: {p:>10,.0f} / {r:,}")

    # Day-level outliers: any day >8x the median of its surrounding 15 days.
    days = sorted(s["by_day"].items())
    vals = [v for _, v in days]
    flagged = []
    for i, (d, v) in enumerate(days):
        window = sorted(vals[max(0, i - 7):i + 8])
        med = window[len(window) // 2] if window else 0
        if med > 0 and v > 8 * med and v > 2000:
            flagged.append((d, v, med))
    if flagged:
        print("\nDay-level outliers (volume >8x the local median - explain each one):")
        for d, v, med in flagged:
            print(f"  {d}: {v:,.0f} photos (local median {med:,.0f})")
    else:
        print("\nNo extreme day-level outliers.")


if __name__ == "__main__":
    main()
