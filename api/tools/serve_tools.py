"""Serve the labelling tools over localhost with caching disabled.

Opening rank_pairs.html as a file:// page works, but the browser caches it, and
a cached copy is indistinguishable from a current one. That cost real data: two
shoots were judged against a stale build whose repeat pairs were unreachable, so
the self-consistency measurement those repeats exist to produce came back empty
both times.

Serving with `Cache-Control: no-store` makes that impossible rather than
unlikely. It also means the page can be opened by automation, which cannot
navigate to file:// URLs at all.

    python tools/serve_tools.py
    python tools/serve_tools.py --port 9000

Then open http://localhost:8765/rank_pairs.html

Everything still runs entirely in the browser. The server only hands over the
HTML; photos are read through the local file picker and never uploaded, and the
comparisons are exported as a normal download.
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

HERE = Path(__file__).resolve().parent


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def end_headers(self):
        # no-store rather than no-cache: no-cache still lets the browser keep a
        # copy and revalidate, and a revalidation that is wrongly satisfied is
        # exactly the failure being prevented.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        # The default logs every request to stderr, which buries the one line
        # actually worth reading when this runs in the background.
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    # Bind to loopback explicitly. The default binds every interface, which
    # would expose these pages to the local network for no benefit.
    with socketserver.TCPServer(("127.0.0.1", a.port), NoCacheHandler) as srv:
        srv.allow_reuse_address = True
        print(f"serving {HERE} at http://localhost:{a.port}/")
        print(f"  judging UI:  http://localhost:{a.port}/rank_pairs.html")
        print("  caching disabled; Ctrl-C to stop")
        srv.serve_forever()


if __name__ == "__main__":
    main()
