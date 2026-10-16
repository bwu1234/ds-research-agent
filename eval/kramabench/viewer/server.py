"""Serve the run viewer on loopback.

The viewer opens the ledger read-only for each request, so it can stay open
while a batch writes. It binds 127.0.0.1 only and answers only requests whose
Host header names that address (or localhost) and its port, so a web page
that rebinds its own domain to 127.0.0.1 cannot read the viewer. Pages are
GET only, carry no forms that write, and are sent with a Content-Security-
Policy that allows only the viewer's own stylesheet and script.
"""

from __future__ import annotations

import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qsl, urlsplit

from ds_research_agent.ledger import Ledger, LedgerError
from eval.kramabench.viewer.pages import ROUTES, Context, NotFound, error_page, layout

STATIC = Path(__file__).parent / "static"
STATIC_TYPES = {
    "viewer.css": "text/css; charset=utf-8",
    "viewer.js": "text/javascript; charset=utf-8",
}
CSP = (
    "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self'; "
    "form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
)
HOST = "127.0.0.1"


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, ctx: Context, ledger_path: Path) -> None:
        self.ctx, self.ledger_path = ctx, ledger_path
        super().__init__((HOST, port), Handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])


class Handler(BaseHTTPRequestHandler):
    server_version = "run-viewer"

    @property
    def viewer(self) -> ViewerServer:
        return cast(ViewerServer, self.server)

    def do_GET(self) -> None:
        self._handle(head=False)

    def do_HEAD(self) -> None:
        self._handle(head=True)

    def _send(self, status: int, body: bytes, ctype: str, head: bool) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _page(self, status: int, html: str, head: bool) -> None:
        self._send(status, html.encode("utf-8"), "text/html; charset=utf-8", head)

    def _handle(self, head: bool) -> None:
        port = self.viewer.port
        if self.headers.get("Host") not in (f"{HOST}:{port}", f"localhost:{port}"):
            self._send(HTTPStatus.FORBIDDEN, b"unexpected Host header\n", "text/plain", head)
            return
        parts = urlsplit(self.path)
        if parts.path.startswith("/static/"):
            name = parts.path.removeprefix("/static/")
            if name not in STATIC_TYPES:
                self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain", head)
                return
            self._send(HTTPStatus.OK, (STATIC / name).read_bytes(), STATIC_TYPES[name], head)
            return
        route = ROUTES.get(parts.path)
        if route is None:
            self._page(HTTPStatus.NOT_FOUND, error_page(404, f"no page {parts.path}"), head)
            return
        query = dict(parse_qsl(parts.query))
        try:
            ledger = Ledger(self.viewer.ledger_path, readonly=True)
        except LedgerError as e:
            self._page(HTTPStatus.SERVICE_UNAVAILABLE, error_page(503, str(e)), head)
            return
        try:
            html = layout(route(ledger, self.viewer.ctx, query))
        except NotFound as e:
            self._page(HTTPStatus.NOT_FOUND, error_page(404, str(e)), head)
            return
        except Exception as e:  # noqa: BLE001 - shown as a 500, traceback on the console
            traceback.print_exc(file=sys.stderr)
            self._page(HTTPStatus.INTERNAL_SERVER_ERROR, error_page(500, repr(e)), head)
            return
        finally:
            ledger.close()
        self._page(HTTPStatus.OK, html, head)


def serve(ctx: Context, ledger_path: Path, port: int) -> int:
    Ledger(ledger_path, readonly=True).close()  # fail now, not on the first page
    with ViewerServer(port, ctx, ledger_path) as srv:
        sealed = "" if ctx.unseal_holdout else " (holdout sealed)"
        print(f"run viewer on http://{HOST}:{srv.port}/{sealed}; Ctrl-C stops it", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0
