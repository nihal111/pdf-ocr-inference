"""Preview the exported website with a plain, loopback-only static file server."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def allowed(self):
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if (self.headers.get("Host") not in hosts
                or self.headers.get("Sec-Fetch-Site") == "cross-site"
                or (origin and origin not in {f"http://{host}" for host in hosts})):
            self.send_error(403, "Only same-origin localhost requests are allowed")
            return False
        relative = unquote(urlsplit(self.path).path).lstrip("/")
        root = Path(self.directory).resolve()
        if ".." in Path(relative).parts or not (root / relative).resolve().is_relative_to(root):
            self.send_error(404)
            return False
        return True

    def do_GET(self):
        if self.allowed():
            super().do_GET()

    def do_HEAD(self):
        if self.allowed():
            super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None


def make_server(root: Path = ROOT / "site", port: int = 8766):
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(root.resolve())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--directory", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    if not (args.directory / "index.html").is_file():
        parser.error("Export first: python -m lab.export_site")
    with make_server(args.directory, args.port) as server:
        print(f"PDF OCR inference: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
