"""Serve frontend/ and forward /api/ and /voice/ to the backend on the same origin.

The phone camera only works on HTTPS pages, and an HTTPS page cannot call an HTTP
backend. With the page and the API on one origin, a single HTTPS tunnel covers both
and no CORS setup is needed. Run from the repository root:

    python frontend/dev_server.py [--port 5173] [--backend http://127.0.0.1:8000]
"""
import argparse
import json
import urllib.error
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent
PROXY_PREFIXES = ("/api/", "/voice/")
RELAYED_HEADERS = ("Content-Type", "X-CakGo-Detector")
MAX_BODY = 6 * 1024 * 1024  # the backend rejects images over 5 MiB itself; this only caps memory
TIMEOUT_S = 60  # Grounding DINO on CPU takes about 15 s per request


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, backend, **kwargs):
        self.backend = backend  # set before super().__init__, which handles the request
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path.startswith(PROXY_PREFIXES):
            self.proxy()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith(PROXY_PREFIXES):
            self.proxy()
        else:
            self.send_error(405)

    def proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            return self.reply(413, {"error": "Image exceeds 5 MiB"})
        request = urllib.request.Request(self.backend + self.path, method=self.command,
                                         data=self.rfile.read(length) if length else None)
        if "Content-Type" in self.headers:
            request.add_header("Content-Type", self.headers["Content-Type"])
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                self.relay(response.status, response.headers, response.read())
        except urllib.error.HTTPError as error:  # the backend answered 4xx/5xx: pass it on
            self.relay(error.code, error.headers, error.read())
        except OSError:  # refused, timed out or unknown host
            self.reply(502, {"error": "Backend is not reachable"})

    def relay(self, status, headers, body):
        self.send_response(status)
        for name in RELAYED_HEADERS:
            if headers.get(name):
                self.send_header(name, headers[name])
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def reply(self, status, payload):
        self.relay(status, {"Content-Type": "application/json"}, json.dumps(payload).encode())


def make_server(host, port, backend):
    handler = partial(Handler, backend=backend.rstrip("/"), directory=str(FRONTEND_DIR))
    return ThreadingHTTPServer((host, port), handler)


def main():
    parser = argparse.ArgumentParser(description="Serve the CakGo frontend with the backend on one origin.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--backend", default="http://127.0.0.1:8000", help="backend base URL")
    args = parser.parse_args()
    server = make_server(args.host, args.port, args.backend)
    print(f"CakGo frontend: http://{args.host}:{args.port}  (/api/ and /voice/ -> {args.backend})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
