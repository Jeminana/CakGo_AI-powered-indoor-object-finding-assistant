"""Run from repository root: python -m pytest frontend/test_dev_server.py -q"""
import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dev_server import make_server


class FakeBackend(BaseHTTPRequestHandler):
    """Echoes what reached it, so the tests can see what the proxy forwarded."""

    def do_GET(self):
        self.answer(200, "text/javascript", f"// {self.path}".encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.dumps({"path": self.path, "content_type": self.headers["Content-Type"],
                              "body": body.decode()}).encode()
        self.answer(200 if self.path == "/api/detect" else 422, "application/json", payload)

    def answer(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-CakGo-Detector", "mock")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def start(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def servers():
    started = []

    def run(backend_url=None):
        if backend_url is None:
            backend = ThreadingHTTPServer(("127.0.0.1", 0), FakeBackend)
            started.append(backend)
            backend_url = start(backend)
        proxy = make_server("127.0.0.1", 0, backend_url)
        started.append(proxy)
        return start(proxy)

    yield run
    for server in started:
        server.shutdown()
        server.server_close()


def post(url, body=b"--x\r\n", content_type="multipart/form-data; boundary=x"):
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    return urllib.request.urlopen(request, timeout=10)


def test_serves_the_page(servers):
    with urllib.request.urlopen(servers() + "/", timeout=10) as response:
        assert response.status == 200
        assert b"<title>CakGo" in response.read()


def test_forwards_detect_request_and_detector_header(servers):
    with post(servers() + "/api/detect") as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-CakGo-Detector"] == "mock"
        assert json.load(response) == {"path": "/api/detect", "body": "--x\r\n",
                                       "content_type": "multipart/form-data; boundary=x"}


def test_forwards_voice_script(servers):
    with urllib.request.urlopen(servers() + "/voice/guidance.js", timeout=10) as response:
        assert response.headers["Content-Type"] == "text/javascript"
        assert response.read() == b"// /voice/guidance.js"


def test_passes_backend_errors_through(servers):
    with pytest.raises(urllib.error.HTTPError) as error:
        post(servers() + "/api/other")
    assert error.value.code == 422
    assert json.load(error.value)["path"] == "/api/other"


def test_unreachable_backend_is_502(servers):
    with socket.socket() as probe:  # a port with nothing listening on it
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    with pytest.raises(urllib.error.HTTPError) as error:
        post(servers(f"http://127.0.0.1:{closed_port}") + "/api/detect")
    assert error.value.code == 502
    assert json.load(error.value) == {"error": "Backend is not reachable"}


def test_oversized_upload_is_rejected_before_reading(servers):
    host, port = servers().removeprefix("http://").split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=10)
    connection.putrequest("POST", "/api/detect")
    connection.putheader("Content-Length", str(7 * 1024 * 1024))
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 413
    assert json.load(response) == {"error": "Image exceeds 5 MiB"}
    connection.close()
