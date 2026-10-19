"""Send one test frame to /api/detect and print the result (used by start_demo.ps1).

A grey 640x480 JPEG is generated in memory, so no photo is needed. The first request also
warms up the model, so the first real phone request is not slow.

  python scripts/demo_request.py --url http://127.0.0.1:5173 --target keys
"""

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173", help="page origin (dev server or tunnel)")
    parser.add_argument("--target", default="keys")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    image = io.BytesIO()
    Image.new("RGB", (640, 480), (128, 128, 128)).save(image, "JPEG")
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"target\"\r\n\r\n{args.target}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"test.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + image.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(args.url.rstrip("/") + "/api/detect", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.loads(response.read())
            detector = response.headers.get("X-CakGo-Detector", "?")
    except (urllib.error.URLError, OSError) as error:
        print(f"[fail] {args.url}: {error}")
        return 1
    ms = (time.perf_counter() - start) * 1000
    print(f"[ok] detector={detector} target={args.target} detected={result.get('detected')} {ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
