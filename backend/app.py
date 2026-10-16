"""Run from repository root: python -m uvicorn backend.app:app --reload."""
import asyncio
import io
import logging
import math
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.detectors import load_detector

LOG = logging.getLogger("uvicorn.error")
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
VOICE_DIR = Path(__file__).resolve().parents[1] / "voice"
MESSAGES = {
    "left": "찾는 물건이 화면 왼쪽에 보입니다.",
    "center": "찾는 물건이 화면 가운데에 보입니다.",
    "right": "찾는 물건이 화면 오른쪽에 보입니다.",
    None: "현재 카메라 화면에서 찾는 물건이 보이지 않습니다.",
}


def position_for(box, width):
    cx = (box[0] + box[2]) / 2
    return "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "center"


def normalize_result(raw, target, width, height):
    matches = []
    for item in raw["detections"]:
        box = [float(v) for v in item["bbox"]]
        score = float(item["confidence"])
        if len(box) != 4 or not all(math.isfinite(v) for v in box + [score]):
            raise ValueError("Invalid detector coordinates")
        if not 0 <= score <= 1 or not isinstance(item["label"], str):
            raise ValueError("Invalid detector score/label")
        x1, y1, x2, y2 = box
        box = [max(0, min(width, x1)), max(0, min(height, y1)),
               max(0, min(width, x2)), max(0, min(height, y2))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty detector box")
        matches.append({"label": item["label"], "confidence": score, "bbox": box})
    matches.sort(key=lambda d: d["confidence"], reverse=True)
    position = position_for(matches[0]["bbox"], width) if matches else None
    return {"target": target, "detected": bool(matches), "position": position,
            "image_width": width, "image_height": height,
            "detections": matches, "message": MESSAGES[position]}


def decode_image(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image exceeds 16 megapixels")
            return ImageOps.exif_transpose(image).convert("RGB")


def create_app(detector=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.detector = detector if detector is not None else load_detector()
        app.state.inference_lock = asyncio.Lock()
        LOG.info("Detector mode: %s", "injected" if detector else os.getenv("CAKGO_DETECTOR", "mock"))
        yield

    app = FastAPI(title="CakGo Backend and Voice", lifespan=lifespan)
    origins = [s.strip() for s in os.getenv("CAKGO_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if s.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], expose_headers=["X-CakGo-Detector"])

    @app.middleware("http")
    async def request_log(request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-CakGo-Detector"] = os.getenv("CAKGO_DETECTOR", "mock") if detector is None else "injected"
        # No images, query strings, targets, client identifiers or payloads in logs.
        LOG.info("%s %s %.0fms", request.method, response.status_code,
                 (time.perf_counter() - start) * 1000)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"error": "Required fields: image (JPEG/PNG), target (text)"}, status_code=422)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/detect")
    async def detect(image: UploadFile = File(...), target: str = Form(...)):
        target = target.strip()
        if not target or len(target) > 100:
            await image.close()
            return JSONResponse({"error": "target must contain 1–100 characters"}, status_code=422)
        try:
            data = await image.read(MAX_BYTES + 1)
        finally:
            await image.close()
        if len(data) > MAX_BYTES:
            return JSONResponse({"error": "Image exceeds 5 MiB"}, status_code=413)
        try:
            frame = await run_in_threadpool(decode_image, data)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return JSONResponse({"error": "Invalid image; use JPEG/PNG up to 16 megapixels"}, status_code=400)
        try:
            async with app.state.inference_lock:
                raw = await run_in_threadpool(app.state.detector.detect, frame, target)
            return normalize_result(raw, target, *frame.size)
        except ValueError:
            return JSONResponse({"error": "Unsupported target or invalid detector result"}, status_code=422)
        except Exception:
            LOG.error("Detector failed; no request content retained")
            return JSONResponse({"error": "Detection service unavailable"}, status_code=503)
        finally:
            frame.close()

    @app.get("/voice-demo")
    def voice_demo():
        return FileResponse(VOICE_DIR / "demo.html")

    @app.get("/voice/guidance.js")
    def voice_script():
        return FileResponse(VOICE_DIR / "guidance.js", media_type="text/javascript")

    return app


app = create_app()
