"""Adapters receive RGB Pillow images and return target-matched detections."""
import importlib
import os


class MockDetector:
    """Deterministic integration fixture, NOT an object recognition model."""

    def __init__(self, position="left"):
        if position not in {"left", "center", "right", "missing"}:
            raise ValueError("CAKGO_MOCK_POSITION must be left, center, right or missing")
        self.position = position

    def detect(self, image, target):
        if self.position == "missing":
            return {"detections": []}
        w, h = image.size
        cx = {"left": .2, "center": .5, "right": .8}[self.position] * w
        return {"detections": [{"label": target, "confidence": .89,
                               "bbox": [cx - .1*w, .3*h, cx + .1*w, .7*h]}]}


class YoloAdapter:
    def __init__(self):
        from detection.experiments.yolo.detector import YoloDetector
        self.engine = YoloDetector(weights=os.getenv("CAKGO_YOLO_WEIGHTS", "yolo26s.pt"),
                                   device=os.getenv("CAKGO_DEVICE") or None)

    def detect(self, image, target):
        import numpy as np
        # Existing experiment explicitly expects BGR, unlike the shared RGB interface.
        return self.engine.detect(np.asarray(image)[:, :, ::-1].copy(), target)


def load_detector():
    mode = os.getenv("CAKGO_DETECTOR", "mock")
    if mode == "mock":
        return MockDetector(os.getenv("CAKGO_MOCK_POSITION", "left"))
    if mode == "yolo":
        return YoloAdapter()
    if mode == "custom":
        # Team-selected DINO/unified detector can be plugged in without changing the API.
        module, factory = os.environ["CAKGO_DETECTOR_FACTORY"].split(":", 1)
        return getattr(importlib.import_module(module), factory)()
    raise ValueError("CAKGO_DETECTOR must be mock, yolo or custom")
