"""Pretrained YOLO wrapper for the CakGo YOLO experiment (Member 1).

`detect(image, target)` follows docs/API_CONTRACT.md: bbox is [x1, y1, x2, y2]
in ORIGINAL image pixel coordinates. No training or fine-tuning happens here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

EXPERIMENT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = EXPERIMENT_DIR / "weights"
TARGET_MAPPING_PATH = EXPERIMENT_DIR / "target_mapping.json"
DEFAULT_WEIGHTS = "yolo26s.pt"


class UnsupportedTargetError(ValueError):
    """Raised when the pretrained label set has no class for the requested target."""


def resolve_weights(weights: str) -> Path:
    """Bare model names (e.g. 'yolo26s.pt') are downloaded into ./weights/ (gitignored)."""
    path = Path(weights)
    if path.parent == Path("."):
        WEIGHTS_DIR.mkdir(exist_ok=True)
        path = WEIGHTS_DIR / path.name
    return path


class YoloDetector:
    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str | None = None,
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.7, max_det: int = 300):
        self.weights_path = resolve_weights(weights)
        self.model = YOLO(str(self.weights_path))
        self.device = device
        self.settings = {"imgsz": imgsz, "conf": conf, "iou": iou, "max_det": max_det}
        with open(TARGET_MAPPING_PATH, encoding="utf-8") as f:
            self.target_mapping = json.load(f)["categories"]

    @property
    def names(self) -> dict[int, str]:
        return self.model.names

    def predict(self, image: np.ndarray) -> tuple[list[dict], dict]:
        """Run the model on one BGR image. Returns (all detections, timing in ms)."""
        start = time.perf_counter()
        result = self.model.predict(image, device=self.device, verbose=False, **self.settings)[0]
        wall_ms = (time.perf_counter() - start) * 1000
        detections = [
            {
                "label": self.names[int(cls)],
                "class_id": int(cls),
                "confidence": round(float(conf), 4),
                "bbox": [round(float(v), 1) for v in xyxy],
            }
            for xyxy, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                       result.boxes.cls.tolist())
        ]
        timing = {"wall_ms": round(wall_ms, 2), **{f"{k}_ms": round(v, 2) for k, v in result.speed.items()}}
        return detections, timing

    def labels_for(self, target: str) -> list[str]:
        """Map a CakGo category (or a raw COCO label) to the model labels that count as a match."""
        if target in self.target_mapping:
            entry = self.target_mapping[target]
            if not entry["supported"]:
                raise UnsupportedTargetError(f"'{target}': {entry['note']}")
            return entry["coco_classes"]
        if target in self.names.values():
            return [target]
        raise UnsupportedTargetError(f"'{target}' is not a CakGo category or a label of this model.")

    def detect(self, image: np.ndarray, target: str) -> dict:
        labels = set(self.labels_for(target))
        detections, _ = self.predict(image)
        matches = sorted((d for d in detections if d["label"] in labels),
                         key=lambda d: d["confidence"], reverse=True)
        return {
            "target": target,
            "detected": bool(matches),
            "detections": [{k: d[k] for k in ("label", "confidence", "bbox")} for d in matches],
        }
