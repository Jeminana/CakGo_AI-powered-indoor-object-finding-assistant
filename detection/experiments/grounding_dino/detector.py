"""Grounding DINO wrapper exposing the shared ``detect(image, target)`` interface.

Results follow docs/API_CONTRACT.md: ``bbox`` is ``[x1, y1, x2, y2]`` in original image
pixels and detections are sorted by confidence, highest first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_pil_rgb(image):
    """Accept a PIL image or an HxWx3 uint8 RGB array (decoded camera frame)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def prompt_text(phrases):
    """Grounding DINO expects lowercase phrases, each terminated by a period: "glasses. sunglasses." """
    return " ".join(f"{phrase.lower().strip().rstrip('.')}." for phrase in phrases)


def build_result(target, raw_detections, image_size):
    """Turn raw ``(score, [x1, y1, x2, y2])`` pairs into the API contract's result shape."""
    width, height = image_size
    detections = []
    for score, box in sorted(raw_detections, key=lambda d: d[0], reverse=True):
        x1, y1, x2, y2 = (round(v) for v in box)
        x1, x2 = min(max(x1, 0), width), min(max(x2, 0), width)
        y1, y2 = min(max(y1, 0), height), min(max(y2, 0), height)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({"label": target, "confidence": round(float(score), 4), "bbox": [x1, y1, x2, y2]})
    return {"target": target, "detected": bool(detections), "detections": detections}


class GroundingDinoDetector:
    def __init__(self, model_id, device="auto", box_threshold=0.35, text_threshold=0.25, prompts=None):
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompts = {name.lower(): phrases for name, phrases in (prompts or {}).items()}
        self._model = None
        self._processor = None

    @classmethod
    def from_config(cls, config_dir=CONFIG_DIR, **overrides):
        """Build from config/model.json and config/prompts.json; ``None`` overrides are ignored."""
        config_dir = Path(config_dir)
        settings = load_json(config_dir / "model.json")
        settings.update({key: value for key, value in overrides.items() if value is not None})
        return cls(prompts=load_json(config_dir / "prompts.json"), **settings)

    def phrases_for(self, target):
        """Configured prompt phrases for a target name, or the target text itself."""
        return self.prompts.get(target.lower(), [target])

    def detect(self, image, target):
        target = target.strip()
        if not target:
            raise ValueError("target must be a non-empty string")
        image = to_pil_rgb(image)
        return build_result(target, self._predict(image, self.phrases_for(target)), image.size)

    def _load(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()

    def _predict(self, image, phrases):
        """Run the model; returns ``[(score, [x1, y1, x2, y2]), ...]`` in original image pixels."""
        import torch

        if self._model is None:
            self._load()
        inputs = self._processor(images=image, text=prompt_text(phrases), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        return [(float(score), box.tolist()) for score, box in zip(result["scores"], result["boxes"])]


_default_detector = None


def detect(image, target):
    """Shared detector interface for the backend (see docs/API_CONTRACT.md)."""
    global _default_detector
    if _default_detector is None:
        _default_detector = GroundingDinoDetector.from_config()
    return _default_detector.detect(image, target)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python detector.py IMAGE_PATH TARGET")
    print(json.dumps(detect(Image.open(sys.argv[1]), sys.argv[2]), ensure_ascii=False, indent=2))
