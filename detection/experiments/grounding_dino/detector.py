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


def phrase_token_spans(prompt, phrases, offsets):
    """Token indices of each phrase in ``prompt``, from the tokenizer's (start, end) character offsets."""
    spans, pos = [], 0
    for phrase in phrases:
        start = prompt.index(f"{phrase}.", pos)
        end = pos = start + len(phrase)
        spans.append([i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end])
    return spans


def keep_target_boxes(phrase_scores, phrase_categories, boxes, target, threshold):
    """Multi-prompt rule: each box goes to the phrase it matches best; keep it only if that phrase
    belongs to ``target`` and its score reaches ``threshold``.

    ``phrase_scores`` is one list of per-phrase scores per box; ``phrase_categories`` gives the
    category of each phrase (``None`` for a look-alike).
    """
    kept = []
    for scores, box in zip(phrase_scores, boxes):
        best = max(range(len(scores)), key=scores.__getitem__)
        if phrase_categories[best] == target and scores[best] >= threshold:
            kept.append((float(scores[best]), list(box)))
    return kept


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
    def __init__(self, model_id, device="auto", box_threshold=0.35, text_threshold=0.25, prompts=None,
                 prompt_mode="single", look_alikes=None):
        if prompt_mode not in ("single", "multi"):
            raise ValueError("prompt_mode must be 'single' or 'multi'")
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompts = {name.lower(): phrases for name, phrases in (prompts or {}).items()}
        self.prompt_mode = prompt_mode
        self.look_alikes = list(look_alikes or [])
        self._model = None
        self._processor = None
        self._multi_spans = None

    @classmethod
    def from_config(cls, config_dir=CONFIG_DIR, **overrides):
        """Build from config/model.json, prompts.json and look_alikes.json; ``None`` overrides are ignored."""
        config_dir = Path(config_dir)
        settings = load_json(config_dir / "model.json")
        settings.update({key: value for key, value in overrides.items() if value is not None})
        look_alikes_path = config_dir / "look_alikes.json"
        look_alikes = load_json(look_alikes_path)["phrases"] if look_alikes_path.exists() else []
        return cls(prompts=load_json(config_dir / "prompts.json"), look_alikes=look_alikes, **settings)

    def phrases_for(self, target):
        """Configured prompt phrases for a target name, or the target text itself."""
        return self.prompts.get(target.lower(), [target])

    def multi_phrases(self):
        """All configured target phrases followed by the look-alike phrases, with their categories."""
        phrases = [(p, name) for name, ps in self.prompts.items() for p in ps]
        return phrases + [(p, None) for p in self.look_alikes]

    def prompt_for(self, target):
        """The exact text prompt sent to the model for ``target``."""
        if self.prompt_mode == "multi" and target.lower() in self.prompts:
            return prompt_text([p for p, _ in self.multi_phrases()])
        return prompt_text(self.phrases_for(target))

    def detect(self, image, target):
        target = target.strip()
        if not target:
            raise ValueError("target must be a non-empty string")
        image = to_pil_rgb(image)
        if self.prompt_mode == "multi" and target.lower() in self.prompts:
            raw = self._predict_multi(image, target.lower())
        else:  # single mode, or a free-text target that has no configured phrases
            raw = self._predict(image, self.phrases_for(target))
        return build_result(target, raw, image.size)

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

    def _predict_multi(self, image, target):
        """One prompt with every target and look-alike phrase. A box counts for ``target`` only if
        ``target`` is the phrase it matches best, so e.g. a mouse goes to "computer mouse" instead of
        "remote control". Returns ``[(score, [x1, y1, x2, y2]), ...]`` in original image pixels."""
        import torch

        if self._model is None:
            self._load()
        phrases = self.multi_phrases()
        prompt = prompt_text([p for p, _ in phrases])
        if self._multi_spans is None:
            offsets = self._processor.tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
            self._multi_spans = phrase_token_spans(prompt, [p.lower() for p, _ in phrases], offsets)
        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        token_scores = outputs.logits.sigmoid()[0]  # (queries, tokens)
        phrase_scores = torch.stack([token_scores[:, span].max(-1).values for span in self._multi_spans], -1)
        cx, cy, w, h = outputs.pred_boxes[0].unbind(-1)  # normalized centre/size
        boxes = torch.stack([(cx - w / 2) * image.width, (cy - h / 2) * image.height,
                             (cx + w / 2) * image.width, (cy + h / 2) * image.height], -1)
        return keep_target_boxes(phrase_scores.tolist(), [c for _, c in phrases], boxes.tolist(),
                                 target, self.box_threshold)


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

def create_detector():
    return GroundingDinoDetector.from_config()
