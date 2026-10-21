"""Zero-shot YOLOE-26 and SAM 3 detectors with the shared ``detect(image, target)`` interface.

Both use the same prompt words as Grounding DINO (detection/experiments/grounding_dino/config/
prompts.json and look_alikes.json) and return results in Grounding DINO's format (``build_result``:
label = target, ``[x1, y1, x2, y2]`` original-image pixels, highest confidence first).

prompt_mode "single": only the target's phrases. "multi": every target phrase plus the look-alike
phrases; a box counts for the target only if a target phrase is the phrase it matches best, which is
Grounding DINO's multi rule (see grounding_dino/MULTI_PROMPT.md).

- YOLOE scores all phrases of one prompt in one pass: each box gets its best phrase, and class-agnostic
  NMS keeps one box per object, so the rule holds as in Grounding DINO.
- SAM 3 scores one phrase per pass. Boxes from different phrases with IoU >= SAME_OBJECT_IOU are treated
  as the same object, and the highest-scoring phrase wins (``keep_best_phrase``).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GDINO_DIR = REPO_ROOT / "detection" / "experiments" / "grounding_dino"
sys.path.insert(0, str(GDINO_DIR))
from detector import CONFIG_DIR, build_result, load_json, to_pil_rgb  # noqa: E402
from evaluate import iou  # noqa: E402

SAME_OBJECT_IOU = 0.7  # also the NMS IoU of YOLOE (Ultralytics default)


def load_prompts(config_dir: Path = CONFIG_DIR) -> tuple[dict, list]:
    """Grounding DINO's prompt words: {category: [phrases]} and the look-alike phrases."""
    prompts = {name.lower(): phrases for name, phrases in load_json(Path(config_dir) / "prompts.json").items()}
    return prompts, load_json(Path(config_dir) / "look_alikes.json")["phrases"]


def phrase_table(prompts: dict, look_alikes: list, target: str, prompt_mode: str) -> list[tuple[str, str | None]]:
    """``(phrase, category)`` pairs sent to the model for ``target``; category ``None`` = look-alike.
    Same phrases and order as GroundingDinoDetector.phrases_for / multi_phrases."""
    if prompt_mode == "multi" and target in prompts:
        return [(p, c) for c, ps in prompts.items() for p in ps] + [(p, None) for p in look_alikes]
    return [(p, target) for p in prompts.get(target, [target])]


def nms(dets: list[tuple[float, list]], iou_thr: float) -> list[tuple[float, list]]:
    """Keep the highest-scoring box of every group of boxes that overlap at IoU >= ``iou_thr``."""
    kept = []
    for score, box in sorted(dets, key=lambda d: d[0], reverse=True):
        if all(iou(box, k) < iou_thr for _, k in kept):
            kept.append((score, box))
    return kept


def keep_best_phrase(dets: list[tuple[float, list, str | None]], target: str,
                     iou_thr: float = SAME_OBJECT_IOU) -> list[tuple[float, list]]:
    """``dets`` are ``(score, box, category)`` from separate one-phrase passes. A target box survives only
    if no box of another category (other target or look-alike) overlaps it at IoU >= ``iou_thr`` with a
    higher score; the surviving target boxes are merged with NMS."""
    survivors = [(s, b) for s, b, c in dets if c == target and not any(
        c2 != target and s2 > s and iou(b, b2) >= iou_thr for s2, b2, c2 in dets)]
    return nms(survivors, iou_thr)


class YoloeDetector:
    family = "YOLOE-26 (Ultralytics, text prompts, zero-shot, no fine-tuning)"

    def __init__(self, weights: str = "yoloe-26s-seg.pt", prompt_mode: str = "single", box_threshold: float = 0.05,
                 device: str = "auto", imgsz: int = 640, iou_thr: float = SAME_OBJECT_IOU):
        if prompt_mode not in ("single", "multi"):
            raise ValueError("prompt_mode must be 'single' or 'multi'")
        self.model_id = weights
        self.prompt_mode = prompt_mode
        self.box_threshold = box_threshold
        self.device = device
        self.imgsz = imgsz
        self.iou_thr = iou_thr
        self.prompts, self.look_alikes = load_prompts()
        self._models = {}  # phrase list -> YOLOE model with those classes set once, like the app at startup

    def phrases(self, target: str) -> list[tuple[str, str | None]]:
        return phrase_table(self.prompts, self.look_alikes, target.lower(), self.prompt_mode)

    def prompt_for(self, target: str) -> str:
        return " | ".join(p for p, _ in self.phrases(target))

    def _model_for(self, phrases: list[str]):
        key = tuple(phrases)
        if key not in self._models:
            import torch
            from ultralytics import YOLOE

            if self.device in ("auto", "cuda"):
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = YOLOE(self.model_id)
            model.set_classes(list(phrases))
            self._models[key] = model
        return self._models[key]

    def prepare(self, categories: list[str]) -> None:
        """Load the models and encode the prompt words before anything is timed."""
        for category in categories:
            self._model_for([p for p, _ in self.phrases(category)])

    def weights_path(self) -> str | None:
        model = next(iter(self._models.values()), None)
        return getattr(model, "ckpt_path", None) if model is not None else None

    def detect(self, image, target: str) -> dict:
        target = target.strip()
        if not target:
            raise ValueError("target must be a non-empty string")
        image = to_pil_rgb(image)
        table = self.phrases(target)
        model = self._model_for([p for p, _ in table])
        result = model.predict(image, conf=self.box_threshold, iou=self.iou_thr, imgsz=self.imgsz,
                               agnostic_nms=True, device=self.device, verbose=False)[0]
        boxes = result.boxes
        raw = [(score, box) for score, box, k in zip(boxes.conf.tolist(), boxes.xyxy.tolist(), boxes.cls.tolist())
               if table[int(k)][1] == target.lower()]
        return build_result(target, raw, image.size)


class Sam3Detector:
    family = "SAM 3 (Hugging Face transformers, text prompts, zero-shot, no fine-tuning)"

    def __init__(self, model_id: str = "facebook/sam3", prompt_mode: str = "single", box_threshold: float = 0.05,
                 device: str = "auto"):
        if prompt_mode not in ("single", "multi"):
            raise ValueError("prompt_mode must be 'single' or 'multi'")
        self.model_id = model_id
        self.prompt_mode = prompt_mode
        self.box_threshold = box_threshold
        self.device = device
        self.prompts, self.look_alikes = load_prompts()
        self._model = None
        self._processor = None

    def phrases(self, target: str) -> list[tuple[str, str | None]]:
        return phrase_table(self.prompts, self.look_alikes, target.lower(), self.prompt_mode)

    def prompt_for(self, target: str) -> str:
        return " | ".join(p for p, _ in self.phrases(target))

    def prepare(self, categories: list[str]) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam3Processor.from_pretrained(self.model_id)
        self._model = Sam3Model.from_pretrained(self.model_id, dtype=torch.float32).to(self.device).eval()

    def detect(self, image, target: str) -> dict:
        """One request: the image is encoded once, then each phrase is decoded on the same features."""
        import torch

        target = target.strip()
        if not target:
            raise ValueError("target must be a non-empty string")
        self.prepare([])
        image = to_pil_rgb(image)
        image_inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        sizes = image_inputs.get("original_sizes").tolist()
        dets = []
        with torch.inference_mode():
            vision_embeds = self._model.get_vision_features(pixel_values=image_inputs.pixel_values)
            for phrase, category in self.phrases(target):
                text_inputs = self._processor(text=phrase, return_tensors="pt").to(self.device)
                outputs = self._model(vision_embeds=vision_embeds, **text_inputs)
                found = self._processor.post_process_instance_segmentation(
                    outputs, threshold=self.box_threshold, mask_threshold=0.5, target_sizes=sizes)[0]
                dets += [(score, box, category) for score, box in zip(found["scores"].tolist(), found["boxes"].tolist())]
        return build_result(target, keep_best_phrase(dets, target.lower()), image.size)


def create(model: str, weights: str | None, prompt_mode: str, box_threshold: float, device: str):
    """Detector for ``run_zero_shot.py``: 'gdino' (the app's detector), 'yoloe' or 'sam3'."""
    if model == "gdino":
        from detector import GroundingDinoDetector

        detector = GroundingDinoDetector.from_config(device=None if device == "auto" else device,
                                                     box_threshold=box_threshold, prompt_mode=prompt_mode)
        detector.family = "Grounding DINO (Hugging Face transformers, zero-shot, no fine-tuning)"
        detector.prepare = lambda categories: None  # loads on the first (untimed) warm-up query
        return detector
    if model == "yoloe":
        return YoloeDetector(weights or "yoloe-26s-seg.pt", prompt_mode, box_threshold, device)
    if model == "sam3":
        return Sam3Detector(weights or "facebook/sam3", prompt_mode, box_threshold, device)
    raise ValueError(f"unknown model {model!r}")
