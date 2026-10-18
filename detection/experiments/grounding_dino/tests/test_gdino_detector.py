from pathlib import Path

import pytest
from PIL import Image

from detector import GroundingDinoDetector, keep_target_boxes, load_json, phrase_token_spans, prompt_text

REPO_ROOT = Path(__file__).resolve().parents[4]


class FakeDetector(GroundingDinoDetector):
    """Skips the real model: returns canned raw detections and records the prompt phrases."""

    def __init__(self, raw, prompts=None):
        super().__init__(model_id="fake", prompts=prompts)
        self.raw = raw
        self.phrases = None

    def _predict(self, image, phrases):
        self.phrases = phrases
        return self.raw


def test_result_follows_api_contract():
    detector = FakeDetector([(0.41, [10.2, 20.7, 50.4, 60.4]), (0.86, [180, 210, 260, 290])])

    result = detector.detect(Image.new("RGB", (640, 480)), "keys")

    assert result == {
        "target": "keys",
        "detected": True,
        "detections": [
            {"label": "keys", "confidence": 0.86, "bbox": [180, 210, 260, 290]},
            {"label": "keys", "confidence": 0.41, "bbox": [10, 21, 50, 60]},
        ],
    }


def test_boxes_are_clamped_to_the_image_and_empty_boxes_dropped():
    detector = FakeDetector([(0.9, [-5, -3, 700, 500]), (0.5, [100, 100, 100.2, 140])])

    result = detector.detect(Image.new("RGB", (640, 480)), "wallet")

    assert [d["bbox"] for d in result["detections"]] == [[0, 0, 640, 480]]


def test_no_detection():
    result = FakeDetector([]).detect(Image.new("RGB", (64, 48)), "glasses")

    assert result == {"target": "glasses", "detected": False, "detections": []}


def test_configured_targets_expand_and_free_text_passes_through():
    detector = FakeDetector([], prompts={"keys": ["keys", "key ring"]})

    detector.detect(Image.new("RGB", (8, 8)), " Keys ")
    assert detector.phrases == ["keys", "key ring"]

    detector.detect(Image.new("RGB", (8, 8)), "black spectacles")
    assert detector.phrases == ["black spectacles"]


def test_accepts_rgb_array_frames():
    np = pytest.importorskip("numpy")
    detector = FakeDetector([(0.7, [1, 2, 3, 4])])

    result = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8), "wallet")

    assert result["detections"][0]["bbox"] == [1, 2, 3, 4]


def test_empty_target_is_rejected():
    with pytest.raises(ValueError):
        FakeDetector([]).detect(Image.new("RGB", (8, 8)), "  ")


def test_config_files_load_without_loading_the_model():
    detector = GroundingDinoDetector.from_config()

    assert detector.model_id == "IDEA-Research/grounding-dino-tiny"
    assert 0 < detector.box_threshold < 1 and 0 < detector.text_threshold < 1
    assert detector._model is None


def test_prompts_cover_the_shared_dataset_categories():
    categories = load_json(REPO_ROOT / "tests" / "dataset" / "class_mapping.json")["categories"]
    detector = GroundingDinoDetector.from_config()

    assert set(detector.prompts) == set(categories)
    assert prompt_text(detector.phrases_for("glasses")) == "glasses. sunglasses."
    assert prompt_text(detector.phrases_for("remote_control")) == "remote control."


def test_config_overrides_ignore_none():
    assert GroundingDinoDetector.from_config(device="cpu").device == "cpu"
    assert GroundingDinoDetector.from_config(device=None).device == "auto"


# --- prompt_mode "multi": all targets + look-alikes in one prompt ---------------------------------

def test_box_goes_to_its_best_phrase_and_look_alikes_are_dropped():
    categories = ["remote_control", "glasses", "glasses", None]  # remote control, glasses, sunglasses, mouse
    scores = [
        [0.40, 0.05, 0.02, 0.70],  # best phrase is the look-alike "computer mouse" -> dropped
        [0.62, 0.10, 0.01, 0.20],  # remote control wins -> kept
        [0.30, 0.10, 0.02, 0.05],  # remote control wins but below the threshold -> dropped
        [0.10, 0.20, 0.55, 0.01],  # "sunglasses" also counts for glasses
    ]
    boxes = [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]]

    assert keep_target_boxes(scores, categories, boxes, "remote_control", 0.35) == [(0.62, [1, 1, 2, 2])]
    assert keep_target_boxes(scores, categories, boxes, "glasses", 0.35) == [(0.55, [3, 3, 4, 4])]
    assert keep_target_boxes(scores, categories, boxes, "keys", 0.35) == []


def test_phrase_token_spans_follow_character_offsets():
    prompt = "remote control. computer mouse."
    offsets = [(0, 0), (0, 6), (7, 14), (14, 15), (16, 24), (25, 30), (30, 31), (0, 0)]

    assert phrase_token_spans(prompt, ["remote control", "computer mouse"], offsets) == [[1, 2], [4, 5]]


class FakeMultiDetector(GroundingDinoDetector):
    def __init__(self):
        super().__init__(model_id="fake", prompts={"keys": ["keys"]}, prompt_mode="multi", look_alikes=["pen"])
        self.calls = []

    def _predict(self, image, phrases):
        self.calls.append(("single", phrases))
        return []

    def _predict_multi(self, image, target):
        self.calls.append(("multi", target))
        return [(0.5, [1, 1, 5, 5])]


def test_multi_mode_is_used_for_configured_targets_and_free_text_stays_single():
    detector = FakeMultiDetector()

    assert detector.detect(Image.new("RGB", (8, 8)), "Keys")["detections"][0]["label"] == "Keys"
    detector.detect(Image.new("RGB", (8, 8)), "umbrella")

    assert detector.calls == [("multi", "keys"), ("single", ["umbrella"])]
    assert detector.prompt_for("keys") == "keys. pen."
    assert detector.prompt_for("umbrella") == "umbrella."


def test_unknown_prompt_mode_is_rejected():
    with pytest.raises(ValueError):
        GroundingDinoDetector(model_id="fake", prompt_mode="all")


def test_config_uses_multi_mode_with_look_alikes_that_are_not_targets():
    detector = GroundingDinoDetector.from_config()
    target_phrases = {p.lower() for phrases in detector.prompts.values() for p in phrases}

    assert detector.prompt_mode == "multi"
    assert detector.look_alikes and not target_phrases & {p.lower() for p in detector.look_alikes}
    assert GroundingDinoDetector.from_config(prompt_mode="single").prompt_for("glasses") == "glasses. sunglasses."
