from pathlib import Path

import pytest
from PIL import Image

from detector import GroundingDinoDetector, load_json, prompt_text

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
