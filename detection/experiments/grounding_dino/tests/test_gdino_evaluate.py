import json

import pytest
from PIL import Image

from evaluate import evaluate, iou, load_manifest


def result(*boxes):
    detections = [{"label": "keys", "confidence": 0.9, "bbox": list(box)} for box in boxes]
    return {"target": "keys", "detected": bool(detections), "detections": detections}


class CannedDetector:
    """No model: answers by image size, so each synthetic image gets a fixed result."""

    def __init__(self, results_by_size):
        self.results_by_size = results_by_size
        self.calls = 0

    def detect(self, image, target):
        self.calls += 1
        return self.results_by_size[image.size]


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_evaluate_reports_detection_recall_false_positives_and_tags(tmp_path):
    sizes = {"hit.png": (100, 100), "miss.png": (101, 100), "clean.png": (102, 100), "false_alarm.png": (103, 100)}
    for name, size in sizes.items():
        Image.new("RGB", size).save(tmp_path / name)
    samples = [
        {"image": "hit.png", "target": "keys", "boxes": [[10, 10, 50, 50]], "tags": ["dim"]},
        {"image": "miss.png", "target": "keys", "boxes": [[10, 10, 50, 50]]},
        {"image": "clean.png", "target": "keys", "boxes": [], "tags": ["dim"]},
        {"image": "false_alarm.png", "target": "keys", "boxes": []},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    detector = CannedDetector(
        {
            (100, 100): result([12, 12, 48, 50]),  # right place
            (101, 100): result([60, 60, 90, 90]),  # detected, wrong place
            (102, 100): result(),
            (103, 100): result([0, 0, 20, 20]),  # false alarm
        }
    )

    report = evaluate(detector, load_manifest(manifest), iou_threshold=0.5, warmup=1)

    overall = report["overall"]
    assert (overall["samples"], overall["positives"], overall["negatives"]) == (4, 2, 2)
    assert overall["detection_rate"] == 1.0
    assert overall["top1_recall"] == 0.5
    assert overall["false_positive_rate"] == 0.5
    assert overall["latency_ms"]["p95"] >= overall["latency_ms"]["p50"] >= 0
    assert report["by_tag"]["dim"]["top1_recall"] == 1.0
    assert report["by_tag"]["dim"]["false_positive_rate"] == 0.0
    assert detector.calls == 5  # 1 warm-up + 4 timed


def test_manifest_requires_image_and_target(tmp_path):
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"image": "a.png"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_manifest(manifest)
