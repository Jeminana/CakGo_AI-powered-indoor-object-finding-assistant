import json
import sys

import pytest

import evaluate as gd_eval

GT_IMAGES = [
    {"image_id": "remote_control_001", "target_category": "remote_control", "objects": [
        {"category": "remote_control", "bbox_xyxy": [10, 10, 60, 40]},
        {"category": "remote_control", "bbox_xyxy": [100, 100, 150, 130]}]},
    {"image_id": "remote_control_002", "target_category": "remote_control", "objects": [
        {"category": "remote_control", "bbox_xyxy": [20, 20, 80, 50]},
        {"category": None, "bbox_xyxy": [0, 0, 5, 5]}]},
    {"image_id": "keys_001", "target_category": "keys", "objects": [
        {"category": "keys", "bbox_xyxy": [30, 30, 70, 70]},
        {"category": "remote_control", "bbox_xyxy": [200, 200, 260, 230]}]},
    {"image_id": "negative_001", "target_category": "negative", "objects": [
        {"category": None, "bbox_xyxy": [5, 5, 50, 50]}]},
]
# image_id -> remote_control detections as (confidence, bbox)
DETECTIONS = {
    "remote_control_001": [(0.9, [11, 9, 61, 41]), (0.6, [12, 11, 59, 39]), (0.2, [101, 99, 149, 131])],
    "remote_control_002": [(0.4, [300, 300, 340, 330])],
    "keys_001": [(0.7, [201, 201, 259, 229])],
    "negative_001": [(0.5, [5, 5, 50, 50]), (0.1, [0, 0, 10, 10])],
}


# What detection/experiments/yolo/evaluate.py (PR #1, commit d299bb4) printed for this fixture, captured once on
# 2026-10-16. Pinned so this folder never imports the YOLO folder; if the shared scoring rules change, re-capture
# these numbers and update both evaluators together.
YOLO_REFERENCE = {
    (0.3, 0.5): {"status": "evaluated", "num_gt_boxes": 4, "num_predictions": 5, "true_positives": 2,
                 "false_positives": 3, "false_negatives": 2, "precision": 0.4, "recall": 0.5, "ap50": 0.625,
                 "num_target_images": 2, "target_images_detected": 1, "image_level_detection_rate": 0.5,
                 "negative_images": 1, "negative_images_with_false_positive": 1},
    (0.1, 0.5): {"status": "evaluated", "num_gt_boxes": 4, "num_predictions": 7, "true_positives": 3,
                 "false_positives": 4, "false_negatives": 1, "precision": 0.4286, "recall": 0.75, "ap50": 0.625,
                 "num_target_images": 2, "target_images_detected": 1, "image_level_detection_rate": 0.5,
                 "negative_images": 1, "negative_images_with_false_positive": 1},
    (0.5, 0.75): {"status": "evaluated", "num_gt_boxes": 4, "num_predictions": 4, "true_positives": 2,
                  "false_positives": 2, "false_negatives": 2, "precision": 0.5, "recall": 0.5, "ap50": 0.625,
                  "num_target_images": 2, "target_images_detected": 1, "image_level_detection_rate": 0.5,
                  "negative_images": 1, "negative_images_with_false_positive": 1},
}
YOLO_REFERENCE_LATENCY = ([12.0, 3.5, 7.25, 30.0, 9.0],
                          {"mean": 12.35, "median": 9.0, "p95": 30.0, "min": 3.5, "max": 30.0})


def detections(image_id):
    return [{"label": "remote_control", "confidence": c, "bbox": b} for c, b in DETECTIONS[image_id]]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


@pytest.fixture
def ground_truth(tmp_path):
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps({"categories": ["remote_control"], "images": GT_IMAGES}), encoding="utf-8")
    return path


def test_hand_computed_metrics():
    predictions = {i: {"detections": detections(i)} for i in DETECTIONS}
    gt_images = {g["image_id"]: g for g in GT_IMAGES}

    m = gd_eval.score_category("remote_control", predictions, gt_images, conf_thr=0.3, iou_thr=0.5)

    assert (m["num_gt_boxes"], m["true_positives"], m["false_positives"], m["false_negatives"]) == (4, 2, 3, 2)
    assert (m["precision"], m["recall"], m["ap50"]) == (0.4, 0.5, 0.625)
    assert (m["target_images_detected"], m["num_target_images"]) == (1, 2)
    assert m["negative_images_with_false_positive"] == 1


@pytest.mark.parametrize(("conf_thr", "iou_thr"), list(YOLO_REFERENCE))
def test_scores_match_the_yolo_evaluator_reference(conf_thr, iou_thr):
    """Same predictions, same rules: our numbers equal what the YOLO evaluator produced."""
    predictions = {i: {"detections": detections(i)} for i in DETECTIONS}
    gt_images = {g["image_id"]: g for g in GT_IMAGES}

    ours = gd_eval.score_category("remote_control", predictions, gt_images, conf_thr, iou_thr)

    assert ours == YOLO_REFERENCE[(conf_thr, iou_thr)]


def test_latency_stats_match_the_yolo_evaluator_reference():
    values, expected = YOLO_REFERENCE_LATENCY
    assert gd_eval.latency_stats(values) == expected


def test_main_writes_metrics_and_per_query_latency(tmp_path, ground_truth, monkeypatch):
    run_dir = tmp_path / "results" / "run"
    write_jsonl(run_dir / "predictions.jsonl", [
        {"image_id": i, "detections": detections(i), "timing": {"wall_ms": {"remote_control": 100.0 + n}}}
        for n, i in enumerate(DETECTIONS)])
    (run_dir / "run_info.json").write_text(json.dumps({
        "model": {"model_id": "fake"}, "prompts": {"remote_control": "remote control."},
        "inference_settings": {"box_threshold": 0.05}}), encoding="utf-8")
    monkeypatch.setattr(gd_eval, "GROUND_TRUTH_PATH", ground_truth)
    monkeypatch.setattr(gd_eval, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--run", "run", "--conf-threshold", "0.3"])

    gd_eval.main()

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["categories"]["remote_control"]["ap50"] == 0.625
    assert metrics["categories"]["remote_control"]["prompt"] == "remote control."
    assert metrics["latency_ms"]["wall_ms_per_query"]["mean"] == 101.5
    assert (run_dir / "metrics.md").exists()
