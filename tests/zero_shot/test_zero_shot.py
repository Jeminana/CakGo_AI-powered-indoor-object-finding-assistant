"""Model-free checks of the zero-shot adapters and scoring helpers (no torch or downloads needed).

  python -m pytest tests/zero_shot -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_zero_shot  # noqa: E402
import detectors  # noqa: E402
from detector import GroundingDinoDetector  # noqa: E402  (grounding_dino/detector.py)

WALLET, MOUSE = [0, 0, 100, 100], [2, 2, 101, 99]  # IoU ~0.95: the same object
ELSEWHERE = [300, 300, 400, 400]


def test_phrase_table_matches_grounding_dino():
    gdino = GroundingDinoDetector.from_config()
    prompts, look_alikes = detectors.load_prompts()
    assert detectors.phrase_table(prompts, look_alikes, "keys", "multi") == gdino.multi_phrases()
    for target in prompts:
        single = detectors.phrase_table(prompts, look_alikes, target, "single")
        assert [p for p, _ in single] == gdino.phrases_for(target)
        assert {c for _, c in single} == {target}
    assert detectors.phrase_table(prompts, look_alikes, "red mug", "multi") == [("red mug", "red mug")]


def test_look_alike_with_higher_score_takes_the_box():
    dets = [(0.6, WALLET, "wallet"), (0.8, MOUSE, None)]
    assert detectors.keep_best_phrase(dets, "wallet") == []


def test_target_keeps_box_when_it_scores_higher():
    dets = [(0.8, WALLET, "wallet"), (0.6, MOUSE, None), (0.9, ELSEWHERE, None)]
    assert detectors.keep_best_phrase(dets, "wallet") == [(0.8, WALLET)]


def test_two_phrases_of_one_target_merge_into_one_box():
    dets = [(0.5, WALLET, "glasses"), (0.7, MOUSE, "glasses"), (0.4, ELSEWHERE, "glasses")]
    assert detectors.keep_best_phrase(dets, "glasses") == [(0.7, MOUSE), (0.4, ELSEWHERE)]


def test_other_target_counts_like_a_look_alike():
    dets = [(0.5, WALLET, "keys"), (0.6, MOUSE, "wallet")]
    assert detectors.keep_best_phrase(dets, "keys") == []
    assert detectors.keep_best_phrase(dets, "wallet") == [(0.6, MOUSE)]


class FakeBoxes:
    def __init__(self, rows):
        self.conf = FakeTensor([r[0] for r in rows])
        self.xyxy = FakeTensor([r[1] for r in rows])
        self.cls = FakeTensor([float(r[2]) for r in rows])


class FakeTensor(list):
    def tolist(self):
        return list(self)


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeYoloe:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [FakeResult(self.rows)]


def test_yoloe_multi_keeps_only_target_classes(monkeypatch):
    from PIL import Image

    detector = detectors.YoloeDetector(prompt_mode="multi", device="cpu")
    names = [p for p, _ in detector.phrases("glasses")]
    rows = [(0.9, [10, 10, 50, 50], names.index("sunglasses")),
            (0.8, [60, 60, 90, 90], names.index("computer mouse")),
            (0.3, [5, 5, 20, 20], names.index("glasses"))]
    fake = FakeYoloe(rows)
    monkeypatch.setattr(detector, "_model_for", lambda phrases: fake)
    result = detector.detect(Image.new("RGB", (100, 100)), "glasses")
    assert [d["bbox"] for d in result["detections"]] == [[10, 10, 50, 50], [5, 5, 20, 20]]
    assert all(d["label"] == "glasses" for d in result["detections"])
    assert fake.calls[0]["agnostic_nms"] is True and fake.calls[0]["conf"] == 0.05


def test_centre_diagnostic_accepts_a_box_on_one_key_of_a_bunch():
    gt = {"k": {"target_category": "keys", "objects": [{"category": "keys", "bbox_xyxy": [0, 0, 100, 100]}]}}
    one_key = {"k": {"detections": [{"label": "keys", "confidence": 0.9, "bbox": [10, 10, 30, 30]}]}}
    elsewhere = {"k": {"detections": [{"label": "keys", "confidence": 0.9, "bbox": [90, 90, 200, 200]}]}}
    assert compare_zero_shot.top1_rate("keys", one_key, gt, 0.5) == "0/1"
    assert compare_zero_shot.top1_centre_hits("keys", one_key, gt, 0.5) == (1, 1)
    assert compare_zero_shot.top1_centre_hits("keys", elsewhere, gt, 0.5) == (0, 1)


def test_false_alarms_count_queries_for_absent_objects():
    gt = {"a": {"objects": [{"category": "wallet", "bbox_xyxy": [0, 0, 1, 1]}]},
          "n": {"objects": [{"category": None, "bbox_xyxy": [0, 0, 1, 1]}]}}
    preds = {"a": {"detections": [{"label": "wallet", "confidence": 0.9, "bbox": [0, 0, 1, 1]},
                                  {"label": "keys", "confidence": 0.4, "bbox": [0, 0, 1, 1]}]},
             "n": {"detections": [{"label": "keys", "confidence": 0.2, "bbox": [0, 0, 1, 1]}]}}
    assert compare_zero_shot.false_alarms(preds, gt, ["wallet", "keys"], 0.3) == (1, 3)
    assert compare_zero_shot.false_alarms(preds, gt, ["wallet", "keys"], 0.1) == (2, 3)


def test_instance_one_shot_helpers():
    import instance_one_shot as ios

    a2 = ios.photo_info("remote_control_cheolyeon_A2.jpg")
    assert (a2["object"], a2["design"], a2["instance"], a2["condition"]) == (
        "remote_control", "A", "remote_control_A", "cluttered")
    assert ios.photo_info("none_cheolyeon_N1.jpg")["instance"] is None
    assert ios.relation("wallet_A", ios.photo_info("wallet_cheolyeon_A3.jpg")) == "same object"
    assert ios.relation("wallet_A", ios.photo_info("wallet_cheolyeon_B1.jpg")) == "same type, other object"
    assert ios.relation("wallet_A", ios.photo_info("keys_cheolyeon_A1.jpg")) == "no such object"
    assert ios.centre_inside([10, 10, 30, 30], [0, 0, 100, 100])
    assert not ios.centre_inside([90, 90, 200, 200], [0, 0, 100, 100])
    assert ios.top([(0.2, [0, 0, 1, 1]), (0.7, [5, 5, 9, 9])]) == (0.7, [5, 5, 9, 9])
    assert ios.top([]) == (0.0, None)


def test_text_answers_are_reused_for_every_example():
    import instance_one_shot as ios

    photos = ["wallet_x_A1.jpg", "wallet_x_A2.jpg", "keys_x_A1.jpg"]
    gold = {"wallet_x_A1.jpg": [0, 0, 10, 10], "wallet_x_A2.jpg": [0, 0, 10, 10], "keys_x_A1.jpg": [0, 0, 10, 10]}
    rows = [{"model": "gdino_multi", "example": "", "query": q, "word": w, "score": 0.5, "box": [1, 1, 3, 3]}
            for q in photos for w in ("wallet", "keys")]
    records = ios.evaluate(rows, gold, photos)
    pairs = {(r["example"], r["query"], r["word"]) for r in records}
    assert ("wallet_x_A1.jpg", "wallet_x_A2.jpg", "wallet") in pairs
    assert ("keys_x_A1.jpg", "wallet_x_A1.jpg", "keys") in pairs
    assert all(r["example"] != r["query"] for r in records) and len(records) == 3 * 2


def test_valuables_summary_counts_roles_separately():
    import pytest

    pytest.importorskip("ultralytics")  # valuables_test imports the visual-prompt helpers
    import valuables_test as vt

    rows = [
        {"method": "text", "model": "YOLOE-26x", "object": "keys", "role": "search", "score": 0.9, "on_object": True},
        {"method": "text", "model": "YOLOE-26x", "object": "keys", "role": "search", "score": 0.9, "on_object": False},
        {"method": "text", "model": "YOLOE-26x", "object": "keys", "role": "none", "score": 0.3, "on_object": False},
        {"method": "text", "model": "YOLOE-26x", "object": "keys", "role": "other_keys", "score": 0.6, "on_object": False},
        {"method": "text", "model": "YOLOE-26x", "object": "earbud", "role": "other_keys", "score": 0.6, "on_object": False},
    ]
    text = vt.summarise(rows)
    assert "| text | YOLOE-26x | 0.25 | 1/2 (50%) | 2/2 (100%) | 1/1 (100%) |" in text


def test_shrink_puts_the_item_at_the_asked_size():
    import pytest

    pytest.importorskip("ultralytics")  # valuables_yoloe_scale imports YOLOE
    from PIL import Image
    import valuables_yoloe_scale as vs

    image = Image.new("RGB", (1280, 960), (255, 0, 0))
    canvas, box = vs.shrink(image, [100, 100, 500, 300], 40)  # long side 400 px -> 40 px
    assert canvas.size == image.size
    assert round(vs.long_side(box)) == 40
    assert canvas.getpixel((0, 0)) == vs.GREY
    assert canvas.getpixel((round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2))) == (255, 0, 0)
    assert vs.shrink(image, [100, 100, 500, 300], 800) == (image, [100, 100, 500, 300])  # never enlarged


def test_dino_verify_pipelines():
    import dino_verify as dv

    cands = [(0.6, [0, 0, 10, 10]), (0.3, [50, 50, 60, 60]), (0.04, [90, 90, 99, 99])]
    d = dv.decide(cands, [0.2, 0.7, 0.9])
    assert (d["top_score"], d["top_sim"]) == (0.6, 0.2)  # top box: Grounding DINO's first
    assert (d["rr_box"], d["rr_sim"]) == ([50, 50, 60, 60], 0.7)  # rerank skips the box under the 0.05 floor
    assert dv.decide([(0.2, [0, 0, 1, 1])], [0.9])["top_box"] is None  # below 0.35: no top box
    rec = {**d, "kind": "target", "gold": [45, 45, 65, 65]}
    assert dv.rates([rec], "gdino")["target"] == (0, 1)  # top box is not on the item
    assert dv.rates([rec], "rerank", 0.5)["target"] == (1, 1)
    assert dv.rates([rec], "filter", 0.1)["target"] == (0, 1)
    none = {**dv.decide(cands, [0.2, 0.3, 0.1]), "kind": "none", "gold": None}
    assert dv.rates([none], "filter", 0.25)["none"] == (0, 1) and dv.rates([none], "filter", 0.2)["none"] == (1, 1)
    # rerank: item found up to tau 0.7, false alarm up to 0.3 -> best from 0.35 to 0.7; ties go to the lowest
    assert dv.pick_tau([rec, none], "rerank") == 0.35
    assert dv.auc([0.9, 0.8], [0.1, 0.8]) == 0.875


def test_dino_verify_crop_is_a_padded_square():
    from PIL import Image
    import dino_verify as dv

    c = dv.crop(Image.new("RGB", (100, 100), (255, 255, 255)), [10, 40, 50, 60])  # 40 x 20 box, 10% margin
    assert c.size == (48, 48)
    assert c.getpixel((0, 0)) == dv.MEAN_RGB and c.getpixel((24, 24)) == (255, 255, 255)
