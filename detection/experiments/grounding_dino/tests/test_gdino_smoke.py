"""Opt-in real-model smoke test: GDINO_SMOKE=1 pytest tests/test_gdino_smoke.py

Downloads IDEA-Research/grounding-dino-tiny (~690 MB) on first run. It uses a synthetic drawing,
not evaluation data, so it only checks that the wrapper runs and returns boxes in original pixels.
"""

import os

import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(not os.environ.get("GDINO_SMOKE"), reason="set GDINO_SMOKE=1 to run the real model")


def test_real_model_localizes_a_synthetic_object():
    from detector import GroundingDinoDetector
    from evaluate import iou

    circle = [460, 180, 580, 300]  # right third of a 640x480 frame
    image = Image.new("RGB", (640, 480), "white")
    ImageDraw.Draw(image).ellipse(circle, fill="red")

    result = GroundingDinoDetector.from_config().detect(image, "red circle")

    assert result["target"] == "red circle"
    assert result["detected"]
    top = result["detections"][0]
    assert all(isinstance(v, int) for v in top["bbox"])
    assert iou(top["bbox"], circle) > 0.5
