import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import create_app, normalize_result, position_for
from backend.detectors import MockDetector


def png():
    output = io.BytesIO()
    Image.new("RGB", (300, 200)).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("mode", ["left", "center", "right", "missing"])
def test_detection_contract(mode):
    with TestClient(create_app(MockDetector(mode))) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        response = client.post("/api/detect", data={"target": "keys"},
                               files={"image": ("frame.png", png(), "image/png")})
        assert response.status_code == 200
        result = response.json()
        assert set(result) == {"target", "detected", "position", "image_width", "image_height", "detections", "message"}
        assert result["position"] == (None if mode == "missing" else mode)
        assert result["detected"] == (mode != "missing")
        assert (result["image_width"], result["image_height"]) == (300, 200)
        assert client.get("/voice-demo").status_code == 200
        assert client.get("/voice/guidance.js").status_code == 200


def test_invalid_input_errors():
    with TestClient(create_app(MockDetector())) as client:
        for data, files, status in [
            ({"target": "keys"}, {"image": ("a.png", b"invalid")}, 400),
            ({"target": " "}, {"image": ("a.png", png())}, 422),
            ({}, {"image": ("a.png", png())}, 422),
            ({"target": "keys"}, {"image": ("a.png", b"x" * (5*1024*1024+1))}, 413),
        ]:
            response = client.post("/api/detect", data=data, files=files)
            assert response.status_code == status
            assert "error" in response.json()


def test_boundaries_and_highest_confidence():
    assert position_for([90, 0, 110, 10], 300) == "center"
    assert position_for([190, 0, 210, 10], 300) == "center"
    result = normalize_result({"detections": [
        {"label": "keys", "confidence": .6, "bbox": [10, 10, 30, 30]},
        {"label": "keys", "confidence": .9, "bbox": [250, 10, 280, 30]},
    ]}, "keys", 300, 200)
    assert result["position"] == "right"
    assert len(result["detections"]) == 2


def test_model_failure_is_not_reported_as_missing():
    class Broken:
        def detect(self, image, target):
            raise RuntimeError("private detector detail")
    with TestClient(create_app(Broken())) as client:
        response = client.post("/api/detect", data={"target": "keys"},
                               files={"image": ("a.png", png())})
        assert response.status_code == 503
        assert "private" not in response.text
