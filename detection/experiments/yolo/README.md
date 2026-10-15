# YOLO Experiment & Evaluation — Member 1

## Purpose
This directory houses the experimentation, benchmarking, and wrapper implementation for YOLO (Ultralytics) to assess its viability as the primary object detection engine for CakGo.

## Assigned Team Member
- **Member 1**
  - **Initial Responsibility:** YOLO testing, setup, and comparative benchmark execution.
  - **Post-Testing Responsibility:** Final model implementation, optimization, and integration into the core `detection/` pipeline.

## Dataset
The evaluation images and ground truth are **shared** with the Grounding DINO experiment and live in
[`tests/dataset/`](../../../tests/dataset/README.md). Nothing is copied into this folder.

- Images: `tests/dataset/<file_path>` for every row of `tests/dataset/manifest.csv`
- Ground truth: `tests/dataset/annotations/ground_truth.json` (`[x1, y1, x2, y2]` pixels)
- Sources, licenses and limitations: [`tests/dataset/dataset_info.md`](../../../tests/dataset/dataset_info.md)

If `manifest.csv` is missing, the subset has not been built yet; follow `tests/dataset/README.md`.

## Model and supported categories
- **Model:** Ultralytics `yolo26s.pt` by default (pretrained on COCO-80). Choose another with `--weights`, e.g. `yolo11s.pt`.
- **No training or fine-tuning** is done at this stage.
- Category support is defined in [`target_mapping.json`](target_mapping.json). Only exact semantic matches count:

| CakGo category | COCO class | Status |
|---|---|---|
| Wallet | — | **Unsupported**: COCO has no wallet class (`handbag` is not used as a substitute) |
| Keys | — | **Unsupported**: no key class |
| Glasses / Sunglasses | — | **Unsupported**: no eyeglasses class (`wine glass` is drinkware) |
| Remote control | `remote` (id 65) | Supported |

Unsupported categories are reported as `unsupported` in the metrics and are not scored as 0 %.
The model still runs on those images, and all its detections are saved.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
# (GPU) install a CUDA build of torch first: https://pytorch.org/get-started/locally/
pip install -r detection/experiments/yolo/requirements.txt
```
Weights are downloaded automatically into `detection/experiments/yolo/weights/` (gitignored) on the first run.

## Running the benchmark (from the repository root)
```bash
python tests/dataset/prepare_dataset.py verify                     # same files as Grounding DINO
python detection/experiments/yolo/run_inference.py                  # default: yolo26s.pt, imgsz 640, conf 0.25
python detection/experiments/yolo/evaluate.py --run <run_name>      # run_name is printed by run_inference.py
```
Useful options: `--weights`, `--imgsz`, `--conf` (use `0.001` for an untruncated AP50), `--iou` (NMS), `--device cpu|0`, `--warmup`.

### Outputs (`results/<run_name>/`)
| File | Content |
|---|---|
| `predictions.jsonl` | Per image: all detections (label, class id, confidence, `[x1, y1, x2, y2]` bbox) and timing in ms |
| `run_info.json` | Weights name and sha256; ultralytics, torch, CUDA and GPU versions; inference settings; manifest sha256 |
| `metrics.json`, `metrics.md` | Per category: TP/FP/FN, precision, recall, AP50, target-image detection rate, false positives on negative images; latency mean/median/p95 |

Timing: `wall_ms` is the `model.predict()` call (preprocess, inference and NMS) after the warm-up passes.
Image file reading is excluded. Results are produced only by actually running these scripts; nothing is pre-filled.

## Structure
```text
detection/experiments/yolo/
├── README.md            # This guide
├── requirements.txt     # Pinned ultralytics version
├── target_mapping.json  # CakGo category -> COCO class(es), unsupported classes documented
├── detector.py          # YoloDetector: predict() + detect(image, target) per API contract
├── run_inference.py     # Runs the model on tests/dataset/manifest.csv, saves predictions + run info
├── evaluate.py          # Scores predictions against tests/dataset/annotations/ground_truth.json
├── results/             # One folder per run
└── weights/             # Downloaded weights (gitignored)
```

## Expected Inputs & Outputs
- **Input:** Decoded frame (BGR/RGB image array) and target label string (a CakGo category such as `"remote_control"`, or a COCO label such as `"cell phone"`).
- **Output:** Structured detection result conforming to the API contract:
  ```python
  {
      "target": "cell phone",
      "detected": True,
      "detections": [
          {"label": "cell phone", "confidence": 0.92, "bbox": [150, 80, 320, 410]}
      ]
  }
  ```
  `bbox` is `[x1, y1, x2, y2]` in original image pixel coordinates. `detect()` raises `UnsupportedTargetError` for targets the pretrained label set cannot detect.

## Instructions for AI Coding Agents
1. Work exclusively inside `detection/experiments/yolo/`; read data from `tests/dataset/` without modifying or copying it.
2. Do not train or fine-tune, and do not invent test metrics. Results must come from `run_inference.py` and `evaluate.py`.
3. Keep the detection interface compatible with [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
4. Do not commit weights, datasets or API keys.
