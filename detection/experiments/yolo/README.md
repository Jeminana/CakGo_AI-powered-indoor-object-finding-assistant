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
- The pretrained model is used as-is for the baseline. A fine-tuned 4-class model can be trained separately; see [Fine-tuning](#fine-tuning-on-the-4-cakgo-classes).
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

## Fine-tuning on the 4 CakGo classes
The pretrained model cannot detect wallets, keys or glasses, so we fine-tune it on the Objects In House
**train split** (6,912 images, unused by the shared evaluation) with 4 classes: `wallet`, `keys`, `glasses`,
`remote_control`.

```bash
# 1. Raw export (once; needs ROBOFLOW_API_KEY, see tests/dataset/README.md)
python tests/dataset/prepare_dataset.py download
# 2. Build the YOLO-format training set -> detection/experiments/yolo/data/cakgo4/ (gitignored)
python detection/experiments/yolo/prepare_finetune_data.py
# 3. Quick check that training runs on your machine, then the real run
python detection/experiments/yolo/train.py --epochs 1 --name smoke_test
python detection/experiments/yolo/train.py
# 4. Evaluate on the SAME 70 shared images as the pretrained YOLO and Grounding DINO
python detection/experiments/yolo/run_inference.py --weights detection/experiments/yolo/training_runs/<name>/weights/best.pt \
    --mapping target_mapping_finetuned.json --conf 0.001 --run-name ft_<name>
python detection/experiments/yolo/evaluate.py --run ft_<name>
```

How the training data is built (`prepare_finetune_data.py`):
- **train** = export train split; **val** = export valid split (early stopping and threshold tuning). The test split is not used.
- **No leakage:** the 70 shared evaluation images, other crops of the same original photo, and near-duplicates (dHash) are removed from train and val. The shared evaluation stays a true held-out test.
- Images with only non-target labels (bag, broom, shoes, earphone) are kept as **background images**, which teach the model what is *not* a target and should reduce false alarms.
- Images with an unmapped label (`remote-control-ac`, AC remotes) or with no annotations are dropped as ambiguous.
- Counts per split and class are written to `data/cakgo4/dataset_stats.json` and copied into `training_info.json`.

Settings are in [`finetune.yaml`](finetune.yaml): `yolo26s.pt` start, 60 epochs, early-stop patience 15, imgsz 640,
batch 16 (use `--batch 8` if the GPU runs out of memory), seed 0, deterministic, default Ultralytics augmentation.
Each run writes `training_runs/<name>/` with `training_info.json` (settings, versions, dataset stats, best-weights
sha256), `results.csv` and plots. These can be committed; `weights/` is gitignored. An interrupted run continues with
`train.py --resume training_runs/<name>/weights/last.pt`.

Caveats:
- The training labels have the same gaps as the evaluation labels; for example, sunglasses visible but unlabeled
  in keys photos (see `tests/dataset/dataset_info.md`). The model may learn to ignore some real targets.
- Train and test images come from the same dataset, so the 70-image score will be optimistic. Confirm on real
  phone photos before choosing the final model.

### Results so far
| Run | Change | Validation mAP50 / wallet recall | 70 shared images: top-1 correct (wallet / keys / glasses / remote) | False alarms on the 10 negatives |
|---|---|---|---|---|
| `yolo26s_cakgo4_v1` | `finetune.yaml`, 60 epochs | 0.917 / 0.63 | 8 / 14 / 14 / 14 of 15 | 0 |
| `yolo26s_cakgo4_v2` | `finetune_v2.yaml`: 120 epochs + rotation, mixup, more brightness/zoom variation | 0.932 / 0.69 | 7 / 14 / 14 / 14 of 15 | 0 |

- **Wallet overfitting (v1):** 100 % of training wallets found but only 63 % of validation wallets (keys, glasses
  and remote: 94–99 %). It is not about box size, confidence threshold or duplicate labels: about a quarter of
  validation wallets get no wallet box even at confidence 0.01. v2 tried more image variation instead of only
  more epochs; validation improved slightly, but the 70 shared images did not (a 2-image difference, within
  noise). **v1 stays the default.**
- **Real iPhone frames (phone test, 2026-10-19): fine-tuned YOLO does not generalise.** On 4 real photos of
  cluttered desks, v1 found 1 of 4 targets and v2 found 0 of 4; Grounding DINO found 4 of 4. Likely causes:
  - the training images are stretched to 640×640 (Objects In House v5) and are mostly stock-style close-ups
  - `remote-control-ac` (air-conditioner remotes) was dropped from training, so the user's AC remote was never
    a training example

  The 70-image scores above are therefore not representative of phone use.
- Ideas not tried yet: count `remote-control-ac` as `remote_control`, freeze the backbone (`freeze`) to keep
  COCO's real-world features, and add real-home images (e.g. the secondary Home item detection dataset).

## Structure
```text
detection/experiments/yolo/
├── README.md                     # This guide
├── requirements.txt              # Pinned ultralytics version
├── target_mapping.json           # Pretrained: CakGo category -> COCO class(es), unsupported classes documented
├── target_mapping_finetuned.json # Fine-tuned: CakGo category -> the model's own 4 classes
├── detector.py                   # YoloDetector: predict() + detect(image, target) per API contract
├── run_inference.py              # Runs a model on tests/dataset/manifest.csv, saves predictions + run info
├── evaluate.py                   # Scores predictions against tests/dataset/annotations/ground_truth.json
├── prepare_finetune_data.py      # Builds the YOLO training set from the export's train/valid splits
├── finetune.yaml                 # Training settings (v1)
├── finetune_v2.yaml              # v2 experiment: longer training + stronger augmentation
├── train.py                      # Fine-tunes YOLO and records training_info.json
├── results/                      # One folder per inference run
├── training_runs/                # One folder per training run (weights/ gitignored)
├── data/                         # Generated training set (gitignored)
└── weights/                      # Downloaded pretrained weights (gitignored)
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
2. Do not invent test metrics. Results must come from `run_inference.py` and `evaluate.py`. Never train on, or tune thresholds with, the 70 shared evaluation images.
3. Keep the detection interface compatible with [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
4. Do not commit weights, datasets or API keys.
