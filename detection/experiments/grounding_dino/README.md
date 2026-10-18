# Grounding DINO Experiment & Evaluation — Member 4

## Purpose
This directory houses the experimentation, benchmarking, and wrapper implementation for Grounding DINO (open-vocabulary zero-shot detector) to evaluate its capability for detecting arbitrary text-prompted indoor household items.

## Assigned Team Member
- **Member 4**
  - **Initial Responsibility:** Grounding DINO testing, setup, prompt tuning, and comparative evaluation.
  - **Post-Testing Responsibility:** Documentation, evaluation reports, and testing (`tests/`, `docs/`).

## Dataset Selection Status
> **Dataset selection pending team discussion and confirmation.**
> We are currently considering several options, including the Microsoft ORBIT Dataset and other datasets containing everyday household objects. The final dataset will be selected later after discussion and confirmation with team members.
>
> - **Do not download any dataset or create actual evaluation annotations at this time.**
> - Once confirmed, **Member 4 (Grounding DINO)** and **Member 1 (YOLO)** will evaluate both models on the **identical dataset**.
> - Proposed object categories, number of test images, and test conditions remain provisional until confirmed.

## Tasks & Development Priorities
1. **Environment & Model Setup:** Prepare Grounding DINO environment (PyTorch, transformers, groundingdino package) and config files.
2. **Standard Adapter Preparation:** Implement `detector.py` implementing `detect(image, target)` conforming to [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
3. **Prompt Engineering & Evaluation (Awaiting Confirmation):**
   - Once dataset is confirmed, test text prompts for small everyday objects (e.g., `"keys"`, `"glasses"`, `"wallet"`).
   - Evaluate detection sensitivity, box accuracy, and confidence thresholds.
   - Measure inference latency (ms per inference) and GPU VRAM consumption.
4. **Comparative Analysis:** Compare performance and resource trade-offs against YOLO alongside Member 1, storing structured results in `tests/results/`.

## Expected Inputs & Outputs
- **Input:** Decoded frame (RGB image) and arbitrary target text prompt (e.g., `"metal house keys"`, `"black spectacles"`).
- **Output:** Structured detection result conforming to API contract:
  ```python
  {
      "target": "keys",
      "detected": True,
      "detections": [
          {"label": "keys", "confidence": 0.86, "bbox": [180, 210, 260, 290]}
      ]
  }
  ```
  Where `bbox` represents `[x1, y1, x2, y2]` in original image pixel coordinates.

## Structure
```text
detection/experiments/grounding_dino/
├── README.md           # This guide
├── requirements.txt    # torch, transformers>=4.49, pillow, numpy, pytest
├── detector.py         # Standardized Grounding DINO inference wrapper: detect(image, target)
├── run_inference.py    # Runs the model on tests/dataset/manifest.csv, saves predictions + run info
├── evaluate.py         # Scores predictions against tests/dataset/annotations/ground_truth.json (same rules as YOLO)
├── MULTI_PROMPT.md     # Multi-prompt mode: method, tuning and results (fewer false alarms)
├── config/
│   ├── model.json      # Checkpoint, device, box/text thresholds, prompt_mode
│   ├── prompts.json    # CakGo category -> prompt phrases
│   └── look_alikes.json # Look-alike phrases used in prompt_mode "multi"
├── results/            # One folder per run
└── tests/              # Unit tests (fake model, pinned YOLO-evaluator reference numbers) + opt-in real-model smoke test
```

## Setup & Usage
- **Implementation:** Hugging Face `transformers` port of Grounding DINO with the official `IDEA-Research/grounding-dino-tiny` checkpoint (Apache-2.0, ~690 MB, downloaded to the Hugging Face cache on first use; weights are never committed). This avoids compiling the original `groundingdino` package and runs on CPU or CUDA (e.g. Colab).
- **Install:** `pip install -r requirements.txt` (Colab already ships `torch` and `transformers`).
- **Detect:**
  ```python
  from PIL import Image
  from detector import detect

  detect(Image.open("frame.jpg"), "keys")
  # {"target": "keys", "detected": True, "detections": [{"label": "keys", "confidence": 0.62, "bbox": [x1, y1, x2, y2]}]}
  ```
  Or from the shell: `python detector.py frame.jpg keys`. CakGo categories in `config/prompts.json` expand to their prompt phrases (`glasses` -> `glasses. sunglasses.`, `remote_control` -> `remote control.`); any other text is used as the prompt as-is. The `label` of every detection is the requested target.
- **Prompt mode (`config/model.json`):** `"multi"` (default) puts all four targets plus look-alike objects (`config/look_alikes.json`: computer mouse, laptop, keyboard, …) into one prompt. A box counts for the requested target only if that target is the phrase it matches best. This cut false alarms on images without the target from 53–94 % to 5–16 % on validation images, at the cost of lower wallet recall. Method and numbers: [MULTI_PROMPT.md](MULTI_PROMPT.md). `"single"` is the original one-target prompt; free-text targets always use it.
- **Provisional settings:** `box_threshold` 0.35 / `text_threshold` 0.25 (the original Grounding DINO demo values) stay provisional until the team fixes the confidence threshold ("Day 1 decisions to confirm" in the API contract). Prompts were chosen from the category names in `tests/dataset/class_mapping.json` before looking at results and are recorded in each run's `run_info.json`.
- **Benchmark on the shared dataset (from the repository root):**
  ```bash
  python tests/dataset/prepare_dataset.py verify                          # same files as YOLO
  python detection/experiments/grounding_dino/run_inference.py             # default: all 4 categories, saves conf >= 0.05
  python detection/experiments/grounding_dino/evaluate.py --run <run_name> # scores at box_threshold (0.35) unless --conf-threshold
  python detection/experiments/grounding_dino/run_inference.py --prompt-mode single  # benchmark the original prompt mode
  ```
  Each image is queried once per CakGo category (the app asks for one target at a time), so every category is scored on every image, like YOLO's classes. `evaluate.py` uses the same matching, metrics and output layout as `detection/experiments/yolo/evaluate.py` at PR #1 (per category TP/FP/FN, precision, recall, AP50, target-image detection rate, negative images with a false positive). The scoring functions are copied, not imported, so this folder never reads the YOLO folder; `tests/test_gdino_evaluate.py` pins the numbers the YOLO evaluator produced for a test fixture. Latency is `wall_ms_per_query`: one `detect()` call (pre-processing, inference, post-processing) after warm-up, image reading excluded. Use `--device cpu` to measure CPU latency.
- **Outputs (`results/<run_name>/`):** `predictions.jsonl` (per image: detections of every category query and ms per query), `run_info.json` (model revision, prompts, thresholds, versions, GPU, peak VRAM, manifest sha256), `metrics.json` / `metrics.md`.
- **Tests:** `pytest tests` runs the unit tests with a fake model (no download). The real-model smoke test on a synthetic drawing is opt-in: `GDINO_SMOKE=1 pytest tests/test_gdino_smoke.py`.

## Instructions for AI Coding Agents
1. Work exclusively inside `detection/experiments/grounding_dino/`.
2. Do not download external datasets or invent fabricated test measurements.
3. Keep the detection interface compatible with [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
4. Clearly state when dummy/provisional data is used.
