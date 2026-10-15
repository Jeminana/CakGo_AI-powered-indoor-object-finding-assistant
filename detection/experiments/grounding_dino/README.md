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

## Planned Structure
```text
detection/experiments/grounding_dino/
├── README.md           # This guide
├── detector.py         # Standardized Grounding DINO inference wrapper
├── evaluate.py         # Benchmark and latency measurement script
└── config/             # Model architecture configs and prompt definitions
```

## Instructions for AI Coding Agents
1. Work exclusively inside `detection/experiments/grounding_dino/`.
2. Do not download external datasets or invent fabricated test measurements.
3. Keep the detection interface compatible with [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
4. Clearly state when dummy/provisional data is used.
