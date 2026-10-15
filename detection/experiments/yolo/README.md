# YOLO Experiment & Evaluation — Member 1

## Purpose
This directory houses the experimentation, benchmarking, and wrapper implementation for YOLO (e.g., Ultralytics YOLOv8/v11) to assess its viability as the primary object detection engine for CakGo.

## Assigned Team Member
- **Member 1**
  - **Initial Responsibility:** YOLO testing, setup, and comparative benchmark execution.
  - **Post-Testing Responsibility:** Final model implementation, optimization, and integration into the core `detection/` pipeline.

## Dataset Selection Status
> **Dataset selection pending team discussion and confirmation.**
> We are currently considering several options, including the Microsoft ORBIT Dataset and other datasets containing everyday household objects. The final dataset will be selected later after discussion and confirmation with team members.
>
> - **Do not download any dataset or create actual evaluation annotations at this time.**
> - Once confirmed, **Member 1 (YOLO)** and **Member 4 (Grounding DINO)** will benchmark both models on the **identical dataset**.
> - Proposed object categories (e.g., keys, wallet, phone, glasses), number of test images, and test conditions remain provisional until confirmed.

## Tasks & Development Priorities
1. **Environment Setup:** Configure YOLO dependencies (e.g. `ultralytics`) and test baseline inference on dummy frames.
2. **Standard Adapter Preparation:** Implement `detector.py` implementing `detect(image, target)` compatible with [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
3. **Dataset Benchmark (Awaiting Confirmation):**
   - Once dataset is confirmed, load identical test images.
   - Run inference across lighting variations, clutter, and partial occlusions.
   - Record detection confidence, false positive/negative rates, and latency in milliseconds.
4. **Export Findings:** Log benchmark metrics to `tests/results/` for joint review with Member 4.

## Expected Inputs & Outputs
- **Input:** Decoded frame (RGB image) and target label string (e.g., `"cell phone"`, `"remote"`).
- **Output:** Structured detection result conforming to API contract:
  ```python
  {
      "target": "cell phone",
      "detected": True,
      "detections": [
          {"label": "cell phone", "confidence": 0.92, "bbox": [150, 80, 320, 410]}
      ]
  }
  ```
  Where `bbox` represents `[x1, y1, x2, y2]` in original image pixel coordinates.

## Planned Structure
```text
detection/experiments/yolo/
├── README.md           # This guide
├── detector.py         # Standardized YOLO inference wrapper
└── evaluate.py         # Benchmark and latency measurement script
```

## Instructions for AI Coding Agents
1. Work exclusively inside `detection/experiments/yolo/`.
2. Do not download external datasets or invent fabricated test metrics.
3. Keep the detection interface compatible with [`docs/API_CONTRACT.md`](../../../docs/API_CONTRACT.md).
4. Clearly state when dummy/provisional data is used.
