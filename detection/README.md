# Detection Module — Member 1 & Member 4

## Component Purpose
The `detection/` component provides the core visual object detection capability for CakGo. It houses comparative experiment submodules for evaluating candidate architectures and will host the unified detector adapter once a model is selected.

## Assigned Team Members
- **Initial Phase (Model Comparison):**
  - **Member 1:** YOLO testing (`experiments/yolo/`)
  - **Member 4:** Grounding DINO testing (`experiments/grounding_dino/`)
- **Post-Comparison Phase:**
  - **Member 1:** Final model selection, implementation, adapter optimization, and continuous improvement.

## Dataset & Model Status
- **Dataset (confirmed):** *Objects In House* v5 (CC0); 70 shared test images in [`tests/dataset/`](../tests/dataset/README.md). Microsoft ORBIT remains a future option.
- **Real-photo test:** 94 team phone photos, same photos for every model: [`COMPARISON.md`](../tests/results/real_photos/COMPARISON.md).
- **Chosen model: Grounding DINO multi-prompt** (`prompt_mode: "multi"`, threshold 0.35). Fine-tuned YOLO missed most objects on real phone photos.
- **Demo:** `scripts\start_demo.cmd` ([`scripts/README.md`](../scripts/README.md)).

## Planned Structure
```text
detection/
├── README.md               # This documentation
├── detector.py             # (Post-selection) Unified detector adapter implementing detect(image, target)
└── experiments/
    ├── yolo/               # Member 1: YOLO benchmarking & setup
    │   └── README.md
    └── grounding_dino/     # Member 4: Grounding DINO benchmarking & setup
        └── README.md
```

## Tasks & Development Priorities
1. **Experiment Setup:** Configure environment and dependencies for both YOLO and Grounding DINO inside `experiments/`.
2. **Dataset Confirmation:** Wait for team consensus on the evaluation dataset.
3. **Comparative Evaluation:**
   - Run inference across the identical test samples.
   - Benchmark detection rate for target indoor items (e.g., keys, wallet, glasses, phone).
   - Measure inference latency (FPS / ms per frame) and hardware requirements (CPU vs GPU / VRAM).
   - Analyze behavior on small, partially occluded, or cluttered scenes.
4. **Adapter Implementation:** Implement the chosen model behind the standard `detect(image, target)` callable interface for the backend.

## Expected Inputs & Outputs
In accordance with [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md):
- **Input:**
  - `image`: Decoded image array or frame buffer (RGB).
  - `target`: Target object name or text prompt (string).
- **Output:**
  - Python dictionary or structured data:
    ```python
    {
        "target": "keys",
        "detected": True,
        "detections": [
            {
                "label": "keys",
                "confidence": 0.88,
                "bbox": [120, 200, 240, 310]  # [x1, y1, x2, y2] in original image pixels
            }
        ]
    }
    ```

## Connection to CakGo & Component Dependencies
- **Upstream:** None directly; invoked by `backend/` via the detector interface.
- **Downstream:** `backend/` consumes detection bounding boxes to compute spatial position (`left`, `center`, `right`) and trigger `voice/` feedback.
- **Independence:** Detection development runs independently from frontend and voice.

## Instructions for AI Coding Agents
1. Read the root `README.md`, this file, and [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md).
2. For YOLO-specific tasks, operate strictly within `detection/experiments/yolo/`.
3. For Grounding DINO-specific tasks, operate strictly within `detection/experiments/grounding_dino/`.
4. The dataset and model are decided (see Dataset & Model Status). Changing the model, prompts or thresholds requires re-running the real-photo test in `tests/real_photos/`.
5. Ensure bounding box outputs always match the `[x1, y1, x2, y2]` pixel coordinate convention.
