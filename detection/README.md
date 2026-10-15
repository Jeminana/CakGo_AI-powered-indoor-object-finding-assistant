# Detection Module — Member 1 & Member 4

## Component Purpose
The `detection/` component provides the core visual object detection capability for CakGo. It houses comparative experiment submodules for evaluating candidate architectures and will host the unified detector adapter once a model is selected.

## Assigned Team Members
- **Initial Phase (Model Comparison):**
  - **Member 1:** YOLO testing (`experiments/yolo/`)
  - **Member 4:** Grounding DINO testing (`experiments/grounding_dino/`)
- **Post-Comparison Phase:**
  - **Member 1:** Final model selection, implementation, adapter optimization, and continuous improvement.

## Dataset Selection Status
> **Dataset selection pending team discussion and confirmation.**
> We are currently considering several options, including the Microsoft ORBIT Dataset and other datasets containing everyday household objects. The final dataset will be selected later after discussion and confirmation with the team members.
> 
> - **Do not download datasets or generate mock evaluation annotations yet.**
> - Member 1 and Member 4 will evaluate both YOLO and Grounding DINO on the **exact same dataset** once confirmed by the team.
> - Proposed test categories, sample counts, and evaluation metrics remain provisional until confirmation.

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
4. Do not download or assume a dataset until the team explicitly provides the confirmed dataset details.
5. Ensure bounding box outputs always match the `[x1, y1, x2, y2]` pixel coordinate convention.
