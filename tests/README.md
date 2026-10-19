# Testing & Evaluation — Member 4

## Component Purpose
The `tests/` directory manages quality assurance, test data organization, comparative model evaluation benchmarks, and integration testing for CakGo.

## Assigned Team Member
- **Member 4**
  - **Initial Responsibility:** Grounding DINO testing (`detection/experiments/grounding_dino/`).
  - **Post-Testing Responsibility:** Documentation, evaluation reports, integration QA, and final testing (`tests/`, `docs/`).

## Dataset & Model Status
- **Dataset (confirmed):** *Objects In House* v5 (CC0); 70 shared test images in [`tests/dataset/`](../tests/dataset/README.md). Microsoft ORBIT remains a future option.
- **Real-photo test:** 94 team phone photos, same photos for every model: [`COMPARISON.md`](../tests/results/real_photos/COMPARISON.md).
- **Chosen model: Grounding DINO multi-prompt** (`prompt_mode: "multi"`, threshold 0.35). Fine-tuned YOLO missed most objects on real phone photos.
- **Demo:** `scripts\start_demo.cmd` ([`scripts/README.md`](../scripts/README.md)).

## Directory Structure
```text
tests/
├── README.md           # This document
├── dataset/            # Shared evaluation dataset (Objects In House v5, 70 images; images not committed)
│   └── README.md       # How to download, build, fetch and verify the dataset
├── real_photos/        # Real phone photo test: scoring script (team photos are gitignored)
└── results/            # Benchmark logs, latency logs, and comparative evaluation metrics
    └── README.md       # Results documentation
```

## Comparative Model Evaluation Protocol (Once Dataset Confirmed)
When the team confirms the dataset:
1. **Identical Sample Set:** Both Member 1 (YOLO) and Member 4 (Grounding DINO) will run evaluation on the identical set of test images.
2. **Standard Evaluation Metrics:**
   - **Detection Rate / Recall:** Accuracy on target indoor objects (e.g. keys, wallet, phone, glasses).
   - **Inference Latency:** Average milliseconds per frame across CPU and GPU.
   - **Robustness:** Performance under clutter, dim lighting, and partial occlusions.
   - **False Positive Rate:** Identifying non-target objects incorrectly.
3. **Log Storage:** All output logs, JSON evaluations, and latency records will be stored in `tests/results/`.

## End-to-End Integration QA
- **Mock Integration Tests:** Verify frontend communication with backend mock endpoints before full model wiring.
- **E2E Pipeline Testing:** Verify smartphone camera frame capture → backend processing → model detection → Korean audio playback.
- **Accessibility & Edge Cases:** Camera denied, server timeout, missing target object, low network bandwidth.

## Instructions for AI Coding Agents
1. Get data only through the documented scripts (`tests/dataset/prepare_dataset.py`, the team Drive for real photos); never invent benchmark data or commit images.
2. Maintain the separation between `tests/dataset/` (inputs) and `tests/results/` (outputs).
3. Provide reproducible, automated pytest or benchmark scripts.
