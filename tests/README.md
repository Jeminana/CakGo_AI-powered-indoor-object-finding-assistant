# Testing & Evaluation — Member 4

## Component Purpose
The `tests/` directory manages quality assurance, test data organization, comparative model evaluation benchmarks, and integration testing for CakGo.

## Assigned Team Member
- **Member 4**
  - **Initial Responsibility:** Grounding DINO testing (`detection/experiments/grounding_dino/`).
  - **Post-Testing Responsibility:** Documentation, evaluation reports, integration QA, and final testing (`tests/`, `docs/`).

## Dataset Selection Status
> **Dataset selection pending team discussion and confirmation.**
> We have not finalized which dataset to use for comparing YOLO and Grounding DINO. We are currently considering several options, including the Microsoft ORBIT Dataset and other datasets containing everyday household objects. However, the final dataset will be selected later, after discussion and confirmation with the team members.
>
> **Mandatory Guidelines for Testing:**
> - **Do not select or finalize any dataset prematurely.**
> - **Do not download datasets.**
> - **Do not automatically use ORBIT or any other dataset.**
> - **Do not create actual evaluation images or annotations.**
> - Keep the dataset folder structure (`tests/dataset/`) prepared for future use.
> - The two members responsible for model evaluation (**Member 1** for YOLO and **Member 4** for Grounding DINO) will use the **same dataset** once it has been confirmed.
> - The proposed number of images, object categories, and evaluation conditions should remain provisional.
> - The team will provide the confirmed dataset details later.

## Directory Structure
```text
tests/
├── README.md           # This document
├── dataset/            # Prepared directory for the evaluation dataset once confirmed
│   └── README.md       # Dataset guidelines & pending notice
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
1. Do not download external datasets or invent fake benchmark data.
2. Maintain the separation between `tests/dataset/` (pending confirmation) and `tests/results/`.
3. Provide reproducible, automated pytest or benchmark scripts.
