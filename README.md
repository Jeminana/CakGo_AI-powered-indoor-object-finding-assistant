# CakGo (찾고) — AI-powered Indoor Object Finder

**Repository:** [https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant)

**Goal:** A one-week accessibility-focused prototype using a phone camera to help people with visual difficulties find visible everyday objects through Korean voice guidance. Anyone who has trouble finding small items (e.g., keys, wallet, glasses) may also benefit.

## MVP Scope
1. **Smartphone browser camera capture** and target selection (large accessible controls).
2. **Pretrained object detection model** (compare YOLO and Grounding DINO, then select one).
3. **Spoken Korean left/center/right guidance** and visual bounding boxes.

*Optional/Future:* Vibration feedback, last-seen location memory, multi-camera support, custom item identification. Note: A phone cannot find itself without an external device; fully obscured items cannot be detected directly.

## Team Responsibilities

| Member | Initial Responsibility | Responsibility After Model Testing |
|---|---|---|
| **Member 1** | YOLO testing (`detection/experiments/yolo/`) | Final model implementation and improvement (`detection/`) |
| **Member 2** | Smartphone camera and frontend (`frontend/`) | Frontend development and integration (`frontend/`) |
| **Member 3** | Backend and voice guidance (`backend/`, `voice/`) | Backend, API, and voice integration (`backend/`, `voice/`) |
| **Member 4** | Grounding DINO testing (`detection/experiments/grounding_dino/`) | Documentation, evaluation reports, and testing (`tests/`, `docs/`) |

- **Model Evaluation (done):** Member 1 (YOLO) and Member 4 (Grounding DINO) compared their models on the same shared dataset and on the team's real phone photos. See [Dataset & Model Status](#dataset--model-status).
- **Independent Development:** Members 2 and 3 can develop and test frontend and voice/backend features independently using mock detection results specified in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

## Dataset & Model Status
- **Dataset (confirmed):** *Objects In House* v5 (Roboflow Universe, CC0). The 70 shared test images (wallet, keys, glasses, remote control, and target-absent) are reproduced locally with `tests/dataset/prepare_dataset.py`. See [`tests/dataset/README.md`](tests/dataset/README.md) and [`dataset_info.md`](tests/dataset/dataset_info.md). Microsoft ORBIT remains a future option.
- **Real-photo test:** 94 phone photos taken by the team. YOLO and Grounding DINO were scored on exactly the same photos: [`tests/results/real_photos/COMPARISON.md`](tests/results/real_photos/COMPARISON.md).
- **Chosen model: Grounding DINO with the multi-prompt** (`detection/experiments/grounding_dino/`, `prompt_mode: "multi"`, threshold 0.35). Fine-tuned YOLO looked good on the stock-style test images but missed most objects in real phone photos. Details: [`MULTI_PROMPT.md`](detection/experiments/grounding_dino/MULTI_PROMPT.md) and the YOLO README.
- **Run the demo:** `scripts\start_demo.cmd` starts the backend with the chosen model, the frontend and the HTTPS phone link. Step-by-step instructions: [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md). It needs an NVIDIA GPU for real-time use.

**Guidelines for all developers and AI agents:**
- Never commit dataset images, model weights, API keys, or the team's private phone photos (`tests/real_photos/` is gitignored).
- Download data only with the documented scripts (`prepare_dataset.py`); keep the Roboflow API key in an environment variable.
- Changing the model, prompts or thresholds requires re-running the real-photo test (`tests/real_photos/`) and updating `COMPARISON.md`.

## Repository Structure
```text
CakGo/
├── README.md
├── requirements.txt
├── detection/
│   ├── README.md
│   └── experiments/
│       ├── yolo/
│       │   └── README.md
│       └── grounding_dino/
│           └── README.md
├── frontend/
│   └── README.md
├── backend/
│   └── README.md
├── voice/
│   └── README.md
├── tests/
│   ├── README.md
│   ├── dataset/
│   ├── real_photos/
│   └── results/
├── scripts/
│   └── start_demo.cmd
└── docs/
    ├── README.md
    ├── API_CONTRACT.md
    └── DEVELOPMENT_PLAN.md
```

## Workflow & Collaboration
- **Branches:** `feature/detection`, `feature/frontend`, `feature/backend-voice`, `feature/qa-docs`.
- **Review:** Use Pull Requests to merge into `main`. Integrate components via the shared [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).
- **Privacy:** Do not commit personal images or private room captures to this public repository.

## Instructions for AI Coding Agents
1. First read this `README.md`, your assigned component's `README.md`, and [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).
2. Keep edits strictly scoped within your assigned component directory.
3. Adhere to the shared API contract for inputs and outputs; do not change endpoint contracts unilaterally.
4. Follow the [Dataset & Model Status](#dataset--model-status): the dataset and the model (Grounding DINO multi-prompt) are decided; don't change them without re-running the real-photo test.
5. Provide reproducible test instructions and clean, verifiable code. Human review is required before merging.

**Status (2026-10-20):** End-to-end MVP working on a real phone (camera → backend → Grounding DINO multi-prompt → boxes → Korean voice with two-frame confirmation). Remaining: second real-phone test, final report and presentation.
