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

- **Model Evaluation:** Member 1 and Member 4 will compare their respective models using the **same dataset** once it has been confirmed.
- **Independent Development:** Members 2 and 3 can develop and test frontend and voice/backend features independently using mock detection results specified in [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md).

## Dataset Status
> **Dataset selection pending team discussion and confirmation.**
> We are currently considering several options, including the Microsoft ORBIT Dataset and other datasets containing everyday household objects. The final dataset will be selected later after discussion and confirmation with team members.
>
> **Guidelines for all developers and AI agents:**
> - Do not select or finalize any dataset prematurely.
> - Do not download datasets or upload large binary dataset files to git.
> - Do not automatically assume ORBIT or any other specific dataset is finalized.
> - Do not create provisional evaluation images or annotations until the team confirms the dataset.
> - Keep the dataset folder structure (`tests/dataset/`) prepared for future use.

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
│   └── results/
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
4. **Do not assume a finalized dataset**; respect the pending dataset status banner.
5. Provide reproducible test instructions and clean, verifiable code. Human review is required before merging.

**Status:** Repository scaffold and development guidelines initialized. Application development and model benchmarking commence upon dataset confirmation and mock integration.
