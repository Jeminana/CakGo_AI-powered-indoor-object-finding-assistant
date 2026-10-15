# CakGo (찾고) — AI-powered Indoor Object Finder

**Goal:** A one-week accessibility-focused prototype using a phone camera to help people with visual difficulties find visible everyday objects through Korean voice guidance. Anyone who has trouble finding small items may also benefit.

## MVP scope
1. Smartphone browser camera capture and target selection (large accessible controls).
2. Pretrained object detection model (compare YOLO and Grounding DINO, then select one).
3. Spoken Korean left/center/right guidance and visual bounding boxes.

Optional if time permits: vibration. Future: last-seen location memory, extra cameras, individual item identification. A phone cannot find itself without another device. Fully obscured items cannot be directly seen.

## Team
- **Member 1:** `detection/` — model evaluation and detector function.
- **Member 2:** `frontend/` — mobile browser camera, UI and bounding boxes.
- **Member 3:** `backend/`, `voice/` — HTTP API, integration and speech.
- **Member 4:** `tests/`, `docs/` — test data, documentation, QA, demo support.

Read [the shared API contract](docs/API_CONTRACT.md) before coding. Follow [development plan](docs/DEVELOPMENT_PLAN.md).

## Workflow
Create separate branches `feature/detection`, `feature/frontend`, `feature/backend-voice`, `feature/qa-docs`. Use PRs; integrate on Day 3. Mock results let UI, speech and backend develop in parallel. Do not upload private frames to this public repository.

## AI coding agent instructions
First read this README, your component README, and `docs/API_CONTRACT.md`. Keep changes in your assigned component; preserve the shared API and provide reproducible test/run instructions. Ask before changing contracts or dependencies. Human review required.

**Status:** Documentation-only scaffold. No working application code is included yet.
