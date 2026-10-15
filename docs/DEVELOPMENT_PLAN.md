# Seven-day Parallel Development Plan

## Team Ownership & Updated Responsibilities

| Member | Initial Responsibility | Responsibility After Model Testing |
|---|---|---|
| **Member 1** | YOLO testing (`detection/experiments/yolo/`) | Final model implementation and improvement (`detection/`) |
| **Member 2** | Smartphone camera and frontend (`frontend/`) | Frontend development and integration (`frontend/`) |
| **Member 3** | Backend and voice guidance (`backend/`, `voice/`) | Backend, API, and voice integration (`backend/`, `voice/`) |
| **Member 4** | Grounding DINO testing (`detection/experiments/grounding_dino/`) | Documentation, evaluation reports, and testing (`tests/`, `docs/`) |

## Dataset Status Notice
> **Dataset selection pending team discussion and confirmation.**
> We are currently considering several options, including the Microsoft ORBIT Dataset and other datasets containing everyday household objects. The final dataset will be selected later after discussion and confirmation with the team members.
>
> - **Member 1** and **Member 4** will compare the models using the **same dataset** once it has been confirmed.
> - Proposed object categories, test sample counts, and evaluation conditions remain provisional.
> - The team will provide confirmed dataset details later.

## Independent vs. Dependent Work
- **Independent Work:**
  - **Model Testing:** Member 1 (YOLO) and Member 4 (Grounding DINO) set up independent experiment environments in `detection/experiments/`.
  - **Frontend & Voice:** Member 2 (Frontend) and Member 3 (Backend & Voice) work independently using simulated mock detection responses adhering to [`API_CONTRACT.md`](API_CONTRACT.md).
- **Dependent Work:**
  - Connecting the chosen detector adapter to the backend server.
  - End-to-end phone camera capture → backend detection → Korean voice feedback.

## Daily Development Schedule

| Day | Member 1 (YOLO / Detection) | Member 2 (Frontend) | Member 3 (Backend & Voice) | Member 4 (Grounding DINO / QA) |
|---|---|---|---|---|
| **Day 1** | Set up YOLO experiment environment & dummy inference | Phone camera preview & permission test | FastAPI mock server & TTS audio logic | Set up Grounding DINO experiment & text configs |
| **Day 2** | Prepare YOLO benchmark scripts (pending dataset) | Capture frame & render mock detection result | Mock `/api/detect` with spatial calculations | Prepare Grounding DINO benchmark scripts (pending dataset) |
| **Day 3** | Run model benchmark on confirmed dataset with Member 4; select detector | Submit phone camera frame to `POST /api/detect` | Wire selected detector adapter into backend API | Run model benchmark on confirmed dataset with Member 1; log results |
| **Day 4–5** | Optimize chosen detector & implement `detect()` adapter | Bounding box rendering & accessibility controls | Stabilize API timeouts, error handling & speech | Run E2E regression tests & track bug fixes |
| **Day 6** | Measure latency, memory, and detection limits | Real-phone physical testing across browsers | Optimize latency & audio debounce | Compile measured QA report in `tests/results/` |
| **Day 7** | Final model adapter sign-off | Polished mobile UI for demo | End-to-end pipeline stabilization | Final project documentation & demo support |

## Integration Checkpoints
- **Day 1:** Agree on API contract, component boundaries, and HTTPS hosting approach.
- **Day 2:** Mock image submission and mock directional voice feedback working independently.
- **Day 3:** Dataset confirmed → Model benchmark completed → Selected model adapter integrated into backend.
- **Day 5:** End-to-end MVP stable (Phone camera → Backend API → Detection → Korean speech).
- **Days 6–7:** Final evaluation, benchmark reporting, and demo rehearsals.

## Collaboration Workflow
- **Git Branches:** `feature/detection`, `feature/frontend`, `feature/backend-voice`, `feature/qa-docs`.
- **Pull Requests:** Open PRs against `main`. No direct push to `main` for experimental code.
- **API Contract:** All cross-component communication must strictly conform to [`API_CONTRACT.md`](API_CONTRACT.md).
