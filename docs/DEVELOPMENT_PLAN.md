# Seven-day Parallel Development Plan

## Team ownership
| Member | Work |
|---|---|
| 1 | `detection/`: compare YOLO vs Grounding DINO, choose model, expose detector |
| 2 | `frontend/`: phone camera, accessible UI, upload/results |
| 3 | `backend/` + `voice/`: API, model integration, Korean guidance |
| 4 | `tests/` + `docs/`: images, QA, documentation, demo assistance |

## Independent work
Model comparison, UI with mock JSON, speech from mock JSON, server mock endpoints, test-image preparation and documentation can all begin on Day 1.

## Dependent work
Actual detector/backend connection; phone/backend upload; live speech integration; final end-to-end QA. Use `API_CONTRACT.md` to avoid waiting.

| Day | Member 1 | Member 2 | Member 3 | Member 4 |
|---|---|---|---|---|
| 1 | Set up two models | Test phone camera | Mock server + TTS | Sample images + test plan |
| 2 | Compare same images | Mock camera search UI | Mock API + guidance | Log model/test findings |
| 3 | Select detector + adapter | Send phone frame | Integrate real response | End-to-end QA |
| 4–5 | Improve detection | Boxes, status and accessibility | Stabilize API and speech | Regression tests + bugs |
| 6 | Detection evaluation | Real-phone testing | Audio/latency testing | Measured test report |
| 7 | Model results | Demo interface | Final pipeline | Final documentation and demo |

## Integration checkpoints
- Day 1: Agree target labels, API, phone/browser + HTTPS approach.
- Day 2: Mock image submission and simulated guidance working.
- Day 3: Real phone frame → selected model → spoken result.
- Day 5: End-to-end MVP stable.
- Days 6–7: No major new features; evaluate and present.

## Collaboration
Suggested branches: `feature/detection`, `feature/frontend`, `feature/backend-voice`, `feature/qa-docs`. Review pull requests before merging, integrate early and document changed contracts.

## Working with AI agents
Ask the agent to first read root README, owned component README and shared contract. Prefer small changes and verifiable tests; humans review code and results. No fabricated evaluation data.
