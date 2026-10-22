# Documentation & Project Records — Member 4

## Purpose
The `docs/` directory maintains project-wide specifications, interface contracts, development schedules, meeting reports, and research documentation for CakGo.

## Assigned Team Member
- **Member 4**
  - **Initial Responsibility:** Grounding DINO testing (`detection/experiments/grounding_dino/`).
  - **Post-Testing Responsibility:** Documentation, evaluation reports, and testing (`tests/`, `docs/`).

## Dataset & Model Status
- **Dataset (confirmed):** *Objects In House* v5 (CC0); 70 shared test images in [`tests/dataset/`](../tests/dataset/README.md). Microsoft ORBIT remains a future option.
- **Real-photo test:** 94 team phone photos, same photos for every model: [`COMPARISON.md`](../tests/results/real_photos/COMPARISON.md).
- **Chosen model: Grounding DINO multi-prompt** (`prompt_mode: "multi"`, threshold 0.35). Fine-tuned YOLO missed most objects on real phone photos.
- **Demo:** `scripts\start_demo.cmd` ([`scripts/README.md`](../scripts/README.md)).

## Main Documents
- [`DEMO_GUIDE.md`](DEMO_GUIDE.md): **How to run the live demo**: setup, phone, demo sequence, sharing the link, troubleshooting.
- [`2026-10-20/`](2026-10-20/README.md): **Final results**: how the model was chosen, presentation plan, progress.
- [`2026-10-17/`](2026-10-17/README.md): Earlier documentation package and slide deck (before the phone tests).
- [`API_CONTRACT.md`](API_CONTRACT.md): Shared REST API specification and coordinate conventions. Must remain consistent across all components.
- [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md): 7-day development plan, member ownership, integration checkpoints, and collaboration workflow.
- [`REPORT.md`](REPORT.md): Official IC-PBL Team Activity Report (preserved).
- [`TEAM_ACTIVITY_REPORT.md`](TEAM_ACTIVITY_REPORT.md): Initial IC-PBL Team Activity Report record (preserved).
- [`TEAM_ACTIVITY_REPORT_2026-10-20.md`](TEAM_ACTIVITY_REPORT_2026-10-20.md): IC-PBL Team Activity Report for the last meeting (20 October 2026), before the final presentation on 22 October.

## Guidelines for Developers & AI Agents
1. Treat [`API_CONTRACT.md`](API_CONTRACT.md) as the single source of truth for component communication.
2. Meeting reports (`REPORT.md`, `TEAM_ACTIVITY_REPORT.md`) must be preserved in their existing format and location.
3. Keep documentation factual and avoid fabricated performance metrics.
