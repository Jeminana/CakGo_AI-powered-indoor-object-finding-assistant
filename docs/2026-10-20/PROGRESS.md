# Project progress on 20 October 2026

Follows [`../2026-10-17/PROGRESS.md`](../2026-10-17/PROGRESS.md). All PRs listed are merged into `main`.

## Since 17 October

| Date | Work | Owner | PR |
|---|---|---|---|
| 17 Oct | Korean voice connected in the frontend (spoken on the phone) | 상싱위 | #8 |
| 17 Oct | Slides, progress and result analysis (first version) | 상싱위 | #9 |
| 18 Oct | Grounding DINO multi-prompt: about 10× fewer false alarms | 김철연 | #10 |
| 18 Oct | YOLO v2 training experiment and phone-test findings recorded | 나즈미 | #11 |
| 19 Oct | **Phone test 1** on iPhone 15 Pro: full chain works; fine-tuned YOLO fails on real frames | 나즈미 | (#10, #11) |
| 19 Oct | Voice: speak only after two matching frames, "not visible" at most every 10 s | 상싱위 | #12 |
| 19 Oct | Real-photo scoring script; 94 team photos collected (private Drive) | 나즈미, team | #13, #14 |
| 19 Oct | Real-photo test: YOLO run, Grounding DINO run (Colab) | 나즈미, 김철연 | #14, #16 |
| 19 Oct | Larger text in the app | 오가암 | #15 |
| 20 Oct | **Final model: Grounding DINO multi-prompt** (team decision) | team | — |
| 20 Oct | One-command demo start + model comparison write-up | 나즈미 | #17 |
| 20 Oct | Outdated "dataset pending" notes replaced in all READMEs | 나즈미 | #18 |

## Current state

- **End-to-end MVP works on a real iPhone:** camera → HTTPS link → backend → Grounding DINO multi → boxes →
  Korean voice with two-frame confirmation.
- **Demo:** `scripts\start_demo.cmd` on the RTX 4050 laptop starts everything and prints the phone link.
- **Model choice is documented:** [`RESULT_ANALYSIS.md`](RESULT_ANALYSIS.md) and
  [`tests/results/real_photos/COMPARISON.md`](../../tests/results/real_photos/COMPARISON.md).

## Remaining

| Task | Owner | Note |
|---|---|---|
| Phone test 2 with the final setup: font, voice, all 4 objects, one empty area; Android if available | team | Use `scripts\start_demo.cmd` |
| Measure phone → spoken direction time | 나즈미 | During phone test 2; fill in `[after phone test 2]` in the slides |
| Fix bugs found in phone test 2 | owner of each part | Keep a short list: what, phone, object |
| Check Grounding DINO's box pictures from the real-photo run | 김철연 | So "found" is verified, as for YOLO (26/28) |
| Update the slide deck from [`PRESENTATION_NOTES.md`](PRESENTATION_NOTES.md) | 상싱위 | Keep slides 1–6 and 11, replace the rest |
| Rehearse the demo; record a backup video; test the backup laptop with `start_demo.ps1 -Check` | team | Start the demo about 10 min early |
| Optional: vibration or directional sound cues | 나즈미 | Personal extra after the presentation work |
