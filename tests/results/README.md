# Evaluation Results & Benchmarks — tests/results/

## Purpose
This directory stores comparative evaluation logs, inference latency records, and test run summaries generated during model evaluation and QA testing.

## Contents (Once Testing Commences)
- **Per-run model results** live next to each experiment, one folder per run: `detection/experiments/yolo/results/<run>/` and `detection/experiments/grounding_dino/results/<run>/` (`predictions.jsonl`, `run_info.json`, `metrics.json` / `.md`).
- **`yolo_reference_runs/`**: YOLO runs made by Member 4 with Member 1's unmodified scripts for the Day 3 comparison, kept here so the YOLO folder stays untouched. Not Member 1's official results; see its README.
- **`model_comparison_report.md`**: Joint comparison document summarizing trade-offs, strengths, weaknesses, and final model selection rationale. Its tables are printed by `tests/compare_models.py`.
- **`zero_shot_70/`**: Member 4's zero-shot screening of YOLOE-26 and SAM 3 against Grounding DINO on the 70 Day 3 images (code: `tests/zero_shot/`).
- **`few_shot/`**: Member 4's "register my item" experiments on real photos (YOLOE visual prompts, SAM 3 exemplars, Grounding DINO + DINOv3 verification). Aggregate results only; the photos and per-photo data stay local.
- **`e2e_test_logs/`**: Integration and QA test results across mobile browser, backend, and voice output.

## Guidelines
- Do not fabricate benchmark metrics or synthetic results.
- Ensure all comparisons between YOLO and Grounding DINO utilize the same confirmed dataset and consistent testing hardware specifications.
