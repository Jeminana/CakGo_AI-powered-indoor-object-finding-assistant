# Evaluation Results & Benchmarks — tests/results/

## Purpose
This directory stores comparative evaluation logs, inference latency records, and test run summaries generated during model evaluation and QA testing.

## Contents (Once Testing Commences)
- **Per-run model results** live next to each experiment, one folder per run: `detection/experiments/yolo/results/<run>/` and `detection/experiments/grounding_dino/results/<run>/` (`predictions.jsonl`, `run_info.json`, `metrics.json` / `.md`).
- **`model_comparison_report.md`**: Joint comparison document summarizing trade-offs, strengths, weaknesses, and final model selection rationale. Its tables are printed by `tests/compare_models.py`.
- **`e2e_test_logs/`**: Integration and QA test results across mobile browser, backend, and voice output.

## Guidelines
- Do not fabricate benchmark metrics or synthetic results.
- Ensure all comparisons between YOLO and Grounding DINO utilize the same confirmed dataset and consistent testing hardware specifications.
