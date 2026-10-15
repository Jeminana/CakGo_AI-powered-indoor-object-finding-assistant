# Evaluation Results & Benchmarks — tests/results/

## Purpose
This directory stores comparative evaluation logs, inference latency records, and test run summaries generated during model evaluation and QA testing.

## Contents (Once Testing Commences)
- **`yolo_benchmark_results.json` / `.md`**: Accuracy, latency, and resource metrics for YOLO evaluated by Member 1 on the confirmed dataset.
- **`grounding_dino_benchmark_results.json` / `.md`**: Accuracy, latency, and resource metrics for Grounding DINO evaluated by Member 4 on the confirmed dataset.
- **`model_comparison_report.md`**: Joint comparison document summarizing trade-offs, strengths, weaknesses, and final model selection rationale.
- **`e2e_test_logs/`**: Integration and QA test results across mobile browser, backend, and voice output.

## Guidelines
- Do not fabricate benchmark metrics or synthetic results.
- Ensure all comparisons between YOLO and Grounding DINO utilize the same confirmed dataset and consistent testing hardware specifications.
