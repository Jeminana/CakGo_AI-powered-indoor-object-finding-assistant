# YOLO reference runs (made by Member 4)

**These are not Member 1's official YOLO results.** On 2026-10-16 the repository had no YOLO results
yet: PR #1 shipped the YOLO scripts, but the evaluation subset had not been built. To compare the
models on Day 3, Member 4 ran Member 1's scripts **unmodified** on the same Colab Tesla T4 VM and the
same `tests/dataset/manifest.csv` (sha256 `ca73e2c1…5bde4b`) as Grounding DINO.

They are kept here so that `detection/experiments/yolo/` stays untouched. Member 1 can confirm these
numbers or replace them with an own run on the same manifest; `tests/compare_models.py
--yolo-results-dir <folder>` compares any set of YOLO run folders.

| Run | How it was made |
|---|---|
| `t4_yolo26s_conf0.25` | `run_inference.py --run-name t4_yolo26s_conf0.25` (default settings: `yolo26s.pt`, imgsz 640, conf 0.25), then `evaluate.py` |
| `t4_yolo26s_conf0.001` | `run_inference.py --conf 0.001` for an untruncated AP50; `evaluate.py` at 0.25 (`metrics.json`) and at 0.1 / 0.35 / 0.5 (`metrics_conf<t>.json`) |
| `colabcpu_yolo26s_latency-sample` | `run_inference.py --device cpu` on 10 images (two per bucket, `manifest_cpu_sample.csv`), for CPU latency only |

The scripts write to `detection/experiments/yolo/results/<run>/` by default; the folders were moved
here unchanged after the run. Environment: ultralytics 8.4.177, torch 2.11.0+cu130, Python 3.13.15
(see each `run_info.json`).
