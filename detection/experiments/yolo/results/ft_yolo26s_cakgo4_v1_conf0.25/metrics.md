# YOLO evaluation - ft_yolo26s_cakgo4_v1_conf0.25

- Model: `best.pt` (fine-tuned, training run `yolo26s_cakgo4_v1`)
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |
|---|---|---|---|---|---|---|---|---|---|
| wallet | evaluated | 17 | 11 | 5 | 6 | 0.6875 | 0.6471 | 0.5916 | 10/15 |
| keys | evaluated | 32 | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9678 | 14/15 |
| glasses | evaluated | 18 | 17 | 0 | 1 | 1.0 | 0.9444 | 0.9444 | 14/15 |
| remote_control | evaluated | 18 | 17 | 0 | 1 | 1.0 | 0.9444 | 0.9444 | 14/15 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms | 15.22 | 15.29 | 16.12 |
| preprocess_ms | 0.93 | 0.92 | 1.1 |
| inference_ms | 12.53 | 12.48 | 13.16 |
| postprocess_ms | 1.4 | 1.48 | 1.93 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence; re-run inference with --conf 0.001 for an un-truncated AP.
- Unsupported categories are not scored; the model has no class for them.
