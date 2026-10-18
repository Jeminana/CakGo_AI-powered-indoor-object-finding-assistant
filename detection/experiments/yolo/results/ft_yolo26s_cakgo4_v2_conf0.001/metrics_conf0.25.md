# YOLO evaluation - ft_yolo26s_cakgo4_v2_conf0.001

- Model: `best.pt` (fine-tuned, training run `yolo26s_cakgo4_v2`)
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |
|---|---|---|---|---|---|---|---|---|---|
| wallet | evaluated | 17 | 9 | 1 | 8 | 0.9 | 0.5294 | 0.6366 | 8/15 |
| keys | evaluated | 32 | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9658 | 14/15 |
| glasses | evaluated | 18 | 17 | 1 | 1 | 0.9444 | 0.9444 | 0.9942 | 14/15 |
| remote_control | evaluated | 18 | 17 | 1 | 1 | 0.9444 | 0.9444 | 0.9861 | 14/15 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms | 14.31 | 14.27 | 15.59 |
| preprocess_ms | 0.88 | 0.85 | 1.01 |
| inference_ms | 11.79 | 11.71 | 12.42 |
| postprocess_ms | 1.32 | 1.41 | 1.8 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence; re-run inference with --conf 0.001 for an un-truncated AP.
- Unsupported categories are not scored; the model has no class for them.
