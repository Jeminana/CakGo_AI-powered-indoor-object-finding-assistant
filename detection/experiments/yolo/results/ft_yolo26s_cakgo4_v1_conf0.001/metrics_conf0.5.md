# YOLO evaluation - ft_yolo26s_cakgo4_v1_conf0.001

- Model: `best.pt` (fine-tuned, training run `yolo26s_cakgo4_v1`)
- Images: 70; confidence >= 0.5; IoU >= 0.5

| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |
|---|---|---|---|---|---|---|---|---|---|
| wallet | evaluated | 17 | 10 | 3 | 7 | 0.7692 | 0.5882 | 0.8057 | 9/15 |
| keys | evaluated | 32 | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9678 | 14/15 |
| glasses | evaluated | 18 | 17 | 0 | 1 | 1.0 | 0.9444 | 1.0 | 14/15 |
| remote_control | evaluated | 18 | 17 | 0 | 1 | 1.0 | 0.9444 | 1.0 | 14/15 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms | 15.62 | 15.54 | 16.72 |
| preprocess_ms | 0.98 | 0.95 | 1.23 |
| inference_ms | 12.72 | 12.65 | 13.39 |
| postprocess_ms | 1.53 | 1.53 | 1.95 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence; re-run inference with --conf 0.001 for an un-truncated AP.
- Unsupported categories are not scored; the model has no class for them.
