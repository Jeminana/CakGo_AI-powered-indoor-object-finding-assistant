# YOLO evaluation - ft_yolo26s_cakgo4_v2_conf0.25

- Model: `best.pt` (fine-tuned, training run `yolo26s_cakgo4_v2`)
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |
|---|---|---|---|---|---|---|---|---|---|
| wallet | evaluated | 17 | 9 | 1 | 8 | 0.9 | 0.5294 | 0.5 | 8/15 |
| keys | evaluated | 32 | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9658 | 14/15 |
| glasses | evaluated | 18 | 17 | 1 | 1 | 0.9444 | 0.9444 | 0.9414 | 14/15 |
| remote_control | evaluated | 18 | 17 | 1 | 1 | 0.9444 | 0.9444 | 0.9444 | 14/15 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms | 14.09 | 13.88 | 15.02 |
| preprocess_ms | 0.83 | 0.81 | 0.98 |
| inference_ms | 11.66 | 11.61 | 12.56 |
| postprocess_ms | 1.31 | 1.37 | 1.93 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence; re-run inference with --conf 0.001 for an un-truncated AP.
- Unsupported categories are not scored; the model has no class for them.
