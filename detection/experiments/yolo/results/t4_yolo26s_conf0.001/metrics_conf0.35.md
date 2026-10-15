# YOLO evaluation - t4_yolo26s_conf0.001

- Model: `yolo26s.pt` (pretrained, no fine-tuning)
- Images: 70; confidence >= 0.35; IoU >= 0.5

| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |
|---|---|---|---|---|---|---|---|---|---|
| wallet | unsupported | - | - | - | - | - | - | - | - |
| keys | unsupported | - | - | - | - | - | - | - | - |
| glasses | unsupported | - | - | - | - | - | - | - | - |
| remote_control | evaluated | 18 | 15 | 0 | 3 | 1.0 | 0.8333 | 0.8704 | 13/15 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms | 14.42 | 13.24 | 23.15 |
| preprocess_ms | 0.73 | 0.71 | 0.84 |
| inference_ms | 12.12 | 10.96 | 20.73 |
| postprocess_ms | 1.21 | 1.18 | 1.58 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence; re-run inference with --conf 0.001 for an un-truncated AP.
- Unsupported categories are not scored; a COCO-pretrained model has no class for them.
