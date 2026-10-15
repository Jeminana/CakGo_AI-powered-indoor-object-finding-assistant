# YOLO evaluation - t4_yolo26s_conf0.25

- Model: `yolo26s.pt` (pretrained, no fine-tuning)
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |
|---|---|---|---|---|---|---|---|---|---|
| wallet | unsupported | - | - | - | - | - | - | - | - |
| keys | unsupported | - | - | - | - | - | - | - | - |
| glasses | unsupported | - | - | - | - | - | - | - | - |
| remote_control | evaluated | 18 | 15 | 0 | 3 | 1.0 | 0.8333 | 0.8333 | 13/15 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms | 15.25 | 13.17 | 25.91 |
| preprocess_ms | 0.7 | 0.68 | 0.82 |
| inference_ms | 12.77 | 11.12 | 23.69 |
| postprocess_ms | 1.43 | 1.05 | 1.3 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence; re-run inference with --conf 0.001 for an un-truncated AP.
- Unsupported categories are not scored; a COCO-pretrained model has no class for them.
