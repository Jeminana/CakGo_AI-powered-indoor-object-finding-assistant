# Zero-shot evaluation - t4_yoloe-26s_multi

- Model: `/content/work/yoloe-26s-seg.pt` (YOLOE-26 (Ultralytics, text prompts, zero-shot, no fine-tuning))
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 17 | 4 | 2 | 13 | 0.6667 | 0.2353 | 0.201 | 4/15 | 0/10 |
| keys | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 32 | 6 | 0 | 26 | 1.0 | 0.1875 | 0.4626 | 5/15 | 0/10 |
| glasses | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 18 | 6 | 1 | 12 | 0.8571 | 0.3333 | 0.5413 | 6/15 | 0/10 |
| remote_control | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 18 | 8 | 1 | 10 | 0.8889 | 0.4444 | 0.537 | 7/15 | 0/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 23.04 | 22.71 | 27.02 |
