# Zero-shot evaluation - t4_yoloe-26x_multi

- Model: `/content/work/yoloe-26x-seg.pt` (YOLOE-26 (Ultralytics, text prompts, zero-shot, no fine-tuning))
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 17 | 4 | 3 | 13 | 0.5714 | 0.2353 | 0.26 | 4/15 | 1/10 |
| keys | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 32 | 12 | 4 | 20 | 0.75 | 0.375 | 0.5223 | 8/15 | 0/10 |
| glasses | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 18 | 8 | 3 | 10 | 0.7273 | 0.4444 | 0.4809 | 8/15 | 0/10 |
| remote_control | `wallet | keys | glasses | sunglasses | remote control | computer mouse | laptop | keyboard | pen | bottle | can | cup | microphone | cable | charger | nail clipper | coin` | 18 | 13 | 2 | 5 | 0.8667 | 0.7222 | 0.806 | 12/15 | 0/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 137.9 | 137.94 | 140.94 |
