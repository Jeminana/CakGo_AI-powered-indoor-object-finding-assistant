# Zero-shot evaluation - t4_yoloe-26x_single

- Model: `/content/work/yoloe-26x-seg.pt` (YOLOE-26 (Ultralytics, text prompts, zero-shot, no fine-tuning))
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet` | 17 | 4 | 3 | 13 | 0.5714 | 0.2353 | 0.2574 | 4/15 | 1/10 |
| keys | `keys` | 32 | 13 | 4 | 19 | 0.7647 | 0.4062 | 0.6174 | 9/15 | 0/10 |
| glasses | `glasses | sunglasses` | 18 | 8 | 3 | 10 | 0.7273 | 0.4444 | 0.5984 | 8/15 | 0/10 |
| remote_control | `remote control` | 18 | 13 | 2 | 5 | 0.8667 | 0.7222 | 0.795 | 12/15 | 0/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 136.56 | 136.66 | 140.21 |
