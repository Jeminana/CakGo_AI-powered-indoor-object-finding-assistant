# Zero-shot evaluation - t4_yoloe-26s_single

- Model: `/content/work/yoloe-26s-seg.pt` (YOLOE-26 (Ultralytics, text prompts, zero-shot, no fine-tuning))
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet` | 17 | 4 | 2 | 13 | 0.6667 | 0.2353 | 0.2377 | 4/15 | 0/10 |
| keys | `keys` | 32 | 7 | 0 | 25 | 1.0 | 0.2188 | 0.5299 | 5/15 | 0/10 |
| glasses | `glasses | sunglasses` | 18 | 6 | 1 | 12 | 0.8571 | 0.3333 | 0.5244 | 6/15 | 0/10 |
| remote_control | `remote control` | 18 | 13 | 1 | 5 | 0.9286 | 0.7222 | 0.8039 | 12/15 | 0/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 22.41 | 21.92 | 25.57 |
