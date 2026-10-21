# Zero-shot evaluation - t4_sam3_single

- Model: `facebook/sam3` (SAM 3 (Hugging Face transformers, text prompts, zero-shot, no fine-tuning))
- Images: 70; confidence >= 0.5; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet` | 17 | 16 | 6 | 1 | 0.7273 | 0.9412 | 0.8029 | 14/15 | 1/10 |
| keys | `keys` | 32 | 16 | 346 | 16 | 0.0442 | 0.5 | 0.0788 | 12/15 | 0/10 |
| glasses | `glasses | sunglasses` | 18 | 18 | 11 | 0 | 0.6207 | 1.0 | 0.7094 | 15/15 | 1/10 |
| remote_control | `remote control` | 18 | 18 | 0 | 0 | 1.0 | 1.0 | 1.0 | 15/15 | 0/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 2289.21 | 2239.99 | 2523.39 |
