# Zero-shot evaluation - t4_gdino-tiny_multi

- Model: `IDEA-Research/grounding-dino-tiny` (Grounding DINO (Hugging Face transformers, zero-shot, no fine-tuning))
- Images: 70; confidence >= 0.35; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 17 | 12 | 7 | 5 | 0.6316 | 0.7059 | 0.5838 | 12/15 | 1/10 |
| keys | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 32 | 21 | 4 | 11 | 0.84 | 0.6562 | 0.7053 | 13/15 | 0/10 |
| glasses | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 18 | 16 | 9 | 2 | 0.64 | 0.8889 | 0.6905 | 15/15 | 0/10 |
| remote_control | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 18 | 18 | 10 | 0 | 0.6429 | 1.0 | 0.9374 | 15/15 | 1/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 384.8 | 382.92 | 401.74 |
