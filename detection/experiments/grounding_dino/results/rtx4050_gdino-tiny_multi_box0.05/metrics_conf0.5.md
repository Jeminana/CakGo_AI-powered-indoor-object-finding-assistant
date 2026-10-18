# Grounding DINO evaluation - rtx4050_gdino-tiny_multi_box0.05

- Model: `IDEA-Research/grounding-dino-tiny` (zero-shot, no fine-tuning)
- Images: 70; confidence >= 0.5; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 17 | 6 | 2 | 11 | 0.75 | 0.3529 | 0.5838 | 6/15 | 0/10 |
| keys | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 32 | 4 | 1 | 28 | 0.8 | 0.125 | 0.6953 | 3/15 | 0/10 |
| glasses | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 18 | 13 | 5 | 5 | 0.7222 | 0.7222 | 0.6905 | 12/15 | 0/10 |
| remote_control | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 18 | 16 | 1 | 2 | 0.9412 | 0.8889 | 0.9374 | 14/15 | 0/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 320.91 | 319.08 | 342.68 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence (run_inference.py --box-threshold).
- Each image is queried once per category; latency is per query, i.e. per app request.
