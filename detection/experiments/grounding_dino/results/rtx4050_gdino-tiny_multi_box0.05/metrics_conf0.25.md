# Grounding DINO evaluation - rtx4050_gdino-tiny_multi_box0.05

- Model: `IDEA-Research/grounding-dino-tiny` (zero-shot, no fine-tuning)
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 17 | 13 | 11 | 4 | 0.5417 | 0.7647 | 0.5838 | 12/15 | 1/10 |
| keys | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 32 | 23 | 19 | 9 | 0.5476 | 0.7188 | 0.6953 | 14/15 | 0/10 |
| glasses | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 18 | 16 | 22 | 2 | 0.4211 | 0.8889 | 0.6905 | 15/15 | 1/10 |
| remote_control | `wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup. microphone. cable. charger. nail clipper. coin.` | 18 | 18 | 22 | 0 | 0.45 | 1.0 | 0.9374 | 15/15 | 3/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 320.91 | 319.08 | 342.68 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence (run_inference.py --box-threshold).
- Each image is queried once per category; latency is per query, i.e. per app request.
