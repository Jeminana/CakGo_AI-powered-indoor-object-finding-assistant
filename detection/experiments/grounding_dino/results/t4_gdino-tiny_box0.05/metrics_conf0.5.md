# Grounding DINO evaluation - t4_gdino-tiny_box0.05

- Model: `IDEA-Research/grounding-dino-tiny` (zero-shot, no fine-tuning)
- Images: 70; confidence >= 0.5; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet.` | 17 | 11 | 20 | 6 | 0.3548 | 0.6471 | 0.5213 | 11/15 | 2/10 |
| keys | `keys.` | 32 | 25 | 64 | 7 | 0.2809 | 0.7812 | 0.3334 | 12/15 | 7/10 |
| glasses | `glasses. sunglasses.` | 18 | 15 | 27 | 3 | 0.3571 | 0.8333 | 0.6956 | 14/15 | 2/10 |
| remote_control | `remote control.` | 18 | 18 | 27 | 0 | 0.4 | 1.0 | 0.941 | 15/15 | 2/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 333.88 | 331.48 | 363.78 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence (run_inference.py --box-threshold).
- Each image is queried once per category; latency is per query, i.e. per app request.
