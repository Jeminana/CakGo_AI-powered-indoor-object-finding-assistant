# Grounding DINO evaluation - t4_gdino-tiny_box0.05

- Model: `IDEA-Research/grounding-dino-tiny` (zero-shot, no fine-tuning)
- Images: 70; confidence >= 0.4; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet.` | 17 | 15 | 33 | 2 | 0.3125 | 0.8824 | 0.5213 | 13/15 | 4/10 |
| keys | `keys.` | 32 | 29 | 84 | 3 | 0.2566 | 0.9062 | 0.3334 | 15/15 | 7/10 |
| glasses | `glasses. sunglasses.` | 18 | 17 | 39 | 1 | 0.3036 | 0.9444 | 0.6956 | 15/15 | 5/10 |
| remote_control | `remote control.` | 18 | 18 | 54 | 0 | 0.25 | 1.0 | 0.941 | 15/15 | 6/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 333.88 | 331.48 | 363.78 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence (run_inference.py --box-threshold).
- Each image is queried once per category; latency is per query, i.e. per app request.
