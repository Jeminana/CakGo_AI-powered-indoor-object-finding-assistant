# Grounding DINO evaluation - t4_gdino-tiny_box0.05

- Model: `IDEA-Research/grounding-dino-tiny` (zero-shot, no fine-tuning)
- Images: 70; confidence >= 0.25; IoU >= 0.5

| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Negative images with FP |
|---|---|---|---|---|---|---|---|---|---|---|
| wallet | `wallet.` | 17 | 15 | 61 | 2 | 0.1974 | 0.8824 | 0.5213 | 13/15 | 5/10 |
| keys | `keys.` | 32 | 32 | 181 | 0 | 0.1502 | 1.0 | 0.3334 | 15/15 | 10/10 |
| glasses | `glasses. sunglasses.` | 18 | 18 | 88 | 0 | 0.1698 | 1.0 | 0.6956 | 15/15 | 9/10 |
| remote_control | `remote control.` | 18 | 18 | 144 | 0 | 0.1111 | 1.0 | 0.941 | 15/15 | 10/10 |

| Latency (ms) | mean | median | p95 |
|---|---|---|---|
| wall_ms_per_query | 333.88 | 331.48 | 363.78 |

- AP50 only uses predictions saved at confidence >= inference_min_confidence (run_inference.py --box-threshold).
- Each image is queried once per category; latency is per query, i.e. per app request.
