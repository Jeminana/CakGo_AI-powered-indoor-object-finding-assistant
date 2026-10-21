<!-- manifest sha256: ca73e2c1f12464df03b1f2b3824109cd237ea1e9dbba46fa8d5ac4c6db5bde4b; GPUs of new runs: Tesla T4 -->

### Reproduction check (same model, earlier run, each at 0.35)

| Run | Earlier run | Category | TP / FP / FN now | TP / FP / FN earlier | AP50 now / earlier |
|---|---|---|---|---|---|
| t4_gdino-tiny_multi | rtx4050_gdino-tiny_multi_box0.05 | wallet | 12 / 7 / 5 | 12 / 7 / 5 | 0.584 / 0.584 |
| t4_gdino-tiny_multi | rtx4050_gdino-tiny_multi_box0.05 | keys | 21 / 4 / 11 | 21 / 4 / 11 | 0.705 / 0.695 |
| t4_gdino-tiny_multi | rtx4050_gdino-tiny_multi_box0.05 | glasses | 16 / 9 / 2 | 16 / 9 / 2 | 0.691 / 0.691 |
| t4_gdino-tiny_multi | rtx4050_gdino-tiny_multi_box0.05 | remote_control | 18 / 10 / 0 | 18 / 10 / 0 | 0.937 / 0.937 |

### App view at each model's operating threshold (all 70 images)

| Model | Conf | Top-1 correct | Target images detected | False alarms | Negatives with FP | Mean AP50 | Median ms / request | p95 ms | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|
| Grounding DINO tiny multi | 0.35 | 54/60 | 55/60 | 24/220 | 2/40 | 0.729 | 382.92 | 401.74 | 1532.1 |
| YOLOE-26s single | 0.25 | 27/60 | 27/60 | 1/220 | 0/40 | 0.524 | 21.92 | 25.57 | 309.7 |
| YOLOE-26s multi | 0.25 | 22/60 | 22/60 | 1/220 | 0/40 | 0.435 | 22.71 | 27.02 | 150.8 |
| YOLOE-26x single | 0.25 | 30/60 | 33/60 | 6/220 | 1/40 | 0.567 | 136.66 | 140.21 | 1390.4 |
| YOLOE-26x multi | 0.25 | 29/60 | 32/60 | 6/220 | 1/40 | 0.517 | 137.94 | 140.94 | 494.9 |
| SAM 3 single | 0.5 | 44/60 | 56/60 | 24/220 | 2/40 | 0.648 | 2239.99 | 2523.39 | 4570.7 |
| SAM 3 multi | 0.5 | 44/60 | 56/60 | 23/220 | 2/40 | 0.648 | 6526.31 | 6578.8 | 4570.7 |
| Grounding DINO tiny single (Day 3, T4) | 0.35 | 55/60 | 58/60 | 176/220 | 28/40 | 0.623 | 331.48 | 363.78 | 1480.8 |

### Same, excluding 12 images with unlabeled targets

| Model | Conf | Top-1 correct | Target images detected | False alarms | Negatives with FP | Mean AP50 | Median ms / request | p95 ms | Peak VRAM (MB) |
|---|---|---|---|---|---|---|---|---|---|
| Grounding DINO tiny multi | 0.35 | 43/48 | 43/48 | 12/184 | 2/40 | 0.773 | 382.92 | 401.74 | 1532.1 |
| YOLOE-26s single | 0.25 | 22/48 | 22/48 | 1/184 | 0/40 | 0.531 | 21.92 | 25.57 | 309.7 |
| YOLOE-26s multi | 0.25 | 17/48 | 17/48 | 1/184 | 0/40 | 0.445 | 22.71 | 27.02 | 150.8 |
| YOLOE-26x single | 0.25 | 25/48 | 26/48 | 3/184 | 1/40 | 0.595 | 136.66 | 140.21 | 1390.4 |
| YOLOE-26x multi | 0.25 | 24/48 | 25/48 | 3/184 | 1/40 | 0.518 | 137.94 | 140.94 | 494.9 |
| SAM 3 single | 0.5 | 42/48 | 45/48 | 14/184 | 2/40 | 0.733 | 2239.99 | 2523.39 | 4570.7 |
| SAM 3 multi | 0.5 | 42/48 | 45/48 | 13/184 | 2/40 | 0.733 | 6526.31 | 6578.8 | 4570.7 |
| Grounding DINO tiny single (Day 3, T4) | 0.35 | 44/48 | 46/48 | 147/184 | 28/40 | 0.606 | 331.48 | 363.78 | 1480.8 |

### Common thresholds (all 70 images): top-1 correct / false alarms

| Model | 0.1 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|
| Grounding DINO tiny multi | 55 / 100 | 55 / 50 | 54 / 38 | 54 / 27 | 54 / 24 | 52 / 19 | 35 / 8 |
| YOLOE-26s single | 32 / 15 | 29 / 4 | 27 / 1 | 23 / 1 | 22 / 1 | 22 / 0 | 18 / 0 |
| YOLOE-26s multi | 27 / 10 | 24 / 3 | 22 / 1 | 20 / 1 | 19 / 1 | 19 / 0 | 16 / 0 |
| YOLOE-26x single | 35 / 17 | 32 / 9 | 30 / 6 | 28 / 5 | 26 / 3 | 24 / 3 | 21 / 2 |
| YOLOE-26x multi | 32 / 13 | 31 / 7 | 29 / 6 | 28 / 5 | 26 / 3 | 24 / 3 | 21 / 2 |
| SAM 3 single | 44 / 42 | 44 / 30 | 44 / 29 | 44 / 29 | 44 / 27 | 44 / 26 | 44 / 24 |
| SAM 3 multi | 44 / 37 | 44 / 30 | 44 / 29 | 44 / 29 | 44 / 27 | 44 / 26 | 44 / 23 |
| Grounding DINO tiny single (Day 3, T4) | 55 / 215 | 55 / 204 | 55 / 198 | 55 / 190 | 55 / 176 | 55 / 157 | 51 / 111 |

Top-1 correct is out of 60 target images; false alarms are out of 220 queries for an object the image has no box of.

### Diagnostic: top box correct (IoU >= 0.5) / top box centre on a labeled object

Not part of the protocol. Each cell: target images out of 15, at the model's operating threshold.

| Model | wallet | keys | glasses | remote_control |
|---|---|---|---|---|
| Grounding DINO tiny multi | 11 / 11 | 13 / 14 | 15 / 15 | 15 / 15 |
| YOLOE-26s single | 4 / 4 | 5 / 5 | 6 / 7 | 12 / 12 |
| YOLOE-26s multi | 4 / 4 | 5 / 5 | 6 / 7 | 7 / 7 |
| YOLOE-26x single | 3 / 3 | 7 / 8 | 8 / 9 | 12 / 12 |
| YOLOE-26x multi | 3 / 3 | 6 / 7 | 8 / 9 | 12 / 12 |
| SAM 3 single | 12 / 13 | 2 / 15 | 15 / 15 | 15 / 15 |
| SAM 3 multi | 12 / 13 | 2 / 15 | 15 / 15 | 15 / 15 |
| Grounding DINO tiny single (Day 3, T4) | 13 / 15 | 12 / 12 | 15 / 15 | 15 / 15 |

### Per category at each model's operating threshold (all 70 images)

| Category | Model | Conf | AP50 | Precision | Recall | Target images detected | Top-1 correct | Negatives with FP |
|---|---|---|---|---|---|---|---|---|
| wallet | Grounding DINO tiny multi | 0.35 | 0.584 | 0.632 | 0.706 | 12/15 | 11/15 | 1/10 |
| wallet | YOLOE-26s single | 0.25 | 0.238 | 0.667 | 0.235 | 4/15 | 4/15 | 0/10 |
| wallet | YOLOE-26s multi | 0.25 | 0.201 | 0.667 | 0.235 | 4/15 | 4/15 | 0/10 |
| wallet | YOLOE-26x single | 0.25 | 0.257 | 0.571 | 0.235 | 4/15 | 3/15 | 1/10 |
| wallet | YOLOE-26x multi | 0.25 | 0.260 | 0.571 | 0.235 | 4/15 | 3/15 | 1/10 |
| wallet | SAM 3 single | 0.5 | 0.803 | 0.727 | 0.941 | 14/15 | 12/15 | 1/10 |
| wallet | SAM 3 multi | 0.5 | 0.803 | 0.727 | 0.941 | 14/15 | 12/15 | 1/10 |
| wallet | Grounding DINO tiny single (Day 3, T4) | 0.35 | 0.521 | 0.273 | 0.882 | 13/15 | 13/15 | 4/10 |
| keys | Grounding DINO tiny multi | 0.35 | 0.705 | 0.840 | 0.656 | 13/15 | 13/15 | 0/10 |
| keys | YOLOE-26s single | 0.25 | 0.530 | 1.000 | 0.219 | 5/15 | 5/15 | 0/10 |
| keys | YOLOE-26s multi | 0.25 | 0.463 | 1.000 | 0.188 | 5/15 | 5/15 | 0/10 |
| keys | YOLOE-26x single | 0.25 | 0.617 | 0.765 | 0.406 | 9/15 | 7/15 | 0/10 |
| keys | YOLOE-26x multi | 0.25 | 0.522 | 0.750 | 0.375 | 8/15 | 6/15 | 0/10 |
| keys | SAM 3 single | 0.5 | 0.079 | 0.044 | 0.500 | 12/15 | 2/15 | 0/10 |
| keys | SAM 3 multi | 0.5 | 0.079 | 0.045 | 0.500 | 12/15 | 2/15 | 0/10 |
| keys | Grounding DINO tiny single (Day 3, T4) | 0.35 | 0.333 | 0.229 | 1.000 | 15/15 | 12/15 | 9/10 |
| glasses | Grounding DINO tiny multi | 0.35 | 0.691 | 0.640 | 0.889 | 15/15 | 15/15 | 0/10 |
| glasses | YOLOE-26s single | 0.25 | 0.524 | 0.857 | 0.333 | 6/15 | 6/15 | 0/10 |
| glasses | YOLOE-26s multi | 0.25 | 0.541 | 0.857 | 0.333 | 6/15 | 6/15 | 0/10 |
| glasses | YOLOE-26x single | 0.25 | 0.598 | 0.727 | 0.444 | 8/15 | 8/15 | 0/10 |
| glasses | YOLOE-26x multi | 0.25 | 0.481 | 0.727 | 0.444 | 8/15 | 8/15 | 0/10 |
| glasses | SAM 3 single | 0.5 | 0.709 | 0.621 | 1.000 | 15/15 | 15/15 | 1/10 |
| glasses | SAM 3 multi | 0.5 | 0.709 | 0.621 | 1.000 | 15/15 | 15/15 | 1/10 |
| glasses | Grounding DINO tiny single (Day 3, T4) | 0.35 | 0.696 | 0.258 | 0.944 | 15/15 | 15/15 | 6/10 |
| remote_control | Grounding DINO tiny multi | 0.35 | 0.937 | 0.643 | 1.000 | 15/15 | 15/15 | 1/10 |
| remote_control | YOLOE-26s single | 0.25 | 0.804 | 0.929 | 0.722 | 12/15 | 12/15 | 0/10 |
| remote_control | YOLOE-26s multi | 0.25 | 0.537 | 0.889 | 0.444 | 7/15 | 7/15 | 0/10 |
| remote_control | YOLOE-26x single | 0.25 | 0.795 | 0.867 | 0.722 | 12/15 | 12/15 | 0/10 |
| remote_control | YOLOE-26x multi | 0.25 | 0.806 | 0.867 | 0.722 | 12/15 | 12/15 | 0/10 |
| remote_control | SAM 3 single | 0.5 | 1.000 | 1.000 | 1.000 | 15/15 | 15/15 | 0/10 |
| remote_control | SAM 3 multi | 0.5 | 1.000 | 1.000 | 1.000 | 15/15 | 15/15 | 0/10 |
| remote_control | Grounding DINO tiny single (Day 3, T4) | 0.35 | 0.941 | 0.198 | 1.000 | 15/15 | 15/15 | 9/10 |

