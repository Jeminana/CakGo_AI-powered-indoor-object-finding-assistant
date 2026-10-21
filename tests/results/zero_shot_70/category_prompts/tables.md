# Category-level few-shot on the 70 shared images

## A. YOLOE visual prompts (threshold 0.25)

| Model | Prompt | Top-1 correct (of 60) | False alarms (of 220) | Negatives with a box (of 40) | Mean AP50 | Top-1 without the 12 unlabeled-target images (of 48) | False alarms without them |
|---|---|---|---|---|---|---|---|
| yoloe-26s | text (earlier run) | 27/60 | 1/220 | 0/40 | 0.524 | 22/48 | 1/184 |
| yoloe-26s | visual, leave-one-out (14 examples) | 32/60 | 7/220 | 1/40 | 0.571 | 25/48 | 3/184 |
| yoloe-26s | visual, one-shot (mean of 15 runs, min–max) | 20.1/60 (12–28) | 5.2/220 (0–13) | 0.9/40 | 0.393 | - | - |
| yoloe-26x | text (earlier run) | 30/60 | 6/220 | 1/40 | 0.567 | 25/48 | 3/184 |
| yoloe-26x | visual, leave-one-out (14 examples) | 40/60 | 14/220 | 0/40 | 0.756 | 31/48 | 5/184 |
| yoloe-26x | visual, one-shot (mean of 15 runs, min–max) | 24.5/60 (14–36) | 8.3/220 (0–16) | 1.3/40 | 0.471 | - | - |

Top-1 correct per category (wallet | keys | glasses | remote control):

| Model | Prompt | wallet | keys | glasses | remote control |
|---|---|---|---|---|---|
| yoloe-26s | text (earlier run) | 4/15 | 5/15 | 6/15 | 12/15 |
| yoloe-26s | visual, leave-one-out (14 examples) | 0/15 | 12/15 | 7/15 | 13/15 |
| yoloe-26x | text (earlier run) | 3/15 | 7/15 | 8/15 | 12/15 |
| yoloe-26x | visual, leave-one-out (14 examples) | 3/15 | 14/15 | 9/15 | 14/15 |

Threshold sweep, top-1 correct / false alarms (all 70 images):

| Model | Prompt | 0.1 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| yoloe-26s | text (earlier run) | 32 / 15 | 29 / 4 | 27 / 1 | 23 / 1 | 22 / 1 | 22 / 0 | 18 / 0 |
| yoloe-26s | visual, leave-one-out (14 examples) | 38 / 22 | 32 / 9 | 32 / 7 | 30 / 3 | 29 / 2 | 28 / 2 | 26 / 1 |
| yoloe-26s | visual, one-shot (mean) | 25.3 / 17.4 | 21.5 / 7.3 | 20.1 / 5.2 | 18.2 / 3.5 | 17.3 / 2.7 | 16.1 / 2.0 | 13.7 / 1.0 |
| yoloe-26x | text (earlier run) | 35 / 17 | 32 / 9 | 30 / 6 | 28 / 5 | 26 / 3 | 24 / 3 | 21 / 2 |
| yoloe-26x | visual, leave-one-out (14 examples) | 48 / 23 | 44 / 17 | 40 / 14 | 39 / 12 | 36 / 10 | 35 / 9 | 34 / 6 |
| yoloe-26x | visual, one-shot (mean) | 31.7 / 20.0 | 26.2 / 10.8 | 24.5 / 8.3 | 22.5 / 6.8 | 20.7 / 5.5 | 19.0 / 4.3 | 16.1 / 2.3 |

## B. Grounding DINO multi + DINOv3 verification

Threshold: from the references (10th percentile of their own leave-one-image-out similarities).

| Verifier | Similarity | Pipeline | Top-1 correct (of 60) | False alarms (of 220) | Negatives with a box (of 40) | Top-1 without the 12 (of 48) | False alarms without them | Median threshold |
|---|---|---|---|---|---|---|---|---|
| - | - | Grounding DINO alone (0.35) | 54/60 | 24/220 | 2/40 | 43/48 | 12/184 | - |
| dinov3-vitb16 | max | filter | 47/60 | 9/220 | 1/40 | 37/48 | 1/184 | 0.59 |
| dinov3-vitb16 | max | rerank | 43/60 | 10/220 | 1/40 | 33/48 | 1/184 | 0.59 |
| dinov3-vitb16 | prototype | filter | 47/60 | 10/220 | 1/40 | 37/48 | 1/184 | 0.60 |
| dinov3-vitb16 | prototype | rerank | 45/60 | 10/220 | 1/40 | 35/48 | 1/184 | 0.60 |
| dinov3-vitl16 | max | filter | 45/60 | 8/220 | 0/40 | 36/48 | 0/184 | 0.63 |
| dinov3-vitl16 | max | rerank | 39/60 | 8/220 | 0/40 | 33/48 | 0/184 | 0.63 |
| dinov3-vitl16 | prototype | filter | 46/60 | 8/220 | 0/40 | 37/48 | 0/184 | 0.62 |
| dinov3-vitl16 | prototype | rerank | 40/60 | 9/220 | 0/40 | 33/48 | 0/184 | 0.62 |

Fixed-threshold sweep, top-1 correct / false alarms (all 70 images):

| Verifier | Similarity | Pipeline | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| dinov3-vitb16 | max | filter | 54 / 22 | 54 / 17 | 54 / 13 | 53 / 13 | 51 / 12 | 43 / 5 | 33 / 3 | 17 / 0 |
| dinov3-vitb16 | max | rerank | 47 / 121 | 47 / 68 | 47 / 25 | 47 / 14 | 46 / 12 | 40 / 9 | 31 / 3 | 18 / 0 |
| dinov3-vitb16 | prototype | filter | 54 / 16 | 53 / 15 | 53 / 13 | 51 / 13 | 47 / 12 | 44 / 6 | 32 / 4 | 19 / 0 |
| dinov3-vitb16 | prototype | rerank | 49 / 71 | 49 / 32 | 49 / 15 | 47 / 14 | 44 / 12 | 41 / 10 | 33 / 4 | 21 / 0 |
| dinov3-vitl16 | max | filter | 54 / 22 | 54 / 16 | 53 / 13 | 53 / 12 | 51 / 9 | 43 / 7 | 35 / 4 | 24 / 0 |
| dinov3-vitl16 | max | rerank | 44 / 111 | 44 / 59 | 44 / 20 | 44 / 14 | 42 / 9 | 37 / 8 | 33 / 4 | 22 / 0 |
| dinov3-vitl16 | prototype | filter | 54 / 20 | 54 / 13 | 53 / 12 | 53 / 12 | 45 / 10 | 42 / 7 | 34 / 2 | 20 / 0 |
| dinov3-vitl16 | prototype | rerank | 44 / 88 | 44 / 29 | 44 / 15 | 44 / 12 | 38 / 10 | 36 / 9 | 30 / 5 | 20 / 0 |

Verifier alone (AUC): labeled object vs the most similar box in images without the category:

| Verifier | Similarity | AUC | Median labeled object | Median best box without the category |
|---|---|---|---|---|
| dinov3-vitb16 | max | 0.97 | 0.74 | 0.20 |
| dinov3-vitb16 | prototype | 0.97 | 0.74 | 0.10 |
| dinov3-vitl16 | max | 0.97 | 0.78 | 0.18 |
| dinov3-vitl16 | prototype | 0.97 | 0.75 | 0.13 |
