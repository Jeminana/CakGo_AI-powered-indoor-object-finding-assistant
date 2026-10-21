# Category-level few-shot on the 70 shared images (2026-10-21)

Every photo of the 70 shows a different object, so "register my item" cannot be tested here. These tests use
example photos the way YOLOE's visual prompts are designed: as examples of the **category**.

**Questions**

- Used this way, do YOLOE visual prompts beat YOLOE's text prompts?
- Does a DINOv3 check against labeled examples of the category cut Grounding DINO multi's false alarms on the team's
  shared benchmark?

**Answers**

- **YOLOE visual prompts with 14 examples beat YOLOE's text prompts, but not Grounding DINO.**
  - YOLOE-26x gets 40/60 top-1 correct against 30/60 with the word (AP50 0.76 vs 0.57), at the same speed.
  - The gain is mostly keys: 14/15 against 7/15.
  - It raises more false alarms: 14/220 against 6/220.
  - **One example is unreliable:** 14–36/60 depending on which photo is the example (mean 24.5, below the word).
  - Grounding DINO multi stays far ahead at 54/60.
- **The DINOv3 check removes Grounding DINO's genuine false alarms.**
  - With the "filter" pipeline (ViT-L/16) and a threshold taken from the labeled examples alone, false alarms fall
    from 24/220 to 8/220.
  - All 8 remaining are "glasses" on key photos whose sunglasses are visible but unlabeled, so they may well be
    correct. On the 48 fully labeled images, false alarms fall from 12/184 to 0/184.
  - The cost is top-1 correct: 54/60 → 46/60, two per category.
- **Picking the most similar box ("rerank") lands on the right object but often on a part of it** (one key of a
  bunch, one lens). It scores 40/60 by the IoU rule but 51/60 by box centre, which is what the app's guidance uses.
- **DINOv3 separates the category from everything else very well:** AUC 0.97.

## Setup

| Item | Value |
|---|---|
| Images | The 70 Day 3 images (`tests/dataset/manifest.csv`, Objects In House, CC0): 15 each of wallet, keys, glasses, remote control, plus 10 without any |
| A. YOLOE visual prompts | YOLOE-26s/26x, `refer_image` with the example's labeled boxes; one class per query, same settings as the text runs (conf 0.05, iou 0.7, imgsz 640, agnostic NMS). **One-shot**: 15 runs, in run k the k-th image of each category is the example (the next one when the query is that image). **14 examples**: the other images of the category, embeddings averaged and L2-normalised |
| B. DINOv3 check | Candidates: the saved boxes of the Grounding DINO multi run (`t4_gdino-tiny_multi`). References: DINOv3 ViT-L/16 or ViT-B/16 crops of the category's labeled boxes in the other images. Similarity: highest cosine ("max") or cosine with their mean ("prototype"). **filter**: Grounding DINO's top box (≥ 0.35) kept if similar enough. **rerank**: the most similar of its 5 best boxes (≥ 0.05) kept if similar enough |
| Threshold of the check | From the references alone: the 10th percentile of their own leave-one-image-out similarities (median about 0.6). No test labels used |
| Scoring | `compare_zero_shot.summary` (Day 3 protocol: top-1 correct = IoU ≥ 0.5) and its box-centre diagnostic; YOLOE at its default 0.25. "Without the 12" = without the images whose targets are not fully labeled (`GT_ISSUES`) |
| Fixed before the run | All of the above ([`category_prompts_70.py`](../../../zero_shot/category_prompts_70.py) docstring) |
| Where | Colab T4, ultralytics 8.4.177, transformers 5.20.1 |

## A. YOLOE visual prompts (threshold 0.25)

| Model | Prompt | Top-1 correct (of 60) | False alarms (of 220) | Mean AP50 | Without the 12: top-1 / false alarms |
|---|---|---|---|---|---|
| YOLOE-26s | word | 27 | 1 | 0.524 | 22/48, 1/184 |
| YOLOE-26s | 14 examples | 32 | 7 | 0.571 | 25/48, 3/184 |
| YOLOE-26s | 1 example (mean of 15 runs, range) | 20.1 (12–28) | 5.2 (0–13) | 0.393 | - |
| YOLOE-26x | word | 30 | 6 | 0.567 | 25/48, 3/184 |
| YOLOE-26x | **14 examples** | **40** | 14 | **0.756** | 31/48, 5/184 |
| YOLOE-26x | 1 example (mean of 15 runs, range) | 24.5 (14–36) | 8.3 (0–16) | 0.471 | - |
| *Grounding DINO multi (app, 0.35)* | *word* | *54* | *24* | *0.729* | *43/48, 12/184* |

**Per category**, top-1 correct of 15 (word → 14 examples):

| Model | Wallet | Keys | Glasses | Remote control |
|---|---|---|---|---|
| YOLOE-26s | 4 → 0 | 5 → 12 | 6 → 7 | 12 → 13 |
| YOLOE-26x | 3 → 3 | 7 → 14 | 8 → 9 | 12 → 14 |

**At matched false alarms** (sweep in [`tables.md`](tables.md)):

- YOLOE-26x with 14 examples at 0.5: 34/60 with 6 false alarms.
- That is more than its word prompt at 0.25 (30/60 with 6), and close to Grounding DINO multi at 0.5 (35/60 with 8).
- **Speed per query** (T4) is the same as with the word: 22 ms (26s) and 144 ms (26x).

## B. Grounding DINO multi + DINOv3 check (threshold from the references)

| Verifier | Similarity | Pipeline | Top-1 correct (of 60) | Top box centre on the object (of 60) | False alarms (of 220) | Without the 12: top-1 / false alarms |
|---|---|---|---|---|---|---|
| - | - | Grounding DINO alone (0.35) | 54 | 55 | 24 | 43/48, 12/184 |
| ViT-L/16 | prototype | filter | 46 | 47 | **8** | 37/48, **0/184** |
| ViT-L/16 | max | filter | 45 | 46 | **8** | 36/48, **0/184** |
| ViT-B/16 | max | filter | 47 | 48 | 9 | 37/48, 1/184 |
| ViT-B/16 | prototype | filter | 47 | 48 | 10 | 37/48, 1/184 |
| ViT-L/16 | prototype | rerank | 40 | **51** | 9 | 33/48, 0/184 |
| ViT-L/16 | max | rerank | 39 | 49 | 8 | 33/48, 0/184 |
| ViT-B/16 | max | rerank | 43 | 51 | 10 | 33/48, 1/184 |
| ViT-B/16 | prototype | rerank | 45 | 51 | 10 | 35/48, 1/184 |

**With a fixed threshold** (ViT-L/16 max, filter; read off these images, so not a fair operating point):

| Threshold | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|
| Top-1 correct / false alarms | 53 / 13 | 53 / 12 | 51 / 9 | 43 / 7 |

**Verifier alone:** the labeled object against the most similar box in images without the category gives AUC 0.97
for both models and both similarities (median similarity 0.74–0.78 against 0.10–0.20).

## Why

- **YOLOE's visual prompts need many examples.** Averaging 14 examples works; a single example depends on the
  photo (14–36 of 60). This matches the YOLOE paper, which evaluates visual prompts with 16 examples per category.
- **Keys gain the most from examples.** A bunch of keys is easier to show than to name; the word prompt finds 7 of
  15, the examples 14 of 15.
- **False alarms look nothing like the category** (similarity about 0.1–0.2), so the check removes them. The
  references' own similarities set a strict threshold (about 0.6). It also rejects some real objects that look
  unlike the other examples: two per category.
- **Rerank's boxes are on the right object but not of the labeled extent.** By box centre it keeps 51 of 60, by
  IoU only 40. The category references match parts of an object (one key, one lens) about as well as the whole.

## Takeaways

- **Keep Grounding DINO multi; add the DINOv3 check as a false-alarm filter.** On this benchmark it removes the
  genuine false alarms. A threshold calibrated on the references costs 8 of 60 top-1 answers; a looser one (around
  0.5) would cost about 3. This needs confirming on other photos before it is fixed.
- **YOLOE visual prompts are an option only with many examples** (e.g. a category gallery), not with one photo.
  They still trail Grounding DINO multi.
- **For "find my item" the earlier real-photo results stand** ([`../../few_shot/README.md`](../../few_shot/README.md)).
  Category examples tell a wallet from a remote, not one wallet from another.

## Limits

- 15 images per category, stock-style photos.
- 12 images have targets that are visible but unlabeled. Their "false alarms" may be correct; both views are shown.
- The DINOv3 threshold rule (10th percentile) and the crop rule were fixed before the run. The fixed-threshold
  sweep is read off the same images.

**Files**

- [`tables.md`](tables.md): all tables.
- `t4_yoloe-*_visual_loo/`, `t4_yoloe-*_visual_1shot/`: YOLOE predictions in the usual format.
- [`dinov3_per_query.csv`](dinov3_per_query.csv): every DINOv3 decision.
- Code: [`tests/zero_shot/category_prompts_70.py`](../../../zero_shot/category_prompts_70.py).
