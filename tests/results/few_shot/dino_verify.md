# Grounding DINO + DINOv3 verification on Member 4's photos (2026-10-21)

**Question:** after Grounding DINO finds a box, does comparing it with the registration photos (DINOv3 features)
help find *my* item and reject everything else?

**Answer: it removes the false alarms, but it does not tell my item from a look-alike.**

- **False alarms dropped to zero.** On photos without the item they fell from 16% to 0% (28 photos) and from 14–28%
  to 0% (valuables).
- **The registered item is still found when the most similar of 5 boxes is taken ("rerank").** On the 28 photos it
  stays at 71%; on the valuables it goes from 25% to 30% (the app's detector, four close-ups). Checking only
  Grounding DINO's top box ("filter") loses some: 62% and 20%.
- **Look-alikes are halved, not solved.** "Said yes" to the other object of the same type fell from 75% to 36–38%
  (28 photos). The user's keys and the other key set were not told apart (5 vs 3 photos, AUC 0.60).
- **The threshold does not carry over from one registration style to another.** It was chosen on the 28 photos
  (0.45–0.5, registration = a photo of the scene) and is too strict for close-up registrations, whose similarities
  are lower (median 0.32 against 0.62). Read off the valuables themselves (not a fair number), rerank at 0.3 finds
  50% with no false alarms, but also says yes to the other key set 2 times of 3.
- **ViT-L is slightly ahead of ViT-B**, by 1–3 photos.

## Setup

| Item | Value |
|---|---|
| Datasets | **28 photos**: 8 objects (4 types × designs A, B) × 3 photos plus 4 empty; one-shot, leave-one-out (each object photo registers with its box, the other 27 are searched). **Valuables**: sunglasses, earbud, keys; 4 close-up registrations each; 20 search photos, 10 no-item photos, 3 photos of another key set |
| Stage 1: candidates | Grounding DINO tiny, top 10 boxes at box threshold 0.05. "app" = the app's multi-prompt setting (single word for the earbud, which is not an app object); "single" = the object word alone |
| Stage 2: verification | DINOv3 ViT-L/16 and ViT-B/16 (LVD-1689M), CLS token (`pooler_output`), L2-normalised. Box + 10% margin, padded to a square, resized to 224 px. Cosine similarity with the registration crops; four-shot = highest of the 4 |
| Pipelines | **Grounding DINO alone**: top box ≥ 0.35 (the app). **filter**: that box, accepted if similarity ≥ τ. **rerank**: the 5 best boxes, the most similar one accepted if similarity ≥ τ |
| Threshold τ | Chosen on the 28 photos (largest found − look-alike said yes − false alarms; ViT-L filter 0.45, others 0.5) and applied unchanged to the valuables. The 28-photo numbers at τ are therefore in-sample; the valuables numbers are not |
| Scoring | "Found" = accepted box with its centre inside the item's box. "Said yes" / "false alarm" = any box accepted |
| Where | Colab T4. Photos sent at the app frame size (long side 1280 px, JPEG quality 95) |

## Results (ViT-L/16, app detector)

**28 photos** (48 same-object searches, 72 other-object searches, 528 photos without the type):

| Pipeline | τ | Same object found | Other object of the type: said yes | False alarms |
|---|---|---|---|---|
| Grounding DINO alone | - | 71% (34/48) | 75% (54/72) | 16% (84/528) |
| filter | 0.45 | 62% (30/48) | 36% (26/72) | **0%** (0/528) |
| rerank | 0.5 | **71%** (34/48) | 38% (27/72) | **0%** (0/528) |

**Valuables, four close-ups** (20 search photos, 36 no-item answers, 3 other-key photos):

| Pipeline | τ (from the 28 photos) | Found | Other key set: said yes | False alarms |
|---|---|---|---|---|
| Grounding DINO alone | - | 25% (5/20) | 1/3 | 14% (5/36) |
| filter | 0.45 | 20% (4/20) | 1/3 | **0%** |
| rerank | 0.5 | **30%** (6/20) | 1/3 | **0%** |
| *rerank, τ 0.3 (read off these photos)* | *0.3* | *50% (10/20)* | *2/3* | *0%* |

Per object, rerank at 0.5: sunglasses 4/8, keys 2/5, earbud 0/7. At 0.3: 6/8, 3/5, 1/7.

For comparison on the same valuables photos ([valuables test](valuables.md)): SAM 3 with the word 45% found, 0%
false alarms, other key set 3/3; Grounding DINO single word 40%, 28%, 2/3.

**With the single-word detector** (more boxes found, more false alarms), ViT-L:

| Data | Grounding DINO alone | filter | rerank |
|---|---|---|---|
| 28 photos: found / look-alike yes / false alarms | 92% / 96% / 60% | 79% / 36% / 0% | 79% / 38% / 0% |
| Valuables four-shot: found / other keys / false alarms | 40% / 2 of 3 / 28% | 30% / 0 of 3 / 0% | 30% / 2 of 3 / 0% |

**The verifier alone** (similarity of the items' own crops; ViT-L, ViT-B in brackets):

| Comparison | AUC | Median similarity, positive vs negative |
|---|---|---|
| 28 photos: same object vs the other object of the type | 0.79 (0.79) | 0.62 vs 0.41 |
| 28 photos: same object vs the best box in photos without the type | 1.00 (1.00) | 0.62 vs 0.13 |
| Valuables: the item vs the best box in no-item photos | 0.79 (0.80) | 0.32 vs 0.11 |
| Valuables: the user's keys vs the best box on the other key set | 0.60 (0.53) | 0.36 vs 0.41 (5 vs 3 photos) |

## Why

- **Things that are not the item look nothing like it** (similarity about 0.1 against 0.3–0.6), so a threshold
  removes the false alarms almost for free.
- **The other object of the same type looks similar** (0.41 against 0.62), so about half of them pass.
- **Small items match the close-ups weakly.** A 37 px earbud cropped and enlarged to 224 px has little detail left,
  and the close-ups show the items 3–10× larger. A threshold set on scene registrations rejects many of them.
- **Rerank helps because the right box is often among Grounding DINO's candidates but not accepted.** On the
  valuables, rerank found two items for which Grounding DINO had no box at 0.35.

## Takeaways

- **DINOv3 rerank is a good false-alarm filter for the app.** It keeps the registered item about as often as
  Grounding DINO alone and removes nearly all false alarms.
- **It is not "my item" recognition.** Half of the look-alikes pass, and the user's keys were not told from the
  other key set.
- **The threshold has to come from the registration itself,** for example calibrated from the user's own
  registration photos; one fixed value does not fit both scene photos and close-ups.
- **ViT-B is close enough** if speed matters. Speed was not measured.

## Limits

One photographer; 20 valuables search photos and 3 other-key photos. The τ rule was fixed before the run, but on
the 28 photos τ is in-sample. Grounding DINO alone gave the same yes/no as the earlier Mac runs in 636 of 648
(28 photos), 38 of 39 and 59 of 59 (valuables) answers. Item boxes are semi-automatic. Per-photo answers and
candidate boxes are kept locally with the photos.

Code: [`tests/zero_shot/dino_verify.py`](../../zero_shot/dino_verify.py) (`run` on the VM, `report` without a model).
