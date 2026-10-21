# Register-then-find test on Member 4's valuables photos (2026-10-21)

**Question:** with 1 or 4 close-up example photos, do YOLOE (visual prompts) or SAM 3 (exemplar boxes) find a
registered item in a room better than with a word?

**Answer: no.** The examples never beat the word.

- **Few-shot found less than the word, in both models.**
  - YOLOE-26x: 8% with one example, 10% with four (embeddings averaged). With the word: 20%.
  - SAM 3 with the examples (a workaround, see Setup): 29–30%. With the word: 45%.
- **SAM 3 with the word did best of all:** 45% found, no false alarms. On keys and sunglasses (the app's objects):
  8/13 found with 0/23 false alarms, against Grounding DINO multi's 5/13 with 2/23. It is about 6× slower per
  request (2.24 s vs 0.38 s on a T4, [70-image test](../zero_shot_70/COMPARISON.md)).
- **No method tells the user's keys from another key set.** SAM 3 with the word said yes to the other set 3 times
  out of 3.
- **Closing the size gap helped YOLOE few-shot only a little:** at best 18–20%, the level of the word (follow-up
  below).
- **The earbud was almost never found.** It is about 37 px in the search photos.

## Setup

| Item | Value |
|---|---|
| Objects | Sunglasses, one wireless earbud, a key ring (the photographer's own); another key set as the look-alike |
| Photos (52) | 12 registration close-ups (4 per object, different angles, same surface); 20 search photos with the item visible (sunglasses 8, earbud 7, keys 5); 10 photos of the same scenes without the items; 3 photos with only the other key set; 7 excluded because the item, although in the scene, was not visible |
| YOLOE, Grounding DINO | YOLOE-26s/26x: one-shot (each example alone, `refer_image` + its box), four-shot (4 embeddings averaged, L2-normalised), text ("sunglasses", "earbud", "keys"). Grounding DINO tiny: single (the same words) and multi = the app's setting (keys and glasses only; the earbud is not an app object) |
| SAM 3 | transformers `Sam3Model`, fp32. Text: the same words. Exemplars: SAM 3 takes exemplar boxes only inside the image it searches, so the registration crops were pasted into a strip next to the search photo and given as positive boxes; only detections in the original photo count. One-shot, four-shot, text + four-shot. **A workaround, not an official API** (other integrations composite exemplars the same way) |
| Queries | Every method asks about the object's search photos, the no-item photos and the other-key photos. False alarms are counted on the no-item photos (other-key photos too for sunglasses and earbud) |
| "Box on it" | Top box centre inside the item's box |
| Thresholds | YOLOE 0.25, Grounding DINO 0.35, SAM 3 0.5 (library defaults) |
| Where | YOLOE, Grounding DINO: Member 4's Mac (MPS). SAM 3 and the follow-up: Colab T4 |

## Result (default thresholds)

| Method | Box on the item (20 search photos) | False alarms (no-item photos) | Other key set: said yes |
|---|---|---|---|
| YOLOE-26x one-shot | 8% (6/80 searches) | 1% | 0/12 |
| YOLOE-26x four-shot | 10% (2/20) | 3% | 0/3 |
| YOLOE-26x text | 20% (4/20) | **0%** | 1/3 |
| Grounding DINO single (word) | 40% (8/20) | 28% | 2/3 |
| Grounding DINO multi (app; keys + sunglasses) | 38% (5/13) | 9% | 1/3 |
| **SAM 3 text** | **45%** (9/20) | **0%** | 3/3 |
| SAM 3 one-shot (exemplar strip) | 29% (23/80 searches) | 1% | 4/12 |
| SAM 3 four-shot (exemplar strip) | 30% (6/20) | **0%** | 1/3 |
| SAM 3 text + four-shot (exemplar strip) | 30% (6/20) | **0%** | 1/3 |
| YOLOE-26s (one-shot, four-shot, text) | 0% | 0% | 0 |

One-shot rows pool the 4 registration photos (each searched separately).

**Per object** (box on the item / search photos, false alarms / no-item photos):

| Method | Sunglasses | Earbud | Keys |
|---|---|---|---|
| YOLOE-26x one-shot | 2/32, FA 0/52 | 0/28, FA 1/52 | 4/20, FA 0/40 |
| YOLOE-26x four-shot | 1/8, FA 0/13 | 0/7, FA 0/13 | 1/5, FA 1/10 |
| YOLOE-26x text | 3/8, FA 0/13 | 0/7, FA 0/13 | 1/5, FA 0/10 |
| Grounding DINO single | 5/8, FA 5/13 | 0/7, FA 3/13 | 3/5, FA 2/10 |
| Grounding DINO multi (app) | 4/8, FA 2/13 | - | 1/5, FA 0/10 |
| SAM 3 text | 5/8, FA 0/13 | 1/7, FA 0/13 | 3/5, FA 0/10 |
| SAM 3 one-shot (exemplar strip) | 17/32, FA 0/52 | 0/28, FA 2/52 | 6/20, FA 0/40 |
| SAM 3 four-shot (exemplar strip) | 5/8, FA 0/13 | 0/7, FA 0/13 | 1/5, FA 0/10 |

**Threshold sweep** (box on the item / false alarms, all objects):

| Method | 0.05 | 0.1 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|---|
| YOLOE-26x one-shot | 15% / 14% | 15% / 8% | 8% / 1% | 5% / 0% | 5% / 0% |
| YOLOE-26x text | 35% / 8% | 25% / 3% | 20% / 0% | 20% / 0% | 15% / 0% |
| Grounding DINO single | 55% / 100% | 55% / 100% | 55% / 86% | 40% / 28% | 25% / 0% |
| Grounding DINO multi (app) | 69% / 87% | 69% / 61% | 62% / 22% | 38% / 9% | 23% / 0% |
| SAM 3 text | 55% / 25% | 50% / 11% | 50% / 3% | 50% / 3% | 45% / 0% |
| SAM 3 four-shot (exemplar strip) | 45% / 94% | 45% / 58% | 30% / 11% | 30% / 3% | 30% / 0% |

## Why

- **The items are small in the search photos.** Median long side in the 1280 px frame: earbud 37 px, sunglasses
  163 px, keys 162 px. YOLOE sees the frame at 640×480, so the earbud becomes about 18 px; Grounding DINO sees it
  at 1066×800 (about 31 px).
- **The examples are 3–10× larger than the targets** (median long side; 11–94× by area): earbud 358 px (9.7×),
  keys 534 px (3.3×), sunglasses 1,046 px (6.4×) in the close-ups.
- **YOLOE few-shot only worked when the target looked like the example:** its finds were the keys lying on the
  same surface as the registration photos and the largest sunglasses.
- **YOLOE's visual prompts are meant for categories and collapse across photos** ([one-shot test](one_shot_28_photos.md)).
  Averaging four examples did not change that.
- **SAM 3's examples did not help either.** Adding them to the word lowered its hits from 45% to 30% at 0.5. A
  likely reason, not tested: SAM 3 learns exemplars from boxes in the same image, and a strip of crops from other
  photos is outside that.

## Follow-up: does closing the size gap help YOLOE?

Fixes chosen before the run, YOLOE-26x, default threshold 0.25, Colab T4:

- **Input resolution:** 1280 px instead of 640, for registration and search.
- **Several sizes:** each registration photo also shrunk so the item is 320, 160, 80 and 40 px; the 5 embeddings
  averaged ("avg") or kept as 5 prototypes, best match wins ("max").
- **Scale-matched** (diagnostic upper bound): the item shrunk to the median size of that object in the search photos.

The plain rows at 640 on the T4 gave the same hit as the earlier Mac run for 353 of 354 answers.

| YOLOE-26x | Box on the item @640 | False alarms @640 | Box on the item @1280 | False alarms @1280 |
|---|---|---|---|---|
| one-shot | 6% (5/80) | 1% | 8% (6/80) | 1% |
| one-shot, several sizes avg | 5% | 0% | 11% | 0% |
| one-shot, several sizes max | 11% | 1% | **18%** (14/80) | 1% |
| one-shot, scale-matched (upper bound) | 4% | 0% | 6% | 0% |
| four-shot | 10% (2/20) | 3% | 10% | 3% |
| four-shot, several sizes avg | 5% | 0% | 15% | 0% |
| four-shot, several sizes max | 15% | 3% | **20%** (4/20) | 3% |
| four-shot, scale-matched (upper bound) | 10% | 0% | 5% | 0% |
| text | 20% (4/20) | 0% | 20% | 14% |

- **Resolution alone changed little:** text stayed at 20% while its false alarms rose from 0% to 14%.
- **Matching the example's size to the target did not help (4–10%):** shrinking the example also throws away detail.
- **The best fix reached 18–20%:** the level of the word, still below Grounding DINO multi (38%) and SAM 3 (45%).
- **Every few-shot find was an item of 124 px or more;** no earbud was found with examples.
- **So the size gap is part of the problem, not the main part.** In the 28-photo test the items were larger
  (median 278 px) and one-shot still found 27% against Grounding DINO's 71%.

## Takeaways

- **Grounding DINO multi stays the app's model; SAM 3 with the word is the one to watch** (more found, fewer false
  alarms, about 6× slower).
- **Few-shot (YOLOE visual prompts, SAM 3 exemplar strip) is not a way to find a registered item in a room.**
- **Small items are the real limit.** Untested: tiling the frame; app-side: guiding the user to scan closer.

## Limits

20 search photos and one photographer, so this is a rough signal. The item boxes are semi-automatic; the smallest
target is 27 px. The follow-up tried 16 variants on the same 20 search photos, so a difference of 1–2 photos is
noise. Per-photo answers are kept locally with the photos.

Code: [`valuables_test.py`](../../zero_shot/valuables_test.py), [`valuables_sam3.py`](../../zero_shot/valuables_sam3.py),
[`valuables_yoloe_scale.py`](../../zero_shot/valuables_yoloe_scale.py).
