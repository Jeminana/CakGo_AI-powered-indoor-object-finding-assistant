# Zero-shot and few-shot experiments — Member 4 (2026-10-20/21)

Exploratory tests on branch `feature/zero-shot-yoloe-sam3`, not a model decision. Two questions:

- Does a newer open-vocabulary model (YOLOE-26, SAM 3) beat the app's Grounding DINO multi-prompt?
- Can a user register an item with one or a few photos so the app finds *that* item?

The real-photo tests used Member 4's own photos. The photos and everything derived from them (file names, boxes,
per-photo answers) stay on Member 4's machine; these pages hold the aggregate results only.

## Experiments and results

| # | Experiment | Question | Result | Details |
|---|---|---|---|---|
| 1 | Zero-shot, 70 shared images (T4) | Does a newer model beat Grounding DINO with words only? | **No.** Grounding DINO multi: 54/60 top-1, 24/220 false alarms, 0.38 s per request. SAM 3: box on 58/60, same false alarms, 2.24 s. YOLOE: 22–30/60 top-1, 1–6 false alarms, 0.02–0.14 s | [`../zero_shot_70/COMPARISON.md`](../zero_shot_70/COMPARISON.md) |
| 2 | One-shot registration, 28 photos (YOLOE visual prompts) | Does one example photo beat the word? | **No.** 15–27% against Grounding DINO multi's 71%; says yes to the other object of the same type about as often | [`one_shot_28_photos.md`](one_shot_28_photos.md) |
| 3 | Register-then-find, valuables (4 close-ups) | Do 4 example photos help? YOLOE and SAM 3 | **No.** YOLOE 8–10% (word 20%); SAM 3 exemplars 29–30% (word 45%, no false alarms) | [`valuables.md`](valuables.md) |
| 4 | YOLOE size follow-up | Does fixing the size gap help? | **A little.** Best 18–20%, the level of the word | [`valuables.md`](valuables.md#follow-up-does-closing-the-size-gap-help-yoloe) |
| 5 | Grounding DINO + DINOv3 check | Does comparing boxes with the registration photos help? | **Yes, against false alarms:** 16% → 0% (28 photos), 14% → 0% (valuables), item still found (71%; 25% → 30%). Look-alikes only halved (75% → 38%) | [`dino_verify.md`](dino_verify.md) |
| 6 | 70 shared images: YOLOE visual prompts as category examples | Do examples help when used as designed? | **With many examples, yes for YOLOE:** 14 examples 40/60 top-1 against the word's 30/60 (YOLOE-26x); one example 14–36/60. Still below Grounding DINO multi (54/60) | [`category_prompts/SUMMARY.md`](../zero_shot_70/category_prompts/SUMMARY.md) |
| 7 | 70 shared images: Grounding DINO + DINOv3 check with category examples | Does the check cut false alarms on the shared benchmark? | **Yes:** 24 → 8/220, all 8 on photos with visible but unlabeled sunglasses (12 → 0 on the 48 fully labeled photos); top-1 54 → 46/60 with a strict threshold | [`category_prompts/SUMMARY.md`](../zero_shot_70/category_prompts/SUMMARY.md) |

## What it adds up to

1. **For finding a specific item, words beat example photos.** YOLOE's visual prompts describe a category and
   collapse across photos; SAM 3 has no official way to take exemplars from another photo. Neither beat its own
   word prompt. As *category* examples (14 per category, 70 shared images), YOLOE visual prompts do beat YOLOE's
   word prompt (40 vs 30 of 60), but not Grounding DINO multi (54).
2. **Example photos are useful as a check.** Re-ranking Grounding DINO's boxes by DINOv3 similarity to the
   registration photos removed the false alarms and kept the item. On the 70 shared images, the same check with
   category examples removed Grounding DINO's genuine false alarms (24 → 8, all 8 on photos with unlabeled
   sunglasses) and cost 8 of 60 top-1 answers with a strict threshold. This is the most promising addition so far.
3. **"My item vs a look-alike of the same type" is not solved.** The best result, DINOv3, halves the look-alikes.
4. **Small items are the main limit.** An earbud is about 37 px in a phone photo; no method found it reliably, and
   a higher input resolution alone did not help.
5. **Speed vs accuracy.** SAM 3 with the word found the most valuables but is about 6× slower than Grounding DINO;
   YOLOE is fast but misses about half.

**For the app:** keep Grounding DINO multi; consider adding a DINOv3 check against example photos as a false-alarm
filter. Its threshold needs calibrating: strict cost 8 of 60 correct answers on the shared images, a looser one
about 3.

## Not tried yet

- DINOv3 re-ranking inside the app, with its latency measured.
- A threshold calibrated from the user's own registration photos (one fixed value did not fit both scene photos
  and close-ups).
- Patch-level DINOv3 matching for fine details (keychain, stickers), and a small "my item vs other items of the
  type" classifier trained at registration.
- Tiling the frame for small items, or guiding the user to scan closer.
- More photographers and more look-alike pairs.

## Limits

One photographer for the real photos, 20 valuables search photos: rough signals. Item boxes are semi-automatic.
Library-default thresholds, nothing tuned except where marked.

Code and how it was run: [`tests/zero_shot/README.md`](../../zero_shot/README.md).
