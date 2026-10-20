# Model result analysis: final

Prepared by **나즈미 / Jeminana**, 20 October 2026. This replaces the 17 October analysis
([`../2026-10-17/RESULT_ANALYSIS.md`](../2026-10-17/RESULT_ANALYSIS.md)), which was written before any real-phone
test. All values come from committed experiment files; sources are listed at the end.

## Decision

**CakGo uses Grounding DINO tiny with the multi-prompt** (`prompt_mode: "multi"`, threshold 0.35). The team
decided this on 20 October from the real-phone-photo test.

The short version of how we got there:

1. On the 70 stock-style dataset images, **fine-tuned YOLO looked best**: few false alarms, 15 ms per image.
2. On a **real iPhone**, fine-tuned YOLO found almost nothing. Grounding DINO found the objects, but also called a
   mouse, a laptop keyboard and a router "remote control".
3. A **multi-prompt** (asking for all objects plus look-alikes at once) cut Grounding DINO's false alarms about
   tenfold on validation images.
4. On **94 real phone photos taken by the team**, Grounding DINO multi had the best balance of found objects and
   false alarms.

**Lesson: the dataset test did not predict real use.** Only the real-photo test should decide the model.

## 1. Dataset benchmark (70 stock-style images)

The shared evaluation set (`tests/dataset/`) has 70 images from *Objects In House* v5: 15 per object and 10
without any target. These are mostly stock-style product photos, stretched to 640×640.

| Model | What it can find | Result on the 70 images |
|---|---|---|
| Pretrained YOLO26s (COCO) | Remote control only | Remote: AP50 0.870, 0 false alarms on the 10 negatives |
| Grounding DINO tiny, single prompt | All four via text | Remote: AP50 0.941, but false alarms on 4–9 of 10 negatives |
| **Fine-tuned YOLO26s v1** (6,514 training images) | All four | Mean AP50 0.943, 0 false alarms on the 10 negatives, about 15 ms per image |

Fine-tuned YOLO looked like the clear winner. Two warning signs were already visible:

- **Wallet overfitting:** v1 found 100 % of wallets in its training images but only 63 % in validation images
  (other objects 94–99 %). It was not about box size, threshold or duplicate labels: about a quarter of validation
  wallets got no wallet box even at confidence 0.01.
- **v2** (120 epochs + rotation, mixup, brightness and zoom variation) improved validation (mAP50 0.917 → 0.932,
  wallet recall 0.63 → 0.69) but not the 70 test images (wallet 8/15 → 7/15). v1 stayed the default.

Training and test images come from the same source collection, so these scores were likely optimistic.

## 2. Phone test 1 (19 October, iPhone 15 Pro)

The full chain worked: HTTPS link → camera → backend → boxes → Korean voice. On 4 real desk photos (cluttered,
one dim):

| Model | Targets found (of 4) | False alarms (12 questions about absent objects) |
|---|---|---|
| Fine-tuned YOLO v1 | 1 | 0 |
| Fine-tuned YOLO v2 | 0 | 1 |
| Grounding DINO, single prompt | 4 | 9 |

Live on the phone, Grounding DINO boxed a **computer mouse, a laptop keyboard and a white router** as "remote
control". The user's remote was an **air-conditioner remote**, a label that had been left out of YOLO's training
data. Testers also reported that the **voice spoke too often** and the **text was too small**. Both were fixed
afterwards (PR #12, PR #15).

Likely reasons YOLO failed on real frames: the training images are stretched stock-style close-ups, real rooms
have clutter, distance and indoor light, and AC remotes were never trained.

## 3. Fix: multi-prompt for Grounding DINO (PR #10)

With a one-object prompt ("keys."), Grounding DINO gives the word to the most similar object in the picture. The
multi-prompt lists **all four targets plus look-alikes** (computer mouse, laptop, keyboard, pen, bottle, can, cup,
microphone, cable, charger, nail clipper, coin). Each box goes to the phrase it matches best and counts only if
that phrase is the requested object. Speed is unchanged: one model pass per request.

The look-alike list was **tuned on validation images, not on the test images**. Words that describe a target
itself (card, bag, book, box, paper, mobile phone) took boxes away from real wallets, so they were removed.

| Validation images, threshold 0.35 | Top answer correct: single → multi | False alarms on 100 images without the object: single → multi |
|---|---|---|
| Wallet | 52/60 → 36/60 | 53 → 5 |
| Keys | 42/42 → 41/42 | 94 → 5 |
| Glasses | 58/60 → 58/60 | 69 → 12 |
| Remote control | 58/60 → 58/60 | 82 → 16 |

On the 70 test images (checked once, after tuning), precision rose from 0.20–0.27 to 0.63–0.84 and keys AP50
from 0.33 to 0.70. False alarms on the 10 negatives dropped from 4/9/6/9 to 1/0/0/1. The trade-off is lower
wallet recall.

## 4. Real-photo test (94 team photos)

- **Photos:** taken by the team on 19 October. Wallet 24, keys 20, glasses 24, remote control 12, and 14 without
  any target. Three photographers: 김철연 28, 나즈미 36, and 30 without names in the file names.
- **Excluded:** 15 PNG screenshots of web images, because they are not phone photos.
- **Same photos for every model:** both runs show the photo-set fingerprint `01d7103be4d5ddf8…`.
- **Method:** every model is asked about all four objects in every photo, at its default threshold. Nothing was
  tuned on these photos. The photos stay private and are not in the repository.
- **Definitions:** *Found* = share of photos of the object where the model says yes. *False alarm* = share of
  photos without it where it still says yes. *Balance* = found − false alarm (1 is perfect, 0 is guessing).

| Object | Fine-tuned YOLO v1 | **Grounding DINO multi** | Grounding DINO single |
|---|---|---|---|
| Wallet (found / false alarm) | 17 % / 1 % | **50 % / 11 %** | 96 % / 50 % |
| Keys | 30 % / 1 % | **50 % / 15 %** | 100 % / 85 % |
| Glasses | 46 % / 10 % | **50 % / 3 %** | 96 % / 53 % |
| Remote control | 58 % / 1 % | **92 % / 21 %** | 100 % / 93 % |
| **Balance, mean** | 0.34 | **0.48** | 0.28 |
| Photos without any target that got a false alarm | 1 of 14 | 3 of 14 | 13 of 14 |

- **Single prompt** says yes to almost everything (85–93 % false alarms for keys and remote). That would guide
  users to the wrong object.
- **Fine-tuned YOLO** is cautious but misses 42–83 % of real objects.
- **Multi prompt** has the best balance overall and for keys, glasses and remote control.

**Conditions** (photos named by the plan): multi found 15/16 in clutter, 7/8 partly hidden and 7/8 easy, but only
4/8 in dim light and **2/8 far away** (2–3 m). **Photographer** mattered a lot: multi found 75 % of 김철연's
objects but 38 % of the unnamed set.

YOLO's 28 "found" boxes were checked by eye: 26 are on the right object. Grounding DINO's boxes still need the
same check. Full tables, including the threshold sweep:
[`tests/results/real_photos/COMPARISON.md`](../../tests/results/real_photos/COMPARISON.md).

## 5. Speed

| Measurement | Hardware | Time |
|---|---|---|
| Grounding DINO multi, one request (model only) | RTX 4050 laptop | 321 ms mean (70 images) |
| Grounding DINO multi through the HTTPS link, one test request | RTX 4050 laptop + Cloudflare tunnel | about 0.45 s |
| Fine-tuned YOLO, one request (model only) | RTX 4050 laptop | 15 ms (not chosen) |
| Phone camera → spoken direction | iPhone + laptop | **[after phone test 2]** |

The frontend sends a frame, waits for the answer, then pauses 1 s. Since PR #12 the voice only speaks after **two
frames in a row** agree, which adds roughly one more frame cycle before speaking. Without an NVIDIA GPU, Grounding
DINO needs about 15 s per request, which is too slow for live use.

## 6. Limitations

- **Single frames:** the photo test scores one photo at a time. In the app the camera keeps sending frames and the
  voice waits for two matching frames. That should find more objects and filter one-frame false alarms, but it
  has not been measured.
- **Weak spots:** far objects (2/8), dim light (4/8), about half of wallets, keys and glasses missed in a single
  frame, and remote-control false alarms on button-like objects (21 %).
- **Small, uneven test set:** 94 photos from 3 people, only 12 remote-control photos, and strong differences
  between photographers.
- **Box check:** "found" means the model said yes. Only YOLO's boxes were verified by eye so far.
- **No user study:** benefit for people with visual difficulties has not been measured.
- **Hardware:** live use needs a laptop or server with an NVIDIA GPU.

## Sources

- Dataset benchmark: [`tests/results/model_comparison_report.md`](../../tests/results/model_comparison_report.md),
  fine-tuned YOLO: [`detection/experiments/yolo/README.md`](../../detection/experiments/yolo/README.md) (Results so far)
- Multi-prompt method and validation tuning:
  [`detection/experiments/grounding_dino/MULTI_PROMPT.md`](../../detection/experiments/grounding_dino/MULTI_PROMPT.md),
  run `detection/experiments/grounding_dino/results/rtx4050_gdino-tiny_multi_box0.05/`
- Real-photo test: [`tests/results/real_photos/COMPARISON.md`](../../tests/results/real_photos/COMPARISON.md), runs
  `team_photos_yolo` and `team_photos_gdino`
- Phone test 1: PR #10 and PR #11 descriptions; team messages from 19 October
- Voice and font fixes: PR #12, PR #15. Demo start: [`scripts/README.md`](../../scripts/README.md)
