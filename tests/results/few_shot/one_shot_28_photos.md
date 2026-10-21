# One-shot "register my item" test on Member 4's 28 photos (2026-10-20)

**Question:** a user registers an object with one example photo (a YOLOE visual prompt). Does YOLOE then find that
object better than the app's Grounding DINO multi, which only gets the object word?

**Answer: no.**

- **It finds less than either text model.** YOLOE one-shot puts its top box on the registered object in 15–27% of
  searches. YOLOE with the object word does better (38–58%), and Grounding DINO multi does best (71%).
- **It does not tell "my wallet" from another wallet.** It says yes to the other object of the same type
  (18–26%) about as often as to the registered one.
- **Bad registration photos almost never work.** With YOLOE-26x, registering from the far photo or the dim photo
  succeeded in 0 of 8 searches each.
- **Its only strength is few false alarms (2–4%).**

## Setup

| Item | Value |
|---|---|
| Photos | 4 object types (wallet, keys, glasses, remote control) × designs A and B, which are different physical objects; 3 photos each (A1 easy, A2 cluttered, A3 partly hidden, B1 far, B2 dim, B3 cluttered), plus 4 photos without any of them. Processed like the app: EXIF rotation, long side 1280 px |
| One-shot | For each of the 8 objects and each of its 3 photos (leave-one-out): YOLOE gets that photo and the object's box (Ultralytics `refer_image`) and is asked about the other 27 photos → 24 registrations |
| Baselines | Grounding DINO tiny multi (app, 0.35) and YOLOE-26s/26x with the object word (0.25), on the same photos |
| "Box on it" | Top box centre inside the object's box |
| Thresholds | Library defaults, fixed before the run: YOLOE 0.25, Grounding DINO 0.35 |
| Where | Member 4's Mac (Apple MPS); ultralytics 8.4.177, transformers 5.20.1 |

## Result (default thresholds)

| Model | Same object: found | Same object: box on it | Same type, other object: said yes | No such object: false alarms |
|---|---|---|---|---|
| YOLOE-26s one-shot | 17% (8/48) | 15% | 18% (13/72) | **2%** (13/528) |
| YOLOE-26x one-shot | 29% (14/48) | 27% | 26% (19/72) | 4% (19/528) |
| **Grounding DINO multi (app)** | **75%** (36/48) | **71%** | 75% (54/72) | 14% (72/528) |
| YOLOE-26s text | 38% (18/48) | 38% | 38% (27/72) | **0%** (0/528) |
| YOLOE-26x text | 62% (30/48) | 58% | 62% (45/72) | 7% (36/528) |

"Same object" = the other two photos of the registered object. "Same type, other object" = the 3 photos of the
other design of that type. Text models cannot tell objects of one type apart, so for them that column is just how
often they find the type.

**By the photo used for registration** (YOLOE-26x one-shot, box on the same object, of 8 searches):

| A1 easy | A2 cluttered | A3 partly hidden | B1 far | B2 dim | B3 cluttered |
|---|---|---|---|---|---|
| 3 | 3 | 4 | 0 | 0 | 3 |

**Threshold sweep** (box on the same object / other object of the same type / false alarms):

| Model | 0.1 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| YOLOE-26x one-shot | 38% / 35% / 7% | 27% / 26% / 4% | 25% / 17% / 2% | 15% / 11% / 2% |
| Grounding DINO multi | 79% / 96% / 68% | 71% / 79% / 30% | 71% / 75% / 14% | 50% / 50% / 2% |
| YOLOE-26x text | 75% / 79% / 16% | 58% / 62% / 7% | 54% / 58% / 3% | 46% / 46% / 0% |

## Why one-shot does poorly

- **It is not a usage error.** The code follows the documented Ultralytics path (`predict(..., refer_image=...,
  visual_prompts=...)`, then plain `predict`), as the authors' demo does. The docs' own cross-image example works
  in the same environment.
- **Visual prompts are meant for categories, not individual objects.** The YOLOE paper says visual prompts
  indicate the object category, and its evaluation averages 16 examples per category. The embeddings agree: two
  photos of the same object are hardly more similar than photos of two different objects of the same type (mean
  cosine 0.51 vs 0.48; other types 0.29).
- **Within one photo it works; across photos the score collapses.** Prompting and searching the same photo puts
  the top box on the object in 22 of 24 photos. Across photos the object is still the highest-scoring region in
  35 of 48 searches, but its median score is 0.04, and only 13 of 48 reach the 0.25 default.
- **Lowering the threshold does not close the gap.** At 0.05, one-shot finds 44%, says yes to the other object of
  the type in 49%, and raises 12% false alarms. YOLOE-26x with the word is already at 58% / 62% / 7% at 0.25.

## Takeaways

- **One-shot visual prompting with YOLOE is not usable for "find my registered item".**
- **Keep Grounding DINO multi.**
- **YOLOE-26x with the word is a lower-false-alarm option:** 58% found with 7% false alarms at 0.25.

## Limits

8 objects, one photographer, 3 photos per object: a rough signal. The object boxes are semi-automatic.
Per-photo answers are kept locally with the photos.

Code: [`tests/zero_shot/instance_one_shot.py`](../../zero_shot/instance_one_shot.py),
[`diagnose_visual_prompt.py`](../../zero_shot/diagnose_visual_prompt.py).
