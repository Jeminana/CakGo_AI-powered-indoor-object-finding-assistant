# Model comparison on real phone photos (task 5 → decision for task 6)

**Result: Grounding DINO with the multi-prompt (`prompt_mode: "multi"`) is the best of the three models on real
phone photos and is the recommended final model.** Fine-tuned YOLO misses most real objects. Grounding DINO with
the original single-object prompt says "found" to almost everything.

## Setup

- **94 real phone photos** taken by the team on 2026-10-19: wallet 24, keys 20, glasses 24, remote control 12,
  and 14 photos with none of the four. By photographer: cheolyeon 28 (Galaxy S23 Ultra), najmi 36, and 30 from
  a third person without a name in the file names. 15 screenshot PNGs in the shared folder were excluded
  because they are small web images, not phone photos.
- Both runs used **identical photos**: photo-set fingerprint `01d7103be4d5ddf8…` in both summaries.
- YOLO run: `team_photos_yolo` (RTX 4050). Grounding DINO run: `team_photos_gdino` (Colab).
  Script: `tests/real_photos/score_real_photos.py`. Photos are prepared like app frames (long side 1280 px).
- Each model is asked about **all four objects in every photo**, as the app would be. Default thresholds:
  YOLO 0.25, Grounding DINO 0.35. Nothing was tuned on these photos.
- **Found** = share of photos of the object where the model said yes.
  **False alarm** = share of photos *without* the object where it still said yes.
  **Balance** = found − false alarm: 1 = perfect, 0 = no better than guessing.

## Main result

| Object | YOLO v1 (fine-tuned) found / false alarm | **Grounding DINO multi** found / false alarm | Grounding DINO single found / false alarm |
|---|---|---|---|
| Wallet | 17 % / 1 % | **50 % / 11 %** | 96 % / 50 % |
| Keys | 30 % / 1 % | **50 % / 15 %** | 100 % / 85 % |
| Glasses | 46 % / 10 % | **50 % / 3 %** | 96 % / 53 % |
| Remote control | 58 % / 1 % | **92 % / 21 %** | 100 % / 93 % |
| **Balance (mean)** | 0.34 | **0.48** | 0.28 |
| Balance per object (wallet / keys / glasses / remote) | 0.15 / 0.29 / 0.36 / 0.57 | **0.39 / 0.35 / 0.47 / 0.71** | 0.46 / 0.15 / 0.43 / 0.07 |
| `none` photos with any false alarm (of 14) | 1 | 3 | 13 |

- **Grounding DINO single** finds almost everything only because it says yes to almost everything. For keys and
  remote control it raises a false alarm on 85–93 % of photos that don't contain them. That would send a user
  in the wrong direction most of the time.
- **YOLO v1** is cautious (almost no false alarms) but misses 42–83 % of real objects.
- **Grounding DINO multi** has the best balance for keys, glasses and remote control, and the best average. For
  wallets, single mode scores slightly higher on balance (0.46 vs 0.39), but with a false alarm on every second
  photo.

## Where it works and where it doesn't

**By condition** (only photos named by the plan; `unknown` = codes outside the plan or no name):

| Condition | Photos | YOLO v1 | Grounding DINO multi | Grounding DINO single |
|---|---|---|---|---|
| Easy | 8 | 6 | 7 | 8 |
| Cluttered | 16 | 4 | **15** | 16 |
| Partly hidden | 8 | 3 | **7** | 8 |
| Dim | 8 | 3 | 4 | 7 |
| Far (2–3 m) | 8 | 1 | **2** | 7 |
| Unknown | 32 | 11 | 10 | 32 |

**By photographer** (objects found, of the photos that contain one):

| Photographer | Photos | YOLO v1 | Grounding DINO multi |
|---|---|---|---|
| cheolyeon | 24 | 8 | 18 |
| najmi | 32 | 13 | 18 |
| unnamed (sent via WeChat) | 24 | 7 | 9 |

- Grounding DINO multi handles **clutter and partly hidden objects well**. It is weak on **far** objects and
  middling in **dim** light.
- Results **vary strongly by photographer**: multi found 75 % of cheolyeon's objects but only 38 % of the unnamed
  set. Likely factors are distance and framing; the WeChat photos were also recompressed. A few people's
  photos are not a complete picture of real use.

## Effect of the threshold (Grounding DINO multi)

| Threshold | Wallet found / false alarm | Keys | Glasses | Remote control |
|---|---|---|---|---|
| 0.25 | 67 % / 17 % | 65 % / 30 % | 63 % / 11 % | 100 % / 49 % |
| **0.35 (current)** | **50 % / 11 %** | **50 % / 15 %** | **50 % / 3 %** | **92 % / 21 %** |
| 0.5 | 13 % / 3 % | 0 % / 3 % | 29 % / 0 % | 67 % / 4 % |

0.35 is a reasonable middle. Lowering it finds more but doubles the false alarms; raising it to 0.5 loses most
objects.

## Checking the boxes

There are no ground-truth boxes, so "found" means the model said yes. YOLO's 28 "found" boxes were checked by
eye: 26 were on the right object (one wallet box on a monitor, one doubtful glasses box). Grounding DINO's box
pictures (`--sheets`) should be checked the same way before the numbers are final.

## Stock test images vs real photos

On the 70 dataset test images (stock-style photos from the same source as YOLO's training data), YOLO v1 looked
as good as Grounding DINO multi: top-1 correct for wallet / keys / glasses / remote 8 / 14 / 14 / 14 of 15 vs 11 /
13 / 15 / 15. On real phone photos the gap is large. **The real-photo test, not the dataset test, should decide
the model.**

## Recommendation and limits

- **Use Grounding DINO multi** (`prompt_mode: "multi"`, threshold 0.35) as the app's detector.
  `scripts/start_demo.ps1` starts it by default.
- In the app the camera keeps sending frames while the user moves the phone, so an object missed in one frame
  can be found in the next. The voice guidance only speaks after **two frames in a row** agree (PR #12), which
  should filter many one-frame false alarms. Neither effect is measured in this single-photo test.
- Weak spots to state honestly: **far** objects and **dim** light, about **half of wallets, keys and glasses
  missed in a single frame**, and **remote-control false alarms** on button-like objects (21 %).
- It needs a GPU for real-time use: about 0.3–0.5 s per request on an RTX 4050, about 15 s on a CPU.
