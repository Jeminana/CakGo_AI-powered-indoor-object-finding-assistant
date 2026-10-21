# Zero-shot comparison on the 70 shared images: YOLOE-26 and SAM 3 vs Grounding DINO

Member 4, 2026-10-20. Screening test, not a model decision: the team's real-photo test (94 phone photos)
showed that these stock-style images can be misleading, so the real photos decide.

**Result: on these images no candidate beats the app's Grounding DINO multi-prompt on both accuracy and
speed.**

- **SAM 3** finds the objects at least as well, with the same number of false alarms. It is better on wallets
  and remote controls. Its keys boxes cover single keys instead of the labeled bunch, so they fail the IoU
  rule. The cost is speed: about **6× slower per request** on the same T4 (2.24 s vs 0.38 s), with 3× the
  GPU memory.
- **YOLOE-26** is fast (22 ms per request for 26s) and almost never raises a false alarm (1–6 of 220), but it
  **misses about half of the objects** (top-1 correct 22–30 of 60). Wallets are its weakest object (3–4 of 15).
- Adding the look-alike words (multi mode) does not help either new model: SAM 3 is unchanged at 3× the cost,
  and YOLOE loses boxes to the look-alikes.

## Setup (identical for every model)

| Item | Value |
|---|---|
| Images | `tests/dataset/manifest.csv` (sha256 `ca73e2c1…5bde4b`): 70 Objects In House images, 15 per object + 10 without any; checked with `prepare_dataset.py verify` on the VM |
| Zero-shot | Text prompts only: no fine-tuning, no example images |
| Words | Grounding DINO's `config/prompts.json`; multi mode adds the 12 words of `config/look_alikes.json`, and a box counts for the target only if a target word matches it best |
| Queries | One request per image per object (280 per run), each timed; 3 untimed warm-up requests per object |
| Saved boxes | Confidence >= 0.05 (for AP50 and the threshold sweep) |
| Operating thresholds | Fixed before the run, each library's default: Grounding DINO 0.35 (app config), YOLOE 0.25 (Ultralytics default), SAM 3 0.5 (transformers examples) |
| Hardware | One Colab VM via the Colab CLI (team account): Tesla T4, fp32, one process per run, all 7 runs back to back (08:51–09:39 UTC) |
| Software | Python 3.13.15, torch 2.11.0+cu130, transformers 5.20.1, ultralytics 8.4.177 |
| Models | `IDEA-Research/grounding-dino-tiny` (revision `a2bb814d`); `yoloe-26s-seg.pt` (sha256 `48f24206…`) and `yoloe-26x-seg.pt` (`d08d390a…`); `facebook/sam3` (revision `3c879f39`) |
| Scoring | Day 3 code, imported: `grounding_dino/evaluate.py` `score_category` (IoU >= 0.5, greedy), `tests/compare_models.py` `top1_rate` and `GT_ISSUES` |

Code and run instructions: [`tests/zero_shot/`](../../zero_shot/README.md). All tables: [`tables.md`](tables.md).

## Main result (each model at its operating threshold, all 70 images)

| Model | Top-1 correct (of 60) | Top box on the object* (of 60) | False alarms (of 220) | Target-free images with a box (of 40) | Mean AP50 | Median per request (T4) | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| **Grounding DINO tiny multi (app)** | **54** | 55 | 24 | 2 | **0.729** | 383 ms | 1.5 GB |
| YOLOE-26s single | 27 | 28 | **1** | **0** | 0.524 | **22 ms** | 0.3 GB |
| YOLOE-26s multi | 22 | 23 | **1** | **0** | 0.435 | 23 ms | 0.2 GB |
| YOLOE-26x single | 30 | 32 | 6 | 1 | 0.567 | 137 ms | 1.4 GB |
| YOLOE-26x multi | 29 | 31 | 6 | 1 | 0.517 | 138 ms | 0.5 GB |
| SAM 3 single | 44 | **58** | 24 | 2 | 0.648 | 2,240 ms | 4.6 GB |
| SAM 3 multi | 44 | **58** | 23 | 2 | 0.648 | 6,526 ms | 4.6 GB |
| *Grounding DINO tiny single (Day 3 run, T4)* | *55* | *57* | *176* | *28* | *0.623* | *331 ms* | *1.5 GB* |

\* Diagnostic, not part of the protocol: the centre of the highest-confidence box lies inside a labeled box of
that object. The app's left/centre/right guidance uses the box centre.

**Top-1 correct per object** (of 15; in brackets: AP50):

| Object | Grounding DINO multi | YOLOE-26s single | YOLOE-26x single | SAM 3 single |
|---|---|---|---|---|
| Wallet | 11 (0.58) | 4 (0.24) | 3 (0.26) | **12 (0.80)** |
| Keys | **13 (0.71)** | 5 (0.53) | 7 (0.62) | 2 (0.08); top box on the bunch: 15 |
| Glasses | 15 (0.69) | 6 (0.52) | 8 (0.60) | 15 (0.71) |
| Remote control | 15 (0.94) | 12 (0.80) | 12 (0.80) | **15 (1.00)** |

**Common thresholds**, top-1 correct / false alarms (full sweep in `tables.md`):

| Model | 0.1 | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| Grounding DINO tiny multi | 55 / 100 | 54 / 38 | 54 / 24 | 35 / 8 |
| YOLOE-26s single | 32 / 15 | 27 / 1 | 22 / 1 | 18 / 0 |
| YOLOE-26x single | 35 / 17 | 30 / 6 | 26 / 3 | 21 / 2 |
| SAM 3 single | 44 / 42 | 44 / 29 | 44 / 27 | 44 / 24 |

Without the 12 images that have unlabeled targets: Grounding DINO multi 43/48 top-1 and 12/184 false alarms,
SAM 3 single 42/48 and 14/184, YOLOE-26x single 25/48 and 3/184.

## What is behind the numbers

- **SAM 3 and keys.** SAM 3 boxes every key of a bunch (4–11 boxes per image), while the dataset has one box
  per bunch. On all 15 key images the top box lies fully inside a labeled bunch, but its IoU is 0.11–0.47, so
  the official rule counts it wrong. Grounding DINO boxes the whole bunch: its top box reaches IoU 0.61–0.98 on
  13 of the 14 key images where it answers.
- **Where the false alarms are.** Grounding DINO multi (24): remote control 10, glasses 8, wallet 5, keys 1.
  SAM 3 single (24): 11 are "keys" on remote-control images (buttons), and 8 are "glasses" on key images, mostly
  the sunglasses that the labels miss. 12 (Grounding DINO) and 10 (SAM 3) of the 24 are on the 12 images with
  unlabeled targets.
- **Multi mode does not catch SAM 3's buttons.** A button box is much smaller than the remote box (IoU < 0.7),
  so the best-word rule treats them as different objects (23 vs 24 false alarms), at 6.5 s instead of 2.2 s per
  request.
- **YOLOE is confident only when it is sure.** Even at 0.1, YOLOE-26x finds the top box on 35 of 60. Look-alike
  words take boxes away from targets: YOLOE-26s finds 12 of 15 remote controls in single mode and 7 in multi.
- **SAM 3 is threshold-insensitive.** Top-1 stays at 44 from 0.1 to 0.5, so its 0.5 default is not the reason
  for its numbers.

## Checks

- The Grounding DINO multi re-run on the T4 reproduces the RTX 4050 run (`rtx4050_gdino-tiny_multi_box0.05`):
  same TP / FP / FN for every object at 0.35. The keys AP50 is 0.705 vs 0.695, a difference in low-score boxes.
- A first pass of Grounding DINO and the four YOLOE runs gave bit-identical detections on all 70 images, and
  median latency within 3%. A file upload overlapped three of those first-pass runs, so the reported numbers
  come from the second, undisturbed pass.
- Latency includes pre- and post-processing of one `detect()` call. The T4 slows slightly as it warms up:
  within one run, Grounding DINO goes from 377 to 387 ms and SAM 3 single from 2.22 to 2.32 s (medians per
  10 images). fp16, and reusing one image encoding for the four questions of a frame, would speed SAM 3 up.
  Neither was tested.

## Limits

- 15 images per object: differences of one or two images are noise.
- The look-alike words were tuned on Grounding DINO validation images, which gives Grounding DINO multi a small
  head start.
- SAM 3 multi decides "same object" at IoU >= 0.7. Grounding DINO decides it within one pass, so SAM 3 multi is
  an approximation of the same rule.
- Stock-style images from the same source as the earlier tests, not phone photos.

## Suggested next step (team decision)

Keep Grounding DINO multi for the MVP. If a model is tested on the 94 real photos next, SAM 3 single is the
candidate: equal or better accuracy here, but 2.2 s per request on a T4 and "keys" on remote buttons. Its
adapter already has the app's `detect(image, target)` interface, so it can be plugged into
`tests/real_photos/score_real_photos.py`. YOLOE-26 zero-shot misses too many wallets, keys and glasses to
replace Grounding DINO. Fine-tuned on the Objects In House train split, it might do better; that would be a
separate test.

## How SAM 3 reached the VM

`facebook/sam3` is gated; the Member 4 Hugging Face account has access. The files were downloaded on the Mac with
the existing login and copied into the VM's Hugging Face cache. A single 3.4 GB upload is rejected (limit
between 64 and 256 MB per file), so the file went as 52 parts of 64 MB, was joined on the VM and matched its
sha256 (`6d06f0a5…`). SAM 3 then ran with `HF_HUB_OFFLINE=1`: no token was on the VM. The whole Colab session
used 1.56 compute units.

## Files

`t4_gdino-tiny_multi/`, `t4_yoloe-26s_single/`, `t4_yoloe-26s_multi/`, `t4_yoloe-26x_single/`, `t4_yoloe-26x_multi/`,
`t4_sam3_single/`, `t4_sam3_multi/`: `predictions.jsonl`, `run_info.json`, `metrics.json`, `metrics.md` each.

Reproduce the tables (from the repository root):

```bash
python tests/zero_shot/compare_zero_shot.py --runs t4_gdino-tiny_multi t4_yoloe-26s_single t4_yoloe-26s_multi \
    t4_yoloe-26x_single t4_yoloe-26x_multi t4_sam3_single t4_sam3_multi \
    --reference "Grounding DINO tiny single (Day 3, T4)=detection/experiments/grounding_dino/results/t4_gdino-tiny_box0.05" \
    --reproduce t4_gdino-tiny_multi=detection/experiments/grounding_dino/results/rtx4050_gdino-tiny_multi_box0.05
```
