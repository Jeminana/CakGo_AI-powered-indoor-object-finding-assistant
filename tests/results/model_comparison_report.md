# Model Comparison Report — YOLO vs Grounding DINO (Day 3)

Input for the Day 4 model decision; the decision itself belongs to the team.
Prepared by Member 4 on 2026-10-16.

> **About the YOLO numbers:** the repository had no YOLO results on 2026-10-16. PR #1 shipped the YOLO
> scripts, but the evaluation subset had not been built yet. So Member 4 ran Member 1's YOLO scripts
> **unmodified** on the same Colab VM and the same 70 images as Grounding DINO. These are reference numbers,
> not Member 1's official results. Their run folders are in `tests/results/yolo_reference_runs/`, so nothing
> was added to `detection/experiments/yolo/`. Member 1 can confirm them or replace them with an own run on the
> same `manifest.csv`.

## Summary

- **Coverage.** The COCO-pretrained YOLO26s has a class for only one of the four targets (remote control).
  Wallet, keys and glasses need fine-tuning before YOLO can find them at all. Grounding DINO tiny handles all
  four zero-shot through text prompts.
- **When the object is in the frame,** Grounding DINO's highest-confidence box is correct for 15/15 remote
  controls, 15/15 glasses, 13/15 wallets and 12/15 keys. YOLO gets 13/15 remote controls.
- **When the object is not in the frame,** Grounding DINO still reports it in many of the 10 target-free
  images: 4–9 of them at the provisional threshold 0.35, 2–7 at 0.5. YOLO reports a remote control in none.
  For a user who cannot see the screen, these false alarms are Grounding DINO's main risk.
- **Remote control, the only shared category:** AP50 0.941 (Grounding DINO) vs 0.870 (YOLO). YOLO has
  precision 1.0 at recall 0.83; Grounding DINO has recall 1.0 at precision 0.20 (threshold 0.35) or 0.40 (0.5).
- **Cost per request:** 334 ms vs 15 ms on a T4 GPU (about 22×), 15.4 s vs 0.27 s on the Colab CPU, and
  1.48 GB vs 0.13 GB of peak GPU memory.

## Recommendation (Member 4's input for Day 4)

1. **Use Grounding DINO tiny for the MVP**, because it is the only tested option that can find wallets, keys
   and glasses without training. It must run on a GPU server; about 0.33 s per request fits the API contract's
   periodic frame uploads, while 15 s on a CPU does not.
2. **Treat false alarms as the main Day 4–5 work item.** Raise the threshold above the provisional 0.35
   (0.5 reduces false alarms to 2/10 for wallet, glasses and remote control, but keys stay at 7/10), and choose
   it on separate images rather than on this test set. Candidate mitigations that were **not** tested here:
   prompting all categories at once so a box can go to a better-fitting label, and only speaking when the same
   detection appears in consecutive frames.
3. **Keep fine-tuning YOLO as the fallback.** The Objects In House train split (6,912 images) was left untouched
   for this. YOLO's zero false alarms on remote controls suggest a closed-set detector trained on our four
   classes would fix Grounding DINO's weakness at a fraction of the latency, but it needs training time and a
   new evaluation.

## Setup (identical for both models)

| Item | Value |
|---|---|
| Dataset | `tests/dataset/manifest.csv` (sha256 `ca73e2c1…5bde4b`): 70 images, Objects In House v5 test+valid, seed `cakgo-eval-v1`; 15 each of wallet, keys, glasses, remote control, plus 10 target-free images; all 640×640 |
| Hardware | One Colab VM: Tesla T4, fp32 for both models; CPU runs on the same VM's 2 vCPUs (Intel Xeon @ 2.00 GHz) |
| Software | Python 3.13.15, torch 2.11.0+cu130; ultralytics 8.4.177; transformers 5.18.0 |
| YOLO | Member 1's scripts from PR #1, unmodified, run by Member 4. `yolo26s.pt` (sha256 `646f8bc3…db384a1b`), imgsz 640, NMS IoU 0.7. Accuracy from run `t4_yolo26s_conf0.001` (untruncated AP), latency from `t4_yolo26s_conf0.25` (default settings); run folders in `tests/results/yolo_reference_runs/` |
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` (revision `a2bb814d`, read from the VM cache after the run), text threshold 0.25, boxes saved from 0.05. One query per image per category (280 queries), because the app asks for one target at a time. Prompts, chosen before the run: `wallet.`, `keys.`, `glasses. sunglasses.`, `remote control.` |
| Scoring | Same code for both: per category, greedy matching at IoU ≥ 0.5 across all 70 images, all-point AP50. `tests/compare_models.py` reproduces YOLO's own `metrics.json` exactly, and `tests/test_gdino_evaluate.py` checks the Grounding DINO evaluator against YOLO's |
| Operating thresholds | YOLO 0.25 (Member 1's default), Grounding DINO 0.35 (provisional config). Sweeps below show both models across thresholds |

**Top-1 correct** is the app's view: on each target image, is the single highest-confidence box (at or above the
threshold) a correct box? **Negatives with FP** counts target-free images where the model returned any box for
that category, i.e. a false alarm in the app.

## Results

### Accuracy, all 70 images (official protocol)

| Category | Model | Conf | AP50 | Precision | Recall | Target images detected | Top-1 correct | Negatives with FP |
|---|---|---|---|---|---|---|---|---|
| wallet | YOLO26s | unsupported | - | - | - | - | - | - |
| keys | YOLO26s | unsupported | - | - | - | - | - | - |
| glasses | YOLO26s | unsupported | - | - | - | - | - | - |
| remote_control | YOLO26s | 0.25 | 0.870 | 1.000 | 0.833 | 13/15 | 13/15 | 0/10 |
| wallet | Grounding DINO tiny | 0.35 | 0.521 | 0.273 | 0.882 | 13/15 | 13/15 | 4/10 |
| keys | Grounding DINO tiny | 0.35 | 0.333 | 0.229 | 1.000 | 15/15 | 12/15 | 9/10 |
| glasses | Grounding DINO tiny | 0.35 | 0.696 | 0.258 | 0.944 | 15/15 | 15/15 | 6/10 |
| remote_control | Grounding DINO tiny | 0.35 | 0.941 | 0.198 | 1.000 | 15/15 | 15/15 | 9/10 |

### Sensitivity check: excluding 12 images with unlabeled targets

An AI pre-screen of the images (not a human review) found 12 images where a target is visible but has no box
(listed in `tests/dataset/dataset_info.md`, section 6). Correct detections of those objects count as false
positives above. This table drops those images; it is a check, not the official result.

| Category | Model | Conf | AP50 | Precision | Recall | Target images detected | Top-1 correct | Negatives with FP |
|---|---|---|---|---|---|---|---|---|
| remote_control | YOLO26s | 0.25 | 0.870 | 1.000 | 0.833 | 13/15 | 13/15 | 0/10 |
| wallet | Grounding DINO tiny | 0.35 | 0.474 | 0.255 | 0.857 | 10/12 | 10/12 | 4/10 |
| keys | Grounding DINO tiny | 0.35 | 0.164 | 0.109 | 1.000 | 6/6 | 4/6 | 9/10 |
| glasses | Grounding DINO tiny | 0.35 | 0.838 | 0.327 | 0.944 | 15/15 | 15/15 | 6/10 |
| remote_control | Grounding DINO tiny | 0.35 | 0.948 | 0.273 | 1.000 | 15/15 | 15/15 | 9/10 |

The missing labels mostly hurt Grounding DINO's glasses score (AP50 0.696 → 0.838). Keys get worse instead,
because the 9 excluded keys images were ones it handled well. YOLO's remote control images are unaffected.

### Confidence sweep: precision / recall / negatives with FP

| Model | Category | 0.1 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| YOLO26s | remote_control | 0.938 / 0.833 / 0 | 1.000 / 0.833 / 0 | 1.000 / 0.833 / 0 | 1.000 / 0.833 / 0 | 1.000 / 0.833 / 0 | 1.000 / 0.833 / 0 | 1.000 / 0.778 / 0 |
| Grounding DINO tiny | wallet | 0.064 / 0.941 / 10 | 0.144 / 0.882 / 8 | 0.197 / 0.882 / 5 | 0.224 / 0.882 / 5 | 0.273 / 0.882 / 4 | 0.312 / 0.882 / 4 | 0.355 / 0.647 / 2 |
| Grounding DINO tiny | keys | 0.042 / 1.000 / 10 | 0.113 / 1.000 / 10 | 0.150 / 1.000 / 10 | 0.188 / 1.000 / 9 | 0.229 / 1.000 / 9 | 0.257 / 0.906 / 7 | 0.281 / 0.781 / 7 |
| Grounding DINO tiny | glasses | 0.038 / 1.000 / 10 | 0.126 / 1.000 / 10 | 0.170 / 1.000 / 9 | 0.214 / 1.000 / 8 | 0.258 / 0.944 / 6 | 0.304 / 0.944 / 5 | 0.357 / 0.833 / 2 |
| Grounding DINO tiny | remote_control | 0.033 / 1.000 / 10 | 0.081 / 1.000 / 10 | 0.111 / 1.000 / 10 | 0.150 / 1.000 / 9 | 0.198 / 1.000 / 9 | 0.250 / 1.000 / 6 | 0.400 / 1.000 / 2 |

### App view sweep: top-1 correct on target images / negatives with any box

| Model | Category | 0.1 | 0.2 | 0.25 | 0.3 | 0.35 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| YOLO26s | remote_control | 13/15 / 0 | 13/15 / 0 | 13/15 / 0 | 13/15 / 0 | 13/15 / 0 | 13/15 / 0 | 12/15 / 0 |
| Grounding DINO tiny | wallet | 13/15 / 10 | 13/15 / 8 | 13/15 / 5 | 13/15 / 5 | 13/15 / 4 | 13/15 / 4 | 11/15 / 2 |
| Grounding DINO tiny | keys | 12/15 / 10 | 12/15 / 10 | 12/15 / 10 | 12/15 / 9 | 12/15 / 9 | 12/15 / 7 | 11/15 / 7 |
| Grounding DINO tiny | glasses | 15/15 / 10 | 15/15 / 10 | 15/15 / 9 | 15/15 / 8 | 15/15 / 6 | 15/15 / 5 | 14/15 / 2 |
| Grounding DINO tiny | remote_control | 15/15 / 10 | 15/15 / 10 | 15/15 / 10 | 15/15 / 9 | 15/15 / 9 | 15/15 / 6 | 15/15 / 2 |

### Where Grounding DINO's false boxes land (conf ≥ 0.35)

| Category | False boxes | Near a GT box | Near full frame | On another labeled object | Elsewhere | Top score on target images (min / median) | Top score on negatives (max / median) |
|---|---|---|---|---|---|---|---|
| wallet | 40 | 5 | 1 | 24 | 10 | 0.44 / 0.68 | 0.72 / 0.35 |
| keys | 108 | 8 | 3 | 53 | 44 | 0.48 / 0.70 | 0.77 / 0.52 |
| glasses | 49 | 0 | 1 | 29 | 19 | 0.43 / 0.72 | 0.52 / 0.41 |
| remote_control | 73 | 0 | 1 | 46 | 26 | 0.65 / 0.83 | 0.59 / 0.42 |

About half (49–63 %) of the false boxes sit on another labeled object (a bag, shoe, earphone or a different
target): with a single-category prompt the model tends to give the requested label to whatever object is there.
Few are duplicates of a true box (0–8 per category) or whole-frame boxes (1–3). Only remote control separates
cleanly by score in this sample: the lowest top score on remote control images is 0.65 and the highest on
target-free images is 0.59. For keys and wallets, target-free images score as high as 0.77 and 0.72.

### Latency per request and memory

| Model | Hardware | Measured | mean (ms) | median | p95 | min | max |
|---|---|---|---|---|---|---|---|
| YOLO26s | T4 GPU | 70 requests (`t4_yolo26s_conf0.25`) | 15.25 | 13.17 | 25.91 | 12.38 | 58.67 |
| Grounding DINO tiny | T4 GPU | 280 requests (`t4_gdino-tiny_box0.05`) | 333.88 | 331.48 | 363.78 | 307.75 | 370.06 |
| YOLO26s | Colab CPU | 10 requests (`colabcpu_yolo26s_latency-sample`) | 270.93 | 268.75 | 299.02 | 256.79 | 299.02 |
| Grounding DINO tiny | Colab CPU | 10 requests (`colabcpu_gdino-tiny_latency-sample`) | 15422.05 | 15423.90 | 15538.04 | 15313.74 | 15538.04 |

A request is one `predict()` call for YOLO (all classes at once) and one single-category query for Grounding
DINO; image file reading is excluded and warm-up passes are not timed. Peak GPU memory (PyTorch allocator):
Grounding DINO 1,480.8 MB over the whole run (`run_info.json`); YOLO 128.1 MB, measured separately with five
`predict()` calls on the same VM because the YOLO script does not record it.

## Limitations

- **YOLO run by Member 4:** the YOLO numbers are a reference run of Member 1's unmodified scripts, not Member 1's
  own results. If Member 1's run on the same manifest differs, use Member 1's run and re-run the comparison.
- **Small sample:** 15 images per category and 10 target-free images, so the numbers are indicative only.
- **Not phone frames:** the source images are stretched to 640×640 and many are stock photos with watermarks.
- **Unreviewed ground truth:** labels are community annotations, and `review_status` is still `unreviewed`.
  The 12 images with unlabeled targets come from an AI pre-screen and still need a human check.
- **Different defaults:** the two operating thresholds differ by design (each model's default); use the sweeps
  for like-for-like reading. Thresholds that look good here were picked by looking at the test set, so they must
  be confirmed on other images.
- **CPU timings:** measured on 10 images on Colab's 2 vCPUs (Grounding DINO: remote control query only), not on
  the deployment machine.
- **Unused label:** the export's `remote-control-ac` label (26 boxes, AC remotes) is unmapped; none of the 70
  images contain it.
- **Model revision:** this run's `run_info.json` recorded the Grounding DINO revision as `null`; the revision
  above was read from the VM's Hugging Face cache, and `run_inference.py` now records it.

## Reproduce (from the repository root)

```bash
python tests/dataset/prepare_dataset.py verify
python detection/experiments/yolo/run_inference.py --run-name t4_yolo26s_conf0.25
python detection/experiments/yolo/run_inference.py --conf 0.001 --run-name t4_yolo26s_conf0.001
python detection/experiments/grounding_dino/run_inference.py --run-name t4_gdino-tiny_box0.05
python detection/experiments/yolo/evaluate.py --run t4_yolo26s_conf0.25
python detection/experiments/yolo/evaluate.py --run t4_yolo26s_conf0.001
python detection/experiments/grounding_dino/evaluate.py --run t4_gdino-tiny_box0.05
python tests/compare_models.py --yolo-run t4_yolo26s_conf0.001 --yolo-latency-run t4_yolo26s_conf0.25 \
    --gdino-run t4_gdino-tiny_box0.05 --yolo-cpu-run colabcpu_yolo26s_latency-sample \
    --gdino-cpu-run colabcpu_gdino-tiny_latency-sample
```

The YOLO scripts write their run folders to `detection/experiments/yolo/results/`. For this report those folders
were moved unchanged to `tests/results/yolo_reference_runs/`, which `compare_models.py` reads by default. To
compare other YOLO runs, pass `--yolo-results-dir detection/experiments/yolo/results`.
The `metrics_conf<t>.json` files in each run folder come from re-running the evaluator with `--conf-threshold <t>`.
The CPU runs used a 10-image manifest (two images per bucket), which is saved next to their predictions.
