# Zero-shot and few-shot experiments — Member 4

Exploratory tests, not a model decision. They ask two questions:

- Does a newer open-vocabulary model (YOLOE-26, SAM 3) beat the app's detector (Grounding DINO tiny, multi-prompt)?
- Does registering an item with example photos help find *that* item?

**Main findings**

- **Grounding DINO multi is still the best balance.** SAM 3 finds as much or more but is about 6× slower. YOLOE is
  fast but misses about half.
- **Example photos never beat the object word.** Neither YOLOE visual prompts nor SAM 3 exemplars did.
- **Example photos work as a check.** Comparing Grounding DINO's boxes with the registration photos (DINOv3)
  removed the false alarms without losing the item. Look-alikes were halved, not solved.
- **Small items are the main limit.** An earbud is about 37 px in a phone photo.

## Experiments

| Experiment | Scripts | Results |
|---|---|---|
| Zero-shot on the 70 shared images | `run_zero_shot.py`, `compare_zero_shot.py`, `colab_job.sh` | [`zero_shot_70/COMPARISON.md`](../results/zero_shot_70/COMPARISON.md) |
| One-shot registration on Member 4's 28 photos (YOLOE visual prompts), and why it fails | `instance_one_shot.py`, `diagnose_visual_prompt.py` | kept locally |
| Valuables register-then-find: YOLOE one/four-shot, Grounding DINO, SAM 3 text and exemplars | `valuables_test.py`, `valuables_sam3.py` | kept locally |
| YOLOE size follow-up: 1280 px input, example photos at several sizes | `valuables_yoloe_scale.py` | kept locally |
| Grounding DINO + DINOv3 verification | `dino_verify.py` | kept locally |

- **Shared code:** `detectors.py` holds the YOLOE-26 and SAM 3 adapters with the app's `detect(image, target)`
  result format.
- **Tests:** `test_zero_shot.py` has model-free checks.

**Photos.** The real-photo experiments use Member 4's own photos. The photos and everything derived from them
(gold boxes, per-photo answers, summaries) stay on Member 4's machine and are not in the repository. The
scripts expect those files at their default paths; the findings above are the aggregate results.

## 70-image zero-shot test: same conditions for every model

| Item | Rule |
|---|---|
| Images | The 70 images of `tests/dataset/manifest.csv`; every run records the manifest sha256 (`ca73e2c1…`) |
| Words | Grounding DINO's `config/prompts.json` (+ `config/look_alikes.json` in multi mode), fixed before the run |
| Prompt modes | `single`: only the target's words. `multi`: all target words + 12 look-alikes; a box counts for the target only if a target word matches it best (Grounding DINO's rule, see `detectors.py`) |
| Queries | One query per image per category (280), like the app's one-target requests; every query timed |
| Saved boxes | Confidence >= 0.05, so AP50 and threshold sweeps can be computed afterwards |
| Operating threshold | Each library's default, fixed before the run: Grounding DINO 0.35 (app config), YOLOE 0.25 (Ultralytics default), SAM 3 0.5 (transformers examples). Common sweep 0.1–0.5 as well |
| Hardware | One Colab VM (Tesla T4), fp32, one process per run, 3 untimed warm-up queries per category |
| Scoring | Day 3 code, imported not copied: `grounding_dino/evaluate.py` `score_category`, `tests/compare_models.py` `top1_rate` and `GT_ISSUES` |

How it was run (Colab CLI, team account):

```bash
CFG=<colab-cli config of the team account>
colab --config $CFG new -s cakgo-zs --gpu T4
colab --config $CFG upload -s cakgo-zs zs_bundle.tar.gz /content/zs_bundle.tar.gz   # repo files + the 70 images
# on the VM: unpack to /content/repo, python tests/dataset/prepare_dataset.py verify
colab --config $CFG install -s cakgo-zs "ultralytics==8.4.177" "git+https://github.com/ultralytics/CLIP.git"
# on the VM, in the background: bash tests/zero_shot/colab_job.sh
colab --config $CFG stop -s cakgo-zs
```

SAM 3 (`facebook/sam3`) is gated. For this run its files were downloaded on the Member 4 Mac with the existing
Hugging Face login and copied into the VM's Hugging Face cache (`hub/models--facebook--sam3/snapshots/<revision>/`
plus `refs/main`), so `colab_job.sh` runs SAM 3 with `HF_HUB_OFFLINE=1`. `colab upload` rejects a single 3.4 GB
file (the limit lies between 64 and 256 MB), so `model.safetensors` went as 64 MB parts (`split -b 64m`), was
joined on the VM and checked against its sha256.

## Real-photo runs

- **Mac (Apple MPS):** the 28-photo one-shot test, and YOLOE and Grounding DINO on the valuables.
- **Colab T4 (team account):** SAM 3 on the valuables, the YOLOE size follow-up, and the DINOv3 verification.
- **Gated models (SAM 3, DINOv3):**
  - The photographer uploaded their own Hugging Face token to the VM.
  - The VM downloaded the model at a pinned revision and checked its sha256.
  - The token file was deleted right after the download.
- **Photo size:** for the DINOv3 run the photos were sent at the app frame size (long side 1280 px, JPEG quality 95).
- **Resuming a stopped run:** `valuables_sam3.py` and `dino_verify.py` save their answers query by query, so a
  stopped run can resume.

Local check (no GPU needed):

```bash
python -m pytest tests/zero_shot -q
```
