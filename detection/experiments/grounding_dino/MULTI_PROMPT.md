# Multi-prompt mode: fewer false alarms

`config/model.json` sets `"prompt_mode": "multi"` (the default since 2026-10-19). Set it to `"single"` for the
original behaviour; `run_inference.py --prompt-mode single|multi` overrides it for one benchmark run.

## Problem

With a single-object prompt (`"keys."`), Grounding DINO almost always returns a box. On images that do **not**
contain the object it still answered "found" on 53–94 % of validation images (table below). The phone test on
2026-10-19 showed the same thing: asked for "remote control", it boxed a computer mouse, a laptop keyboard and a
white router. With nothing better to match, the model gives the requested word to the most similar object.

## Method

The prompt lists **all four targets plus look-alike objects**:

```
wallet. keys. glasses. sunglasses. remote control. computer mouse. laptop. keyboard. pen. bottle. can. cup.
microphone. cable. charger. nail clipper. coin.
```

For each box, the score of every phrase is the highest token score inside that phrase. The box goes to the phrase
with the highest score, and it counts for the requested object only if that phrase belongs to it (for example,
"sunglasses" counts for glasses). Its confidence is that phrase's score, compared against `box_threshold` (0.35);
`text_threshold` is not used in this mode. There is still one model pass per request, so speed is unchanged.

The look-alike list is in `config/look_alikes.json`. It was **tuned on validation images, not on the 70 shared
test images**: 60 images each of wallet, glasses and remote control, 42 of keys, and 100 images with none of
them. All come from the Objects In House v5 valid split, excluding the 70 shared images. Words that describe a
target itself took boxes away from real wallets during tuning, so they were removed: *card, bag, book, box,
paper* (V0 → V1) and *mobile phone* (V1 → V3).

## Results

### Validation images used for tuning (threshold 0.35)

"Top-1 correct" means the highest-confidence box for the requested object is a correct box (IoU ≥ 0.5).

| Object | Top-1 correct, single | Top-1 correct, multi (V3) | False alarms on 100 images without it, single | False alarms, multi (V3) |
|---|---|---|---|---|
| Wallet | 52/60 | 36/60 | 53 | 5 |
| Keys | 42/42 | 41/42 | 94 | 5 |
| Glasses | 58/60 | 58/60 | 69 | 12 |
| Remote control | 58/60 | 58/60 | 82 | 16 |

### 70 shared test images (checked once, after tuning; threshold 0.35)

Single = run `t4_gdino-tiny_box0.05`; multi = run `rtx4050_gdino-tiny_multi_box0.05`.

| Object | Top-1 correct, single → multi | Negative images with a false alarm, single → multi | Precision, single → multi | AP50, single → multi |
|---|---|---|---|---|
| Wallet | 13/15 → 11/15 | 4/10 → 1/10 | 0.27 → 0.63 | 0.52 → 0.58 |
| Keys | 12/15 → 13/15 | 9/10 → 0/10 | 0.23 → 0.84 | 0.33 → 0.70 |
| Glasses | 15/15 → 15/15 | 6/10 → 0/10 | 0.26 → 0.64 | 0.70 → 0.69 |
| Remote control | 15/15 → 15/15 | 9/10 → 1/10 | 0.20 → 0.64 | 0.94 → 0.94 |

Latency of multi mode: mean 321 ms per request (RTX 4050 Laptop GPU, fp32).

### 4 real iPhone 15 Pro photos (cluttered desks, one dim)

Each photo contains one target. All 4 targets were asked about in every photo.

| | Single | Multi |
|---|---|---|
| Target found | 4/4 | 4/4 |
| False alarms (12 questions about absent objects) | 9 | 4 |

Remaining false alarms: "remote control" on a microphone and on a cluttered desk area; "glasses" once at 0.36;
"wallet" on a navy card holder (arguably a wallet).

## Limitations

- **Wallet recall drops** (validation 52 → 36 of 60). Wallets are flat dark rectangles, and part of their boxes
  still go to other phrases. If missed wallets matter more than false alarms, use single mode for wallet only.
- **Remote control is still the weakest on real photos.** Real rooms contain many "things with buttons".
- The look-alike list is tuned to this dataset. Real homes contain objects that are not on the list.
- 4 real photos is a very small sample. The team's real-phone photo set should confirm these results.
- Borderline false alarms change from frame to frame, while real objects stay detected. Only speaking after the
  same answer appears in 2 consecutive frames (voice guidance) should remove more of them.

## Reproduce

```bash
python detection/experiments/grounding_dino/run_inference.py --run-name rtx4050_gdino-tiny_multi_box0.05
python detection/experiments/grounding_dino/evaluate.py --run rtx4050_gdino-tiny_multi_box0.05
python detection/experiments/grounding_dino/run_inference.py --prompt-mode single --run-name <name>  # old behaviour
```
