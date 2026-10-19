# Dataset Information — CakGo Shared Evaluation Set

Verified on **2026-10-16** from the public Roboflow Universe pages listed below. The evaluation subset was built
the same day with `prepare_dataset.py build` (Roboflow API key required for the download); section 4 has the
final counts.

## 1. Sources

### Primary — Objects In House (cropdev)

| Field | Verified value |
|---|---|
| URL | <https://universe.roboflow.com/cropdev/objects-in-house> |
| Task / annotation type | Object detection, bounding boxes |
| Classes (8, exact source labels) | `wallet`, `keys`, `glasses-sunglasses`, `remote-control`, `backpack-bag`, `broom`, `casual-shoes`, `earphone` |
| Source images | 9,868 |
| License | **CC0 1.0 / Public Domain** (<https://creativecommons.org/publicdomain/zero/1.0/>) |
| Last modified | 2023-11-03 |
| Download | Free, but requires a Roboflow account (page metadata: `requiresSubscription: true`; API returns *"This method requires your API key"*) |
| Export formats offered | COCO JSON, YOLO (v5–v12, YOLO26) TXT, Pascal VOC XML, TFRecord, CreateML, PaliGemma, and others |

Published versions (from the version pages):

| Version | Generated | Images | Split (train / valid / test) | Preprocessing | Augmentation |
|---|---|---|---|---|---|
| v3 | 2023-10-22 | 9,868 | 7,472 / 1,577 / 819 | Auto-orient, stretch to 640×640 | none |
| **v5 (used)** | 2023-10-31 | 9,868 | 6,912 / 1,965 / 991 | Auto-orient, stretch to 640×640 | none |
| v11 | 2023-11-02 | 16,780 | 13,824 / 1,965 / 991 | Auto-orient, fit (white edges) in 640×640 | yes (train only) |
| v12 | 2023-11-03 | 16,780 | 13,824 / 1,965 / 991 | Auto-orient, fit (white edges) in 1024×1024 | yes (train only) |

**Why v5:** it has no augmentation, so there are no synthetic near-duplicate images. Its valid/test sizes
(1,965 / 991) are also identical to v11/v12. That suggests v5 shares their held-out split (inferred from the
counts only, not confirmed). The evaluation subset is drawn from the **test and valid** splits only.

**Checked after download (2026-10-16):** the v5 COCO export also uses a ninth label, `remote-control-ac`
(26 boxes in test+valid). It is not in `class_mapping.json`, so `build` treats it as non-target; none of the
70 selected images contain it. Whether AC remotes should count as *Remote control* is a team decision.

### Secondary — Home item detection (EmulxionAI)

Used **only** if the primary dataset cannot fill a category.

| Field | Verified value |
|---|---|
| URL | <https://universe.roboflow.com/emulxionai-w5sq2/home-item-detection-7kqro> |
| Classes (9, exact source labels) | `box for eye glasses`, `car key with keychain`, `car key without keychain`, `eye glasses`, `keys for apartment`, `pills blisters`, `pills bottles`, `pills boxes`, `remote controller` |
| Source images | 707 |
| Versions | 1 (v1, generated 2025-12-12) |
| v1 export | 1,697 images: 1,485 / 141 / 71 (train / valid / test); auto-orient, stretch to 512×512; ×3 augmentation (flip, 90° rotate, crop, ±15° rotation, shear, 15 % grayscale, hue, …) |
| License | **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>), so attribution is required |
| Download | Free, requires a Roboflow account / API key |

Note: this dataset has **no wallet class**. Its augmented copies are in the train split, which is not used.

### Future — Microsoft ORBIT

<https://github.com/microsoft/ORBIT-Dataset>. **Not downloaded at this stage**, by team decision.

## 2. Attribution

- *Objects In House* by cropdev, Roboflow Universe, CC0 1.0. Attribution is not required but is given.
- *Home item detection* by EmulxionAI, Roboflow Universe, licensed under CC BY 4.0. If any of its images are
  used, this attribution must stay with the dataset and any report or demo that shows them.

## 3. Category mapping

Original labels are kept unchanged in `annotations/ground_truth.json`; the mapping below is defined in
`class_mapping.json`.

| CakGo category | Folder | Target count | Objects In House label | Home item detection labels |
|---|---|---|---|---|
| Wallet | `images/wallet/` | 15 | `wallet` | (none) |
| Keys | `images/keys/` | 15 | `keys` | `car key with keychain`, `car key without keychain`, `keys for apartment` |
| Glasses / Sunglasses | `images/glasses/` | 15 | `glasses-sunglasses` | `eye glasses` |
| Remote control | `images/remote_control/` | 15 | `remote-control` | `remote controller` |
| Target-absent | `images/negative/` | 10 | an image whose labels are only `backpack-bag`, `broom`, `casual-shoes`, `earphone` | only `box for eye glasses` / `pills *` |

Not mapped (kept as-is, treated as non-target): `backpack-bag`, `broom`, `casual-shoes`, `earphone`,
`box for eye glasses`, `pills blisters`, `pills bottles`, `pills boxes`.

## 4. Evaluation subset (built 2026-10-16)

| Category | Target | Selected | Shortfall / notes |
|---|---|---|---|
| Wallet | 15 | 15 | none |
| Keys | 15 | 15 | none |
| Glasses / Sunglasses | 15 | 15 | none |
| Remote control | 15 | 15 | none |
| Target-absent | 10 | 10 | none |

Selection seed: `cakgo-eval-v1`. The exact numbers are in `annotations/selection_report.json`.
Built on 2026-10-15 20:56 UTC from v5 test+valid (2,956 candidate images); 4 near-duplicates were skipped.
No shortfalls, so the secondary dataset was not downloaded or used. All 70 images are 640×640.

## 5. Model coverage (for interpreting results)

| Category | COCO-pretrained YOLO | Grounding DINO |
|---|---|---|
| Wallet | **Not supported** (no COCO class) | Open-vocabulary text prompt (Member 4) |
| Keys | **Not supported** | Text prompt |
| Glasses / Sunglasses | **Not supported** | Text prompt |
| Remote control | Supported (`remote`, COCO id 65) | Text prompt |

## 6. Known limitations

- **Not original camera frames.** Roboflow exports are preprocessed: v5 images are *stretched* to 640×640, so
  aspect ratios are distorted. Both models see the same distortion, but the absolute results may differ from
  phone-camera frames.
- **Testing conditions unknown.** The sources provide no lighting, clutter or occlusion metadata, so
  `testing_condition` is `unknown` unless someone fills it in manually.
- **Annotation quality not audited.** Labels are community annotations. Negatives are defined by their labels
  and could contain an unlabeled target, so manual review (`review_status`) is required.
- **Small sample.** About 70 images gives indicative, not statistically strong, comparisons.
- **Class balance of the primary dataset** (boxes in test+valid, counted after download): casual-shoes 928,
  glasses-sunglasses 594, wallet 432, broom 377, remote-control 374, keys 325, backpack-bag 258, earphone 184,
  remote-control-ac 26.
- **Unlabeled targets in the subset.** An AI pre-screen of contact sheets (not a human review) found 12 images
  where a target is visible but has no box: sunglasses/glasses in `keys_001`, `keys_003`, `keys_004`, `keys_007`,
  `keys_008`, `keys_009`, `keys_012`, `keys_013`; a wallet in `keys_015`; keys in `wallet_003`; a second wallet in
  `wallet_006` and `wallet_007`. Correct detections of those objects are scored as false positives. The
  10 negatives showed no visible target. `review_status` is still `unreviewed` until a human checks the images.
- **Download depends on Roboflow availability and a personal API key.** If Roboflow regenerates an export,
  file hashes may change; `verify` detects this (see README section 2b).
