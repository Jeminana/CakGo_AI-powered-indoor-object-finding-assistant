# Real phone photo test

Scores the detectors on the team's own phone photos, which are closer to real app use than the 70 stock-style
dataset images. The photos show teammates' homes, so **they are never committed**: `photos/` and `sheets/` are
gitignored, and only the summary tables in `tests/results/real_photos/` are committed.

## 1. Get the photos

Download the teammates-only Drive folder `real_phone_photos/` and put it at `tests/real_photos/photos/`:

```text
tests/real_photos/photos/
├── wallet/          wallet_<name>_A1.jpg … B3.jpg
├── keys/
├── glasses/
├── remote_control/
└── none/            none_<name>_N1.jpg … N4.jpg
```

The folder gives the object in the photo (one per photo; `none` = none of the four). The code at the end of
the file name gives the condition: A1 easy, A2 cluttered, A3 partly hidden, B1 far, B2 dim, B3 cluttered,
N1–N4 none photos. JPG and iPhone HEIC/HEIF are accepted (HEIC needs `pip install pillow-heif`).

The script is tolerant of how the shared Drive folder actually looks:
- `empty/` counts as `none/`, and `remote controller/` counts as `remote_control/`.
- The downloaded `real_photos/` folder can be used directly with `--photos tests/real_photos/real_photos`.
  It is gitignored too.
- **PNG files are excluded by default.** In the shared folder they are small screenshots of web images, not
  phone photos. They are listed in the summary; `--include-screenshots` keeps them.
- Codes outside the plan (C1, D2, A4, B5, …) and names without a person's name get condition `unknown`.
- Every run records a **photo set fingerprint** (sha256 over all photos). Separate runs with the same
  fingerprint used exactly the same photos.

## 2. Run (from the repository root)

```bash
python tests/real_photos/score_real_photos.py --photos tests/real_photos/photos --run-name team_photos_v1 --sheets
```

- Default models: `gdino_multi`, `gdino_single`, `yolo_v1`. Others: `yolo_v2`, `yolo_pretrained`
  (COCO, remote control only). Choose with `--models …`.
- `gdino_multi` and `gdino_single` need the multi-prompt Grounding DINO (PR #10).
- `yolo_v1` / `yolo_v2` need the fine-tuned weights (`best.pt` from the team Drive).
- Photos are prepared like app frames: iPhone rotation applied, long side scaled to 1280 px.
- Takes about 2–3 minutes for 84 photos on an RTX 4050.

### Running it as a team

- **Freeze the photo folder first.** Check that it has 84 photos with a photo code in every file name; the
  script warns about files without one. After that, nobody adds, renames or deletes photos: every run must use
  exactly the same photos.
- **Runs can be split by model:** for example `--models gdino_multi gdino_single` on one machine and
  `--models yolo_v1` on another. The scores are per model, so separate runs compare fairly when the photos and
  the default thresholds are the same.
- **Grounding DINO needs a GPU for this.** On a CPU it takes about 15 s per question, so about 3 hours for
  84 photos × 4 objects × 2 modes. On an RTX 4050 it takes a few minutes. Uploading the photos to Colab sends
  them to Google, so only do that if everyone agrees.
- **Don't tune on these photos before the comparison.** Keep the thresholds and `config/look_alikes.json` as
  they are. If someone tunes afterwards, report it as a separate second result.

## 3. Read the results

`tests/results/real_photos/<run>/summary.md` (also `summary.json` and `per_photo.csv`):

| Column | Meaning |
|---|---|
| **Found** | Photos of the object where the model returned a box at its threshold (higher is better) |
| **False alarms** | Photos **without** the object where it still returned a box, out of all other photos (lower is better) |
| **Found by condition** | Found rate for easy / cluttered / far / dim / partly hidden photos |
| **Threshold sweep** | Found and false alarms at 0.25 / 0.35 / 0.5 |

There are no ground-truth boxes, so "found" only means the model said yes. `--sheets` draws each model's top box
for every photo into `tests/real_photos/sheets/<run>/` (gitignored). Look through them and count a "found" as
wrong if its box is on a different object.
