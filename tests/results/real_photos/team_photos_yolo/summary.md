# Real phone photos - team_photos_yolo

Photos: 94 (wallet 24, keys 20, glasses 24, remote_control 12, none 14). Each model is asked about all 4 objects in every photo.

By person (from the file name; `?` = no name in the file name): ? 30, cheolyeon 28, najmi 36.

Photo set fingerprint: `01d7103be4d5ddf863418214b63b173a53213f001f4f139fb3e584043127b985`. Runs with the same fingerprint used exactly the same photos.

Excluded: 15 screenshot PNGs (not phone photos): `glasses/glasses 2026-10-19 185850.png`, `glasses/glasses 2026-10-19 191026.png`, `keys/屏幕截图 2026-10-19 192935.png`, `keys/屏幕截图 2026-10-19 193003.png`, `keys/屏幕截图 2026-10-19 193119.png`, `keys/屏幕截图 2026-10-19 193206.png`, `keys/屏幕截图 2026-10-19 193236.png`, `keys/屏幕截图 2026-10-19 193307.png`, `keys/屏幕截图 2026-10-19 193407.png`, `remote controller/屏幕截图 2026-10-19 192333.png`, `remote controller/屏幕截图 2026-10-19 192420.png`, `remote controller/屏幕截图 2026-10-19 192547.png`, `remote controller/屏幕截图 2026-10-19 192642.png`, `remote controller/屏幕截图 2026-10-19 192722.png`, `remote controller/屏幕截图 2026-10-19 192809.png`.

**Found** = photos of the object where the model returned a box. **False alarms** = photos without the object where it still returned one. No ground-truth boxes: check box positions with `--sheets`.

| Model (threshold) | Object | Found | False alarms (all photos without it) | False alarms on `none` photos |
|---|---|---|---|---|
| yolo_v1 (0.25) | wallet | 4/24 (17%) | 1/70 (1%) | 0/14 |
| yolo_v1 (0.25) | keys | 6/20 (30%) | 1/74 (1%) | 0/14 |
| yolo_v1 (0.25) | glasses | 11/24 (46%) | 7/70 (10%) | 1/14 |
| yolo_v1 (0.25) | remote_control | 7/12 (58%) | 1/82 (1%) | 0/14 |

## Found by condition (supported objects only)

| Model | cluttered | dim | easy | far | partly hidden | unknown |
|---|---|---|---|---|---|---|
| yolo_v1 | 4/16 (25%) | 3/8 (38%) | 6/8 (75%) | 1/8 (12%) | 3/8 (38%) | 11/32 (34%) |

## Threshold sweep: found / false alarms

| Model | Object | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| yolo_v1 | wallet | 4/24 / 1/70 | 3/24 / 0/70 | 3/24 / 0/70 |
| yolo_v1 | keys | 6/20 / 1/74 | 6/20 / 1/74 | 5/20 / 0/74 |
| yolo_v1 | glasses | 11/24 / 7/70 | 10/24 / 5/70 | 9/24 / 1/70 |
| yolo_v1 | remote_control | 7/12 / 1/82 | 7/12 / 0/82 | 7/12 / 0/82 |
