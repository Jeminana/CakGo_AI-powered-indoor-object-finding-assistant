# Real phone photos - team_photos_gdino

Photos: 94 (wallet 24, keys 20, glasses 24, remote_control 12, none 14). Each model is asked about all 4 objects in every photo.

By person (from the file name; `?` = no name in the file name): ? 30, cheolyeon 28, najmi 36.

Photo set fingerprint: `01d7103be4d5ddf863418214b63b173a53213f001f4f139fb3e584043127b985`. Runs with the same fingerprint used exactly the same photos.

Excluded: 15 screenshot PNGs (not phone photos): `glasses/glasses 2026-10-19 185850.png`, `glasses/glasses 2026-10-19 191026.png`, `keys/屏幕截图 2026-10-19 192935.png`, `keys/屏幕截图 2026-10-19 193003.png`, `keys/屏幕截图 2026-10-19 193119.png`, `keys/屏幕截图 2026-10-19 193206.png`, `keys/屏幕截图 2026-10-19 193236.png`, `keys/屏幕截图 2026-10-19 193307.png`, `keys/屏幕截图 2026-10-19 193407.png`, `remote controller/屏幕截图 2026-10-19 192333.png`, `remote controller/屏幕截图 2026-10-19 192420.png`, `remote controller/屏幕截图 2026-10-19 192547.png`, `remote controller/屏幕截图 2026-10-19 192642.png`, `remote controller/屏幕截图 2026-10-19 192722.png`, `remote controller/屏幕截图 2026-10-19 192809.png`.

**Found** = photos of the object where the model returned a box. **False alarms** = photos without the object where it still returned one. No ground-truth boxes: check box positions with `--sheets`.

| Model (threshold) | Object | Found | False alarms (all photos without it) | False alarms on `none` photos |
|---|---|---|---|---|
| gdino_multi (0.35) | wallet | 12/24 (50%) | 8/70 (11%) | 2/14 |
| gdino_multi (0.35) | keys | 10/20 (50%) | 11/74 (15%) | 0/14 |
| gdino_multi (0.35) | glasses | 12/24 (50%) | 2/70 (3%) | 1/14 |
| gdino_multi (0.35) | remote_control | 11/12 (92%) | 17/82 (21%) | 3/14 |
| gdino_single (0.35) | wallet | 23/24 (96%) | 35/70 (50%) | 4/14 |
| gdino_single (0.35) | keys | 20/20 (100%) | 63/74 (85%) | 9/14 |
| gdino_single (0.35) | glasses | 23/24 (96%) | 37/70 (53%) | 2/14 |
| gdino_single (0.35) | remote_control | 12/12 (100%) | 76/82 (93%) | 12/14 |

## Found by condition (supported objects only)

| Model | cluttered | dim | easy | far | partly hidden | unknown |
|---|---|---|---|---|---|---|
| gdino_multi | 15/16 (94%) | 4/8 (50%) | 7/8 (88%) | 2/8 (25%) | 7/8 (88%) | 10/32 (31%) |
| gdino_single | 16/16 (100%) | 7/8 (88%) | 8/8 (100%) | 7/8 (88%) | 8/8 (100%) | 32/32 (100%) |

## Threshold sweep: found / false alarms

| Model | Object | 0.25 | 0.35 | 0.5 |
|---|---|---|---|---|
| gdino_multi | wallet | 16/24 / 12/70 | 12/24 / 8/70 | 3/24 / 2/70 |
| gdino_multi | keys | 13/20 / 22/74 | 10/20 / 11/74 | 0/20 / 2/74 |
| gdino_multi | glasses | 15/24 / 8/70 | 12/24 / 2/70 | 7/24 / 0/70 |
| gdino_multi | remote_control | 12/12 / 40/82 | 11/12 / 17/82 | 8/12 / 3/82 |
| gdino_single | wallet | 23/24 / 50/70 | 23/24 / 35/70 | 18/24 / 12/70 |
| gdino_single | keys | 20/20 / 74/74 | 20/20 / 63/74 | 13/20 / 41/74 |
| gdino_single | glasses | 24/24 / 58/70 | 23/24 / 37/70 | 17/24 / 12/70 |
| gdino_single | remote_control | 12/12 / 82/82 | 12/12 / 76/82 | 11/12 / 46/82 |
