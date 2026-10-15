# CakGo Shared API Contract v0 — Proposal

All developers agree on this before coding. Change requests must be discussed with the team.

## System
Phone browser camera → Python backend → chosen detector → JSON result → mobile UI + Korean text-to-speech.

## Proposed endpoints
- `GET /api/health` → `{ "status": "ok" }`
- `POST /api/detect`, `multipart/form-data`: `image` (JPEG/PNG), `target` (agreed model label or text prompt)

### Example: detected (illustrative only)
```json
{
  "target": "cell phone",
  "detected": true,
  "position": "right",
  "image_width": 640,
  "image_height": 480,
  "detections": [
    {"label": "cell phone", "confidence": 0.91, "bbox": [360, 100, 530, 300]}
  ],
  "message": "찾는 물건이 화면 오른쪽에 보입니다."
}
```
### Example: no detection
```json
{
  "target": "cell phone",
  "detected": false,
  "position": null,
  "image_width": 640,
  "image_height": 480,
  "detections": [],
  "message": "현재 카메라 화면에서 찾는 물건이 보이지 않습니다."
}
```

## Conventions
- `bbox` is `[x1,y1,x2,y2]` in ORIGINAL IMAGE pixel coordinates; frontend scales for display.
- Position is based on detected target box center / width: `<1/3` left, `1/3–2/3` center, `>2/3` right.
- If multiple matches, report all in `detections`; guide to highest-confidence one.
- No detection means *not visible in this frame*, not *not present in the room*.
- Invalid image/service error: appropriate HTTP error with `{ "error": "..." }`.
- Uploaded imagery is processed ephemerally by default. Secure HTTPS origin is required for normal phone camera permissions.
- No location memory or physical distance guarantee in core MVP.

## Day 1 decisions to confirm
Exact target-label list, size limit, confidence threshold, image frequency, HTTPS hosting, and browser/device targets.
