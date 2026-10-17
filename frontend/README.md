# Smartphone Frontend — Member 2

## Korean voice connected (상싱위, 2026-10-17)

The page loads `/voice/guidance.js` from the same origin. **Start Camera** enables
Korean speech directly inside the user's click (before waiting for camera access).
Each current detection response is passed to `voice.guide(result)` after the boxes
and Korean text are displayed. **Voice guidance: on/off** independently mutes or
reenables speech; **Stop Camera** and leaving the page cancel speech and camera
tracks. Changing targets cancels old speech, and stale responses are ignored.
The shared voice module retains its four-second duplicate cooldown and prevents
overlapping utterances. Network/camera failures call the module's error cues.

If the module fails to load, detection and screen text still work and the voice
button reports it unavailable. If browser synthesis fails, Korean status text is
retained. A Korean device voice is recommended; audio/device compatibility must
still be checked on real iOS Safari and Android Chrome (the automated tests below
do not prove audible phone output).

Automated wiring checks (no camera permission or model download):
`node frontend/test_voice_integration.cjs`. Existing module checks:
`node voice/test_guidance.cjs`.

Manual phone acceptance: open the frontend via HTTPS, tap Start Camera, confirm
audible Korean directions and missing-target guidance, check mute/unmute, repeated
frames, target change, Stop Camera, and a backend disconnect. Test both iOS Safari
and Android Chrome; record device/browser, detector mode and observed results.

## Backend connection (Member 4)

While the camera is on, `index.html` captures one frame (JPEG, long side at most
1280 px), posts it with the selected target to `POST /api/detect`, and pauses 1 s
after each answer before the next frame (30 s timeout). Every returned box is drawn
over the preview in original-image coordinates scaled to the screen; the thicker box
is the highest-confidence one, which the guidance message is based on. The Korean
`message` appears under the preview. A note says when the backend runs in mock mode
(`X-CakGo-Detector: mock`), because those boxes are fixtures, not detections.

The page calls the API on its own origin. `dev_server.py` (Python standard library
only) serves this folder and forwards `/api/` and `/voice/` to the backend, so no
CORS setup is needed and one HTTPS tunnel covers a phone. From the repository root,
in two terminals (backend dependencies: see [`backend/README.md`](../backend/README.md)):

```bash
python -m uvicorn backend.app:app --port 8000   # mock detector by default
python frontend/dev_server.py                   # http://localhost:5173
```

Open http://localhost:5173 on the computer (localhost may use the camera without
HTTPS), choose a target and press Start Camera. Start the backend with
`CAKGO_MOCK_POSITION=left|center|right|missing` to check each case, or point the dev
server at another backend with `--backend https://<host>`.

Phone: the camera needs HTTPS. Expose the dev server through an HTTPS tunnel, for
example `cloudflared tunnel --url http://localhost:5173` or `ngrok http 5173`, and
open the printed https URL on the phone. Anyone with that URL can use the page and
the backend while the tunnel runs, so stop it after testing.

Tests: `python -m pytest frontend/test_dev_server.py -q` (proxy and static serving).
Not yet checked on real devices: iOS Safari, Android Chrome, and the real Grounding DINO
backend with audible speech. Frontend voice wiring is implemented above. The target
list, the 1 s pace and HTTPS hosting are still open team
decisions ("Day 1 decisions to confirm" in [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md)).

## Component Purpose
The `frontend/` component provides an accessible, mobile-first web interface that allows users (including those with low vision) to capture video frames using their smartphone's rear camera, select a target indoor object, submit frames for detection, and visualize detection boxes.

## Assigned Team Member
- **Member 2**
  - **Initial Responsibility:** Smartphone camera setup, accessible UI layout, camera frame capture.
  - **Post-Testing Responsibility:** Full frontend development, integration with backend API, and live guidance display.

## Independent Work via Mock Data
Member 2 does not need to wait for real object detection models to be ready. Development and testing proceed independently using mock JSON responses conforming to [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md).

## Tasks & Development Priorities
1. **Camera Stream Access:** Implement accessible rear-camera access via `navigator.mediaDevices.getUserMedia` with proper error handling (permission denied, no camera).
2. **Accessible Target Selection:** Implement large, high-contrast target buttons for common household items (e.g., Keys, Phone, Glasses, Wallet).
3. **Frame Capture & Submission:** Grab individual video frames from `<video>` onto `<canvas>` and encode as JPEG/PNG for transmission to `POST /api/detect`.
4. **Bounding Box Visualization:** Overlay bounding boxes on the preview display, properly scaling normalized or pixel coordinates `[x1, y1, x2, y2]` from the original frame dimensions to the screen.
5. **Guidance & Voice Integration:** Connect visual indicators with voice prompts generated by `voice/` or the backend.

## Expected Inputs & Outputs
- **Input:** Camera stream from device; user selection of target item.
- **Output:** HTTP request to Backend:
  - Endpoint: `POST /api/detect`
  - Body: `multipart/form-data` with `image` (binary file) and `target` (string).
- **Receives from Backend:**
  ```json
  {
    "target": "cell phone",
    "detected": true,
    "position": "center",
    "image_width": 640,
    "image_height": 480,
    "detections": [
      {"label": "cell phone", "confidence": 0.93, "bbox": [200, 150, 440, 380]}
    ],
    "message": "찾는 물건이 화면 가운데에 보입니다."
  }
  ```

## Mobile Constraints & Requirements
- **HTTPS / Secure Context:** Mobile browsers (iOS Safari, Android Chrome) strictly require HTTPS or `localhost` to access `getUserMedia`. When testing on a physical phone connecting to a local dev server, use a tunneling tool (e.g., ngrok) or local SSL certificates.
- **Responsive & Accessible Design:** Use high-contrast color schemes, large touch targets (min 48x48px), clear focus states, and screen reader-friendly aria labels.

## Instructions for AI Coding Agents
1. Operate strictly within `frontend/`.
2. Do not modify [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md) unilaterally.
3. Test with mock JSON data first before attempting backend integration.
4. Scale bounding boxes dynamically based on `image_width` and `image_height` relative to the displayed viewport.
