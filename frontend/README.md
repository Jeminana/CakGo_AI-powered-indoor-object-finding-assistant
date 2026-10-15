# Smartphone Frontend — Member 2

## Objective
Create a phone-friendly page with rear-camera preview, large target-selection controls, capture/search button, accessible status and detection display.

## Tasks
- Day 1: Mock page; test camera permission on actual phone.
- Day 2: Capture single frame and show mock result.
- Day 3: Send frame to `POST /api/detect`; receive JSON.
- Days 4–5: Draw boxes; integrate audio instructions with Member 3.
- Days 6–7: Test phone usability and fallback workflow.

## Cautions
Phone browser `getUserMedia` generally requires HTTPS/secure context. A laptop's localhost is not automatically secure when reached by another phone. Verify phone permissions and an HTTPS development origin on Day 1. Vibration browser support varies; treat it as optional. Scale original-image box coordinates to on-screen image.

## Done when
Phone can capture, submit, and present mock/real result. Use `../docs/API_CONTRACT.md`; never portray mock detections as real.
