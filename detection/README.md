# Detection — Member 1

## Objective
Compare pretrained YOLO and Grounding DINO for an indoor object-finding proof of concept; choose one based on supported classes, detection quality and speed. Keys, wallets and glasses may need more data: never promise recognition without testing.

## Tasks
- Day 1: Set up both models and prepare the same safe sample images.
- Day 2: Compare clearly visible, cluttered, partially obscured examples; log missed/false detections and latency.
- Day 3: Select model; implement a common `detect(image, target)` adapter.
- Days 4–5: Improve model stability and integrate through backend.
- Days 6–7: Test and document actual performance and limitations.

## Interface
Input: decoded frame + target label/prompt. Output: detections with `label`, `confidence`, `bbox=[x1,y1,x2,y2]` in original-image pixels. See `../docs/API_CONTRACT.md`. Keep detector independent of frontend and speech.

## Done when
Real image processed, matching detections returned, no-detection safely handled, classes/latency documented. AI agents must read root README and API contract before changing code.
