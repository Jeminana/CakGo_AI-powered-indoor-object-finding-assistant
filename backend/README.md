# Backend — Member 3

## Objective
Serve the mobile frontend, accept images, call detector, compute left/center/right and return a stable JSON response.

## Tasks
- Day 1: Agree on API contract, server skeleton.
- Day 2: Mock `/api/detect` and `/api/health` endpoint.
- Day 3: Integrate detector and real phone requests.
- Days 4–5: Add errors, timeouts and search-response stability.
- Days 6–7: Test latency and demo reproducibility.

## Contract
Use `../docs/API_CONTRACT.md`. `POST /api/detect`: multipart `image` and `target`; JSON with `target`, `detected`, `position`, image dimensions, `detections`, and `message`. Return proper error codes for invalid images. Do not store uploaded frames by default.

## Done when
Mock and real detector results have same schema; no-match works; backend processes phone images. AI agent must not change frontend contract unilaterally.
