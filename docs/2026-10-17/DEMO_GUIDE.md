# Prototype operation and demo record

## Before the demonstration

Use a local checkout of the team repository. For the connected speech UI, use `feature/frontend-voice` until PR #8 merges into main. Use the repository's tested dependency instructions. Avoid mixing a stock YOLO checkpoint with unsupported target classes.

From the repository root:

```sh
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.app:app --port 8000
```

In a second terminal:

```sh
python frontend/dev_server.py
```

Open `http://localhost:5173`. Default backend mode is **mock**, which returns fixed sample boxes and performs no recognition. Use it to check UI and speech wiring only.

## Grounding DINO backend

Install the detector's requirements using its README. Then select its existing factory:

```sh
CAKGO_DETECTOR=custom \
CAKGO_DETECTOR_FACTORY=detection.experiments.grounding_dino.detector:create_detector \
python -m uvicorn backend.app:app --port 8000
```

This shell syntax is for macOS/Linux. The first request may load/download model files and take longer than a warmed-up call. GPU hosting is recommended based on the recorded CPU latency. The frontend's 30-second timeout can expire during a cold start, even though the upload demo has a longer timeout. Warm up with a request before presenting and record any timeout rather than treating it as a missing target. Real-model operation was not newly validated for this documentation task.

## Fine-tuned YOLO backend

Install the YOLO experiment dependencies and obtain the matching `best.pt` through the team's agreed channel. Weights are excluded from git.

```sh
CAKGO_DETECTOR=yolo \
CAKGO_YOLO_WEIGHTS=/absolute/path/to/best.pt \
python -m uvicorn backend.app:app --port 8000
```

The recorded best-weights SHA256 is `09d41cb206f3a46be6cfe04aaec766c8194a5b63bec2680932341e9d954ec4e3`.

## Phone and speech acceptance checks

Phone camera access needs HTTPS. Use the team's existing deployment or an approved HTTPS tunnel to the frontend proxy. A computer's localhost URL does not point to that computer when typed on a phone.

1. Choose the target and press Start Camera. Allow camera access and confirm Korean speech starts.
2. Move a visible object left, center and right. Verify the box, text and speech agree.
3. Remove all target objects. Check that the spoken result refers only to the current frame.
4. Test repeat suppression, mute/unmute and changing targets.
5. Press Stop Camera. Confirm capture and audio stop. Repeat while camera permission is pending.
6. Disconnect the backend. Check error text and voice behavior.
7. Repeat on iOS Safari and Android Chrome. Record missing Korean voices or permission issues.

| Date/device/browser | Detector and weights | Scenario | Expected | Observed | Pass/fail | Evidence |
|---|---|---|---|---|---|---|
| Not yet recorded | Not yet recorded | Left / center / right / missing | Matching text, box and speech | Pending test | Pending | Add consented clip or log reference |

## New phone-photo evaluation

The team assigned the other three members 28 photos each, due 19 October: 6 per item plus 4 with none of the target items, for 84 total. Each item uses designs A/B and the prescribed easy, cluttered, occluded, far and dim conditions. Store photos only in the teammates-only Drive folders. Do not upload private room photos to public GitHub.

The set is **test-only**. Do not train, tune prompts or pick thresholds using these final test outcomes. Before inference, record filenames, design, scene, target boxes and all four absent labels for the none images. Preserve original photos as instructed, without cropping or editing. If design B repeats design A, record that exception.

## Demonstration sequence

Explain the target user and current model first. Show one successful target, one missing-target scene, and one known failure. State whether the demonstration uses mock or a real model. End with the measured limitations and the next validation step. A successful mock demonstration is not recognition evidence.
