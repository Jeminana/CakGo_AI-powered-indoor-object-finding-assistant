# Voice Guidance Module — Member 3

## Implemented voice helper (상싱위)

`guidance.js` exports browser-global `CakGoVoice`. Load the script, construct
`new CakGoVoice({onStatus: text => { /* update UI text */ }})`, and call `enable()`
inside a user click. Pass each successful detection response to `guide(result)`.
Call `stop()` to disable audio and cancel queued speech; call `error('camera')`
or `error('network')` for failures. Detection speech requires the same target and
answer (left/center/right/not found) in two consecutive successful responses.
Text still updates immediately. A changed answer or target starts confirmation
again; an error, stop, or enable also resets confirmation. This suppresses
single-frame spoken alerts, but cannot rule out persistent false detections.

Repeated found directions retain the four-second cooldown. Not-found speech
repeats at most every ten seconds per target in a scan, even if another answer
or error intervenes. Stop clears this history for the next scan. The optional
`missingCooldownMs` constructor setting defaults to `10000`. Startup and error
cues remain immediate, and speech is never queued over an active utterance.

Browser Web Speech synthesis avoids a server TTS dependency and generated audio
storage. A ko-KR voice is selected when available; otherwise the browser default
voice is used. Text status survives disabled/failed audio. No distance estimate
is announced. Open the backend's `/voice-demo` for manual checks.

Automated logic checks: `node voice/test_guidance.cjs` (Node.js required).
Frontend lifecycle checks: `node frontend/test_voice_integration.cjs`.
Actual Korean speech on iOS Safari and Android Chrome still requires testing;
the automated checks do not establish device compatibility or audio quality.

## Component Purpose
The `voice/` component handles spoken Korean directional audio feedback for CakGo. It enables visually impaired users and those with low vision to locate everyday items without needing to examine bounding boxes on the screen.

## Assigned Team Member
- **Member 3**
  - **Initial Responsibility:** Voice guidance script design, Korean TTS testing, and audio message logic.
  - **Post-Testing Responsibility:** Full integration with backend detection API and frontend audio playback.

## Independent Work via Mock Data
Member 3 can design and validate speech synthesis (Web Speech API in browser or server-side gTTS/edge-tts) immediately using mock JSON detection results, without waiting for the detection model or real smartphone frames.

## Tasks & Development Priorities
1. **TTS Strategy Selection:** Test browser-native `window.speechSynthesis` (Korean voice) versus server-side audio file generation (`gTTS`) to find the lowest latency and highest mobile reliability.
2. **Standard Message Mapping:** Map spatial coordinates (`left`, `center`, `right`, `null`) to clear Korean phrases:
   - **Right:** `"찾는 물건이 화면 오른쪽에 보입니다."`
   - **Center:** `"찾는 물건이 화면 가운데에 보입니다."`
   - **Left:** `"찾는 물건이 화면 왼쪽에 보입니다."`
   - **Not Found:** `"현재 카메라 화면에서 찾는 물건이 보이지 않습니다."`
3. **Repetition Control:** Implement a throttle/debounce mechanism to prevent jarring audio repetition when frames are processed continuously.
4. **Fallback Handling:** Provide immediate auditory cues for error states (e.g., camera disconnected, network timeout).

## Inputs & Outputs
- **Input:** Detection JSON payload conforming to [`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md) (specifically `position`, `detected`, and `target`).
- **Output:** Spoken Korean audio via browser speech synthesis or synthesized audio stream.

## Instructions for AI Coding Agents
1. Work within `voice/`.
2. Do not announce physical distance estimates (e.g., "30cm away") unless supported by calibrated hardware/depth sensors; stick to directional cues (left/center/right).
3. Test TTS compatibility across both mobile Safari (iOS) and Chrome (Android).
