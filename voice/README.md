# Voice Guidance Module — Member 3

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
