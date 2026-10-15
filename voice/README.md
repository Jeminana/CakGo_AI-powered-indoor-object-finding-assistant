# Voice Guidance — Member 3

## Objective
Make CakGo usable without reading bounding boxes by giving concise Korean spoken directions.

## Tasks
- Day 1: Test Korean text-to-speech on actual phone/browser.
- Day 2: Convert mock detection JSON into Korean voice messages.
- Day 3: Integrate actual backend response.
- Days 4–5: Prevent incessant repetition; handle not-found/error states.
- Days 6–7: Test clarity and accessibility.

## Messages
- Right: "찾는 물건이 화면 오른쪽에 보입니다."
- Center: "찾는 물건이 화면 가운데에 보입니다."
- Missing: "현재 카메라 화면에서 찾는 물건이 보이지 않습니다."

Avoid announcing physical distances that have not been measured. Object selection buttons are the fallback if speech recognition is unavailable. Read `../docs/API_CONTRACT.md` and test with mock JSON first.
