# Project progress on 17 October 2026

Documentation owner: **상싱위 / Thehxliday68**.

## Current task ownership

| Task | Owner | Evidence and status |
|---|---|---|
| YOLO fine-tuning | 나즈미 | PR #7 merged. Training metadata, curves and 70-image evaluation files are in main. This establishes a completed training run, not real-phone performance. |
| Smartphone camera and detection display to backend | 김철연 | PR #5 merged. Camera frame upload, same-origin proxy and detection-box display implemented. Real-phone results still need recording. |
| Grounding DINO and Korean voice integration | 오가암 | PR #6 merged. DINO factory and demo adjustment present. Real-device voice plus model acceptance remains unverified in this package. |
| Slides, progress, results analysis and related documentation | 상싱위 | This documentation package and editable slide deck. |

## Additional round

| Task | Owner | Due | Status |
|---|---|---|
| Connect voice to frontend | 상싱위 | 18 October | PR #8 open at preparation. Implementation and automated checks complete. Team review and real-phone audible tests pending. |
| Collect real-phone test photos | Other three teammates, according to team message | 19 October | Planned: 28 per person, 84 total. Upload completion and test results not verified. |

## Completed work supported by repository records

- Smartphone UI and camera workflow: PR #4 merged.
- Backend image API, direction logic and speech helper: PR #3 merged.
- Grounding DINO benchmark: PR #2 merged, with reference YOLO runs on the same Colab VM.
- Frontend API connection: PR #5 merged.
- DINO factory and integration demo: PR #6 merged.
- Four-class YOLO fine-tuning and saved evaluation outputs: PR #7 merged.
- Frontend voice PR #8 includes activation inside the user click, speech for current detection results, mute/unmute, cancellation on stop/target change/page exit and stale-request handling.

PR links: [#2](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/2), [#3](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/3), [#4](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/4), [#5](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/5), [#6](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/6), [#7](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/7), [#8](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/8).

## Validation recorded for PR #8

- Frontend voice wiring checks passed with fake browser devices and actual page/voice scripts.
- Shared speech helper tests passed.
- 13 backend and frontend proxy Python tests passed.
- These checks validate code behavior. They do not prove audible Korean speech on a physical phone or real-model recognition accuracy.
- The coordinator reported that merged components work well. This is a team status report, not a substitute for a dated device test record.

## Remaining work

1. Review and merge PR #8 after checking its changes.
2. Record iOS Safari and Android Chrome results with the selected backend, including permission, speech, target change, stop and network failure.
3. Collect and label the new phone photos without putting private images in public GitHub.
4. Evaluate DINO and fine-tuned YOLO on the exact same new test set with fixed settings.
5. Repeat latency measurements on the same hardware before reporting a direct speed ratio for the fine-tuned model.
6. Review the final model and operating thresholds with the team, then update this package.

The initial meeting reports in `docs/REPORT.md` and `docs/TEAM_ACTIVITY_REPORT.md` remain historical records. The older root README and component ownership labels do not fully reflect the current rounds of work.
