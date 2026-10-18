# Presentation script and sources

Presenter: **상싱위 / Thehxliday68**. Language: English. Snapshot: 17 October 2026.

## 1. CakGo

CakGo is a smartphone prototype for finding visible everyday objects with Korean voice guidance. This update covers what we built, what the experiments show and what we still need to validate.

## 2. Target users and scope

Our intended users include people with visual difficulties and people who have trouble locating small belongings. The first version looks for wallets, keys, glasses and TV remotes. It describes screen direction rather than physical distance. We have not yet measured benefit with users.

## 3. System workflow

The phone captures a frame and sends it with the requested target to the FastAPI backend. A detector returns original-image boxes. The backend selects the highest-confidence match for left, center or right guidance. The phone draws boxes and speaks Korean through the existing voice module. No result means the target was not visible in that frame.

## 4. Team progress

The team's camera/backend connection, DINO factory and YOLO fine-tuning work are merged. My additional frontend speech connection is in PR #8 and still awaits review in this snapshot. Its automated checks passed, but physical-phone speech and the complete real-model path still need recorded tests.

## 5. Baseline detector comparison

Pretrained YOLO supports only remote control among the four requested objects. DINO supports all four with prompts. On remote control, DINO has higher AP50 and recall, while its current operating threshold produces many more false positives. That motivated initial DINO integration and parallel YOLO fine-tuning.

## 6. Fine-tuned YOLO

The new four-class model now has saved evaluation results. The mean of its four AP50 values is 0.9434 on the existing 70 images. Wallet remains weaker than the other categories. These are same-source-dataset results, so we need the new phone photos before making a final model choice.

## 7. False detections

In ten target-free images, DINO at threshold 0.35 gives false alerts on four wallet images, nine keys images, six glasses images and nine remote-control images. Fine-tuned YOLO at 0.25 gives zero on that small negative subset. This does not mean it never makes mistakes. Repeating speech less often does not make a wrong box correct.

## 8. Request time

The original baseline comparison ran both models on the same T4. YOLO took about 15 milliseconds and DINO about 334 milliseconds per request. Fine-tuned YOLO took about 15 milliseconds on a different RTX 4050 laptop. We should not use those different machines to claim a direct fine-tuning speedup, and none of these numbers is phone end-to-end latency.

## 9. Real-phone test plan

The next collection contains 84 planned photos from three members, with two designs per item and varied conditions. These are test photos only. We will preserve their separation from training and evaluate false positives, detection quality and phone operation. The collection is due 19 October.

## 10. Next milestones

The voice wiring deadline is 18 October and the implementation is ready for review. We next need phone audio checks, collection completion, reviewed annotations and identical candidate settings for evaluation. The team will then decide the model and thresholds from the evidence.

## 11. Training evidence appendix

The plot comes from the saved 60-epoch training run. Its validation curves describe the training validation split, not the 70-image test or the forthcoming phone photos. The preparation script removes shared evaluation photos and duplicates. That helps avoid leakage but does not eliminate the need for a new environment test.

## Questions to prepare for

**Why not immediately switch to fine-tuned YOLO?** Its existing-set results are promising, but real-phone generalization and a final team decision remain outstanding.

**Does zero false positives mean safe operation?** No. The negative subset contains only ten images, and wallet already has false boxes elsewhere in the test. We need broader negative scenes and application-level tests.

**Is the app real time?** The frontend sends periodic images. Model timings alone do not establish video FPS or camera-to-speech latency.

**Is the voice task finished?** Code wiring and automated validation are complete in PR #8. Merge and device audio acceptance remain pending at this snapshot.

## Slide source map

Repository base: https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/blob/c3a4356a875bc4b41b052a4148cbd98133aa77b3/

- Slides 1–3: root README and docs/API_CONTRACT.md.
- Slide 4: merged PRs #3–#7, PR #8 state, and team task messages supplied on 17 October.
- Slides 5, 7, 8: tests/results/model_comparison_report.md and the respective JSON experiment metrics/run_info records.
- Slide 6: detection/experiments/yolo/results/ft_yolo26s_cakgo4_v1_conf0.001/metrics_conf0.25.json and training_runs/yolo26s_cakgo4_v1/training_info.json.
- Slides 9–10: team photo instructions supplied on 17 October and PR #8.
- Slide 11: detection/experiments/yolo/training_runs/yolo26s_cakgo4_v1/results.png and training_info.json.

Exact source links and speaking notes are also embedded in each slide's speaker notes. The live prototype is not represented by a staged screenshot in this presentation.
