# CakGo project documentation

Prepared by **상싱위 / Thehxliday68**, 17 October 2026.

This package covers the documentation task assigned on 17 October: presentation slides, progress, model result analysis and operating information.

## Files

- [Editable project presentation](CakGo_Project_Update.pptx)
- [Progress and team responsibilities](PROGRESS.md)
- [Model results and interpretation](RESULT_ANALYSIS.md)
- [Running the prototype and recording a demo](DEMO_GUIDE.md)
- [Presentation script and source map](PRESENTATION_NOTES.md)

## Current conclusion

The team chose Grounding DINO for initial integration because the pretrained YOLO baseline did not cover three target classes. Fine-tuning has now produced a four-class YOLO model and evaluation records. Its promising results still need real-phone evaluation before a final model decision.

Frontend voice wiring is submitted in [PR #8](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/pull/8), which was open when this package was prepared. The implementation and automatic checks are complete, while phone audio and full real-model checks remain pending.

Evidence snapshot: main commit `c3a4356a875bc4b41b052a4148cbd98133aa77b3`; frontend voice branch commit `c042da5a8684ed6459f88b772617db0e8f2745cb`. This package records a point in time. Later merges and phone-test results should update it.
