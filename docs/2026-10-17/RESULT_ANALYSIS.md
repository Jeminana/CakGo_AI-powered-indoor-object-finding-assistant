# Model result analysis

Prepared by **상싱위 / Thehxliday68**, 17 October 2026. Values below summarize committed experiment files, not new experiments performed for this document.

## Question and current decision

CakGo needs to find a requested wallet, keys, glasses or TV remote in a phone image and provide useful direction guidance. A false positive can cause the app to announce a position for the wrong object. Missing a target also reduces usefulness, so both precision and recall matter.

The team initially selected Grounding DINO for integration and chose to fine-tune YOLO in parallel. The idea that fine-tuning would require less work than correcting DINO was a planning judgment. Fine-tuning results now exist, but the final deployment model still needs a team decision informed by real-phone testing.

## Evaluation setup

The shared evaluation contains 70 images, 15 assigned target images per category and 10 target-free images. All runs reference manifest SHA256 `ca73e2c1f12464df03b1f2b3824109cd237ea1e9dbba46fa8d5ac4c6db5bde4b`. A target image can contain multiple annotated objects, so object recall and image detection rate have different denominators.

- Correct box matching: IoU at least 0.5.
- Baseline YOLO26s: COCO pretrained, no exact class for wallet, keys or glasses. Those categories are unsupported, not 0% accurate.
- Grounding DINO tiny: text prompts, confidence threshold 0.35 in the operating comparison.
- Fine-tuned YOLO26s: four CakGo classes, confidence threshold 0.25 in the operating comparison.
- AP50 uses the saved low-confidence predictions across thresholds. It is distinct from precision/recall at one selected threshold. YOLO predictions start at 0.001 and DINO at 0.05, so the AP tails have different truncation.

## Baseline findings

For remote control, the one category both baseline models support:

| Model | Confidence | AP50 | Precision | Recall | Target-free images with a false positive |
|---|---:|---:|---:|---:|---:|
| Pretrained YOLO26s | 0.25 | 0.870 | 1.000 | 0.833 | 0/10 |
| Grounding DINO tiny | 0.35 | 0.941 | 0.198 | 1.000 | 9/10 |

The baseline DINO detector covers more categories and misses fewer annotated remote-control boxes at this threshold. Its additional false detections reduce precision substantially. A higher AP50 does not imply that the current threshold is reliable for spoken guidance.

The baseline YOLO figures are reference runs made by the DINO evaluator using the YOLO owner's unmodified scripts, as disclosed in the original comparison report.

## Fine-tuned YOLO results

Source: `detection/experiments/yolo/results/ft_yolo26s_cakgo4_v1_conf0.001/metrics_conf0.25.json`. Using this explicit threshold file avoids ambiguity in the generic `metrics.json`, which currently reports confidence 0.5 despite its run name.

| Category | TP | FP | FN | Precision | Recall | AP50 | Target images detected | Target-free images with FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wallet | 11 | 5 | 6 | 0.6875 | 0.6471 | 0.8057 | 10/15 | 0/10 |
| Keys | 31 | 1 | 1 | 0.9688 | 0.9688 | 0.9678 | 14/15 | 0/10 |
| Glasses | 17 | 0 | 1 | 1.0000 | 0.9444 | 1.0000 | 14/15 | 0/10 |
| Remote control | 17 | 0 | 1 | 1.0000 | 0.9444 | 1.0000 | 14/15 | 0/10 |

The arithmetic mean AP50 is **0.9434** across the four categories, calculated from those AP50 values. This is not mAP@0.5:0.95. Wallet remains the main weakness. Zero false alarms on only 10 target-free images does not establish a zero false-alarm rate in real use. Wallet also has five false boxes elsewhere in the full evaluation, even though it has no false boxes in the ten target-free images.

Training metadata records 60 configured epochs, batch 16, image size 640 and seed 0. `results.csv` contains epoch 60. After filtering, the dataset contains 6,514 training and 1,536 validation images, with shared evaluation images and duplicate/similar photos excluded by the data preparation script. Both training and evaluation still come from the same source collection, which limits generalization evidence.

## Grounding DINO errors and threshold trade-off

| Category | Precision at 0.35 | Recall at 0.35 | Target-free FP images at 0.35 | Target-free FP images at 0.5 |
|---|---:|---:|---:|---:|
| Wallet | 0.2727 | 0.8824 | 4/10 | 2/10 |
| Keys | 0.2286 | 1.0000 | 9/10 | 7/10 |
| Glasses | 0.2576 | 0.9444 | 6/10 | 2/10 |
| Remote control | 0.1978 | 1.0000 | 9/10 | 2/10 |

Raising the threshold reduces some false alarms but does not resolve the keys case. The original report finds that many false boxes cover another labeled object. Consecutive-frame confirmation and revised prompts are candidate mitigations, not proven improvements in this package. The speech helper's repetition cooldown only reduces repeated audio. It does not validate a detection or eliminate false positives.

## Latency and comparison limits

| Model and run | Hardware | Mean time per model request | Interpretation |
|---|---|---:|---|
| Pretrained YOLO26s, confidence 0.25 | Colab T4 | 15.25 ms | Matched baseline VM |
| Grounding DINO tiny | Same Colab T4 | 333.88 ms | One requested category per call |
| Fine-tuned YOLO26s, confidence 0.25 | RTX 4050 Laptop GPU | 15.22 ms | Different machine and software environment |

The matched baseline T4 comparison gives approximately 21.9 times the model request time for DINO versus pretrained YOLO. Do not attribute this ratio to the fine-tuned model. The fine-tuned run uses Windows, Python 3.12.10 and a laptop RTX 4050; the DINO run uses the earlier Colab T4 environment. Timings exclude image file reading and do not measure camera capture, upload, response processing or speech startup. Reciprocal latency is not the phone application's measured FPS.

## Interpretation and next experiment

Fine-tuning resolves the baseline YOLO class-coverage gap and yields promising precision and AP50 on the source-dataset test. DINO retains higher wallet recall at its selected operating threshold. No single number determines the best model for users.

Evaluate both candidates on the forthcoming 84 real-phone photos without training on them. Report per-category TP/FP/FN, AP50, target-image detection, false alarms on target-free images, and correct highest-confidence guidance. Record latency on the same deployment machine. Fix thresholds using separate validation data before inspecting final phone-test outcomes.

## Limitations to include in the presentation

- Small existing evaluation: 15 target images per category, 10 target-free images.
- Community annotations have missing labels. Twelve possible cases came from an AI prescreen, not a completed human audit.
- Existing images include stock-style photos and resized 640-by-640 images, not the new phone scenes.
- Different model thresholds are operational settings, not equivalent calibrated probabilities.
- Fine-tuned model timing and DINO timing come from different hardware.
- Successful automated API/voice checks do not establish real-device audio or user benefit.

## Sources

All repository paths below refer to main commit `c3a4356a875bc4b41b052a4148cbd98133aa77b3`.

- [Original comparison report](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/blob/c3a4356a875bc4b41b052a4148cbd98133aa77b3/tests/results/model_comparison_report.md)
- [Fine-tuned YOLO metrics at confidence 0.25](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/blob/c3a4356a875bc4b41b052a4148cbd98133aa77b3/detection/experiments/yolo/results/ft_yolo26s_cakgo4_v1_conf0.001/metrics_conf0.25.json)
- [Fine-tuned latency run metadata](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/blob/c3a4356a875bc4b41b052a4148cbd98133aa77b3/detection/experiments/yolo/results/ft_yolo26s_cakgo4_v1_conf0.25/run_info.json)
- [Training metadata](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/blob/c3a4356a875bc4b41b052a4148cbd98133aa77b3/detection/experiments/yolo/training_runs/yolo26s_cakgo4_v1/training_info.json)
- [DINO operating metrics](https://github.com/Jeminana/CakGo_AI-powered-indoor-object-finding-assistant/blob/c3a4356a875bc4b41b052a4148cbd98133aa77b3/detection/experiments/grounding_dino/results/t4_gdino-tiny_box0.05/metrics.json)
- Team messages supplied on 17 October establish the current task ownership, initial integration decision and new phone-photo plan.
