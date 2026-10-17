"""Evaluate a YOLO run against the shared ground truth in tests/dataset/annotations/.

For every CakGo category whose target is SUPPORTED by the model's label set
(see target_mapping.json), predictions of the mapped classes are matched to
ground-truth boxes of that category across ALL evaluation images (greedy, by
confidence, IoU >= --iou-threshold). Unsupported categories are reported as
"unsupported" and are NOT scored as zero.

Writes results/<run>/metrics.json and results/<run>/metrics.md.

Example:
  python detection/experiments/yolo/evaluate.py --run <run_name>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
GROUND_TRUTH_PATH = REPO_ROOT / "tests" / "dataset" / "annotations" / "ground_truth.json"
TARGET_MAPPING_PATH = EXPERIMENT_DIR / "target_mapping.json"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds: list[dict], gts: list[list[float]], iou_thr: float) -> list[bool]:
    """Greedy matching of confidence-sorted preds to GT boxes; returns a TP flag per pred."""
    used = [False] * len(gts)
    flags = []
    for p in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if not used[j]:
                v = iou(p["bbox"], g)
                if v > best:
                    best, best_j = v, j
        if best_j >= 0 and best >= iou_thr:
            used[best_j] = True
            flags.append(True)
        else:
            flags.append(False)
    return flags


def average_precision(scored: list[tuple[float, bool]], num_gt: int) -> float | None:
    """All-point interpolated AP (VOC 2010+ style)."""
    if num_gt == 0:
        return None
    scored = sorted(scored, key=lambda s: s[0], reverse=True)
    tp = fp = 0
    recalls, precisions = [0.0], [1.0]
    for _, is_tp in scored:
        tp += is_tp
        fp += not is_tp
        recalls.append(tp / num_gt)
        precisions.append(tp / (tp + fp))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum((recalls[i] - recalls[i - 1]) * precisions[i] for i in range(1, len(recalls)))


def latency_stats(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(values[min(len(values) - 1, int(0.95 * len(values)))], 2),
        "min": round(values[0], 2),
        "max": round(values[-1], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="folder name under results/")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--mapping", help="mapping file in this folder (default: the one recorded in run_info.json)")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        sys.exit(f"{pred_path} not found. Run run_inference.py first.")
    if not GROUND_TRUTH_PATH.exists():
        sys.exit(f"{GROUND_TRUTH_PATH} not found. See tests/dataset/README.md.")

    with open(pred_path, encoding="utf-8") as f:
        predictions = {p["image_id"]: p for p in map(json.loads, f)}
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        gt_images = {g["image_id"]: g for g in json.load(f)["images"]}
    with open(run_dir / "run_info.json", encoding="utf-8") as f:
        run_info = json.load(f)
    mapping_name = args.mapping or run_info["model"].get("target_mapping", TARGET_MAPPING_PATH.name)
    with open(EXPERIMENT_DIR / mapping_name, encoding="utf-8") as f:
        mapping = json.load(f)["categories"]

    missing = set(gt_images) - set(predictions)
    if missing:
        sys.exit(f"{len(missing)} ground-truth images have no prediction (e.g. {sorted(missing)[0]}).")

    negatives = [i for i, g in gt_images.items() if g["target_category"] == "negative"]
    per_category = {}
    for category, entry in mapping.items():
        target_images = {i for i, g in gt_images.items() if g["target_category"] == category}
        if not entry["supported"]:
            per_category[category] = {"status": "unsupported", "reason": entry["note"],
                                      "num_target_images": len(target_images)}
            continue
        labels = set(entry["coco_classes"])
        scored = []
        num_gt = num_pred = tp = image_hits = 0
        for image_id, gt in gt_images.items():
            gts = [o["bbox_xyxy"] for o in gt["objects"] if o["category"] == category]
            preds = sorted((d for d in predictions[image_id]["detections"] if d["label"] in labels),
                           key=lambda d: d["confidence"], reverse=True)
            flags = match(preds, gts, args.iou_threshold)
            scored += [(p["confidence"], fl) for p, fl in zip(preds, flags)]
            kept = [fl for p, fl in zip(preds, flags) if p["confidence"] >= args.conf_threshold]
            num_gt += len(gts)
            num_pred += len(kept)
            tp += sum(kept)
            if image_id in target_images and any(kept):
                image_hits += 1
        neg_fp_images = sum(
            any(d["label"] in labels and d["confidence"] >= args.conf_threshold
                for d in predictions[i]["detections"]) for i in negatives)
        ap = average_precision(scored, num_gt)
        per_category[category] = {
            "status": "evaluated",
            "model_classes": sorted(labels),
            "num_gt_boxes": num_gt,
            "num_predictions": num_pred,
            "true_positives": tp,
            "false_positives": num_pred - tp,
            "false_negatives": num_gt - tp,
            "precision": round(tp / num_pred, 4) if num_pred else None,
            "recall": round(tp / num_gt, 4) if num_gt else None,
            "ap50": round(ap, 4) if ap is not None else None,
            "num_target_images": len(target_images),
            "target_images_detected": image_hits,
            "image_level_detection_rate": round(image_hits / len(target_images), 4) if target_images else None,
            "negative_images": len(negatives),
            "negative_images_with_false_positive": neg_fp_images,
        }

    timings = [p["timing"] for p in predictions.values()]
    latency = {k: latency_stats([t[k] for t in timings]) for k in timings[0]}

    metrics = {
        "run": args.run,
        "model": run_info["model"]["weights"],
        "fine_tuned": run_info["model"].get("fine_tuned", False),
        "training_run": run_info["model"].get("training_run"),
        "target_mapping": mapping_name,
        "thresholds": {"confidence": args.conf_threshold, "iou": args.iou_threshold,
                       "inference_min_confidence": run_info["inference_settings"]["conf"]},
        "notes": [
            "AP50 only uses predictions saved at confidence >= inference_min_confidence; "
            "re-run inference with --conf 0.001 for an un-truncated AP.",
            "Unsupported categories are not scored; the model has no class for them.",
        ],
        "num_images": len(gt_images),
        "categories": per_category,
        "latency_ms": latency,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [f"# YOLO evaluation - {args.run}", "",
             f"- Model: `{metrics['model']}` "
             + (f"(fine-tuned, training run `{metrics['training_run']}`)" if metrics["fine_tuned"]
                else "(pretrained, no fine-tuning)"),
             f"- Images: {metrics['num_images']}; confidence >= {args.conf_threshold}; IoU >= {args.iou_threshold}",
             "", "| Category | Status | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c, m in per_category.items():
        if m["status"] == "unsupported":
            lines.append(f"| {c} | unsupported | - | - | - | - | - | - | - | - |")
        else:
            lines.append(f"| {c} | evaluated | {m['num_gt_boxes']} | {m['true_positives']} | "
                         f"{m['false_positives']} | {m['false_negatives']} | {m['precision']} | {m['recall']} | "
                         f"{m['ap50']} | {m['target_images_detected']}/{m['num_target_images']} |")
    lines += ["", "| Latency (ms) | mean | median | p95 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean']} | {v['median']} | {v['p95']} |" for k, v in latency.items()]
    lines += [""] + [f"- {n}" for n in metrics["notes"]]
    with open(run_dir / "metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
