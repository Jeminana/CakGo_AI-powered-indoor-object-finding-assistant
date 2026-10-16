"""Evaluate a Grounding DINO run against the shared ground truth in tests/dataset/annotations/.

Scoring follows the rules of detection/experiments/yolo/evaluate.py as of PR #1 (commit d299bb4):
per CakGo category, predictions are matched to that category's ground-truth boxes across ALL
evaluation images (greedy, by confidence, IoU >= --iou-threshold), with the same metrics and
output layout. The scoring functions are copied, not imported, so this folder does not depend
on the YOLO folder; tests/test_gdino_evaluate.py pins the numbers the YOLO evaluator produced
for a test fixture. Every category is supported through its text prompt.

Writes results/<run>/metrics.json and results/<run>/metrics.md.

Example:
  python detection/experiments/grounding_dino/evaluate.py --run <run_name>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from detector import CONFIG_DIR, load_json

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
GROUND_TRUTH_PATH = REPO_ROOT / "tests" / "dataset" / "annotations" / "ground_truth.json"
RESULTS_DIR = EXPERIMENT_DIR / "results"


# iou, match, average_precision and latency_stats are copies of yolo/evaluate.py (commit d299bb4).
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


def score_category(category: str, predictions: dict, gt_images: dict, conf_thr: float, iou_thr: float) -> dict:
    """Per-category metrics, computed exactly like yolo/evaluate.py (label = CakGo category)."""
    target_images = {i for i, g in gt_images.items() if g["target_category"] == category}
    negatives = [i for i, g in gt_images.items() if g["target_category"] == "negative"]
    scored = []
    num_gt = num_pred = tp = image_hits = 0
    for image_id, gt in gt_images.items():
        gts = [o["bbox_xyxy"] for o in gt["objects"] if o["category"] == category]
        preds = sorted((d for d in predictions[image_id]["detections"] if d["label"] == category),
                       key=lambda d: d["confidence"], reverse=True)
        flags = match(preds, gts, iou_thr)
        scored += [(p["confidence"], fl) for p, fl in zip(preds, flags)]
        kept = [fl for p, fl in zip(preds, flags) if p["confidence"] >= conf_thr]
        num_gt += len(gts)
        num_pred += len(kept)
        tp += sum(kept)
        if image_id in target_images and any(kept):
            image_hits += 1
    neg_fp_images = sum(
        any(d["label"] == category and d["confidence"] >= conf_thr for d in predictions[i]["detections"])
        for i in negatives)
    ap = average_precision(scored, num_gt)
    return {
        "status": "evaluated",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="folder name under results/")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="operating confidence (default: box_threshold in config/model.json)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    conf_thr = args.conf_threshold
    if conf_thr is None:
        conf_thr = load_json(CONFIG_DIR / "model.json")["box_threshold"]

    run_dir = RESULTS_DIR / args.run
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        sys.exit(f"{pred_path} not found. Run run_inference.py first.")
    if not GROUND_TRUTH_PATH.exists():
        sys.exit(f"{GROUND_TRUTH_PATH} not found. See tests/dataset/README.md.")

    with open(pred_path, encoding="utf-8") as f:
        predictions = {p["image_id"]: p for p in map(json.loads, f)}
    ground_truth = load_json(GROUND_TRUTH_PATH)
    gt_images = {g["image_id"]: g for g in ground_truth["images"]}
    run_info = load_json(run_dir / "run_info.json")

    missing = set(gt_images) - set(predictions)
    if missing:
        sys.exit(f"{len(missing)} ground-truth images have no prediction (e.g. {sorted(missing)[0]}).")

    per_category = {}
    for category in ground_truth["categories"]:
        if category not in run_info["prompts"]:
            per_category[category] = {"status": "not_queried"}
            continue
        per_category[category] = {"prompt": run_info["prompts"][category],
                                  **score_category(category, predictions, gt_images, conf_thr, args.iou_threshold)}

    query_ms = [ms for p in predictions.values() for ms in p["timing"]["wall_ms"].values()]
    metrics = {
        "run": args.run,
        "model": run_info["model"]["model_id"],
        "thresholds": {"confidence": conf_thr, "iou": args.iou_threshold,
                       "inference_min_confidence": run_info["inference_settings"]["box_threshold"]},
        "notes": [
            "AP50 only uses predictions saved at confidence >= inference_min_confidence "
            "(run_inference.py --box-threshold).",
            "Each image is queried once per category; latency is per query, i.e. per app request.",
        ],
        "num_images": len(gt_images),
        "categories": per_category,
        "latency_ms": {"wall_ms_per_query": latency_stats(query_ms)},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [f"# Grounding DINO evaluation - {args.run}", "",
             f"- Model: `{metrics['model']}` (zero-shot, no fine-tuning)",
             f"- Images: {metrics['num_images']}; confidence >= {conf_thr}; IoU >= {args.iou_threshold}",
             "", "| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 "
                 "| Target images detected | Negative images with FP |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, m in per_category.items():
        if m["status"] != "evaluated":
            lines.append(f"| {c} | {m['status']} | - | - | - | - | - | - | - | - | - |")
        else:
            lines.append(f"| {c} | `{m['prompt']}` | {m['num_gt_boxes']} | {m['true_positives']} | "
                         f"{m['false_positives']} | {m['false_negatives']} | {m['precision']} | {m['recall']} | "
                         f"{m['ap50']} | {m['target_images_detected']}/{m['num_target_images']} | "
                         f"{m['negative_images_with_false_positive']}/{m['negative_images']} |")
    lines += ["", "| Latency (ms) | mean | median | p95 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean']} | {v['median']} | {v['p95']} |" for k, v in metrics["latency_ms"].items()]
    lines += [""] + [f"- {n}" for n in metrics["notes"]]
    with open(run_dir / "metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
