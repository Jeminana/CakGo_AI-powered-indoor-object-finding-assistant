"""Compare the YOLO and Grounding DINO runs on the shared evaluation set (Day 3 benchmark).

Both runs are scored with the same function (grounding_dino/evaluate.py `score_category`, whose
numbers tests/test_gdino_evaluate.py pins to what yolo/evaluate.py produced), and the YOLO
numbers are cross-checked against the YOLO run's own metrics.json. Prints the Markdown tables
used in tests/results/model_comparison_report.md.

YOLO run folders and the COCO-to-CakGo label mapping are read from tests/results/yolo_reference_runs/
by default (Member 4's reference runs of Member 1's scripts, plus a copy of target_mapping.json from
PR #1). Nothing under detection/experiments/yolo/ is imported or read; pass --yolo-results-dir and
--yolo-mapping to compare other YOLO runs.

Example (from the repository root):
  python tests/compare_models.py --yolo-run t4_yolo26s_conf0.001 --yolo-latency-run t4_yolo26s_conf0.25 \
      --gdino-run t4_gdino-tiny_box0.05 --yolo-cpu-run colabcpu_yolo26s_latency-sample \
      --gdino-cpu-run colabcpu_gdino-tiny_latency-sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GDINO_DIR = REPO_ROOT / "detection" / "experiments" / "grounding_dino"
GROUND_TRUTH_PATH = REPO_ROOT / "tests" / "dataset" / "annotations" / "ground_truth.json"
YOLO_REFERENCE_RUNS = REPO_ROOT / "tests" / "results" / "yolo_reference_runs"
sys.path.insert(0, str(GDINO_DIR))
from evaluate import iou, latency_stats, match, score_category  # noqa: E402

THRESHOLDS = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
# Visible targets that the source annotations miss (AI pre-screen of contact sheets on 2026-10-16,
# not a human review; manifest.csv review_status is still "unreviewed").
GT_ISSUES = {
    "keys_001": "unlabeled sunglasses",
    "keys_003": "unlabeled sunglasses",
    "keys_004": "unlabeled sunglasses",
    "keys_007": "unlabeled sunglasses",
    "keys_008": "unlabeled glasses",
    "keys_009": "unlabeled sunglasses",
    "keys_012": "unlabeled sunglasses",
    "keys_013": "unlabeled sunglasses",
    "keys_015": "unlabeled wallet",
    "wallet_003": "unlabeled keys",
    "wallet_006": "second wallet unlabeled",
    "wallet_007": "second wallet unlabeled",
}


def read_jsonl(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {p["image_id"]: p for p in map(json.loads, f)}


def yolo_predictions(run_dir: Path, mapping_path: Path) -> tuple[dict, dict]:
    """YOLO detections relabelled to CakGo categories (supported categories only)."""
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["categories"]
    to_category = {label: c for c, m in mapping.items() if m["supported"] for label in m["coco_classes"]}
    preds = read_jsonl(run_dir / "predictions.jsonl")
    for p in preds.values():
        p["detections"] = [{**d, "label": to_category[d["label"]]} for d in p["detections"]
                           if d["label"] in to_category]
    return preds, {c: m["supported"] for c, m in mapping.items()}


def top1_rate(category: str, predictions: dict, gt_images: dict, thr: float, iou_thr: float = 0.5):
    """App view: on target images, is the highest-confidence box (>= thr) a correct box?"""
    targets = [i for i, g in gt_images.items() if g["target_category"] == category]
    hits = 0
    for i in targets:
        preds = [d for d in predictions[i]["detections"] if d["label"] == category and d["confidence"] >= thr]
        if preds:
            top = max(preds, key=lambda d: d["confidence"])
            gts = [o["bbox_xyxy"] for o in gt_images[i]["objects"] if o["category"] == category]
            hits += any(iou(top["bbox"], g) >= iou_thr for g in gts)
    return f"{hits}/{len(targets)}"


def false_positive_kinds(category: str, predictions: dict, gt_images: dict, thr: float) -> dict:
    """Where do the unmatched boxes (>= thr) land? Diagnostic only, not part of the protocol."""
    kinds = {"near_gt_box": 0, "near_full_frame": 0, "on_other_labeled_object": 0, "elsewhere": 0}
    for image_id, gt in gt_images.items():
        gts = [o["bbox_xyxy"] for o in gt["objects"] if o["category"] == category]
        others = [o["bbox_xyxy"] for o in gt["objects"] if o["category"] != category]
        preds = sorted((d for d in predictions[image_id]["detections"] if d["label"] == category),
                       key=lambda d: d["confidence"], reverse=True)
        for p, is_tp in zip(preds, match(preds, gts, 0.5)):
            if is_tp or p["confidence"] < thr:
                continue
            b = p["bbox"]
            if any(iou(b, g) > 0.3 for g in gts):
                kinds["near_gt_box"] += 1
            elif (b[2] - b[0]) * (b[3] - b[1]) > 0.8 * gt["width"] * gt["height"]:
                kinds["near_full_frame"] += 1
            elif any(iou(b, o) > 0.3 for o in others):
                kinds["on_other_labeled_object"] += 1
            else:
                kinds["elsewhere"] += 1
    return kinds


def top_scores(category: str, predictions: dict, gt_images: dict, bucket: str) -> list[float]:
    """Highest confidence for the category on each image of a bucket (0 when nothing was saved)."""
    return sorted(max([d["confidence"] for d in predictions[i]["detections"] if d["label"] == category] or [0.0])
                  for i, g in gt_images.items() if g["target_category"] == bucket)


def fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def metric_rows(name, supported, predictions, gt_images, categories, thr):
    rows = []
    for c in categories:
        if not supported.get(c, True):
            rows.append(f"| {c} | {name} | unsupported | - | - | - | - | - | - |")
            continue
        m = score_category(c, predictions, gt_images, thr, 0.5)
        rows.append(f"| {c} | {name} | {thr} | {fmt(m['ap50'])} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                    f"{m['target_images_detected']}/{m['num_target_images']} | "
                    f"{top1_rate(c, predictions, gt_images, thr)} | "
                    f"{m['negative_images_with_false_positive']}/{m['negative_images']} |")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yolo-run", required=True, help="YOLO run for accuracy (ideally --conf 0.001)")
    parser.add_argument("--yolo-latency-run", required=True, help="YOLO run with default settings, for latency")
    parser.add_argument("--gdino-run", required=True)
    parser.add_argument("--yolo-cpu-run")
    parser.add_argument("--gdino-cpu-run")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO operating confidence")
    parser.add_argument("--gdino-conf", type=float, default=0.35, help="Grounding DINO operating confidence")
    parser.add_argument("--yolo-results-dir", type=Path, default=YOLO_REFERENCE_RUNS,
                        help="folder that holds the YOLO run folders (default: tests/results/yolo_reference_runs)")
    parser.add_argument("--yolo-mapping", type=Path,
                        help="YOLO target_mapping.json (default: the one inside --yolo-results-dir)")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    categories = ground_truth["categories"]
    gt_all = {g["image_id"]: g for g in ground_truth["images"]}
    gt_clean = {i: g for i, g in gt_all.items() if i not in GT_ISSUES}
    yolo, yolo_supported = yolo_predictions(args.yolo_results_dir / args.yolo_run,
                                            args.yolo_mapping or args.yolo_results_dir / "target_mapping.json")
    gdino = read_jsonl(GDINO_DIR / "results" / args.gdino_run / "predictions.jsonl")

    # Cross-check: our scoring of the YOLO run must equal the YOLO run's own metrics.json.
    own = json.loads((args.yolo_results_dir / args.yolo_run / "metrics.json").read_text(encoding="utf-8"))
    own_conf = own["thresholds"]["confidence"]
    for c, m in own["categories"].items():
        if m["status"] == "evaluated":
            ours = score_category(c, yolo, gt_all, own_conf, own["thresholds"]["iou"])
            assert all(ours[k] == m[k] for k in ours), f"YOLO cross-check failed for {c}"
    print(f"<!-- cross-check: YOLO metrics.json reproduced at conf {own_conf} -->\n")

    header = ["| Category | Model | Conf | AP50 | Precision | Recall | Target images detected "
              "| Top-1 correct | Negatives with FP |", "|---|---|---|---|---|---|---|---|---|"]
    for title, gt in (("All 70 images (official protocol)", gt_all),
                      (f"Excluding {len(GT_ISSUES)} images with unlabeled targets (sensitivity check)", gt_clean)):
        print(f"### {title}\n")
        print("\n".join(header + metric_rows("YOLO26s", yolo_supported, yolo, gt, categories, args.yolo_conf)
                        + metric_rows("Grounding DINO tiny", {}, gdino, gt, categories, args.gdino_conf)))
        print()

    print("### Confidence sweep (all 70 images): precision / recall / negatives with FP\n")
    print("| Model | Category | " + " | ".join(str(t) for t in THRESHOLDS) + " |")
    print("|---|---|" + "---|" * len(THRESHOLDS))
    for name, preds, supported in (("YOLO26s", yolo, yolo_supported), ("Grounding DINO tiny", gdino, {})):
        for c in categories:
            if not supported.get(c, True):
                continue
            cells = []
            for t in THRESHOLDS:
                m = score_category(c, preds, gt_all, t, 0.5)
                cells.append(f"{fmt(m['precision'])} / {fmt(m['recall'])} / "
                             f"{m['negative_images_with_false_positive']}")
            print(f"| {name} | {c} | " + " | ".join(cells) + " |")
    print()

    print("### App view sweep (all 70 images): top-1 correct on target images / negatives with any box\n")
    print("| Model | Category | " + " | ".join(str(t) for t in THRESHOLDS) + " |")
    print("|---|---|" + "---|" * len(THRESHOLDS))
    for name, preds, supported in (("YOLO26s", yolo, yolo_supported), ("Grounding DINO tiny", gdino, {})):
        for c in categories:
            if not supported.get(c, True):
                continue
            cells = []
            for t in THRESHOLDS:
                m = score_category(c, preds, gt_all, t, 0.5)
                cells.append(f"{top1_rate(c, preds, gt_all, t)} / {m['negative_images_with_false_positive']}")
            print(f"| {name} | {c} | " + " | ".join(cells) + " |")
    print()

    print(f"### Grounding DINO diagnostics (conf >= {args.gdino_conf}, all 70 images)\n")
    print("| Category | False boxes | Near a GT box | Near full frame | On another labeled object | Elsewhere "
          "| Top score on target images (min / median) | Top score on negatives (max / median) |")
    print("|---|---|---|---|---|---|---|---|")
    for c in categories:
        k = false_positive_kinds(c, gdino, gt_all, args.gdino_conf)
        tgt, neg = top_scores(c, gdino, gt_all, c), top_scores(c, gdino, gt_all, "negative")
        print(f"| {c} | {sum(k.values())} | {k['near_gt_box']} | {k['near_full_frame']} | "
              f"{k['on_other_labeled_object']} | {k['elsewhere']} | {tgt[0]:.2f} / {tgt[len(tgt) // 2]:.2f} | "
              f"{neg[-1]:.2f} / {neg[len(neg) // 2]:.2f} |")
    print()

    print("### Latency per request (ms)\n")
    print("| Model | Hardware | Measured | mean | median | p95 | min | max |")
    print("|---|---|---|---|---|---|---|---|")
    runs = [("YOLO26s", "T4 GPU", args.yolo_results_dir, args.yolo_latency_run),
            ("Grounding DINO tiny", "T4 GPU", GDINO_DIR / "results", args.gdino_run),
            ("YOLO26s", "Colab CPU", args.yolo_results_dir, args.yolo_cpu_run),
            ("Grounding DINO tiny", "Colab CPU", GDINO_DIR / "results", args.gdino_cpu_run)]
    for name, hardware, results_dir, run in runs:
        if not run:
            continue
        preds = read_jsonl(results_dir / run / "predictions.jsonl")
        values = [v for p in preds.values() for v in
                  (p["timing"]["wall_ms"].values() if isinstance(p["timing"]["wall_ms"], dict)
                   else [p["timing"]["wall_ms"]])]
        s = latency_stats(values)
        print(f"| {name} | {hardware} | {len(values)} requests (`{run}`) | {s['mean']} | {s['median']} | "
              f"{s['p95']} | {s['min']} | {s['max']} |")


if __name__ == "__main__":
    main()
