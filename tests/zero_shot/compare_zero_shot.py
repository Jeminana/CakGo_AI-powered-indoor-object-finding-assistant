"""Score the zero-shot runs on the 70 shared images and print the comparison tables.

Scoring is the Day 3 protocol, reused rather than copied: grounding_dino/evaluate.py `score_category`
(pinned by its tests to the YOLO evaluator) and tests/compare_models.py `top1_rate`, `GT_ISSUES` and
`THRESHOLDS`. Each model is read at its own operating threshold from run_info.json (fixed before the
run), and at the common thresholds of the sweep.

Writes metrics.json / metrics.md into every run folder and tests/results/zero_shot_70/tables.md.

Example (from the repository root):
  python tests/zero_shot/compare_zero_shot.py --runs t4_gdino-tiny_multi t4_yoloe-26s_single ... \
      --reference "Grounding DINO single (Day 3, T4)=detection/experiments/grounding_dino/results/t4_gdino-tiny_box0.05" \
      --reproduce t4_gdino-tiny_multi=detection/experiments/grounding_dino/results/rtx4050_gdino-tiny_multi_box0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "tests" / "results" / "zero_shot_70"
GROUND_TRUTH_PATH = REPO_ROOT / "tests" / "dataset" / "annotations" / "ground_truth.json"
sys.path.insert(0, str(REPO_ROOT / "tests"))
from compare_models import GT_ISSUES, THRESHOLDS, fmt, metric_rows, read_jsonl, top1_rate  # noqa: E402
from evaluate import latency_stats, score_category  # noqa: E402  (grounding_dino/evaluate.py)

DEFAULT_OPERATING = 0.35  # reference runs without an operating_threshold are Grounding DINO runs


def load_run(path: Path) -> dict:
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    settings = info["inference_settings"]
    return {"path": path, "info": info, "predictions": read_jsonl(path / "predictions.jsonl"),
            "threshold": settings.get("operating_threshold", DEFAULT_OPERATING)}


def label(run: dict) -> str:
    """Short model label, e.g. 'YOLOE-26s single'."""
    info = run["info"]
    model_id = info["model"]["model_id"]
    mode = info["inference_settings"]["prompt_mode"]
    if "grounding-dino" in model_id:
        name = "Grounding DINO tiny"
    elif "sam3" in model_id:
        name = "SAM 3"
    else:
        name = Path(model_id).stem.replace("yoloe-", "YOLOE-").replace("-seg", "")
    return f"{name} {mode}"


def false_alarms(predictions: dict, gt_images: dict, categories: list, thr: float) -> tuple[int, int]:
    """Queries for a category that the image has no box of, answered with any box >= thr."""
    hits = total = 0
    for image_id, gt in gt_images.items():
        present = {o["category"] for o in gt["objects"]}
        for c in categories:
            if c not in present:
                total += 1
                hits += any(d["label"] == c and d["confidence"] >= thr for d in predictions[image_id]["detections"])
    return hits, total


def top1_centre_hits(category: str, predictions: dict, gt_images: dict, thr: float) -> tuple[int, int]:
    """Diagnostic, not the protocol: on target images, is the centre of the highest-confidence box (>= thr)
    inside a labeled box of the category? The app's left/centre/right guidance uses the box centre, and this
    does not penalise a box on one key of a labeled key bunch."""
    targets = [i for i, g in gt_images.items() if g["target_category"] == category]
    hits = 0
    for i in targets:
        boxes = [d for d in predictions[i]["detections"] if d["label"] == category and d["confidence"] >= thr]
        if boxes:
            b = max(boxes, key=lambda d: d["confidence"])["bbox"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            gts = [o["bbox_xyxy"] for o in gt_images[i]["objects"] if o["category"] == category]
            hits += any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in gts)
    return hits, len(targets)


def summary(run: dict, gt_images: dict, categories: list, thr: float) -> dict:
    preds = run["predictions"]
    scores = [score_category(c, preds, gt_images, thr, 0.5) for c in categories]
    top1 = [list(map(int, top1_rate(c, preds, gt_images, thr).split("/"))) for c in categories]
    fa, fa_total = false_alarms(preds, gt_images, categories, thr)
    aps = [s["ap50"] for s in scores if s["ap50"] is not None]
    return {
        "threshold": thr,
        "top1": (sum(h for h, _ in top1), sum(n for _, n in top1)),
        "detected": (sum(s["target_images_detected"] for s in scores), sum(s["num_target_images"] for s in scores)),
        "negatives_fp": (sum(s["negative_images_with_false_positive"] for s in scores),
                         sum(s["negative_images"] for s in scores)),
        "false_alarms": (fa, fa_total),
        "mean_ap50": round(sum(aps) / len(aps), 3) if aps else None,
    }


def ratio(pair: tuple[int, int]) -> str:
    return f"{pair[0]}/{pair[1]}"


def write_metrics(run: dict, gt_images: dict, categories: list) -> None:
    """metrics.json / metrics.md in the layout of grounding_dino/evaluate.py, at the run's operating threshold."""
    thr, preds, info = run["threshold"], run["predictions"], run["info"]
    per_category = {c: {"prompt": info["prompts"][c], **score_category(c, preds, gt_images, thr, 0.5)}
                    for c in categories}
    query_ms = [ms for p in preds.values() for ms in p["timing"]["wall_ms"].values()]
    metrics = {"run": run["path"].name, "model": info["model"]["model_id"],
               "thresholds": {"confidence": thr, "iou": 0.5,
                              "inference_min_confidence": info["inference_settings"]["box_threshold"]},
               "num_images": len(gt_images), "categories": per_category,
               "latency_ms": {"wall_ms_per_query": latency_stats(query_ms)}}
    (run["path"] / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = [f"# Zero-shot evaluation - {run['path'].name}", "",
             f"- Model: `{metrics['model']}` ({info['model']['family']})",
             f"- Images: {len(gt_images)}; confidence >= {thr}; IoU >= 0.5", "",
             "| Category | Prompt | GT boxes | TP | FP | FN | Precision | Recall | AP50 | Target images detected "
             "| Negative images with FP |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, m in per_category.items():
        lines.append(f"| {c} | `{m['prompt']}` | {m['num_gt_boxes']} | {m['true_positives']} | {m['false_positives']} "
                     f"| {m['false_negatives']} | {m['precision']} | {m['recall']} | {m['ap50']} | "
                     f"{m['target_images_detected']}/{m['num_target_images']} | "
                     f"{m['negative_images_with_false_positive']}/{m['negative_images']} |")
    s = metrics["latency_ms"]["wall_ms_per_query"]
    lines += ["", "| Latency (ms) | mean | median | p95 |", "|---|---|---|---|",
              f"| wall_ms_per_query | {s['mean']} | {s['median']} | {s['p95']} |"]
    (run["path"] / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="run folders under tests/results/zero_shot_70/")
    parser.add_argument("--reference", action="append", default=[], metavar="LABEL=PATH",
                        help="earlier run shown for comparison (not re-scored into its folder)")
    parser.add_argument("--reproduce", action="append", default=[], metavar="RUN=REFERENCE_PATH",
                        help="check that RUN gives the same TP/FP/FN as an earlier run of the same model")
    args = parser.parse_args()

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    categories = ground_truth["categories"]
    gt_all = {g["image_id"]: g for g in ground_truth["images"]}
    gt_clean = {i: g for i, g in gt_all.items() if i not in GT_ISSUES}

    runs = []
    for name in args.runs:
        run = load_run(RESULTS_DIR / name)
        run["label"] = label(run)
        write_metrics(run, gt_all, categories)
        runs.append(run)
    references = []
    for item in args.reference:
        ref_label, path = item.split("=", 1)
        run = load_run(REPO_ROOT / path)
        run["label"] = ref_label
        references.append(run)
    everything = runs + references

    out = []
    manifests = {r["info"]["dataset"]["manifest_sha256"] for r in everything}
    gpus = sorted({str(r["info"]["environment"].get("gpu")) for r in runs})
    out += [f"<!-- manifest sha256: {', '.join(sorted(manifests))}; GPUs of new runs: {', '.join(gpus)} -->", ""]

    if args.reproduce:
        out += ["### Reproduction check (same model, earlier run, each at 0.35)", "",
                "| Run | Earlier run | Category | TP / FP / FN now | TP / FP / FN earlier | AP50 now / earlier |",
                "|---|---|---|---|---|---|"]
        for item in args.reproduce:
            name, path = item.split("=", 1)
            now, before = load_run(RESULTS_DIR / name), load_run(REPO_ROOT / path)
            for c in categories:
                a = score_category(c, now["predictions"], gt_all, 0.35, 0.5)
                b = score_category(c, before["predictions"], gt_all, 0.35, 0.5)
                out.append(f"| {name} | {Path(path).name} | {c} | {a['true_positives']} / {a['false_positives']} / "
                           f"{a['false_negatives']} | {b['true_positives']} / {b['false_positives']} / "
                           f"{b['false_negatives']} | {fmt(a['ap50'])} / {fmt(b['ap50'])} |")
        out.append("")

    header = ["| Model | Conf | Top-1 correct | Target images detected | False alarms | Negatives with FP | "
              "Mean AP50 | Median ms / request | p95 ms | Peak VRAM (MB) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for title, gt in (("App view at each model's operating threshold (all 70 images)", gt_all),
                      (f"Same, excluding {len(GT_ISSUES)} images with unlabeled targets", gt_clean)):
        out += [f"### {title}", ""] + header
        for r in everything:
            s = summary(r, gt, categories, r["threshold"])
            lat = latency_stats([ms for p in r["predictions"].values() for ms in p["timing"]["wall_ms"].values()])
            out.append(f"| {r['label']} | {r['threshold']} | {ratio(s['top1'])} | {ratio(s['detected'])} | "
                       f"{ratio(s['false_alarms'])} | {ratio(s['negatives_fp'])} | {fmt(s['mean_ap50'])} | "
                       f"{lat['median']} | {lat['p95']} | {r['info']['environment'].get('peak_vram_mb') or '-'} |")
        out.append("")

    out += ["### Common thresholds (all 70 images): top-1 correct / false alarms", "",
            "| Model | " + " | ".join(str(t) for t in THRESHOLDS) + " |", "|---|" + "---|" * len(THRESHOLDS)]
    for r in everything:
        cells = []
        for t in THRESHOLDS:
            s = summary(r, gt_all, categories, t)
            cells.append(f"{s['top1'][0]} / {s['false_alarms'][0]}")
        out.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    fa_total = false_alarms(everything[0]["predictions"], gt_all, categories, 1.0)[1]
    out += ["", f"Top-1 correct is out of 60 target images; false alarms are out of {fa_total} queries for an "
                "object the image has no box of.", ""]

    out += ["### Diagnostic: top box correct (IoU >= 0.5) / top box centre on a labeled object", "",
            "Not part of the protocol. Each cell: target images out of 15, at the model's operating threshold.", "",
            "| Model | " + " | ".join(categories) + " |", "|---|" + "---|" * len(categories)]
    for r in everything:
        cells = []
        for c in categories:
            strict = top1_rate(c, r["predictions"], gt_all, r["threshold"]).split("/")[0]
            centre = top1_centre_hits(c, r["predictions"], gt_all, r["threshold"])[0]
            cells.append(f"{strict} / {centre}")
        out.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["### Per category at each model's operating threshold (all 70 images)", "",
            "| Category | Model | Conf | AP50 | Precision | Recall | Target images detected | Top-1 correct "
            "| Negatives with FP |", "|---|---|---|---|---|---|---|---|---|"]
    for c in categories:
        for r in everything:
            out += [row for row in metric_rows(r["label"], {}, r["predictions"], gt_all, [c], r["threshold"])]
    out.append("")

    text = "\n".join(out)
    (RESULTS_DIR / "tables.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
