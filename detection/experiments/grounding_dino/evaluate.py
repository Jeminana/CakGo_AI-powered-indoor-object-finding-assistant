"""Benchmark Grounding DINO on a manifest of images (evaluation dataset pending team confirmation).

Manifest: JSON Lines, one sample per line. The format is provisional; align it with Member 1's
YOLO script before the comparison.

    {"image": "images/0001.jpg", "target": "keys", "boxes": [[x1, y1, x2, y2]], "tags": ["dim"]}

- image: path relative to the manifest file (or absolute).
- boxes: ground-truth boxes in original pixels; [] means the target is not in the image.
- tags: optional conditions (e.g. "clutter", "dim", "occluded") that get their own metrics.

Usage:
    python evaluate.py --manifest path/to/manifest.jsonl --output report.json [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from PIL import Image

from detector import CONFIG_DIR, GroundingDinoDetector


def iou(a, b):
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_manifest(path):
    path = Path(path)
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if "image" not in sample or "target" not in sample:
                raise ValueError(f"{path}:{line_no}: each sample needs 'image' and 'target'")
            sample["image"] = str(path.parent / sample["image"])
            samples.append(sample)
    return samples


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def score_sample(sample, result, iou_threshold):
    """Top-1 is what matters: the app guides the user to the highest-confidence box."""
    gt_boxes = sample.get("boxes", [])
    top = result["detections"][0]["bbox"] if result["detections"] else None
    return {
        "positive": bool(gt_boxes),
        "detected": result["detected"],
        "top1_hit": top is not None and any(iou(top, gt) >= iou_threshold for gt in gt_boxes),
    }


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def percentile(sorted_values, q):
    """Nearest-rank percentile."""
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(q / 100 * len(sorted_values)) - 1)]


def summarize(rows):
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    latencies = sorted(row["latency_ms"] for row in rows)
    return {
        "samples": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "detection_rate": ratio(sum(row["detected"] for row in positives), len(positives)),
        "top1_recall": ratio(sum(row["top1_hit"] for row in positives), len(positives)),
        "false_positive_rate": ratio(sum(row["detected"] for row in negatives), len(negatives)),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
    }


def evaluate(detector, samples, iou_threshold=0.5, warmup=1):
    """Run every sample once; latency covers the detect() call only (image decoding excluded)."""
    for sample in samples[:warmup]:
        detector.detect(load_image(sample["image"]), sample["target"])

    rows = []
    for sample in samples:
        image = load_image(sample["image"])
        start = time.perf_counter()
        result = detector.detect(image, sample["target"])
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append(
            {
                "image": sample["image"],
                "target": sample["target"],
                "tags": sample.get("tags", []),
                "latency_ms": round(latency_ms, 2),
                **score_sample(sample, result, iou_threshold),
                "detections": result["detections"],
            }
        )

    tags = sorted({tag for row in rows for tag in row["tags"]})
    return {
        "iou_threshold": iou_threshold,
        "overall": summarize(rows),
        "by_tag": {tag: summarize([row for row in rows if tag in row["tags"]]) for tag in tags},
        "samples": rows,
    }


def environment(detector):
    import torch
    import transformers

    env = {
        "model_id": detector.model_id,
        "device": detector.device,
        "box_threshold": detector.box_threshold,
        "text_threshold": detector.text_threshold,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if str(detector.device).startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(0)
        env["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark Grounding DINO on a JSON Lines manifest.")
    parser.add_argument("--manifest", required=True, help="JSON Lines file (see module docstring)")
    parser.add_argument("--output", required=True, help="where to write the JSON report")
    parser.add_argument("--config-dir", default=str(CONFIG_DIR))
    parser.add_argument("--device", help="override the config device, e.g. cpu or cuda")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs before measuring")
    args = parser.parse_args(argv)

    detector = GroundingDinoDetector.from_config(args.config_dir, device=args.device)
    report = evaluate(detector, load_manifest(args.manifest), args.iou_threshold, args.warmup)
    report = {"environment": environment(detector), **report}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"environment": report["environment"], "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
