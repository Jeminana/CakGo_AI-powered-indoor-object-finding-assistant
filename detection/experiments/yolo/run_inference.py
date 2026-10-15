"""Run pretrained YOLO on the shared evaluation set (tests/dataset/manifest.csv).

Saves, under results/<run_name>/:
  predictions.jsonl  one line per image: every detection (all COCO classes),
                     confidence, [x1, y1, x2, y2] pixel bbox, and timing in ms
  run_info.json      exact model / library versions, weights sha256, inference
                     settings, hardware, and the manifest sha256 used

Example (from the repository root):
  python detection/experiments/yolo/run_inference.py --weights yolo26s.pt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import ultralytics

from detector import DEFAULT_WEIGHTS, EXPERIMENT_DIR, YoloDetector

REPO_ROOT = EXPERIMENT_DIR.parents[2]
DATASET_DIR = REPO_ROOT / "tests" / "dataset"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DATASET_DIR / "manifest.csv")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Ultralytics pretrained weights name or path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence kept in predictions")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0'; default = Ultralytics auto-select")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warm-up passes before measuring")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"{args.manifest} not found. See tests/dataset/README.md to build or fetch the dataset.")
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dataset_dir = args.manifest.resolve().parent
    missing = [r["file_path"] for r in rows if not (dataset_dir / r["file_path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images listed in the manifest are missing locally (e.g. {missing[0]}). "
                 "Run: python tests/dataset/prepare_dataset.py fetch")

    detector = YoloDetector(args.weights, device=args.device, imgsz=args.imgsz,
                            conf=args.conf, iou=args.iou, max_det=args.max_det)
    model_name = detector.weights_path.stem
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{model_name}_img{args.imgsz}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    first = cv2.imread(str(dataset_dir / rows[0]["file_path"]))
    for _ in range(args.warmup):
        detector.predict(first)

    print(f"[run] {len(rows)} images -> {out_dir.relative_to(REPO_ROOT)}")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            image = cv2.imread(str(dataset_dir / row["file_path"]))
            if image is None:
                sys.exit(f"Could not read {row['file_path']}")
            detections, timing = detector.predict(image)
            f.write(json.dumps({
                "image_id": row["image_id"],
                "file_path": row["file_path"],
                "target_category": row["target_category"],
                "image_size": [image.shape[1], image.shape[0]],
                "timing": timing,
                "detections": detections,
            }) + "\n")

    cuda = torch.cuda.is_available() and args.device != "cpu"
    run_info = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "family": "Ultralytics YOLO (pretrained COCO-80 detection, no fine-tuning)",
            "weights": detector.weights_path.name,
            "weights_sha256": sha256_file(detector.weights_path),
            "num_classes": len(detector.names),
        },
        "inference_settings": {**detector.settings, "device_arg": args.device, "warmup_passes": args.warmup,
                               "half_precision": False, "input": "BGR numpy array read with cv2.imread"},
        "timing_notes": "wall_ms = model.predict() call (preprocess + inference + NMS); image file "
                        "reading is excluded. *_ms fields are Ultralytics' own per-stage timings.",
        "dataset": {
            "manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256_file(args.manifest),
            "num_images": len(rows),
        },
        "environment": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "cpu": platform.processor(),
            "os": platform.platform(),
        },
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"[ok] wrote predictions.jsonl and run_info.json. Next: python "
          f"detection/experiments/yolo/evaluate.py --run {run_name}")


if __name__ == "__main__":
    main()
