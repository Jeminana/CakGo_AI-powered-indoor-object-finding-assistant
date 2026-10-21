"""Run one zero-shot detector on the shared evaluation set (tests/dataset/manifest.csv).

Same procedure and output format as detection/experiments/grounding_dino/run_inference.py, for
Grounding DINO, YOLOE-26 or SAM 3: each image is queried once per CakGo category (the app asks for one
target at a time), boxes are saved from --box-threshold so AP50 and threshold sweeps can be computed
later, and every query is timed. The only difference: the untimed warm-up runs --warmup queries for
EVERY category, because YOLOE in single mode holds one model per category.

Saves, under tests/results/zero_shot_70/<run_name>/:
  predictions.jsonl  one line per image: detections of every category query, ms per query
  run_info.json      model, prompts, thresholds, library versions, GPU, peak VRAM, manifest sha256

Examples (from the repository root):
  python tests/zero_shot/run_zero_shot.py --model gdino --prompt-mode multi --run-name t4_gdino-tiny_multi
  python tests/zero_shot/run_zero_shot.py --model yoloe --weights yoloe-26s-seg.pt --prompt-mode single
  python tests/zero_shot/run_zero_shot.py --model sam3 --prompt-mode single
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import detectors

REPO_ROOT = detectors.REPO_ROOT
DATASET_DIR = REPO_ROOT / "tests" / "dataset"
RESULTS_DIR = REPO_ROOT / "tests" / "results" / "zero_shot_70"
sys.path.insert(0, str(detectors.GDINO_DIR))
from run_inference import load_image, model_revision, sha256_file  # noqa: E402

# Operating thresholds fixed before the run: each library's own default, nothing tuned on these images.
OPERATING_THRESHOLDS = {
    "gdino": (0.35, "grounding_dino/config/model.json box_threshold (the app's setting)"),
    "yoloe": (0.25, "Ultralytics predict() default conf"),
    "sam3": (0.5, "threshold in the Hugging Face transformers SAM3 examples"),
}


def environment(device: str) -> dict:
    import PIL
    import torch
    import transformers

    cuda = str(device).startswith("cuda")
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "cpu": platform.processor(),
        "os": platform.platform(),
    }
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    return info


def model_info(args, detector) -> dict:
    info = {"family": detector.family, "model_id": detector.model_id}
    if args.model in ("gdino", "sam3"):
        info["revision"] = model_revision(detector.model_id)
    else:
        path = detector.weights_path()
        info["weights_path"] = path
        info["weights_sha256"] = sha256_file(Path(path)) if path and Path(path).exists() else None
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["gdino", "yoloe", "sam3"], required=True)
    parser.add_argument("--weights", default=None, help="YOLOE checkpoint or Hugging Face model id")
    parser.add_argument("--prompt-mode", choices=["single", "multi"], required=True)
    parser.add_argument("--manifest", type=Path, default=DATASET_DIR / "manifest.csv")
    parser.add_argument("--box-threshold", type=float, default=0.05, help="minimum confidence kept in predictions")
    parser.add_argument("--device", default="auto", help="e.g. cuda or cpu")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warm-up queries per category")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"{args.manifest} not found. See tests/dataset/README.md to build or fetch the dataset.")
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:args.limit]
    dataset_dir = args.manifest.resolve().parent
    missing = [r["file_path"] for r in rows if not (dataset_dir / r["file_path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images listed in the manifest are missing locally (e.g. {missing[0]}). "
                 "Run: python tests/dataset/prepare_dataset.py fetch")
    categories = list(detectors.load_json(dataset_dir / "class_mapping.json")["categories"])

    detector = detectors.create(args.model, args.weights, args.prompt_mode, args.box_threshold, args.device)
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{args.model}_{args.prompt_mode}"
    out_dir = args.results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    detector.prepare(categories)
    first = load_image(dataset_dir / rows[0]["file_path"])
    for category in categories:
        for _ in range(args.warmup):
            detector.detect(first, category)

    print(f"[run] {len(rows)} images x {len(categories)} queries -> {out_dir}", flush=True)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for n, row in enumerate(rows, 1):
            image = load_image(dataset_dir / row["file_path"])
            detections, wall_ms = [], {}
            for category in categories:
                start = time.perf_counter()
                result = detector.detect(image, category)
                wall_ms[category] = round((time.perf_counter() - start) * 1000, 2)
                detections += result["detections"]
            f.write(json.dumps({
                "image_id": row["image_id"],
                "file_path": row["file_path"],
                "target_category": row["target_category"],
                "image_size": [image.width, image.height],
                "timing": {"wall_ms": wall_ms},
                "detections": detections,
            }) + "\n")
            if n % 10 == 0:
                print(f"[run] {n}/{len(rows)} images", flush=True)

    operating, operating_source = OPERATING_THRESHOLDS[args.model]
    run_info = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_info(args, detector),
        "prompts": {category: detector.prompt_for(category) for category in categories},
        "inference_settings": {
            "prompt_mode": detector.prompt_mode,
            "look_alikes": detector.look_alikes if detector.prompt_mode == "multi" else [],
            "box_threshold": detector.box_threshold,
            "operating_threshold": operating,
            "operating_threshold_source": operating_source,
            "device": str(detector.device),
            "warmup_queries_per_category": args.warmup,
            "queries_per_image": len(categories),
            "half_precision": False,
            "input": "RGB PIL image read with Pillow",
            "same_object_iou": detectors.SAME_OBJECT_IOU if args.model != "gdino" else None,
        },
        "timing_notes": "wall_ms[category] = one detect() call (preprocess + inference + post-process) "
                        "with that category's prompt; image file reading is excluded.",
        "dataset": {
            "manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256_file(args.manifest),
            "num_images": len(rows),
        },
        "environment": environment(str(detector.device)),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"[ok] wrote {out_dir}/predictions.jsonl and run_info.json", flush=True)


if __name__ == "__main__":
    main()
