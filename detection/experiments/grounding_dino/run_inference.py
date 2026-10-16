"""Run Grounding DINO on the shared evaluation set (tests/dataset/manifest.csv).

Mirrors detection/experiments/yolo/run_inference.py so both models are scored by the same
rules. Each image is queried once per CakGo category, because the app asks for one target
at a time; every category is therefore scored on every image, like YOLO's classes.

Saves, under results/<run_name>/:
  predictions.jsonl  one line per image: the detections of every category query
                     (label = CakGo category), confidence, [x1, y1, x2, y2] pixel bbox,
                     and wall-clock ms per query
  run_info.json      model revision, prompts, thresholds, library versions, hardware,
                     and the manifest sha256 used

Example (from the repository root):
  python detection/experiments/grounding_dino/run_inference.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from detector import GroundingDinoDetector, load_json, prompt_text

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
DATASET_DIR = REPO_ROOT / "tests" / "dataset"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def model_revision(model_id: str) -> str | None:
    """Commit hash of the cached Hugging Face snapshot that was loaded."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def environment(detector: GroundingDinoDetector) -> dict:
    import PIL
    import torch
    import transformers

    cuda = str(detector.device).startswith("cuda")
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DATASET_DIR / "manifest.csv")
    parser.add_argument("--categories", nargs="+", help="CakGo categories to query (default: all in class_mapping.json)")
    parser.add_argument("--box-threshold", type=float, default=0.05, help="minimum confidence kept in predictions")
    parser.add_argument("--device", default=None, help="override the config device, e.g. cpu or cuda")
    parser.add_argument("--warmup", type=int, default=3, help="untimed warm-up queries before measuring")
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
    categories = args.categories or list(load_json(dataset_dir / "class_mapping.json")["categories"])

    detector = GroundingDinoDetector.from_config(device=args.device, box_threshold=args.box_threshold)
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_{detector.model_id.split('/')[-1]}"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    first = load_image(dataset_dir / rows[0]["file_path"])
    for _ in range(args.warmup):
        detector.detect(first, categories[0])

    print(f"[run] {len(rows)} images x {len(categories)} queries -> {out_dir.relative_to(REPO_ROOT)}")
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
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

    run_info = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "family": "Grounding DINO (Hugging Face transformers, zero-shot, no fine-tuning)",
            "model_id": detector.model_id,
            "revision": model_revision(detector.model_id),
        },
        "prompts": {category: prompt_text(detector.phrases_for(category)) for category in categories},
        "inference_settings": {
            "box_threshold": detector.box_threshold,
            "text_threshold": detector.text_threshold,
            "device": detector.device,
            "warmup_queries": args.warmup,
            "queries_per_image": len(categories),
            "half_precision": False,
            "input": "RGB PIL image read with Pillow",
        },
        "timing_notes": "wall_ms[category] = one detect() call (preprocess + inference + post-process) "
                        "with that category's prompt; image file reading is excluded.",
        "dataset": {
            "manifest": str(args.manifest.resolve().relative_to(REPO_ROOT)).replace("\\", "/"),
            "manifest_sha256": sha256_file(args.manifest),
            "num_images": len(rows),
        },
        "environment": environment(detector),
    }
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"[ok] wrote predictions.jsonl and run_info.json. Next: python "
          f"detection/experiments/grounding_dino/evaluate.py --run {run_name}")


if __name__ == "__main__":
    main()
