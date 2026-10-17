"""Fine-tune YOLO on the 4 CakGo classes (wallet, keys, glasses, remote_control).

Prerequisite: python detection/experiments/yolo/prepare_finetune_data.py

Settings come from finetune.yaml; flags below override them. Output goes to
training_runs/<name>/ (gitignored except training_info.json and the result plots
you choose to commit): weights/best.pt, weights/last.pt, results.csv, plots, and
training_info.json with the exact settings, versions and dataset statistics.

Examples (from the repository root):
  python detection/experiments/yolo/train.py
  python detection/experiments/yolo/train.py --epochs 3 --name smoke_test     # quick check
  python detection/experiments/yolo/train.py --resume training_runs/<name>/weights/last.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from detector import EXPERIMENT_DIR, resolve_weights

CONFIG_PATH = EXPERIMENT_DIR / "finetune.yaml"
DATA_DIR = EXPERIMENT_DIR / "data" / "cakgo4"
TRAINING_DIR = EXPERIMENT_DIR / "training_runs"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "data.yaml")
    parser.add_argument("--model", help="starting weights (overrides finetune.yaml)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", help="e.g. 0 or cpu (default: Ultralytics auto-select)")
    parser.add_argument("--name", help="run folder name under training_runs/")
    parser.add_argument("--resume", type=Path, help="continue an interrupted run from its weights/last.pt")
    args = parser.parse_args()

    if args.resume:
        YOLO(str(args.resume)).train(resume=True)
        return

    if not args.data.exists():
        sys.exit(f"{args.data} not found. Run: python detection/experiments/yolo/prepare_finetune_data.py")
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("model", "epochs", "batch", "imgsz", "device"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    start_weights = resolve_weights(cfg.pop("model"))
    model_name = start_weights.stem
    name = args.name or f"{datetime.now():%Y%m%d_%H%M%S}_{model_name}_cakgo4"
    run_dir = TRAINING_DIR / name
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; choose another --name.")

    started = datetime.now(timezone.utc)
    model = YOLO(str(start_weights))
    model.train(data=str(args.data.resolve()), project=str(TRAINING_DIR), name=name, exist_ok=False, **cfg)
    finished = datetime.now(timezone.utc)

    best = run_dir / "weights" / "best.pt"
    stats_path = args.data.parent / "dataset_stats.json"
    info = {
        "run_name": name,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "start_weights": start_weights.name,
        "start_weights_sha256": sha256_file(start_weights),
        "train_settings": cfg,
        "data_yaml": str(args.data.resolve()),
        "dataset_stats": json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else None,
        "best_weights": str(best.relative_to(EXPERIMENT_DIR)).replace("\\", "/") if best.exists() else None,
        "best_weights_sha256": sha256_file(best) if best.exists() else None,
        "environment": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "os": platform.platform(),
        },
        "notes": "Validation metrics in results.csv are on the export's valid split (minus the shared "
                 "evaluation images), not on tests/dataset. Use run_inference.py + evaluate.py for the "
                 "comparable held-out result.",
    }
    with open(run_dir / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    rel_best = f"detection/experiments/yolo/{info['best_weights']}"
    print(f"\n[ok] training finished: {run_dir}")
    print("Evaluate on the shared 70-image set:")
    print(f"  python detection/experiments/yolo/run_inference.py --weights {rel_best} "
          f"--mapping target_mapping_finetuned.json --conf 0.001 --run-name ft_{name}")
    print(f"  python detection/experiments/yolo/evaluate.py --run ft_{name}")


if __name__ == "__main__":
    main()
