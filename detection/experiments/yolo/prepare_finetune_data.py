"""Build the YOLO-format fine-tuning dataset for the 4 CakGo classes.

Reads the Objects In House export that tests/dataset/prepare_dataset.py downloads into
tests/dataset/.cache/ (run `prepare_dataset.py download` first) and writes
detection/experiments/yolo/data/cakgo4/ (gitignored):

  images/{train,val}/  labels/{train,val}/  data.yaml  dataset_stats.json

Splits
  train  <- the export's train split
  val    <- the export's valid split, used for early stopping / threshold tuning
  (the export's test split is not used at all)

Leakage guard: any image that is one of the 70 shared evaluation images
(tests/dataset/manifest.csv), comes from the same original photo, or is a
near-duplicate of one (dHash), is excluded from BOTH train and val, so the
shared evaluation stays a held-out test.

Labels
  - Source labels are mapped with tests/dataset/class_mapping.json.
  - Images whose only labels are non-targets (bag, broom, shoes, earphone) are kept
    as background images with an empty label file; they teach the model what is NOT a target.
  - Images containing a label that class_mapping.json does not list (e.g. remote-control-ac,
    AC remotes) are dropped, because it is ambiguous whether they should count as a target.
  - Images with no annotations at all are dropped (they may simply be unlabeled).

Example (from the repository root):
  python detection/experiments/yolo/prepare_finetune_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "dataset"))
import prepare_dataset as shared  # noqa: E402  (shared dataset helpers, read-only use)

DEFAULT_OUT = EXPERIMENT_DIR / "data" / "cakgo4"
SOURCE_KEY = "objects_in_house"
SPLITS = {"train": "train", "val": "valid"}  # our split name -> Roboflow split name


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link when possible (same drive, no extra disk space), otherwise copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def eval_guard(source: dict) -> tuple[set, list[int]]:
    """Original-photo stems and dHashes of the 70 shared evaluation images."""
    if not shared.MANIFEST_PATH.exists():
        sys.exit("tests/dataset/manifest.csv not found; the leakage guard needs it.")
    with open(shared.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stems, hashes = set(), []
    export = shared.export_dir(SOURCE_KEY, source)
    for row in rows:
        stems.add(shared.original_stem(row["source_filename"]))
        path = export / row["source_split"] / row["source_filename"]
        if not path.exists():
            path = shared.DATASET_DIR / row["file_path"]
        if not path.exists():
            sys.exit(f"Cannot find evaluation image {row['image_id']} for the leakage guard. "
                     "Run tests/dataset/prepare_dataset.py download first.")
        hashes.append(shared.dhash(path))
    return stems, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="delete and rebuild an existing --out folder")
    args = parser.parse_args()

    mapping = shared.load_mapping()
    source = mapping["sources"][SOURCE_KEY]
    categories = list(mapping["categories"])  # class id = index: wallet, keys, glasses, remote_control
    class_id = {c: i for i, c in enumerate(categories)}

    if not shared.export_dir(SOURCE_KEY, source).exists():
        sys.exit("No local Objects In House export. Run: python tests/dataset/prepare_dataset.py download")
    if args.out.exists():
        if not args.force:
            sys.exit(f"{args.out} already exists (use --force to rebuild).")
        shutil.rmtree(args.out)

    eval_stems, eval_hashes = eval_guard(source)
    stats = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "source": f"{source['name']} v{source['version']}", "classes": categories, "splits": {}}

    for split, rf_split in SPLITS.items():
        records = shared.load_source_records(SOURCE_KEY, {**source, "split_preference": [rf_split]})
        if not records:
            sys.exit(f"Split '{rf_split}' not found in the export.")
        (args.out / "images" / split).mkdir(parents=True)
        (args.out / "labels" / split).mkdir(parents=True)
        counts = Counter()
        boxes = Counter()
        for rec in records:
            labels = {o["source_label"] for o in rec["objects"]}
            if not rec["objects"]:
                counts["dropped_unannotated"] += 1
                continue
            if labels - set(source["label_map"]):
                counts["dropped_unmapped_label"] += 1
                continue
            if shared.original_stem(rec["file_name"]) in eval_stems:
                counts["dropped_eval_same_photo"] += 1
                continue
            h = shared.dhash(rec["path"])
            if any(bin(h ^ e).count("1") <= shared.DHASH_MAX_DISTANCE for e in eval_hashes):
                counts["dropped_eval_near_duplicate"] += 1
                continue

            lines = []
            for o in rec["objects"]:
                if o["category"] is None:
                    continue
                x1, y1, x2, y2 = o["bbox_xyxy"]
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(rec["width"]), x2), min(float(rec["height"]), y2)
                if x2 <= x1 or y2 <= y1:
                    counts["skipped_degenerate_box"] += 1
                    continue
                cx, cy = (x1 + x2) / 2 / rec["width"], (y1 + y2) / 2 / rec["height"]
                w, h_ = (x2 - x1) / rec["width"], (y2 - y1) / rec["height"]
                lines.append(f"{class_id[o['category']]} {cx:.6f} {cy:.6f} {w:.6f} {h_:.6f}")
                boxes[o["category"]] += 1

            link_or_copy(rec["path"], args.out / "images" / split / rec["file_name"])
            label_file = args.out / "labels" / split / (Path(rec["file_name"]).stem + ".txt")
            label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts["images"] += 1
            counts["background_images" if not lines else "images_with_targets"] += 1

        stats["splits"][split] = {"roboflow_split": rf_split, "candidates": len(records),
                                  **dict(counts), "boxes_per_class": {c: boxes[c] for c in categories}}
        print(f"[{split}] {json.dumps(stats['splits'][split])}")

    data_yaml = "\n".join([
        "# Generated by prepare_finetune_data.py - do not edit by hand.",
        f"path: {args.out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
        *[f"  {i}: {c}" for i, c in enumerate(categories)],
    ]) + "\n"
    (args.out / "data.yaml").write_text(data_yaml, encoding="utf-8")
    with open(args.out / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[ok] wrote {args.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
