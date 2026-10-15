"""Build / reproduce the shared CakGo evaluation subset in tests/dataset/.

Both detector experiments (YOLO, Grounding DINO) read the SAME files produced
here. Images are copied byte-for-byte from the Roboflow export (no resizing or
re-encoding), and the original source labels/boxes are preserved.

Sub-commands
------------
  download  Download the Roboflow COCO exports into tests/dataset/.cache/
            (needs ROBOFLOW_API_KEY, or use --zip to point at a manually
            downloaded export zip).
  build     Select ~70 images (deterministic, seeded), copy them into
            images/<category>/, write manifest.csv, annotations/ground_truth.json
            and annotations/selection_report.json. Run ONCE by the dataset
            owner; the outputs (except images) are committed to git.
  fetch     Re-create images/ from the committed manifest.csv using a local
            export (for teammates). Never re-selects.
  verify    Check every file in manifest.csv exists and matches its sha256.

Usage (from the repository root)
--------------------------------
  python tests/dataset/prepare_dataset.py download --source objects_in_house
  python tests/dataset/prepare_dataset.py build
  python tests/dataset/prepare_dataset.py fetch
  python tests/dataset/prepare_dataset.py verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATASET_DIR / ".cache"
IMAGES_DIR = DATASET_DIR / "images"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
GROUND_TRUTH_PATH = ANNOTATIONS_DIR / "ground_truth.json"
REPORT_PATH = ANNOTATIONS_DIR / "selection_report.json"
CLASS_MAPPING_PATH = DATASET_DIR / "class_mapping.json"

DEFAULT_SEED = "cakgo-eval-v1"
DHASH_MAX_DISTANCE = 6  # Hamming distance (of 64 bits) treated as "near-duplicate"
ROBOFLOW_SUFFIX = re.compile(r"^(?P<stem>.+?)_(?:jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+\.\w+$", re.I)

MANIFEST_COLUMNS = [
    "image_id", "file_path", "sha256", "width", "height",
    "target_category", "contains_categories", "gt_boxes_xyxy", "num_gt_boxes",
    "source_labels", "annotation_available", "testing_condition", "review_status",
    "source_dataset", "source_url", "source_version", "source_split",
    "source_filename", "license",
]


# --------------------------------------------------------------------------- utils

def load_mapping() -> dict:
    with open(CLASS_MAPPING_PATH, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_dir(source_key: str, source: dict) -> Path:
    return CACHE_DIR / source_key / f"v{source['version']}_{source['export_format']}"


def original_stem(file_name: str) -> str:
    """Roboflow renames IMG_1.jpg to IMG_1_jpg.rf.<hash>.jpg; recover IMG_1."""
    m = ROBOFLOW_SUFFIX.match(file_name)
    return m.group("stem") if m else Path(file_name).stem


def dhash(path: Path) -> int:
    from PIL import Image  # imported lazily so `verify` works without Pillow

    with Image.open(path) as im:
        px = im.convert("L").resize((9, 8)).tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest not in target.parents and target != dest:
                raise RuntimeError(f"Refusing unsafe path in zip: {member.filename}")
        zf.extractall(dest)


# ------------------------------------------------------------------------ download

def cmd_download(args: argparse.Namespace) -> None:
    mapping = load_mapping()
    source = mapping["sources"][args.source]
    dest = export_dir(args.source, source)
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(f"[skip] {dest} already exists (use --force to re-download)")
        return
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / f"{dest.name}.zip"

    if args.zip:
        shutil.copyfile(args.zip, zip_path)
    else:
        import requests

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            sys.exit(
                "ROBOFLOW_API_KEY is not set. Create a free Roboflow account, copy the key from\n"
                "Settings -> API Keys, and set it as an environment variable (never commit it).\n"
                f"Alternatively download the {source['export_format']} export of version {source['version']}\n"
                f"manually from {source['url']} and pass --zip <file>."
            )
        url = (f"https://api.roboflow.com/{source['workspace']}/{source['project']}/"
               f"{source['version']}/{source['export_format']}")
        resp = requests.get(url, params={"api_key": api_key}, timeout=60)
        if resp.status_code != 200:
            sys.exit(f"Roboflow API returned HTTP {resp.status_code}: {resp.text[:300]}")
        link = resp.json().get("export", {}).get("link")
        if not link:
            sys.exit("Roboflow did not return an export link (the export may still be generating). "
                     "Retry in a minute, or download the zip manually and pass --zip.")
        print(f"[download] {source['name']} v{source['version']} ({source['export_format']})")
        with requests.get(link, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)

    safe_extract(zip_path, dest)
    zip_path.unlink()
    print(f"[ok] extracted to {dest}")


# --------------------------------------------------------------------------- build

def load_source_records(source_key: str, source: dict) -> list[dict]:
    """Return one record per image (from preferred splits) with its original annotations."""
    root = export_dir(source_key, source)
    if not root.exists():
        return []
    records = []
    for split in source["split_preference"]:
        ann_file = root / split / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        objects = defaultdict(list)
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            label = cat_names[a["category_id"]]
            objects[a["image_id"]].append({
                "source_label": label,
                "category": source["label_map"].get(label),
                "bbox_xyxy": [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)],
                "bbox_source_coco_xywh": [x, y, w, h],
            })
        for img in coco["images"]:
            records.append({
                "source_key": source_key,
                "split": split,
                "file_name": img["file_name"],
                "path": root / split / img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "objects": objects.get(img["id"], []),
            })
    unmapped = {o["source_label"] for r in records for o in r["objects"]} - set(source["label_map"])
    if unmapped:
        print(f"[warn] {source_key}: labels not in class_mapping.json (treated as non-target): {sorted(unmapped)}")
    return records


def select_images(mapping: dict, quotas: dict, seed: str) -> tuple[dict, dict, dict]:
    """Fill each bucket from the primary source first, then from later sources for shortfalls."""
    categories = list(mapping["categories"])
    source_order = sorted(mapping["sources"], key=lambda k: mapping["sources"][k]["role"] != "primary")
    selected: dict[str, list[dict]] = {b: [] for b in quotas}
    used_stems, used_hashes = set(), []
    skipped = defaultdict(int)
    sources_used = {}

    def order_key(rec: dict) -> str:
        return hashlib.sha256(f"{seed}|{rec['source_key']}|{rec['file_name']}".encode()).hexdigest()

    for source_key in source_order:
        if all(len(selected[b]) >= quotas[b] for b in quotas):
            break
        source = mapping["sources"][source_key]
        records = load_source_records(source_key, source)
        if not records:
            print(f"[info] no local export for {source_key} (run: download --source {source_key})")
            continue
        print(f"[info] {source_key}: {len(records)} candidate images from splits {source['split_preference']}")
        sources_used[source_key] = source
        for rec in records:
            rec["cats"] = sorted({o["category"] for o in rec["objects"] if o["category"]})

        for bucket in categories + ["negative"]:
            need = quotas[bucket] - len(selected[bucket])
            if need <= 0:
                continue
            if bucket == "negative":
                pool = [r for r in records if r["objects"] and not r["cats"]]
            else:
                pool = [r for r in records if bucket in r["cats"]]
            # Prefer images where this is the only target category present.
            pool.sort(key=lambda r: (len(r["cats"]) > 1, order_key(r)))
            for rec in pool:
                if need <= 0:
                    break
                stem_key = (rec["source_key"], original_stem(rec["file_name"]))
                if stem_key in used_stems:
                    skipped["same_original_image"] += 1
                    continue
                h = dhash(rec["path"])
                if any(bin(h ^ other).count("1") <= DHASH_MAX_DISTANCE for other in used_hashes):
                    skipped["near_duplicate"] += 1
                    continue
                used_stems.add(stem_key)
                used_hashes.append(h)
                selected[bucket].append(rec)
                need -= 1
    return selected, sources_used, dict(skipped)


def cmd_build(args: argparse.Namespace) -> None:
    mapping = load_mapping()
    categories = list(mapping["categories"])
    quotas = {c: mapping["categories"][c]["quota"] for c in categories}
    quotas["negative"] = mapping["negative"]["quota"]

    if MANIFEST_PATH.exists() and not args.force:
        sys.exit("manifest.csv already exists. Use fetch to reproduce it, or build --force to re-select.")

    selected, sources_used, skipped = select_images(mapping, quotas, args.seed)
    if not sources_used:
        sys.exit("No source exports found in .cache/. Run the download step first.")

    # Clear previously selected images (keep .gitkeep), then copy the new selection byte-for-byte.
    for bucket in quotas:
        folder = IMAGES_DIR / bucket
        folder.mkdir(parents=True, exist_ok=True)
        for f in folder.iterdir():
            if f.is_file() and f.name != ".gitkeep":
                f.unlink()
    ANNOTATIONS_DIR.mkdir(exist_ok=True)

    rows, gt_images = [], []
    for bucket, recs in selected.items():
        for i, rec in enumerate(recs, 1):
            src = mapping["sources"][rec["source_key"]]
            image_id = f"{bucket}_{i:03d}"
            rel_path = f"images/{bucket}/{image_id}{rec['path'].suffix.lower()}"
            shutil.copyfile(rec["path"], DATASET_DIR / rel_path)
            target_boxes = [o["bbox_xyxy"] for o in rec["objects"] if o["category"] == bucket]
            rows.append({
                "image_id": image_id,
                "file_path": rel_path,
                "sha256": sha256_file(DATASET_DIR / rel_path),
                "width": rec["width"],
                "height": rec["height"],
                "target_category": bucket,
                "contains_categories": ";".join(rec["cats"]),
                "gt_boxes_xyxy": json.dumps(target_boxes),
                "num_gt_boxes": len(target_boxes),
                "source_labels": ";".join(sorted({o["source_label"] for o in rec["objects"]})),
                "annotation_available": "yes" if rec["objects"] else "no",
                "testing_condition": "unknown",
                "review_status": "unreviewed",
                "source_dataset": src["name"],
                "source_url": src["url"],
                "source_version": src["version"],
                "source_split": rec["split"],
                "source_filename": rec["file_name"],
                "license": src["license"],
            })
            gt_images.append({
                "image_id": image_id,
                "file_path": rel_path,
                "width": rec["width"],
                "height": rec["height"],
                "target_category": bucket,
                "objects": rec["objects"],
            })

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "bbox_format": "xyxy in absolute pixels of the image file at file_path",
            "note": "objects lists EVERY original source annotation (all labels); category is the "
                    "CakGo category, or null when the source label is not a target.",
            "categories": categories,
            "images": gt_images,
        }, f, indent=1)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "dhash_max_distance": DHASH_MAX_DISTANCE,
        "sources": {k: {"name": s["name"], "version": s["version"], "splits": s["split_preference"]}
                    for k, s in sources_used.items()},
        "counts": {b: {"selected": len(selected[b]), "target": quotas[b]} for b in quotas},
        "shortfalls": {b: quotas[b] - len(selected[b]) for b in quotas if len(selected[b]) < quotas[b]},
        "skipped": skipped,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["counts"], indent=2))
    if report["shortfalls"]:
        print(f"[warn] shortfalls (document these in dataset_info.md): {report['shortfalls']}")
    print("[ok] wrote manifest.csv, annotations/ground_truth.json, annotations/selection_report.json")


# ------------------------------------------------------------------- fetch/verify

def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        sys.exit("manifest.csv not found. The dataset owner must run build and commit the manifest first.")
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_fetch(args: argparse.Namespace) -> None:
    mapping = load_mapping()
    by_name = {s["name"]: (k, s) for k, s in mapping["sources"].items()}
    missing_exports = set()
    for row in read_manifest():
        key, source = by_name[row["source_dataset"]]
        if int(row["source_version"]) != source["version"]:
            sys.exit(f"{row['image_id']}: manifest uses v{row['source_version']} but class_mapping.json "
                     f"has v{source['version']}")
        src = export_dir(key, source) / row["source_split"] / row["source_filename"]
        if not src.exists():
            missing_exports.add(key)
            continue
        dst = DATASET_DIR / row["file_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    if missing_exports:
        sys.exit(f"Missing local exports for {sorted(missing_exports)}. Run download --source <name> first.")
    cmd_verify(args)


def cmd_verify(args: argparse.Namespace) -> None:
    rows = read_manifest()
    bad = 0
    for row in rows:
        path = DATASET_DIR / row["file_path"]
        if not path.exists():
            print(f"[missing] {row['file_path']}")
            bad += 1
        elif sha256_file(path) != row["sha256"]:
            print(f"[sha256 mismatch] {row['file_path']}")
            bad += 1
    if bad:
        sys.exit(f"[fail] {bad}/{len(rows)} files missing or different.")
    print(f"[ok] all {len(rows)} images match manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download a Roboflow export into .cache/")
    p.add_argument("--source", default="objects_in_house", choices=list(load_mapping()["sources"]))
    p.add_argument("--zip", type=Path, help="use a manually downloaded export zip instead of the API")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("build", help="select the evaluation subset (dataset owner, once)")
    p.add_argument("--seed", default=DEFAULT_SEED)
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest.csv")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("fetch", help="re-create images/ from the committed manifest.csv")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("verify", help="check images against the manifest.csv sha256 values")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
