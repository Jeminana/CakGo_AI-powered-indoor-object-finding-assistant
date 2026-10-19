"""Score detectors on the team's real phone photos (task: real-photo test).

Expected layout (as in the photo instructions; the Drive folder can be downloaded as is):

    <photos>/wallet/wallet_<name>_A1.jpg ... <photos>/none/none_<name>_N4.jpg
    subfolders: wallet, keys, glasses, remote_control, none

The folder says which object is in the photo (one per photo, "none" = none of the four).
Every model is asked about all four objects in every photo, like the app would be:

  found        = photos of that object where the model returned a box   (higher is better)
  false alarm  = photos WITHOUT that object where it still returned one (lower is better)

There are no ground-truth boxes, so "found" only means "said yes". Use --sheets to draw each
model's top box and check by eye that it is on the right object.

Photos are prepared like app frames: EXIF rotation applied, long side scaled to 1280 px.

Run from the repository root:
  python tests/real_photos/score_real_photos.py --photos <folder> --run-name team_photos_v1
  python tests/real_photos/score_real_photos.py --photos <folder> --models gdino_multi yolo_v1 --sheets
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "tests" / "results" / "real_photos"
SHEETS_DIR = Path(__file__).resolve().parent / "sheets"  # gitignored: contains the private photos
CATEGORIES = ["wallet", "keys", "glasses", "remote_control"]
FOLDERS = CATEGORIES + ["none"]
FOLDER_ALIASES = {"empty": "none", "remote controller": "remote_control", "remote": "remote_control"}
CONDITIONS = {  # photo code -> condition, from the photo instructions
    "A1": "easy", "A2": "cluttered", "A3": "partly hidden",
    "B1": "far", "B2": "dim", "B3": "cluttered",
    "N1": "look-alikes", "N2": "open bag", "N3": "dim", "N4": "far",
}
# Photo code anywhere in the name, e.g. "keys_najmi_A1", "wallet Image_A1_768_2", "glasses _B5_2".
# Codes outside the plan (A4, B5, C1, D2, ...) are kept with condition "unknown".
CODE = re.compile(r"(?:^|[_\s])([A-DN][0-9])(?=$|[_\s])", re.I)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
# PNG files in the shared folder are small screenshots of web images, not phone photos.
SCREENSHOT_SUFFIXES = {".png"}
MAX_SIDE = 1280      # same as frontend/index.html
SAVE_MIN_CONF = 0.05  # keep low-confidence boxes so thresholds can be swept afterwards
SWEEP = [0.25, 0.35, 0.5]


def load_module(name: str, path: Path):
    """Both experiment folders have a detector.py, so load each under its own name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


class YoloModel:
    threshold = 0.25  # detector default

    def __init__(self, weights: str):
        mod = load_module("yolo_detector", REPO_ROOT / "detection/experiments/yolo/detector.py")
        self.engine = mod.YoloDetector(weights, conf=SAVE_MIN_CONF)
        self.unsupported = mod.UnsupportedTargetError

    def detect(self, image: Image.Image, target: str):
        try:
            result = self.engine.detect(np.asarray(image)[:, :, ::-1].copy(), target)  # expects BGR
        except self.unsupported:
            return None
        return result["detections"]


class GdinoModel:
    def __init__(self, prompt_mode: str):
        mod = load_module("gdino_detector", REPO_ROOT / "detection/experiments/grounding_dino/detector.py")
        if not hasattr(mod.GroundingDinoDetector, "prompt_for"):
            sys.exit("This Grounding DINO version has no prompt_mode yet. Merge PR #10 (multi-prompt) first, "
                     "or run with --models yolo_v1 yolo_pretrained.")
        self.engine = mod.GroundingDinoDetector.from_config(box_threshold=SAVE_MIN_CONF, prompt_mode=prompt_mode)
        self.threshold = mod.load_json(mod.CONFIG_DIR / "model.json")["box_threshold"]

    def detect(self, image: Image.Image, target: str):
        return self.engine.detect(image, target)["detections"]


MODELS = {
    "gdino_multi": lambda: GdinoModel("multi"),
    "gdino_single": lambda: GdinoModel("single"),
    "yolo_v1": lambda: YoloModel("detection/experiments/yolo/training_runs/yolo26s_cakgo4_v1/weights/best.pt"),
    "yolo_v2": lambda: YoloModel("detection/experiments/yolo/training_runs/yolo26s_cakgo4_v2/weights/best.pt"),
    "yolo_pretrained": lambda: YoloModel("yolo26s.pt"),  # COCO: remote_control only
}


def folder_category(name: str) -> str | None:
    name = name.lower().strip()
    name = FOLDER_ALIASES.get(name, name)
    return name if name in FOLDERS else None


def person_from_name(stem: str, code_match) -> str:
    """'keys_najmi_A1' -> 'najmi'; names without a person token (e.g. 'Weixin Image_...') -> '?'."""
    if not code_match:
        return "?"
    tokens = [t for t in re.split(r"[_\s]+", stem[:code_match.start(1)]) if t]
    item_words = {"wallet", "key", "keys", "glasses", "remote", "control", "none", "image", "weixin"}
    return tokens[-1].lower() if len(tokens) >= 2 and tokens[-1].isalpha() and tokens[-1].lower() not in item_words else "?"


def find_photos(root: Path, include_screenshots: bool) -> tuple[list[dict], list[Path]]:
    photos, skipped = [], []
    for path in sorted(root.rglob("*")):
        truth = folder_category(path.parent.name)
        if path.suffix.lower() not in IMAGE_SUFFIXES or truth is None:
            continue
        if path.suffix.lower() in SCREENSHOT_SUFFIXES and not include_screenshots:
            skipped.append(path)
            continue
        match = CODE.search(path.stem)
        code = match.group(1).upper() if match else "?"
        person = person_from_name(path.stem, match)
        photos.append({
            "path": path,
            "truth": truth,
            "code": code,
            # Only names in the planned "item_name_code" form are trusted to follow the condition plan.
            "condition": CONDITIONS.get(code, "unknown") if person != "?" else "unknown",
            "person": person,
        })
    return photos, skipped


def photo_set_fingerprint(photos: list[dict]) -> tuple[str, list[dict]]:
    """sha256 of every photo plus one combined hash, so separate runs can prove they used the same set."""
    entries = []
    for p in photos:
        digest = hashlib.sha256(p["path"].read_bytes()).hexdigest()
        entries.append({"file": f"{p['path'].parent.name}/{p['path'].name}", "sha256": digest})
    combined = hashlib.sha256("\n".join(sorted(e["sha256"] for e in entries)).encode()).hexdigest()
    return combined, entries


def open_photo(path: Path) -> Image.Image:
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            import pillow_heif
        except ImportError:
            sys.exit(f"{path.name} is HEIC. Run `pip install pillow-heif`, or export the photos as JPG.")
        pillow_heif.register_heif_opener()
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = min(1.0, MAX_SIDE / max(image.size))
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return image


def rate(hits: int, total: int) -> str:
    return f"{hits}/{total} ({hits / total:.0%})" if total else "-"


def summarise(photos, results, model_names, thresholds):
    """results[(photo_index, model, category)] = list of confidences (None = unsupported)."""
    summary = {}
    for model in model_names:
        thr = thresholds[model]
        per_cat = {}
        for cat in CATEGORIES:
            if all(results[(i, model, cat)] is None for i in range(len(photos))):
                per_cat[cat] = None
                continue
            said_yes = {i: any(c >= thr for c in results[(i, model, cat)]) for i in range(len(photos))}
            pos = [i for i, p in enumerate(photos) if p["truth"] == cat]
            neg = [i for i, p in enumerate(photos) if p["truth"] != cat]
            per_cat[cat] = {
                "found": sum(said_yes[i] for i in pos), "photos_with_object": len(pos),
                "false_alarms": sum(said_yes[i] for i in neg), "photos_without_object": len(neg),
                "false_alarms_on_none": sum(said_yes[i] for i in neg if photos[i]["truth"] == "none"),
                "sweep": {str(t): {"found": sum(any(c >= t for c in results[(i, model, cat)]) for i in pos),
                                   "false_alarms": sum(any(c >= t for c in results[(i, model, cat)]) for i in neg)}
                          for t in SWEEP},
            }
        by_condition = {}
        for cond in sorted({p["condition"] for p in photos if p["truth"] != "none"}):
            idx = [i for i, p in enumerate(photos) if p["truth"] != "none" and p["condition"] == cond
                   and results[(i, model, p["truth"])] is not None]
            by_condition[cond] = {"found": sum(any(c >= thr for c in results[(i, model, photos[i]["truth"])])
                                               for i in idx), "photos": len(idx)}
        summary[model] = {"threshold": thr, "categories": per_cat, "found_by_condition": by_condition}
    return summary


def markdown(summary, photos, run_name, fingerprint, skipped):
    counts = {f: sum(p["truth"] == f for p in photos) for f in FOLDERS}
    people = {}
    for p in photos:
        people[p["person"]] = people.get(p["person"], 0) + 1
    lines = [f"# Real phone photos - {run_name}", "",
             f"Photos: {len(photos)} (" + ", ".join(f"{f} {n}" for f, n in counts.items()) + "). "
             "Each model is asked about all 4 objects in every photo.", "",
             "By person (from the file name; `?` = no name in the file name): "
             + ", ".join(f"{k} {v}" for k, v in sorted(people.items())) + ".", "",
             f"Photo set fingerprint: `{fingerprint}`. Runs with the same fingerprint used exactly the same photos.", ""]
    if skipped:
        lines += [f"Excluded: {len(skipped)} screenshot PNGs (not phone photos): "
                  + ", ".join(f"`{p.parent.name}/{p.name}`" for p in skipped) + ".", ""]
    lines += [
             "**Found** = photos of the object where the model returned a box. "
             "**False alarms** = photos without the object where it still returned one. "
             "No ground-truth boxes: check box positions with `--sheets`.", "",
             "| Model (threshold) | Object | Found | False alarms (all photos without it) | False alarms on `none` photos |",
             "|---|---|---|---|---|"]
    for model, s in summary.items():
        for cat, m in s["categories"].items():
            if m is None:
                lines.append(f"| {model} ({s['threshold']}) | {cat} | unsupported | - | - |")
            else:
                lines.append(f"| {model} ({s['threshold']}) | {cat} | {rate(m['found'], m['photos_with_object'])} | "
                             f"{rate(m['false_alarms'], m['photos_without_object'])} | "
                             f"{m['false_alarms_on_none']}/{counts['none']} |")
    conds = sorted({c for s in summary.values() for c in s["found_by_condition"]})
    lines += ["", "## Found by condition (supported objects only)", "",
              "| Model | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for model, s in summary.items():
        lines.append(f"| {model} | " + " | ".join(rate(s["found_by_condition"][c]["found"],
                                                         s["found_by_condition"][c]["photos"])
                                                    if c in s["found_by_condition"] else "-" for c in conds) + " |")
    lines += ["", "## Threshold sweep: found / false alarms", "",
              "| Model | Object | " + " | ".join(str(t) for t in SWEEP) + " |", "|---|---|" + "---|" * len(SWEEP)]
    for model, s in summary.items():
        for cat, m in s["categories"].items():
            if m is not None:
                lines.append(f"| {model} | {cat} | " + " | ".join(
                    f"{m['sweep'][str(t)]['found']}/{m['photos_with_object']} / "
                    f"{m['sweep'][str(t)]['false_alarms']}/{m['photos_without_object']}" for t in SWEEP) + " |")
    return "\n".join(lines) + "\n"


def draw_sheet(image, boxes_by_model, out_path, title):
    tiles = []
    for model, box in boxes_by_model.items():
        tile = image.copy()
        draw = ImageDraw.Draw(tile)
        if box:
            draw.rectangle(box["bbox"], outline=(255, 0, 0), width=max(3, tile.width // 150))
        label = f"{model}: {box['confidence']:.2f}" if box else f"{model}: -"
        draw.rectangle([0, 0, tile.width, tile.height // 14], fill=(0, 0, 0))
        draw.text((10, 5), f"{label}  [{title}]", fill=(255, 255, 255), font_size=max(14, tile.height // 24))
        tiles.append(tile.resize((360, round(360 * tile.height / tile.width))))
    sheet = Image.new("RGB", (360 * len(tiles), max(t.height for t in tiles)), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * 360, 0))
    sheet.save(out_path, quality=80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, required=True, help="folder with wallet/ keys/ glasses/ remote_control/ none/")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["gdino_multi", "gdino_single", "yolo_v1"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--include-screenshots", action="store_true",
                        help="also score .png files (by default they are excluded: screenshots, not phone photos)")
    parser.add_argument("--sheets", action="store_true",
                        help=f"draw each model's top box per photo into {SHEETS_DIR.relative_to(REPO_ROOT)} (gitignored)")
    args = parser.parse_args()

    photos, skipped = find_photos(args.photos, args.include_screenshots)
    if not photos:
        sys.exit(f"No photos found under {args.photos} in subfolders {', '.join(FOLDERS)} "
                 f"(or {', '.join(FOLDER_ALIASES)}).")
    if skipped:
        print(f"[info] excluded {len(skipped)} screenshot PNGs (use --include-screenshots to keep them)")
    unnamed = [p["path"].name for p in photos if p["code"] == "?"]
    if unnamed:
        print(f"[warn] {len(unnamed)} file names have no photo code (A1-B3, N1-N4), condition unknown: {unnamed[:5]}")
    fingerprint, photo_hashes = photo_set_fingerprint(photos)
    print(f"[info] {len(photos)} photos, photo set fingerprint {fingerprint[:16]}…")
    run_name = args.run_name or f"{datetime.now():%Y%m%d_%H%M%S}_real_photos"
    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    images = [open_photo(p["path"]) for p in photos]
    results, top_boxes, thresholds = {}, {}, {}
    for name in args.models:
        print(f"[model] {name}")
        model = MODELS[name]()
        thresholds[name] = model.threshold
        for i, image in enumerate(images):
            for cat in CATEGORIES:
                dets = model.detect(image, cat)
                results[(i, name, cat)] = None if dets is None else [d["confidence"] for d in dets]
                top_boxes[(i, name, cat)] = max(dets, key=lambda d: d["confidence"]) if dets else None
        del model

    summary = summarise(photos, results, args.models, thresholds)
    (out_dir / "summary.md").write_text(markdown(summary, photos, run_name, fingerprint, skipped), encoding="utf-8")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_name": run_name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "num_photos": len(photos), "max_side_px": MAX_SIDE, "photo_set_sha256": fingerprint,
                   "excluded_screenshots": [f"{p.parent.name}/{p.name}" for p in skipped],
                   "models": summary, "photos": photo_hashes}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "per_photo.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["photo", "person", "code", "condition", "truth"] +
                        [f"{m}:{c}" for m in args.models for c in CATEGORIES])
        for i, p in enumerate(photos):
            row = [p["path"].name, p["person"], p["code"], p["condition"], p["truth"]]
            for m in args.models:
                for c in CATEGORIES:
                    confs = results[(i, m, c)]
                    row.append("n/a" if confs is None else (f"{max(confs):.3f}" if confs else "0"))
            writer.writerow(row)

    if args.sheets:
        sheet_dir = SHEETS_DIR / run_name
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(photos):
            asked = p["truth"] if p["truth"] != "none" else "keys"  # for "none" photos show the keys query
            boxes = {m: (top_boxes[(i, m, asked)] if top_boxes[(i, m, asked)] and
                         top_boxes[(i, m, asked)]["confidence"] >= thresholds[m] else None) for m in args.models}
            draw_sheet(images[i], boxes, sheet_dir / f"{p['truth']}__{p['path'].stem}.jpg", f"asked: {asked}")
        print(f"[ok] box pictures: {sheet_dir}")

    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"[ok] wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
