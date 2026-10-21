"""One-shot "register my item" test on Member 4's 28 real phone photos.

Photos (README inside the photo zip): 4 objects x designs A and B, where A and B are different physical objects,
3 photos each (A1 easy, A2 cluttered, A3 partly hidden, B1 far, B2 dim, B3 cluttered), plus 4 photos without any.

- YOLOE one-shot: for every instance and each of its 3 photos as the example (leave-one-out), YOLOE gets that photo
  with its gold box as a visual prompt (Ultralytics `refer_image`) and is asked about the other 27 photos.
- Zero-shot baselines on the same photos: Grounding DINO multi (the app) and YOLOE text, asked for the object word.

Gold boxes (--boxes) were made semi-automatically: Grounding DINO multi's top box checked by eye, or drawn on a
coordinate grid, then checked by the photographer. Because 18 of the 24 gold boxes are Grounding DINO's own top
boxes, a top box counts as "on the object" when its centre lies inside the gold box, not by IoU.

Example (from the repository root):
  python tests/zero_shot/instance_one_shot.py --photos tests/real_photos/photos \
      --boxes tests/results/real_photos/instance_one_shot_cheolyeon/gold_boxes.json --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also works under `python -I`
import detectors  # noqa: E402  (also puts grounding_dino/ on sys.path)

REPO_ROOT = detectors.REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "tests" / "real_photos"))
from score_real_photos import CONDITIONS, open_photo  # noqa: E402

CATEGORIES = ["wallet", "keys", "glasses", "remote_control"]
THRESHOLDS = [0.1, 0.25, 0.35, 0.5]
DEFAULT_THRESHOLD = {"gdino_multi": 0.35, "yoloe_text": 0.25, "yoloe_visual": 0.25}
NAME = re.compile(r"^(wallet|keys|glasses|remote_control|none)_.+_([ABN]\d)\.jpe?g$", re.I)


def photo_info(name: str) -> dict:
    m = NAME.match(name)
    if not m:
        raise ValueError(f"unexpected photo name {name}")
    obj, code = m.group(1).lower(), m.group(2).upper()
    return {"photo": name, "object": obj, "code": code, "design": code[0],
            "instance": f"{obj}_{code[0]}" if obj != "none" else None,
            "condition": CONDITIONS.get(code, "none photo")}


def centre_inside(box: list, gold: list) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return gold[0] <= cx <= gold[2] and gold[1] <= cy <= gold[3]


def top(detections: list) -> tuple[float, list | None]:
    """Highest-confidence (score, box), or (0.0, None) when nothing was returned."""
    best = max(detections, key=lambda d: d[0], default=None)
    return (best[0], best[1]) if best else (0.0, None)


def yoloe_boxes(result) -> list:
    b = result.boxes
    return list(zip(b.conf.tolist(), b.xyxy.tolist()))


def run_text_models(photos: dict, args) -> list[dict]:
    """Zero-shot: every photo asked for every object word. One row per (model, photo, word)."""
    rows = []
    models = {"gdino_multi": lambda: detectors.create("gdino", None, "multi", 0.05, args.device),
              **{f"yoloe_text:{w}": (lambda w=w: detectors.create("yoloe", str(args.weights_dir / w), "single",
                                                                  0.05, args.device)) for w in args.yoloe}}
    for key, make in models.items():
        detector = make()
        detector.prepare(CATEGORIES)
        for name, image in photos.items():
            for word in CATEGORIES:
                found = detector.detect(image, word)["detections"]
                score, box = top([(d["confidence"], d["bbox"]) for d in found])
                rows.append({"model": key, "example": "", "query": name, "word": word, "score": round(score, 4),
                             "box": box})
        print(f"[text] {key} done", flush=True)
    return rows


def run_visual(photos: dict, gold: dict, args) -> list[dict]:
    """One-shot: register each item photo (gold box) and ask about every other photo."""
    import numpy as np
    from ultralytics import YOLOE

    rows = []
    for w in args.yoloe:
        for example in (n for n in photos if photo_info(n)["object"] != "none"):
            model = YOLOE(str(args.weights_dir / w))
            prompt = {"bboxes": np.array([gold[example]], dtype=float), "cls": np.array([0])}
            model.predict(photos[example], refer_image=photos[example], visual_prompts=prompt, conf=0.05,
                          device=args.device, verbose=False)  # sets the class from the example
            for name, image in photos.items():
                if name == example:
                    continue
                score, box = top(yoloe_boxes(model.predict(image, conf=0.05, device=args.device, verbose=False)[0]))
                rows.append({"model": f"yoloe_visual:{w}", "example": example, "query": name,
                             "word": photo_info(example)["object"], "score": round(score, 4), "box": box})
        print(f"[visual] {w} done", flush=True)
    return rows


def relation(instance: str, query: dict) -> str:
    obj = instance.rsplit("_", 1)[0]
    if query["instance"] == instance:
        return "same object"
    if query["object"] == obj:
        return "same type, other object"
    return "no such object"


def evaluate(rows: list[dict], gold: dict, photos: list[str]) -> list[dict]:
    """Expand to one record per (model, registered instance, example, query), as the one-shot test sees it.
    Text models have no example: their answer for the instance's object word is reused for every example."""
    info = {n: photo_info(n) for n in photos}
    text = defaultdict(dict)
    for r in rows:
        if not r["example"]:
            text[r["model"]][(r["query"], r["word"])] = r
    records = []
    for r in rows:
        if r["example"]:
            records.append(dict(r))
    for model, answers in text.items():
        for example in (n for n in photos if info[n]["object"] != "none"):
            for q in photos:
                if q != example:
                    records.append({**answers[(q, info[example]["object"])], "model": model, "example": example})
    for rec in records:
        ex, q = info[rec["example"]], info[rec["query"]]
        rec.update({"instance": ex["instance"], "example_condition": ex["condition"], "query_condition": q["condition"],
                    "relation": relation(ex["instance"], q),
                    "on_object": bool(rec["box"]) and q["photo"] in gold and centre_inside(rec["box"], gold[q["photo"]])})
    return records


def model_family(model: str) -> str:
    return model.split(":")[0]


def summarise(records: list[dict]) -> str:
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model"]].append(r)
    lines = ["| Model | Conf | Same object found | ...box on it | Same type, other object: said yes | "
             "No such object: false alarms |", "|---|---|---|---|---|---|"]
    sweep = ["| Model | " + " | ".join(f"{t}" for t in THRESHOLDS) + " |", "|---|" + "---|" * len(THRESHOLDS)]

    def rates(rs, thr):
        same = [r for r in rs if r["relation"] == "same object"]
        other = [r for r in rs if r["relation"] == "same type, other object"]
        none = [r for r in rs if r["relation"] == "no such object"]
        yes = lambda r: r["score"] >= thr
        return (sum(map(yes, same)), sum(yes(r) and r["on_object"] for r in same), len(same),
                sum(map(yes, other)), len(other), sum(map(yes, none)), len(none))

    for model, rs in by_model.items():
        thr = DEFAULT_THRESHOLD[model_family(model)]
        f, on, n, o, no, fa, nfa = rates(rs, thr)
        lines.append(f"| {model} | {thr} | {f}/{n} ({f / n:.0%}) | {on}/{n} ({on / n:.0%}) | {o}/{no} ({o / no:.0%}) | "
                     f"{fa}/{nfa} ({fa / nfa:.0%}) |")
        cells = []
        for t in THRESHOLDS:
            f, on, n, o, no, fa, nfa = rates(rs, t)
            cells.append(f"{on / n:.0%} / {o / no:.0%} / {fa / nfa:.0%}")
        sweep.append(f"| {model} | " + " | ".join(cells) + " |")

    by_example = ["| Model | " + " | ".join(["A1 easy", "A2 cluttered", "A3 partly hidden", "B1 far", "B2 dim",
                                             "B3 cluttered"]) + " |", "|---|" + "---|" * 6]
    by_query = list(by_example)
    for model, rs in by_model.items():
        thr = DEFAULT_THRESHOLD[model_family(model)]
        same = [r for r in rs if r["relation"] == "same object"]
        for table, key in ((by_example, "example"), (by_query, "query")):
            if key == "example" and model_family(model) != "yoloe_visual":
                continue  # text models never see the example
            cells = []
            for code in ("A1", "A2", "A3", "B1", "B2", "B3"):
                sel = [r for r in same if photo_info(r[key])["code"] == code]
                hit = sum(r["score"] >= thr and r["on_object"] for r in sel)
                cells.append(f"{hit}/{len(sel)}")
            table.append(f"| {model} | " + " | ".join(cells) + " |")

    return "\n".join(
        ["### At each model's default threshold", "", *lines, "",
         "Same object found: the other photos of the registered object (2 per example). Box on it: the top box "
         "centre lies inside the gold box. Text models do not see the example; they answer the object word.", "",
         "### Threshold sweep: box on the same object / other object of the same type / false alarms", "", *sweep, "",
         "### Same object, box on it, by the example photo used for registration", "", *by_example, "",
         "### Same object, box on it, by the photo being searched", "", *by_query, ""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True, help="gold boxes JSON {photo: {box: [x1, y1, x2, y2]}}")
    parser.add_argument("--yoloe", nargs="+", default=["yoloe-26s-seg.pt", "yoloe-26x-seg.pt"])
    parser.add_argument("--weights-dir", type=Path, default=Path("."))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "tests/results/real_photos/instance_one_shot_cheolyeon")
    args = parser.parse_args()

    gold = {k: v["box"] for k, v in json.loads(args.boxes.read_text()).items() if not k.startswith("_")}
    names = sorted(p.name for p in args.photos.rglob("*.jpg"))
    photos = {}
    for n in names:
        folder = photo_info(n)["object"]
        photos[n] = open_photo(args.photos / folder / n)
    missing = [n for n in names if photo_info(n)["object"] != "none" and n not in gold]
    if missing:
        sys.exit(f"no gold box for {missing}")
    print(f"[run] {len(photos)} photos, {len(gold)} gold boxes", flush=True)

    rows = run_text_models(photos, args) + run_visual(photos, gold, args)
    records = evaluate(rows, gold, names)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["model", "instance", "example", "example_condition", "query", "query_condition", "relation", "word",
                  "score", "on_object", "box"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    text = summarise(records)
    (args.out / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
