"""Register-then-find test on Member 4's valuables photos (sunglasses, earbud, keys).

Gold file (--gold): every photo with its role. register = close-up example of the object (4 per object); search = the
object is visible; none = no valuables (the search scenes again, objects removed); other_keys = only another key set;
hidden = excluded, the object is in the scene but not visible in the photo.

Methods, all asked about the same photos (the search photos of the object, the 10 none photos, the 3 other_keys
photos):
- YOLOE one-shot: one registration photo + its gold box (`refer_image`), each of the 4 in turn.
- YOLOE four-shot: the 4 registration embeddings averaged and L2-normalised, as Ultralytics' validator does.
- YOLOE text: the object word.
- Grounding DINO single: the object word.
- Grounding DINO multi (the app's setting): keys and sunglasses only, the words configured for them.
A top box counts as "on the object" when its centre lies inside the gold box (some gold boxes came from Grounding
DINO's own candidates).

Example (from the repository root):
  python tests/zero_shot/valuables_test.py --photos tests/real_photos/photos/valuables \
      --gold tests/results/real_photos/valuables_cheolyeon/gold.json --weights-dir . --device mps
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_one_shot as ios  # noqa: E402
from diagnose_visual_prompt import register, register_embedding  # noqa: E402

THRESHOLDS = [0.05, 0.1, 0.25, 0.35, 0.5]


def default_threshold(model: str) -> float:
    """Each library's default: SAM 3 0.5 (transformers examples), Grounding DINO 0.35 (app), YOLOE 0.25."""
    return 0.5 if "SAM" in model else 0.35 if "DINO" in model else 0.25
APP_CATEGORY = {"keys": "keys", "sunglasses": "glasses"}  # grounding_dino/config/prompts.json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--yoloe", nargs="+", default=["yoloe-26s-seg.pt", "yoloe-26x-seg.pt"])
    parser.add_argument("--weights-dir", type=Path, default=Path("."))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.gold.parent

    import torch
    import torch.nn.functional as F
    from ultralytics import YOLOE

    gold = json.loads(args.gold.read_text())
    words, photos = gold["words"], gold["photos"]
    used = {k: v for k, v in photos.items() if v["role"] != "hidden"}
    images = {k: ios.open_photo(args.photos / k) for k in used}
    objects = list(words)
    queries = {o: [k for k, v in used.items() if (v["role"] == "search" and v["object"] == o)
                   or v["role"] in ("none", "other_keys")] for o in objects}
    print(f"[run] {len(images)} photos; queries per object: { {o: len(q) for o, q in queries.items()} }", flush=True)

    rows = []

    def ask(method, model_name, obj, example, search_fn):
        for q in queries[obj]:
            score, box = ios.top(search_fn(images[q]))
            rows.append({"method": method, "model": model_name, "object": obj, "example": example, "query": q,
                         "role": used[q]["role"], "score": round(score, 4), "box": box})

    def yoloe_search(model):
        return lambda im: ios.yoloe_boxes(model.predict(im, conf=0.05, device=args.device, verbose=False)[0])

    for w in args.yoloe:
        path = str(args.weights_dir / w)
        name = Path(w).stem.replace("-seg", "").replace("yoloe-", "YOLOE-")
        for obj in objects:
            regs = sorted(k for k, v in used.items() if v["role"] == "register" and v["object"] == obj)
            vpes = []
            for r in regs:
                model = register(path, images[r], used[r]["box"], args.device)
                vpes.append(model.model.pe.detach().float().cpu().clone())
                ask("one-shot", name, obj, r, yoloe_search(model))
            avg = F.normalize(torch.stack(vpes).mean(0), dim=-1, p=2)
            ask("four-shot", name, obj, "average of 4", yoloe_search(register_embedding(path, avg)))
            text = YOLOE(path)
            text.set_classes([words[obj]])
            ask("text", name, obj, "", yoloe_search(text))
        print(f"[yoloe] {w} done", flush=True)

    from detector import GroundingDinoDetector  # grounding_dino/, on sys.path via instance_one_shot

    single = GroundingDinoDetector(model_id="IDEA-Research/grounding-dino-tiny", device=args.device, box_threshold=0.05,
                                   prompts={}, prompt_mode="single")
    multi = GroundingDinoDetector.from_config(device=args.device, box_threshold=0.05, prompt_mode="multi")
    gd = lambda det, word: (lambda im: [(d["confidence"], d["bbox"]) for d in det.detect(im, word)["detections"]])  # noqa: E731
    for obj in objects:
        ask("text", "Grounding DINO single", obj, "", gd(single, words[obj]))
        if obj in APP_CATEGORY:
            ask("text (app config)", "Grounding DINO multi", obj, "", gd(multi, APP_CATEGORY[obj]))
    print("[gdino] done", flush=True)

    for r in rows:
        g = photos[r["query"]]
        r["on_object"] = bool(r["box"]) and g["role"] == "search" and g["object"] == r["object"] and \
            ios.centre_inside(r["box"], g["box"])
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "model", "object", "example", "query", "role", "score",
                                               "on_object", "box"])
        writer.writeheader()
        writer.writerows(rows)
    text = summarise(rows)
    (out / "tables.md").write_text(text, encoding="utf-8")
    print(text)


def summarise(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["model"])].append(r)

    def stats(rs, thr, obj=None):
        rs = [r for r in rs if obj is None or r["object"] == obj]
        search = [r for r in rs if r["role"] == "search"]
        none = [r for r in rs if r["role"] == "none" or (r["role"] == "other_keys" and r["object"] != "keys")]
        other = [r for r in rs if r["role"] == "other_keys" and r["object"] == "keys"]
        on = sum(r["score"] >= thr and r["on_object"] for r in search)
        fa = sum(r["score"] >= thr for r in none)
        ok = sum(r["score"] >= thr for r in other)
        return on, len(search), fa, len(none), ok, len(other)

    def pct(a, b):
        return f"{a}/{b} ({a / b:.0%})" if b else "-"

    head = ["| Method | Model | Conf | Box on the object (search photos) | False alarms (none photos) | "
            "Other keys said yes |", "|---|---|---|---|---|---|"]
    lines, per_obj, sweep = list(head), [], []
    for (method, model), rs in groups.items():
        thr = default_threshold(model)
        on, n, fa, nfa, ok, nok = stats(rs, thr)
        lines.append(f"| {method} | {model} | {thr} | {pct(on, n)} | {pct(fa, nfa)} | {pct(ok, nok)} |")
        cells = []
        for obj in ("sunglasses", "earbud", "keys"):
            o_on, o_n, o_fa, o_nfa, _, _ = stats(rs, thr, obj)
            cells.append(f"{o_on}/{o_n}, FA {o_fa}/{o_nfa}" if o_n else "-")
        per_obj.append(f"| {method} | {model} | " + " | ".join(cells) + " |")
        sw = []
        for t in THRESHOLDS:
            on, n, fa, nfa, _, _ = stats(rs, t)
            sw.append(f"{on / n:.0%} / {fa / nfa:.0%}" if n and nfa else "-")
        sweep.append(f"| {method} | {model} | " + " | ".join(sw) + " |")
    return "\n".join(
        ["### All objects, each method at its default threshold", "", *lines, "",
         "One-shot rows pool the 4 registration photos (each searched separately).", "",
         "### Per object (box on the object / search photos, false alarms / none photos)", "",
         "| Method | Model | Sunglasses | Earbud | Keys |", "|---|---|---|---|---|", *per_obj, "",
         "### Threshold sweep, all objects: box on the object / false alarms", "",
         "| Method | Model | " + " | ".join(str(t) for t in THRESHOLDS) + " |",
         "|---|---|" + "---|" * len(THRESHOLDS), *sweep, ""])


if __name__ == "__main__":
    main()
