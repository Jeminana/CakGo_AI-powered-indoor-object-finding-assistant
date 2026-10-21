"""Why does YOLOE one-shot visual prompting do poorly on Member 4's real photos? Four diagnostics, run locally.

1. Official example: the docs' cross-image example (person box on bus.jpg, target zidane.jpg) in this environment.
2. Same photo: prompt and search the same photo (the "Intra-Image" mode of the authors' demo) for all 24 photos.
3. Score at the object: one example → each other photo of the same object, with conf 0.001. Highest score whose
   box centre is on the object vs. the highest score anywhere else in the photo.
4. Two examples: the visual embeddings of two photos averaged (then L2-normalised, as Ultralytics' validator does)
   vs. each single example, searching the remaining third photo.
Plus the cosine similarity of the visual embeddings: same object, same type / other object, other type, and
against the text embedding of the object word.

Uses only the documented path: `YOLOE.predict(..., refer_image=..., visual_prompts=...)` stores the embedding with
`set_classes`, then plain `predict` searches.

Example (from the repository root):
  python tests/zero_shot/diagnose_visual_prompt.py --photos tests/real_photos/photos \
      --boxes tests/results/real_photos/instance_one_shot_cheolyeon/gold_boxes.json --weights yoloe-26x-seg.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_one_shot as ios  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from ultralytics import YOLOE  # noqa: E402
from ultralytics.utils import ASSETS  # noqa: E402


def register(weights: str, image, box: list, device: str) -> YOLOE:
    model = YOLOE(weights)
    prompt = {"bboxes": np.array([box], dtype=float), "cls": np.array([0])}
    model.predict(image, refer_image=image, visual_prompts=prompt, conf=0.25, device=device, verbose=False)
    return model


def register_embedding(weights: str, vpe: torch.Tensor) -> YOLOE:
    model = YOLOE(weights)
    model.set_classes(["object0"], vpe)
    return model


def search(model: YOLOE, image, device: str, conf: float = 0.001) -> list:
    r = model.predict(image, conf=conf, device=device, verbose=False, max_det=300)[0]
    return list(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))


def object_vs_rest(dets: list, gold: list) -> tuple[float, float]:
    on = [s for s, b in dets if ios.centre_inside(b, gold)]
    off = [s for s, b in dets if not ios.centre_inside(b, gold)]
    return max(on, default=0.0), max(off, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--weights", default="yoloe-26x-seg.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=ios.REPO_ROOT / "tests/results/real_photos/instance_one_shot_cheolyeon")
    args = parser.parse_args()
    w, dev = args.weights, args.device
    gold = {k: v["box"] for k, v in json.loads(args.boxes.read_text()).items() if not k.startswith("_")}
    items = sorted(gold)
    photos = {n: ios.open_photo(args.photos / ios.photo_info(n)["object"] / n) for n in items}
    instances = {}
    for n in items:
        instances.setdefault(ios.photo_info(n)["instance"], []).append(n)
    out = [f"# Visual prompt diagnostics ({Path(w).name}, device {dev})", ""]

    # 1. Official example
    model = YOLOE(w)
    prompt = {"bboxes": np.array([[221.52, 405.8, 344.98, 857.54]]), "cls": np.array([0])}  # person on bus.jpg
    r = model.predict(str(ASSETS / "zidane.jpg"), refer_image=str(ASSETS / "bus.jpg"), visual_prompts=prompt,
                      conf=0.25, device=dev, verbose=False)[0]
    out += ["## 1. Official cross-image example (person box on bus.jpg, search zidane.jpg)", "",
            f"Boxes >= 0.25: {len(r.boxes)}; scores {[round(c, 2) for c in r.boxes.conf.tolist()]}", ""]

    # 2. Same photo
    hits = []
    for n in items:
        m = YOLOE(w)
        prompt = {"bboxes": np.array([gold[n]], dtype=float), "cls": np.array([0])}
        r = m.predict(photos[n], visual_prompts=prompt, conf=0.25, device=dev, verbose=False)[0]
        dets = list(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))
        best = max(dets, default=(0.0, None))
        hits.append(best[1] is not None and ios.centre_inside(best[1], gold[n]))
    out += ["## 2. Prompt and search the same photo (Intra-Image)", "",
            f"Top box on the object: {sum(hits)}/{len(hits)}", ""]

    # 3. Score at the object vs elsewhere, and embeddings
    vpes, rows = {}, []
    for n in items:
        m = register(w, photos[n], gold[n], dev)
        vpes[n] = m.model.pe.detach().float().cpu().clone()
        for q in instances[ios.photo_info(n)["instance"]]:
            if q != n:
                obj, rest = object_vs_rest(search(m, photos[q], dev), gold[q])
                rows.append((n, q, obj, rest))
    obj_scores = [r[2] for r in rows]
    out += ["## 3. One example → other photos of the same object (conf 0.001)", "",
            "| Example | Searched | Best score on the object | Best score elsewhere |", "|---|---|---|---|"]
    out += [f"| {n} | {q} | {o:.3f} | {e:.3f} |" for n, q, o, e in rows]
    out += ["", f"Median best score on the object: {statistics.median(obj_scores):.3f}; on the object >= 0.25 in "
                f"{sum(o >= 0.25 for o in obj_scores)}/{len(rows)}; object beats everything else in "
                f"{sum(o > e for _, _, o, e in rows)}/{len(rows)}", ""]

    # Embedding similarity
    def cos(a, b):
        return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)))
    groups = {"same object": [], "same type, other object": [], "other type": []}
    for a, b in itertools.combinations(items, 2):
        ia, ib = ios.photo_info(a), ios.photo_info(b)
        key = ("same object" if ia["instance"] == ib["instance"] else
               "same type, other object" if ia["object"] == ib["object"] else "other type")
        groups[key].append(cos(vpes[a], vpes[b]))
    tm = YOLOE(w)
    words = ios.CATEGORIES
    tpe = tm.get_text_pe([{"remote_control": "remote control"}.get(c, c) for c in words]).float().cpu()[0]
    own = [cos(vpes[n], tpe[words.index(ios.photo_info(n)["object"])]) for n in items]
    other = [cos(vpes[n], tpe[i]) for n in items for i, c in enumerate(words) if c != ios.photo_info(n)["object"]]
    out += ["## Cosine similarity of the visual embeddings", "", "| Pair | n | mean | min | max |", "|---|---|---|---|---|"]
    out += [f"| {k} | {len(v)} | {statistics.fmean(v):.3f} | {min(v):.3f} | {max(v):.3f} |" for k, v in groups.items()]
    out += [f"| visual vs text of its own word | {len(own)} | {statistics.fmean(own):.3f} | {min(own):.3f} | {max(own):.3f} |",
            f"| visual vs text of another word | {len(other)} | {statistics.fmean(other):.3f} | {min(other):.3f} | "
            f"{max(other):.3f} |", ""]

    # 4. Two examples averaged vs one
    single, double = [], []
    for inst, names in instances.items():
        for q in names:
            ex = [n for n in names if n != q]
            for n in ex:
                single.append(object_vs_rest(search(register_embedding(w, vpes[n]), photos[q], dev, 0.25), gold[q]))
            avg = F.normalize(torch.stack([vpes[n] for n in ex]).mean(0), dim=-1, p=2)
            double.append(object_vs_rest(search(register_embedding(w, avg), photos[q], dev, 0.25), gold[q]))
    hit = lambda pairs: sum(o > 0 and o >= e for o, e in pairs)  # noqa: E731  top box on the object at 0.25
    out += ["## 4. One example vs two examples averaged (search the third photo, conf 0.25)", "",
            f"Top box on the object: one example {hit(single)}/{len(single)}, two averaged {hit(double)}/{len(double)}",
            ""]

    text = "\n".join(out)
    (args.out / f"diagnostics_{Path(w).stem}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
