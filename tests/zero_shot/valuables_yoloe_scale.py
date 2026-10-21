"""Does closing the size gap help YOLOE few-shot? Valuables photos, YOLOE-26x, same queries and scoring as
valuables_test.py.

The registration close-ups show the item 3-10x larger than the search photos do, and YOLOE sees the 1280 px app
frame at 640 px. Fixes tried, fixed before the run:
- Input resolution: registration and search at imgsz 1280 (the app frame's long side), against the default 640.
- Multi-scale registration: each registration photo is also shrunk so the item's long side is 320, 160, 80 and
  40 px (app-frame pixels), centred on a grey canvas of the original size. The 5 embeddings are used two ways:
  averaged (L2-normalised, as Ultralytics' validator does), or kept as 5 prototypes (one class each, so the top box
  is the best match over the sizes).
- Scale-matched registration (diagnostic upper bound, uses the gold boxes): the item shrunk to the median size of
  that object in the search photos.
The plain one-shot / four-shot / text rows at 640 repeat valuables_test.py on this device as the baseline.

Example (from the repository root):
  python tests/zero_shot/valuables_yoloe_scale.py --photos tests/real_photos/photos/valuables \
      --gold tests/real_photos/private_results/valuables/gold.json --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_one_shot as ios  # noqa: E402
import valuables_test as vt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLOE  # noqa: E402

SIZES = [320, 160, 80, 40]  # item long side after shrinking, app-frame pixels
GREY = (114, 114, 114)  # Ultralytics' letterbox padding colour


def long_side(box: list) -> float:
    return max(box[2] - box[0], box[3] - box[1])


def shrink(image: Image.Image, box: list, side: float) -> tuple[Image.Image, list]:
    """The photo shrunk so the item's long side is `side` px, centred on a grey canvas of the original size."""
    s = side / long_side(box)
    if s >= 1:
        return image, box
    small = image.resize((max(1, round(image.width * s)), max(1, round(image.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", image.size, GREY)
    ox, oy = (image.width - small.width) // 2, (image.height - small.height) // 2
    canvas.paste(small, (ox, oy))
    return canvas, [ox + box[0] * s, oy + box[1] * s, ox + box[2] * s, oy + box[3] * s]


def embed(weights: str, image: Image.Image, box: list, device: str, imgsz: int) -> torch.Tensor:
    """Visual prompt embedding of one box, by the documented refer_image path. Shape (1, 1, D)."""
    model = YOLOE(weights)
    prompt = {"bboxes": np.array([box], dtype=float), "cls": np.array([0])}
    model.predict(image, refer_image=image, visual_prompts=prompt, conf=0.25, device=device, imgsz=imgsz,
                  verbose=False)
    return model.model.pe.detach().float().cpu().clone()


def average(pes: list) -> torch.Tensor:
    return F.normalize(torch.cat(pes, dim=1).mean(1, keepdim=True), dim=-1, p=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--weights", default="yoloe-26x-seg.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.gold.parent
    w, dev = args.weights, args.device

    gold = json.loads(args.gold.read_text())
    words, photos = gold["words"], gold["photos"]
    used = {k: v for k, v in photos.items() if v["role"] != "hidden"}
    images = {k: ios.open_photo(args.photos / k) for k in used}
    queries = {o: [k for k, v in used.items() if (v["role"] == "search" and v["object"] == o)
                   or v["role"] in ("none", "other_keys")] for o in words}
    rows = []

    def ask(method, obj, example, pes, imgsz):
        model = YOLOE(w)
        pe = torch.cat(pes, dim=1)
        model.set_classes([f"object{i}" for i in range(pe.shape[1])], pe)
        for q in queries[obj]:
            r = model.predict(images[q], conf=0.05, device=dev, imgsz=imgsz, verbose=False)[0]
            score, box = ios.top(ios.yoloe_boxes(r))
            rows.append({"method": f"{method} @{imgsz}", "model": "YOLOE-26x", "object": obj, "example": example,
                         "query": q, "role": used[q]["role"], "score": round(score, 4), "box": box})

    for imgsz in (640, 1280):
        for obj in words:
            regs = sorted(k for k, v in used.items() if v["role"] == "register" and v["object"] == obj)
            matched = statistics.median(long_side(v["box"]) for v in used.values()
                                        if v["role"] == "search" and v["object"] == obj)
            base, scaled, oracle = {}, {}, {}
            for r in regs:
                image, box = images[r], used[r]["box"]
                base[r] = embed(w, image, box, dev, imgsz)
                scaled[r] = [embed(w, *shrink(image, box, s), dev, imgsz) for s in SIZES]
                oracle[r] = embed(w, *shrink(image, box, matched), dev, imgsz)
                ask("one-shot", obj, r, [base[r]], imgsz)
                ask("one-shot multi-scale avg", obj, r, [average([base[r], *scaled[r]])], imgsz)
                ask("one-shot multi-scale max", obj, r, [base[r], *scaled[r]], imgsz)
                ask("one-shot scale-matched (oracle)", obj, r, [oracle[r]], imgsz)
            ask("four-shot", obj, "average of 4", [average(list(base.values()))], imgsz)
            ask("four-shot multi-scale avg", obj, "average of 4 x 5 sizes",
                [average([e for r in regs for e in (base[r], *scaled[r])])], imgsz)
            ask("four-shot multi-scale max", obj, "5 sizes, each the average of 4",
                [average(list(base.values()))] + [average([scaled[r][i] for r in regs]) for i in range(len(SIZES))],
                imgsz)
            ask("four-shot scale-matched (oracle)", obj, f"average of 4 at {matched:.0f} px",
                [average(list(oracle.values()))], imgsz)
            text = YOLOE(w)
            text.set_classes([words[obj]])
            for q in queries[obj]:
                r = text.predict(images[q], conf=0.05, device=dev, imgsz=imgsz, verbose=False)[0]
                score, box = ios.top(ios.yoloe_boxes(r))
                rows.append({"method": f"text @{imgsz}", "model": "YOLOE-26x", "object": obj, "example": "",
                             "query": q, "role": used[q]["role"], "score": round(score, 4), "box": box})
            print(f"[imgsz {imgsz}] {obj} done", flush=True)

    for r in rows:
        g = photos[r["query"]]
        r["on_object"] = bool(r["box"]) and g["role"] == "search" and g["object"] == r["object"] and \
            ios.centre_inside(r["box"], g["box"])
    with open(out / "per_query_yoloe_scale.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "model", "object", "example", "query", "role", "score",
                                               "on_object", "box"])
        writer.writeheader()
        writer.writerows(rows)
    text = vt.summarise(rows)
    (out / "tables_yoloe_scale.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
