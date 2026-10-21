"""SAM 3 on the valuables register-then-find test (same photos and gold as valuables_test.py).

- text: the object word (the documented PCS usage).
- exemplars (workaround, not an official API): SAM 3 takes exemplar boxes only inside the image it searches. So the
  registration crops are pasted into a strip added to the right of the search photo, their boxes are given as positive
  exemplars, and only detections whose centre lies in the original photo count. This is a variant of the "collage"
  used in the SAM 3 few-shot literature. Methods: one-shot (each registration crop alone), four-shot (all 4 crops),
  text + four-shot.
Scores come from Sam3Processor.post_process_object_detection (sigmoid(query) x sigmoid(presence), same as the
instance-segmentation path, without building masks). Default threshold 0.5 (transformers examples).

Example (from the repository root; SAM 3 snapshot folder downloaded beforehand):
  python tests/zero_shot/valuables_sam3.py --photos tests/real_photos/photos/valuables \
      --gold tests/results/real_photos/valuables_cheolyeon/gold.json --model-dir <sam3 snapshot> --dtype float16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instance_one_shot as ios  # noqa: E402
import valuables_test as vt  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

STRIP = 0.25  # strip width as a share of the photo's long side


def crop_example(image: Image.Image, box: list, margin: float = 0.15):
    """The registration object with a margin, and its box inside the crop."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    cx1, cy1 = max(0, round(x1 - mx)), max(0, round(y1 - my))
    cx2, cy2 = min(image.width, round(x2 + mx)), min(image.height, round(y2 + my))
    return image.crop((cx1, cy1, cx2, cy2)), [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]


def make_canvas(photo: Image.Image, examples: list) -> tuple[Image.Image, list]:
    """Photo on the left, a grey strip on the right holding the example crops stacked vertically."""
    w, h = photo.size
    s = round(STRIP * max(w, h))
    canvas = Image.new("RGB", (w + s, h), (128, 128, 128))
    canvas.paste(photo, (0, 0))
    cell = h // len(examples)
    boxes = []
    for i, (crop, b) in enumerate(examples):
        scale = min((s - 8) / crop.width, (cell - 8) / crop.height)
        c = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
        x0, y0 = w + (s - c.width) // 2, i * cell + (cell - c.height) // 2
        canvas.paste(c, (x0, y0))
        boxes.append([x0 + b[0] * scale, y0 + b[1] * scale, x0 + b[2] * scale, y0 + b[3] * scale])
    return canvas, boxes


class Sam3:
    def __init__(self, model_dir: str, device: str, dtype: torch.dtype):
        from transformers import Sam3Model, Sam3Processor

        self.proc = Sam3Processor.from_pretrained(model_dir)
        self.model = Sam3Model.from_pretrained(model_dir, dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype

    def detect(self, image: Image.Image, text: str | None = None, boxes: list | None = None) -> list:
        kwargs = {"images": image, "return_tensors": "pt"}
        if text:
            kwargs["text"] = text
        if boxes:
            kwargs["input_boxes"] = [boxes]
            kwargs["input_boxes_labels"] = [[1] * len(boxes)]
        inputs = self.proc(**kwargs).to(self.device)
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(self.dtype)
        with torch.inference_mode():
            out = self.model(**inputs)
        res = self.proc.post_process_object_detection(out, threshold=0.05,
                                                      target_sizes=inputs.get("original_sizes").tolist())[0]
        dets = list(zip(res["scores"].float().tolist(), res["boxes"].float().tolist()))
        del out
        if self.device == "mps":
            torch.mps.empty_cache()
        return dets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--model-dir", required=True, help="local facebook/sam3 snapshot folder")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--limit", type=int, default=None, help="queries per object (smoke test)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.gold.parent

    gold = json.loads(args.gold.read_text())
    words, photos = gold["words"], gold["photos"]
    used = {k: v for k, v in photos.items() if v["role"] != "hidden"}
    images = {k: ios.open_photo(args.photos / k) for k in used}
    sam = Sam3(args.model_dir, args.device, getattr(torch, args.dtype))
    model_name = f"SAM 3 ({args.dtype})"
    fields = ["method", "model", "object", "example", "query", "role", "score", "on_object", "box"]
    csv_path = out / "per_query_sam3.csv"
    done = set()
    if csv_path.exists():  # resume: every query is saved as soon as it is finished
        with open(csv_path, encoding="utf-8") as f:
            done = {(r["object"], r["query"]) for r in csv.DictReader(f)}
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    for obj in words:
        regs = sorted(k for k, v in used.items() if v["role"] == "register" and v["object"] == obj)
        crops = [crop_example(images[r], used[r]["box"]) for r in regs]
        queries = [k for k, v in used.items() if (v["role"] == "search" and v["object"] == obj)
                   or v["role"] in ("none", "other_keys")][:args.limit]
        for n, q in enumerate(queries, 1):
            if (obj, q) in done:
                continue
            photo = images[q]
            inside = lambda dets: [(s, b) for s, b in dets if (b[0] + b[2]) / 2 < photo.width]  # noqa: E731
            runs = [("text", "", sam.detect(photo, text=words[obj]))]
            for r, c in zip(regs, crops):
                canvas, boxes = make_canvas(photo, [c])
                runs.append(("one-shot (exemplar strip)", r, inside(sam.detect(canvas, boxes=boxes))))
            canvas, boxes = make_canvas(photo, crops)
            runs.append(("four-shot (exemplar strip)", "4 crops", inside(sam.detect(canvas, boxes=boxes))))
            runs.append(("text + four-shot (exemplar strip)", "4 crops",
                         inside(sam.detect(canvas, text=words[obj], boxes=boxes))))
            g = photos[q]
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                for method, example, dets in runs:
                    score, box = ios.top(dets)
                    writer.writerow({"method": method, "model": model_name, "object": obj, "example": example,
                                     "query": q, "role": g["role"], "score": round(score, 4), "box": box,
                                     "on_object": bool(box) and g["role"] == "search" and g["object"] == obj
                                     and ios.centre_inside(box, g["box"])})
            print(f"[{obj}] {n}/{len(queries)} {q}", flush=True)

    rows = []
    for name in ("per_query.csv", "per_query_sam3.csv"):  # earlier YOLOE / Grounding DINO answers, then SAM 3
        if (out / name).exists():
            with open(out / name, encoding="utf-8") as f:
                rows += [{**r, "score": float(r["score"]), "on_object": r["on_object"] == "True"}
                         for r in csv.DictReader(f)]
    text = vt.summarise(rows)
    (out / "tables_with_sam3.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
