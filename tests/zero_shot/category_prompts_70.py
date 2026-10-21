"""Category-level few-shot on the 70 shared images: YOLOE visual prompts, and Grounding DINO + DINOv3 verification.

Every photo of the 70 (tests/dataset/manifest.csv, Objects In House, CC0) shows a different object, so "register my
item" cannot be tested here. These tests use example photos the way YOLOE's visual prompts are meant to be used: as an
example of the *category*. Everything below was fixed before the run.

A. YOLOE-26s/26x visual prompts. The visual embedding of an example image comes from its labeled boxes of the
   category (Ultralytics `refer_image`). Each query asks for one category with one class, with the same settings
   as the text runs (conf 0.05, iou 0.7, imgsz 640, agnostic NMS):
   - one-shot: 15 runs. In run k the example of each category is its k-th image (the next one when the query is
     that image itself).
   - leave-one-out: the embeddings of the category's other 14 images averaged and L2-normalised.
   Scored with compare_zero_shot.summary at YOLOE's default 0.25, next to the YOLOE text runs.
B. Grounding DINO + DINOv3. Candidates: the saved boxes (>= 0.05) of the Grounding DINO multi run
   (t4_gdino-tiny_multi). References: DINOv3 crops (dino_verify.crop) of the category's labeled boxes in the other
   images. Similarity: the highest cosine with a reference ("max"), or the cosine with their normalised mean
   ("prototype"). Pipelines as in dino_verify.py: filter (top box >= 0.35, accepted if similar enough) and rerank
   (the 5 best boxes >= 0.05, the most similar one accepted if similar enough). "Similar enough" comes from the
   references alone: the 10th percentile of their own leave-one-image-out similarities. A fixed-threshold sweep and
   AUCs are reported too.

`run` (GPU) writes the YOLOE runs (predictions.jsonl in the usual format) and the DINOv3 embeddings; `report` scores
everything. Example, from the repository root:
  python tests/zero_shot/category_prompts_70.py run --emb-dir /content/emb --dinov3 /content/dinov3/dinov3-vitl16
  python tests/zero_shot/category_prompts_70.py report --emb-dir /content/emb
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_zero_shot as cz  # noqa: E402  (also puts tests/ and grounding_dino/ on sys.path)
import detectors  # noqa: E402
import dino_verify as dv  # noqa: E402

REPO = cz.REPO_ROOT
DATASET = REPO / "tests" / "dataset"
OUT = cz.RESULTS_DIR / "category_prompts"
GDINO_RUN = cz.RESULTS_DIR / "t4_gdino-tiny_multi"
TEXT_RUNS = {"yoloe-26s": "t4_yoloe-26s_single", "yoloe-26x": "t4_yoloe-26x_single"}
YOLOE_THRESHOLD, GDINO_THRESHOLD = 0.25, 0.35
SELF_PERCENTILE = 10


def load_gt():
    gt = json.loads(cz.GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    return gt["categories"], {g["image_id"]: g for g in gt["images"]}


def examples(images: dict, category: str) -> list[str]:
    return sorted(i for i, g in images.items() if g["target_category"] == category)


def boxes_of(g: dict, category: str) -> list:
    return [o["bbox_xyxy"] for o in g["objects"] if o["category"] == category]


def percentile(values: list, q: float) -> float:
    """Linear-interpolation percentile (numpy's default)."""
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo = int(k)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (k - lo)


def crop_list(images: dict, categories: list) -> list[dict]:
    """Every labeled box, and the top Grounding DINO multi boxes of every (image, category) query."""
    candidates = cz.read_jsonl(GDINO_RUN / "predictions.jsonl")
    crops = []
    for i, g in images.items():
        crops += [{"kind": "gold", "image_id": i, "category": o["category"], "rank": "", "score": "",
                   "box": json.dumps(o["bbox_xyxy"])} for o in g["objects"]]
        for c in categories:
            ds = sorted((d for d in candidates[i]["detections"] if d["label"] == c), key=lambda d: -d["confidence"])
            crops += [{"kind": "candidate", "image_id": i, "category": c, "rank": r, "score": d["confidence"],
                       "box": json.dumps(d["bbox"])} for r, d in enumerate(ds[:dv.TOP_K])]
    return crops


# ---------- run (GPU) ----------

def run(args) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    categories, images = load_gt()
    pil = {i: Image.open(DATASET / g["file_path"]).convert("RGB") for i, g in images.items()}
    manifest_sha = hashlib.sha256((DATASET / "manifest.csv").read_bytes()).hexdigest()
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLOE
    for w in args.yoloe:
        name = Path(w).stem.replace("-seg", "")
        one_dir, loo_dir = out / f"t4_{name}_visual_1shot", out / f"t4_{name}_visual_loo"
        if (loo_dir / "predictions.jsonl").exists() and len(list(one_dir.glob("run_*.jsonl"))) == 15:
            continue
        vpe = {}
        for c in categories:
            for e in examples(images, c):
                b = boxes_of(images[e], c)
                m = YOLOE(w)
                m.predict(pil[e], refer_image=pil[e], visual_prompts={"bboxes": np.array(b, dtype=float),
                                                                      "cls": np.zeros(len(b), dtype=int)},
                          conf=0.25, device=args.device, verbose=False)
                vpe[(c, e)] = m.model.pe.detach().float().cpu().clone()
        print(f"[yoloe] {name}: {len(vpe)} example embeddings", flush=True)
        model = YOLOE(w)

        def search(pe, ids, c, out):
            model.set_classes(["object0"], pe)
            for i in ids:
                t = time.perf_counter()
                r = model.predict(pil[i], conf=0.05, iou=detectors.SAME_OBJECT_IOU, imgsz=640, agnostic_nms=True,
                                  device=args.device, verbose=False)[0]
                ms = round((time.perf_counter() - t) * 1000, 2)
                raw = list(zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()))
                out[i]["detections"] += detectors.build_result(c, raw, pil[i].size)["detections"]
                out[i]["timing"]["wall_ms"][c] = ms

        def fresh():
            return {i: {"image_id": i, "file_path": g["file_path"], "target_category": g["target_category"],
                        "image_size": list(pil[i].size), "timing": {"wall_ms": {}}, "detections": []}
                    for i, g in images.items()}

        def save(path, preds, mode, note):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(p) + "\n" for p in preds.values())
            info = {"model": {"model_id": w, "family": "YOLOE-26 (Ultralytics, visual prompts, no fine-tuning)"},
                    "inference_settings": {"prompt_mode": mode, "box_threshold": 0.05,
                                           "operating_threshold": YOLOE_THRESHOLD, "device": args.device,
                                           "same_object_iou": detectors.SAME_OBJECT_IOU, "imgsz": 640,
                                           "agnostic_nms": True, "note": note},
                    "prompts": {c: "visual" for c in categories},
                    "dataset": {"manifest_sha256": manifest_sha}, "environment": env}
            (path.parent / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        for k in range(15):
            path = one_dir / f"run_{k:02d}.jsonl"
            if path.exists():
                continue
            preds = fresh()
            for c in categories:
                ex = examples(images, c)
                e, alt = ex[k], ex[(k + 1) % len(ex)]
                search(vpe[(c, e)], [i for i in images if i != e], c, preds)
                search(vpe[(c, alt)], [e], c, preds)
            save(path, preds, "visual one-shot", "run k: the k-th image of each category is the example")
            print(f"[yoloe] {name} one-shot run {k + 1}/15", flush=True)
        if not (loo_dir / "predictions.jsonl").exists():
            preds = fresh()
            for c in categories:
                ex = examples(images, c)
                avg = lambda keys: F.normalize(torch.cat([vpe[(c, x)] for x in keys], 1).mean(1, keepdim=True),  # noqa: E731
                                               dim=-1, p=2)
                for e in ex:
                    search(avg([x for x in ex if x != e]), [e], c, preds)
                search(avg(ex), [i for i in images if images[i]["target_category"] != c], c, preds)
            save(loo_dir / "predictions.jsonl", preds, "visual leave-one-out",
                 "the other images of the category averaged (L2-normalised)")
            print(f"[yoloe] {name} leave-one-out done", flush=True)
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.dinov3:
        from transformers import AutoImageProcessor, AutoModel

        crops = crop_list(images, categories)
        args.emb_dir.mkdir(parents=True, exist_ok=True)
        with open(args.emb_dir / "crops.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "kind", "image_id", "category", "rank", "score", "box"])
            writer.writeheader()
            writer.writerows({"id": n, **c} for n, c in enumerate(crops))
        for model_dir in args.dinov3:
            name = Path(model_dir).name
            proc = AutoImageProcessor.from_pretrained(model_dir)
            model = AutoModel.from_pretrained(model_dir).to(args.device).eval()
            embs = []
            for s in range(0, len(crops), 32):
                batch = [dv.crop(pil[c["image_id"]], json.loads(c["box"])) for c in crops[s:s + 32]]
                inputs = proc(images=batch, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    embs.append(F.normalize(model(**inputs).pooler_output.float(), dim=-1).cpu().numpy())
            np.save(args.emb_dir / f"emb_{name}.npy", np.concatenate(embs))
            print(f"[dinov3] {name}: {len(crops)} crops", flush=True)
            del model


# ---------- report ----------

def ratio(pair) -> str:
    return f"{pair[0]}/{pair[1]}"


def verify_records(emb_dir: Path, name: str, categories: list, images: dict, agg: str) -> list[dict]:
    import numpy as np

    emb = np.load(emb_dir / f"emb_{name}.npy")
    crops = list(csv.DictReader(open(emb_dir / "crops.csv", encoding="utf-8")))
    gold, cand = defaultdict(list), defaultdict(list)
    for c in crops:
        if c["kind"] == "gold":
            gold[c["category"]].append((c["image_id"], int(c["id"]), json.loads(c["box"])))
        else:
            cand[(c["image_id"], c["category"])].append((int(c["rank"]), float(c["score"]), json.loads(c["box"]),
                                                         int(c["id"])))

    def scorer(refs):
        r = emb[[k for _, k, _ in refs]]
        if agg == "max":
            return lambda x: float((r @ x).max())
        p = r.mean(0)
        p = p / np.linalg.norm(p)
        return lambda x: float(p @ x)

    records = []
    for i, g in images.items():
        present = {o["category"] for o in g["objects"]}
        for c in categories:
            refs = [t for t in gold[c] if t[0] != i]
            sim = scorer(refs)
            own = [scorer([t for t in refs if t[0] != img])(emb[k]) for img, k, _ in refs]
            cs = sorted(cand[(i, c)])
            d = dv.decide([(s, b) for _, s, b, _ in cs], [sim(emb[k]) for _, _, _, k in cs])
            gold_sims = [sim(emb[k]) for img, k, _ in gold[c] if img == i]
            records.append({"image_id": i, "category": c, "present": c in present, "tau": percentile(own, SELF_PERCENTILE),
                            "gold_sim": max(gold_sims) if gold_sims else None, **d})
    return records


def as_predictions(records: list, images: dict, pipeline: str, tau=None) -> dict:
    """Accepted boxes only, with confidence 1, so compare_zero_shot.summary at any threshold < 1 scores the decision."""
    preds = {i: {"image_id": i, "detections": []} for i in images}
    for r in records:
        ok, box = dv.accepted(r, pipeline, r["tau"] if tau is None else tau)
        if ok:
            preds[r["image_id"]]["detections"].append({"label": r["category"], "confidence": 1.0, "bbox": box})
    return preds


def report(args) -> None:
    import numpy as np

    categories, images = load_gt()
    gt_all = images
    gt_clean = {i: g for i, g in images.items() if i not in cz.GT_ISSUES}
    lines = ["# Category-level few-shot on the 70 shared images", ""]

    def summary(preds, gt, thr):
        return cz.summary({"predictions": preds}, gt, categories, thr)

    def per_category(preds, thr):
        return " | ".join(cz.top1_rate(c, preds, gt_all, thr) for c in categories)

    # A. YOLOE visual prompts
    lines += ["## A. YOLOE visual prompts (threshold 0.25)", "",
              "| Model | Prompt | Top-1 correct (of 60) | False alarms (of 220) | Negatives with a box (of 40) | "
              "Mean AP50 | Top-1 without the 12 unlabeled-target images (of 48) | False alarms without them |",
              "|---|---|---|---|---|---|---|---|"]
    per_cat = ["", "Top-1 correct per category (wallet | keys | glasses | remote control):", "",
               "| Model | Prompt | wallet | keys | glasses | remote control |", "|---|---|---|---|---|---|"]
    sweep = ["", "Threshold sweep, top-1 correct / false alarms (all 70 images):", "",
             "| Model | Prompt | " + " | ".join(str(t) for t in cz.THRESHOLDS) + " |",
             "|---|---|" + "---|" * len(cz.THRESHOLDS)]
    for name, text_run in TEXT_RUNS.items():
        loo = args.out / f"t4_{name}_visual_loo" / "predictions.jsonl"
        if not loo.exists():
            continue
        variants = [("text (earlier run)", cz.read_jsonl(cz.RESULTS_DIR / text_run / "predictions.jsonl")),
                    ("visual, leave-one-out (14 examples)", cz.read_jsonl(loo))]
        for prompt, preds in variants:
            a, b = summary(preds, gt_all, YOLOE_THRESHOLD), summary(preds, gt_clean, YOLOE_THRESHOLD)
            lines.append(f"| {name} | {prompt} | {ratio(a['top1'])} | {ratio(a['false_alarms'])} | "
                         f"{ratio(a['negatives_fp'])} | {cz.fmt(a['mean_ap50'])} | {ratio(b['top1'])} | "
                         f"{ratio(b['false_alarms'])} |")
            per_cat.append(f"| {name} | {prompt} | {per_category(preds, YOLOE_THRESHOLD)} |")
            sweep.append(f"| {name} | {prompt} | " + " | ".join(
                f"{summary(preds, gt_all, t)['top1'][0]} / {summary(preds, gt_all, t)['false_alarms'][0]}"
                for t in cz.THRESHOLDS) + " |")
        runs = [cz.read_jsonl(p) for p in sorted((args.out / f"t4_{name}_visual_1shot").glob("run_*.jsonl"))]
        stats = [summary(p, gt_all, YOLOE_THRESHOLD) for p in runs]
        top1 = [s["top1"][0] for s in stats]
        fa = [s["false_alarms"][0] for s in stats]
        neg = [s["negatives_fp"][0] for s in stats]
        aps = [s["mean_ap50"] for s in stats if s["mean_ap50"] is not None]
        lines.append(f"| {name} | visual, one-shot (mean of {len(runs)} runs, min–max) | "
                     f"{np.mean(top1):.1f}/60 ({min(top1)}–{max(top1)}) | {np.mean(fa):.1f}/220 ({min(fa)}–{max(fa)}) | "
                     f"{np.mean(neg):.1f}/40 | {np.mean(aps):.3f} | - | - |")
        sweep.append(f"| {name} | visual, one-shot (mean) | " + " | ".join(
            f"{np.mean([summary(p, gt_all, t)['top1'][0] for p in runs]):.1f} / "
            f"{np.mean([summary(p, gt_all, t)['false_alarms'][0] for p in runs]):.1f}" for t in cz.THRESHOLDS) + " |")
    lines += per_cat + sweep + [""]

    # B. Grounding DINO + DINOv3
    gdino = cz.read_jsonl(GDINO_RUN / "predictions.jsonl")
    base, base_clean = summary(gdino, gt_all, GDINO_THRESHOLD), summary(gdino, gt_clean, GDINO_THRESHOLD)
    lines += ["## B. Grounding DINO multi + DINOv3 verification", "",
              "Threshold: from the references (10th percentile of their own leave-one-image-out similarities).", "",
              "| Verifier | Similarity | Pipeline | Top-1 correct (of 60) | False alarms (of 220) | "
              "Negatives with a box (of 40) | Top-1 without the 12 (of 48) | False alarms without them | "
              "Median threshold |", "|---|---|---|---|---|---|---|---|---|",
              f"| - | - | Grounding DINO alone (0.35) | {ratio(base['top1'])} | {ratio(base['false_alarms'])} | "
              f"{ratio(base['negatives_fp'])} | {ratio(base_clean['top1'])} | {ratio(base_clean['false_alarms'])} | - |"]
    sweep_b = ["", "Fixed-threshold sweep, top-1 correct / false alarms (all 70 images):", "",
               "| Verifier | Similarity | Pipeline | " + " | ".join(str(t) for t in dv.TAUS[::2]) + " |",
               "|---|---|---|" + "---|" * len(dv.TAUS[::2])]
    aucs = ["", "Verifier alone (AUC): labeled object vs the most similar box in images without the category:", "",
            "| Verifier | Similarity | AUC | Median labeled object | Median best box without the category |",
            "|---|---|---|---|---|"]
    rows = []
    for emb in sorted(args.emb_dir.glob("emb_*.npy")):
        name = emb.stem[4:]
        for agg in ("max", "prototype"):
            recs = verify_records(args.emb_dir, name, categories, images, agg)
            rows += [{**r, "verifier": name, "similarity": agg} for r in recs]
            for p in ("filter", "rerank"):
                preds = as_predictions(recs, images, p)
                a, b = summary(preds, gt_all, 0.5), summary(preds, gt_clean, 0.5)
                lines.append(f"| {name} | {agg} | {p} | {ratio(a['top1'])} | {ratio(a['false_alarms'])} | "
                             f"{ratio(a['negatives_fp'])} | {ratio(b['top1'])} | {ratio(b['false_alarms'])} | "
                             f"{np.median([r['tau'] for r in recs]):.2f} |")
                sweep_b.append(f"| {name} | {agg} | {p} | " + " | ".join(
                    f"{summary(as_predictions(recs, images, p, t), gt_all, 0.5)['top1'][0]} / "
                    f"{summary(as_predictions(recs, images, p, t), gt_all, 0.5)['false_alarms'][0]}"
                    for t in dv.TAUS[::2]) + " |")
            pos = [r["gold_sim"] for r in recs if r["gold_sim"] is not None and images[r["image_id"]]["target_category"]
                   == r["category"]]
            neg = [r["rr_sim"] for r in recs if not r["present"] and r["rr_sim"] is not None]
            aucs.append(f"| {name} | {agg} | {dv.auc(pos, neg):.2f} | {np.median(pos):.2f} | {np.median(neg):.2f} |")
    lines += sweep_b + aucs + [""]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    fields = ["verifier", "similarity", "image_id", "category", "present", "tau", "gold_sim", "top_score", "top_sim",
              "top_box", "rr_score", "rr_sim", "rr_box"]
    with open(args.out / "dinov3_per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else
                                 round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in fields})
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--yoloe", nargs="*", default=["yoloe-26s-seg.pt", "yoloe-26x-seg.pt"])
    r.add_argument("--dinov3", nargs="*", default=[], help="local DINOv3 snapshot folders")
    r.add_argument("--emb-dir", type=Path, required=True, help="where the DINOv3 crops and embeddings go")
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--out", type=Path, default=OUT)
    p = sub.add_parser("report")
    p.add_argument("--emb-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
