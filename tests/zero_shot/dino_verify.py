"""Grounding DINO + DINOv3 verification: does comparing the found box with the registration photos help find *my* item?

Stage 1, candidates: Grounding DINO tiny, top 10 boxes at box threshold 0.05. Modes: "multi" (the app's setting, for
app objects) and "single" (the object word alone). "app" below = multi for app objects, single for the earbud (not an
app object).
Stage 2, verification: each box is cropped (10% margin), padded to a square with the mean colour, and embedded by
DINOv3 (CLS token = pooler_output, L2-normalised; the processor resizes to 224 x 224). The similarity is the cosine
with the registration crops (gold boxes), the highest over the registrations used.

Pipelines, all fixed before the run:
- Grounding DINO alone: top box with score >= 0.35 (the app's default).
- filter: that top box, accepted only if its similarity >= tau.
- rerank: the top 5 boxes with score >= 0.05; the most similar one is chosen and accepted if its similarity >= tau.
tau is swept from 0.10 to 0.80. The operating tau of each model and pipeline is chosen on the 28 photos (largest
"same object found - other object said yes - false alarms") and applied unchanged to the valuables photos.

Datasets:
- valuables: Member 4's valuables photos (valuables_cheolyeon/gold.json). 4 close-up registrations per object;
  one-shot (each alone) and four-shot (highest of the 4). Same queries and roles as valuables_test.py.
- c28: Member 4's 28 photos (instance_one_shot_cheolyeon/gold_boxes.json). One-shot, leave-one-out: each photo with a
  gold box registers its object, the other 27 photos are searched (as in instance_one_shot.py).

`run` computes the candidates and embeddings (GPU); `report` scores them (no model). Examples, from the repository root:
  python tests/zero_shot/dino_verify.py run --out /content/dino_verify --dinov3 /content/dinov3-vitl16 /content/dinov3-vitb16
  python tests/zero_shot/dino_verify.py report --run <downloaded run folder> --out tests/results/real_photos/dino_verify_cheolyeon
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

REPO = ios.REPO_ROOT
VAL_PHOTOS = REPO / "tests/real_photos/photos/valuables"
VAL_GOLD = REPO / "tests/results/real_photos/valuables_cheolyeon/gold.json"
C28_PHOTOS = REPO / "tests/real_photos/photos"
C28_GOLD = REPO / "tests/results/real_photos/instance_one_shot_cheolyeon/gold_boxes.json"
APP_CATEGORY = {"keys": "keys", "sunglasses": "glasses"}  # valuables object -> app category (prompts.json)
C28_WORDS = {"wallet": "wallet", "keys": "keys", "glasses": "glasses", "remote_control": "remote control"}
TOP_K, RERANK_K, FLOOR, APP_THRESHOLD = 10, 5, 0.05, 0.35
MARGIN, MEAN_RGB = 0.10, (124, 116, 104)
TAUS = [round(0.10 + 0.05 * i, 2) for i in range(15)]
CANDIDATE_FIELDS = ["dataset", "mode", "object", "photo", "rank", "score", "box"]


# ---------- data ----------

def load_valuables():
    gold = json.loads(VAL_GOLD.read_text())
    photos = {k: v for k, v in gold["photos"].items() if v["role"] != "hidden"}
    queries = {o: [k for k, v in photos.items() if (v["role"] == "search" and v["object"] == o)
                   or v["role"] in ("none", "other_keys")] for o in gold["words"]}
    return gold["words"], photos, queries


def load_c28():
    gold = {k: v["box"] for k, v in json.loads(C28_GOLD.read_text()).items() if not k.startswith("_")}
    names = sorted(p.name for p in C28_PHOTOS.glob("*/*.jpg") if ios.NAME.match(p.name))
    return gold, names


def c28_path(name: str) -> Path:
    return C28_PHOTOS / ios.photo_info(name)["object"] / name


def crop(image, box: list, margin: float = MARGIN):
    """Box with a margin, padded to a square with the mean colour (the processor would stretch it otherwise)."""
    from PIL import Image

    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    l, t = max(0, round(x1 - mx)), max(0, round(y1 - my))
    r, b = min(image.width, max(l + 1, round(x2 + mx))), min(image.height, max(t + 1, round(y2 + my)))
    c = image.crop((l, t, r, b))
    side = max(c.size)
    square = Image.new("RGB", (side, side), MEAN_RGB)
    square.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
    return square


# ---------- run (GPU) ----------

def run(args) -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from detector import GroundingDinoDetector  # detection/experiments/grounding_dino, on sys.path via ios
    from transformers import AutoImageProcessor, AutoModel

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    words, vphotos, vqueries = load_valuables()
    c28_gold, c28_names = load_c28()
    images = {("valuables", k): ios.open_photo(VAL_PHOTOS / k) for k in vphotos}
    images.update({("c28", n): ios.open_photo(c28_path(n)) for n in c28_names})

    jobs = []  # (dataset, mode, object, photo, prompt target)
    for o in words:
        for q in vqueries[o]:
            jobs.append(("valuables", "single", o, q, words[o]))
            if o in APP_CATEGORY:
                jobs.append(("valuables", "multi", o, q, APP_CATEGORY[o]))
    for n in c28_names:
        for cat in ios.CATEGORIES:
            jobs.append(("c28", "multi", cat, n, cat))
            jobs.append(("c28", "single", cat, n, C28_WORDS[cat]))

    path = out / "candidates.csv"
    done = set()
    if path.exists():  # resume: every query is written as soon as it is answered
        with open(path, encoding="utf-8") as f:
            done = {(r["dataset"], r["mode"], r["object"], r["photo"]) for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS).writeheader()
    detectors = {
        "single": GroundingDinoDetector(model_id="IDEA-Research/grounding-dino-tiny", device=args.device,
                                        box_threshold=FLOOR, prompts={}, prompt_mode="single"),
        "multi": GroundingDinoDetector.from_config(device=args.device, box_threshold=FLOOR, prompt_mode="multi"),
    }
    for i, (ds, mode, obj, photo, target) in enumerate(jobs, 1):
        if (ds, mode, obj, photo) in done:
            continue
        found = detectors[mode].detect(images[(ds, photo)], target)["detections"][:TOP_K]
        rows = [{"dataset": ds, "mode": mode, "object": obj, "photo": photo, "rank": r, "score": d["confidence"],
                 "box": json.dumps([round(v, 2) for v in d["bbox"]])} for r, d in enumerate(found)]
        if not rows:  # keep a marker so the query counts as done
            rows = [{"dataset": ds, "mode": mode, "object": obj, "photo": photo, "rank": -1, "score": 0.0, "box": ""}]
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS).writerows(rows)
        if i % 25 == 0 or i == len(jobs):
            print(f"[gdino] {i}/{len(jobs)}", flush=True)
    del detectors
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    crops = [{"dataset": "valuables", "kind": "gold", "photo": k, "object": v["object"], "mode": "", "rank": "",
              "box": json.dumps(v["box"])} for k, v in vphotos.items() if v.get("box")]
    crops += [{"dataset": "c28", "kind": "gold", "photo": n, "object": ios.photo_info(n)["object"], "mode": "",
               "rank": "", "box": json.dumps(b)} for n, b in c28_gold.items()]
    with open(path, encoding="utf-8") as f:
        crops += [{"dataset": r["dataset"], "kind": "candidate", "photo": r["photo"], "object": r["object"],
                   "mode": r["mode"], "rank": r["rank"], "box": r["box"]} for r in csv.DictReader(f) if r["box"]]
    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "dataset", "kind", "photo", "object", "mode", "rank", "box"])
        writer.writeheader()
        writer.writerows({"id": i, **c} for i, c in enumerate(crops))
    print(f"[crops] {len(crops)}", flush=True)

    info = {"device": args.device, "torch": torch.__version__, "models": {}}
    for model_dir in args.dinov3:
        name = Path(model_dir).name
        proc = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir).to(args.device).eval()
        embs = []
        for s in range(0, len(crops), args.batch):
            batch = [crop(images[(c["dataset"], c["photo"])], json.loads(c["box"])) for c in crops[s:s + args.batch]]
            inputs = proc(images=batch, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                embs.append(F.normalize(model(**inputs).pooler_output.float(), dim=-1).cpu().numpy())
        np.save(out / f"emb_{name}.npy", np.concatenate(embs))
        info["models"][name] = {"dir": str(model_dir), "dim": int(embs[0].shape[1])}
        print(f"[dinov3] {name} done", flush=True)
        del model
    import transformers
    info["transformers"] = transformers.__version__
    (out / "run_info.json").write_text(json.dumps(info, indent=2))


# ---------- report (no model) ----------

def auc(pos: list, neg: list) -> float:
    """Probability that a positive scores above a negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def decide(cands: list, sims: list) -> dict:
    """cands: [(score, box)] in Grounding DINO order; sims: their similarities. The three pipelines' inputs."""
    top = cands[0] if cands and cands[0][0] >= APP_THRESHOLD else None
    pool = [(s, c) for s, c in zip(sims, cands) if c[0] >= FLOOR][:RERANK_K]
    best = max(pool, key=lambda t: t[0], default=None)
    return {"top_score": top[0] if top else 0.0, "top_box": top[1] if top else None,
            "top_sim": sims[0] if top else None,
            "rr_sim": best[0] if best else None, "rr_box": best[1][1] if best else None,
            "rr_score": best[1][0] if best else 0.0}


def accepted(rec: dict, pipeline: str, tau: float | None = None) -> tuple[bool, list | None]:
    if pipeline == "gdino":
        return rec["top_box"] is not None, rec["top_box"]
    if pipeline == "filter":
        return rec["top_box"] is not None and rec["top_sim"] >= tau, rec["top_box"]
    return rec["rr_box"] is not None and rec["rr_sim"] >= tau, rec["rr_box"]


def rates(recs: list, pipeline: str, tau: float | None = None) -> dict:
    n = defaultdict(int)
    hit = defaultdict(int)
    for r in recs:
        ok, box = accepted(r, pipeline, tau)
        n[r["kind"]] += 1
        if r["kind"] == "target":
            hit["target"] += ok and box is not None and ios.centre_inside(box, r["gold"])
        else:
            hit[r["kind"]] += ok
    return {k: (hit[k], n[k]) for k in ("target", "other", "none")}


def pick_tau(recs: list, pipeline: str) -> float:
    """Largest found - other said yes - false alarms; ties go to the lower tau."""
    def score(t):
        r = rates(recs, pipeline, t)
        return sum((1 if k == "target" else -1) * (h / n if n else 0) for k, (h, n) in r.items())
    return max(TAUS, key=lambda t: (round(score(t), 9), -t))


def build_records(run_dir: Path, emb_name: str) -> list[dict]:
    import numpy as np

    emb = np.load(run_dir / f"emb_{emb_name}.npy")
    crops = list(csv.DictReader(open(run_dir / "crops.csv", encoding="utf-8")))
    gold_id = {(c["dataset"], c["photo"]): int(c["id"]) for c in crops if c["kind"] == "gold"}
    cand = defaultdict(list)  # (dataset, mode, object, photo) -> [(score, box, crop id)]
    for c in crops:
        if c["kind"] == "candidate":
            cand[(c["dataset"], c["mode"], c["object"], c["photo"])].append((int(c["rank"]), int(c["id"]), c))
    scores = {}
    for r in csv.DictReader(open(run_dir / "candidates.csv", encoding="utf-8")):
        scores[(r["dataset"], r["mode"], r["object"], r["photo"], r["rank"])] = float(r["score"])

    def candidates(ds, mode, obj, photo):
        items = sorted(cand.get((ds, mode, obj, photo), []))
        return [(scores[(ds, mode, obj, photo, str(rank))], json.loads(c["box"]), i) for rank, i, c in items]

    records = []
    words, vphotos, vqueries = load_valuables()
    for o in words:
        regs = sorted(k for k, v in vphotos.items() if v["role"] == "register" and v["object"] == o)
        for mode in ("app", "single"):
            gmode = ("multi" if o in APP_CATEGORY else "single") if mode == "app" else "single"
            for q in vqueries[o]:
                g = vphotos[q]
                kind = "target" if g["role"] == "search" else ("other" if g["role"] == "other_keys" and o == "keys"
                                                                else "none")
                cs = candidates("valuables", gmode, o, q)
                for regset, used in [*[("one-shot", [r]) for r in regs], ("four-shot", regs)]:
                    ref = emb[[gold_id[("valuables", r)] for r in used]]
                    sims = [float((emb[i] @ ref.T).max()) for _, _, i in cs]
                    records.append({"dataset": "valuables", "mode": mode, "regset": regset,
                                    "example": used[0] if regset == "one-shot" else "4 close-ups", "object": o,
                                    "query": q, "kind": kind, "gold": g.get("box"),
                                    **decide([(s, b) for s, b, _ in cs], sims)})
    c28_gold, names = load_c28()
    info = {n: ios.photo_info(n) for n in names}
    for ex in sorted(c28_gold):
        inst, cat = info[ex]["instance"], info[ex]["object"]
        ref = emb[gold_id[("c28", ex)]]
        for mode in ("app", "single"):
            gmode = "multi" if mode == "app" else "single"
            for q in names:
                if q == ex:
                    continue
                rel = ios.relation(inst, info[q])
                kind = {"same object": "target", "same type, other object": "other"}.get(rel, "none")
                cs = candidates("c28", gmode, cat, q)
                sims = [float(emb[i] @ ref) for _, _, i in cs]
                records.append({"dataset": "c28", "mode": mode, "regset": "one-shot", "example": ex, "object": cat,
                                "query": q, "kind": kind, "gold": c28_gold.get(q),
                                **decide([(s, b) for s, b, _ in cs], sims)})
    return records


def gold_aucs(run_dir: Path, emb_name: str, records: list[dict]) -> dict:
    """Verifier alone: the registration vs gold crops of the same object / the other object of the same type, and vs
    the most similar rerank candidate in photos without the object."""
    import numpy as np

    emb = np.load(run_dir / f"emb_{emb_name}.npy")
    crops = list(csv.DictReader(open(run_dir / "crops.csv", encoding="utf-8")))
    gid = {(c["dataset"], c["photo"]): int(c["id"]) for c in crops if c["kind"] == "gold"}
    c28_gold, names = load_c28()
    info = {n: ios.photo_info(n) for n in names}
    same, other = [], []
    for ex in c28_gold:
        for q in c28_gold:
            if q == ex or info[q]["object"] != info[ex]["object"]:
                continue
            s = float(emb[gid[("c28", ex)]] @ emb[gid[("c28", q)]])
            (same if info[q]["instance"] == info[ex]["instance"] else other).append(s)
    none28 = [r["rr_sim"] for r in records if r["dataset"] == "c28" and r["mode"] == "app" and r["kind"] == "none"
              and r["rr_sim"] is not None]
    words, vphotos, _ = load_valuables()
    vtarget = []  # (object, similarity of its gold crop to the closest of the 4 registrations)
    for o in words:
        regs = [gid[("valuables", k)] for k, v in vphotos.items() if v["role"] == "register" and v["object"] == o]
        for k, v in vphotos.items():
            if v["role"] == "search" and v["object"] == o:
                vtarget.append((o, float((emb[gid[("valuables", k)]] @ emb[regs].T).max())))
    vpos = [s for _, s in vtarget]
    vkeys = [s for o, s in vtarget if o == "keys"]
    vrec = [r for r in records if r["dataset"] == "valuables" and r["mode"] == "app" and r["regset"] == "four-shot"]
    vnone = [r["rr_sim"] for r in vrec if r["kind"] == "none" and r["rr_sim"] is not None]
    vother = [r["rr_sim"] for r in vrec if r["kind"] == "other" and r["rr_sim"] is not None]
    med = lambda xs: float(np.median(xs)) if xs else float("nan")  # noqa: E731
    row = lambda pos, neg: (auc(pos, neg), med(pos), med(neg), len(pos), len(neg))  # noqa: E731
    return {"c28: same object vs the other object of the type (gold crops)": row(same, other),
            "c28: same object (gold) vs best box in photos without the type": row(same, none28),
            "valuables: item (gold) vs best box in no-valuables photos": row(vpos, vnone),
            "valuables: my keys (gold) vs best box on the other key set": row(vkeys, vother)}


def reproduction(records: list[dict]) -> list[str]:
    """Grounding DINO alone here vs the earlier runs (valuables: Mac MPS; 28 photos: Mac MPS), same decisions?"""
    old_v = {(r["model"], r["object"], r["query"]): r for r in csv.DictReader(open(
        VAL_GOLD.parent / "per_query.csv", encoding="utf-8")) if r["model"].startswith("Grounding DINO")}
    old_c = {(r["example"], r["query"]): r for r in csv.DictReader(open(
        C28_GOLD.parent / "per_query.csv", encoding="utf-8")) if r["model"] == "gdino_multi"}
    same = defaultdict(lambda: [0, 0])
    for r in records:
        if r["dataset"] == "valuables" and r["regset"] == "four-shot":
            if r["mode"] == "app" and r["object"] not in APP_CATEGORY:
                continue  # the earbud's "app" boxes are its single-word boxes, counted once below
            name = "Grounding DINO single" if r["mode"] == "single" else "Grounding DINO multi"
            o = old_v.get((name, r["object"], r["query"]))
            key = f"valuables {name}"
        elif r["dataset"] == "c28" and r["mode"] == "app":
            o, key = old_c.get((r["example"], r["query"])), "c28 Grounding DINO multi"
        else:
            continue
        if o is None:
            continue
        old_yes = float(o["score"]) >= APP_THRESHOLD
        same[key][0] += old_yes == (r["top_box"] is not None)
        same[key][1] += 1
    return [f"| {k} | {a}/{n} |" for k, (a, n) in same.items()]


def report(args) -> None:
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    models = sorted(p.stem[4:] for p in args.run.glob("emb_*.npy"))
    pct = lambda h, n: f"{h}/{n} ({h / n:.0%})" if n else "-"  # noqa: E731
    lines, all_rows, chosen = [], [], {}
    for name in models:
        recs = build_records(args.run, name)
        all_rows += [{**r, "model": name} for r in recs]
        c28 = [r for r in recs if r["dataset"] == "c28" and r["mode"] == "app"]
        for p in ("filter", "rerank"):
            chosen[(name, p)] = pick_tau(c28, p)
        lines += [f"## {name}", ""]
        for ds, regsets in (("c28", ["one-shot"]), ("valuables", ["one-shot", "four-shot"])):
            for mode in ("app", "single"):
                for regset in regsets:
                    rs = [r for r in recs if r["dataset"] == ds and r["mode"] == mode and r["regset"] == regset]
                    lines += [f"### {ds}, {mode}, {regset}", "",
                              "| Pipeline | tau | Item: box on it | Other object of the type: said yes | "
                              "False alarms |", "|---|---|---|---|---|"]
                    g = rates(rs, "gdino")
                    lines.append(f"| Grounding DINO alone | - | {pct(*g['target'])} | {pct(*g['other'])} | "
                                 f"{pct(*g['none'])} |")
                    for p in ("filter", "rerank"):
                        t = chosen[(name, p)]
                        r = rates(rs, p, t)
                        lines.append(f"| {p} (tau from c28) | {t} | {pct(*r['target'])} | {pct(*r['other'])} | "
                                     f"{pct(*r['none'])} |")
                    lines += ["", "Sweep (box on it / other said yes / false alarms):", "",
                              "| Pipeline | " + " | ".join(str(t) for t in TAUS[::2]) + " |",
                              "|---|" + "---|" * len(TAUS[::2])]
                    for p in ("filter", "rerank"):
                        cells = []
                        for t in TAUS[::2]:
                            r = rates(rs, p, t)
                            cells.append(" / ".join(f"{h / n:.0%}" if n else "-" for h, n in r.values()))
                        lines.append(f"| {p} | " + " | ".join(cells) + " |")
                    lines.append("")
        lines += ["### Verifier alone (AUC, median similarity positive / negative, n)", "",
                  "| Comparison | AUC | Median pos | Median neg | n pos | n neg |", "|---|---|---|---|---|---|"]
        for k, (a, mp, mn, npos, nneg) in gold_aucs(args.run, name, recs).items():
            lines.append(f"| {k} | {a:.2f} | {mp:.2f} | {mn:.2f} | {npos} | {nneg} |")
        lines.append("")
        if name == models[0]:
            lines += ["### Grounding DINO alone: same yes/no at 0.35 as the earlier Mac runs", "",
                      "| Run | Same answer |", "|---|---|", *reproduction(recs), ""]
    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    fields = ["model", "dataset", "mode", "regset", "example", "object", "query", "kind", "top_score", "top_sim",
              "top_box", "rr_score", "rr_sim", "rr_box", "gold"]
    with open(out / "per_query.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else
                                 round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in fields})
    (out / "chosen_tau.json").write_text(json.dumps({f"{m} {p}": t for (m, p), t in chosen.items()}, indent=2))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--dinov3", nargs="+", required=True, help="local DINOv3 snapshot folders")
    r.add_argument("--device", default="cuda")
    r.add_argument("--batch", type=int, default=32)
    p = sub.add_parser("report")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    main()
