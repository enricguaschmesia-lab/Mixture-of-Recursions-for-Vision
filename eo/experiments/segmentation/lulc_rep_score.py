#!/usr/bin/env python
"""LULC as RGB map tokens against TerraMind's LULC tokens: DECODING and SCORING (`a2seg` env; tf3's read needs no GPU).
The generation half and the protocol: eo/experiments/segmentation/lulc_rep_gen.py; docs/plans/a2_segmentation.md,
"Follow-up: LULC through the map tokenizer against TerraMind's LULC tokens" (written before any number of it).

  decode1   part 1's codes (the four 256-row shards joined, rows checked) -> class maps (224 x 224):
              TerraMind's LULC tokens  its ViT decoder (eo/terramesh_tok/tokenizers), argmax over classes 1-9;
              map codes                the frozen aligned tokenizer (hashes checked), harness.class_scores on
                                       LULC's 224 grid, the frozen rule (nearest).
            Each of the K samples is decoded and voted per pixel (seg.vote); the true tokens of each representation
            are decoded the same way (the ceilings). Per scene: the 10 x 10 confusion of the vote and of sample 0,
            against the val LULC raster's 224 centre crop (no data = class 0 ignored).
  score1    pooled mIoU over the 9 classes (eo/maps/scoring.Confusion's formula), pixel accuracy, per-scene mIoU over the
            classes present (decode_eo.lulc_metrics), per corpus, per class; the paired bootstrap of the reading rule
            (10,000 resamples of scenes, seed 0, the pooled mIoU recomputed from the resampled confusions); controls:
            ceilings, another scene (the ceiling of the row half the list away), none.
  decode3   part 3's generated targets and their true tokens through TerraMind's decoders at tm_compare's protocol
            (10 diffusion steps, seed 0 per batch of 8), standardized float16.
  score3    MAE / RMSE / SSIM against the original rasters in physical units (tm_compare_eval's functions), paired
            bootstrap of MAE; and tf3's teacher-forced CE: each source's gain over none, map against TerraMind's tokens,
            paired per row (10,000 resamples).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python eo/experiments/segmentation/lulc_rep_score.py decode1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "eo", REPO / "eo" / "scripts", REPO / "eo" / "experiments" / "generation"):
    sys.path.insert(0, str(p))
os.environ.setdefault("HF_HOME", "/data/enric/hf")
os.environ.setdefault("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")

import numpy as np  # noqa: E402
import torch  # noqa: E402

GEN = Path(os.environ.get("LULC_REP_OUT", "/data/enric/generations/a2_lulc_rep"))
REPORT = Path(os.environ.get("LULC_REP_REPORT", "/data/enric/reports/a2_seg/lulc_rep"))
N_SHARDS, NC, IGNORE = 4, 10, 0
COND1 = [f"{m}_{t}_{s}" for m, t in (("base", "lulc"), ("r1", "lulc"), ("r1", "map")) for s in ("none", "s2", "all")]
COMPARE1 = {"asked": ("r1_map", "base_lulc"), "representation": ("r1_map", "r1_lulc"), "mid_training": ("r1_lulc", "base_lulc")}
TARGETS3 = ("S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI")
COND3 = ("base_lulc", "r1_lulc", "r1_map", "base_none", "r1_none")
NB, BOOT_SEED = 10_000, 0


def gpu(allow_titanv):
    if os.environ.get("LULC_REP_CPU") == "1":            # the scorer's own dry run on a throwaway directory only
        return "cpu"
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...)")
    name = torch.cuda.get_device_name(0)
    if "TITAN V" in name and not allow_titanv:
        raise SystemExit("the TITAN V only under the power watchdog: pass --allow-titanv from its runner")
    return name


# ------------------------------------------------------------------ part 1
def load1(cond):
    """A condition's four shards joined: rows, corpus, codes (N, K, 196), truth (N, 196)."""
    parts = []
    for s in range(N_SHARDS):
        f = GEN / "part1" / f"shard{s}" / f"{cond}.npz"
        if not f.with_suffix(".json").exists():
            raise SystemExit(f"{f}: not done")
        parts.append(np.load(f))
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("rows", "corpus", "codes", "truth")}
    if len(np.unique(out["rows"])) != len(out["rows"]):
        raise SystemExit(f"{cond}: a row twice")
    return out


def true_lulc(rows):
    import decode_eo as DE
    from terramesh_tok import preprocess as PP
    ras = DE.read_rasters("LULC", np.asarray(rows), "clean")
    gt = np.stack([PP.center_crop(a[0] if a.ndim == 4 else a)[0] for a in ras]).astype(np.int64)
    if gt.shape[1:] != (224, 224) or gt.min() < 0 or gt.max() >= NC:
        raise SystemExit(f"LULC truth {gt.shape}, classes {gt.min()}..{gt.max()}")
    return gt


class Decoders:
    def __init__(self, device):
        from eo.maps import frozen
        from terramesh_tok import tokenizers as TK
        self.dev, self.TK = device, TK
        self.tm = TK.build("LULC", device=device)
        self.map = frozen.load("aligned", device)
        self.rule = frozen.record("aligned")["rule"]

    @torch.no_grad()
    def lulc(self, codes):
        """(B, 196) TerraMind LULC codes -> (B, 224, 224) classes 1-9, and the unrestricted argmax."""
        g = torch.as_tensor(codes.astype(np.int64), device=self.dev).view(-1, 14, 14)
        out = self.TK.decode(self.tm, g, seed=0).float()
        if out.shape[1:] != (NC, 224, 224):
            raise SystemExit(f"TerraMind's LULC decoder gave {tuple(out.shape)}")
        return out[:, 1:].argmax(1) + 1, out.argmax(1)

    @torch.no_grad()
    def mapc(self, codes):
        """(B, 196) aligned map codes (raster order) -> (B, 224, 224) classes by the frozen rule."""
        from eo.maps import decode as D, harness as H
        g = torch.as_tensor(codes.astype(np.int64), device=self.dev).view(-1, 14, 14)
        sc = H.class_scores(self.map, g, "lulc", "universal", "crop224")
        pr = D.apply_rule(sc, IGNORE, self.rule, self.map.output)
        if pr.shape[1:] != (224, 224) or bool((pr == IGNORE).any()):
            raise SystemExit("map decode: wrong shape, or a no-data prediction")
        return pr


def confusions(gt, pred):
    """(N, H, W) truth and prediction -> (N, NC, NC) per-scene confusion, no-data truth pixels left out."""
    # numpy, not torch: torch's CPU bincount took > 180 s for 1,024 scenes on the shared 4 cores, which the TITAN V's
    # watchdog read as a stall (2026-11-23); numpy gives the identical matrices in < 1 s.
    gt = np.asarray(gt).reshape(len(gt), -1).astype(np.int64)
    pred = (pred.cpu().numpy() if torch.is_tensor(pred) else np.asarray(pred)).reshape(len(gt), -1).astype(np.int64)
    out = np.zeros((len(gt), NC, NC), np.int64)
    for i in range(len(gt)):
        v = gt[i] != IGNORE
        out[i] = np.bincount(gt[i][v] * NC + pred[i][v], minlength=NC * NC).reshape(NC, NC)
    return out


def done1(cond):
    return all((GEN / "part1" / f"shard{s}" / f"{cond}.json").exists() for s in range(N_SHARDS))


def truth1(a):
    """CPU: the val LULC rasters of part 1's rows (a tar pass), before any GPU step (the TITAN V's watchdog would read
    a long CPU read as a stall)."""
    cond = next(c for c in COND1 if done1(c))
    d = load1(cond)
    (GEN / "decoded1").mkdir(parents=True, exist_ok=True)
    gt = true_lulc(d["rows"])
    np.savez(GEN / "decoded1" / "truth.npz", rows=d["rows"], corpus=d["corpus"], gt=gt.astype(np.uint8))
    print(f"truth: {len(gt)} rows from {cond}'s; class pixels {np.bincount(gt.ravel(), minlength=NC).tolist()}")
    return 0


def decode1(a):
    """Every part-1 condition that is done (others are skipped: run again later), and the two ceilings."""
    from eo.eval import seg as SG
    name = gpu(a.allow_titanv)
    dev = "cpu" if name == "cpu" else "cuda"
    outdir = GEN / "decoded1"
    tr = np.load(outdir / "truth.npz")
    rows, gt = tr["rows"], tr["gt"].astype(np.int64)
    dec = Decoders(dev)
    jobs = [(c, None) for c in COND1] + [("ceiling_lulc", "lulc"), ("ceiling_map", "map")]
    truths = {}
    for cond, ceil in jobs:
        f = outdir / f"{cond}.npz"
        if f.exists():
            print(f"[skip] {cond}: decoded", flush=True)
            continue
        src = cond if ceil is None else ("r1_map_s2" if ceil == "map" else "r1_lulc_s2")
        if not done1(src):
            print(f"[wait] {cond}: {src} not generated yet", flush=True)
            continue
        t0 = time.time()
        if ceil is None:
            d = load1(cond)
            if not np.array_equal(d["rows"], rows):
                raise SystemExit(f"{cond}: rows differ from {COND1[0]}'s")
            rep = "map" if "_map_" in cond else "lulc"
            if rep in truths and not np.array_equal(truths[rep], d["truth"]):
                raise SystemExit(f"{cond}: its true codes differ from another condition's")
            truths[rep] = d["truth"]
            codes = d["codes"]                                   # (N, K, 196)
        else:
            d = load1(src)
            if not np.array_equal(d["rows"], rows):
                raise SystemExit(f"{src}: rows differ from the truth's")
            codes = d["truth"][:, None]                          # (N, 1, 196)
            rep = ceil
        N, K, _ = codes.shape
        vote = np.zeros((N, 224, 224), np.uint8)
        first_s = np.zeros((N, 224, 224), np.uint8)
        free = np.zeros((N, 224, 224), np.uint8) if rep == "lulc" else None
        per = 32
        for s in range(0, N, per):
            ks, kfree = [], []
            for j in range(K):
                if rep == "lulc":
                    p, pf = dec.lulc(codes[s:s + per, j])
                    kfree.append(pf)
                else:
                    p = dec.mapc(codes[s:s + per, j])
                ks.append(p)
            vote[s:s + per] = SG.vote(torch.stack(ks), NC).to(torch.uint8).cpu().numpy()
            first_s[s:s + per] = ks[0].to(torch.uint8).cpu().numpy()
            if rep == "lulc":
                free[s:s + per] = SG.vote(torch.stack(kfree), NC).to(torch.uint8).cpu().numpy()
        arrays = {"rows": rows, "vote": vote, "first": first_s, "cm_vote": confusions(gt, vote),
                  "cm_first": confusions(gt, first_s)}
        if free is not None:
            arrays["cm_unrestricted"] = confusions(gt, free)
            arrays["nodata_pred_share"] = np.array([(free == IGNORE).mean()])
        np.savez(f.with_suffix(".tmp.npz"), **arrays)
        os.replace(f.with_suffix(".tmp.npz"), f)
        print(f"{cond}: decoded {N} x {K} in {time.time() - t0:.0f} s ({name})", flush=True)
    return 0


def miou(cm):
    """eo/maps/scoring.Confusion's mIoU of a (NC, NC) confusion (rows truth): classes seen in truth or prediction,
    no data left out."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(den > 0, tp / np.where(den > 0, den, 1), 0.0)
    w = ((cm.sum(0) + cm.sum(1)) > 0).astype(np.float64)
    w[IGNORE] = 0
    return float((w * iou).sum() / w.sum()), iou


def miou_batch(cms):
    """(B, NC, NC) -> (B,) mIoU, vectorised for the bootstrap."""
    cm = cms.astype(np.float64)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    den = cm.sum(1) + cm.sum(2) - tp
    iou = np.where(den > 0, tp / np.where(den > 0, den, 1), 0.0)
    w = ((cm.sum(1) + cm.sum(2)) > 0).astype(np.float64)
    w[:, IGNORE] = 0
    return (w * iou).sum(1) / w.sum(1)


def scene_miou(cms):
    """Per scene: mIoU over the classes present in its truth (decode_eo.lulc_metrics), and pixel accuracy."""
    out_m, out_a = [], []
    for cm in cms:
        cm = cm.astype(np.float64)
        tp = np.diag(cm)
        present = cm.sum(1) > 0
        present[IGNORE] = False
        den = cm.sum(0) + cm.sum(1) - tp
        out_m.append(float((tp[present] / den[present]).mean()) if present.any() else np.nan)
        out_a.append(float(tp.sum() / cm.sum()) if cm.sum() else np.nan)
    return np.array(out_m), np.array(out_a)


def boot_pooled(cma, cmb, idx):
    """Pooled mIoU(a) - mIoU(b) over resampled scenes (idx (NB, N)), and the point difference."""
    d = miou(cma.sum(0))[0] - miou(cmb.sum(0))[0]
    bs = np.empty(len(idx))
    for i0 in range(0, len(idx), 500):
        sl = idx[i0:i0 + 500]
        bs[i0:i0 + 500] = miou_batch(cma[sl].sum(1)) - miou_batch(cmb[sl].sum(1))
    return {"diff": d, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "boot_mean": float(bs.mean()), "share_positive": float((bs > 0).mean())}


def verdict(ci):
    return "map better" if ci[0] > 0 else ("TerraMind's tokens better" if ci[1] < 0 else "level")


def score1(a):
    from eo.maps import palette as P
    d1 = GEN / "decoded1"
    tr = np.load(d1 / "truth.npz")
    rows, corpus = tr["rows"], tr["corpus"].astype(str)
    N = len(rows)
    idx = np.random.default_rng(BOOT_SEED).integers(0, N, (NB, N))
    cms, extra = {}, {}
    for cond in COND1 + ["ceiling_lulc", "ceiling_map"]:
        z = np.load(d1 / f"{cond}.npz")
        if not np.array_equal(z["rows"], rows):
            raise SystemExit(f"{cond}: rows differ")
        cms[cond] = {"vote": z["cm_vote"], "first": z["cm_first"]}
        if "cm_unrestricted" in z:
            cms[cond]["unrestricted"] = z["cm_unrestricted"]
            extra[cond] = {"nodata_pred_share_unrestricted": float(z["nodata_pred_share"][0])}
    half = np.roll(np.arange(N), N // 2)                      # another scene: the ceiling of the row half the list away
    gt = tr["gt"].astype(np.int64)
    for rep in ("lulc", "map"):
        z = np.load(d1 / f"ceiling_{rep}.npz")
        cms[f"another_scene_{rep}"] = {"vote": confusions(gt, z["vote"][half].astype(np.int64))}
    res = {"rows": int(N), "corpora": {c: int((corpus == c).sum()) for c in np.unique(corpus)},
           "classes": list(P.LULC_NAMES), "conditions": {}, "compare": {}, "validity": {}}
    for cond, kinds in cms.items():
        r = {}
        for kind, cm in kinds.items():
            m, iou = miou(cm.sum(0))
            sm, sa = scene_miou(cm)
            r[kind] = {"mIoU": m, "pixel_acc": float(np.trace(cm.sum(0)) / cm.sum()),
                       "scene_mIoU": float(np.nanmean(sm)), "per_class_IoU": dict(zip(P.LULC_NAMES, iou.tolist())),
                       "per_corpus_mIoU": {c: miou(cm[corpus == c].sum(0))[0] for c in np.unique(corpus)}}
        r.update(extra.get(cond, {}))
        res["conditions"][cond] = r
    ceil = {"lulc": res["conditions"]["ceiling_lulc"]["vote"]["mIoU"], "map": res["conditions"]["ceiling_map"]["vote"]["mIoU"]}
    for cond in COND1:
        rep = "map" if "_map_" in cond else "lulc"
        res["conditions"][cond]["share_of_ceiling"] = res["conditions"][cond]["vote"]["mIoU"] / ceil[rep]
    for src in ("s2", "all", "none"):
        for cname, (x, y) in COMPARE1.items():
            ca, cb = f"{x}_{src}", f"{y}_{src}"
            out = {}
            for kind in ("vote", "first"):
                out[kind] = boot_pooled(cms[ca][kind], cms[cb][kind], idx)
                out[kind]["per_corpus"] = {}
                for c in np.unique(corpus):
                    sel = np.nonzero(corpus == c)[0]
                    ic = np.random.default_rng(BOOT_SEED).integers(0, len(sel), (NB, len(sel)))
                    out[kind]["per_corpus"][c] = boot_pooled(cms[ca][kind][sel], cms[cb][kind][sel], ic)
            out["verdict"] = verdict(out["vote"]["ci95"]) if src != "none" else "(prior: not read)"
            res["compare"][f"{cname}|{src}"] = {"a": ca, "b": cb, **out}
    for cond in COND1:
        if cond.endswith("_none"):
            continue
        rep = "map" if "_map_" in cond else "lulc"
        m = res["conditions"][cond]["vote"]["mIoU"]
        res["validity"][cond] = {
            "above_another_scene": m > res["conditions"][f"another_scene_{rep}"]["vote"]["mIoU"],
            "above_none": m > res["conditions"][cond.rsplit("_", 1)[0] + "_none"]["vote"]["mIoU"],
            "below_ceiling": m < ceil[rep]}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "part1.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["| condition | mIoU (K = 8 vote) | K = 1 | pixel acc | scene mIoU | Major TOM | ssl4eos12 | of ceiling |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, r in res["conditions"].items():
        v = r["vote"]
        lines.append(f"| {cond} | {100 * v['mIoU']:.2f} | {100 * r.get('first', v)['mIoU']:.2f} | "
                     f"{100 * v['pixel_acc']:.2f} | {100 * v['scene_mIoU']:.2f} | "
                     f"{100 * v['per_corpus_mIoU'].get('majortom', np.nan):.2f} | "
                     f"{100 * v['per_corpus_mIoU'].get('ssl4eos12', np.nan):.2f} | "
                     f"{r.get('share_of_ceiling', float('nan')):.3f} |")
    lines += ["", "| comparison | Δ mIoU (vote) | 95% interval | K = 1 Δ | Major TOM Δ | ssl4eos12 Δ | verdict |",
              "|---|---|---|---|---|---|---|"]
    for k, c in res["compare"].items():
        v = c["vote"]
        lines.append(f"| {k} ({c['a']} − {c['b']}) | {100 * v['diff']:+.2f} | [{100 * v['ci95'][0]:+.2f}, "
                     f"{100 * v['ci95'][1]:+.2f}] | {100 * c['first']['diff']:+.2f} | "
                     f"{100 * v['per_corpus'].get('majortom', {}).get('diff', np.nan):+.2f} | "
                     f"{100 * v['per_corpus'].get('ssl4eos12', {}).get('diff', np.nan):+.2f}"
                     f" | {c['verdict']} |")
    (REPORT / "part1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("validity:", json.dumps(res["validity"]))
    return 0


# ------------------------------------------------------------------ part 3
def load3(cond, T):
    f = GEN / "part3" / f"{cond}_{T}.npz"
    if not f.with_suffix(".json").exists():
        raise SystemExit(f"{f}: not done")
    return np.load(f)


def decode3(a):
    from terramesh_tok import tokenizers as TK
    import tm_compare_eval as TME
    name = gpu(a.allow_titanv)
    out = GEN / "decoded3"
    for T in (a.targets or TARGETS3):
        tok = TK.build(T, device="cuda")
        ref = load3(COND3[0], T)
        jobs = {"ceiling": ref["truth"]}
        for cond in COND3:
            z = load3(cond, T)
            if not (np.array_equal(z["rows"], ref["rows"]) and np.array_equal(z["truth"], ref["truth"])):
                raise SystemExit(f"{cond} {T}: rows or truth differ")
            for smp in ("t1", "p08"):
                jobs[f"{cond}_{smp}"] = z[f"{smp}|tokens"]
        for cond, codes in jobs.items():
            f = out / cond / f"{T}.npy"
            if f.exists():
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            t0, outs = time.time(), []
            for i in range(0, len(codes), TME.BATCH):
                g = torch.from_numpy(codes[i:i + TME.BATCH].astype(np.int64)).view(-1, 14, 14).cuda()
                outs.append(TK.decode(tok, g, timesteps=TME.STEPS, seed=TME.DECODE_SEED).float().cpu().numpy())
            np.save(f.with_suffix(".tmp.npy"), np.concatenate(outs).astype(np.float16))
            os.replace(f.with_suffix(".tmp.npy"), f)
            print(f"{T:6s} {cond:16s} n {len(codes)}  {time.time() - t0:.0f} s ({name})", flush=True)
        del tok
        torch.cuda.empty_cache()
    return 0


def boot_mean(d, n=NB, seed=BOOT_SEED):
    d = np.asarray(d, np.float64)
    bs = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def verdict_lower(ci):
    """For a difference map - TerraMind where LOWER is better (CE, MAE)."""
    return "map better" if ci[1] < 0 else ("TerraMind's tokens better" if ci[0] > 0 else "level")


def score3(a):
    res = {"tf3": {}, "decoded": {}}
    # teacher-forced CE, every held-out row
    z = {m: np.load(GEN / "tf3" / f"{m}.npz") for m in ("base", "r1")}
    if not np.array_equal(z["base"]["rows"], z["r1"]["rows"]):
        raise SystemExit("tf3: rows differ between models")
    corpus = z["base"]["corpus"].astype(str)
    for T in ("S2L2A", "S1", "DEM", "NDVI"):
        ce = {"base_none": z["base"][f"none|{T}"], "base_lulc": z["base"][f"lulc|{T}"],
              "r1_none": z["r1"][f"none|{T}"], "r1_lulc": z["r1"][f"lulc|{T}"], "r1_map": z["r1"][f"map|{T}"]}
        gain = {k: ce[k.split("_")[0] + "_none"] - ce[k] for k in ("base_lulc", "r1_lulc", "r1_map")}
        r = {"ce": {k: float(v.mean()) for k, v in ce.items()}, "gain_over_none": {k: float(v.mean()) for k, v in gain.items()},
             "ce_per_corpus": {c: {k: float(v[corpus == c].mean()) for k, v in ce.items()} for c in np.unique(corpus)},
             "compare": {}}
        for cname, (x, y) in COMPARE1.items():
            # CE difference = -(gain difference) within a model; across models the gains are compared (each over its own none)
            dg = gain[x] - gain[y]
            b = boot_mean(-dg)                           # as a CE change: negative = x's representation helps more
            b["verdict"] = verdict_lower(b["ci95"]) if cname != "mid_training" else (
                "r1 better" if b["ci95"][1] < 0 else ("base better" if b["ci95"][0] > 0 else "level"))
            b["per_corpus"] = {c: boot_mean(-dg[corpus == c]) for c in np.unique(corpus)}
            r["compare"][f"{cname} ({x} - {y})"] = b
        res["tf3"][T] = r
    # decoded, 256 rows per target
    import decode_eo as DE
    import tm_compare_eval as TME
    from terramesh_tok import preprocess as PP
    d3 = GEN / "decoded3"
    for T in TARGETS3:
        if not (d3 / "ceiling" / f"{T}.npy").exists():
            continue
        R = load3(COND3[0], T)["rows"]
        refs = np.stack([PP.center_crop(x[0] if x.ndim == 4 else x).astype(np.float32)
                         for x in DE.read_rasters(T, R, "clean")]) * TME.SCALE.get(T, 1.0)
        rng = float(np.nanmax(refs) - np.nanmin(refs))
        per = {}
        for cdir in sorted(p.parent.name for p in d3.glob(f"*/{T}.npy")):
            pred = TME.physical(np.load(d3 / cdir / f"{T}.npy"), T)
            per[cdir] = TME.scene_metrics(pred, refs, rng)
            if cdir == "ceiling":
                per["another_scene"] = TME.scene_metrics(np.roll(pred, len(pred) // 2, axis=0), refs, rng)
        out = {"n": int(len(R)), "data_range": rng, "conditions": {}, "compare": {}}
        for c, m in per.items():
            out["conditions"][c] = {"mae": float(m[:, 0].sum() / m[:, 2].sum()),
                                    "rmse": float(np.sqrt(m[:, 1].sum() / m[:, 2].sum())), "ssim": float(m[:, 3].mean())}
        for smp in ("t1", "p08"):
            for cname, (x, y) in COMPARE1.items():
                ka, kb = f"{x}_{smp}", f"{y}_{smp}"
                if ka in per and kb in per:
                    b = boot_mean(per[ka][:, 0] / per[ka][:, 2] - per[kb][:, 0] / per[kb][:, 2])
                    b["verdict"] = verdict_lower(b["ci95"]) if cname != "mid_training" else (
                        "r1 better" if b["ci95"][1] < 0 else ("base better" if b["ci95"][0] > 0 else "level"))
                    out["compare"][f"{cname}|{smp} ({ka} - {kb}) scene MAE"] = b
        res["decoded"][T] = out
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "part3.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["| target | CE none (base / r1) | gain LULC base | gain LULC r1 | gain map r1 | map − LULC (r1), CE | 95% | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for T, r in res["tf3"].items():
        c = r["compare"]["representation (r1_map - r1_lulc)"]
        g = r["gain_over_none"]
        lines.append(f"| {T} | {r['ce']['base_none']:.4f} / {r['ce']['r1_none']:.4f} | {g['base_lulc']:.4f} | "
                     f"{g['r1_lulc']:.4f} | {g['r1_map']:.4f} | {c['diff']:+.4f} | [{c['ci95'][0]:+.4f}, "
                     f"{c['ci95'][1]:+.4f}] | {c['verdict']} |")
    lines += ["", "| target | " + " | ".join(f"{c} MAE" for c in ("ceiling", "another_scene", "base_none_p08", "base_lulc_p08",
                                                                       "r1_none_p08", "r1_lulc_p08", "r1_map_p08")) + " |",
              "|---" * 8 + "|"]
    for T, o in res["decoded"].items():
        cs = o["conditions"]
        lines.append(f"| {T} | " + " | ".join(f"{cs[c]['mae']:.4g}" if c in cs else "–" for c in
                                               ("ceiling", "another_scene", "base_none_p08", "base_lulc_p08", "r1_none_p08",
                                                "r1_lulc_p08", "r1_map_p08")) + " |")
    (REPORT / "part3.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["truth1", "decode1", "score1", "decode3", "score3"])
    ap.add_argument("--allow-titanv", action="store_true")
    ap.add_argument("--targets", nargs="*", choices=TARGETS3)
    a = ap.parse_args()
    return {"truth1": truth1, "decode1": decode1, "score1": score1, "decode3": decode3, "score3": score3}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
