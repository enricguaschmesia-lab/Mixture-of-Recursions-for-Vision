#!/usr/bin/env python
"""Continuous-I/O pilot, analysis: how many numbers does a patch need, and how the variants compare. (`.venv`)

    python eo/experiments/cio_analysis.py pca        # CPU, ~5 min
    python eo/experiments/cio_analysis.py report     # after cio_train eval/rollout and cio_decode

pca     Per modality (S2L2A, S1RTC, DEM): PCA of 16x16xC patches fitted on 768 train rows, and an
        UNTRAINED 2-D DCT kept to its lowest frequencies, scored on the 256 held-out rows the
        tokenizer round trip was decoded on (cio_decode.py roundtrip). Reported per k kept numbers:
        reconstruction MSE in z units^2, and the k at which each matches the FSQ token's round trip.
        A token is log2(15,360) = 13.9 bits; this asks how many continuous dimensions carry as much.
report  Tables (markdown + json) and figures from every run's eval_tf.json and rollouts.

Output: /data/enric/experiments/continuous_io/analysis/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eo.experiments.cio_train import (CONT, MODS, NCH, NP, P, PAIRS, patchify, unpatchify,   # noqa: E402
                                      hf_energy, miou)

ROOT = Path("/data/enric/experiments/continuous_io")
OUT = ROOT / "analysis"
N_RT = 256          # rows decoded by cio_decode.py roundtrip


def dct_matrix(n: int = P) -> np.ndarray:
    u, x = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    m = np.cos(np.pi * (2 * x + 1) * u / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m                                               # orthonormal: m @ m.T == I


def dct_order(n: int = P) -> np.ndarray:
    """Coefficient order for one channel: by u+v (frequency band), then by u."""
    u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return np.lexsort((u.ravel(), (u + v).ravel()))


def cmd_pca(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rep = {}
    M = torch.from_numpy(dct_matrix()).float()
    order = dct_order()
    for m in CONT:
        C, D = NCH[m], NCH[m] * P * P
        tr = np.load(ROOT / f"data/train_{m}.npy", mmap_mode="r")
        rows = np.sort(rng.choice(len(tr), a.rows, replace=False))
        X = patchify(torch.from_numpy(tr[rows].astype(np.float32))).reshape(-1, D).double()
        mu = X.mean(0)
        Xc = X - mu
        cov = (Xc.T @ Xc) / len(Xc)
        ev, evec = torch.linalg.eigh(cov)
        ev, evec = ev.flip(0), evec.flip(1)
        del X, Xc, cov
        te = np.load(ROOT / f"data/test_{m}.npy", mmap_mode="r")[:N_RT].astype(np.float32)
        Y = patchify(torch.from_numpy(te)).reshape(-1, D).double()
        tot = ((Y - mu) ** 2).sum(1).mean() / D                      # MSE of the mean patch
        coef = (Y - mu) @ evec                                         # (n, D)
        pca_mse = tot - (coef ** 2).mean(0).cumsum(0) / D              # after keeping k components
        # DCT: per channel, same transform for any band; keep the m lowest frequencies of each
        Yp = Y.float().reshape(-1, C, P, P)
        dc = torch.einsum("ux,ncxy,vy->ncuv", M, Yp, M).reshape(-1, C, P * P)[:, :, order]
        e_dct = (dc.double() ** 2).mean(0)                              # (C, 256) energy per coef
        dct_mse_per_m = (e_dct.sum() - e_dct.cumsum(1).sum(0)) / D      # keep m coefs per channel
        # the tokenizer's round trip on the same rows
        dec = np.load(ROOT / f"roundtrip/{m}_gt_n{N_RT}_decoded.npy").astype(np.float32)
        fsq_mse = float(((dec - te) ** 2).mean())
        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, D]
        ks = sorted({k for k in ks if k <= D})
        k_pca = int((pca_mse <= fsq_mse).nonzero()[0][0]) + 1 if (pca_mse <= fsq_mse).any() else None
        m_dct = int((dct_mse_per_m <= fsq_mse).nonzero()[0][0]) + 1 if (dct_mse_per_m <= fsq_mse).any() else None
        frac = (ev / ev.sum()).cumsum(0)
        rep[m] = {
            "dim": D, "train_patches": int(len(rows) * NP), "test_patches": int(N_RT * NP),
            "mean_patch_mse": float(tot), "fsq_roundtrip_mse": fsq_mse,
            "pca_mse": {k: float(pca_mse[k - 1]) for k in ks},
            "dct_mse": {mm * C: float(dct_mse_per_m[mm - 1]) for mm in (1, 2, 4, 8, 16, 32, 64, 128, 256)},
            "k_pca_matching_fsq": k_pca, "k_dct_matching_fsq": None if m_dct is None else m_dct * C,
            "k_var": {p: int((frac >= p).nonzero()[0][0]) + 1 for p in (0.9, 0.95, 0.99, 0.999)},
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "fsq_bits_per_patch": float(np.log2(15360)),
        }
        print(m, json.dumps({k: v for k, v in rep[m].items() if k not in ("pca_mse", "dct_mse")}), flush=True)
        print("   pca", {k: round(v, 4) for k, v in rep[m]["pca_mse"].items()})
        print("   dct", {k: round(v, 4) for k, v in rep[m]["dct_mse"].items()}, flush=True)
    # LULC: the tokenizer's round trip in pixel accuracy / mIoU
    te = np.load(ROOT / "data/test_LULC.npy")[:N_RT]
    dec = np.load(ROOT / f"roundtrip/LULC_gt_n{N_RT}_decoded.npy")
    conf = torch.zeros(10, 10, dtype=torch.long)
    idx = torch.from_numpy(te.astype(np.int64)).reshape(-1) * 10 + torch.from_numpy(dec.astype(np.int64)).reshape(-1)
    conf += torch.bincount(idx, minlength=100).reshape(10, 10)
    mi, acc = miou(conf)
    rep["LULC"] = {"fsq_roundtrip_pix_acc": acc, "fsq_roundtrip_miou": mi}
    print("LULC", rep["LULC"])
    (OUT / "pca_dct.json").write_text(json.dumps(rep, indent=1))
    return 0


# ----------------------------------------------------------------------------- report
TOKEN_RUNS = ("tok_in", "lin_in", "linraw_in", "linrel_in", "band_in", "bandrop_in")      # FSQ-token targets
CONT_RUNS = ("reg_out", "flow_out", "flowdct_out")               # continuous targets
N_RO = 128          # rows the rollouts were generated on
SUBSETS = {"1,2,3": "RGB (B02-B04)", "1,2,3,7": "RGB + NIR (B02-B04, B08)",
           "1,2,3,4,5,6,7,8,10,11": "PASTIS's 10 bands (no B01, B09)", "7": "B08 alone"}


def load_json(f: Path):
    return json.loads(f.read_text()) if f.exists() else None


def run_dir(run_id: str) -> Path:
    return ROOT / "runs" / run_id


def boot_diff(a, b, B=5000, seed=0):
    """Paired mean difference a - b over rows, with a 95% bootstrap interval."""
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def to_image(arr: np.ndarray, m: str) -> np.ndarray:
    """A rollout or decode -> (N, C, 224, 224) float32 z-scores, or (N, 224, 224) classes."""
    if m == "LULC":
        return arr.astype(np.int64)
    if arr.ndim == 3:                                    # (N, 196, D) patches
        return unpatchify(torch.from_numpy(arr.astype(np.float32)), NCH[m]).numpy()
    return arr.astype(np.float32)


def cont_scores(samples: list, y: np.ndarray) -> dict:
    """samples: K arrays (N, C, H, W); y (N, C, H, W). Per-value-normalised energy score."""
    K, n = len(samples), y[0].size
    X = np.stack(samples).reshape(K, len(y), -1)
    Y = y.reshape(len(y), -1)
    d_obs = np.sqrt(((X - Y[None]) ** 2).sum(-1)) / np.sqrt(n)                   # (K, N)
    es = d_obs.mean(0)
    if K > 1:
        pair = [np.sqrt(((X[k] - X[l]) ** 2).sum(-1)) / np.sqrt(n) for k in range(K) for l in range(K) if k != l]
        es = es - 0.5 * np.mean(pair, 0)
    hp, tp = hf_energy(torch.from_numpy(np.stack(samples).reshape(-1, *y.shape[1:])))
    hg, tg = hf_energy(torch.from_numpy(y))
    hp, tp = hp.reshape(K, *hg.shape).mean(0), tp.reshape(K, *tg.shape).mean(0)
    return {"k": K, "mse_sample": float(((X - Y[None]) ** 2).mean()),
            "mse_mean": float(((X.mean(0) - Y) ** 2).mean()),
            "energy_score": float(es.mean()),
            # medians over (image, channel): a smooth DEM or open-water scene has ~0 true HF power,
            # and a mean of per-image ratios is then dominated by those (2026-10-28: 295,790)
            "hf_ratio": float((hp / hg.clamp_min(1e-9)).median()),
            "power_ratio": float((tp / tg.clamp_min(1e-9)).median())}


def lulc_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    conf = torch.bincount(torch.from_numpy(y.reshape(-1) * 10 + pred.reshape(-1)), minlength=100).reshape(10, 10)
    mi, acc = miou(conf)
    return {"miou": mi, "pix_acc": acc}


def rollout_files(run_id: str, s: str, t: str, mode: str, par: bool, decoded: bool):
    d = run_dir(run_id) / "rollout"
    suf = ("_par" if par else "") + ("_decoded" if decoded else "")
    return sorted(d.glob(f"{s}_{t}_k*_{mode}{suf}.npy"))


def _cell(r, t=None):
    """t: the target modality, to put the per-patch energy score per value (/ sqrt(dim))."""
    if r is None:
        return ""
    if "mse_mean_pred" in r:
        es = r["energy_score"] / np.sqrt(NCH[t] * P * P) if t else r["energy_score"]
        return f"{r['mse_mean_pred']:.3f} / {es:.3f}"
    if "miou" in r:
        return f"{r['ce']:.3f} ({r['miou']:.2f})"
    return f"{r['ce']:.3f} ({100 * r['acc']:.1f}%)"


def cmd_report(a) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep, md = {"eval": {}, "rollouts": {}, "contrasts": {}, "seeds": {}, "subsets": {}}, []
    runs = sorted({p.name for p in (ROOT / "runs").glob("*_s[0-9]") if (p / "eval_tf.json").exists()})
    for rid in runs:
        for f in run_dir(rid).glob("eval_*.json"):
            if f.stem.endswith("_rows"):
                continue
            rep["eval"][f"{rid}:{f.stem[5:]}"] = load_json(f)["results"]
    ev = lambda rid, tag, key: rep["eval"].get(f"{rid}:{tag}", {}).get(key)
    # T1: token targets, source only (r = 1), the shuffled-source control and the gain
    tok = [r for r in (f"{v}_s0" for v in TOKEN_RUNS) if r in runs]
    md.append("## T1. Token targets, source only (r = 1): CE nats/token (top-1), 1,024 held-out rows\n")
    md.append("gain = CE with another row's source - CE with the true source: what the source representation "
              "tells the model about the target.\n")
    md.append("| pair | " + " | ".join(f"{r} | {r} shuffled | {r} gain" for r in tok) + " |")
    md.append("|" + "---|" * (1 + 3 * len(tok)))
    for s_, t_ in PAIRS:
        k = f"{s_}>{t_}"
        cells = []
        for rid in tok:
            tr, sh = ev(rid, "mask1", k), ev(rid, "mask1_shufsrc", k)
            cells += [_cell(tr), _cell(sh), "" if (tr is None or sh is None) else f"{sh['ce'] - tr['ce']:+.3f}"]
        md.append(f"| {k} | " + " | ".join(cells) + " |")
    # T2: token targets, autoregressive (r = 0)
    md.append("\n## T2. Token targets, autoregressive teacher forcing (r = 0): CE nats/token (top-1)\n")
    md.append("| pair | " + " | ".join(tok) + " |"); md.append("|" + "---|" * (1 + len(tok)))
    for s_, t_ in PAIRS:
        k = f"{s_}>{t_}"
        md.append(f"| {k} | " + " | ".join(_cell(ev(rid, "tf", k)) for rid in tok) + " |")
    # paired contrasts against tok_in, per pair, both ends
    for tag in ("mask1", "tf"):
        rows = {rid: load_json(run_dir(rid) / f"eval_{tag}_rows.json") for rid in tok}
        for rid in tok:
            if rid == "tok_in_s0" or rows.get(rid) is None or rows.get("tok_in_s0") is None:
                continue
            for s_, t_ in PAIRS:
                k = f"{s_}>{t_}:ce"
                rep["contrasts"][f"{rid}-tok_in_s0:{tag}:{s_}>{t_}"] = boot_diff(rows[rid][k], rows["tok_in_s0"][k])
    if rep["contrasts"]:
        md.append("\n## T3. Paired CE difference against tok_in_s0 (nats/token, 95% bootstrap CI over rows; "
                  "negative = better than FSQ-token input)\n")
        md.append("| run | eval | " + " | ".join(f"{s_}>{t_}" for s_, t_ in PAIRS) + " | mean |")
        md.append("|" + "---|" * (3 + len(PAIRS)))
        for rid in tok[1:]:
            for tag in ("mask1", "tf"):
                cs = [rep["contrasts"].get(f"{rid}-tok_in_s0:{tag}:{s_}>{t_}") for s_, t_ in PAIRS]
                if not any(cs):
                    continue
                cells = ["" if c is None else f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]" for c in cs]
                mean = np.mean([c[0] for c in cs if c is not None])
                md.append(f"| {rid} | {'r=1' if tag == 'mask1' else 'r=0'} | " + " | ".join(cells) + f" | {mean:+.3f} |")
    # T4: seed spread of the input contrast
    seeds = {}
    for v in ("tok_in", "lin_in"):
        for sd in (0, 1):
            rid = f"{v}_s{sd}"
            for tag in ("mask1", "tf"):
                r = rep["eval"].get(f"{rid}:{tag}")
                if r:
                    seeds[f"{rid}:{tag}"] = float(np.mean([r[f"{s_}>{t_}"]["ce"] for s_, t_ in PAIRS]))
    rep["seeds"] = seeds
    if seeds:
        md.append("\n## T4. Mean CE over the 12 pairs, per seed (nats/token)\n")
        md.append("| run | r=1 (source only) | r=0 (AR) |"); md.append("|---|---|---|")
        for v in ("tok_in", "lin_in"):
            for sd in (0, 1):
                rid = f"{v}_s{sd}"
                if f"{rid}:tf" in seeds or f"{rid}:mask1" in seeds:
                    md.append(f"| {rid} | {seeds.get(f'{rid}:mask1', float('nan')):.4f} | {seeds.get(f'{rid}:tf', float('nan')):.4f} |")
    # T5: continuous-output runs, teacher forced
    cont = [r for r in (f"{v}_s0" for v in CONT_RUNS) if r in runs]
    md.append("\n## T5. Continuous-output runs: MSE of the predicted mean (z units^2; regression's output, or "
              "the mean of 4 flow samples) / energy score per value; LULC: per-pixel CE (mIoU)\n")
    md.append("| pair | " + " | ".join(f"{r} r=0 | {r} r=1 | {r} r=1 shuffled" for r in cont) + " |")
    md.append("|" + "---|" * (1 + 3 * len(cont)))
    for s_, t_ in PAIRS:
        k = f"{s_}>{t_}"
        md.append(f"| {k} | " + " | ".join(_cell(ev(rid, tag, k), t_) for rid in cont
                                           for tag in ("tf", "mask1", "mask1_shufsrc")) + " |")
    # T6: band subsets through the band-agnostic embedders
    sub_runs = [r for r in ("band_in_s0", "bandrop_in_s0") if r in runs]
    if any(f"{rid}:mask1_bands{k.replace(',', '-')}" in rep["eval"] for rid in sub_runs for k in SUBSETS):
        md.append("\n## T6. S2L2A source with a band subset never chosen by hand (r = 1, CE nats/token)\n")
        md.append("| run | bands | S2L2A>LULC | S2L2A>DEM | S2L2A>S1RTC |"); md.append("|---|---|---|---|---|")
        for rid in sub_runs:
            md.append(f"| {rid} | all 12 | " + " | ".join(_cell(ev(rid, "mask1", f"S2L2A>{t_}")) for t_ in ("LULC", "DEM", "S1RTC")) + " |")
            for k, name in SUBSETS.items():
                tag = f"mask1_bands{k.replace(',', '-')}"
                md.append(f"| {rid} | {name} | " + " | ".join(_cell(ev(rid, tag, f"S2L2A>{t_}")) for t_ in ("LULC", "DEM", "S1RTC")) + " |")
            md.append(f"| {rid} | another row's source | " + " | ".join(_cell(ev(rid, "mask1_shufsrc", f"S2L2A>{t_}")) for t_ in ("LULC", "DEM", "S1RTC")) + " |")
    # T7: rollouts, scored in pixels
    gt = {m: np.load(ROOT / f"data/test_{m}.npy", mmap_mode="r")[:N_RO] for m in MODS}
    ceil = {}
    for m in MODS:
        f = ROOT / f"roundtrip/{m}_gt_n{N_RT}_decoded.npy"
        if f.exists():
            dec = np.load(f)[:N_RO]
            ceil[m] = lulc_metrics(dec.astype(np.int64), gt[m].astype(np.int64)) if m == "LULC" else \
                cont_scores([dec.astype(np.float32)], gt[m].astype(np.float32))
    rep["tokenizer_ceiling"] = ceil
    for rid in ("tok_in_s0", "lin_in_s0", "reg_out_s0", "flow_out_s0", "flowdct_out_s0"):
        tokens = rid.rsplit("_s", 1)[0] in TOKEN_RUNS
        for s_, t_ in [tuple(x.split(":")) for x in ("S1RTC:S2L2A", "LULC:S2L2A", "S2L2A:DEM", "S2L2A:S1RTC",
                                                     "S2L2A:LULC", "S1RTC:LULC")]:
            for par in (False, True):
                if t_ == "LULC":
                    files = rollout_files(rid, s_, t_, "greedy", par, tokens)
                    if files:
                        pred = to_image(np.load(files[0]), "LULC")
                        rep["rollouts"][f"{rid}:{s_}>{t_}:{'par' if par else 'ar'}"] = lulc_metrics(pred, gt["LULC"].astype(np.int64))
                    continue
                flow = rid.startswith("flow")
                for mode in (["sample", "greedy"] if par else ["sample"]):
                    files = rollout_files(rid, s_, t_, mode, par, tokens)
                    if not files or (par and flow and mode == "greedy"):
                        continue
                    y = gt[t_].astype(np.float32)
                    rep["rollouts"][f"{rid}:{s_}>{t_}:{'par' if par else 'ar'}-{mode}"] = cont_scores(
                        [to_image(np.load(f), t_) for f in files], y)
    md.append("\n## T7. Generation from the source, scored in pixels (128 held-out rows)\n")
    md.append("ar = autoregressive, sampled (k = 2 draws; regression feeds back its mean); par = one pass with "
              "every target input masked (sample: k = 2 independent draws per patch; greedy: argmax tokens, or "
              "the regression mean). Token runs are decoded by TerraMind's decoders (50 steps, unclamped). "
              "MSE in z units^2. ES = energy score per value (a proper score: lower is better; it rewards both "
              "accuracy and calibrated spread). HF = high-frequency power relative to the truth, median over images and bands (1 = as sharp; "
              "power = total variance relative to the truth, same median).\n")
    md.append("| pair | run | mode | k | MSE (one draw) | MSE (mean of draws) | energy score | HF ratio | power ratio |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for key, r in sorted(rep["rollouts"].items(), key=lambda kv: (kv[0].split(":")[1], kv[0].split(":")[2], kv[0])):
        rid, pair, mode = key.split(":")
        if "miou" in r:
            continue
        md.append(f"| {pair} | {rid} | {mode} | {r['k']} | {r['mse_sample']:.3f} | {r['mse_mean']:.3f} | "
                  f"{r['energy_score']:.3f} | {r['hf_ratio']:.2f} | {r['power_ratio']:.2f} |")
    for m in ("S2L2A", "DEM", "S1RTC"):
        if m in ceil:
            c = ceil[m]
            md.append(f"| tokenizer round trip ({m}) | GT tokens | decode | 1 | {c['mse_sample']:.3f} | | "
                      f"{c['energy_score']:.3f} | {c['hf_ratio']:.2f} | {c['power_ratio']:.2f} |")
        y = gt[m].astype(np.float32)
        md.append(f"| predict z = 0 ({m}) | none | | | {float((y ** 2).mean()):.3f} | | | | |")
    md.append("\n### T7b. LULC targets (mIoU / pixel accuracy, greedy)\n")
    md.append("| pair | run | mode | mIoU | pixel acc |"); md.append("|---|---|---|---|---|")
    for key, r in sorted(rep["rollouts"].items()):
        rid, pair, mode = key.split(":")
        if "miou" in r:
            md.append(f"| {pair} | {rid} | {mode} | {r['miou']:.3f} | {r['pix_acc']:.3f} |")
    if "LULC" in ceil:
        md.append(f"| tokenizer round trip | GT tokens | decode | {ceil['LULC']['miou']:.3f} | {ceil['LULC']['pix_acc']:.3f} |")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    (OUT / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


# ----------------------------------------------------------------------------- figures
# The reference categorical palette (dataviz skill, references/palette.md), slots in fixed order;
# inks and surface from the same file. LULC keeps the project's ESRI class colours
# (s12_decoded_eval.py), copied rather than imported to keep that module's imports out.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LULC_NAMES = ["no data", "water", "trees", "flooded veg.", "crops", "built", "bare", "snow/ice",
              "clouds", "rangeland"]
LULC_COLORS = ["#d9d9d9", "#419bdf", "#397d49", "#7a87c6", "#e49635", "#c4281b", "#a59b8f",
               "#a8ebff", "#616161", "#e3e2c3"]


def _style(ax):
    ax.set_facecolor(SURF)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_pca(plt):
    rep_ = load_json(OUT / "pca_dct.json")
    if rep_ is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=SURF)
    for ax, m in zip(axes, CONT):
        r = rep_[m]
        _style(ax)
        D = r["dim"]
        floor = 1e-6           # full-rank points are exactly 0 and cannot sit on a log axis
        kp = sorted(int(k) for k in r["pca_mse"] if int(k) < D)
        kd = sorted(int(k) for k in r["dct_mse"] if int(k) < D)
        yp = [max(r["pca_mse"][str(k)], floor) for k in kp]
        yd = [max(r["dct_mse"][str(k)], floor) for k in kd]
        ax.plot(kp, yp, color=SERIES[0], lw=2, marker="o", ms=4, label="PCA, fitted per modality")
        ax.plot(kd, yd, color=SERIES[1], lw=2, marker="s", ms=4, label="DCT, untrained, same for any band")
        f = r["fsq_roundtrip_mse"]
        ax.axhline(f, color=INK2, lw=1.5, ls="--", label="one FSQ token, decoded")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        lo = max(min(yp + yd + [f]) / 3, floor / 2)
        ax.set_ylim(lo, max(r["mean_patch_mse"], f) * 3)
        kf = r["k_pca_matching_fsq"]
        ax.set_title(f"{m} ({D:,} values per patch)", color=INK, fontsize=10.5, loc="left")
        ax.text(0.97, 0.95, f"token ≈ PCA k = {kf}\ntoken ≈ DCT k = {r['k_dct_matching_fsq']}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK2)
        ax.set_xlabel("continuous numbers kept per 16x16 patch", color=INK2, fontsize=9)
        if m == CONT[0]:
            ax.set_ylabel("reconstruction MSE, held-out (z units²)", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("How many continuous numbers per patch carry what one 13.9-bit FSQ token carries (in MSE)?",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "fig_pca_dct.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def fig_source_gain(plt):
    """Nats/token gained from the true source over another row's source, at r = 1."""
    order = ("tok_in", "linraw_in", "linrel_in", "band_in")      # four slots of the reference palette
    runs = [r for r in order if (ROOT / "runs" / f"{r}_s0" / "eval_mask1_shufsrc.json").exists()]
    if not runs:
        return
    label = {"tok_in": "FSQ tokens (today)", "linraw_in": "raw patch, per-modality linear",
             "linrel_in": "raw patch, level + relief, linear", "band_in": "raw patch, one band-agnostic embedder"}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), facecolor=SURF, sharey=False)
    for ax, t_ in zip(axes, MODS):
        _style(ax)
        srcs = [s_ for s_ in MODS if s_ != t_]
        w = 0.8 / len(runs)
        for i, run in enumerate(runs):
            tru = load_json(ROOT / "runs" / f"{run}_s0" / "eval_mask1.json")["results"]
            shf = load_json(ROOT / "runs" / f"{run}_s0" / "eval_mask1_shufsrc.json")["results"]
            g = [shf[f"{s_}>{t_}"]["ce"] - tru[f"{s_}>{t_}"]["ce"] for s_ in srcs]
            ax.bar(np.arange(len(srcs)) + (i - (len(runs) - 1) / 2) * w, g, width=w * 0.92,
                   color=SERIES[order.index(run)], label=label[run], edgecolor=SURF, linewidth=1)
        ax.set_xticks(range(len(srcs))); ax.set_xticklabels([f"{s_}" for s_ in srcs], fontsize=9, color=INK2)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title(f"target {t_}", color=INK, fontsize=10.5, loc="left")
        if t_ == MODS[0]:
            ax.set_ylabel("nats/token gained from the true source", color=INK2, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=len(runs), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Source-only prediction (every target input masked): nats/token the true source is worth, "
                 "against another row's source", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(OUT / "fig_source_gain.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def _rgb(z):
    """S2L2A z-scores (12, H, W) -> RGB in [0, 1]: B04/B03/B02, reflectance 0-0.3, gamma 1/1.6."""
    from eo.terramesh_tok import contract as C
    mean = np.asarray(C.V1_TOK_MEAN["S2L2A"])[[3, 2, 1], None, None]
    std = np.asarray(C.V1_TOK_STD["S2L2A"])[[3, 2, 1], None, None]
    refl = (z[[3, 2, 1]] * std + mean) / 10000.0
    return np.clip(refl / 0.3, 0, 1).transpose(1, 2, 0) ** (1 / 1.6)


def _panel(ax, m, img, ref=None):
    from matplotlib.colors import ListedColormap
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    if img is None:
        ax.set_facecolor(SURF); return
    if m == "S2L2A":
        ax.imshow(_rgb(img))
    elif m == "LULC":
        ax.imshow(img, cmap=ListedColormap(LULC_COLORS), vmin=0, vmax=9, interpolation="nearest")
    else:
        r = ref if ref is not None else img
        lo, hi = np.percentile(r[0], 2), np.percentile(r[0], 98)
        ax.imshow(img[0], cmap="Greys_r", vmin=lo, vmax=max(hi, lo + 1e-6))


def fig_examples(plt, s_, t_, rows=(0, 1, 2)):
    gt_t = np.load(ROOT / f"data/test_{t_}.npy", mmap_mode="r")
    gt_s = np.load(ROOT / f"data/test_{s_}.npy", mmap_mode="r")
    cols = [("source " + s_, None), ("truth " + t_, None)]
    mode_ar = "greedy" if t_ == "LULC" else "sample"
    spec = [("tok_in", False, "FSQ in, FSQ out (AR)"), ("lin_in", False, "pixels in, FSQ out (AR)"),
            ("reg_out", False, "pixels in/out, regression (AR)"), ("flow_out", False, "pixels in/out, flow (AR)"),
            ("flowdct_out", False, "pixels in, DCT flow (AR)"),
            ("lin_in", True, "pixels in, FSQ out (1 pass)"), ("reg_out", True, "regression (1 pass)"),
            ("flow_out", True, "flow (1 pass)")]
    preds = []
    for run, par, name in spec:
        tokens = run in TOKEN_RUNS
        mode = mode_ar if not par else ("sample" if (run.startswith("flow") and t_ != "LULC") else "greedy")
        f = rollout_files(f"{run}_s0", s_, t_, mode, par, tokens)
        preds.append((name, to_image(np.load(f[0]), t_) if f else None))
    n_c = 2 + len(preds)
    fig, axes = plt.subplots(len(rows), n_c, figsize=(1.75 * n_c, 1.85 * len(rows) + 0.6), facecolor=SURF)
    for i, r in enumerate(rows):
        src = gt_s[r].astype(np.int64) if s_ == "LULC" else gt_s[r].astype(np.float32)
        tru = gt_t[r].astype(np.int64) if t_ == "LULC" else gt_t[r].astype(np.float32)
        _panel(axes[i, 0], s_, src)
        _panel(axes[i, 1], t_, tru)
        for j, (name, pr) in enumerate(preds):
            _panel(axes[i, 2 + j], t_, None if pr is None else pr[r], ref=tru)
    titles = [f"source {s_}", f"truth {t_}"] + [n for n, _ in preds]
    for j, ttl in enumerate(titles):
        axes[0, j].set_title(ttl, fontsize=7.5, color=INK, wrap=True)
    fig.suptitle(f"{s_} -> {t_}, held-out rows {list(rows)}", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_examples_{s_}_{t_}.png", dpi=150, facecolor=SURF)
    plt.close(fig)


def cmd_figures(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pca(plt)
    fig_source_gain(plt)
    for s_, t_ in (("S1RTC", "S2L2A"), ("LULC", "S2L2A"), ("S2L2A", "DEM"), ("S2L2A", "LULC"), ("S2L2A", "S1RTC")):
        fig_examples(plt, s_, t_)
    print("figures in", OUT)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = p.add_subparsers(dest="cmd", required=True)
    pc = sp.add_parser("pca"); pc.add_argument("--rows", type=int, default=768)
    sp.add_parser("report")
    sp.add_parser("figures")
    a = p.parse_args()
    return {"pca": cmd_pca, "report": cmd_report, "figures": cmd_figures}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
