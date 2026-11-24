#!/usr/bin/env python
"""A2 R1's read, by the reading rule written before the run (docs/plans/a2_segmentation.md, Step 5.1; `.venv`).

    curves      every logged eval and train loss of the run (and its extensions, in order) from trainer_state.json; the
                extension test at 0.9 N (map CE more than 1% below its value 300 steps earlier, and the logged
                eval_single_loss within +0.02 of step 0's); the first eval at which the map CE fell less than 1% over
                300 steps (description)
    forgetting  eval_forgetting.py's two fp32 readings on the 4,416 held-out rows (the map base "before", an R1
                checkpoint "after"): the difference of E1's ce_all_body_tokens, paired per token, a row bootstrap
                (2,000 resamples, seed 0), per modality; admissible iff the difference is at most +0.02 nats
    corpus      the held-out map CE of the `map` and `map_s2` eval sets per row on one checkpoint (GPU), split by corpus:
                val's Major TOM rows carry S1RTC beside a map, which training never shows. The rows go through the model
                --chunk at a time (default 2: a 16-row micro-batch's fp32 logits are 12.3 GB); --limit N micro-batches
                and --device cpu make a smoke. Beside it, the checkpoint's last logged eval (bf16, in training)

    python eo/experiments/segmentation/r1_read.py curves --run <run dir> [--run <extension dir> ...]
    python eo/experiments/segmentation/r1_read.py forgetting --before <tag dir>/extended_base --after <tag dir>/checkpoint-N
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<card> \\
        python eo/experiments/segmentation/r1_read.py corpus --checkpoint <run>/checkpoint-N [--precision bf16|fp32] \\
        [--chunk 2] [--limit N] [--device cuda|cpu]

Each writes /data/enric/reports/a2_seg/r1/<what>_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path("/data/enric/reports/a2_seg/r1")
MAP_KEY = "eval_map_loss_MAP"
EXT_WINDOW, EXT_REL, FORGET_MARGIN = 300, 0.01, 0.02


def write(name: str, doc: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(doc, indent=1, default=float))
    print(f"-> {p}")
    return p


def history(runs) -> dict:
    """{step: {key: value}} over the runs in order (a later run's steps replace an earlier run's, as an extension
    resumed from 0.9 N replaces the first run's decayed tail)."""
    out = {}
    for r in runs:
        st = json.loads((Path(r) / "trainer_state.json").read_text()) if (Path(r) / "trainer_state.json").is_file() \
            else json.loads(sorted(Path(r).glob("checkpoint-*/trainer_state.json"),
                                   key=lambda p: int(p.parent.name.split("-")[1]))[-1].read_text())
        steps = {}
        for h in st["log_history"]:
            steps.setdefault(int(h["step"]), {}).update(h)
        first = min(steps) if steps else 0
        out = {k: v for k, v in out.items() if k < first} | steps
        out["_max_steps"] = st.get("max_steps")
    return out


def curves(a) -> None:
    h = history(a.run)
    N = int(h.pop("_max_steps"))
    ev = {s: d for s, d in sorted(h.items()) if MAP_KEY in d}
    single = {s: d["eval_single_loss"] for s, d in sorted(h.items()) if "eval_single_loss" in d}
    s2 = {s: d["eval_map_s2_loss"] for s, d in sorted(h.items()) if "eval_map_s2_loss" in d}
    mp = {s: d[MAP_KEY] for s, d in ev.items()}
    t = int(round(0.9 * N))
    test = None
    if t in mp and t - EXT_WINDOW in mp and 0 in single and t in single:
        still = mp[t] < (1 - EXT_REL) * mp[t - EXT_WINDOW]
        within = single[t] <= single[0] + FORGET_MARGIN
        test = {"t": t, "map_ce_t": mp[t], "map_ce_t_minus_300": mp[t - EXT_WINDOW],
                "relative_drop": 1 - mp[t] / mp[t - EXT_WINDOW], "still_falling": bool(still),
                "single_t": single[t], "single_0": single[0], "single_within_0.02": bool(within),
                "extend": bool(still and within)}
    flat = next((s for s in mp if s >= EXT_WINDOW and s - EXT_WINDOW in mp and s < t + 1
                 and mp[s] >= (1 - EXT_REL) * mp[s - EXT_WINDOW]), None)
    train = {s: {k: d[k] for k in d if k in ("loss", "loss_MAP", "grad_norm", "learning_rate")}
             for s, d in sorted(h.items()) if "loss" in d and "grad_norm" in d}
    gn = [v["grad_norm"] for v in train.values()]
    doc = {"runs": [str(r) for r in a.run], "max_steps": N, "map_ce": mp, "map_s2_ce": s2, "single": single,
           "single_rise_last": (list(single.values())[-1] - single[0]) if single else None,
           "extension_test": test, "flatten_step": flat,
           "grad_norm": {"max": max(gn) if gn else None, "median": float(np.median(gn)) if gn else None,
                         "nonfinite": int(sum(not np.isfinite(g) for g in gn))},
           "train": train}
    print(json.dumps({k: doc[k] for k in ("max_steps", "extension_test", "flatten_step", "single_rise_last",
                                          "grad_norm")}, indent=1, default=float))
    write(f"curves_{a.tag or Path(a.run[-1]).name}", doc)


def forgetting(a) -> None:
    before, after = Path(a.before), Path(a.after)
    jb, ja = (json.loads((d / "forgetting.json").read_text()) for d in (before, after))
    if jb["precision"] != ja["precision"] or jb["arm_config"] != ja["arm_config"] or jb["n_rows"] != ja["n_rows"]:
        raise SystemExit(f"the readings differ in precision / arm-config / rows: {jb['precision']} {ja['precision']}")
    tb, ta = np.load(before / "tokens_ce.npz"), np.load(after / "tokens_ce.npz")
    if not (np.array_equal(tb["rows"], ta["rows"]) and np.array_equal(tb["label_mod"], ta["label_mod"])):
        raise SystemExit("the two readings did not score the same rows and tokens")
    cb, ca, lm = tb["ce"].astype(np.float64), ta["ce"].astype(np.float64), tb["label_mod"]
    body = lm > 0
    d_tok = (ca - cb) * body
    n_row = body.sum(1)
    mean = float(d_tok.sum() / body.sum())
    rng = np.random.default_rng(0)
    rs, ns = d_tok.sum(1), n_row
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(rs), len(rs))
        boot.append(rs[i].sum() / ns[i].sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    from eo.data import eo_vocab_nc as V
    per = {}
    for mid, name in V.ID_TO_MODALITY.items():
        m = lm == mid
        if m.any():
            per[name] = {"before": float((cb * m).sum() / m.sum()), "after": float((ca * m).sum() / m.sum()),
                         "diff": float(((ca - cb) * m).sum() / m.sum()), "tokens": int(m.sum())}
    doc = {"before": str(before), "after": str(after), "precision": ja["precision"],
           "ce_before": jb["ce_all_body_tokens"], "ce_after": ja["ce_all_body_tokens"],
           "diff": mean, "diff_ci95": [float(lo), float(hi)], "margin": FORGET_MARGIN,
           "admissible": bool(mean <= FORGET_MARGIN), "per_modality": per,
           "check_reproduces_json": [abs(float((cb * body).sum() / body.sum()) - jb["ce_all_body_tokens"]) < 1e-3,
                                     abs(float((ca * body).sum() / body.sum()) - ja["ce_all_body_tokens"]) < 1e-3]}
    print(json.dumps({k: doc[k] for k in ("ce_before", "ce_after", "diff", "diff_ci95", "admissible",
                                          "check_reproduces_json")}, indent=1))
    write(f"forgetting_{a.tag or after.name}", doc)


def corpus(a) -> None:
    import pandas as pd
    import torch
    from hydra import compose, initialize_config_dir
    from eo.data import map_mix_dataset as MD
    from eo.data import map_vocab as MV
    from eo.finetune.init import strict_load
    from eo.generate import conditional as G
    from util.env import load_dotenv
    load_dotenv()
    import os
    ov = [f"r1.val_maps={a.val_maps}"] if a.val_maps else []
    if not os.environ.get("TERRAMESH_TRAIN_TOK_ROOT"):
        # the workstation has no train root: build_eval resolves one but reads only val, so the gate's two-shard
        # fixture stands in (eo/experiments/segmentation/r1_mini_root.py; verify_seg_r1.py does the same)
        tm = Path("/data/enric/data/TerraMesh")
        os.environ["TERRAMESH_TRAIN_TOK_ROOT"] = str(tm / "train_tok_partial_r1")
        ov += ["++multimodal.allow_partial_train_root=true", f"r1.train_maps={tm / 'maps/mini_r1/MAP_aligned'}"]
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        cfg = compose(config_name=a.config, overrides=ov)
    ev = MD.build_eval(cfg)
    model, _ = G.build_model(a.config, None, device="cpu")
    strict_load(model, Path(a.checkpoint))
    dev = a.device
    model.to(dev).eval()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.precision]
    corp = pd.read_parquet(Path(os.environ["TERRAMESH_TOK_ROOT"]) / "tok_index.parquet").sort_values("row").corpus.values
    # 2026-11-23: a 16-row `map` micro-batch at 1,188 tokens holds 16 x 1,188 x 161,665 fp32 logits (12.3 GB), beyond a
    # 12 GB card, so the rows go through the model `chunk` at a time (every sequence is full length: no padding moves)
    st = Path(a.checkpoint) / "trainer_state.json"
    logged = {}
    if st.is_file():
        hist = json.loads(st.read_text())["log_history"]
        last = max(int(h["step"]) for h in hist if "eval_map_loss_MAP" in h)
        for h in hist:
            if int(h["step"]) == last:
                logged.update({k: h[k] for k in ("eval_map_loss_MAP", "eval_map_s2_loss") if k in h})
        logged["step"] = last
    out, arrays = {}, {}
    t0 = time.time()
    for name in ("map", "map_s2"):
        ds = ev[name]
        rows, sums, ns, sums_all, ns_all = [], [], [], [], []
        n_mb = len(ds) if a.limit is None else min(len(ds), a.limit)
        with torch.no_grad():
            for m in range(n_mb):
                b = ds[m]
                for i in range(0, b["input_ids"].shape[0], a.chunk):
                    sl = slice(i, i + a.chunk)
                    inp = {k: b[k][sl].to(dev) for k in ("input_ids", "attention_mask", "position_ids")}
                    with torch.autocast(dev, dtype=dtype or torch.float16, enabled=dtype is not None):
                        lg = model(**inp).logits
                    lg = lg[:, :-1].float()
                    lab = b["labels"][sl, 1:].to(dev)
                    mod = b["modality_ids"][sl, 1:].to(dev)
                    lab_ok = lab.ne(-100)
                    keep = lab_ok & mod.eq(MV.MAP_MODALITY_ID)
                    ce = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab.clamp_min(0), reduction="none")
                    del lg
                    sums.append((ce * keep).sum(1).double().cpu())
                    ns.append(keep.sum(1).cpu())
                    sums_all.append((ce * lab_ok).sum(1).double().cpu())
                    ns_all.append(lab_ok.sum(1).cpu())
                rows.append(torch.as_tensor(ds.rows[m * ds.per:(m + 1) * ds.per]))
                if m % 50 == 0:
                    print(f"  {name} {m + 1}/{n_mb} micro-batches, {time.time() - t0:.0f} s", flush=True)
        s, n, r = torch.cat(sums).numpy(), torch.cat(ns).numpy(), torch.cat(rows).numpy()
        sa, na = torch.cat(sums_all).numpy(), torch.cat(ns_all).numpy()
        res = {"pooled": float(s.sum() / n.sum()), "pooled_all_labels": float(sa.sum() / na.sum()), "rows": int(len(r)),
               "map_tokens_per_row": sorted(set(n.tolist()))}
        for c in ("majortom", "ssl4eos12"):
            sel = corp[r] == c
            res[c] = {"ce": float(s[sel].sum() / n[sel].sum()), "rows": int(sel.sum())}
        out[name] = res
        arrays.update({f"{name}_rows": r, f"{name}_ce_sum": s, f"{name}_n": n})
        print(name, json.dumps(res), flush=True)
    tag = a.tag or Path(a.checkpoint).parent.name + "_" + Path(a.checkpoint).name
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"corpus_{tag}.npz", **arrays)
    write(f"corpus_{tag}",
          {"checkpoint": a.checkpoint, "precision": a.precision, "chunk": a.chunk, "limit": a.limit,
           "device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
           "map_code_ce": out, "logged_in_training": logged, "seconds": round(time.time() - t0, 1),
           "written": time.strftime("%Y-%m-%d %H:%M:%S")})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--run", action="append", required=True)
    c.add_argument("--tag", default=None)
    f = sub.add_parser("forgetting")
    f.add_argument("--before", required=True)
    f.add_argument("--after", required=True)
    f.add_argument("--tag", default=None)
    k = sub.add_parser("corpus")
    k.add_argument("--checkpoint", required=True)
    k.add_argument("--config", default="eo_seg/r1_aligned")
    k.add_argument("--val-maps", default=None)
    k.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="fp32")
    k.add_argument("--chunk", type=int, default=2, help="rows per forward pass")
    k.add_argument("--limit", type=int, default=None, help="micro-batches per set (a smoke)")
    k.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    k.add_argument("--tag", default=None)
    a = ap.parse_args()
    {"curves": curves, "forgetting": forgetting, "corpus": corpus}[a.what](a)


if __name__ == "__main__":
    main()
