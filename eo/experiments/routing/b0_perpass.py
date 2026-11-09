"""B0 Step 6, the per-pass test: what a router that decides after each pass could know, on A10cR.

Enric's question (2026-11-08): today's router decides a token's depth once, from x0, the state after layer 0.
A router that decides after each pass would also read the state after pass 1. Does that state say more about
how much a token gains from extra passes? Rules: docs/plans/b0_router_audit.md, Step 6, "Rules, fixed before
the capture runs (2026-11-09, 13:25)".

  capture CK [--smoke N]   GPU (the TITAN V, fp16). Four main passes of A10cR at checkpoint-CK with every
               position forced to depth 1 and 3, under gate seeds 0 and 1, each exactly as eval_checkpoint.py's
               main pass. Hooks keep, for 256 rows (one per spatial group), x0 (the MoR layer's input) and h1
               (the shared block's output at pass 1). They check both bit for bit across the four passes, and the
               state after pass 2 against h1 (the control). Writes /data/enric/reports/b0/perpass/checkpoint-CK/
               {ce.npz, x0.npy, h1.npy, capture.json}. --smoke N: the first N batches, 8 rows, into --out.
  selftest     CPU. The cross-fitted ridge on synthetic data: a known R² is recovered, and noise gives none.
  fit          CPU. Ridge fits cross-fitted over the rows' groups, within each modality; the reliability;
               the rule; the shuffled and planted controls; the description. Writes read.{txt,json}.
  kl CK, klfit The decomposition added 14:10-14:13, description read by no rule: how much of a token's gain any router
               could know before seeing the next token. `kl` (GPU) computes KL(p3 || p1) and the gain's variance
               over the next token under p3, from depth-1 and depth-3 forwards on the stored passes' own gate draws
               (checked against their CE). `klfit` (CPU) writes read_kl.{txt,json}.
  value        CPU, description added at 14:34, read by no rule: what each score is worth in nats at equal compute,
               half the image tokens at depth 3 by score and the rest at depth 1, against a random half and
               uniform depth 2. Writes read_value.{txt,json}.
  value3       CPU, description added at 14:44: small moves off uniform depth 2, f of each modality's tokens to depth 1
               and f to depth 3 by predicted per-pass gains, against uniform 2. Writes read_value3.{txt,json}.

Under every-position forcing, every token takes pass 1, and a token's gate scales only its state at its exit
(model/mor_model/token_choice_router.py:327: final = x0 + w * h_d). So x0 and h1 depend on neither the depth nor
the gate draw, and only CE does. Runs in .venv. ⚠ No vocabulary-sized topk or sort: the capture runs on the
TITAN V.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eo.routing import stats as S  # noqa: E402

RUN = Path("/data/enric/runs/pretrain/phase3/mor10cr_20261012_095834")
CFG = "eo_terramesh/arm_a10cr_random"
OUT = Path("/data/enric/reports/b0/perpass")
STORED = Path("/data/enric/reports/b0/landscape")          # a10cr_tvforce{1,3}: Step 6's TITAN V passes
CKS = (1980, 3300)
PASSES = ((1, 0), (3, 0), (1, 1), (3, 1))                    # (depth, gate seed); the first is the reference
SEED_ROWS, SEED_FOLDS, SEED_CTRL = 20261109, 20261110, 20261111
N_FOLDS = 8
LAMS = 10.0 ** np.arange(-4, 3)                               # ridge penalty per token, on standardised features
N_BOOT = 10_000


def pick_rows(groups: np.ndarray, seed: int = SEED_ROWS) -> np.ndarray:
    """One row index from each spatial group, sorted."""
    rng = np.random.default_rng(seed)
    return np.sort(np.array([rng.choice(np.flatnonzero(groups == g)) for g in range(int(groups.max()) + 1)]))


# --------------------------------------------------------------------------------------------- capture (GPU)

def capture(ck: int, smoke: int = 0, out: Path = OUT) -> int:
    import torch
    from eo.eval import teacher_forced as TF
    from eo.generate import conditional as G
    from eo.routing.telemetry import find_mor_layers, has_learned_router

    d = out / f"checkpoint-{ck}"
    if (d / "capture.json").exists() and not smoke:
        print(f"[skip] {d} is done")
        return 0
    d.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    gpu = torch.cuda.get_device_name(0)
    torch.manual_seed(42)                                     # as eval_checkpoint.py, before the build
    model, cfg = G.build_model(CFG, str(RUN / f"checkpoint-{ck}"))
    if has_learned_router(model):
        raise SystemExit(f"{CFG} has a learned router; the per-pass test is A10cR's")
    if int(cfg.dataloader_num_workers) != 4:
        raise SystemExit("dataloader_num_workers != 4: modality orders would differ from the stored passes")
    ds, loader = TF.eval_loader(cfg, num_workers=4)
    rows = np.asarray(ds.rows)
    stored = {dd: STORED / f"a10cr_tvforce{dd}" / f"checkpoint-{ck}" for dd in (1, 3)}
    if not np.array_equal(rows, np.load(stored[1] / "eval_rows.npy")):
        raise SystemExit("the held-out rows differ from Step 6's stored passes")
    groups = S.eval_groups(rows)
    if smoke:
        sel = np.arange(8)
        n_rows = smoke * loader.batch_size
        batches = lambda: itertools.islice(loader, smoke)      # noqa: E731
    else:
        sel = pick_rows(groups)
        n_rows = len(ds)
        batches = lambda: loader                               # noqa: E731
    want = np.zeros(len(ds), bool)
    want[sel] = True

    layer = find_mor_layers(model)[0]
    lasts = {id(b[-1]): b[-1] for b in layer.block_list}     # one module if the blocks share layer objects
    st = {"row0": 0, "k": 0, "x0": [], "h1": [], "h2": []}

    def pre(_m, _args):
        st["k"] = 0                                           # pass counter, per MoR forward

    def on_last(_m, _args, out_):
        st["k"] += 1
        if st["k"] > 2:
            return
        h = out_[0]
        idx = np.flatnonzero(want[st["row0"]:st["row0"] + h.shape[0]])
        if idx.size:                                          # at passes 1 and 2 every position is active, in order
            st["h1" if st["k"] == 1 else "h2"].append(h[torch.as_tensor(idx, device=h.device)].detach().float().cpu())

    def on_mor(_m, args, _out):
        x = args[0]
        idx = np.flatnonzero(want[st["row0"]:st["row0"] + x.shape[0]])
        if idx.size:
            st["x0"].append(x[torch.as_tensor(idx, device=x.device)].detach().float().cpu())
        st["row0"] += x.shape[0]

    handles = [layer.register_forward_pre_hook(pre), layer.register_forward_hook(on_mor)]
    handles += [m.register_forward_hook(on_last) for m in lasts.values()]
    doc = {"checkpoint": ck, "gpu": gpu, "arm_config": CFG, "run": str(RUN), "n_rows": n_rows,
           "sel": sel.tolist(), "groups_sel": groups[sel].tolist(), "n_groups": int(groups.max()) + 1,
           "shared_last_layer_objects": len(lasts), "passes": [], "checks": {}}
    ref, ces, keep = None, {}, {}
    try:
        for depth, seed in PASSES:
            st.update(row0=0, x0=[], h1=[], h2=[])
            rf = TF.install_random_depth(model, depth)
            ov = TF.install_override(model, "router")
            if ov is not None:
                raise SystemExit("a random router has no override")
            rf.reseed(seed)
            t0 = time.perf_counter()
            res = TF.run_pass(model, batches(), override=None, dtype=torch.float16)
            tok = res.pop("tokens")
            if not (tok["depth"] == depth).all():
                raise SystemExit(f"forced depth {depth}, captured {np.unique(tok['depth'], return_counts=True)}")
            if st["row0"] != n_rows:
                raise SystemExit(f"the hooks saw {st['row0']} rows of {n_rows}")
            name = f"d{depth}_s{seed}"
            ces[name] = tok["ce"]
            if depth == 1 and seed == 0:
                keep = {k: tok[k] for k in ("label_mod", "input_mod", "label")}
            x0, h1 = torch.cat(st["x0"]), torch.cat(st["h1"])
            if len(x0) != len(sel) or len(h1) != len(sel):
                raise SystemExit(f"captured {len(x0)}, {len(h1)} rows of {len(sel)}")
            p = {"depth": depth, "gate_seed": seed, "seconds": round(time.perf_counter() - t0, 1),
                 "ce_all_body_tokens": res["ce_all_body_tokens"]}
            if ref is None:
                ref = (x0, h1)
                p["x0_vs_h1_max_abs"] = float((h1 - x0).abs().max())
            else:
                p["x0_equal_ref"] = bool(torch.equal(x0, ref[0]))
                p["h1_equal_ref"] = bool(torch.equal(h1, ref[1]))
                p["x0_max_abs_ref"] = float((x0 - ref[0]).abs().max())
                p["h1_max_abs_ref"] = float((h1 - ref[1]).abs().max())
            if st["h2"]:                                      # the control: the state after pass 2 is not h1
                h2 = torch.cat(st["h2"])
                p["h2_equal_h1"] = bool(torch.equal(h2, h1))
                p["h2_vs_h1_max_abs"] = float((h2 - h1).abs().max())
                del h2
            doc["passes"].append(p)
            print(f"{ck} {name}: {p['seconds']}s CE(all) {p['ce_all_body_tokens']:.6f} "
                  + " ".join(f"{k}={v}" for k, v in p.items() if k.startswith(("x0_", "h1_", "h2_"))), flush=True)
    finally:
        for h in handles:
            h.remove()

    # the checks, each with its control
    c = doc["checks"]
    for dd in (1, 3):
        s = np.load(stored[dd] / "tokens_router.npz")
        c[f"ce_d{dd}_s0_equals_stored"] = bool(np.array_equal(ces[f"d{dd}_s0"], s["ce"][:n_rows]))
        c[f"ce_d{dd}_s0_max_abs_stored"] = float(np.abs(ces[f"d{dd}_s0"].astype(np.float64)
                                                       - s["ce"][:n_rows].astype(np.float64)).max())
    s3 = np.load(stored[3] / "tokens_router.npz")["ce"][:n_rows]
    c["control_ce_d1_equals_stored_d3"] = bool(np.array_equal(ces["d1_s0"], s3))
    c["states_invariant"] = all(p.get("x0_equal_ref", True) and p.get("h1_equal_ref", True) for p in doc["passes"])
    c["control_h2_differs"] = any(p.get("h2_equal_h1") is False for p in doc["passes"])
    c["control_h1_differs_x0"] = doc["passes"][0]["x0_vs_h1_max_abs"] > 0
    ok = (c["ce_d1_s0_equals_stored"] and c["ce_d3_s0_equals_stored"] and not c["control_ce_d1_equals_stored_d3"]
          and c["states_invariant"] and c["control_h2_differs"] and c["control_h1_differs_x0"])
    c["all_pass"] = bool(ok)
    np.savez_compressed(d / "ce.npz", **ces, **keep)
    np.save(d / "x0.npy", ref[0].numpy())
    np.save(d / "h1.npy", ref[1].numpy())
    (d / "capture.json").write_text(json.dumps(doc, indent=1))
    print(json.dumps(c, indent=1))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------- kl (GPU)

def kl_capture(ck: int, smoke: int = 0, out: Path = OUT) -> int:
    """The decomposition added 2026-11-09, between 14:10 and 14:13, after the fit's first lines (description, read by no rule).

    Per position of the capture's rows: KL(p3 || p1), the gain expected under the depth-3 belief; the variance of
    the gain over the next token under that belief; and the realised CE at both depths, which must equal the
    stored seed-0 passes after fp16 rounding (the control: against the seed-1 passes it must not). A depth-1 and a
    depth-3 forward run on every batch holding a chosen row, each depth with its own seed-0 gate stream. On the
    other batches each stream is advanced by the same draw a forward makes, so the gates are the stored passes'."""
    import torch
    from eo.eval import teacher_forced as TF
    from eo.generate import conditional as G
    from eo.routing.telemetry import find_mor_layers

    d = out / f"checkpoint-{ck}"
    if (d / "kl.npz").exists() and not smoke:
        print(f"[skip] {d / 'kl.npz'} exists")
        return 0
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    cap = json.loads((d / "capture.json").read_text())
    sel = np.asarray(cap["sel"])
    stored = np.load(d / "ce.npz")
    torch.manual_seed(42)
    model, cfg = G.build_model(CFG, str(RUN / f"checkpoint-{ck}"))
    ds, loader = TF.eval_loader(cfg, num_workers=4)
    want = np.zeros(len(ds), bool)
    want[sel] = True
    layer = find_mor_layers(model)[0]
    dev = torch.device("cuda", torch.cuda.current_device())
    rfs = {}
    for dd in (1, 3):
        rfs[dd] = TF.install_random_depth(model, dd)
        rfs[dd].reseed(0)
        rfs[dd].gen = torch.Generator(device=dev).manual_seed(0)    # as its first call would make it
    gens = {dd: rfs[dd].gen for dd in (1, 3)}
    model.eval()
    acc = {k: [] for k in ("kl", "v", "ce1", "ce3")}
    row0, n_fwd = 0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in (itertools.islice(loader, smoke) if smoke else loader):
            batch = dict(batch)
            mids = batch.pop("modality_ids")
            labels = batch.pop("labels")
            B, L = batch["input_ids"].shape
            idx = np.flatnonzero(want[row0:row0 + B])
            row0 += B
            if idx.size == 0:
                for dd in (1, 3):
                    torch.rand(B, L, rfs[dd].n_trained, generator=rfs[dd].gen, device=dev)
                continue
            inputs = {k: v.to(dev) for k, v in batch.items()}
            ix = torch.as_tensor(idx, device=dev)
            lp, ce_rp = {}, {}
            lab = labels[idx, 1:].to(dev).clamp_min(0)
            for dd in (1, 3):
                layer.rand_override = rfs[dd]
                rfs[dd].mod_ids = mids.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**inputs).logits
                lg = logits[ix, :-1].float()
                # CE exactly as run_pass computes it (layout included), for the check against the stored passes
                ce_rp[dd] = torch.nn.functional.cross_entropy(lg.transpose(1, 2), lab, reduction="none")
                lp[dd] = torch.log_softmax(lg, -1)
                del logits, lg
                if rfs[dd].gen is not gens[dd]:
                    raise SystemExit("the gate stream was rebuilt: the draws would not be the stored passes'")
            diff = lp[3] - lp[1]                                  # the gain for every possible next token
            p3 = lp[3].exp()
            kl = (p3 * diff).sum(-1)
            acc["kl"].append(kl.cpu())
            acc["v"].append(((p3 * diff.square()).sum(-1) - kl.square()).clamp_min(0).cpu())
            acc["ce1"].append(ce_rp[1].cpu())
            acc["ce3"].append(ce_rp[3].cpu())
            n_fwd += 1
            del lp, diff, p3, kl
    res = {k: torch.cat(v).numpy() for k, v in acc.items()}
    if len(res["kl"]) != len(sel):
        raise SystemExit(f"{len(res['kl'])} rows of {len(sel)}")
    lmod = stored["label_mod"][sel]
    valid = lmod > 0
    checks = {}
    for dd in (1, 3):
        mine = res[f"ce{dd}"].astype(np.float16)
        checks[f"ce{dd}_equal_seed0_fp16"] = float((mine[valid] == stored[f"d{dd}_s0"][sel][valid]).mean())
        checks[f"ce{dd}_equal_seed1_fp16_control"] = float((mine[valid] == stored[f"d{dd}_s1"][sel][valid]).mean())
    checks["max_abs_gain_vs_stored"] = float(np.abs((res["ce1"] - res["ce3"])[valid]
                                                    - (stored["d1_s0"][sel].astype(np.float64)
                                                       - stored["d3_s0"][sel].astype(np.float64))[valid]).max())
    ok = (min(checks["ce1_equal_seed0_fp16"], checks["ce3_equal_seed0_fp16"]) >= 0.999
          and max(checks["ce1_equal_seed1_fp16_control"], checks["ce3_equal_seed1_fp16_control"]) < 0.9)
    checks["all_pass"] = bool(ok)
    checks["batches_forwarded"] = n_fwd
    checks["seconds"] = round(time.perf_counter() - t0, 1)
    np.savez_compressed(d / "kl.npz", **res)
    (d / "kl.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    return 0 if ok else 1


def klfit() -> int:
    """How much of the gain could any router know: the decomposition's readings (description, read by no rule)."""
    from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITY_TO_ID
    report, lines = {}, []
    P = lines.append
    for ck in CKS:
        d = OUT / f"checkpoint-{ck}"
        chk = json.loads((d / "kl.json").read_text())
        if not chk["all_pass"]:
            raise SystemExit(f"{ck}: the decomposition's checks failed; nothing is read")
        sel = np.asarray(json.loads((d / "capture.json").read_text())["sel"])
        ce = np.load(d / "ce.npz")
        kz = np.load(d / "kl.npz")
        lmod = ce["label_mod"][sel]
        g0 = ce["d1_s0"][sel].astype(np.float64) - ce["d3_s0"][sel].astype(np.float64)
        g1 = ce["d1_s1"][sel].astype(np.float64) - ce["d3_s1"][sel].astype(np.float64)
        x0 = np.load(d / "x0.npy", mmap_mode="r")
        h1 = np.load(d / "h1.npy", mmap_mode="r")
        fold_of_row = np.random.default_rng(SEED_FOLDS).permutation(len(sel)) % N_FOLDS
        per = {}
        for m in list(IMAGE_MODALITIES) + ["Coords"]:
            r_, t_ = _rows_by_modality(lmod, MODALITY_TO_ID[m])
            if len(r_) == 0:
                continue
            y, y1 = g0[r_, t_], g1[r_, t_]
            kl, v = kz["kl"][r_, t_].astype(np.float64), kz["v"][r_, t_].astype(np.float64)
            fold = fold_of_row[r_]
            res = {"n": len(y), "var_g": float(y.var()), "rv": float(np.mean((y - y.mean()) * (y1 - y1.mean()))),
                   "mean_g": float(y.mean()), "mean_kl": float(kl.mean()), "var_kl": float(kl.var()),
                   "mean_v": float(v.mean())}
            res["ceiling_calibrated"] = 1 - res["mean_v"] / res["var_g"]
            # the model's own expected gain as a predictor of the realised gain, cross-fitted (one feature)
            pr, _ = Ridge(kl[:, None], fold, N_FOLDS).predict(y[:, None])
            res["share_g_from_kl"] = float(1 - ((y - pr[:, 0]) ** 2).mean() / y.var())
            Xb = np.asarray(x0[r_, t_], dtype=np.float64)
            Xc = np.asarray(h1[r_, t_], dtype=np.float64)
            for name, X in (("b", Xb), ("c", Xc), ("d", np.concatenate([Xc, Xc - Xb], 1))):
                pr, _ = Ridge(X, fold, N_FOLDS).predict(kl[:, None])
                res[f"share_kl_from_{name}"] = float(1 - ((kl - pr[:, 0]) ** 2).mean() / kl.var())
            per[m] = res
            print(f"{ck} {m}: n {res['n']}, var g {res['var_g']:.4f}, E[v] {res['mean_v']:.4f}, ceiling "
                  f"{res['ceiling_calibrated']:+.3f}, g from KL {res['share_g_from_kl']:+.4f}, KL from b "
                  f"{res['share_kl_from_b']:+.3f} c {res['share_kl_from_c']:+.3f} d {res['share_kl_from_d']:+.3f}",
                  flush=True)
        img = [m for m in IMAGE_MODALITIES if m in per]
        def pool(key, var):                                  # explained variance summed over modalities, as a share
            return float(sum(per[m]["n"] * per[m][key] * per[m][var] for m in img)
                         / sum(per[m]["n"] * per[m][var] for m in img))
        pooled = {"ceiling_calibrated": pool("ceiling_calibrated", "var_g"),
                  "share_g_from_kl": pool("share_g_from_kl", "var_g"),
                  **{f"share_kl_from_{f}": pool(f"share_kl_from_{f}", "var_kl") for f in "bcd"}}
        report[ck] = {"checks": chk, "per_modality": per, "pooled": pooled}
        P(f"== checkpoint-{ck}: what any router could know (description; checks: realised CE equals the stored "
          f"seed-0 passes at {min(chk['ce1_equal_seed0_fp16'], chk['ce3_equal_seed0_fp16']):.4f} of positions "
          f"after fp16 rounding, seed 1 at {max(chk['ce1_equal_seed1_fp16_control'], chk['ce3_equal_seed1_fp16_control']):.3f})")
        P(f"   {'modality':8} {'Var g':>7} {'E[v]':>7} {'ceiling':>8} {'g~KL':>7} | KL from: {'(b) x0':>7} "
          f"{'(c) h1':>7} {'(d)':>7} | mean g {'mean KL':>8}")
        for m in per:
            q = per[m]
            P(f"   {m:8} {q['var_g']:7.4f} {q['mean_v']:7.4f} {q['ceiling_calibrated']:+8.3f} "
              f"{q['share_g_from_kl']:+7.4f} |          {q['share_kl_from_b']:+7.3f} {q['share_kl_from_c']:+7.3f} "
              f"{q['share_kl_from_d']:+7.3f} | {q['mean_g']:+.4f} {q['mean_kl']:8.4f}")
        P(f"   pooled image tokens (variance-weighted shares): ceiling {pooled['ceiling_calibrated']:+.3f}, g from KL "
          f"{pooled['share_g_from_kl']:+.4f}; KL from b {pooled['share_kl_from_b']:+.3f}, c "
          f"{pooled['share_kl_from_c']:+.3f}, d {pooled['share_kl_from_d']:+.3f}")
        P("")
    (OUT / "read_kl.json").write_text(json.dumps(report, indent=1, default=float))
    (OUT / "read_kl.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def value() -> int:
    """What a score is worth in nats at equal compute (description added 2026-11-09, 14:34, read by no rule).

    The top half of image tokens by a score goes to depth 3, the rest to depth 1: mean depth 2, A10cR's compute.
    Each token's CE is read additively from the forced passes (Step 6: good to about 0.001 nats). Baselines: a
    random half, whose expectation is the mean of the two depths, and uniform depth 2 on the same rows. Scores are
    cross-fitted on the fit's folds; KL itself and the realised gain are references no router can have."""
    from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITY_TO_ID
    ids = [MODALITY_TO_ID[m] for m in IMAGE_MODALITIES]
    report, lines = {}, []
    P = lines.append
    for ck in CKS:
        d = OUT / f"checkpoint-{ck}"
        sel = np.asarray(json.loads((d / "capture.json").read_text())["sel"])
        ce = np.load(d / "ce.npz")
        kz = np.load(d / "kl.npz")
        t2 = np.load(STORED / "a10cr_tvforce2" / f"checkpoint-{ck}" / "tokens_router.npz")
        lmod = ce["label_mod"][sel]
        if not np.array_equal(t2["label_mod"][sel], lmod):
            raise SystemExit("uniform depth 2's frame differs from the capture's")
        ce1, ce3 = (ce[k][sel].astype(np.float64) for k in ("d1_s0", "d3_s0"))
        ce2 = t2["ce"][sel].astype(np.float64)
        g0 = ce1 - ce3
        x0 = np.load(d / "x0.npy", mmap_mode="r")
        h1 = np.load(d / "h1.npy", mmap_mode="r")
        fold_of_row = np.random.default_rng(SEED_FOLDS).permutation(len(sel)) % N_FOLDS
        score = {k: np.zeros_like(g0) for k in ("g from x0", "g from h1", "KL from x0", "KL from h1")}
        for m, mid in zip(IMAGE_MODALITIES, ids):
            r_, t_ = _rows_by_modality(lmod, mid)
            if len(r_) == 0:
                continue
            Y = np.stack([g0[r_, t_], kz["kl"][r_, t_].astype(np.float64)], 1)
            for name, X in (("x0", x0), ("h1", h1)):
                pr, _ = Ridge(np.asarray(X[r_, t_], dtype=np.float64), fold_of_row[r_], N_FOLDS).predict(Y)
                score[f"g from {name}"][r_, t_] = pr[:, 0]
                score[f"KL from {name}"][r_, t_] = pr[:, 1]
            print(f"{ck} {m}: scores fitted", flush=True)
        score["KL (needs depth 3)"] = kz["kl"].astype(np.float64)
        score["realised gain (oracle)"] = g0
        img = np.isin(lmod, ids)
        n_r = img.sum(1).astype(np.float64)

        def top_half(s, mask):
            pick = np.zeros(mask.shape, bool)
            idx = np.flatnonzero(mask.ravel())
            order = idx[np.argsort(-s.ravel()[idx], kind="stable")]
            pick.ravel()[order[: len(idx) // 2]] = True
            return pick

        def per_row(x):
            return np.where(img, x, 0.0).sum(1)

        W = S.boot_weights(len(sel), N_BOOT, 20261109)
        rand_r, unif_r = per_row((ce1 + ce3) / 2), per_row(ce2)
        out = {"n_tokens": int(n_r.sum()), "random_half_minus_uniform2": float((rand_r.sum() - unif_r.sum()) / n_r.sum())}
        alloc = {}
        for name, s in score.items():
            for scope in ("within", "pooled"):
                if scope == "pooled":
                    pick = top_half(s, img)
                else:
                    pick = np.zeros(img.shape, bool)
                    for mid in ids:
                        pick |= top_half(s, lmod == mid)
                a_r = per_row(np.where(pick, ce3, ce1))
                alloc[(name, scope)] = a_r
                out[f"{name} | {scope}"] = {"vs_random": float((rand_r.sum() - a_r.sum()) / n_r.sum()),
                                            "vs_uniform2": float((unif_r.sum() - a_r.sum()) / n_r.sum())}
        for tgt in ("g", "KL"):
            for scope in ("within", "pooled"):
                diff = alloc[(f"{tgt} from x0", scope)] - alloc[(f"{tgt} from h1", scope)]   # h1's saving over x0's
                bt = (W @ diff) / (W @ n_r)
                out[f"h1 minus x0, {tgt} score | {scope}"] = {
                    "point": float(diff.sum() / n_r.sum()),
                    "ci": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))]}
        report[ck] = out
        P(f"== checkpoint-{ck}: half the image tokens at depth 3 by score, the rest at depth 1 (mean depth 2); "
          f"saving in nats per image token, positive = better; {out['n_tokens']} tokens")
        P(f"   random half minus uniform depth 2: {out['random_half_minus_uniform2']:+.5f}")
        P(f"   {'score':26} {'within: vs random':>18} {'vs uniform 2':>13} | {'pooled: vs random':>18} {'vs uniform 2':>13}")
        for name in score:
            w_, p_ = out[f"{name} | within"], out[f"{name} | pooled"]
            P(f"   {name:26} {w_['vs_random']:+18.5f} {w_['vs_uniform2']:+13.5f} | {p_['vs_random']:+18.5f} "
              f"{p_['vs_uniform2']:+13.5f}")
        for tgt in ("g", "KL"):
            for scope in ("within", "pooled"):
                q = out[f"h1 minus x0, {tgt} score | {scope}"]
                P(f"   h1 minus x0, {tgt} score, {scope}: {q['point']:+.5f} [{q['ci'][0]:+.5f}, {q['ci'][1]:+.5f}]")
        P("")
    (OUT / "read_value.json").write_text(json.dumps(report, indent=1, default=float))
    (OUT / "read_value.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def value3() -> int:
    """Small moves off uniform depth 2 (description added 2026-11-09, 14:44, read by no rule).

    Within each modality, the fraction f of tokens with the lowest predicted CE(1) - CE(2) stop at depth 1, the same
    number with the highest predicted CE(2) - CE(3) go to depth 3, the rest stay at 2: mean depth stays 2. Scores
    are cross-fitted from x0 and from h1 (h1 stands in for both decisions, which understates the second); the
    realised gains are the oracle. CE is read additively from the forced passes, CE(2) from a10cr_tvforce2."""
    from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITY_TO_ID
    ids = [MODALITY_TO_ID[m] for m in IMAGE_MODALITIES]
    FR = (0.02, 0.05, 0.10, 0.20, 1 / 3)
    report, lines = {}, []
    P = lines.append
    for ck in CKS:
        d = OUT / f"checkpoint-{ck}"
        sel = np.asarray(json.loads((d / "capture.json").read_text())["sel"])
        ce = np.load(d / "ce.npz")
        t2 = np.load(STORED / "a10cr_tvforce2" / f"checkpoint-{ck}" / "tokens_router.npz")
        lmod = ce["label_mod"][sel]
        if not np.array_equal(t2["label_mod"][sel], lmod):
            raise SystemExit("uniform depth 2's frame differs from the capture's")
        ce1, ce3 = (ce[k][sel].astype(np.float64) for k in ("d1_s0", "d3_s0"))
        ce2 = t2["ce"][sel].astype(np.float64)
        a, b = ce1 - ce2, ce2 - ce3                      # what pass 2 buys, and what pass 3 buys
        x0 = np.load(d / "x0.npy", mmap_mode="r")
        h1 = np.load(d / "h1.npy", mmap_mode="r")
        fold_of_row = np.random.default_rng(SEED_FOLDS).permutation(len(sel)) % N_FOLDS
        pa = {k: np.zeros_like(a) for k in ("x0", "h1")}
        pb = {k: np.zeros_like(a) for k in ("x0", "h1")}
        for m, mid in zip(IMAGE_MODALITIES, ids):
            r_, t_ = _rows_by_modality(lmod, mid)
            if len(r_) == 0:
                continue
            Y = np.stack([a[r_, t_], b[r_, t_]], 1)
            for name, X in (("x0", x0), ("h1", h1)):
                pr, _ = Ridge(np.asarray(X[r_, t_], dtype=np.float64), fold_of_row[r_], N_FOLDS).predict(Y)
                pa[name][r_, t_], pb[name][r_, t_] = pr[:, 0], pr[:, 1]
            print(f"{ck} {m}: scores fitted", flush=True)
        pa["oracle"], pb["oracle"] = a, b
        img = np.isin(lmod, ids)
        n_r = img.sum(1).astype(np.float64)
        unif_r = np.where(img, ce2, 0.0).sum(1)
        W = S.boot_weights(len(sel), N_BOOT, 20261109)
        out = {"mean_pass2_gain": float(a[img].mean()), "mean_pass3_gain": float(b[img].mean())}
        for f in FR:
            res, alloc = {}, {}
            for name in ("x0", "h1", "oracle"):
                depth = np.full(lmod.shape, 2, dtype=np.int8)
                for mid in ids:
                    idx = np.flatnonzero((lmod == mid).ravel())
                    k = int(round(f * len(idx)))
                    if k == 0:
                        continue
                    to1 = idx[np.argsort(pa[name].ravel()[idx], kind="stable")][:k]
                    rest = np.setdiff1d(idx, to1, assume_unique=True)
                    to3 = rest[np.argsort(-pb[name].ravel()[rest], kind="stable")][:k]
                    depth.ravel()[to1] = 1
                    depth.ravel()[to3] = 3
                if abs(float(depth[img].mean()) - 2.0) > 1e-3:
                    raise SystemExit(f"mean depth {depth[img].mean()} != 2")
                a_r = np.where(img, np.where(depth == 1, ce1, np.where(depth == 3, ce3, ce2)), 0.0).sum(1)
                alloc[name] = a_r
                bt = (W @ (unif_r - a_r)) / (W @ n_r)
                res[name] = {"vs_uniform2": float((unif_r - a_r).sum() / n_r.sum()),
                             "ci": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))]}
            dd = alloc["x0"] - alloc["h1"]
            bt = (W @ dd) / (W @ n_r)
            res["h1_minus_x0"] = {"point": float(dd.sum() / n_r.sum()),
                                  "ci": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))]}
            out[f"{f:.3f}"] = res
        report[ck] = out
        P(f"== checkpoint-{ck}: within each modality, f of tokens stop at depth 1 and f go to depth 3 by score, the "
          f"rest stay at 2 (mean depth 2); saving against uniform depth 2, nats per image token, positive = better")
        P(f"   mean CE(1) - CE(2) {out['mean_pass2_gain']:+.5f}, mean CE(2) - CE(3) {out['mean_pass3_gain']:+.5f}")
        P(f"   {'f':>6} | {'x0':>27} | {'h1':>27} | {'h1 - x0':>27} | {'oracle':>9}")
        for f in FR:
            r = out[f"{f:.3f}"]
            cell = lambda q: f"{q['vs_uniform2'] if 'vs_uniform2' in q else q['point']:+.5f} [{q['ci'][0]:+.5f}, {q['ci'][1]:+.5f}]"  # noqa: E731
            P(f"   {f:6.3f} | {cell(r['x0']):>27} | {cell(r['h1']):>27} | {cell(r['h1_minus_x0']):>27} | "
              f"{r['oracle']['vs_uniform2']:+9.5f}")
        P("")
    (OUT / "read_value3.json").write_text(json.dumps(report, indent=1, default=float))
    (OUT / "read_value3.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------------------------- ridge (CPU)

class Ridge:
    """Ridge on standardised features, cross-fitted over folds, lambda by leaving one training fold out.

    Sufficient statistics per fold, so every training set's moments are sums. The eigendecomposition depends on
    X and the folds only, and is shared by every target y."""

    def __init__(self, X: np.ndarray, fold: np.ndarray, K: int):
        self.X, self.fold, self.K = X, fold, K
        self.ix = [np.flatnonzero(fold == k) for k in range(K)]
        self.n = np.array([len(i) for i in self.ix], dtype=np.float64)
        self.sx = np.stack([X[i].sum(0) for i in self.ix])
        self.sxx = np.stack([X[i].T @ X[i] for i in self.ix])

    def _train(self, ks):
        n, sx, sxx = self.n[ks].sum(), self.sx[ks].sum(0), self.sxx[ks].sum(0)
        mu = sx / n
        C = sxx / n - np.outer(mu, mu)
        sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
        ev, V = np.linalg.eigh(C / np.outer(sd, sd))
        return n, mu, sd, np.clip(ev, 0.0, None), V

    def _betas(self, fit, sy, sxy):
        n, mu, sd, ev, V = fit
        my = sy / n
        cz = (sxy / n - mu * my) / sd
        proj = V.T @ cz
        return my, V @ (proj[:, None] / (ev[:, None] + LAMS[None, :]))      # (p, n_lams)

    def predict(self, Y: np.ndarray):
        """Y (n, T) targets -> cross-fitted predictions (n, T) and the lambda index chosen per (outer fold, target)."""
        K = self.K
        sy = np.stack([Y[i].sum(0) for i in self.ix])                       # (K, T)
        sxy = np.stack([self.X[i].T @ Y[i] for i in self.ix])              # (K, p, T)
        pred = np.zeros_like(Y)
        chosen = np.zeros((K, Y.shape[1]), dtype=int)
        for k in range(K):
            train = [j for j in range(K) if j != k]
            err = np.zeros((len(LAMS), Y.shape[1]))
            for j in train:
                inner = [i for i in train if i != j]
                fit = self._train(inner)
                Z = (self.X[self.ix[j]] - fit[1]) / fit[2]
                for t in range(Y.shape[1]):
                    my, B = self._betas(fit, sy[inner, t].sum(), sxy[inner, :, t].sum(0))
                    err[:, t] += ((Y[self.ix[j], t][:, None] - my - Z @ B) ** 2).sum(0)
            fit = self._train(train)
            Z = (self.X[self.ix[k]] - fit[1]) / fit[2]
            for t in range(Y.shape[1]):
                li = int(np.argmin(err[:, t]))
                chosen[k, t] = li
                my, B = self._betas(fit, sy[train, t].sum(), sxy[train, :, t].sum(0))
                pred[self.ix[k], t] = my + Z @ B[:, li]
        return pred, chosen


def selftest() -> int:
    """A known out-of-sample R² is recovered on grouped data, and pure noise explains nothing."""
    rng = np.random.default_rng(0)
    G_, per, p = 256, 120, 64
    grp = np.repeat(np.arange(G_), per)
    X = rng.normal(size=(G_ * per, p)) + rng.normal(size=(G_, p))[grp] * 0.5   # group-correlated features
    beta = rng.normal(size=p)
    sig = X @ beta
    sig = (sig - sig.mean()) / sig.std()
    noise = rng.normal(size=len(sig))
    y_true = np.sqrt(0.10) * sig + np.sqrt(0.90) * noise                       # R² 0.10 by construction
    folds = rng.permutation(G_) % N_FOLDS
    r = Ridge(X, folds[grp], N_FOLDS)
    pred, chosen = r.predict(np.stack([y_true, noise], 1))
    ev = [1 - ((y - q) ** 2).mean() / y.var() for y, q in zip((y_true, noise), pred.T)]
    print(f"selftest: R² planted 0.100 -> cross-fitted {ev[0]:.4f}; noise -> {ev[1]:+.5f}; "
          f"lambda chosen {LAMS[np.bincount(chosen[:, 0]).argmax()]:g}")
    ok = 0.085 < ev[0] < 0.105 and ev[1] < 0.002
    print("selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------- fit (CPU)

def _rows_by_modality(lmod: np.ndarray, mid: int):
    """(row, position) of every position predicting modality `mid` (label_mod frame), in row-major order."""
    return np.nonzero(lmod == mid)


def fit() -> int:
    from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITY_TO_ID
    mods = list(IMAGE_MODALITIES) + ["Coords"]
    report, lines = {}, []
    P = lines.append
    for ck in CKS:
        d = OUT / f"checkpoint-{ck}"
        cap = json.loads((d / "capture.json").read_text())
        if not cap["checks"]["all_pass"]:
            raise SystemExit(f"{ck}: the capture's checks failed; nothing is read")
        sel = np.asarray(cap["sel"])
        ce = np.load(d / "ce.npz")
        lmod = ce["label_mod"][sel]
        g0 = ce["d1_s0"][sel].astype(np.float64) - ce["d3_s0"][sel].astype(np.float64)
        g1 = ce["d1_s1"][sel].astype(np.float64) - ce["d3_s1"][sel].astype(np.float64)
        # the same reliability over all 4,416 rows (description)
        lm_all = ce["label_mod"]
        G0 = ce["d1_s0"].astype(np.float64) - ce["d3_s0"].astype(np.float64)
        G1 = ce["d1_s1"].astype(np.float64) - ce["d3_s1"].astype(np.float64)
        x0 = np.load(d / "x0.npy", mmap_mode="r")
        h1 = np.load(d / "h1.npy", mmap_mode="r")
        n_rows = len(sel)
        fold_of_row = np.random.default_rng(SEED_FOLDS).permutation(n_rows) % N_FOLDS
        rng = np.random.default_rng(SEED_CTRL)
        per = {}
        rowsums = {}                                         # per modality: per-row sums for the bootstrap
        for m in mods:
            mid = MODALITY_TO_ID[m]
            r_, t_ = _rows_by_modality(lmod, mid)
            if len(r_) == 0:
                continue
            y0, y1 = g0[r_, t_], g1[r_, t_]
            Xb = np.asarray(x0[r_, t_], dtype=np.float64)
            Xc = np.asarray(h1[r_, t_], dtype=np.float64)
            Xd = np.concatenate([Xc, Xc - Xb], 1)
            n = len(y0)
            var0, var1 = y0.var(), y1.var()
            rv = float(np.mean((y0 - y0.mean()) * (y1 - y1.mean())))
            # controls and description targets
            perm = rng.permutation(n)
            y_shuf, y1_shuf = y0[perm], y1[perm]
            u = rng.normal(size=Xb.shape[1]) / np.sqrt(Xb.shape[1])
            z = (Xc - Xb) @ u
            A = np.concatenate([np.ones((n, 1)), Xb], 1)
            z = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]                 # regressed off x0
            plant_var = 0.2 * max(rv, 0.01 * var0)
            z = z / z.std() * np.sqrt(plant_var)
            y_plant = y0 + z
            counts = np.bincount(r_, minlength=n_rows)
            have = np.flatnonzero(counts)
            if len(set(counts[have])) == 1:                  # rows permuted, position within the block kept
                k = counts[have][0]
                blocks = y0.reshape(len(have), k)
                y_rowp = blocks[rng.permutation(len(have))].ravel()
            else:
                y_rowp = None
            Y = [y0, y_shuf, y_plant] + ([y_rowp] if y_rowp is not None else [])
            Y = np.stack(Y, 1)
            res = {"n": n, "var": float(var0), "var_seed1": float(var1), "rv": rv,
                   "r": rv / np.sqrt(var0 * var1) if var0 > 0 and var1 > 0 else float("nan"),
                   "plant_var": plant_var, "mean_gain": float(y0.mean()), "sd_gain": float(y0.std())}
            fold = fold_of_row[r_]
            preds = {}
            for name, X in (("b", Xb), ("c", Xc), ("d", Xd)):
                t0 = time.perf_counter()
                pr, ch = Ridge(X, fold, N_FOLDS).predict(Y)
                preds[name] = pr
                res[f"lambda_{name}"] = float(LAMS[np.bincount(ch[:, 0], minlength=len(LAMS)).argmax()])
                for ti, tname in enumerate(["real", "shuffled", "planted", "rowperm"][:Y.shape[1]]):
                    yy = Y[:, ti]
                    res[f"ev_{name}_{tname}"] = float(yy.var() - ((yy - pr[:, ti]) ** 2).mean())
                res[f"ev_{name}_seed1"] = float(var1 - ((y1 - pr[:, 0]) ** 2).mean())
                res[f"seconds_{name}"] = round(time.perf_counter() - t0, 1)
            # (a) the modality: its cross-fitted mean is a constant within the modality
            per[m] = res
            rowsums[m] = {
                "n": counts.astype(np.float64),
                "tot": np.bincount(r_, weights=(y0 - y0.mean()) ** 2, minlength=n_rows),
                "cov": np.bincount(r_, weights=(y0 - y0.mean()) * (y1 - y1.mean()), minlength=n_rows),
                **{f"err_{f}_{tn}": np.bincount(r_, weights=(Y[:, ti] - preds[f][:, ti]) ** 2, minlength=n_rows)
                   for f in "bcd" for ti, tn in enumerate(["real", "shuffled", "planted"])},
                **{f"tot_{tn}": np.bincount(r_, weights=(Y[:, ti] - Y[:, ti].mean()) ** 2, minlength=n_rows)
                   for ti, tn in enumerate(["real", "shuffled", "planted"])},
                "cov_planted": np.bincount(r_, weights=(y_plant - y_plant.mean()) * (y1 + z - (y1 + z).mean()),
                                           minlength=n_rows),
                "cov_shuffled": np.bincount(r_, weights=(y_shuf - y_shuf.mean()) * (y1_shuf - y1_shuf.mean()),
                                            minlength=n_rows),
            }
            print(f"{ck} {m}: n {n}, var {var0:.4f}, rv {rv:.4f} (r {res['r']:.3f}), "
                  f"EV b {res['ev_b_real']:+.5f} c {res['ev_c_real']:+.5f} d {res['ev_d_real']:+.5f}", flush=True)

        # (a): cross-fitted modality means, pooled over the image modalities
        img = [m for m in IMAGE_MODALITIES if m in per]
        ys, folds_ = [], []
        for m in img:
            r_, t_ = _rows_by_modality(lmod, MODALITY_TO_ID[m])
            ys.append((g0[r_, t_], fold_of_row[r_], np.full(len(r_), m)))
        yy = np.concatenate([a for a, _, _ in ys]); ff = np.concatenate([b for _, b, _ in ys])
        mm = np.concatenate([c for _, _, c in ys])
        pa = np.zeros_like(yy)
        for k in range(N_FOLDS):
            for m in img:
                tr, te = (ff != k) & (mm == m), (ff == k) & (mm == m)
                pa[te] = yy[tr].mean()
        ev_a = float(yy.var() - ((yy - pa) ** 2).mean())

        # pooled over the image modalities, with the rule and its bootstrap over the rows (= the groups)
        W = S.boot_weights(n_rows, N_BOOT, 20261109)
        def pooled(key):                                     # noqa: E306
            return sum(rowsums[m][key] for m in img)
        N_r = pooled("n")
        out = {"n_tokens": int(N_r.sum()), "ev_a_real": ev_a, "var_pooled_total": float(yy.var()),
               "share_a": ev_a / float(yy.var())}
        for tn in ("real", "shuffled", "planted"):
            tot = pooled(f"tot_{tn}" if tn != "real" else "tot")
            cov = pooled({"real": "cov", "shuffled": "cov_shuffled", "planted": "cov_planted"}[tn])
            ev = {f: (tot - pooled(f"err_{f}_{tn}")) for f in "bcd"}            # per-row explained sums
            point = {f: float(ev[f].sum() / N_r.sum()) for f in "bcd"}
            rv_ = float(cov.sum() / N_r.sum())
            best = "c" if point["c"] >= point["d"] else "d"
            boot = {f: (W @ ev[f]) / (W @ N_r) for f in "bcd"}
            rvb = (W @ cov) / (W @ N_r)
            d1 = point[best] - 2 * max(point["b"], 0.0)
            d2 = point[best] - 0.10 * rv_
            d1b = boot[best] - 2 * np.maximum(boot["b"], 0.0)
            d2b = boot[best] - 0.10 * rvb
            lo1, lo2 = float(np.quantile(d1b, 0.025)), float(np.quantile(d2b, 0.025))
            if d1 >= 0 and d2 >= 0 and lo1 > 0 and lo2 > 0:
                verdict = "worth building"
            elif d1 >= 0 and d2 >= 0:
                verdict = "suggestive"
            else:
                verdict = "not worth building"
            out[tn] = {"ev": point, "rv": rv_, "best": best, "D1": d1, "D2": d2, "D1_lo": lo1, "D2_lo": lo2,
                       "ev_ci": {f: [float(np.quantile(boot[f], q)) for q in (0.025, 0.975)] for f in "bcd"},
                       "ev_over_rv": {f: point[f] / rv_ if rv_ > 0 else float("nan") for f in "bcd"},
                       "verdict": verdict}
        # the controls
        R, Sh, Pl = out["real"], out["shuffled"], out["planted"]
        plant_tot = sum(per[m]["plant_var"] * per[m]["n"] for m in img) / sum(per[m]["n"] for m in img)
        out["control_shuffled_ok"] = bool(max(Sh["ev"].values()) < 0.10 * R["rv"] and Sh["verdict"] != "worth building")
        out["control_planted_ok"] = bool(Pl["ev"]["d"] - R["ev"]["d"] >= 0.5 * plant_tot
                                         and Pl["ev"]["b"] - R["ev"]["b"] < 0.25 * plant_tot)
        out["planted_var_pooled"] = plant_tot
        out["seed1"] = {f: float(sum(per[m][f"ev_{f}_seed1"] * per[m]["n"] for m in img)
                                 / sum(per[m]["n"] for m in img)) for f in "bcd"}
        if all("ev_b_rowperm" in per[m] for m in img):
            out["rowperm"] = {f: float(sum(per[m][f"ev_{f}_rowperm"] * per[m]["n"] for m in img)
                                       / sum(per[m]["n"] for m in img)) for f in "bcd"}
        # reliability over all 4,416 rows, per modality (description)
        out["r_all_rows"] = {}
        for m in mods:
            mk = lm_all == MODALITY_TO_ID[m]
            a, b = G0[mk], G1[mk]
            out["r_all_rows"][m] = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan")
        report[ck] = {"per_modality": per, "pooled": out}

        # text
        P(f"== A10cR checkpoint-{ck}: 256 rows, one per spatial group; gain = CE(d1) - CE(d3), every position forced")
        P(f"   capture: {cap['gpu']}; seed-0 CE equals a10cr_tvforce{{1,3}} bit for bit; x0 and h1 identical "
          f"across the four passes; the state after pass 2 differs from h1")
        P(f"   {'modality':8} {'n':>6} {'mean g':>8} {'SD g':>7} {'RV':>8} {'r':>6} | EV/RV:  {'(b) x0':>8} "
          f"{'(c) h1':>8} {'(d) h1,dh':>9} | lambda b c d")
        for m in mods:
            if m not in per:
                continue
            q = per[m]
            f = (lambda v: f"{v / q['rv']:+8.3f}") if q["rv"] > 0 else (lambda v: f"{v:+8.5f}")
            P(f"   {m:8} {q['n']:6d} {q['mean_gain']:+8.4f} {q['sd_gain']:7.4f} {q['rv']:8.5f} {q['r']:6.3f} |"
              f"         {f(q['ev_b_real'])} {f(q['ev_c_real'])} {f(q['ev_d_real']):>9} | "
              f"{q['lambda_b']:g} {q['lambda_c']:g} {q['lambda_d']:g}")
        P(f"   pooled image tokens {out['n_tokens']}: Var {out['var_pooled_total']:.5f}; (a) modality explains "
          f"{100 * out['share_a']:.3f}% of it")
        for tn in ("real", "shuffled", "planted"):
            o = out[tn]
            P(f"   [{tn:8}] EV b {o['ev']['b']:+.5f} c {o['ev']['c']:+.5f} d {o['ev']['d']:+.5f}  RV {o['rv']:.5f}  "
              f"EV/RV b {o['ev_over_rv']['b']:+.3f} c {o['ev_over_rv']['c']:+.3f} d {o['ev_over_rv']['d']:+.3f}  "
              f"best {o['best']}: D1 {o['D1']:+.5f} (lo {o['D1_lo']:+.5f}), D2 {o['D2']:+.5f} (lo {o['D2_lo']:+.5f})"
              f"  -> {o['verdict']}")
        P(f"   controls: shuffled {'OK' if out['control_shuffled_ok'] else 'FAILED'}; planted "
          f"{'OK' if out['control_planted_ok'] else 'FAILED'} (planted variance {plant_tot:.5f}, EV(d) rose by "
          f"{Pl['ev']['d'] - R['ev']['d']:+.5f}, EV(b) by {Pl['ev']['b'] - R['ev']['b']:+.5f})")
        P(f"   description: seed-0 fits on the seed-1 gain: b {out['seed1']['b']:+.5f} c {out['seed1']['c']:+.5f} "
          f"d {out['seed1']['d']:+.5f}" + (f"; rows permuted, position kept: b {out['rowperm']['b']:+.5f} "
                                           f"c {out['rowperm']['c']:+.5f} d {out['rowperm']['d']:+.5f}"
                                           if "rowperm" in out else ""))
        P("   reliability r over all 4,416 rows: " + ", ".join(f"{m} {v:.3f}" for m, v in out["r_all_rows"].items()))
        P("")
    verdicts = [report[ck]["pooled"]["real"]["verdict"] for ck in CKS]
    controls = all(report[ck]["pooled"]["control_shuffled_ok"] and report[ck]["pooled"]["control_planted_ok"]
                   for ck in CKS)
    overall = ("not read: a control failed" if not controls else
               "worth building" if all(v == "worth building" for v in verdicts) else
               "mixed" if any(v != "not worth building" for v in verdicts) else "not worth building")
    report["overall"] = overall
    P(f"OVERALL: {overall} ({', '.join(f'{ck}: {v}' for ck, v in zip(CKS, verdicts))})")
    (OUT / "read.json").write_text(json.dumps(report, indent=1, default=float))
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("checkpoint", type=int, choices=CKS)
    c.add_argument("--smoke", type=int, default=0, help="the first N batches only, 8 rows, into --out")
    c.add_argument("--out", type=Path, default=OUT)
    k = sub.add_parser("kl", help="the decomposition's GPU part (description, added 2026-11-09, between 14:10 and 14:13)")
    k.add_argument("checkpoint", type=int, choices=CKS)
    k.add_argument("--smoke", type=int, default=0, help="the first N batches only, on a smoke capture in --out")
    k.add_argument("--out", type=Path, default=OUT)
    sub.add_parser("selftest")
    sub.add_parser("fit")
    sub.add_parser("klfit")
    sub.add_parser("value", help="what a score is worth in nats at equal compute (description, added 14:34)")
    sub.add_parser("value3", help="small moves off uniform depth 2 (description, added 14:44)")
    a = ap.parse_args()
    if a.cmd in ("capture", "kl"):
        if a.smoke and a.out == OUT:
            raise SystemExit("--smoke writes elsewhere: pass --out")
        return (capture if a.cmd == "capture" else kl_capture)(a.checkpoint, a.smoke, a.out)
    if a.cmd == "selftest":
        return selftest()
    return {"fit": fit, "klfit": klfit, "value": value, "value3": value3}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
