#!/usr/bin/env python
"""B0 Step 2: the forces on the router's logits at a checkpoint (H1, docs/plans/b0_router_audit.md).

The LM loss reaches the router only through the gate: `topk` is not differentiable, so the one
path from the logits to the loss is the top probability that scales a token's block output at its
exit. Making the router's output a leaf and running backward from the summed CE gives the real
force on every logit, with only the last layer and the head in the graph (the recursion's blocks
hold no path from the leaf). Balancing and z-loss are analytic, in the same units, which are the
training loss x N, i.e. nats per unit logit, per position:

    LM           dCE_sum / dl_j
    balancing    coeff * p_j (f_j - sum_i p_i f_i)         f_i = 3 x this micro-batch's load
    z-loss       z_coeff * 2 * logsumexp(l) * p_j

From the chosen depth's entry: dCE/dl_c = dCE/dw * p_c (1 - p_c), so dCE/dw follows. Positive means
the loss wants a smaller gate: descent lowers the chosen depth's probability.

Everything is also kept per CHOSEN depth. In a sliding tie the depth with the larger push down
loses tokens, so at equilibrium the split between two tied depths should follow the ratio of their
forces, plus the balancing term. That difference is the only signal about depth the router gets.

⚠ fp32: fp16 gradients of this size underflow. So routing here is fp32's, and a tie can fall the
other way from fp16's, which does not change a modality's forces.
⚠ Stratified rows: the held-out rows sort majortom-first, so a head holds no S1GRD.
⚠ --rows-from train is the force training felt; heldout is not. By the last checkpoints the model
has seen a training row ten times, and the sign of dCE/dw can differ between the two (overfitting
wants smaller updates on rows it has not seen). A rule about training dynamics reads train.

Controls in every unit, written to the report:
  directional  raising every chosen logit by eps changes the summed CE by the gradient's prediction
  padding      padding positions predict nothing and nothing reads them: LM force exactly zero

    CUDA_VISIBLE_DEVICES=... python eo/experiments/routing/b0_forces.py --arm-tag a10c \
        --run-dir <run> [--rows 256] checkpoint-330 ...

Writes /data/enric/reports/b0/forces/<tag>/<ck>.{json,npz}. Runs in `.venv`.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eo.data.eval_split import load_eval_rows, load_row_table, train_rows_from_eval  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.generate import conditional as G  # noqa: E402
from eo.routing import stats as S  # noqa: E402
from eo.routing.telemetry import find_mor_layers  # noqa: E402
from eo.scripts.capture_router import sample_rows, subset  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]
EPS = 1e-3


def summed_ce(logits, labels):
    lg = logits[:, :-1]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1),
                           ignore_index=-100, reduction="none").double().sum()


def one_unit(tag, run_dir, ck, cfg_name, rows_n, out_root, workers, rows_from="heldout"):
    torch.manual_seed(42)
    ck_path = None if ck == "untrained" else str(Path(run_dir) / ck)
    model, cfg = G.build_model(cfg_name, ck_path)
    model.eval()
    model.requires_grad_(False)
    layer = find_mor_layers(model)[0]
    coeff, z_coeff = float(cfg.mor.token.coeff), float(cfg.mor.z_coeff)
    ds, _ = TF.eval_loader(cfg, num_workers=workers)
    table = load_row_table(ROOT)
    corpus = table.corpus.values
    pool = np.asarray(ds.rows) if rows_from == "heldout" else train_rows_from_eval(
        load_eval_rows(root_dir=ROOT), len(table))
    rows = sample_rows(pool, corpus, rows_n)
    dl = torch.utils.data.DataLoader(subset(ds, rows), batch_size=4, num_workers=workers)
    cap = {}
    bump = {"eps": None}

    def hook(_m, _i, o):
        o = o.detach().float().requires_grad_(True)
        cap["w"] = o
        if bump["eps"] is None:
            return o
        c = o.detach().argmax(-1, keepdim=True)
        return o + torch.zeros_like(o).scatter_(-1, c, bump["eps"])

    h = layer.mor_router.register_forward_hook(hook)
    keys = ("gc", "dw", "pos", "bal", "z", "lm3", "bal3", "z3", "p", "n", "g", "chosen",
            "gc_d", "dw_d", "bal_d")
    acc = {k: [] for k in keys}
    pad_max = 0.0
    dir_err = []
    t0 = time.perf_counter()
    for b, batch in enumerate(dl):
        batch = dict(batch)
        mids = batch.pop("modality_ids")
        labels = batch.pop("labels").to("cuda")
        inp = {k: v.to("cuda") for k, v in batch.items()}
        L = summed_ce(model(**inp).logits, labels)
        L.backward()
        w, grad = cap["w"].detach(), cap["w"].grad.detach()
        if b < 4:                                     # the directional-derivative control
            with torch.no_grad():
                bump["eps"] = EPS
                L1 = summed_ce(model(**inp).logits, labels)
                bump["eps"] = None
            c0 = w.argmax(-1, keepdim=True)
            pred = (grad.gather(-1, c0).double() * EPS).sum()
            dir_err.append(float(abs((L1 - L.detach()) - pred) / max(abs(pred), 1e-12)))
        p = torch.softmax(w.double(), -1)
        c = p.argmax(-1, keepdim=True)
        pc = p.gather(-1, c).squeeze(-1)
        load = torch.stack([torch.bincount(x.view(-1), minlength=3) for x in c.split(1)]).sum(0).double()
        f = 3 * load / load.sum()
        fb = coeff * p * (f.view(1, 1, 3) - (p * f.view(1, 1, 3)).sum(-1, keepdim=True))
        lse = torch.logsumexp(w.double(), -1, keepdim=True)
        fz = z_coeff * 2 * lse * p
        gc = grad.gather(-1, c).squeeze(-1).double()
        dw = gc / (pc * (1 - pc)).clamp_min(1e-9)
        lmod = mids[:, 1:].to("cuda")
        val = labels[:, 1:].ne(-100) & lmod.gt(0)
        pad = inp["attention_mask"][:, :-1].eq(0)
        pad_max = max(pad_max, float(grad[:, :-1][pad].abs().max()) if pad.any() else 0.0)
        mid_t = torch.arange(1, 8, device="cuda").view(-1, 1, 1)
        M = (val.unsqueeze(0) & lmod.unsqueeze(0).eq(mid_t)).double()         # (7, B, L-1)
        fr = (lambda x: (M.unsqueeze(-1) * x[:, :-1].unsqueeze(0).unsqueeze(-1)).sum(2).squeeze(-1)
              .t().cpu().numpy() if x.dim() == 2 else
              (M.unsqueeze(-1) * x[:, :-1].unsqueeze(0)).sum(2).permute(1, 0, 2).cpu().numpy())
        acc["n"].append(fr(torch.ones_like(gc)))
        acc["gc"].append(fr(gc))
        acc["dw"].append(fr(dw))
        acc["pos"].append(fr((gc > 0).double()))
        acc["bal"].append(fr(fb.gather(-1, c).squeeze(-1)))
        acc["z"].append(fr(fz.gather(-1, c).squeeze(-1)))
        acc["g"].append(fr(pc))
        acc["lm3"].append(fr(grad.double()))
        acc["bal3"].append(fr(fb))
        acc["z3"].append(fr(fz))
        acc["p"].append(fr(p))
        hot = F.one_hot(c.squeeze(-1), 3).double()
        acc["chosen"].append(fr(hot))
        acc["gc_d"].append(fr(hot * gc.unsqueeze(-1)))
        acc["dw_d"].append(fr(hot * dw.unsqueeze(-1)))
        acc["bal_d"].append(fr(hot * fb.gather(-1, c)))
        del L
    h.remove()
    arr = {k: np.concatenate(v) for k, v in acc.items()}
    units = S.eval_groups(rows)
    report = {"tag": tag, "ck": ck, "rows": len(rows), "rows_from": rows_from,
              "groups": int(units.max()) + 1,
              "seconds": round(time.perf_counter() - t0, 1), "gpu": torch.cuda.get_device_name(0),
              "pad_max_force": pad_max, "directional_rel_err": dir_err, "mods": {}}
    for j, m in enumerate(MODS):
        n = arr["n"][:, j]
        if n.sum() == 0:
            continue
        keep = n > 0
        u = np.unique(units[keep], return_inverse=True)[1]
        r = {"n": int(n.sum()), "rows": int(keep.sum()), "gate": float(arr["g"][:, j].sum() / n.sum())}
        for k, name in (("gc", "force_chosen"), ("dw", "dL_dw"), ("pos", "share_positive"),
                        ("bal", "bal_chosen"), ("z", "z_chosen")):
            e, lo, hi = S.ratio_boot(arr[k][keep, j], n[keep], u, n_draws=1000)
            r[name], r[f"{name}_ci"] = float(e), [float(lo), float(hi)]
        for k in ("lm3", "bal3", "z3", "p", "chosen"):
            r[k] = [float(arr[k][:, j, d].sum() / n.sum()) for d in range(3)]
        for k, name in (("gc_d", "force_by_depth"), ("dw_d", "dLdw_by_depth"), ("bal_d", "bal_by_depth")):
            nd = arr["chosen"][:, j]
            r[name] = [float(arr[k][:, j, d].sum() / nd[:, d].sum()) if nd[:, d].sum() else None
                       for d in range(3)]
        report["mods"][m] = r
    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ck}.json").write_text(json.dumps(report, indent=1) + "\n")
    np.savez_compressed(out / f"{ck}.npz", rows=rows, **arr)
    print(f"{tag} {ck}: {report['seconds']}s, padding max force {pad_max:.2e}, directional rel err "
          f"{max(dir_err):.3f}", flush=True)
    for m, r in report["mods"].items():
        print(f"  {m:7s} gate {r['gate']:.3f} p {np.round(r['p'], 3)}  F_c {r['force_chosen']:+.4f} "
              f"[{r['force_chosen_ci'][0]:+.4f},{r['force_chosen_ci'][1]:+.4f}]  dL/dw {r['dL_dw']:+.4f} "
              f"[{r['dL_dw_ci'][0]:+.4f},{r['dL_dw_ci'][1]:+.4f}]  bal {r['bal_chosen']:+.4f}  z {r['z_chosen']:+.5f}  "
              f"mix {np.round(r['chosen'], 2)} F by depth "
              + " ".join("-" if x is None else f"{x:+.4f}" for x in r["force_by_depth"]), flush=True)
    del model
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_a10c_mor")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--rows-from", choices=("heldout", "train"), default="heldout")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-root", default="/data/enric/reports/b0/forces")
    args = ap.parse_args()
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    for ck in args.checkpoints:
        if (Path(args.out_root) / args.arm_tag / f"{ck}.json").exists():
            print(f"[skip] {args.arm_tag} {ck}")
            continue
        one_unit(args.arm_tag, args.run_dir, ck, args.arm_config, args.rows, args.out_root,
                 args.num_workers, args.rows_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
