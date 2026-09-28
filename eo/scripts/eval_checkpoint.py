#!/usr/bin/env python
"""Teacher-forced per-checkpoint evaluation of one arm (accuracy curves + recursion).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanv) \
    python eo/scripts/eval_checkpoint.py --arm-tag arm_a --arm-config eo_terramesh/arm_a_mor \
        --run-dir /data/enric/runs/pretrain/phase3/<run> untrained checkpoint-2000 ... \
        [--modes router,force1,force2,force3,perm_modality,perm_all]

Per checkpoint, writes /data/enric/reports/arm_eval/<arm-tag>/<ckpt>/:
  metrics.json         per-modality CE / top-1 / top-5 / slot mass, for every
                       mode, plus the target-last pass and the LOGGED eval loss
                       at that step for comparison
  tokens_router.npz    per-token CE, correctness, modality and (arm A) depth
  tokens_<mode>.npz    CE + correctness under each intervention

⚠ THE CHECK THAT THIS PASS IS THE TRAINER'S PASS: `ce_batch_mean` must match
`eval_loss_<mod>` in trainer_state.json at the same step to within fp16
autocast noise. It is printed per modality and stored; a large gap means the
loader, the modality order or the attribution differs from training.

⚠ `untrained` is a legitimate checkpoint name: a randomly initialised model
(seed 42), the step-0 point of every curve.

Arm B works unchanged: it has no router, so it gets the accuracy metrics and
no depth, and any --modes other than `router` are refused for it.

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eo_vocab import MODALITIES  # noqa: E402
from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.generate import conditional as G  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]


def logged_eval(run_dir: Path, step: int):
    """eval_loss + eval_loss_<mod> at `step`. ⚠ They live in SEPARATE log_history
    entries at the same step (handover section 6 item 4) -- join on step."""
    f = run_dir / "trainer_state.json"
    if not f.exists():
        return None
    out = {}
    for e in json.loads(f.read_text())["log_history"]:
        if e.get("step") == step and any(k.startswith("eval_loss") for k in e):
            out.update({k: v for k, v in e.items() if k.startswith("eval_loss")})
    return out or None


def write_json(path: Path, doc) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes((json.dumps(doc, indent=2) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-tag", required=True)
    ap.add_argument("--arm-config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--modes", default="router")
    ap.add_argument("--target-last-rows", type=int, default=512)
    ap.add_argument("--out-root", default="/data/enric/reports/arm_eval")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="must equal the training config's dataloader_num_workers")
    args = ap.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    gpu = torch.cuda.get_device_name(0)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    run_dir = Path(args.run_dir)

    for ck in args.checkpoints:
        out = Path(args.out_root) / args.arm_tag / ck
        out.mkdir(parents=True, exist_ok=True)
        mpath = out / "metrics.json"
        doc = json.loads(mpath.read_text()) if mpath.exists() else {}
        todo = [m for m in modes if m not in doc.get("modes", {})]
        need_tl = "target_last" not in doc
        if not todo and not need_tl:
            print(f"[skip] {ck}")
            continue

        torch.manual_seed(42)
        ck_path = None if ck == "untrained" else str(run_dir / ck)
        model, cfg = G.build_model(args.arm_config, ck_path)
        if int(cfg.dataloader_num_workers) != args.num_workers:
            raise SystemExit(f"--num-workers {args.num_workers} != config's "
                             f"{cfg.dataloader_num_workers}: modality orders would differ from training")
        ov = TF.install_override(model, "router")
        if ov is None and any(m != "router" for m in todo):
            raise SystemExit(f"{ck}: interventions need a router; this arm has none")
        step = 0 if ck == "untrained" else int(ck.split("-")[-1])
        ds, loader = TF.eval_loader(cfg, num_workers=args.num_workers)
        doc.update({"arm_tag": args.arm_tag, "arm_config": args.arm_config, "checkpoint": ck,
                    "step": step, "gpu": gpu, "n_eval_rows": len(ds),
                    "has_router": ov is not None,
                    "logged_eval": logged_eval(run_dir, step) if ck != "untrained" else None})
        doc.setdefault("modes", {})

        for mode in todo:
            t0 = time.perf_counter()
            if ov is not None:
                TF.install_override(model, mode, seed=0)
            res = TF.run_pass(model, loader, override=ov)
            tok = res.pop("tokens")
            keep = tok if mode == "router" else {k: tok[k] for k in ("ce", "correct", "depth") if k in tok}
            np.savez_compressed(out / f"tokens_{mode}.npz", **keep)
            if mode == "router":
                np.save(out / "eval_rows.npy", np.asarray(ds.rows))
            res["seconds"] = round(time.perf_counter() - t0, 1)
            if "depth" in tok:
                v = tok["label_mod"] > 0 if "label_mod" in tok else None
                d = tok["depth"][v] if v is not None else tok["depth"].ravel()
                res["mean_depth_body"] = float(d.mean())
            doc["modes"][mode] = res
            write_json(mpath, doc)
            pm = res["per_modality"]
            print(f"{ck} [{mode}] {res['seconds']}s  CE(all) {res['ce_all_body_tokens']:.4f}"
                  + (f"  depth {res.get('mean_depth_body', 0):.3f}" if ov is not None else ""))
            for n in pm:
                lg = (doc["logged_eval"] or {}).get(f"eval_loss_{n}")
                print(f"   {n:7s} ce {pm[n]['ce']:.4f}  batch-mean {pm[n]['ce_batch_mean']:.4f}"
                      f"  logged {lg if lg is not None else '-':>7}  top1 {pm[n]['top1']:.4f}"
                      f"  top5 {pm[n]['top5']:.4f}  slot {pm[n]['slot_mass']:.4f}")

        if need_tl:
            if ov is not None:
                TF.install_override(model, "router")
            corpus = load_row_table(ROOT).corpus.values
            eval_rows = load_eval_rows(root_dir=ROOT)
            tl = {}
            for T in TARGETS:
                rows = TF.stratified_target_rows(ds, eval_rows, corpus, T, args.target_last_rows)
                dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T), batch_size=4,
                                                 num_workers=args.num_workers)
                res = TF.run_pass(model, dl, override=ov, keep_tokens=ov is not None)
                r = {"n_rows": len(rows), **res["per_modality"][T]}
                if ov is not None:
                    t = res["tokens"]
                    sel = t["label_mod"] == TF.MODALITY_TO_ID[T]
                    r["mean_depth_target"] = float(t["depth"][sel].mean())
                tl[T] = r
                print(f"{ck} [target-last {T}] rows {len(rows)}  ce {r['ce']:.4f}  "
                      f"top1 {r['top1']:.4f}  top5 {r['top5']:.4f}  slot {r['slot_mass']:.4f}")
            doc["target_last"] = tl
            write_json(mpath, doc)

        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
