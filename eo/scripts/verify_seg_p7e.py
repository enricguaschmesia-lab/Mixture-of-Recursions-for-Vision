#!/usr/bin/env python
"""Gate for A2 Step 7.0e (docs/plans/a2_segmentation.md, Step 7, "7.0e, three generation-side tests on PASTIS-7"):
the nested PASTIS-7 map, the TTA root, the votes and the read. Every check has a control that fails it. `.venv`, CPU;
Z33d's re-tokenization runs as a subprocess in the a2seg env on the cards the artifacts were written on.

  Z33r  7.0d's gate (verify_seg_p7d.py, 7.0c's inside it) on this tree: 22/22.
  Z33a  the overlay eo_seg/s70e_grid784 differs from s70_p7 in exactly seg.arm (c4375_224), seg.map_order (nested) and
        max_length (2,193). Control: a stray key.
  Z33b  the nested PASTIS-7 layout (train and val, from the composed overlay): the inputs are the 196-token sequence's
        bit for bit; map code (j, k) at 1,387 + 198 j + k, i.e. 198 (j + 1) after the latest date's patch k; EO_MAP at
        2,177; 785 labels; step_positions the codes' positions; to_raster undoes the order. Refusals: a dates subset or
        a date curriculum with the nested PASTIS-7 map. Control: the raster layout fails the offsets.
  Z33c  to_k0 inverts s2.dihedral for all 8 variants. Control: each variant used as its own inverse fails at k = 1, 3.
  Z33d  the TTA root: 3,856 views, patch-major k = 0..7, every view's 8 frames those of its k = 0 view; the k = 0 rows
        (S2 and maps) equal today's artifacts; the first 64 new S2 rows and 16 new maps re-tokenized on their cards equal
        the stored ones; the true map tokens of each variant, decoded and brought back, score within 0.005 of 0.9143.
        Controls: the same rows tokenized at k = 0 differ; the truth not brought back falls at k = 1..7.
  Z33e  the votes on B1's decoded K = 8 maps: the plain vote equals the scorer's per-patch matrices exactly, as does the
        weighted vote at w = 1; the halves alternate by sorted patch id; the cross-fit applies each half the other's
        alpha. Control: weights in a permuted class order change the matrices.
  Z33f  the read's rules: decode_call, test2_verdict, band_e784; tta_stacks refuses views that are not patch-major.
  Z33p  E784 through pretrain.py on CPU, 2 steps. Control: the nested PASTIS-7 map with a dates subset is refused.
    python eo/scripts/verify_seg_p7e.py [--only Z33a,Z33b] [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "segmentation"))

from eo.data import seg_dataset as SD              # noqa: E402

RESULTS: list = []
A2SEG = Path("/data/enric/runs/pretrain/a2_seg")
B1_BASE = A2SEG / "base/b10cltrxnct_ext_map_nc_ckpt2960"
PARENT, OVERLAY = "eo_seg/s70_p7", "eo_seg/s70e_grid784"
B1_CLI = ["learning_rate=3e-3", "seed=42"]
TTA = Path("/data/enric/data/geobench2_tta")
TODAY = Path("/data/enric/data/geobench2/pastis/tok")
OUT = Path("/data/enric/reports/a2_seg/step70e")
TMP = OUT / "gate_tmp"
A2SEG_PY = "/data/enric/miniforge3/envs/a2seg/bin/python"


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _diff(a, b) -> set:
    return {k for k in set(a) | set(b) if (k in a) != (k in b) or a.get(k) != b.get(k)}


# ------------------------------------------------------------------ Z33r
def z33r() -> None:
    out = TMP / "z33r_gate_p7d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "eo/scripts/verify_seg_p7d.py", "--out", str(out)], cwd=str(REPO),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=7200)
    (TMP / "z33r_gate_p7d.log").write_text(r.stdout + r.stderr)
    res = json.loads(out.read_text()) if out.exists() else []
    n_pass = sum(x["status"] == "PASS" for x in res)
    record("Z33r 7.0d's gate (7.0c's inside) on this tree", r.returncode == 0 and len(res) == 22 and n_pass == 22,
           f"exit {r.returncode}; {n_pass}/{len(res)} pass; {time.time() - t0:.0f} s; {out}")


# ------------------------------------------------------------------ Z33a
def z33a() -> None:
    from omegaconf import OmegaConf
    a = _flat(OmegaConf.to_container(_compose(PARENT, B1_CLI), resolve=False))
    g = _flat(OmegaConf.to_container(_compose(OVERLAY, B1_CLI), resolve=False))
    d = _diff(a, g)
    want = {"seg.arm": "c4375_224", "seg.map_order": "nested", "max_length": 2193}
    ok = d == set(want) and {k: g.get(k) for k in want} == want
    record("Z33a the overlay = s70_p7 but the 784 arm, the nested order and the length", ok,
           f"differs in {sorted(d)} = {({k: g.get(k) for k in want})}")
    c = _flat(OmegaConf.to_container(_compose(OVERLAY, B1_CLI + ["weight_decay=0.01"]), resolve=False))
    record("Z33a control: a stray key", _diff(a, c) != set(want), f"differs in {sorted(_diff(a, c))}")


# ------------------------------------------------------------------ Z33b
def _layout_ok(ds, d196, v) -> tuple:
    s, t = ds.sequence(v), d196.sequence(v)
    n_in = 7 * 201
    inputs = all(torch.equal(s[k][:n_in], t[k][:n_in]) for k in ("input_ids", "position_ids", "modality_ids",
                                                                   "attention_mask"))
    pos = s["position_ids"]
    j, k = torch.arange(784) // 196, torch.arange(784) % 196
    codes = pos[n_in + 1:n_in + 1 + 784]
    offsets = torch.equal(codes - (198 * 6 + 1 + k), 198 * (j + 1))
    lab = (s["labels"] != -100).nonzero().flatten()
    eo = int(pos[lab[-1]]) == 2177 and len(lab) == 785 and int(pos[n_in]) == 1386
    steps = torch.equal(ds.step_positions(), codes)
    raw = np.asarray(ds.maps[v]).astype(np.int64)
    seq = s["input_ids"][n_in + 1:n_in + 1 + 784].numpy() - ds.map_off
    raster = np.array_equal(ds.to_raster(seq), raw)
    return inputs, offsets, eo, steps, raster


def z33b() -> None:
    cfg = _compose(OVERLAY, B1_CLI)
    bad, n = [], 0
    for want in ("eval", "train"):
        spec = SD.spec_from_cfg(cfg, want)
        ds = SD.SegDataset(spec, max_length=int(cfg.max_length), seed=42, batch=32 if want == "train" else 1)
        sp196 = SD.spec_from_cfg(_compose(PARENT, B1_CLI), want)
        d196 = SD.SegDataset(sp196, max_length=1605, seed=42, batch=32 if want == "train" else 1)
        vs = [int(x) for x in ds.views.view.values[[0, 7, 100, len(ds.views) - 1]]]
        for v in vs:
            r = _layout_ok(ds, d196, v)
            if not all(r):
                bad.append((want, v, r))
            n += 1
    refused = {}
    for name, kw in (("dates", dict(dates=(4, 5, 6))), ("curriculum", dict(split="train", in_pinned=True,
                                                                            date_curriculum=2))):
        try:
            SD.SegDataset(SD.SegSpec(**dict(dict(task="pastis", arm="c4375_224", frames="p7", split="val",
                                                 map_order="nested"), **kw)), max_length=2193, batch=32)
            refused[name] = False
        except ValueError:
            refused[name] = True
    record("Z33b the nested PASTIS-7 layout", not bad and all(refused.values()),
           f"{n} sequences (val, train): inputs, offsets 198 (j + 1), EO_MAP 2,177 / 785 labels, step positions, raster "
           f"round trip; problems {bad[:3]}; refusals {refused}")
    rs = SD.SegDataset(SD.SegSpec("pastis", "c4375_224", "p7", "val"))
    d196 = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "val"), max_length=1605)
    v = int(rs.views.view.values[0])
    r = _layout_ok(rs, d196, v)
    record("Z33b control: the raster layout", not r[1], f"offsets hold under raster: {r[1]}")


# ------------------------------------------------------------------ Z33c
def z33c() -> None:
    import step70e_votes as VT
    from eo.geobench import s2 as G
    x = np.random.default_rng(0).integers(0, 20, size=(3, 128, 128)).astype(np.uint8)
    ok = all(np.array_equal(VT.to_k0(G.dihedral(x, k), k), x) for k in range(8))
    record("Z33c to_k0 inverts s2.dihedral", ok, f"8 variants: {ok}")
    self_inv = {k: bool(np.array_equal(G.dihedral(G.dihedral(x, k), k), x)) for k in range(8)}
    record("Z33c control: a variant as its own inverse", not self_inv[1] and not self_inv[3],
           f"self-inverse by k: {self_inv}")


# ------------------------------------------------------------------ Z33d
def _a2seg(part: str, card: str) -> dict:
    gpu = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "eo.train.gpu", card], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1",
               HF_HOME="/data/enric/hf")
    # the TITAN V stalls with no identifiable op (CLAUDE.md; it hung this check twice on 2026-11-27): each try under
    # timeout -s ABRT (seconds normally), one retry
    for attempt in (1, 2):
        r = subprocess.run(["timeout", "-s", "ABRT", "300", A2SEG_PY, "-X", "faulthandler",
                            "eo/experiments/segmentation/step70e_tta_tokens.py", "--part", part], cwd=str(REPO),
                           env=env, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            break
    (TMP / f"z33d_{part}.log").write_text(r.stdout + r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.returncode}


def z33d() -> None:
    import pandas as pd
    import step70e_read as RD
    import step70e_votes as VT
    from step4_score import per_patch_cm
    from step52_read import IGNORE
    from step53_verdict import miou
    tok = TTA / "pastis" / "tok"
    v, x = pd.read_parquet(tok / "views.parquet"), pd.read_parquet(tok / "inputs.parquet")
    v0, x0 = pd.read_parquet(TODAY / "views.parquet"), pd.read_parquet(TODAY / "inputs.parquet")
    pm = len(v) == 3856 and (v.k.values.reshape(-1, 8) == np.arange(8)).all() and (v.split == "val").all()
    fr = x.merge(v[["view", "sample_id", "k"]], on="view")
    k0 = fr[fr.k == 0].set_index(["sample_id", "frame"])[["t", "date"]]
    frames_ok = all((g.set_index(["sample_id", "frame"])[["t", "date"]] == k0.loc[g.set_index(["sample_id", "frame"])
                                                                                  .index]).all().all()
                    for _, g in fr.groupby("k"))
    copy = json.loads((tok / "tta_copy.json").read_text())
    s_new, s_old = np.load(tok / "S2L2A_tok224/tokens.npy", mmap_mode="r"), np.load(TODAY / "S2L2A_tok224/tokens.npy",
                                                                                      mmap_mode="r")
    m_new, m_old = np.load(tok / "MAP_aligned/tokens.npy", mmap_mode="r"), np.load(TODAY / "MAP_aligned/tokens.npy",
                                                                                    mmap_mode="r")
    s_rows = np.array(sorted(int(k) for k in copy["s2"]))
    k0_s2 = np.array_equal(s_new[s_rows], s_old[[copy["s2"][str(r)] for r in s_rows]])
    m_rows = np.array(sorted(int(k) for k in copy["map"]))
    k0_map = np.array_equal(m_new[m_rows], m_old[[copy["map"][str(r)] for r in m_rows]])
    # the copied rows are today's k = 0 rows of the same patch and frame
    old_key = x0.merge(v0[["view", "sample_id", "k"]], on="view").set_index("row")
    new_key = fr.set_index("row")
    keys_ok = all((new_key.loc[r, ["sample_id", "frame"]].tolist() == old_key.loc[copy["s2"][str(r)],
                                                                                   ["sample_id", "frame"]].tolist())
                  and old_key.loc[copy["s2"][str(r)], "k"] == 0 for r in s_rows[::97])
    pres = (np.load(tok / "S2L2A_tok224/present.npy").all() and np.load(tok / "MAP_aligned/present.npy").all())
    cs2, cmap = _a2seg("check_s2", "titanx"), _a2seg("check_maps", "titanv")
    ok = pm and frames_ok and k0_s2 and k0_map and keys_ok and pres and cs2.get("equal") and cmap.get("equal")
    record("Z33d the TTA root's views, frames, copied rows and re-tokenized rows", bool(ok),
           f"patch-major {pm}; frames = k = 0's {frames_ok}; k = 0 S2 {k0_s2}, maps {k0_map}, keys {keys_ok}; all present "
           f"{pres}; re-tokenized S2 {cs2}; maps {cmap}")
    record("Z33d control: the same rows tokenized at k = 0", (cs2.get("control_k0_equal") is False)
           and (cmap.get("control_k0_equal") is False),
           f"S2 equal {cs2.get('control_k0_equal')} (token share {cs2.get('control_k0_token_share_equal')}); maps equal "
           f"{cmap.get('control_k0_equal')} (share {cmap.get('control_k0_token_share_equal')})")
    tm, tv, _ = RD.load_maps("tta_truth")
    ks = np.asarray(tv["k"])
    ceil, raw = {}, {}
    for k in range(8):
        rows = np.where(ks == k)[0]
        lab = RD.labels_for([tv["sample_id"][r] for r in rows])
        back = np.stack([VT.to_k0(np.asarray(tm[r, 0]), k) for r in rows])
        stay = np.stack([np.asarray(tm[r, 0]) for r in rows])
        ceil[k] = miou(per_patch_cm(back, lab, 20, IGNORE["pastis"]).sum(0), IGNORE["pastis"])
        raw[k] = miou(per_patch_cm(stay, lab, 20, IGNORE["pastis"]).sum(0), IGNORE["pastis"])
    ok = all(abs(c - RD.CEILING_K0) <= RD.CEILING_TOL for c in ceil.values())
    record("Z33d the true tokens of every variant, brought back, score the ceiling", ok,
           f"by k: {({k: round(c, 4) for k, c in ceil.items()})} against {RD.CEILING_K0} +- {RD.CEILING_TOL}")
    record("Z33d control: the truth not brought back", all(raw[k] < RD.CEILING_K0 - 0.1 for k in range(1, 8)),
           f"by k: {({k: round(c, 4) for k, c in raw.items()})}")


# ------------------------------------------------------------------ Z33e
def z33e() -> None:
    import step70e_read as RD
    import step70e_votes as VT
    import step70k_read as K
    from step4_score import per_patch_cm
    from step52_read import IGNORE
    maps, vj, _ = RD.load_maps("b1_k8")
    stored = np.load(K.gen_dir(K.B1_RUN) / "full_k8" / "per_patch_cm.npy")
    labels = RD.labels_for(vj["sample_id"])
    n = len(vj["sample_id"])
    plain = RD.cms_of(lambda i: maps[i], n, labels, None)
    ones = RD.cms_of(lambda i: maps[i], n, labels, np.ones(20))
    h = RD.halves(vj["sample_id"])
    ids = np.asarray([int(s) for s in vj["sample_id"]])
    o = np.argsort(ids, kind="stable")
    h_ok = (h[o] == np.arange(n) % 2).all()
    fake = {0.0: np.zeros((n, 20, 20)), 0.5: np.zeros((n, 20, 20))}      # 0.5 adds a correct class 1 everywhere
    fake[0.0][:, 0, 0], fake[0.0][:, 1, 0] = 1, 1
    fake[0.5][:, 0, 0], fake[0.5][:, 1, 1] = 1, 1
    cf, chosen = RD.cross_fit(fake, h)
    ok = (np.array_equal(plain, stored) and np.array_equal(ones, stored) and h_ok and chosen == {0: 0.5, 1: 0.5})
    record("Z33e the votes on B1's decoded maps", ok,
           f"plain = scorer's matrices {np.array_equal(plain, stored)}; w = 1 {np.array_equal(ones, stored)}; halves "
           f"alternate {h_ok} ({int((h == 0).sum())}/{int((h == 1).sum())}); cross-fit chose {chosen}")
    prior = json.loads(RD.PRIOR.read_text())
    w = VT.class_weights(prior, 1.0, 20, IGNORE["pastis"])
    perm = np.roll(w[:19], 7).tolist() + [w[19]]
    a = RD.cms_of(lambda i: maps[i], 40, labels[:40], w)
    b = RD.cms_of(lambda i: maps[i], 40, labels[:40], np.asarray(perm))
    record("Z33e control: weights in a permuted class order", not np.array_equal(a, b),
           f"the 40-patch matrices differ: {not np.array_equal(a, b)}")


# ------------------------------------------------------------------ Z33f
def z33f() -> None:
    import step70e_read as RD
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    calls = [RD.decode_call(G(0.4, 0.1, 0.7)), RD.decode_call(G(0.4, -0.1, 0.7)), RD.decode_call(G(-0.4, -0.7, -0.1)),
             RD.decode_call(None)]
    v = [RD.test2_verdict(G(0.5, 0.1, 0.9), [G(0.2, -0.1, 0.5), G(0.1, -0.2, 0.4)]),
         RD.test2_verdict(G(0.5, 0.1, 0.9), [G(0.2, -0.1, 0.5), G(-0.1, -0.4, 0.2)]),
         RD.test2_verdict(G(-0.5, -0.9, -0.1), [G(-0.2, -0.5, 0.1), G(-0.1, -0.4, 0.2)])]
    bands = [RD.band_e784(G(1.5, 0.3, 2.7))[:3], RD.band_e784(G(1.5, -0.1, 2.7))[:3], RD.band_e784(G(-1.3, -2.4, -0.2))[:3]]
    try:
        RD.tta_stacks(np.zeros((16, 1, 2, 2), np.uint8), {"k": [0] * 16, "sample_id": ["1"] * 16}, "tta64")
        refused = False
    except RuntimeError:
        refused = True
    ok = (calls == ["gains", "level", "costs", "hole"] and v == ["gains", "level", "costs"]
          and bands == ["(a)", "(b)", "(c)"] and refused)
    record("Z33f the read's rules", ok, f"decode_call {calls}; test 2 {v}; band {bands}; non-patch-major refused {refused}")
    record("Z33f control: one replicate below 0 is not a gain", v[1] != "gains", f"verdict {v[1]}")


# ------------------------------------------------------------------ Z33p
def _run(tag: str, extra: dict) -> dict:
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z33p_{tag}"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(B1_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "2", "per_device_train_batch_size": "1", "logging_steps": "1",
          "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false", "learning_rate": "3e-3",
          "seed": "42", "hydra.run.dir": f"{out}/hydra"}
    ov.update(extra)
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", OVERLAY] + \
        [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    hist = json.loads(st.read_text())["log_history"] if st.exists() else []
    return {"rc": r.returncode, "out": out, "seconds": time.time() - t0, "log": log,
            "losses": [h["loss"] for h in hist if "loss" in h],
            "line": next((ln for ln in log.splitlines() if ln.startswith("seg train:")), "")}


def z33p() -> None:
    A = _run("e784", {})
    ok = (A["rc"] == 0 and "arm c4375_224" in A["line"] and "map_order nested" in A["line"]
          and "2193 tokens" in A["line"] and len(A["losses"]) == 2 and all(math.isfinite(x) for x in A["losses"]))
    record("Z33p E784 through pretrain.py on CPU", ok,
           f"exit {A['rc']}; losses {A['losses']}; {A['seconds']:.0f} s; '{A['line'][-110:]}'; log {A['out'] / 'pretrain.log'}")
    C = _run("dates", {"+seg.dates": "[4,5,6]", "max_length": "1389"})
    record("Z33p control: the nested PASTIS-7 map with a dates subset", C["rc"] != 0 and "no dates subset" in C["log"],
           f"exit {C['rc']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    checks = {"Z33r": z33r, "Z33a": z33a, "Z33b": z33b, "Z33c": z33c, "Z33d": z33d, "Z33e": z33e, "Z33f": z33f,
              "Z33p": z33p}
    want = a.only.split(",") if a.only else list(checks)
    t0 = time.time()
    for k in want:
        try:
            checks[k]()
        except Exception as e:
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
