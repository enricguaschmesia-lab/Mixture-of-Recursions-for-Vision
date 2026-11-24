#!/usr/bin/env python
"""A2 Steps 4.3 and 4.4's gate (docs/plans/a2_segmentation.md, Step 4, "Steps 4.3 and 4.4's design and reading", item 7;
2026-11-23). Every check ships with a control that must fail it. A file of its own, so Step 4.3 adds no lines to the gates
other A2 work edits.

    .venv:  python eo/scripts/verify_seg_s43.py [--only Z24,Z24r,Z25,Z26,Z26p] [--out results.json]      (CPU)
    a2seg:  python eo/scripts/verify_seg_s43.py --only Z25g        (one GPU, after MAP_aligned_arb's val codes exist)

  Z24  The chain's sequences (eo/data/seg_dataset.py, CHAIN) on a synthetic LULC artifact, every val view and 64 training
       views: 594 tokens at sequential positions; [LULC][S2][map] with the LULC body equal to the artifact in LULC's slot;
       the S2 block and the map (ids and labels) equal to Route A's sequence of the view; S2 patch k -> map code k 198
       positions; labels only on the map's codes and EO_MAP; prompt ends in BO_MAP_PASTIS; wrong_lulc keeps the S2 and
       takes the deranged view's LULC; wrong takes both from it; wrong_lulc without a chain is refused
                                     | the LULC block after the S2 block: the S2 -> map offset becomes 396
  Z24r The real artifact (pastis/tok/LULCGEN_base, when it exists): its metadata (fp32, the map base's weights hash,
       the cache check passed with its control failing), every pinned training and val view present, no test view,
       codes in LULC's slot, the stored hashes
                                     | a view's codes changed after writing: the tokens' hash no longer matches
  Z25  The palette alias aligned_arb (CPU part): map_vocab aliases it to the aligned slot and geometry, frozen to the
       aligned tokenizer's own directory; its palette (palette.arm_palette, harness.task_palette "universal_arb") moves
       exactly CloudSen12's thick cloud, to colour 13, valid under Z1's rules, and leaves PASTIS unread (no entry); the
       aligned arm keeps decision 6's assignment
                                     | the aligned arm's palette read for the alias: thick cloud on LULC's clouds (8)
       (GPU part, with the ceiling: step52_score.py --ceiling aligned_arb, and the alias's true codes decoded under the
       aligned palette, which must move thick cloud's IoU)
  Z26  Overlays: s43_grid784 differs from r0_s41_lora16 in exactly seg.arm, seg.map_order and max_length; s43_chain in
       exactly seg.chain and max_length; each max_length is its sequence's required length; built from the map base
       each trains R0's 51,014,208 parameters
                                     | a stray key
  Z27  (Step 4.5, 2026-11-24) the 784-token grid on CloudSen12: s44_grid784_cloudsen12 joins Z26 (differs from
       s52_cloudsen12 in exactly seg.arm, seg.map_order, max_length; R0's parameters); Z27 the nested layout on 64 val
       views (codes, positions, labels, the S2 block)  | raster order read as the sequence
       Z27g (a2seg, GPU) the stored val codes reproduce Step 1's frozen CloudSen12 ceiling  | codes one view off
  Z26p The chain through pretrain.py, 2 steps on CPU on the synthetic artifact: the dataset reports the chain, finite
       losses, a save
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RESULTS: list = []
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
ROOT = Path("/data/enric/data/geobench2")
SCRATCH = Path(os.environ.get("S43_SCRATCH", "/data/enric/reports/a2_seg/step43/gate_root"))
LULC_SLOT = (76800, 4375)


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


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def synthetic_root() -> Path:
    """A scratch geobench2 root: PASTIS's real artifacts linked, plus LULCGEN_synth with seeded random LULC codes."""
    import pandas as pd
    t = SCRATCH / "pastis" / "tok"
    t.mkdir(parents=True, exist_ok=True)
    src = ROOT / "pastis" / "tok"
    for n in ("views.parquet", "inputs.parquet", "MAP_aligned", "S2L2A_tok224"):
        if not (t / n).exists():
            (t / n).symlink_to(src / n)
    g = t / "LULCGEN_synth"
    if not (g / "metadata.json").exists():
        g.mkdir(exist_ok=True)
        nv = len(pd.read_parquet(src / "views.parquet"))
        rng = np.random.default_rng(20261123)
        np.save(g / "tokens.npy", rng.integers(0, LULC_SLOT[1], (nv, 196)).astype(np.uint16))
        np.save(g / "present.npy", np.ones(nv, bool))
        (g / "metadata.json").write_text(json.dumps({"n_views": nv, "tokens_per_sample": 196, "synthetic": True}))
    return SCRATCH


def _chain_problems(dc, da, view: int, order_swapped: bool = False) -> list:
    """Z24's assertions on one view; [] when it passes."""
    import torch
    p = []
    s = dc.sequence(view)
    a = da.sequence(view)
    ids, pos, lab = s["input_ids"], s["position_ids"], s["labels"]
    if len(ids) != 594 or not bool((pos == torch.arange(len(pos))).all()):
        p.append("not 594 sequential positions")
    lulc = dc.V.get_modality("LULC")
    lb, s2b = (198, 0) if order_swapped else (0, 198)
    body = ids[lb + 1:lb + 197]
    if ids[lb] != lulc.bo_id or ids[lb + 197] != lulc.eo_id or \
            not bool((body - lulc.codebook_offset == torch.from_numpy(np.asarray(dc.lulc[view]).astype(np.int64))).all()):
        p.append("the LULC block is not the artifact's")
    if not bool((ids[s2b:s2b + 198] == a["input_ids"][:198]).all()):
        p.append("the S2 block differs from Route A's")
    if not (bool((ids[396:] == a["input_ids"][198:]).all()) and bool((lab[396:] == a["labels"][198:]).all())):
        p.append("the map differs from Route A's")
    if int((lab[:397] != -100).sum()) != 0:
        p.append("a label before the map's first code")
    off = int(pos[397] - pos[1 + 198])                 # map code 0 - S2 patch 0 in the [LULC][S2][map] layout
    if order_swapped:
        off = int(pos[397] - pos[1])                    # S2 patch 0 sits at 1 when S2 comes first
    if off != 198:
        p.append(f"S2 -> map offset {off}, not 198")
    return p


def z24() -> None:
    import torch
    from eo.data import seg_dataset as SD
    root = synthetic_root()
    for split, n in (("val", None), ("train", 64)):
        dc = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", split, in_pinned=True if split == "train" else None,
                                      root=root, chain="synth"))
        da = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", split, in_pinned=True if split == "train" else None,
                                      root=root))
        vs = dc.views.view.values if n is None else dc.views.view.values[:n]
        bad = {int(v): _chain_problems(dc, da, int(v)) for v in vs}
        bad = {k: v for k, v in bad.items() if v}
        record(f"Z24 chain layout ({split}, {len(vs)} views)", not bad,
               f"{len(bad)} views fail" + (f", e.g. {next(iter(bad.items()))}" if bad else
                                           "; 594 tokens, [LULC][S2][map], S2 -> map 198, map = Route A's"))
    dc = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val", root=root, chain="synth"))
    v = int(dc.views.view.values[5])
    d = dc.derangement()[v]
    s, wl, w, dd = dc.sequence(v), dc.sequence(v, "wrong_lulc"), dc.sequence(v, "wrong"), dc.sequence(d)
    pr_ids, _ = dc.prompt(v)
    ok = (bool((wl["input_ids"][198:396] == s["input_ids"][198:396]).all())
          and bool((wl["input_ids"][:198] == dd["input_ids"][:198]).all())
          and bool((wl["input_ids"][:198] != s["input_ids"][:198]).any())
          and bool((w["input_ids"][:396] == dd["input_ids"][:396]).all())
          and bool((wl["input_ids"][396:] == s["input_ids"][396:]).all())
          and len(pr_ids) == 397 and int(pr_ids[-1]) == dc.task_token)
    try:
        SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val", root=root, variant="wrong_lulc"))
        refused = False
    except ValueError:
        refused = True
    record("Z24 variants", ok and refused, f"wrong_lulc: own S2, view {d}'s LULC, own map; wrong: both of view {d}; "
           f"prompt 397 tokens ending in BO_MAP_PASTIS; wrong_lulc without a chain refused {refused}")
    # control: the LULC block after the S2 block
    orig = dc.input_chunks

    def swapped(view, variant=None):
        ch, mods = orig(view, variant)
        return ch[::-1], mods[::-1]
    dc.input_chunks = swapped
    da = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val", root=root))
    s2 = dc.sequence(v)
    off = int(s2["position_ids"][397] - s2["position_ids"][1])
    record("Z24 control: LULC after S2", off != 198 and bool(_chain_problems(dc, da, v)),
           f"S2 patch 0 -> map code 0 is {off} positions; the check reports {_chain_problems(dc, da, v)}")
    dc.input_chunks = orig


def z24r() -> None:
    import pandas as pd
    g = ROOT / "pastis" / "tok" / "LULCGEN_base"
    if not (g / "metadata.json").exists():
        record("Z24r the real LULC artifact", False, f"{g} not written yet")
        return
    m = json.loads((g / "metadata.json").read_text())
    views = pd.read_parquet(ROOT / "pastis" / "tok" / "views.parquet")
    tok = np.load(g / "tokens.npy")
    pres = np.load(g / "present.npy")
    want = ((views.split == "train") & (views.in_pinned == True)) | (views.split == "val")  # noqa: E712
    test = (views.split == "test").to_numpy()
    base_sha = json.loads((MAP_BASE / "extension.json").read_text()).get("weights_sha256")
    ok = (m["precision"] == "fp32" and m["weights_sha256"] == base_sha and m["cache_check"]["pass"]
          and not m["cache_check"]["control"]["within"] and m["slot"] == list(LULC_SLOT)
          and bool(pres[want.to_numpy()].all()) and not bool(pres[test].any()) and int(pres.sum()) == int(want.sum())
          and int(tok[pres].max()) < LULC_SLOT[1] and not bool(tok[test].any())
          and m["tokens_sha256"] == _sha(g / "tokens.npy") and m["present_sha256"] == _sha(g / "present.npy"))
    distinct = len(np.unique(tok[pres]))
    record("Z24r the real LULC artifact", ok,
           f"{int(pres.sum())} views ({int(want.sum())} pinned train + val), none of test; weights {m['weights_sha256'][:10]} "
           f"= the map base's; cache check {m['cache_check']['by']} (control ratio {m['cache_check']['control_ratio']:.3g}); "
           f"{distinct} distinct codes; hashes match; {m['gpu']}, {m['seconds']} s")
    t2 = tok.copy()
    v = int(np.flatnonzero(pres)[0])
    t2[v, 0] = (int(t2[v, 0]) + 1) % LULC_SLOT[1]
    h = hashlib.sha256()
    tmp = SCRATCH / "z24r_ctl.npy"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.save(tmp, t2)
    record("Z24r control: a code changed after writing", _sha(tmp) != m["tokens_sha256"],
           f"view {v}'s first code moved by one: the hash differs")
    tmp.unlink()


def z25() -> None:
    from eo.data import map_vocab as MV
    from eo.maps import frozen, harness as H, palette as P
    a_slot, b_slot = MV.LAYOUT.slot("aligned"), MV.LAYOUT.slot("aligned_arb")
    ok_slot = a_slot == b_slot and MV.spec("aligned_arb") == MV.spec("aligned") and frozen.ALL["aligned_arb"] == frozen.ALL["aligned"]
    _, arb, void = H.task_palette("cloudsen12", P.arm_palette("aligned_arb"))
    _, ali, _ = H.task_palette("cloudsen12", P.arm_palette("aligned"))
    t = P.TASKS["cloudsen12"]
    P.validate(P.UNIVERSAL, arb, t.ignore, t.n_classes)
    moved = [k for k in range(t.n_classes) if arb[k] != ali[k]]
    ok = (ok_slot and tuple(ali) == t.assign == (10, 8, 11, 12) and tuple(arb) == (10, 13, 11, 12)
          and moved == [1] and 8 not in arb and "pastis" not in P.ARBITRARY)
    record("Z25 palette alias", ok, f"slot {b_slot} = aligned's {a_slot}; tokenizer dir {frozen.ALL['aligned_arb']}; "
           f"assignment {tuple(arb)} against {tuple(ali)}: moved classes {[t.class_names[k] for k in moved]}")
    wrong = H.task_palette("cloudsen12", P.arm_palette("aligned"))[1]
    record("Z25 control: the aligned palette for the alias", tuple(wrong) != tuple(arb) and wrong[1] == 8,
           f"decision 6's assignment {tuple(wrong)} puts thick cloud on {wrong[1]} (LULC's clouds), not 13")


def z25g() -> None:
    """(a2seg, one GPU) The alias's stored true val codes: their ceiling under their own palette, against the aligned arm's
    ceiling on the same views; control: the same codes decoded under the aligned palette lose thick cloud."""
    import torch
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    from eo.maps import frozen, palette as P
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    tok = frozen.load("aligned", dev)
    res = {}
    for arm in ("aligned", "aligned_arb"):
        ds = SD.SegDataset(SD.SegSpec("cloudsen12", arm, "single", "val"))
        views = ds.views
        codes = np.stack([np.asarray(ds.maps[v]) for v in views.view.values]).astype(np.int64)
        res[arm] = (codes, views)
    names = P.TASKS["cloudsen12"].class_names
    out = {}
    for label, arm, codes_of in (("aligned", "aligned", "aligned"), ("arbitrary", "aligned_arb", "aligned_arb"),
                                 ("control", "aligned", "aligned_arb")):
        codes, views = res[codes_of]
        r = SEG.score("cloudsen12", arm, codes, views, device=dev, tok=tok)
        out[label] = {"mIoU": r["result"]["mIoU"], "per_class": dict(zip(names, r["result"]["iou_per_class"]))}
    diff = abs(out["arbitrary"]["mIoU"] - out["aligned"]["mIoU"])
    (SCRATCH.parent / "z25g.json").write_text(json.dumps({**out, "gpu": gpu}, indent=1, default=str))
    record("Z25g palette ceilings", diff <= 0.002,
           f"true-token ceiling on CloudSen12 val: aligned {out['aligned']['mIoU']:.5f}, arbitrary "
           f"{out['arbitrary']['mIoU']:.5f} (|diff| {diff:.5f}; the plan expects <= 0.002); {gpu}")
    record("Z25g control: the alias's codes under the aligned palette", out["control"]["mIoU"] < out["arbitrary"]["mIoU"] - 0.05,
           f"mIoU {out['control']['mIoU']:.5f} against {out['arbitrary']['mIoU']:.5f}")


def z26() -> None:
    from omegaconf import OmegaConf
    from eo.data import seg_dataset as SD
    from eo.finetune.init import init_from
    from eo.generate.conditional import build_model
    pairs = {"s43_grid784": {"seg.arm", "seg.map_order", "max_length"}, "s43_chain": {"seg.chain", "max_length"},
             "s43_palette": {"seg.arm"}, "s44_grid784_cloudsen12": {"seg.arm", "seg.map_order", "max_length"}}
    parent = {"s43_grid784": "r0_s41_lora16", "s43_chain": "r0_s41_lora16", "s43_palette": "s52_cloudsen12",
              "s44_grid784_cloudsen12": "s52_cloudsen12"}       # Step 4.5 (2026-11-24): the grid on CloudSen12
    a = _flat(OmegaConf.to_container(_compose("eo_seg/r0_s41_lora16"), resolve=False))
    for name, want in pairs.items():
        c = _compose(f"eo_seg/{name}")
        pa = _flat(OmegaConf.to_container(_compose(f"eo_seg/{parent[name]}"), resolve=False))
        b = _flat(OmegaConf.to_container(c, resolve=False))
        d = {k for k in set(pa) | set(b) if pa.get(k) != b.get(k)}
        req = SD.required_length(str(c.seg.task), str(c.seg.frames), str(c.seg.arm), "full", c.seg.get("chain"))
        record(f"Z26 {name}", d == want and int(c.max_length) == req,
               f"differs from {parent[name]} in {sorted(d)}; max_length {c.max_length}, required {req}")
        m, cfg = build_model(f"eo_seg/{name}", None, device="cpu", overrides=[f"finetune.init_from={MAP_BASE}"])
        rep = init_from(m, cfg)
        record(f"Z26 {name} trains R0's parameters", rep["trainable_params"] == 51_014_208,
               f"trainable {rep['trainable_params']:,} (R0: 51,014,208)")
        del m
    b = _flat(OmegaConf.to_container(_compose("eo_seg/s43_chain", ["weight_decay=0.1"]), resolve=False))
    d = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    record("Z26 control: a stray key", d != pairs["s43_chain"], f"with weight_decay=0.1 it differs in {sorted(d)}")


def z27() -> None:
    """(CPU, after MAP_c4375_224's CloudSen12 val codes exist) Step 4.5: the nested 784-token layout on CloudSen12 val.
    For 64 val views: the map codes sit at bo + 1 + 198 j + k (code t = j * 196 + k of the nested order), their ids are
    the stored raster codes reordered by nested_order plus the slot's offset, to_raster inverts it, and labels are on the
    784 codes and EO_MAP only; the S2 block equals the aligned arm's sequence of the same view."""
    from eo.data import map_vocab as MV, seg_dataset as SD
    d = ROOT / "cloudsen12" / "tok" / "MAP_c4375_224"
    if not (d / "metadata.json").exists():
        record("Z27 CloudSen12 nested layout", False, f"{d} not written yet")
        return
    ds = SD.SegDataset(SD.SegSpec("cloudsen12", "c4375_224", "single", "val", map_order="nested"))
    da = SD.SegDataset(SD.SegSpec("cloudsen12", "aligned", "single", "val"))
    off = ds.lay.slot("c4375_224")[0]
    order = SD.nested_order()
    vs = ds.views.view.values[:: max(1, len(ds.views) // 64)][:64]
    bad = []
    for v in vs:
        s, sa = ds.sequence(int(v)), da.sequence(int(v))
        raw = np.asarray(ds.maps[int(v)]).astype(np.int64)
        lab = (s["labels"] != -100).nonzero().flatten()
        codes_at = lab[:-1]                                   # the 784 codes; the last labelled token is EO_MAP
        ids = s["input_ids"][codes_at].numpy() - off
        pos = s["position_ids"][codes_at].numpy()
        bo = int(s["position_ids"][codes_at[0] - 1])
        t = np.arange(784)
        want_pos = bo + 1 + 198 * (t // 196) + t % 196
        n_in = int(codes_at[0]) - 1                           # tokens before BO_MAP
        if len(lab) != 785 or not np.array_equal(ids, raw[order]) or not np.array_equal(ds.to_raster(ids), raw) \
                or not np.array_equal(pos, want_pos) or not np.array_equal(s["input_ids"][:n_in], sa["input_ids"][:n_in]) \
                or not np.array_equal(s["position_ids"][:n_in], sa["position_ids"][:n_in]):
            bad.append(int(v))
    record("Z27 CloudSen12 nested layout", not bad, f"{len(vs)} val views: codes at bo + 1 + 198 j + k, ids = stored codes "
           f"in nested order + {off}, to_raster inverts, labels on 784 codes + EO_MAP, S2 block = aligned's; bad {bad[:5]}")
    v = int(vs[0])
    raw = np.asarray(ds.maps[v]).astype(np.int64)
    s = ds.sequence(v)
    lab = (s["labels"] != -100).nonzero().flatten()
    ids = s["input_ids"][lab[:-1]].numpy() - off
    record("Z27 control: raster order read as the sequence", not np.array_equal(ids, raw),
           f"agreement {(ids == raw).mean():.4f}")


def z27g() -> None:
    """(a2seg, one GPU) Step 4.5: CloudSen12's stored c4375_224 val codes through eval/seg.score (the generated-map path:
    reshape, decode, stitching, scorer) reproduce Step 1's frozen CloudSen12 ceiling to 1e-6; control: every view's codes
    taken from the next view lose it."""
    import torch
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    from eo.maps import frozen
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    ref = frozen.record("c4375_224")["ceilings"]["cloudsen12"]
    tok = frozen.load("c4375_224", dev)
    ds = SD.SegDataset(SD.SegSpec("cloudsen12", "c4375_224", "single", "val"))
    views = ds.views
    codes = np.stack([np.asarray(ds.maps[v]) for v in views.view.values]).astype(np.int64)
    got = SEG.score("cloudsen12", "c4375_224", codes, views, device=dev, tok=tok)["result"]["mIoU"]
    ctl = SEG.score("cloudsen12", "c4375_224", np.roll(codes, 1, axis=0), views, device=dev, tok=tok)["result"]["mIoU"]
    (SCRATCH.parent.parent / "step45").mkdir(parents=True, exist_ok=True)
    (SCRATCH.parent.parent / "step45" / "z27g.json").write_text(json.dumps(
        {"ceiling": got, "frozen": ref, "control": ctl, "gpu": gpu, "views": len(views)}, indent=1))
    record("Z27g CloudSen12 c4375_224 ceiling", abs(got - ref) < 1e-6,
           f"eval/seg {got:.6f} = Step 1's frozen {ref:.6f} on {len(views)} val views; {gpu}")
    record("Z27g control: codes one view off", ctl < got - 0.05, f"mIoU {ctl:.5f} against {got:.5f}")


def z26p() -> None:
    import math
    import subprocess
    from util.env import load_dotenv
    load_dotenv()
    root = synthetic_root()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z26p"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(MAP_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "2", "per_device_train_batch_size": "2", "per_device_eval_batch_size": "8",
          "logging_steps": "1", "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false",
          "learning_rate": "3e-3", "seg.root": str(root), "seg.chain": "synth", "hydra.run.dir": f"{out}/hydra"}
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", "eo_seg/s43_chain"] + \
        [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    losses = [h["loss"] for h in json.loads(st.read_text())["log_history"] if "loss" in h] if st.exists() else []
    ok = (r.returncode == 0 and (out / "checkpoint-2").is_dir() and "chain synth" in log and len(losses) == 2
          and all(math.isfinite(x) for x in losses))
    record("Z26p the chain through pretrain.py", ok, f"exit {r.returncode}; dataset reports the chain "
           f"{'chain synth' in log}; losses {losses}; {time.time() - t0:.0f} s, CPU fp32; log {out / 'pretrain.log'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z24": z24, "Z24r": z24r, "Z25": z25, "Z25g": z25g, "Z26": z26, "Z26p": z26p, "Z27": z27, "Z27g": z27g}
    want = a.only.split(",") if a.only else ["Z24", "Z26", "Z26p"]
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
