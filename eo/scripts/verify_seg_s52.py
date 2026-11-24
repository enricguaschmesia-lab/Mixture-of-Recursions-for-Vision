#!/usr/bin/env python
"""A2 Step 5.2's gate: CloudSen12's R0 overlay and Step 5.2's scorer (2026-11-22; docs/plans/a2_segmentation.md,
Step 5.2). Every check ships with a control that must fail it. Numbered Z16-Z17 because Z14 and Z15 are taken twice
(Step 4.1's verify_seg_recipe.py and R1's verify_seg_r1.py).

    python eo/scripts/verify_seg_s52.py --only Z16 [--out results.json]        # .venv, CPU
    python eo/scripts/verify_seg_s52.py --only Z17 [--out results.json]        # a2seg, one GPU (the GTX Titan X)
    python eo/scripts/verify_seg_s52.py --only Z18 [--out results.json]        # .venv, CPU (about 12 GB)

  Z16 The overlay eo_seg/s52_cloudsen12 against r0_s41_lora16 (the PASTIS recipe): it differs in exactly the task keys
      (seg.task, seg.frames, seg.in_pinned) and the schedule's step counts, which are PASTIS's in 125-step epochs (50
      epochs, warmup one, the last 20 % decaying, eval = save = one epoch); the learning rate and the plateau stop are
      the recipe's. The datasets it builds: 4,000 training tiles of 72 views (one draw per tile an epoch: 125 steps at
      batch 32, = eval_steps), val's 535 tiles x 9 crops = 4,815 views in tiling's order at k = 0; a sequence is 396
      tokens, [BO_S2L2A s1..s196 EO_S2L2A][BO_MAP_CLOUDSEN12 m1..m196 EO_MAP], codes in the aligned slot, labels on the
      codes and EO_MAP only, the prompt ending at BO_MAP_CLOUDSEN12; a tile's draw is one of its views and moves with
      the epoch; the test split is refused. The plateau stop on 125-step epochs fires where it does on 37-step ones
      (Step 4.0's aligned curve: epoch 29, decay 8 epochs, stop at 29 x 125 + 1,000 = 4,625) with the intended LR.
                                     | a stray key (weight_decay) is flagged; PASTIS's 37-step epoch is flagged;
                                       in_pinned true (r0_base's) is refused for CloudSen12; a curve gaining 0.2 % an
                                       epoch never fires on 125-step epochs
  Z17 Step 5.2's scorer (eo/experiments/segmentation/step52_score.py) on CloudSen12: the stored true tokens of val's
      4,815 views, decoded, stitched per tile and scored, reproduce Step 1's frozen ceiling (1e-6), and the 535
      per-tile matrices sum to the scorer's mIoU (checked inside score_codes)
                                     | each tile given the next tile's codes loses the ceiling; two crops swapped
                                       within a tile are refused
  Z18 R1's packaged base (2026-11-23; R1's arm; eo/finetune/init.verify_continued, eo/scripts/package_continued.py):
      its continued.json passes verify_extension at 161,665 ids, its weights hash to checkpoint-4000's as pulled from
      g49 (pull.sha256.g49), and the base it names is the map base's extension.json, weights hash included. A model
      built from either recipe overlay and strict-loaded from it equals checkpoint-4000 tensor for tensor; pretrain.py's
      hook (init_from) on it reports a continued base and LoRA r16's 51,014,208 trainable parameters, as on the map
      base (Z15, verify_seg_recipe.py).
                                     | the same model loaded from the map base differs from checkpoint-4000; the hook
                                       given R1's raw checkpoint (no record) refuses it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo/scripts"))
sys.path.insert(0, str(REPO / "eo/experiments/segmentation"))

RESULTS: list = []
STEP40_SELECT = Path("/data/enric/reports/a2_seg/step4/select.json")
CEIL_OUT = Path("/data/enric/reports/a2_seg/step52/ceiling")
SCHED_KEYS = {"num_train_steps", "stop_steps", "num_warmup_steps", "lr_scheduler_kwargs.num_decay_steps",
              "save_steps", "eval_steps"}
TASK_KEYS = {"seg.task", "seg.frames", "seg.in_pinned"}


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


def _diff(a, b):
    from omegaconf import OmegaConf
    fa = _flat(OmegaConf.to_container(a, resolve=False))
    fb = _flat(OmegaConf.to_container(b, resolve=False))
    return {k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}


def _epoch_ok(c, spe: int) -> bool:
    """PASTIS's schedule in epochs of `spe` steps: 50 epochs, warmup one, the last 20 % decaying, eval = save = one."""
    n = int(c.num_train_steps)
    return (n == 50 * spe == int(c.stop_steps) and int(c.num_warmup_steps) == spe and int(c.eval_steps) == spe
            and int(c.save_steps) == spe and int(c.lr_scheduler_kwargs.num_decay_steps) == n // 5
            and c.lr_scheduler_kwargs.decay_type == "linear" and c.lr_scheduler_type == "warmup_stable_decay"
            and bool(c.eval_on_start))


# ---------------------------------------------------------------- Z16
def z16() -> None:
    import numpy as np
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    r41 = _compose("eo_seg/r0_s41_lora16")
    c12 = _compose("eo_seg/s52_cloudsen12")
    d = _diff(r41, c12)
    tr = SD.build(c12, "train")
    va = SD.build(c12, "val")
    spe = tr.steps_per_epoch()
    ok = (d == TASK_KEYS | SCHED_KEYS and c12.seg.task == "cloudsen12" and c12.seg.frames == "single"
          and c12.seg.in_pinned is None and c12.seg.arm == "aligned" and int(c12.max_length) == 396
          and float(c12.learning_rate) == float(r41.learning_rate) and c12.plateau_stop == r41.plateau_stop
          and c12.finetune.regime == "lora" and int(c12.finetune.lora.r) == 16 and int(c12.finetune.lora.alpha) == 32
          and c12.precision == "bf16" and int(c12.total_batch_size) == 32 and _epoch_ok(c12, spe) and spe == 125
          and _epoch_ok(r41, 37))
    record("Z16 overlay", ok, f"differs from r0_s41_lora16 in {sorted(d)}; {c12.num_train_steps} steps = 50 x {spe}; "
           f"warmup {c12.num_warmup_steps}, decay {c12.lr_scheduler_kwargs.num_decay_steps}, eval {c12.eval_steps}, "
           f"save {c12.save_steps}; LR {c12.learning_rate} (the recipe's 3e-3 on the command line, as on PASTIS); "
           f"plateau_stop {dict(c12.plateau_stop)}; arm {c12.seg.arm}, max_length {c12.max_length}")
    dw = _diff(c12, _compose("eo_seg/s52_cloudsen12", ["weight_decay=0.1"]))
    c37 = _compose("eo_seg/s52_cloudsen12", ["eval_steps=37", "save_steps=37"])
    record("Z16 controls: a stray key; PASTIS's 37-step epoch", dw == {"weight_decay"} and not _epoch_ok(c37, spe),
           f"stray key differs in {sorted(dw)}; eval/save 37 against the dataset's {spe}-step epoch flagged")

    # the datasets
    nv = {int(s): len(v) for s, v in tr._sample_views.items()}
    sv = va.views
    ok = (len(tr.samples) == 4000 and len(tr) == 4000 and spe == 125 == int(c12.eval_steps) and len(tr.views) == 288000
          and set(nv.values()) == {72} and len(va) == 4815 == len(sv) and sv.sample_id.nunique() == 535
          and not SEG.crop_order_problems("cloudsen12", sv) and set(sv.k) == {0} and set(sv.split) == {"val"}
          and set(tr.views.split) == {"train"})
    record("Z16 data", ok, f"train {len(tr.samples)} tiles x {sorted(set(nv.values()))} views, {len(tr)} an epoch = "
           f"{spe} steps; val {len(va)} views = {sv.sample_id.nunique()} tiles x 9 in tiling's order, k {sorted(set(sv.k))}")
    lay, V = tr.lay, tr.V
    s2 = V.get_modality("S2L2A")
    off, size = lay.slot("aligned")
    bad = []
    for ds, idx in ((tr, 0), (tr, 1999), (va, 0), (va, 4814)):
        s = ds[idx]
        ids, lab = s["input_ids"], s["labels"]
        keys = set(s) == {"input_ids", "attention_mask", "labels", "position_ids", "modality_ids"}
        lab_at = (lab != -100).nonzero().flatten().tolist()
        p_ids, _ = ds.prompt(ds.view_at(idx))
        good = (keys and len(ids) == 396 and int(ids[0]) == s2.bo_id and int(ids[197]) == s2.eo_id
                and int(ids[198]) == lay.bo("cloudsen12") and bool(((ids[199:395] >= off) & (ids[199:395] < off + size)).all())
                and int(ids[395]) == lay.eo_map and lab_at == list(range(199, 396))
                and bool((lab[199:396] == ids[199:396]).all()) and len(p_ids) == 199
                and int(p_ids[-1]) == lay.bo("cloudsen12") and s["position_ids"].tolist() == list(range(396)))
        if not good:
            bad.append((ds.spec.split, idx))
    record("Z16 sequences", not bad, f"4 sequences (train 0, 1999; val 0, 4814): 396 tokens, five keys, BO_S2L2A .. "
           f"EO_S2L2A, BO_MAP_CLOUDSEN12, 196 codes in [{off}, {off + size}), EO_MAP; labels on 199..395; prompt 199 "
           f"tokens ending at BO_MAP_CLOUDSEN12; positions 0..395; bad {bad}")
    own = all(tr.draw_view(int(s), e) in set(tr._sample_views[int(s)].tolist()) for s in tr.samples[:200] for e in (0, 1))
    moved = float(np.mean([tr.draw_view(int(s), 0) != tr.draw_view(int(s), 1) for s in tr.samples]))
    refused = False
    try:
        SD.SegDataset(SD.SegSpec("cloudsen12", "aligned", "single", "test"))
    except PermissionError:
        refused = True
    record("Z16 draws and the test split", own and moved > 0.9 and refused,
           f"a tile's draw is one of its own views ({own}); {moved:.3f} of tiles draw another view at epoch 1; "
           f"test refused {refused}")
    pinned_refused = False
    try:
        SD.SegDataset(SD.SegSpec("cloudsen12", "aligned", "single", "train", in_pinned=True))
    except ValueError:
        pinned_refused = True
    record("Z16 control: r0_base's in_pinned true on CloudSen12", pinned_refused, f"refused {pinned_refused}")

    # the plateau stop on 125-step epochs
    import verify_seg_recipe as R
    R.E, R.N, R.WARM, R.DECAY = 125, 6250, 125, 1250
    from eo.train.plateau_stop import fires
    sel = json.loads(STEP40_SELECT.read_text())["arms"]["aligned"]["eval_ce"]
    real = {int(k) // 37: float(v) for k, v in sel.items() if 0 < int(k) <= 31 * 37}
    flat = lambda e: real[e] if e <= 31 else real[31]                     # noqa: E731
    lrs, stopped, cb, _ = R.simulate(flat, eval_steps=125, save_steps=125)
    f = cb.fired or {}
    t0, dd = f.get("step"), f.get("decay_steps")
    want = {s: (R.plain_lr(s) if s < (t0 or 0) else R.PEAK * max(0.0, ((t0 or 0) + (dd or 1) - s) / (dd or 1)))
            for s in lrs}
    err = max(abs(lrs[s] - want[s]) for s in lrs)
    first = [e for e in range(4, 32) if fires(real, e, 3, 1e-3)][:1]
    record("Z16 plateau stop on 125-step epochs", f.get("epoch") == 29 and first == [29] and dd == 8 * 125
           and stopped == 29 * 125 + 1000 == 4625 and err < 1e-12,
           f"fired at epoch {f.get('epoch')} (the rule's first {first}), decay {dd} steps, stopped at {stopped}; max |LR "
           f"- intended| {err:.1e} over {len(lrs)} steps")
    lrs2, stopped2, cb2, _ = R.simulate(lambda e: 9.0 * (1 - 0.002) ** e, eval_steps=125, save_steps=125)
    err2 = max(abs(lrs2[s] - R.plain_lr(s)) for s in lrs2)
    record("Z16 control: 0.2 % an epoch never fires", cb2.fired is None and stopped2 is None and len(lrs2) == 6250
           and err2 < 1e-12, f"fired {cb2.fired}; ran {len(lrs2)} steps; max |LR - plain| {err2:.1e}")


# ---------------------------------------------------------------- Z17
def z17() -> None:
    import numpy as np
    import torch
    from eo.data import seg_dataset as SD
    from eo.maps import frozen
    import step52_score as S52
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = frozen.record("aligned")["ceilings"]["cloudsen12"]
    ds = SD.SegDataset(SD.SegSpec("cloudsen12", "aligned", "single", "val"))
    views = ds.views
    codes = np.asarray(ds.maps[views.view.values])[:, None]
    tok = frozen.load("aligned", dev)
    out = CEIL_OUT / "ceiling_cloudsen12_aligned"
    res = S52.score_codes("cloudsen12", "aligned", codes, views, tok, dev,
                          {"ceiling": True, "task": "cloudsen12", "arm": "aligned", "frames": "single", "split": "val"},
                          out)
    cms = np.load(out / "per_patch_cm.npy")
    got = res["result"]["mIoU"]
    record("Z17 ceiling through Step 5.2's scorer", abs(got - ref) < 1e-6 and cms.shape[0] == 535
           and abs(res["per_patch_sum_mIoU"] - got) < 1e-12,
           f"mIoU {got:.7f} against Step 1's frozen {ref:.7f}; {cms.shape[0]} per-tile matrices summing to "
           f"{res['per_patch_sum_mIoU']:.7f}")
    rolled = np.roll(codes, 9, axis=0)                     # tile i gets tile i - 1's nine crops
    with __import__("tempfile").TemporaryDirectory() as td:
        r2 = S52.score_codes("cloudsen12", "aligned", rolled, views, tok, dev, {"control": "rolled"}, Path(td) / "x")
    swapped = views.copy()
    swapped.iloc[[0, 1]] = swapped.iloc[[1, 0]].to_numpy()
    try:
        S52.score_codes("cloudsen12", "aligned", codes, swapped, tok, dev, {"control": "swapped"}, Path("/nonexistent"))
        refused = False
    except RuntimeError:
        refused = True
    record("Z17 controls: the next tile's codes; two crops swapped", abs(r2["result"]["mIoU"] - ref) > 0.05 and refused,
           f"the next tile's codes read {r2['result']['mIoU']:.4f}; swapped crops refused {refused}")


A2SEG = Path("/data/enric/runs/pretrain/a2_seg")
R1_RUN = A2SEG / "r1aligned_n4000_20261122_193801"
R1_BASE = A2SEG / "base/r1aligned_n4000_ckpt4000"
MAP_BASE = A2SEG / "base/b10cltrxnc_ext_map_nc_ckpt33000"
LORA16_TRAINABLE = 51_014_208                  # verify_seg_recipe.py Z15, on the map base


def z18() -> None:
    import torch
    from safetensors.torch import load_file
    from eo.data import map_vocab as MV
    from eo.finetune import init as FI
    from eo.generate.conditional import build_model
    vocab = MV.TOTAL_VOCAB_SIZE
    doc = FI.verify_extension(R1_BASE, vocab)                        # dispatches to verify_continued; hashes the weights
    g49 = dict(ln.split()[::-1] for ln in (R1_RUN / "pull.sha256.g49").read_text().splitlines() if ln.strip())
    mapx = json.loads((MAP_BASE / "extension.json").read_text())
    ok = (doc.get("kind") == "continued" and doc["weights_sha256"] == g49["checkpoint-4000/model.safetensors"]
          and doc["base"]["extension"] == mapx and doc["base"]["weights_sha256"] == mapx["weights_sha256"]
          and doc["run"] == R1_RUN.name and doc["step"] == 4000 and doc["vocab_size"] == vocab)
    record("Z18 R1's record", ok, f"continued base {doc['run']} step {doc['step']} at {doc['vocab_size']}; weights "
           f"{doc['weights_sha256'][:12]} = g49's checkpoint-4000 {g49['checkpoint-4000/model.safetensors'][:12]}; the "
           f"base = the map base's extension.json ({mapx['weights_sha256'][:12]})")
    ck = load_file(str(R1_RUN / "checkpoint-4000/model.safetensors"))
    for cfg_name in ("eo_seg/r0_s41_lora16", "eo_seg/s52_cloudsen12"):
        model, _ = build_model(cfg_name, None, device="cpu", overrides=[f"finetune.init_from={R1_BASE}"])
        FI.strict_load(model, R1_BASE)
        sd = model.state_dict()
        diff = [k for k, v in ck.items() if not torch.equal(sd[k], v)]
        record(f"Z18 {cfg_name} loads checkpoint-4000", not diff and len(ck) == 263,
               f"{len(ck)} tensors, {len(diff)} differ from checkpoint-4000 {diff[:3]}")
        del model, sd
    model, _ = build_model("eo_seg/r0_s41_lora16", None, device="cpu", overrides=[f"finetune.init_from={MAP_BASE}"])
    FI.strict_load(model, MAP_BASE)
    sd = model.state_dict()
    n_ctrl = sum(not torch.equal(sd[k], v) for k, v in ck.items())
    record("Z18 control: the map base differs from checkpoint-4000", n_ctrl > 0,
           f"{n_ctrl} of {len(ck)} tensors differ")
    del model, sd
    model, cfg = build_model("eo_seg/r0_s41_lora16", None, device="cpu", overrides=[f"finetune.init_from={R1_BASE}"])
    rep = FI.init_from(model, cfg)
    record("Z18 pretrain.py's hook on R1's base", rep["init_kind"] == "continued" and rep["regime"] == "lora"
           and rep["trainable_params"] == LORA16_TRAINABLE and rep["weights_sha256"] == doc["weights_sha256"],
           f"kind {rep['init_kind']}, regime {rep['regime']}, trainable {rep['trainable_params']:,} (map base "
           f"{LORA16_TRAINABLE:,}), weights {rep['weights_sha256'][:12]}")
    del model
    model, cfg = build_model("eo_seg/r0_s41_lora16", None, device="cpu",
                             overrides=[f"finetune.init_from={R1_RUN / 'checkpoint-4000'}"])
    try:
        FI.init_from(model, cfg)
        refused = ""
    except RuntimeError as e:
        refused = str(e).splitlines()[0]
    record("Z18 control: the hook given R1's raw checkpoint", bool(refused), f"refused: {refused[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="Z16,Z17")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.time()
    checks = {"Z16": z16, "Z17": z17, "Z18": z18}
    for name in a.only.split(","):
        try:
            checks[name.strip()]()
        except Exception as e:                                      # a crash is a failed check, with its stack
            traceback.print_exc()
            record(f"{name} crashed", False, f"{type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
