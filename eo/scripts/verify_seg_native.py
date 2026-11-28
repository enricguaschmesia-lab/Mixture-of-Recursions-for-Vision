#!/usr/bin/env python
"""Gate for the NATIVE S2 layout (meeting 2026-11-25, option A; docs/notes/downstream/2026-11-25_meeting_2_patches_positions.md
§8.4). `.venv`, CPU (`CUDA_VISIBLE_DEVICES=`). Every check is paired with a control that must fail it.

    python eo/scripts/verify_seg_native.py [--only ZN1,ZN2,ZN3,ZN4,ZN5,ZN6,ZN7] [--out results.json]

  ZN1  Arm A's positions on every fold-4 view, built by SegDataset itself: BO_S2L2A 0; S2 cell (r, c)'s stored code at
       1 + 14 r + c; EO_S2L2A 197; BO_MAP_PASTIS 198; map cell (R, C)'s stored code at 199 + 198 j + 14 (R // 2) + C // 2,
       j = 2 (R % 2) + C % 2, i.e. 198 (j + 1) after the S2 code of the cell it lies in; EO_MAP 989; positions strictly
       increasing and < 990; labels on the 256 codes and EO_MAP only; 324 tokens. Controls: the same codes at the raster
       positions (199 + 16 R + C); the inner grid transposed.
  ZN2  Generation's step_positions equal the codes' positions, and to_raster inverts the layout's order. Control: the
       sequence-order codes read as raster do not equal the stored map.
  ZN3  Variants: prior keeps the map at its full-layout positions with no input; wrong takes the deranged view's S2 at the
       same positions and the own map. Control: wrong's S2 equals the own S2 on no view (the derangement has no fixed
       point), so a check that wrong's S2 is the own would fail.
  ZN4  The existing layouts unchanged: aligned raster (p1 train / val), short raster, c4375_224 nested, aligned shift 99,
       the chain, PASTIS-7 and CloudSen12, each with its variants, build sequences equal to HEAD's seg_dataset.py loaded
       beside it (64 views per case). Control: HEAD's module with one map position moved differs.
  ZN5  The artifact S2L2A_tok128: shape, present on exactly the train and val p1 rows (absent on test and p7), codes in
       the codebook, its metadata, the contract check passed, and the fresh-process re-encode (verify.json) equal bit for
       bit. Control: the test split's native rows are absent, so the dataset refuses them (allow_test=True).
  ZN6  The overlays m2p_native128 and m2p_short differ from r0_s41_lora16 in exactly their keys, build at 324 / 456
       tokens (= required_length), and train R0's 51,014,208 parameters; A and S differ in A's layout keys only. Control:
       against another parent (s43_grid784) the exact-keys test fails.
  ZN7  Arm A through pretrain.py, 2 steps on CPU at fp32: finite losses, a checkpoint, the dataset's line.
"""
from __future__ import annotations

import argparse
import importlib.util
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

from eo.data import map_vocab as MV  # noqa: E402
from eo.data import seg_dataset as SD  # noqa: E402

RESULTS: list = []
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
SEG_ROOT = Path(os.environ.get("ZN_ROOT", "/data/enric/data/geobench2"))   # ZN1-ZN3 may run on a stand-in root
TOK = Path("/data/enric/data/geobench2/pastis/tok")
NATIVE_KEYS = {"seg.arm", "seg.s2_crop", "seg.map_order", "max_length"}
SHORT_KEYS = {"seg.arm", "max_length"}


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


_DS: dict = {}


def native(split: str = "val", **kw) -> SD.SegDataset:
    key = (split, tuple(sorted(kw.items())))
    if key not in _DS:
        _DS[key] = SD.SegDataset(SD.SegSpec("pastis", "short", "p1", split, s2_crop=128, map_order="nested", root=SEG_ROOT,
                                             **kw))
    return _DS[key]


def expected_positions():
    """(S2 positions (64,), map positions by RASTER cell (256,)) of arm A's layout, from the note's table."""
    r, c = np.divmod(np.arange(64), 8)
    s2 = 1 + 14 * r + c
    R, C = np.divmod(np.arange(256), 16)
    j = 2 * (R % 2) + C % 2
    mp = 199 + 198 * j + 14 * (R // 2) + C // 2
    return s2, mp


def layout_problems(ds: SD.SegDataset, seq: dict, view: int, pos_override=None, code_order=None) -> list:
    """Problems of one built sequence against ZN1's table. `pos_override` replaces the map codes' positions and
    `code_order` the raster cell each sequence slot holds (the controls)."""
    p = []
    ids, pos, lab = seq["input_ids"].numpy(), seq["position_ids"].numpy(), seq["labels"].numpy()
    if pos_override is not None:
        pos = pos.copy()
        pos[67:323] = pos_override
    info = ds.V.get_modality("S2L2A")
    s2 = np.asarray(ds.s2[int(ds.in_rows[view][0])]).astype(np.int64) + info.codebook_offset
    mp = np.asarray(ds.maps[view]).astype(np.int64) + ds.map_off
    exp_s2, exp_mp = expected_positions()
    if len(ids) != 324:
        p.append(f"length {len(ids)}")
    if ids[0] != info.bo_id or pos[0] != 0:
        p.append("BO_S2L2A")
    if not np.array_equal(ids[1:65], s2):
        p.append("S2 codes not the stored row in raster order")
    if not np.array_equal(pos[1:65], exp_s2):
        p.append("S2 positions")
    if ids[65] != info.eo_id or pos[65] != 197:
        p.append("EO_S2L2A")
    if ids[66] != ds.task_token or pos[66] != 198:
        p.append("BO_MAP")
    order = ds.seq_order if code_order is None else code_order
    if not np.array_equal(ids[67:323], mp[order]):
        p.append("map codes not the stored cells in the layout's order")
    if not np.array_equal(pos[67:323], exp_mp[order]):
        p.append("map positions")
    R, C = np.divmod(order, 16)
    own = exp_s2[(R // 2) * 8 + C // 2]
    off = pos[67:323] - own
    if not set(np.unique(off).tolist()) <= {198, 396, 594, 792}:
        p.append(f"offsets {sorted(set(np.unique(off).tolist()))[:6]}")
    elif not np.array_equal(off // 198 - 1, 2 * (R % 2) + C % 2):
        p.append("offset j is not the sub-cell")
    if ids[323] != ds.lay.eo_map:
        p.append("EO_MAP id")
    if pos[323] != 989:
        p.append("EO_MAP position")
    if not (np.diff(pos) > 0).all() or pos.max() >= 990:
        p.append("positions not increasing or past 989")
    want = np.full(324, -100)
    want[67:324] = ids[67:324]
    if not np.array_equal(lab, want):
        p.append("labels")
    return p


def zn1() -> None:
    ds = native("val")
    bad, n = 0, 0
    ex = None
    for v in ds.views.view.values:
        s = ds.sequence(int(v))
        pr = layout_problems(ds, s, int(v))
        n += 1
        if pr:
            bad += 1
            ex = ex or (int(v), pr)
    record("ZN1 A's positions on every fold-4 view", bad == 0 and n == 482, f"{n} views, {bad} with problems {ex or ''}")
    v = int(ds.views.view.values[0])
    s = ds.sequence(v)
    R, C = np.divmod(ds.seq_order, 16)
    raster_pos = 199 + 16 * R + C
    pr = layout_problems(ds, s, v, pos_override=raster_pos)
    record("ZN1 control: the codes at raster positions fail", bool(pr), f"{pr[:3]}")
    tr_order = (C * 16 + R)
    s_t = dict(s)
    ids = s["input_ids"].clone()
    mp = np.asarray(ds.maps[v]).astype(np.int64) + ds.map_off
    ids[67:323] = torch.from_numpy(mp[tr_order])
    s_t["input_ids"] = ids
    pr = layout_problems(ds, s_t, v)
    record("ZN1 control: the inner grid transposed fails", bool(pr), f"{pr[:3]}")


def zn2() -> None:
    ds = native("val")
    sp = ds.step_positions().numpy()
    s = ds.sequence(int(ds.views.view.values[0]))
    exp = s["position_ids"][67:323].numpy()
    ok = np.array_equal(sp, exp)
    bad = 0
    for v in ds.views.view.values[:64]:
        seq_codes = ds.sequence(int(v))["input_ids"][67:323].numpy() - ds.map_off
        if not np.array_equal(ds.to_raster(seq_codes), np.asarray(ds.maps[int(v)]).astype(np.int64)):
            bad += 1
    record("ZN2 step_positions = the codes' positions; to_raster inverts the order", ok and bad == 0,
           f"step_positions equal {ok}; {bad} of 64 views fail to invert; first steps {sp[:5].tolist()}, last {sp[-2:].tolist()}")
    v = int(ds.views.view.values[0])
    seq_codes = ds.sequence(v)["input_ids"][67:323].numpy() - ds.map_off
    same = np.array_equal(seq_codes, np.asarray(ds.maps[v]).astype(np.int64))
    record("ZN2 control: the sequence order read as raster is not the map", not same, f"equal {same}")


def zn3() -> None:
    ds = native("val")
    der = ds.derangement()
    bad_p = bad_w = same_s2 = 0
    for v in ds.views.view.values[:96]:
        v = int(v)
        f, pr, w = ds.sequence(v, "full"), ds.sequence(v, "prior"), ds.sequence(v, "wrong")
        lab = (pr["labels"] != -100).nonzero().flatten()
        if not (pr["input_ids"][0] == ds.task_token and int(pr["position_ids"][0]) == 198
                and torch.equal(pr["position_ids"][1:258], f["position_ids"][67:324])
                and torch.equal(pr["input_ids"][1:258], f["input_ids"][67:324]) and len(lab) == 257):
            bad_p += 1
        s2w = np.asarray(ds.s2[int(ds.in_rows[der[v]][0])]).astype(np.int64) + ds.V.get_modality("S2L2A").codebook_offset
        if not (np.array_equal(w["input_ids"][1:65].numpy(), s2w) and torch.equal(w["position_ids"], f["position_ids"])
                and torch.equal(w["input_ids"][65:], f["input_ids"][65:])):
            bad_w += 1
        same_s2 += int(torch.equal(w["input_ids"][1:65], f["input_ids"][1:65]))
    record("ZN3 prior and wrong keep A's positions", bad_p == 0 and bad_w == 0, f"96 views: prior bad {bad_p}, wrong bad {bad_w}")
    record("ZN3 control: wrong's S2 is the own S2 on no view", same_s2 == 0, f"{same_s2} of 96 equal")


def _head_module():
    src = subprocess.run(["git", "-C", str(REPO), "show", "HEAD:eo/data/seg_dataset.py"], capture_output=True,
                         text=True, check=True).stdout
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"seg_dataset_head_{os.getpid()}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("seg_dataset_head", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["seg_dataset_head"] = mod                 # dataclasses look the module up while decorating
    spec.loader.exec_module(mod)
    return mod


def _same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def zn4() -> None:
    H = _head_module()
    cases = [("pastis", "aligned", "p1", "val", {}, ("full", "prior", "wrong")),
             ("pastis", "aligned", "p1", "train", {"in_pinned": True}, ("full",)),
             ("pastis", "short", "p1", "val", {}, ("full", "prior", "wrong")),
             ("pastis", "c4375_224", "p1", "val", {"map_order": "nested"}, ("full", "prior", "wrong")),
             ("pastis", "aligned", "p1", "val", {"map_shift": 99}, ("full", "prior")),
             ("pastis", "aligned", "p1", "val", {"chain": "base"}, ("full", "wrong", "wrong_lulc")),
             ("pastis", "aligned", "p7", "val", {}, ("full", "one_date", "shuffled_dates")),
             ("cloudsen12", "aligned", "single", "val", {}, ("full", "prior", "wrong"))]
    for task, arm, fr, split, kw, variants in cases:
        try:
            a = SD.SegDataset(SD.SegSpec(task, arm, fr, split, **kw), batch=32 if split == "train" else 1)
            b = H.SegDataset(H.SegSpec(task, arm, fr, split, **kw), batch=32 if split == "train" else 1)
        except FileNotFoundError as e:
            record(f"ZN4 {task} {arm} {fr} {split} {kw}", False, f"missing data: {e}")
            continue
        n = bad = 0
        for var in variants:
            for i in range(64):
                v = int(a.view_at(i, 3)) if split == "train" else int(a.views.view.values[i])
                if split == "train":
                    a.set_draw_epoch(3); b.set_draw_epoch(3)
                    sa, sb = a[i], b[i]
                else:
                    sa, sb = a.sequence(v, var), b.sequence(v, var)
                n += 1
                bad += int(not _same(sa, sb))
        ok = bad == 0 and np.array_equal(a.step_positions().numpy(), b.step_positions().numpy()) \
            and a.required == b.required
        record(f"ZN4 {task} {arm} {fr} {split} {kw or ''} unchanged", ok, f"{n} sequences, {bad} differ; variants {variants}")
    # control: HEAD's module with one map position moved
    b = H.SegDataset(H.SegSpec("pastis", "aligned", "p1", "val"))
    a = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p1", "val"))
    orig = b.map_positions
    b.map_positions = lambda bo: (lambda p: torch.cat([p[:5], p[5:6] + 1, p[6:]]))(orig(bo))
    v = int(a.views.view.values[0])
    diff = not _same(a.sequence(v), b.sequence(v))
    record("ZN4 control: one moved map position differs", diff, f"differs {diff}")


def zn5() -> None:
    from eo.geobench import views as V
    idx = V.load("pastis")
    views, inputs = idx["views"], idx["inputs"]
    d = TOK / "S2L2A_tok128"
    t = np.load(d / "tokens.npy", mmap_mode="r")
    pr = np.load(d / "present.npy")
    meta = json.loads((d / "metadata.json").read_text())
    split_of = views.set_index("view").split
    want = (inputs.frame == "p1").values & inputs.view.map(split_of).isin(["train", "val"]).values
    tk = np.asarray(t[pr])
    con = json.loads(Path(meta["contract_check"]).read_text())
    ver = json.loads((d / "verify.json").read_text()) if (d / "verify.json").exists() else {}
    ok = (t.shape == (len(inputs), 64) and t.dtype == np.uint16 and np.array_equal(pr, want) and int(pr.sum()) == 12122
          and tk.max() < 15360 and meta["crop"] == 128 and meta["tokens_per_sample"] == 64
          and meta["present_by_split"]["test"] == 0 and "TITAN X" in meta["gpu"] and con["rule_result"]["pass"]
          and ver.get("pass") is True and ver.get("control_fails") is True)
    record("ZN5 the native artifact", ok, f"shape {t.shape}, present {int(pr.sum())} = train+val p1 {int(want.sum())}: "
           f"{np.array_equal(pr, want)}; by split {meta['present_by_split']}; max code {int(tk.max())}; gpu {meta['gpu']}; "
           f"contract {con['rule_result']['pass']}; re-encode {ver.get('rows_equal')}/{ver.get('n_rows')} rows equal")
    try:
        SD.SegDataset(SD.SegSpec("pastis", "short", "p1", "test", s2_crop=128, map_order="nested"), allow_test=True)
        refused = False
    except RuntimeError as e:
        refused = "lacks rows" in str(e)
    record("ZN5 control: the test split's native rows are absent and refused", refused, f"refused {refused}")


def zn6() -> None:
    from omegaconf import OmegaConf
    base = _flat(OmegaConf.to_container(_compose("eo_seg/r0_s41_lora16"), resolve=False))
    got = {}
    for name, keys, n in (("m2p_native128", NATIVE_KEYS, 324), ("m2p_short", SHORT_KEYS, 456)):
        b = _flat(OmegaConf.to_container(_compose(f"eo_seg/{name}"), resolve=False))
        cfg = _compose(f"eo_seg/{name}", [f"seg.root={SEG_ROOT}"])
        got[name] = b
        dif = {k for k in set(base) | set(b) if base.get(k) != b.get(k)}
        tr, ev = SD.build(cfg, "train"), SD.build(cfg, "eval")
        req = SD.required_length("pastis", "p1", "short", s2_crop=int(cfg.seg.get("s2_crop", 224) or 224))
        ok = dif == keys and int(cfg.max_length) == n == req and tr.required == ev.required == n \
            and len(tr.samples) == 1200 and len(ev.views) == 482 and tr.steps_per_epoch() == 37
        record(f"ZN6 {name}", ok, f"differs from r0_s41_lora16 in {sorted(dif)}; {n} tokens (required {req}); "
               f"train {len(tr.samples)} samples, {tr.steps_per_epoch()} steps an epoch; eval {len(ev.views)} views")
    dif = {k for k in set(got["m2p_short"]) | set(got["m2p_native128"])
           if got["m2p_short"].get(k) != got["m2p_native128"].get(k)}
    record("ZN6 A and S differ in A's layout keys only (A - S isolates scale and nesting)",
           dif == {"seg.s2_crop", "seg.map_order", "max_length"}, f"differ in {sorted(dif)}")
    g784 = _flat(OmegaConf.to_container(_compose("eo_seg/s43_grid784"), resolve=False))
    dif = {k for k in set(g784) | set(got["m2p_native128"]) if g784.get(k) != got["m2p_native128"].get(k)}
    record("ZN6 control: against another parent (s43_grid784) the exact-keys test fails", dif != NATIVE_KEYS,
           f"differs from s43_grid784 in {sorted(dif)}")
    from eo.finetune.init import init_from
    from eo.generate.conditional import build_model
    m, cfg = build_model("eo_seg/m2p_native128", None, device="cpu", overrides=[f"finetune.init_from={MAP_BASE}"])
    rep = init_from(m, cfg)
    record("ZN6 m2p_native128 trains R0's parameters", rep["trainable_params"] == 51_014_208,
           f"trainable {rep['trainable_params']:,} (R0: 51,014,208)")


def zn7() -> None:
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_zn7"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(MAP_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "4", "per_device_train_batch_size": "2", "logging_steps": "1", "dataloader_num_workers": "0",
          "precision": "fp32", "plateau_stop.enable": "false", "learning_rate": "3e-3", "hydra.run.dir": f"{out}/hydra",
          "seg.root": str(SEG_ROOT)}
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name",
           "eo_seg/m2p_native128"] + [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    hist = json.loads(st.read_text())["log_history"] if st.exists() else []
    losses = [h["loss"] for h in hist if "loss" in h]
    line = next((ln for ln in log.splitlines() if "seg train:" in ln), "")
    ok = (r.returncode == 0 and (out / "checkpoint-2").is_dir() and "s2_crop 128" in line and "324 tokens" in line
          and len(losses) == 2 and all(math.isfinite(x) for x in losses))
    record("ZN7 A through pretrain.py", ok, f"exit {r.returncode}; losses {losses}; {time.time() - t0:.0f} s, CPU fp32; "
           f"'{line[-120:]}'; log {out / 'pretrain.log'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"ZN1": zn1, "ZN2": zn2, "ZN3": zn3, "ZN4": zn4, "ZN5": zn5, "ZN6": zn6, "ZN7": zn7}
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
