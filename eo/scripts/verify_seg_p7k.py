#!/usr/bin/env python
"""Gate for A2 Step 7.0c (docs/plans/a2_segmentation.md, Step 7, "7.0c, the date count and a date curriculum on B1's
base"): the date subsets, the wrong_early control, the LoRA continuation and the read. Every check has a control that
fails it. `.venv`; CPU (run it with CUDA_VISIBLE_DEVICES= ).

  Z31a  the overlays eo_seg/s70k_d1..d7 differ from s70_p7 in exactly seg.dates, max_length and
        finetune.continue_from; dates = the latest k; max_length = the layout's length. Control: a stray key.
  Z31b  a k-date sequence (built from the composed config, train and val) is PASTIS-7's with the earliest 7 - k groups
        cut and every position moved down by 198 (7 - k): ids, labels, modality ids, mask; BO_MAP at 198 k.
        Control: the earliest k groups fail it.
  Z31c  the default paths are HEAD's (1adedde) bit for bit: PASTIS-7 full and its four controls, PASTIS-1, train draws.
        Control: HEAD's module with ONE_DATE_GROUP moved differs.
  Z31d  wrong_early: groups 0-2's images are the wrong control's, every date block, groups 3-6 and the map the view's
        own, positions unchanged. Control: the wrong control fails it.
  Z31e  continue_lora on B1's base (CPU): a saved LoRA state loads exactly, frozen tensors stay the base's, the logits
        equal the saved model's. Controls (each must be refused): another base in finetune_init.json, a moved frozen
        tensor, another rank, regime full, no finetune_init.json, an extra key.
  Z31p  pretrain.py on CPU: stage A (s70k_d2, 2 steps) then stage B (s70k_d3, continue_from A's checkpoint-2, LR 0,
        1 step): B's finetune_init.json names A's run, step and weights; B's checkpoint-1 equals A's checkpoint-2.
        Control: the same stage B without continue_from differs from A's checkpoint-2.
  Z31f  the read: B1 - 7.0 reproduces B1's read exactly; call, kstar, wave2 and pair_of_drops on known cases. Control:
        a bar-only call gives another k* on a crafted case.
    python eo/scripts/verify_seg_p7k.py [--only Z31a,Z31b] [--out gate.json]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import shutil
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
HEAD = "1adedde"
A2SEG = Path("/data/enric/runs/pretrain/a2_seg")
B1_BASE = A2SEG / "base/b10cltrxnct_ext_map_nc_ckpt2960"
A2_BASE = A2SEG / "base/b10cltrxnc_ext_map_nc_ckpt33000"
PARENT = "eo_seg/s70_p7"
TMP = Path("/data/enric/reports/a2_seg/step70k/gate_tmp")
N_VIEWS = 10


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
    """Keys whose value differs, or that one side holds and the other does not (a null key is still a key)."""
    return {k for k in set(a) | set(b) if (k in a) != (k in b) or a.get(k) != b.get(k)}


def latest(k: int):
    return list(range(7 - k, 7))


# ------------------------------------------------------------------ Z31a
def z31a() -> None:
    from omegaconf import OmegaConf
    a = _flat(OmegaConf.to_container(_compose(PARENT), resolve=False))
    bad = []
    for k in range(1, 8):
        b = _flat(OmegaConf.to_container(_compose(f"eo_seg/s70k_d{k}"), resolve=False))
        want = {"seg.dates", "finetune.continue_from"} | ({"max_length"} if k < 7 else set())
        d = _diff(a, b)
        dates = None if k == 7 else latest(k)
        req = SD.required_length("pastis", "p7", "aligned", dates=dates)
        if d != want or b["seg.dates"] != dates or int(b["max_length"]) != req or b["finetune.continue_from"] is not None:
            bad.append((k, sorted(d), b["seg.dates"], b["max_length"], req))
    record("Z31a the overlays s70k_d1..d7 = s70_p7 but dates, length and continue_from", not bad,
           f"7 overlays; problems {bad}" if bad else "7 overlays: diff keys, dates = the latest k, max_length = 201 k + 198")
    c = _flat(OmegaConf.to_container(_compose("eo_seg/s70k_d3", ["weight_decay=0.01"]), resolve=False))
    d2 = _diff(a, c)
    record("Z31a control: a stray key", d2 != {"seg.dates", "finetune.continue_from", "max_length"},
           f"differs in {sorted(d2)}")


# ------------------------------------------------------------------ Z31b
def _ds_from(k: int, want: str):
    cfg = _compose(f"eo_seg/s70k_d{k}")
    spec = SD.spec_from_cfg(cfg, want)
    batch = int(cfg.get("total_batch_size") or 1) if want == "train" else 1
    return SD.SegDataset(spec, max_length=int(cfg.max_length), seed=42, batch=batch), cfg


def _cut_equal(s, f, cut_groups: int, keep_first: bool = False) -> bool:
    """s is f with `cut_groups` date groups removed from the front (or, keep_first, from the back of the inputs)."""
    w = 201 * cut_groups
    if keep_first:                               # the control: s holds the EARLIEST groups
        n_in = len(s["input_ids"]) - 198
        fi = torch.cat([f["input_ids"][:n_in], f["input_ids"][-198:]])
        return torch.equal(s["input_ids"], fi)
    ok = all(torch.equal(s[x], f[x][w:]) for x in ("input_ids", "labels", "modality_ids", "attention_mask"))
    return ok and torch.equal(s["position_ids"], f["position_ids"][w:] - 198 * cut_groups)


def z31b() -> None:
    full_val = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "val"))
    full_tr = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "train", in_pinned=True), batch=32)
    bad, n = [], 0
    for k in range(1, 7):
        dv, _ = _ds_from(k, "eval")
        dt, _ = _ds_from(k, "train")
        if dv.spec.dates != tuple(latest(k)) or dt.spec.dates != tuple(latest(k)):
            bad.append((k, "dates", dv.spec.dates))
        for i in range(N_VIEWS):
            v = int(dv.views.view.values[i * 37])
            s, f = dv.sequence(v), full_val.sequence(v)
            ip, pp = dv.prompt(v)
            if not (_cut_equal(s, f, 7 - k) and len(s["input_ids"]) == 201 * k + 198 and int(pp[-1]) == 198 * k):
                bad.append((k, "val", v))
            n += 1
        for i in range(N_VIEWS):
            vt = dt.view_at(i, 0)
            if vt != full_tr.view_at(i, 0) or not _cut_equal(dt.sequence(vt), full_tr.sequence(vt), 7 - k):
                bad.append((k, "train", i))
            n += 1
        if dt.steps_per_epoch() != 37:
            bad.append((k, "steps", dt.steps_per_epoch()))
    record("Z31b k-date sequences = PASTIS-7's latest k, positions moved down", not bad,
           f"{n} sequences (k = 1..6, val and train draws), 37 steps an epoch; problems {bad[:4]}")
    c = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "val", dates=(0, 1, 2)))
    v = int(c.views.view.values[0])
    fails = not _cut_equal(c.sequence(v), full_val.sequence(v), 4)
    record("Z31b control: the earliest three groups", fails, f"the earliest-3 sequence fails the latest-3 comparison: {fails}")


# ------------------------------------------------------------------ Z31c
def _head_module(patch=None):
    src = subprocess.run(["git", "show", f"{HEAD}:eo/data/seg_dataset.py"], cwd=str(REPO), capture_output=True,
                         text=True, check=True).stdout
    if patch:
        src = src.replace(*patch)
    TMP.mkdir(parents=True, exist_ok=True)
    f = TMP / f"seg_dataset_head_{abs(hash(patch))}.py"
    f.write_text(src)
    name = f"seg_dataset_head_{abs(hash(patch))}"
    spec = importlib.util.spec_from_file_location(name, f)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m                        # dataclasses look the module up by name
    spec.loader.exec_module(m)
    return m


def _same_sequences(H) -> tuple:
    bad, n = [], 0
    for frames, variants in (("p7", ("full", "prior", "wrong", "one_date", "shuffled_dates")), ("p1", ("full", "prior", "wrong"))):
        a = SD.SegDataset(SD.SegSpec("pastis", "aligned", frames, "val"))
        b = H.SegDataset(H.SegSpec("pastis", "aligned", frames, "val"))
        for i in range(N_VIEWS):
            v = int(a.views.view.values[i * 41])
            for var in variants:
                x, y = a.sequence(v, var), b.sequence(v, var)
                if not all(torch.equal(x[t], y[t]) for t in x):
                    bad.append((frames, var, v))
                n += 1
        at = SD.SegDataset(SD.SegSpec("pastis", "aligned", frames, "train", in_pinned=True), batch=32)
        bt = H.SegDataset(H.SegSpec("pastis", "aligned", frames, "train", in_pinned=True), batch=32)
        for i in range(N_VIEWS):
            x, y = at[i], bt[i]
            if not all(torch.equal(x[t], y[t]) for t in x):
                bad.append((frames, "train", i))
            n += 1
    return bad, n


def z31c() -> None:
    H = _head_module()
    bad, n = _same_sequences(H)
    record(f"Z31c the default paths equal HEAD's ({HEAD}) seg_dataset", not bad,
           f"{n} sequences (PASTIS-7 full and four controls, PASTIS-1 full and two, train draws); problems {bad[:4]}")
    Hc = _head_module(("ONE_DATE_GROUP = 3", "ONE_DATE_GROUP = 2"))
    badc, _ = _same_sequences(Hc)
    record("Z31c control: HEAD's module with ONE_DATE_GROUP = 2", any(b[1] == "one_date" for b in badc),
           f"{len(badc)} sequences differ, one_date among them: {any(b[1] == 'one_date' for b in badc)}")


# ------------------------------------------------------------------ Z31d
def _early_ok(ds, v, var) -> bool:
    s, f, w = ds.sequence(v, var), ds.sequence(v), ds.sequence(v, "wrong")
    blk = lambda t, j: t[201 * j + 3: 201 * j + 201]          # noqa: E731
    return (all(torch.equal(blk(s["input_ids"], j), blk(w["input_ids"], j)) for j in SD.EARLY_GROUPS)
            and all(torch.equal(blk(s["input_ids"], j), blk(f["input_ids"], j)) for j in range(3, 7))
            and all(torch.equal(s["input_ids"][201 * j: 201 * j + 3], f["input_ids"][201 * j: 201 * j + 3]) for j in range(7))
            and torch.equal(s["input_ids"][7 * 201:], f["input_ids"][7 * 201:])
            and torch.equal(s["position_ids"], f["position_ids"]) and torch.equal(s["labels"], f["labels"])
            and not all(torch.equal(blk(s["input_ids"], j), blk(f["input_ids"], j)) for j in SD.EARLY_GROUPS))


def z31d() -> None:
    ds = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "val"))
    vs = [int(ds.views.view.values[i * 41]) for i in range(N_VIEWS)]
    ok = all(_early_ok(ds, v, "wrong_early") for v in vs)
    record("Z31d wrong_early: early images deranged, dates, late images and map own", ok, f"{len(vs)} views")
    ctl = not any(_early_ok(ds, v, "wrong") for v in vs)
    record("Z31d control: the wrong control fails it", ctl, f"wrong passes on none of {len(vs)} views: {ctl}")


# ------------------------------------------------------------------ Z31e
def _wrapped(overrides=()):
    """B1's base, LoRA-wrapped by init_from, on CPU; with the s70k_d1 config (+ overrides)."""
    from eo.finetune.init import init_from
    from eo.generate.conditional import build_model
    ov = [f"finetune.init_from={B1_BASE}", "seg.root=/data/enric/data/geobench2"] + list(overrides)
    model, cfg = build_model("eo_seg/s70k_d1", None, device="cpu", overrides=ov)
    rep = init_from(model, cfg)
    return model, cfg, rep


def _batch():
    ds = SD.SegDataset(SD.SegSpec("pastis", "aligned", "p7", "val", dates=(6,)))
    ss = [ds.sequence(int(ds.views.view.values[i])) for i in (0, 100)]
    return {k: torch.stack([s[k] for s in ss]) for k in ("input_ids", "attention_mask", "position_ids")}


def _logits(model, b):
    model.eval()
    with torch.no_grad():
        return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"]).logits


def _save_prev(state, report, run: str, step: int = 7, edit=None) -> Path:
    from safetensors.torch import save_file
    d = TMP / run / f"checkpoint-{step}"
    if d.parent.exists():
        shutil.rmtree(d.parent)
    d.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in state.items()}, str(d / "model.safetensors"))
    rep = copy.deepcopy(report)
    if edit:
        edit(rep)
    if rep is not None:
        (d.parent / "finetune_init.json").write_text(json.dumps(rep, default=str))
    (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    return d


def z31e() -> None:
    torch.manual_seed(0)
    A, cfg, rep = _wrapped()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in A.named_parameters():
            if p.requires_grad:
                p.add_(0.01 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    state = {k: v.detach().clone() for k, v in A.state_dict().items()}
    b = _batch()
    la = _logits(A, b)
    ck = _save_prev(state, json.loads(json.dumps(rep, default=str)), "prev_ok")
    Bm, _, repb = _wrapped([f"finetune.continue_from={ck}"])
    sb = Bm.state_dict()
    same = all(torch.equal(sb[k], state[k]) for k in state)
    lb = _logits(Bm, b)
    c = repb.get("continued") or {}
    ok = (same and torch.equal(la, lb) and c.get("run") == "prev_ok" and c.get("step") == 7
          and c.get("frozen_tensors_equal_base", 0) > 200 and c.get("trainable_tensors_loaded", 0) > 100
          and c.get("max_abs_change_from_fresh", 0) > 0)
    record("Z31e continue_lora loads a LoRA save of B1's base exactly", ok,
           f"state equal {same}; logits max |d| {float((la - lb).abs().max()):.2e}; frozen checked "
           f"{c.get('frozen_tensors_equal_base')}, trainable loaded {c.get('trainable_tensors_loaded')}, "
           f"max change from fresh {c.get('max_abs_change_from_fresh', 0):.3g}")
    del Bm
    a2sha = json.loads((A2_BASE / "extension.json").read_text())["weights_sha256"]
    frozen_name = next(n for n, p in A.named_parameters() if not p.requires_grad and n.endswith("norm.weight"))
    moved = {k: (v + 1e-3 if k == frozen_name else v) for k, v in state.items()}
    extra = dict(state, **{"model.extra_key": torch.zeros(1)})
    cases = {
        "another base": (_save_prev(state, rep, "ctl_base", edit=lambda r: r.update(weights_sha256=a2sha)), [], "another base"),
        "a moved frozen tensor": (_save_prev(moved, rep, "ctl_moved"), [], "frozen tensors differ"),
        "another rank": (_save_prev(state, rep, "ctl_rank", edit=lambda r: r["lora"].update(rank=64)), [], "LoRA rank"),
        "regime full": (ck, ["finetune.regime=full"], "continues a LoRA fine-tune"),
        "no finetune_init.json": (_save_prev(state, None, "ctl_noinit"), [], "missing"),
        "an extra key": (_save_prev(extra, rep, "ctl_extra"), [], "keys differ"),
    }
    for name, (path, ov, needle) in cases.items():
        try:
            _wrapped([f"finetune.continue_from={path}"] + ov)
            record(f"Z31e control: {name}", False, "NOT refused")
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            record(f"Z31e control: {name}", needle in str(e), f"refused: {str(e)[:110]}")
    shutil.rmtree(TMP, ignore_errors=True)


# ------------------------------------------------------------------ Z31p
def _run(tag: str, overlay: str, extra: dict) -> dict:
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z31p_{tag}"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(B1_BASE), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "2", "per_device_train_batch_size": "1", "logging_steps": "1",
          "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false", "learning_rate": "3e-3",
          "seed": "42", "hydra.run.dir": f"{out}/hydra"}          # Z29p's and Z30p's overrides
    ov.update(extra)
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", overlay] + \
        [f"{k}={v}" for k, v in ov.items()]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=7200)
    out.mkdir(parents=True, exist_ok=True)
    log = r.stdout + r.stderr
    (out / "pretrain.log").write_text(log)
    st = out / "trainer_state.json"
    hist = json.loads(st.read_text())["log_history"] if st.exists() else []
    fi = out / "finetune_init.json"
    return {"rc": r.returncode, "out": out, "seconds": time.time() - t0,
            "losses": [h["loss"] for h in hist if "loss" in h],
            "line": next((ln for ln in log.splitlines() if ln.startswith("seg train:")), ""),
            "init": json.loads(fi.read_text()) if fi.exists() else {}}


def _weights(d: Path) -> dict:
    from eo.finetune.init import _load_state, _weights as wf
    return _load_state(wf(d))


def z31p() -> None:
    from eo.finetune.init import _sha256, _weights as wf
    b1_sha = json.loads((B1_BASE / "extension.json").read_text())["weights_sha256"]
    A = _run("A", "eo_seg/s70k_d2", {})
    cka = A["out"] / "checkpoint-2"
    okA = (A["rc"] == 0 and cka.is_dir() and "dates (5, 6)" in A["line"] and "600 tokens" in A["line"]
           and len(A["losses"]) == 2 and all(math.isfinite(x) for x in A["losses"])
           and A["init"].get("weights_sha256") == b1_sha and "continued" not in A["init"])
    record("Z31p stage A: s70k_d2 through pretrain.py from B1's base", okA,
           f"exit {A['rc']}; losses {A['losses']}; {A['seconds']:.0f} s; '{A['line'][-70:]}'; log {A['out'] / 'pretrain.log'}")
    lr0 = {"num_train_steps": "1", "stop_steps": "1", "save_steps": "1", "learning_rate": "0.0"}
    Bc = _run("B", "eo_seg/s70k_d3", dict(lr0, **{"finetune.continue_from": str(cka)}))
    ckb = Bc["out"] / "checkpoint-1"
    c = Bc["init"].get("continued") or {}
    wa = _weights(cka) if cka.is_dir() else {}
    wb = _weights(ckb) if ckb.is_dir() else {}
    same = bool(wa) and set(wa) == set(wb) and all(torch.equal(wa[k], wb[k]) for k in wa)
    okB = (Bc["rc"] == 0 and "dates (4, 5, 6)" in Bc["line"] and c.get("run") == A["out"].name and c.get("step") == 2
           and c.get("weights_sha256") == (_sha256(wf(cka)) if cka.is_dir() else None) and same
           and Bc["init"].get("weights_sha256") == b1_sha)
    record("Z31p stage B: s70k_d3 continued from A's checkpoint-2 at LR 0", okB,
           f"exit {Bc['rc']}; continued {c.get('run')} step {c.get('step')} sha {str(c.get('weights_sha256'))[:12]}; "
           f"checkpoint-1 equals A's checkpoint-2: {same}; '{Bc['line'][-60:]}'")
    C = _run("C", "eo_seg/s70k_d3", lr0)
    wc = _weights(C["out"] / "checkpoint-1") if (C["out"] / "checkpoint-1").is_dir() else {}
    differs = bool(wc) and any(not torch.equal(wa[k], wc[k]) for k in wa if k in wc)
    record("Z31p control: stage B without continue_from", C["rc"] == 0 and differs and "continued" not in C["init"],
           f"exit {C['rc']}; checkpoint-1 differs from A's checkpoint-2: {differs}")


# ------------------------------------------------------------------ Z31f
def z31f() -> None:
    import step70k_read as K
    import step70_read as S70
    rb = json.loads((Path("/data/enric/reports/a2_seg/step70b1/read.json")).read_text())["b1_minus_70"]
    b1, r70 = K.read_one(K.B1_RUN), K.read_one(K.R70_RUN, base=K.A2_BASE_SHA)
    p = K.pair_runs(b1, r70)
    rep = all(abs(p[x] - rb[x]) < 1e-12 for x in ("delta", "lo95", "hi95"))
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    calls = [K.call(G(1.3, 0.1, 2)), K.call(G(1.3, -0.1, 2)), K.call(G(-1.3, -2, -0.1)), K.call(G(1.19, 0.5, 2))]
    gn, lv = G(3, 1, 5), G(0.5, -0.5, 1.5)
    ks = [K.kstar({1: gn, 2: gn, 4: lv}), K.kstar({1: gn, 2: lv, 4: lv}), K.kstar({1: gn, 2: gn, 4: gn, 5: gn, 6: gn}),
          K.kstar({1: lv, 2: gn})]
    wv = [K.wave2({1: gn, 2: gn, 4: gn}), K.wave2({1: gn, 2: gn, 4: lv}), K.wave2({1: gn, 2: lv, 4: lv})]
    full, shuf = S70.cm_of(K.gen_dir(K.B1_RUN) / "full_k8"), S70.cm_of(K.gen_dir(K.B1_RUN) / "shuffled_dates_k8")
    z = K.pair_of_drops(full, shuf, full, shuf)
    one = K.pair_of_drops(full, shuf, full, full)
    ref = S70.pair(full, shuf)
    pod = (z["delta"] == 0 and z["lo95"] == 0 and z["hi95"] == 0
           and all(abs(one[x] - ref[x]) < 1e-12 for x in ("delta", "lo95", "hi95")))
    ok = (rep and b1["valid"] and r70["valid"] and calls == ["gains", "level", "costs", "level"] and ks == [4, 2, 7, 1]
          and wv == [[5, 6], [3], []] and pod)
    record("Z31f the read path and its rules", ok,
           f"B1 - 7.0 {p['delta']:+.4f} [{p['lo95']:+.4f}, {p['hi95']:+.4f}] = B1's read: {rep}; calls {calls}; k* {ks}; "
           f"wave2 {wv}; pair_of_drops self 0 and one-sided = pair: {pod}")
    crafted = {1: gn, 2: G(1.5, -0.2, 3.0), 4: lv}               # R_2 over the bar but its interval reaches 0
    bar_only = lambda R: next((k for k in sorted(R) if R[k]["delta"] < K.BAR), 7)     # noqa: E731
    record("Z31f control: a bar-only rule", bar_only(crafted) != K.kstar(crafted),
           f"bar-only k* {bar_only(crafted)} against the rule's {K.kstar(crafted)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z31a": z31a, "Z31b": z31b, "Z31c": z31c, "Z31d": z31d, "Z31e": z31e, "Z31f": z31f, "Z31p": z31p}
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
