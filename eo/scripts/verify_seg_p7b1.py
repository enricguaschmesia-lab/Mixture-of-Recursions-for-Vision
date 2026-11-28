#!/usr/bin/env python
"""A2 Step 7.0b's B1 gate (docs/plans/a2_segmentation.md, Step 7, "7.0b", B1; 2026-11-26): PASTIS-7 on the nc temporal
pilot. B1 is 7.0's cell with only the base changed, so this gate holds the base, the cell's one change, the read and the
run's init; 7.0's gate (verify_seg_p7.py, Z29a-c) holds the overlay and its layout, which B1 does not touch. Every check
ships with a control that must fail it. `.venv`, CPU.

    CUDA_VISIBLE_DEVICES= python eo/scripts/verify_seg_p7b1.py [--only Z30a,Z30b,Z30c,Z30d,Z30p] [--out results.json]

  Z30a  B1's base record: its extension.json passed identity (<= 1e-4 on every base id) with both controls fired, at
        A2's map layout, extension settings, arm, token set and identity batch; its weights hash to the record; its
        source is the pilot's checkpoint-2960 (SHA-256 as the pilot's SHA256SUMS and as hashed now, trainer state at
        step 2960); the pilot continued (variant continue, every parameter, no LoRA) from the very weights A2's base
        extends (ff15419b..., hashed now)
                                     | A2's base checked as B1's: its source is not the pilot's
  Z30b  B1's weights: every body tensor and base rows 0..81,295 equal the pilot's checkpoint-2960 bit for bit; the
        80,369 map rows are A2's rule (mean + 0.01 std N(0,1), the seed-0 draw) recomputed here from the pilot's table;
        the same draw rebuilds A2's map rows from A2's source table: one rule on both bases. Beside it, deciding
        nothing: how far the pilot moved the date rows PASTIS-7 uses, against A2's base
                                     | A2's base checked against the pilot: its body differs; the rule at seed 1
  Z30c  B1's cell script is 7.0's with exactly the base (INIT), its SHA-256 block (0b) and the output root changed, its
        0b hashes are the local base's; the fine-tune loader (verify_extension at the config's vocabulary, hash
        included) accepts B1's base
                                     | 7.0's cell script checked as B1's: its base is A2's; a copy of B1's record with
                                       the identity marked failed: refused
  Z30d  The read's path (step70b1_read): 7.0's maps against themselves read Delta 0 [0, 0] and band (c); 7.0's maps
        against R0's PASTIS-1 maps reproduce 7.0's Delta(p7 - p1) from its read.json to 1e-9; the runs' bases resolve
        by their finetune_init.json (7.0's to A2)
                                     | that known gap, +12.39, read by the same verdict code: a gain (band b), not (c)
  Z30p  s70_p7 through pretrain.py, 2 steps on CPU at fp32 from B1's base: PASTIS-7 at 1,605 tokens, finite losses, a
        save, finetune_init.json naming B1's weights
                                     | the same 2 steps from A2's base: Z29p's losses (gate_p.json) to 1e-4, unequal to
                                       B1's, and finetune_init.json naming A2's weights
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eo" / "scripts"))
sys.path.insert(0, str(REPO / "eo" / "experiments" / "segmentation"))

RESULTS: list = []
A2SEG = Path("/data/enric/runs/pretrain/a2_seg")
B1_BASE = A2SEG / "base/b10cltrxnct_ext_map_nc_ckpt2960"
A2_BASE = A2SEG / "base/b10cltrxnc_ext_map_nc_ckpt33000"
PILOT = Path("/data/enric/runs/pretrain/temporal/vanilla10cltrxnct_20261117_111531")
PILOT_CK = PILOT / "checkpoint-2960"
TBASE = Path("/data/enric/runs/pretrain/temporal/base/b10cltrxnc_ext_temporal_nc_ckpt33000")
KIT, KIT70 = Path("/data/enric/rcp_a2s4/s70b1"), Path("/data/enric/rcp_a2s4/s70")
CELL, CELL70 = KIT / "jobs/p7b1_cell.sh", KIT70 / "jobs/p7_cell.sh"
OVERLAY = "eo_seg/s70_p7"
V_BASE, N_MAP, VOCAB, P7_LEN = 81296, 80369, 161665, 1605
EMB = "model.embed_tokens.weight"
GATE_P = Path("/data/enric/reports/a2_seg/step70/gate_p.json")
SEGROOT = Path("/data/enric/data/geobench2")


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sums(p: Path) -> dict:
    return {ln.split()[1]: ln.split()[0] for ln in p.read_text().splitlines() if ln.strip()}


# ---------------------------------------------------------------- Z30a
_SHA: dict = {}


def _sha(p: Path) -> str:
    if p not in _SHA:
        _SHA[p] = sha256(p)
    return _SHA[p]


def lineage_problems(base: Path) -> list:
    e = json.loads((base / "extension.json").read_text())
    a2 = json.loads((A2_BASE / "extension.json").read_text())
    probs = []
    if not (e["identity"]["passed"] and e["identity"]["max_abs_diff"] <= 1e-4 and e["identity"]["ids"] == V_BASE):
        probs.append(f"identity {e['identity']['max_abs_diff']:.2e} over {e['identity']['ids']} ids")
    if not (e["control_insert_before_pad"]["fired"] and e["control_untied_resize"]["fired"]):
        probs.append("a control did not fire")
    for k in ("layout", "map_vocab", "extension", "arm_config", "token_set", "arm_overrides", "tied", "weights_file"):
        if e.get(k) != a2.get(k):
            probs.append(f"{k} differs from A2's base")
    if e["identity"]["batch_rows"] != a2["identity"]["batch_rows"]:
        probs.append("another identity batch")
    if _sha(base / e["weights_file"]) != e["weights_sha256"]:
        probs.append("the weights do not hash to the record")
    pilot_sha = sums(PILOT_CK / "SHA256SUMS")["model.safetensors"]
    if _sha(PILOT_CK / "model.safetensors") != pilot_sha:
        probs.append("the pilot's checkpoint-2960 no longer hashes to its SHA256SUMS")
    if e["source_weights_sha256"] != pilot_sha:
        probs.append(f"source {e['source_weights_sha256'][:12]} is not the pilot's checkpoint-2960 ({pilot_sha[:12]})")
    st = json.loads((PILOT_CK / "trainer_state.json").read_text())
    if int(st["global_step"]) != 2960:
        probs.append(f"the pilot's trainer state is at step {st['global_step']}")
    pi = json.loads((PILOT / "pretrain_init.json").read_text())
    if pi.get("variant") != "continue" or pi.get("freeze_body") or pi.get("lora") is not None \
            or pi.get("trainable_params") != pi.get("total_params"):
        probs.append("the pilot was not a full continuation")
    if pi.get("conversion_weights_sha256") != a2["source_weights_sha256"]:
        probs.append("the pilot did not continue from A2's source weights")
    if _sha(TBASE / "model.safetensors") != a2["source_weights_sha256"]:
        probs.append("A2's source temporal base no longer hashes to A2's record")
    return probs


def z30a() -> None:
    probs = lineage_problems(B1_BASE)
    e = json.loads((B1_BASE / "extension.json").read_text())
    record("Z30a B1's base record and lineage", not probs,
           f"identity {e['identity']['max_abs_diff']:.2e} (tol 1e-4), control 1 {e['control_insert_before_pad']['max_abs_diff']:.3g}; "
           f"source = the pilot's checkpoint-2960 {e['source_weights_sha256'][:12]}; the pilot continued from "
           f"{json.loads((PILOT / 'pretrain_init.json').read_text())['conversion_weights_sha256'][:12]} = A2's source; "
           f"weights {e['weights_sha256'][:12]}" + (f"; problems: {probs}" if probs else ""))
    c = lineage_problems(A2_BASE)
    record("Z30a control: A2's base checked as B1's", any("source" in p for p in c), f"flagged: {c[:2]}")


# ---------------------------------------------------------------- Z30b
def rule_rows(old: torch.Tensor, seed: int = 0, noise: float = 0.01) -> torch.Tensor:
    """eo/finetune/extend.extend_vocab's new rows, recomputed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mean = old.float().mean(0, keepdim=True)
    std = old.float().std(0, keepdim=True)
    return (mean + noise * std * torch.randn(N_MAP, old.shape[1], generator=g)).to(old.dtype)


def weight_problems(base: dict, src: dict, rows: torch.Tensor) -> tuple:
    probs = []
    if set(base) != set(src):
        probs.append(f"tensor names differ: {sorted(set(base) ^ set(src))[:3]}")
    body = [k for k in base if k != EMB and k in src]
    diff_body = [k for k in body if not torch.equal(base[k], src[k])]
    if diff_body:
        probs.append(f"{len(diff_body)} of {len(body)} body tensors differ (e.g. {diff_body[0]})")
    t = base[EMB]
    if t.shape != (VOCAB, src[EMB].shape[1]):
        probs.append(f"table {tuple(t.shape)}")
        return probs, float("nan"), len(body)
    if not torch.equal(t[:V_BASE], src[EMB]):
        probs.append("base rows differ from the source's")
    dmax = float((t[V_BASE:] - rows).abs().max())
    if dmax > 1e-6:
        probs.append(f"map rows off the rule by {dmax:.3g}")
    return probs, dmax, len(body)


def date_rows_beside(b1: dict, a2: dict) -> dict:
    """How far the pilot moved the date rows PASTIS-7 uses (deciding nothing): their spread and their change."""
    import pandas as pd
    from eo.data import eo_vocab_nc as EVNC
    from eo.data import seg_dataset as SD
    from eo.data import temporal_vocab_nc as TVNC
    views = pd.read_parquet(SEGROOT / "pastis/tok/views.parquet")
    keep = set(views.loc[views.split.isin(["train", "val"]), "view"])          # never the test fold's views
    inp = pd.read_parquet(SEGROOT / "pastis/tok/inputs.parquet")
    inp = inp[inp.frame.str.startswith("p7_") & inp.view.isin(keep)]
    dates = sorted(set(int(d) for d in inp.date))
    bins = sorted(set(SD.day_bin(d) for d in dates))
    ids = [TVNC.DAY_IDS[b] for b in bins]
    n_pre = EVNC.TOTAL_VOCAB_SIZE                     # 81,188: the pretraining rows (codebooks, BO/EO, PAD)
    out = {"pastis_p7_dates_train_val": len(dates), "day_bins_used": len(bins), "of_bins": len(TVNC.DAY_IDS)}
    for name, sd in (("a2", a2), ("b1", b1)):
        e = sd[EMB].float()
        d = e[ids]
        n = len(ids)
        out[f"{name}_day_rows_mean_pairwise_over_mean_norm"] = float(torch.cdist(d, d).sum() / (n * (n - 1))
                                                                     / d.norm(dim=1).mean())
        out[f"{name}_pretraining_rows_mean_norm"] = float(e[:n_pre].norm(dim=1).mean())
        out[f"{name}_day_rows_mean_norm"] = float(d.norm(dim=1).mean())
    da, db = a2[EMB].float(), b1[EMB].float()
    out["day_rows_change_over_a2_norm"] = float((db[ids] - da[ids]).norm(dim=1).mean() / da[ids].norm(dim=1).mean())
    out["pretraining_rows_change_over_a2_norm"] = float((db[:n_pre] - da[:n_pre]).norm(dim=1).mean()
                                                        / da[:n_pre].norm(dim=1).mean())
    return out


def z30b() -> None:
    from safetensors.torch import load_file
    b1, pk = load_file(str(B1_BASE / "model.safetensors")), load_file(str(PILOT_CK / "model.safetensors"))
    probs, dmax, nbody = weight_problems(b1, pk, rule_rows(pk[EMB]))
    a2, tb = load_file(str(A2_BASE / "model.safetensors")), load_file(str(TBASE / "model.safetensors"))
    probs_a2, dmax_a2, _ = weight_problems(a2, tb, rule_rows(tb[EMB]))
    record("Z30b B1's weights: the pilot's, plus A2's rule; A2's base by the same rule", not probs and not probs_a2,
           f"{nbody} body tensors and rows 0..{V_BASE - 1} equal the pilot's; B1's map rows off the recomputed rule by "
           f"{dmax:.2g}, A2's by {dmax_a2:.2g} (tol 1e-6)" + (f"; problems: {probs + probs_a2}" if probs or probs_a2 else ""))
    c1, _, _ = weight_problems(a2, pk, rule_rows(pk[EMB]))
    record("Z30b control: A2's base checked against the pilot", any("body" in p for p in c1), f"flagged: {c1[:2]}")
    c2, d2, _ = weight_problems(b1, pk, rule_rows(pk[EMB], seed=1))
    record("Z30b control: the rule at seed 1", any("rule" in p for p in c2), f"off by {d2:.3g}")
    try:
        bes = date_rows_beside(b1, a2)
        record("Z30b beside (deciding nothing): the date rows PASTIS-7 uses", True, json.dumps(bes))
    except Exception as e:      # beside items never fail the gate
        print(f"[note] Z30b beside not computed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- Z30c
def _strip(text: str, drop_b1_block: bool) -> list:
    out, skip = [], False
    for ln in text.splitlines():
        if ln.startswith("# --- B1 0b"):
            skip = drop_b1_block
            continue
        if ln.startswith("# --- end B1 0b"):
            skip = False
            continue
        if skip or ln.lstrip().startswith("#"):
            continue
        out.append(ln)
    return out


def cell_problems(cell: Path) -> list:
    t, t70 = cell.read_text(), CELL70.read_text()
    probs = []
    if "# --- B1 0b" not in t:
        probs.append("no base hash block (0b)")
    a, b = _strip(t70, True), _strip(t, True)
    changed = [(x, y) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
               if tag != "equal" for x, y in [("\n".join(a[i1:i2]), "\n".join(b[j1:j2]))]]
    want = {("INIT=$R/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000",
             f"INIT=$R/runs/pretrain/a2_seg/base/{B1_BASE.name}"),
            ("OUT=$R/runs/a2_seg_eval/p7/$RUN_ID", "OUT=$R/runs/a2_seg_eval/p7b1/$RUN_ID")}
    if set(changed) != want:
        probs.append(f"changes other than INIT and OUT: {[c for c in changed if c not in want][:2]}")
    m = re.search(r"^# --- B1 0b.*?^# --- end B1 0b", t, flags=re.M | re.S)
    block = re.findall(r"^([0-9a-f]{64})  (\S+)$", m.group(0), flags=re.M) if m else []
    local = {f: _sha(B1_BASE / f) for f in ("model.safetensors", "extension.json", "config.json", "generation_config.json")}
    if dict((f, h) for h, f in block) != local:
        probs.append(f"its 0b hashes are not the local base's ({len(block)} lines in the block)")
    return probs


def z30c() -> None:
    from omegaconf import OmegaConf
    from eo.finetune.init import verify_extension
    probs = cell_problems(CELL)
    record("Z30c B1's cell = 7.0's but the base, its hash block and the output root", not probs,
           f"{CELL} against {CELL70}" + (f"; problems: {probs}" if probs else ""))
    c = cell_problems(CELL70)
    record("Z30c control: 7.0's cell checked as B1's", bool(c), f"flagged: {c[:2]}")
    cfg = _compose(OVERLAY)
    vs = int(OmegaConf.select(cfg, "model_config.vocab_size"))
    try:
        doc = verify_extension(B1_BASE, vs, check_hash=True)
        ok, det = doc["weights_sha256"] == _sha(B1_BASE / "model.safetensors"), f"accepted at {vs} ids"
    except Exception as e:
        ok, det = False, f"refused: {e}"
    record("Z30c the fine-tune loader accepts B1's base", ok, det)
    with tempfile.TemporaryDirectory(dir="/data/enric/tmp" if Path("/data/enric/tmp").is_dir() else None) as td:
        broken = Path(td) / B1_BASE.name
        broken.mkdir()
        for f in ("model.safetensors", "config.json", "generation_config.json"):
            os.symlink(B1_BASE / f, broken / f)
        e = json.loads((B1_BASE / "extension.json").read_text())
        e["identity"]["passed"] = False
        (broken / "extension.json").write_text(json.dumps(e))
        try:
            verify_extension(broken, vs, check_hash=True)
            refused, why = False, "accepted"
        except Exception as ex:
            refused, why = True, str(ex)[:120]
    record("Z30c control: B1's record with the identity marked failed", refused, why)


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


# ---------------------------------------------------------------- Z30d
def z30d() -> None:
    import step70_read as S70
    import step70b1_read as RB
    d70 = S70.GEN / RB.P70_RUN / "full_k8"
    c70 = S70.cm_of(d70)
    self_pair = RB.S70.pair(c70, c70)
    band_self = RB.verdict_of(25.0, self_pair)
    r70 = json.loads((S70.OUT / "read.json").read_text())
    c1 = S70.cm_of(S70.P1_GEN / S70.P1_RUN / "full_k8")
    p1_pair = RB.S70.pair(c70, c1)
    ref = r70["p7_minus_p1"]
    rep = all(abs(p1_pair[k] - ref[k]) <= 1e-9 for k in ("x", "y", "delta", "lo95", "hi95"))
    b70 = RB.base_of(RB.P70_RUN)
    ok = (self_pair["delta"] == 0 and self_pair["lo95"] == 0 and self_pair["hi95"] == 0 and band_self == "c" and rep
          and b70 == "A2")
    record("Z30d the read's path", ok, f"7.0 against itself {self_pair['delta']:+.2f} [{self_pair['lo95']:+.2f}, "
           f"{self_pair['hi95']:+.2f}] -> band {band_self}; 7.0 - p1 {p1_pair['delta']:+.4f} [{p1_pair['lo95']:+.4f}, "
           f"{p1_pair['hi95']:+.4f}] against read.json's {ref['delta']:+.4f} [{ref['lo95']:+.4f}, {ref['hi95']:+.4f}]; "
           f"7.0's base by finetune_init.json: {b70}; gain threshold {RB.GAIN}")
    band_known = RB.verdict_of(r70["p7"], p1_pair)
    record("Z30d control: a known gap read by the same verdict code", band_known in ("a", "b"),
           f"+{p1_pair['delta']:.2f} -> band {band_known}")


# ---------------------------------------------------------------- Z30p
def _two_steps(base: Path, tag: str) -> dict:
    import subprocess
    from util.env import load_dotenv
    load_dotenv()
    save = os.environ.get("MOR_SAVE_DIR", "")
    if not save.startswith("/data/"):
        raise RuntimeError(f"MOR_SAVE_DIR {save!r} is not under /data")
    rel = f"gate_seg/{time.strftime('%Y%m%d_%H%M%S')}_z30p_{tag}"
    out = Path(save) / "pretrain" / rel
    env = dict(os.environ, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
    ov = {"name": out.name, "output_dir": rel, "wandb_run_name": out.name, "wandb": "false",
          "finetune.init_from": str(base), "num_train_steps": "2", "stop_steps": "2", "num_warmup_steps": "0",
          "lr_scheduler_kwargs.num_decay_steps": "0", "save_steps": "2", "eval_on_start": "false", "eval_steps": "1000",
          "total_batch_size": "2", "per_device_train_batch_size": "1", "logging_steps": "1",
          "dataloader_num_workers": "0", "precision": "fp32", "plateau_stop.enable": "false", "learning_rate": "3e-3",
          "seed": "42", "hydra.run.dir": f"{out}/hydra"}          # Z29p's overrides, the base aside
    cmd = [sys.executable, "pretrain.py", "--config-path", "conf/pretrain_vision", "--config-name", OVERLAY] + \
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
            "init_sha": json.loads(fi.read_text()).get("weights_sha256") if fi.exists() else None,
            "saved": (out / "checkpoint-2").is_dir()}


def z30p() -> None:
    b1_sha = json.loads((B1_BASE / "extension.json").read_text())["weights_sha256"]
    a2_sha = json.loads((A2_BASE / "extension.json").read_text())["weights_sha256"]
    r = _two_steps(B1_BASE, "b1")
    ok = (r["rc"] == 0 and r["saved"] and "pastis p7 arm aligned" in r["line"] and f"{P7_LEN} tokens" in r["line"]
          and len(r["losses"]) == 2 and all(math.isfinite(x) for x in r["losses"]) and r["init_sha"] == b1_sha)
    record("Z30p s70_p7 through pretrain.py from B1's base", ok,
           f"exit {r['rc']}; losses {r['losses']}; init {str(r['init_sha'])[:12]} (B1 {b1_sha[:12]}); "
           f"{r['seconds']:.0f} s, CPU fp32; '{r['line'][-80:]}'; log {r['out'] / 'pretrain.log'}")
    c = _two_steps(A2_BASE, "a2ctrl")
    z29 = [float(x) for x in re.findall(r"losses \[([0-9.]+), ([0-9.]+)\]", json.loads(GATE_P.read_text())[0]["detail"])[0]]
    same_as_z29 = len(c["losses"]) == 2 and all(abs(x - y) <= 1e-4 for x, y in zip(c["losses"], z29))
    differs = len(r["losses"]) == 2 and any(abs(x - y) > 1e-3 for x, y in zip(c["losses"], r["losses"]))
    record("Z30p control: the same 2 steps from A2's base", same_as_z29 and differs and c["init_sha"] == a2_sha,
           f"losses {c['losses']} against Z29p's {z29} (equal: {same_as_z29}) and B1's {r['losses']} (unequal: "
           f"{differs}); init {str(c['init_sha'])[:12]} (A2 {a2_sha[:12]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = {"Z30a": z30a, "Z30b": z30b, "Z30c": z30c, "Z30d": z30d, "Z30p": z30p}
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
