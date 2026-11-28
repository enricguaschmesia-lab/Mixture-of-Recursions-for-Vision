#!/usr/bin/env python
"""Gate for A2 Step 7.0f (docs/plans/a2_segmentation.md, Step 7, "7.0f, the stack: E784 with the 8 orientations and
corrected voting"). Every check has a control that fails it. `.venv`, CPU; Z34a's re-encoding runs as a subprocess in
the a2seg env on the TITAN V (the card today's MAP_c4375_224 was written on).

  Z34r  7.0e's gate (verify_seg_p7e.py, 7.0d's and 7.0c's inside) on this tree: 17/17.
  Z34a  the TTA root's MAP_c4375_224: 3,856 rows present; the k = 0 rows equal today's; the first 16 new maps re-encoded
        on the TITAN V equal the stored ones. Control: the same maps encoded at k = 0 differ.
  Z34b  the true c4375 tokens of every orientation, decoded and brought back, within 0.005 of k = 0's. Control: not
        brought back, k = 1..7 fall.
  Z34c  E784's config on the TTA root: 3,856 val views, 2,193 tokens, every view's inputs equal the aligned TTA
        dataset's (B1's config on the same root), the nested map at 198 (j + 1) after the latest date. Control: E784's
        config on today's root holds 482 views.
  Z34d  jobs/p7f_stack.sh pins the TTA root's 8 files, E784's save (its selected_sha256.txt) and E784's config. Control:
        a hash moved by one character is found.
  Z34e  the read's rules: band_s2 on known pairs. Control: an interval reaching 0 is not (a).
    python eo/scripts/verify_seg_p7f.py [--only Z34a,Z34b] [--out gate.json]
"""
from __future__ import annotations

import argparse
import json
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
TTA = Path("/data/enric/data/geobench2_tta")
TODAY = Path("/data/enric/data/geobench2/pastis/tok")
OUT = Path("/data/enric/reports/a2_seg/step70f")
TMP = OUT / "gate_tmp"
A2SEG_PY = "/data/enric/miniforge3/envs/a2seg/bin/python"
JOB = Path("/data/enric/rcp_a2s4/s70e/jobs/p7f_stack.sh")
E784 = Path("/data/enric/runs/pretrain/a2_seg/p7e784_s42_20261127_004451")
B1_CLI = ["learning_rate=3e-3", "seed=42"]


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def z34r() -> None:
    out = TMP / "z34r_gate_p7e.json"
    t0 = time.time()
    r = subprocess.run([sys.executable, "eo/scripts/verify_seg_p7e.py", "--out", str(out)], cwd=str(REPO),
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=7200)
    (TMP / "z34r_gate_p7e.log").write_text(r.stdout + r.stderr)
    res = json.loads(out.read_text()) if out.exists() else []
    n_pass = sum(x["status"] == "PASS" for x in res)
    record("Z34r 7.0e's gate on this tree", r.returncode == 0 and len(res) == 17 and n_pass == 17,
           f"exit {r.returncode}; {n_pass}/{len(res)} pass; {time.time() - t0:.0f} s; {out}")


def z34a() -> None:
    tok = TTA / "pastis" / "tok"
    copy = json.loads((tok / "tta_copy.json").read_text())["map"]
    new, old = np.load(tok / "MAP_c4375_224/tokens.npy", mmap_mode="r"), np.load(TODAY / "MAP_c4375_224/tokens.npy",
                                                                                 mmap_mode="r")
    rows = np.array(sorted(int(k) for k in copy))
    k0 = np.array_equal(new[rows], old[[copy[str(r)] for r in rows]])
    pres = np.load(tok / "MAP_c4375_224/present.npy")
    gpu = subprocess.run([str(REPO / ".venv/bin/python"), "-m", "eo.train.gpu", "titanv"], cwd=str(REPO),
                         capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE="1",
               HF_HOME="/data/enric/hf")
    # the TITAN V stalls with no identifiable op (CLAUDE.md): one retry, each try under timeout -s ABRT (seconds normally)
    for attempt in (1, 2):
        r = subprocess.run(["timeout", "-s", "ABRT", "300", A2SEG_PY, "-X", "faulthandler",
                            "eo/experiments/segmentation/step70e_tta_tokens.py", "--part", "check_maps", "--arm", "c4375_224"],
                           cwd=str(REPO), env=env, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            break
    (TMP / "z34a_check_maps.log").write_text(r.stdout + r.stderr)
    c = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.returncode}
    ok = len(pres) == 3856 and pres.all() and new.shape == (3856, 784) and k0 and c.get("equal") is True
    record("Z34a the TTA root's MAP_c4375_224", bool(ok),
           f"rows {new.shape}, all present {bool(pres.all())}; k = 0 rows = today's {k0}; re-encoded {c}")
    record("Z34a control: the same maps encoded at k = 0", c.get("control_k0_equal") is False,
           f"equal {c.get('control_k0_equal')} (token share {c.get('control_k0_token_share_equal')})")


def z34b() -> None:
    import step70e_read as RD
    import step70f_read as F
    from step4_score import per_patch_cm
    from step52_read import IGNORE
    from step53_verdict import miou
    ceil = F.truth_ceilings(F.TRUTH)
    record("Z34b the true c4375 tokens of every orientation, brought back", all(abs(v - ceil[0]) <= 0.005
                                                                                 for v in ceil.values()),
           f"by k {({k: round(v, 4) for k, v in ceil.items()})}")
    tm, tv, _ = RD.load_maps(F.TRUTH)
    ks = np.asarray(tv["k"])
    raw = {}
    for k in range(1, 8):
        rows = np.where(ks == k)[0]
        lab = RD.labels_for([tv["sample_id"][r] for r in rows])
        raw[k] = miou(per_patch_cm(np.stack([np.asarray(tm[r, 0]) for r in rows]), lab, 20, IGNORE["pastis"]).sum(0),
                      IGNORE["pastis"])
    record("Z34b control: not brought back", all(v < ceil[0] - 0.1 for v in raw.values()),
           f"by k {({k: round(v, 4) for k, v in raw.items()})}")


def z34c() -> None:
    cfg = _compose("eo_seg/s70e_grid784", B1_CLI + [f"seg.root={TTA}"])
    ds = SD.SegDataset(SD.spec_from_cfg(cfg, "eval"), max_length=None, seed=42)
    ref = SD.SegDataset(SD.spec_from_cfg(_compose("eo_seg/s70_p7", B1_CLI + [f"seg.root={TTA}"]), "eval"),
                        max_length=None, seed=42)
    n_in = 7 * 201
    bad = []
    for v in [int(x) for x in ds.views.view.values[[0, 1, 5, 8 * 100 + 3, 8 * 481 + 7]]]:
        s, t = ds.sequence(v), ref.sequence(v)
        same = all(torch.equal(s[k][:n_in], t[k][:n_in]) for k in ("input_ids", "position_ids", "modality_ids"))
        pos = s["position_ids"][n_in + 1:n_in + 1 + 784]
        j, k = torch.arange(784) // 196, torch.arange(784) % 196
        if not (same and len(s["input_ids"]) == 2193 and torch.equal(pos - (198 * 6 + 1 + k), 198 * (j + 1))):
            bad.append(v)
    ok = len(ds.views) == 3856 and set(ds.views.k) == set(range(8)) and not bad
    record("Z34c E784's config on the TTA root", ok, f"{len(ds.views)} views, k {sorted(set(ds.views.k))}; 5 views' inputs "
           f"= the aligned TTA dataset's and nested offsets; problems {bad}")
    today = SD.SegDataset(SD.spec_from_cfg(_compose("eo_seg/s70e_grid784", B1_CLI), "eval"), max_length=None, seed=42)
    record("Z34c control: E784's config on today's root", len(today.views) == 482, f"{len(today.views)} views")


def z34d() -> None:
    js = JOB.read_text()
    tta = (Path("/data/enric/rcp_a2s4/s70e/local_tta_root_c4375.sha256").read_text().split("\n"))
    files = ["pastis/tok/views.parquet", "pastis/tok/inputs.parquet", "pastis/tok/S2L2A_tok224/metadata.json",
             "pastis/tok/S2L2A_tok224/present.npy", "pastis/tok/S2L2A_tok224/tokens.npy",
             "pastis/tok/MAP_c4375_224/metadata.json", "pastis/tok/MAP_c4375_224/present.npy",
             "pastis/tok/MAP_c4375_224/tokens.npy"]
    now = subprocess.run(["sha256sum"] + files, cwd=str(TTA), capture_output=True, text=True).stdout.strip().splitlines()
    e784 = (E784 / "selected_sha256.txt").read_text().split()[0]
    ok = (all(line in js for line in now) and len(now) == 8 and f"{e784}  model.safetensors" in js
          and "--config eo_seg/s70e_grid784" in js and "checkpoint-1776" in js and [x for x in tta if x] == now)
    record("Z34d the stack's job script pins the data, E784's save and its config", ok,
           f"8 TTA files pinned {all(line in js for line in now)}; E784 save {e784[:12]} pinned "
           f"{f'{e784}  model.safetensors' in js}; config and checkpoint {'--config eo_seg/s70e_grid784' in js}")
    moved = now[0][:5] + ("0" if now[0][5] != "0" else "1") + now[0][6:]
    record("Z34d control: a hash moved by one character", moved not in js, f"found {moved not in js}")


def z34e() -> None:
    import step70f_read as F
    G = lambda d, lo, hi: {"delta": d, "lo95": lo, "hi95": hi}       # noqa: E731
    b = [F.band_s2(G(1.5, 0.2, 2.8))[:3], F.band_s2(G(0.6, -0.4, 1.6))[:3], F.band_s2(G(-1.4, -2.6, -0.3))[:3]]
    record("Z34e the read's rules", b == ["(a)", "(b)", "(c)"], f"bands {b}")
    c = F.band_s2(G(1.5, -0.05, 3.0))[:3]
    record("Z34e control: an interval reaching 0 is not (a)", c != "(a)", f"band {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    checks = {"Z34r": z34r, "Z34a": z34a, "Z34b": z34b, "Z34c": z34c, "Z34d": z34d, "Z34e": z34e}
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
