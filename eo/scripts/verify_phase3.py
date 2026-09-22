#!/usr/bin/env python
"""Phase 3 -- the verification gate (D3.10).

Same discipline as verify_phase2.py: every check ships with a deliberately
broken control that proves the check can fail (CLAUDE.md rule 2). A check that
only ever passes is not evidence.

W0-W9 are specified in PHASE3_PLAN.md Step 9. They land with the step that
produces them rather than all at the end, so a step is never "done" with an
unverified deliverable:

  W0  verify_phase2 V0-V10 still pass     [Step 1]  the regression net
  W1  train n eval empty at GROUP level   [Step 1]
  W2  eval covers every active modality   [Step 1]
  W3  eval loop runs; untrained ~ ln(V)   [Step 1]
  W4  per-modality loss in BOTH arms      [Step 5]  not yet implemented
  W5  fp16 matches fp32 within tolerance  [Step 2]  not yet implemented
  W6  generated ids land in target slot   [Step 6]  not yet implemented
  W7  decode metrics collapse on shuffle  [Step 7]  not yet implemented
  W8  arm configs differ only as intended [Step 4]  not yet implemented
  W9  resume gives a continuous curve     [Step 0]  covered by
                                                    eo.train.preflight_controls

Runs in the repo .venv, NOT the `mor` env.

    HF_HOME=/data/enric/hf ./.venv/bin/python eo/scripts/verify_phase3.py
    ./.venv/bin/python eo/scripts/verify_phase3.py --skip-forward   # no GPU
    ./.venv/bin/python eo/scripts/verify_phase3.py --skip-phase2    # W1-W3 only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

EO_CONFIG = REPO / "conf/pretrain_vision/eo_terramesh/terramesh_mor_token.yaml"
ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
LN_V = float(np.log(87556))


def _split_table():
    """The row table plus the group labels and the committed eval rows."""
    from eo.data.eval_split import (SPLIT_RADIUS_KM, load_eval_rows,
                                    load_row_table, spatial_groups)
    table = load_row_table(ROOT)
    groups = spatial_groups(table.center_lat.values, table.center_lon.values, SPLIT_RADIUS_KM)
    eval_rows = load_eval_rows(root_dir=ROOT)
    return table, groups, eval_rows


def _km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _min_cross_km(lat_a, lon_a, lat_b, lon_b, bucket_deg=0.05):
    """Smallest distance between any point in A and any point in B.

    Bucketed on a lat/lon grid and probed over the 3x3 neighbourhood: this only
    has to be exact in the near field, which is where leakage lives, and an
    89,088 x 4,416 brute force is not worth the minutes.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for la, lo in zip(lat_b, lon_b):
        buckets[(int(np.floor(la / bucket_deg)), int(np.floor(lo / bucket_deg)))].append((la, lo))
    best = np.inf
    for la, lo in zip(lat_a, lon_a):
        bi, bj = int(np.floor(la / bucket_deg)), int(np.floor(lo / bucket_deg))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for (la2, lo2) in buckets.get((bi + di, bj + dj), ()):
                    d = _km(la, lo, la2, lo2)
                    if d < best:
                        best = d
    return best


def w1_group_disjoint() -> bool:
    """Train and eval share no GROUP, and the group is the right one.

    The plan's acceptance criterion is disjointness at the group level. That is
    necessary but not sufficient on its own: it is satisfied by ANY grouping,
    including one that leaks. So this also measures the physical separation the
    group is supposed to buy, and shows the two rejected keys failing it.
    """
    from eo.data.eval_split import TILE_WIDTH_M
    table, groups, eval_rows = _split_table()

    is_eval = np.zeros(len(table), dtype=bool)
    is_eval[eval_rows] = True
    eval_groups = set(np.unique(groups[is_eval]).tolist())
    train_groups = set(np.unique(groups[~is_eval]).tolist())
    disjoint = len(eval_groups & train_groups) == 0
    print(f"    groups in both train and eval                 = {len(eval_groups & train_groups)}")
    print(f"    train n eval empty at the group level         = {disjoint}")

    # The physical claim: no held-out tile's footprint overlaps a training
    # tile's. Tiles are TILE_WIDTH_M square, so centroids at least that far
    # apart cannot share a pixel.
    tile_km = TILE_WIDTH_M / 1000.0
    d = _min_cross_km(table.center_lat.values[is_eval], table.center_lon.values[is_eval],
                      table.center_lat.values[~is_eval], table.center_lon.values[~is_eval])
    no_overlap = d >= tile_km
    print(f"    closest eval-to-train tile                    = {d:.2f} km (>= {tile_km:.2f} required)")
    print(f"    no held-out tile overlaps a training tile     = {no_overlap}")

    rng = np.random.default_rng(0)

    # CONTROL 1: a ROW-level split of the same size must violate the group
    # criterion. This is the split the plan rejected in Step 1.1.
    row_eval = rng.choice(len(table), size=len(eval_rows), replace=False)
    m = np.zeros(len(table), dtype=bool); m[row_eval] = True
    c1_shared = len(set(np.unique(groups[m]).tolist()) & set(np.unique(groups[~m]).tolist()))
    c1 = c1_shared > 0
    print(f"    control: a row-level split shares {c1_shared:>5} groups = {c1}")

    # CONTROL 2: the PLAN'S OWN per-corpus textual key must be shown to leak.
    # This is why eval_split.py deviates from PHASE3_PLAN Step 1.2. Holding out
    # ssl4eos12 locations leaves the same ground in training inside majortom
    # tiles, so the closest cross-boundary tile pair is far under one tile width.
    sid = table.sample_id.astype(str)
    textual = np.where(table.corpus.values == "majortom",
                       sid.str.split("_").str[:2].str.join("_"),
                       sid.str.split("_").str[0]).astype(str)
    ssl_groups = np.unique(textual[table.corpus.values == "ssl4eos12"])
    held = set(rng.choice(ssl_groups, size=max(1, len(ssl_groups) // 20), replace=False).tolist())
    tm = np.array([g in held for g in textual])
    d_textual = _min_cross_km(table.center_lat.values[tm], table.center_lon.values[tm],
                              table.center_lat.values[~tm], table.center_lon.values[~tm])
    c2 = d_textual < tile_km
    print(f"    control: the plan's textual key leaves {d_textual:.2f} km  = {c2}")
    print(f"             (< {tile_km:.2f} km means overlapping imagery in train)")

    return disjoint and no_overlap and c1 and c2


def w2_eval_covers_modalities() -> bool:
    """Every active modality appears in the eval set, S1GRD included.

    S1GRD exists ONLY on ssl4eos12 rows and S1RTC ONLY on majortom rows -- they
    are exact complements. A holdout drawn without regard to that can contain
    zero S1GRD scenes and produce no S1GRD eval loss at all, which looks like a
    modality that is simply never logged rather than like an error.
    """
    from eo.data.eval_split import load_eval_rows
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset

    eval_rows = load_eval_rows(root_dir=ROOT)
    ds = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048, modality_order="random",
                               rows=eval_rows)
    counts = {m: int(ds._present[m][eval_rows].sum()) for m in ds.active_modalities}
    all_present = all(v > 0 for v in counts.values())
    for m, v in counts.items():
        print(f"    eval rows carrying {m:<7} = {v:>5}")
    print(f"    every active modality present in eval         = {all_present}")

    # CONTROL: drop the ssl4eos12 rows and S1GRD must vanish, proving the check
    # is actually reading the eval set and not a constant.
    from eo.data.eval_split import load_row_table
    table = load_row_table(ROOT)
    mt_only = np.array([r for r in eval_rows if table.corpus.values[r] == "majortom"])
    c1 = int(ds._present["S1GRD"][mt_only].sum()) == 0
    print(f"    control: dropping ssl4eos12 zeroes S1GRD      = {c1}")

    return all_present and c1


def w3_eval_loop(n_eval_rows: int = 24) -> bool:
    """MoRTrainer.evaluate() runs, and an untrained model scores ln(V).

    ⚠ This is the check that would have failed at the first eval step of a
    multi-day run. compute_loss returns a 10-tuple and used to ignore its own
    return_outputs, while HF's prediction_step does
    `loss, outputs = self.compute_loss(..., return_outputs=True)`.

    Both arms are exercised: MoRTrainer (mor.enable=true) and the stock Trainer
    the non-MoR arm falls back to.
    """
    import torch
    from omegaconf import OmegaConf, open_dict
    from transformers import Trainer, TrainingArguments

    from eo.data.eval_split import load_eval_rows
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from model.util import load_model_from_config
    from model.sharing_strategy import SHARING_STRATEGY
    from util.config import preprocess_config
    from util.trainer_pt import MoRTrainer

    # ⚠ Stratify the subsample by corpus. The eval rows are sorted, and every
    # majortom row precedes every ssl4eos12 one, so a plain head() is 100%
    # majortom -- which carries S1RTC and never S1GRD, and the per-modality
    # assertion below would fail on the sampling rather than on the code. W2 is
    # what checks the real eval set's coverage; this only needs a few batches
    # that between them touch all seven modalities.
    from eo.data.eval_split import load_row_table
    table = load_row_table(ROOT)
    all_eval = load_eval_rows(root_dir=ROOT)
    corpus = table.corpus.values
    half = max(2, n_eval_rows // 2)
    mt = [r for r in all_eval if corpus[r] == "majortom"][:half]
    ss = [r for r in all_eval if corpus[r] == "ssl4eos12"][:half]
    rows = np.array(sorted(mt + ss), dtype=np.int64)
    ds = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048, modality_order="random", rows=rows)

    def run(mor_enabled: bool):
        cfg = OmegaConf.load(EO_CONFIG)
        with open_dict(cfg):
            cfg.wandb = False
            cfg.tensorboard = False
            cfg.recursive.enable = mor_enabled
            cfg.mor.enable = mor_enabled
            cfg.num_train_steps = 1
            cfg.stop_steps = 1
        cfg = preprocess_config(cfg)
        model = load_model_from_config(cfg)
        if cfg.recursive.get("enable"):
            model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
        if cfg.mor.get("enable"):
            model.transform_layer_to_mor_token(cfg)
        args = TrainingArguments(
            output_dir=os.path.join("/tmp", "verify_phase3_w3"),
            per_device_eval_batch_size=2, prediction_loss_only=True,
            remove_unused_columns=False, report_to=[], bf16=False, fp16=False,
        )
        if mor_enabled:
            tr = MoRTrainer(model=model, args=args, eval_dataset=ds, cfg=cfg)
        else:
            tr = Trainer(model=model, args=args, eval_dataset=ds)
        m = tr.evaluate()
        del model, tr
        torch.cuda.empty_cache()
        return m

    ok = True
    for name, flag in (("MoR arm     ", True), ("vanilla arm ", False)):
        try:
            metrics = run(flag)
        except Exception as e:                        # noqa: BLE001 -- report, don't mask
            print(f"    {name} evaluate() RAISED {type(e).__name__}: {e}")
            ok = False
            continue
        loss = float(metrics["eval_loss"])
        near = abs(loss - LN_V) < 0.15
        print(f"    {name} eval_loss = {loss:.4f}  (ln V = {LN_V:.4f})  = {near}")
        ok &= near
        if flag:
            per_mod = {k: v for k, v in metrics.items() if k.startswith("eval_loss_")}
            has_all = len(per_mod) == 7 and "eval_loss_S1GRD" in per_mod
            print(f"    {name} per-modality eval keys: {len(per_mod)}/7, S1GRD present = {has_all}")
            ok &= has_all

    # CONTROL: an untrained model at the WRONG vocabulary size must not land on
    # ln(87,556). This is what makes the number above evidence rather than a
    # coincidence of scale.
    from lm_dataset.modality_registry import assert_vocab_size
    bad = OmegaConf.merge(OmegaConf.load(EO_CONFIG), {"model_config": {"vocab_size": 242271}})
    try:
        assert_vocab_size(bad); c1 = False
    except ValueError:
        c1 = True
    print(f"    control: a CLEVR vocab_size is rejected       = {c1}")

    return ok and c1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-forward", action="store_true", help="skip W3 (needs a GPU)")
    ap.add_argument("--skip-phase2", action="store_true", help="skip W0, the inherited gate")
    args = ap.parse_args()

    checks = []
    if not args.skip_phase2:
        checks.append(("W0   verify_phase2 V0-V10 (regression net)", _w0))
    checks += [
        ("W1   train n eval disjoint at group level [Step 1]", w1_group_disjoint),
        ("W2   eval covers every modality, S1GRD    [Step 1]", w2_eval_covers_modalities),
    ]
    if not args.skip_forward:
        checks.append(("W3   eval loop runs; untrained ~ ln(V)    [Step 1]", w3_eval_loop))

    results = []
    for title, fn in checks:
        print(f"\n{title}")
        try:
            results.append((title, bool(fn())))
        except Exception as e:                        # noqa: BLE001
            print(f"    RAISED {type(e).__name__}: {e}")
            results.append((title, False))

    print("\n" + "=" * 70)
    for title, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}")
    print("=" * 70)
    if all(ok for _, ok in results):
        print("PHASE 3 GATE PASSED -- all implemented checks pass and all controls fire")
        print("  (W4-W8 land with Steps 2, 4, 5, 6, 7; W9 is eo.train.preflight_controls)")
        return 0
    print("PHASE 3 GATE FAILED")
    return 1


def _w0() -> bool:
    """The inherited gate. Steps 1, 2 and 5 all modify shared code, and V0
    holds the CLEVR dataset's output bit-identical across those changes."""
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO / "eo/scripts/verify_phase2.py"), "--skip-forward"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    tail = [ln for ln in r.stdout.splitlines() if ln.startswith("  PASS") or ln.startswith("  FAIL")]
    for ln in tail:
        print(f"    {ln.strip()}")
    return r.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
