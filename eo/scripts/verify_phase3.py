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
  W4  per-modality loss in BOTH arms      [Step 5]
  W5  fp16 matches fp32 within tolerance  [Step 2]
  W6  generated ids land in target slot   [Step 6]
  W7  decode metrics collapse on shuffle  [Step 7]  not yet implemented
  W8  arm configs differ only as intended [Step 4]
  W9  resume gives a continuous curve     [Step 0]  covered by
                                                    eo.train.preflight_controls

Runs in the repo .venv, NOT the `mor` env.

    HF_HOME=/data/enric/hf ./.venv/bin/python eo/scripts/verify_phase3.py
    ./.venv/bin/python eo/scripts/verify_phase3.py --skip-forward   # no GPU
    ./.venv/bin/python eo/scripts/verify_phase3.py --skip-phase2    # skip W0

⚠ W3 and W5 build six models in one process. Releasing a model needs
accelerate's Accelerator.free_memory(), not just `del` -- see _release().
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ⚠ This gate builds six models across W3 and W5 in one process. pretrain.py
# sets this at module scope but the gate never imports it, so without it the
# later checks OOM on fragmentation left by the earlier ones -- a gate that
# fails on its own memory bookkeeping is worse than no gate.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

EO_CONFIG = REPO / "conf/pretrain_vision/eo_terramesh/terramesh_mor_token.yaml"
ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")

# The contract is pure constants and imports in EITHER env, which is what lets
# W7 check that decode_eo.py (which runs in `mor`) decoded at this crop.
from eo.terramesh_tok.contract import CROP as TOK_CROP  # noqa: E402
LN_V = float(np.log(87556))


def _release(model=None, trainer=None):
    """Give a model's GPU memory back before the next check builds another.

    ⚠ `del` + `empty_cache()` is NOT enough, and neither is adding a
    `gc.collect()`. Measured: with both, W5 still died with 11.13 GiB still
    *allocated* (not merely reserved) on models W3 had finished with. accelerate's
    `Accelerator` keeps its own references to every model and optimizer it has
    prepared, so nothing the caller drops actually frees them. `free_memory()`
    is the documented way to clear those lists; without it this gate fails on
    its own bookkeeping rather than on anything it is testing.
    """
    import gc
    import torch
    if trainer is not None:
        acc = getattr(trainer, "accelerator", None)
        if acc is not None and hasattr(acc, "free_memory"):
            acc.free_memory()
        for attr in ("model_wrapped", "model", "optimizer", "lr_scheduler"):
            if hasattr(trainer, attr):
                setattr(trainer, attr, None)
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


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
        _release(model, tr)
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


def _short_train(precision: str, mixed: bool, steps: int, tbs: int,
                 break_scaler: bool = False, n_rows: int = 512):
    """Train a few steps in-process and return the per-step loss curve.

    Small and self-contained so the gate can run three variants in ~2 min. The
    *measurement* that justified adopting fp16 was done at the shipped settings
    through train_eo.sh and lives in the worklog; this proves the mechanism and,
    more importantly, that the comparison can fail.
    """
    import torch
    from omegaconf import OmegaConf, open_dict
    from transformers import TrainingArguments

    from eo.data.eval_split import load_eval_rows, train_rows_from_eval
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from model.util import load_model_from_config
    from model.sharing_strategy import SHARING_STRATEGY
    from util.config import preprocess_config
    from util.seeding import set_global_seed
    from util.trainer_pt import MoRTrainer

    cfg = OmegaConf.load(EO_CONFIG)
    with open_dict(cfg):
        cfg.wandb = False
        cfg.tensorboard = False
        cfg.precision = precision
        cfg.mixed_precision = mixed
        cfg.total_batch_size = tbs
        # ⚠ Pin the micro-batch rather than inheriting the shipped value. This
        # check runs several models in one process, and inheriting meant that
        # raising per_device_train_batch_size to 4 for throughput made W5 OOM
        # on its own -- a gate whose result depends on an unrelated tuning knob
        # is not measuring what it claims to.
        cfg.per_device_train_batch_size = 2
        cfg.num_train_steps = steps
        cfg.stop_steps = steps
        cfg.num_warmup_steps = 2
        cfg.seed = 42
    cfg = preprocess_config(cfg)

    set_global_seed(42, deterministic_cuda=False)

    eval_rows = load_eval_rows(root_dir=ROOT)
    rows = train_rows_from_eval(eval_rows, 89088)[:n_rows]
    ds = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048,
                               modality_order="random", rows=rows)

    model = load_model_from_config(cfg)
    model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
    model.transform_layer_to_mor_token(cfg)

    args = TrainingArguments(
        output_dir="/tmp/verify_phase3_w5",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_steps=steps, warmup_steps=2, logging_steps=1,
        save_strategy="no", report_to=[], remove_unused_columns=False,
        fp16=(precision == "fp16"), bf16=(precision == "bf16"),
        learning_rate=cfg.learning_rate, max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=0, seed=42, data_seed=42,
    )
    trainer = MoRTrainer(model=model, args=args, train_dataset=ds, cfg=cfg)

    if break_scaler:
        # ⚠ THE CONTROL. Disable loss scaling while keeping fp16 autocast. This
        # is precisely what the GradScaler exists to prevent: unscaled fp16
        # gradients underflow to zero, so the updates differ and the curve must
        # visibly leave the fp32 one. If it does not, the equivalence check
        # above has no power and "fp16 matches fp32" means nothing.
        trainer.accelerator.scaler._enabled = False

    trainer.train()
    curve = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    _release(model, trainer)
    return np.array(curve)


def w5_fp16_equivalence(steps: int = 15, tbs: int = 8) -> bool:
    """fp16-via-autocast computes the same thing as fp32, and the check can fail.

    ⚠ Two latent bugs had to be fixed before fp16 ran at all, and both were
    invisible to fp32 and bf16 (Phase 3 Step 2):
      1. `precision` set the PARAMETER dtype as well as TrainingArguments'
         autocast flag, so fp16 weights met a GradScaler that requires fp32
         masters -> "Attempting to unscale FP16 gradients" at the first
         optimizer step.
      2. `MoRTrainer._inner_training_loop` clips gradients in two groups and
         called `accelerator.clip_grad_norm_` twice, which unscales twice ->
         "unscale_() has already been called". No scaler, no symptom.
    """
    # ⚠ Compare MEAN |Δ|, not max, and the reason is architectural rather than
    # statistical. Token-choice routing makes a DISCRETE top-k decision per
    # token, so a numeric difference far below any tolerance can flip a token's
    # depth at a near-tie and move that one step's loss by ~0.08. Measured, the
    # same spikes occur between two identical fp32 runs (max |Δ| 0.0826 at
    # tbs=8), so they are not an fp16 artifact -- per-step max simply is not a
    # stable quantity for this model.
    #
    # ⚠ And the floor itself is order-dependent: two back-to-back fp32 runs came
    # out identical (0.0000) while the same pair with an fp16 run between them
    # differed by 0.0826, presumably via allocator state changing cuBLAS kernel
    # selection. So a floor measured in-run cannot be the tolerance either. An
    # earlier version of this check tried both and reported a problem with fp16
    # that did not exist.
    #
    # What IS stable across invocations is the mean, and the ratio between the
    # candidate and the deliberately-broken control:
    #     fp32 vs fp16    mean 0.0078, 0.0079   (two invocations)
    #     fp32 vs broken  mean 0.2205, 0.2205
    # The control is ~28x worse. That is the discriminator.
    ABS_TOL = 0.05        # 0.5% of a loss of ~10; fp16 measures 0.008
    MIN_RATIO = 5.0       # the control must be decisively worse, measured ~28x

    fp32 = _short_train("fp32", False, steps, tbs)
    fp16 = _short_train("fp16", True, steps, tbs)
    n = min(len(fp32), len(fp16))
    d = float(np.abs(fp32[:n] - fp16[:n]).mean())
    ok = d < ABS_TOL
    print(f"    fp32 vs fp16              mean |Δ| {d:.4f}  (tol {ABS_TOL}) = {ok}")

    broken = _short_train("fp16", True, steps, tbs, break_scaler=True)
    nb = min(len(fp32), len(broken))
    db = float(np.abs(fp32[:nb] - broken[:nb]).mean())
    ratio = db / d if d > 0 else float("inf")
    fired = db > ABS_TOL and ratio > MIN_RATIO
    print(f"    control: scaler disabled  mean |Δ| {db:.4f}  ({ratio:.1f}x worse) = {fired}")
    if not fired:
        print("      ⚠ the control did NOT diverge -- the equivalence check has no power")

    return bool(ok and fired)


#: Routing metrics are legitimately absent from the non-MoR arm. NOTHING else
#: may differ between the two arms' logged keys.
ROUTING_ONLY_KEYS = {"balancing_loss", "balancing_entropy", "router_z_loss",
                     "bal_tr_ratio", "sam_tr_loss", "sam_tr_acc", "sam_tr_topk_acc"}

EO_MODALITIES = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]

#: Bookkeeping HF emits that says nothing about either arm.
_IGNORED_LOG_KEYS = {
    "epoch", "step", "total_flos", "train_runtime", "train_samples_per_second",
    "train_steps_per_second", "train_loss", "grad_norm", "learning_rate",
    "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
}


def _logged_keys(arm_mor: bool, steps: int = 2, tbs: int = 8,
                 n_train: int = 64, n_eval: int = 16,
                 remove_unused_columns: bool = False):
    """Train a couple of steps, evaluate once, and return the logged key sets."""
    import torch
    from omegaconf import OmegaConf, open_dict
    from transformers import TrainingArguments

    from eo.data.eval_split import load_eval_rows, load_row_table, train_rows_from_eval
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from model.util import load_model_from_config
    from model.sharing_strategy import SHARING_STRATEGY
    from util.config import preprocess_config
    from util.seeding import set_global_seed
    from util.trainer_pt import EOTrainer, MoRTrainer

    cfg = OmegaConf.load(EO_CONFIG)
    with open_dict(cfg):
        cfg.wandb = False
        cfg.tensorboard = False
        cfg.recursive.enable = arm_mor
        cfg.mor.enable = arm_mor
        cfg.total_batch_size = tbs
        cfg.per_device_train_batch_size = 2
        cfg.num_train_steps = steps
        cfg.stop_steps = steps
        cfg.num_warmup_steps = 1
        cfg.seed = 42
    cfg = preprocess_config(cfg)
    set_global_seed(42, deterministic_cuda=False)

    all_eval = load_eval_rows(root_dir=ROOT)
    corpus = load_row_table(ROOT).corpus.values
    # ⚠ stratified: eval rows are sorted majortom-first, and majortom carries
    # S1RTC but never S1GRD, so an unstratified head would make S1GRD look
    # absent from BOTH arms and the check would pass vacuously.
    half = max(2, n_eval // 2)
    ev = np.array(sorted([r for r in all_eval if corpus[r] == "majortom"][:half]
                         + [r for r in all_eval if corpus[r] == "ssl4eos12"][:half]))
    # ⚠ ...and the TRAIN subsample needs stratifying for the same reason. The
    # first version of this check took `train_rows[:64]`, which is 100%
    # majortom, so no batch contained an S1GRD scene and `loss_S1GRD` was
    # absent from BOTH arms -- 6/7, reported as a failure of the code rather
    # than of the sampling. The trap is the same one 6.7/7.7/8.7 warn about;
    # it bites the training side too.
    all_train = train_rows_from_eval(all_eval, 89088)
    h = max(2, n_train // 2)
    tr = np.array(sorted([r for r in all_train if corpus[r] == "majortom"][:h]
                         + [r for r in all_train if corpus[r] == "ssl4eos12"][:h]))

    ds_tr = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048, modality_order="random", rows=tr)
    ds_ev = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048, modality_order="random", rows=ev)

    model = load_model_from_config(cfg)
    if cfg.recursive.enable:
        model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
        model.transform_layer_to_mor_token(cfg)

    args = TrainingArguments(
        output_dir="/tmp/verify_phase3_w4",
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_eval_batch_size=2, prediction_loss_only=True,
        max_steps=steps, warmup_steps=1, logging_steps=1,
        save_strategy="no", eval_strategy="no", report_to=[],
        remove_unused_columns=remove_unused_columns,
        fp16=(cfg.precision == "fp16"), learning_rate=cfg.learning_rate,
        max_grad_norm=cfg.max_grad_norm, dataloader_num_workers=0,
        seed=42, data_seed=42,
    )
    cls = MoRTrainer if arm_mor else EOTrainer
    trainer = cls(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_ev, cfg=cfg)
    trainer.train()
    trainer.evaluate()

    train_keys, eval_keys = set(), set()
    for entry in trainer.state.log_history:
        for k in entry:
            if k in _IGNORED_LOG_KEYS:
                continue
            (eval_keys if k.startswith("eval_") else train_keys).add(k)
    _release(model, trainer)
    return train_keys, eval_keys


def w4_symmetric_logging() -> bool:
    """Both arms log per-modality loss, on the TRAIN and EVAL axes (D3.6).

    ⚠ The asymmetry this guards was structural and silent. `pretrain.py` builds
    MoRTrainer only when `mor.enable` is true, and per-modality loss lived
    inside it -- so the recursion-OFF arm logged none of the quantity the
    comparison is about. Step 1 then added evaluation to MoRTrainer alone,
    reproducing the identical hole on the eval axis, which is the axis the
    Plan 1 / Plan 2 call is read off. Neither failed loudly; both look like a
    normal W&B page with fewer lines.
    """
    tr_a, ev_a = _logged_keys(arm_mor=True)
    tr_b, ev_b = _logged_keys(arm_mor=False)

    want_tr = {f"loss_{m}" for m in EO_MODALITIES}
    want_ev = {f"eval_loss_{m}" for m in EO_MODALITIES}

    ok = True
    for name, tr, ev in (("arm A (MoR)   ", tr_a, ev_a), ("arm B (vanilla)", tr_b, ev_b)):
        t_ok, e_ok = want_tr <= tr, want_ev <= ev
        ok &= t_ok and e_ok
        print(f"    {name} train per-modality {len(want_tr & tr)}/7 = {t_ok}   "
              f"eval per-modality {len(want_ev & ev)}/7 = {e_ok}")

    diff = (tr_a ^ tr_b) | (ev_a ^ ev_b)
    diff_ok = bool(diff) and diff <= ROUTING_ONLY_KEYS
    print(f"    key-set difference between arms: {sorted(diff)}")
    print(f"    ...and it is exactly routing keys = {diff_ok}")

    # CONTROL: remove_unused_columns=True makes HF's RemoveColumnsCollator strip
    # `modality_ids`, and every per-modality key must then VANISH -- silently,
    # which is precisely why this is load-bearing rather than cosmetic.
    tr_c, ev_c = _logged_keys(arm_mor=False, remove_unused_columns=True)
    c1 = not (want_tr & tr_c) and not (want_ev & ev_c)
    print(f"    control: remove_unused_columns=True drops all per-modality keys = {c1}")

    return bool(ok and diff_ok and c1)


#: The ONLY keys the two arms may differ in. Everything else must be identical
#: or the comparison is confounded before it starts.
ARM_DIFF_KEYS = {"recursive.enable", "mor.enable"}

#: Keys whose equality across arms is load-bearing, checked by name as well as
#: by the blanket diff so a failure says WHICH invariant broke.
ARM_MUST_MATCH = [
    "multimodal.eval_split", "seed", "num_train_steps", "stop_steps",
    "total_batch_size", "per_device_train_batch_size", "num_warmup_steps",
    "lr_scheduler_kwargs.num_decay_steps", "eval_steps", "save_steps",
    "precision", "mixed_precision", "max_length", "model_config.vocab_size",
    "learning_rate", "max_grad_norm", "dataloader_num_workers",
    "multimodal.modality_order", "multimodal.active_modalities",
]

ARM_A = "eo_terramesh/arm_a_mor"
ARM_B = "eo_terramesh/arm_b_vanilla"


def w6_generation_slots(target: str = "LULC", n_scenes: int = 2,
                        max_new_tokens: int = 24) -> bool:
    """Generated ids land in the target slot, and BO/EO are well formed (D3.7).

    ⚠ Runs on an UNTRAINED model on purpose. Off-slot rate has a known
    chance level -- `1 - slot/87,556`, i.e. 95.0% for LULC -- so an untrained
    model gives both a control and the baseline any trained arm must beat. It
    also means this check needs no checkpoint and can run before either arm
    finishes.
    """
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from eo.data.eval_split import load_eval_rows
    from eo.generate import conditional as C

    ds = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048, modality_order="fixed")
    rows = [int(r) for r in load_eval_rows(root_dir=ROOT)
            if target in ds.present_modalities(int(r))][:n_scenes]
    model, _ = C.build_model("eo_terramesh/arm_a_mor", None, device="cuda")
    built = [C.build_prompt(ds, r, target) for r in rows]
    prompts = [b["prompt"] for b in built]
    truths = [b["truth"] for b in built]

    masked = C.generate(model, prompts, target, max_new_tokens=max_new_tokens,
                        slot_masked=True, seed=42)
    free = C.generate(model, prompts, target, max_new_tokens=max_new_tokens,
                      slot_masked=False, seed=42)
    s_masked = C.score(masked, truths, target)
    s_free = C.score(free, truths, target)

    in_slot = s_masked["off_slot_rate"] == 0.0
    print(f"    slot-masked generation is 100% in-slot        = {in_slot}")
    no_junk = s_masked["emitted_pad"] == 0 and s_masked["emitted_foreign_bo_eo"] == 0
    print(f"    ...and emits no PAD and no foreign BO/EO      = {no_junk}")

    # CONTROL 1: unconstrained, an UNTRAINED model must sit at chance. If it
    # did not, the off-slot metric would be measuring the mask rather than the
    # model.
    chance = s_free["off_slot_chance"]
    c1 = abs(s_free["off_slot_rate"] - chance) < 0.05
    print(f"    control: untrained off-slot {s_free['off_slot_rate']:.4f} vs chance "
          f"{chance:.4f} = {c1}")

    # CONTROL 2: score the SAME ids against a different modality's slot. It must
    # collapse to ~100% off-slot, proving the check is modality-specific rather
    # than a constant -- the generation counterpart of pointing the run at the
    # wrong registry.
    other = "S2L2A" if target != "S2L2A" else "DEM"
    s_wrong = C.score(masked, truths, other)
    c2 = s_wrong["off_slot_rate"] > 0.95
    print(f"    control: same ids scored as {other} -> off-slot "
          f"{s_wrong['off_slot_rate']:.4f} = {c2}")

    _release(model, None)
    return bool(in_slot and no_junk and c1 and c2)


# ---------------------------------------------------------------------------
# W7 -- decode metrics collapse on shuffled tokens (Step 7)
# ---------------------------------------------------------------------------
#: How far apart the ceiling and the shuffled control must be before the metric
#: counts as measuring spatial structure. 1.25 is deliberately loose: the point
#: is to catch a metric that is BLIND to arrangement, not to grade the decoder.
W7_MARGIN = 1.25

#: Where decode_eo.py writes its metrics artifacts.
STEP7_ROOT = Path(os.environ.get("STEP7_REPORT_ROOT", "/data/enric/reports/phase3_step7"))


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _w7_judge(doc: dict) -> tuple:
    """Pure verdict on one metrics document. Returns (ok, [reasons]).

    Factored out so the controls below can run it against a DOCTORED document
    and prove the check can fail -- CLAUDE.md rule 2. A gate whose controls
    call a different code path than the check is not a gate.
    """
    reasons = []
    # ⚠ Gate on the COLLAPSE metric, which is not the headline metric. For LULC
    # the headline is pixel accuracy and the collapse metric is mIoU: shuffling
    # a class-dominated land-cover scene leaves pixel accuracy at ~0.82 against
    # a 0.99 ceiling, so a margin test on it is unfalsifiable. Measured
    # 2026-09-23; the reasoning is written out in decode_eo.py.
    key = doc.get("collapse_metric") or doc.get("metric")
    summary = doc.get("summary", {})
    for name in ("ceiling", "generated", "shuffled"):
        if name not in summary:
            reasons.append(f"no '{name}' row in the summary")
    if reasons:
        return False, reasons

    # Freshness. The artifact is produced in the other conda env, so this check
    # can only ever read it second-hand; without the hash a months-old file
    # would pass silently after the grids it describes were regenerated.
    prov = doc.get("provenance", {})
    grids = Path(prov.get("decode_dir", "")) / f"decode_grids_{doc.get('target')}.npy"
    if not grids.exists():
        reasons.append(f"the grids it was computed from are gone: {grids}")
    elif _sha256(grids) != prov.get("grids_sha256"):
        reasons.append(f"stale: {grids.name} has changed since these metrics were written")

    if prov.get("crop") != TOK_CROP:
        reasons.append(f"decoded at crop {prov.get('crop')}, contract is {TOK_CROP}")

    c, sh = summary["ceiling"].get(key), summary["shuffled"].get(key)
    if c is None or sh is None:
        reasons.append(f"no '{key}' on the ceiling or shuffled row")
        return False, reasons

    higher_is_better = doc.get("collapse_higher_is_better")
    if higher_is_better is None:
        higher_is_better = key in ("pixel_acc", "miou")
    ok = (c >= sh * W7_MARGIN) if higher_is_better else (sh >= c * W7_MARGIN)
    if not ok:
        reasons.append(
            f"shuffled {key} does not collapse: {sh} against a ceiling of {c} "
            f"(needs a {W7_MARGIN}x margin)")
    return (not reasons), reasons


def w7_decode_collapse(metrics_path=None) -> bool:
    """Decoded metrics collapse on a shuffled token grid (D3.8, plan 7.5).

    ⚠ THIS CHECK CANNOT DECODE. The DiVAE decoders need terratorch, which
    cannot be imported into this env -- that is the whole reason Step 7 is two
    scripts. So W7 verifies the ARTIFACT `decode_eo.py` wrote, and guards
    against the one failure that makes a second-hand check worthless: a stale
    file. The grids' sha256 is recorded at decode time and re-checked here.

    What it asserts: decoding the ground-truth tokens (the 7.4 ceiling) beats
    decoding the SAME tokens spatially permuted, by a margin. If it does not,
    the metric is not measuring spatial arrangement and no other number in
    Step 7 means anything.
    """
    if metrics_path is None:
        hits = sorted(STEP7_ROOT.glob("*/metrics_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not hits:
            print(f"    no metrics artifact under {STEP7_ROOT}")
            print(f"    run: python eo/scripts/prepare_decode.py --gen-dir <gen>/slot_masked")
            print(f"    then (in the `mor` env): python eo/scripts/decode_eo.py --decode-dir <gen>/slot_masked")
            return False
        metrics_path = hits[0]
    doc = json.loads(Path(metrics_path).read_text())
    print(f"    artifact: {metrics_path}")

    ok, reasons = _w7_judge(doc)
    summ = doc["summary"]
    key = doc.get("collapse_metric") or doc["metric"]
    head = doc["metric"]
    print(f"    {doc['target']}: ceiling {key} {summ['ceiling'].get(key)} | "
          f"generated {summ['generated'].get(key)} | shuffled {summ['shuffled'].get(key)}")
    if head != key:
        print(f"    (headline {head}: ceiling {summ['ceiling'].get(head)} | "
              f"generated {summ['generated'].get(head)} | shuffled {summ['shuffled'].get(head)}"
              f" -- reported, not gated on)")
    rep = doc.get("provenance", {}).get("manifest", {}).get("repair", {})
    if rep:
        print(f"    (generated grid was {100 * rep.get('repaired_fraction', 0):.2f}% repaired)")
    for r in reasons:
        print(f"    REASON: {r}")
    print(f"    shuffled collapses against the ceiling       = {ok}")

    # CONTROL 1: a document in which the control did NOT collapse must be
    # rejected. This is the failure the check exists to catch.
    doctored = copy.deepcopy(doc)
    doctored["summary"]["shuffled"][key] = doctored["summary"]["ceiling"][key]
    c1 = not _w7_judge(doctored)[0]
    print(f"    control: shuffled == ceiling is rejected     = {c1}")

    # CONTROL 2: a stale artifact must be rejected. Without this the check
    # would keep passing on a file describing grids that no longer exist.
    stale = copy.deepcopy(doc)
    stale["provenance"]["grids_sha256"] = "0" * 64
    c2 = not _w7_judge(stale)[0]
    print(f"    control: a stale grids hash is rejected      = {c2}")

    return bool(ok and c1 and c2)


def _compose(name, overrides=None):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides or []))


def _flat(cfg, prefix=""):
    """Flatten a config to dotted keys, WITHOUT resolving interpolations.

    ⚠ `resolve=False` is load-bearing. The EO config interpolates
    `${oc.env:WANDB_ENTITY}`, and merely reading the values resolves it, so
    this check used to raise `InterpolationResolutionError` in any shell
    without W&B variables set — it passed only because they happened to be
    exported. Comparing two configs must not require credentials, and the
    comparison is over the interpolation EXPRESSIONS anyway: two arms that both
    say `${oc.env:WANDB_ENTITY}` agree whether or not it is set.
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=False)
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def _arm_diff(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>")}


def w8_arm_configs() -> bool:
    """The two arm configs differ in exactly the intended keys.

    ⚠ This is the check that stops the comparison being confounded before it
    starts. Arms A and B already differ in parameter count and FLOPs per token
    by construction (that is the experiment); any THIRD difference — a stray
    learning rate, a different split, a different seed — makes the Plan 1 /
    Plan 2 call meaningless, and none of those would fail loudly at runtime.
    Two W&B pages side by side will not reveal it either.
    """
    a, b = _compose(ARM_A), _compose(ARM_B)

    diff = _arm_diff(a, b)
    ok = diff == ARM_DIFF_KEYS
    print(f"    arms differ in exactly {sorted(ARM_DIFF_KEYS)}")
    print(f"    measured difference: {sorted(diff)} = {ok}")

    fa, fb = _flat(a), _flat(b)
    match_ok = True
    for k in ARM_MUST_MATCH:
        if k not in fa or k not in fb:
            print(f"    MISSING key in an arm config: {k}")
            match_ok = False
        elif fa[k] != fb[k]:
            print(f"    MISMATCH {k}: A={fa[k]!r} B={fb[k]!r}")
            match_ok = False
    print(f"    all {len(ARM_MUST_MATCH)} load-bearing keys identical  = {match_ok}")

    # CONTROL 1: a stray difference anywhere must be caught.
    c1 = _arm_diff(a, _compose(ARM_B, ["learning_rate=0.001"])) != ARM_DIFF_KEYS
    print(f"    control: a stray learning_rate difference is caught = {c1}")

    # CONTROL 2: pointing the arms at DIFFERENT eval splits must be caught.
    # This is the one that would silently score the arms on different held-out
    # rows, and nothing downstream could detect it.
    b_split = _compose(ARM_B, ["multimodal.eval_split=v2"])
    c2 = _flat(b_split)["multimodal.eval_split"] != fa["multimodal.eval_split"] and \
        _arm_diff(a, b_split) != ARM_DIFF_KEYS
    print(f"    control: different eval_split per arm is caught     = {c2}")

    # CONTROL 3: a different seed must be caught -- it is the quietest of the
    # three, since both runs still look entirely normal.
    c3 = _arm_diff(a, _compose(ARM_B, ["seed=7"])) != ARM_DIFF_KEYS
    print(f"    control: a different seed per arm is caught         = {c3}")

    return bool(ok and match_ok and c1 and c2 and c3)


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
        ("W8   arm configs differ only as intended [Step 4]", w8_arm_configs),
        ("W7   decode metrics collapse on shuffled   [Step 7]", w7_decode_collapse),
    ]
    if not args.skip_forward:
        checks.append(("W3   eval loop runs; untrained ~ ln(V)    [Step 1]", w3_eval_loop))
        checks.append(("W4   per-modality loss in BOTH arms       [Step 5]", w4_symmetric_logging))
        checks.append(("W6   generated ids land in target slot    [Step 6]", w6_generation_slots))
        checks.append(("W5   fp16 matches fp32; scaler control    [Step 2]", w5_fp16_equivalence))

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
        print("  (W9 is eo.train.preflight_controls; W7 reads the artifact decode_eo.py writes)")
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
