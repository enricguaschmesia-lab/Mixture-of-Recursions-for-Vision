#!/usr/bin/env python
"""A2 R1's gate (`.venv`, CPU; A2 Step 5.1): the train LULC map tokens and the mid-training's data, config, init and
training path. Every check ships with a control that must fail it (CLAUDE.md rule 2).

    python eo/scripts/verify_seg_r1.py [--only Z14,Z15a] [--out results.json]

  Z14   The map artifacts (Step 2.6) are the verified ones: eo/experiments/segmentation/step26_verify.py's reports for
        train and val pass (A integrity, B = the harness's own encoding bit for bit, C round trip >= 0.90, each with its
        controls failing), and each artifact's metadata.json still hashes to what its report read; every train shard is
        done and the train maps are exactly the root index's ssl4eos12 rows
                                     | a report read on another artifact (val's for train) is refused
  Z15a  The overlay eo_seg/r1_aligned differs from its base arm in exactly the intended keys; vocab = the map registry;
        bf16; split train, nc, clean, eval split v1, random order inherited
                                     | a stray key (weight_decay) is flagged
  Z15b  Map sequences (mini root): five keys; 6 blocks of 198 (the row's five image blocks, token-identical to
        TerraMeshTokenDataset's, and [BO_MAP_LULC 81,296 + codes EO_MAP] with the row's own codes); modality ids
        (map codes 10, BO/EO 0); labels on every token; positions 0..1,187; the order a permutation drawn from
        (seed, stream, row)
                                     | the next row's codes do not match; a val artifact against the train root
                                       is refused
  Z15c  Base sequences: Major TOM rows only, their five blocks token-identical to TerraMeshTokenDataset's, 990 tokens,
        no map or temporal id
                                     | a map row's sequence carries map ids and is 1,188 long
  Z15d  The mix: even indices map, odd base; 11,880 tokens in every micro-batch, so each step is half and half; the
        pools are the corpora and disjoint; no row repeats within a pass; a micro-batch is a pure function of its index
        (a second instance, and a DataLoader with 2 workers, serve the same tensors); the trainer reads it in index order
                                     | an order keyed on the worker id serves other tensors at another worker count
  Z15e  Eval sets: `single` equals the base arm's own held-out eval set tensor for tensor (4 rows a micro-batch);
        `map` holds the 4,416 held-out rows with their own val map; `map_s2` is [S2L2A][MAP] with labels on the 196
        codes and EO_MAP only, the prompt ending at BO_MAP_LULC
                                     | the next held-out row's codes do not match
  Z15f  The init: verify_init accepts the map base at 161,665 (layout map, the record = map_vocab's now)
                                     | the temporal base at 161,665 and the map base at 81,296 are refused
  Z15g  Preflight's check_map_data passes on the mini root
                                     | the val artifact as train maps; a schedule past the dataset's length; an odd
                                       total_batch_size; another arm; a resume whose record holds another data
                                       setting: each refused
  Z15h  The smoke through pretrain.py, read from its run directory (pulled from RCP; a real micro-batch needs ~30 GB of
        host memory, so it cannot run on the workstation's CPU): every parameter moved from the map base, the map rows
        among them; pretrain_init.json names the map base; temporal_data.json holds this config's data with g49's
        artifacts; every train log finite; eval_single_loss at step 0 equals the base's last logged eval_loss (5.1027)
        within 2e-3; the map CE falls
                                     | check_data_record refuses the record against a changed data setting
  Z15j  The extension's mechanics, on RCP: the smoke resumed in place with a longer schedule (num_train_steps 60, the
        decay moved) continues from step 40 to 60: each eval set's start-of-run eval at step 40 equals the first job's
        step-40 eval exactly (the weights restored), and the learning rate is non-increasing from the new schedule's peak
                                     | the same resume with a changed data setting is refused by preflight
                                       (check_map_data), as its job log shows
  Z15i  The forgetting zero point: on 4 held-out rows the map base's `single` CE equals the temporal base's (the map
        rows carry no mass there) within 1e-4 nats
                                     | the same rows with the map rows' logits raised by 30 move it past 1e-4
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
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eo.data import map_vocab as MV            # noqa: E402

RESULTS: list = []
TM = Path("/data/enric/data/TerraMesh")
MINI_ROOT = TM / "train_tok_partial_r1"
MINI_MAPS = TM / "maps/mini_r1/MAP_aligned"
TRAIN_MAPS = TM / "maps/train/MAP_aligned"
VAL_MAPS = TM / "maps/val/MAP_aligned"
REPORTS = Path("/data/enric/reports/a2_seg/step26")
MAP_BASE = Path("/data/enric/runs/pretrain/a2_seg/base/b10cltrxnc_ext_map_nc_ckpt33000")
TEMPORAL_BASE = Path("/data/enric/runs/pretrain/temporal/base/b10cltrxnc_ext_temporal_nc_ckpt33000")
OVERLAY = "eo_seg/r1_aligned"
BASE_ARM = "eo_terramesh/arm_b10ctrxnc_vanilla"
#: The mini root (one shard a corpus): its 10,000 map rows hold 4 passes of 91 steps at 22 micro-batches a step, so the
#: schedule is cut to fit (the builder refuses a schedule past its length, which Z15g's control shows).
MINI = {"++multimodal.allow_partial_train_root": True, "r1.train_maps": str(MINI_MAPS), "num_train_steps": 300,
        "stop_steps": 300}


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _env():
    from util.env import load_dotenv
    load_dotenv()
    os.environ["TERRAMESH_TRAIN_TOK_ROOT"] = str(MINI_ROOT)
    os.environ.setdefault("TERRAMESH_TOK_ROOT", str(TM / "val"))


def _compose(name: str, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def _ov(d: dict) -> list:
    return [f"{k}={'true' if v is True else 'false' if v is False else v}" for k, v in d.items()]


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def _cfg(extra: dict | None = None):
    return _compose(OVERLAY, _ov({**MINI, **(extra or {})}))


# ---------------------------------------------------------------- Z14
def z14() -> None:
    import pandas as pd
    for split, art in (("train", TRAIN_MAPS), ("val", VAL_MAPS)):
        rp = REPORTS / f"verify_{split}_full.json"
        if not rp.is_file():
            record(f"Z14 {split} report", False, f"{rp} missing (eo/experiments/segmentation/step26_verify.py --tag full)")
            continue
        rep = json.loads(rp.read_text())
        meta_now = json.loads((art / "metadata.json").read_text())
        same_art = rep["artifact"] == str(art)
        ch = rep["checks"]
        ok = (rep["pass"] and same_art and ch["A_integrity"]["pass"] and ch["B_harness"]["pass"] and ch["C_roundtrip"]["pass"]
              and ch["A_integrity"]["absent_rows_nonzero"] == 0 and meta_now.get("complete")
              and Path(rep["card"]["index"]).resolve() == Path(meta_now["index"]).resolve()
              and rep["card"]["index_sha256"] == meta_now["index_sha256"])
        record(f"Z14 {split} maps verified", ok,
               f"report {rp.name}: A {ch['A_integrity']['pass']} ({ch['A_integrity']['n_present']} rows, absent rows "
               f"zero), B {ch['B_harness']['codes_equal']:.4f} of codes equal on {ch['B_harness']['rows']} rows (controls "
               f"{ch['B_harness']['control_B1_one_row_off_agreement']:.3f} / {ch['B_harness']['control_B2_transposed_agreement']:.3f} "
               f"/ {ch['B_harness']['control_B3_colours_rotated_agreement']:.3f}), C mIoU {ch['C_roundtrip']['mIoU']:.4f} "
               f"(control {ch['C_roundtrip']['control_C_next_row_mIoU']:.3f}); artifact {art}, complete "
               f"{meta_now.get('complete')}, {meta_now.get('n_present')} maps, card {meta_now.get('gpu')}")
        # the artifact the report read is this one, unchanged since: its metadata hashes as then, and every done shard's
        # tokens, rehashed now, equal its done record
        idx_a = pd.read_parquet(meta_now["index"], columns=["row", "source_shard"])
        tokens = np.load(art / "tokens.npy", mmap_mode="r")
        bad = []
        done = sorted((art / "done").glob("*.json"))
        for d in done:
            rec = json.loads(d.read_text())
            blk = np.asarray(tokens[rec["row_first"]:rec["row_last"] + 1]) if rec["contiguous"] else \
                np.asarray(tokens[np.sort(idx_a[idx_a.source_shard == rec["shard"]].row.values)])
            if hashlib.sha256(np.ascontiguousarray(blk).tobytes()).hexdigest() != rec["tokens_sha256"]:
                bad.append(rec["shard"])
        same_meta = rep.get("metadata_sha256") == sha256_file(art / "metadata.json")
        record(f"Z14 {split} unchanged since its report", not bad and same_meta and meta_now["shards_done"] == len(done)
               and meta_now["n_present"] == int(np.load(art / "present.npy").sum()),
               f"metadata.json hashes as the report read it ({same_meta}); {len(done)} shards rehashed, {len(bad)} differ")
    # train maps = exactly the train index's ssl4eos12 rows
    meta = json.loads((TRAIN_MAPS / "metadata.json").read_text())
    idx = pd.read_parquet(meta["index"], columns=["row", "corpus"]).sort_values("row")
    pres = np.load(TRAIN_MAPS / "present.npy")
    ssl = idx.corpus.values == "ssl4eos12"
    record("Z14 train maps = the ssl4eos12 rows", bool((pres == ssl).all()) and int(pres.sum()) == 958_176,
           f"{int(pres.sum())} present = {int(ssl.sum())} ssl4eos12 rows of {len(idx)}; Major TOM rows absent")
    # control: the val report read against the train artifact
    rv = json.loads((REPORTS / "verify_val_full.json").read_text())
    record("Z14 control: another artifact's report", rv["artifact"] != str(TRAIN_MAPS),
           f"val's report names {rv['artifact']}, refused for {TRAIN_MAPS}")


# ---------------------------------------------------------------- Z15a
def z15a() -> None:
    from omegaconf import OmegaConf
    from lm_dataset.modality_registry import get_registry
    c = _compose(OVERLAY)
    b = _compose(BASE_ARM)
    f, g = _flat(OmegaConf.to_container(c, resolve=False)), _flat(OmegaConf.to_container(b, resolve=False))
    gone = object()                               # absent and null are different settings
    diff = {k for k in set(f) | set(g) if f.get(k, gone) != g.get(k, gone)}
    intended = {"dataset", "model_config.vocab_size", "max_length", "total_batch_size", "per_device_train_batch_size",
                "per_device_eval_batch_size", "learning_rate", "num_train_steps", "stop_steps", "num_warmup_steps",
                "lr_scheduler_kwargs.num_decay_steps", "save_steps", "eval_steps", "fixed_eval_steps",
                "fixed_save_steps", "dataloader_prefetch_factor", "wandb_project"} | \
               {f"r1.{k}" for k in ("arm", "task", "train_maps", "val_maps", "map_per_microbatch", "base_per_microbatch",
                                    "eval_per", "eval_map_per", "eval_seed", "passes")} | \
               {f"pretrain_init.{k}" for k in ("conversion", "variant", "freeze_body", "from_stage1")}
    mm = c.multimodal
    ok = (diff == intended or diff == intended - {"wandb_project"}) and \
        c.model_config.vocab_size == get_registry(c).TOTAL_VOCAB_SIZE == MV.TOTAL_VOCAB_SIZE and c.precision == "bf16" \
        and mm.split == "train" and mm.vocab == "nc" and mm.token_set == "clean" and mm.eval_split == "v1" \
        and mm.modality_order == "random" and list(mm.active_modalities) == ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"] \
        and c.mixed_precision is True and int(c.max_length) == 1188 and c.num_train_steps % c.save_steps == 0
    record("Z15a overlay", ok, f"differs from {BASE_ARM} in {len(diff)} keys (intended {len(intended)}; extra "
           f"{sorted(diff - intended)}, missing {sorted(intended - diff)}); vocab {c.model_config.vocab_size} = map "
           f"registry; {c.precision}; split {mm.split}, {mm.vocab}, {mm.token_set}, eval {mm.eval_split}, order "
           f"{mm.modality_order}; stop a multiple of save_steps")
    s = _flat(OmegaConf.to_container(_compose(OVERLAY, ["weight_decay=0.05"]), resolve=False))
    d2 = {k for k in set(s) | set(g) if s.get(k, gone) != g.get(k, gone)}
    record("Z15a control: a stray key", d2 - intended != set(), f"flagged: {sorted(d2 - intended)}")


# ---------------------------------------------------------------- Z15b/c/d
def _train(extra=None):
    from eo.data import map_mix_dataset as MD
    return MD.build_train(_cfg(extra))


def blocks(ids: torch.Tensor):
    return [ids[i:i + 198] for i in range(0, len(ids), 198)]


def z15b() -> None:
    from eo.data import map_mix_dataset as MD
    ds = _train()
    sc = ds.scenes
    stream, rows = ds.members(0)
    mb = ds[0]
    probs, ok_ids, ok_map = [], 0, 0
    eo_by_bo = {}
    for m in sc.base.active_modalities:
        info = sc.base._V.get_modality(m)
        eo_by_bo[info.bo_id] = (m, info.eo_id)
    bo_map = MV.BO_MAP["LULC"]
    for j, r in enumerate(rows.tolist()):
        ids, mid = mb["input_ids"][j], mb["modality_ids"][j]
        if tuple(sorted(mb)) != tuple(sorted(("input_ids", "attention_mask", "labels", "position_ids", "modality_ids"))):
            probs.append("keys")
        if not (torch.equal(mb["labels"][j], ids) and int(mb["attention_mask"][j].sum()) == 1188
                and torch.equal(mb["position_ids"][j], torch.arange(1188))):
            probs.append(f"row {r}: labels/mask/positions")
        names = []
        for k, blk in enumerate(blocks(ids)):
            bo = int(blk[0])
            if bo == bo_map:
                names.append("MAP")
                want = torch.tensor([bo_map] + [MV.SLOTS["aligned"][0] + int(c) for c in sc.maps.codes(r)] + [MV.EO_MAP])
                if torch.equal(blk, want):
                    ok_map += 1
                body = mid[k * 198 + 1:k * 198 + 197]
                if not (bool((body == MV.MAP_MODALITY_ID).all()) and int(mid[k * 198]) == 0 and int(mid[k * 198 + 197]) == 0):
                    probs.append(f"row {r}: map modality ids")
            else:
                m, eo = eo_by_bo[bo]
                names.append(m)
                if torch.equal(blk, sc.base._load_chunk(m, r)) and int(blk[-1]) == eo:
                    ok_ids += 1
        if sorted(names) != sorted(sc.base.present_modalities(r) + ["MAP"]) or names != sc.order(r, True, "map"):
            probs.append(f"row {r}: blocks {names}")
    n = len(rows)
    record("Z15b map sequences", stream == "map" and not probs and ok_map == n and ok_ids == 5 * n
           and tuple(mb["input_ids"].shape) == (10, 1188),
           f"micro-batch 0: {n} ssl4eos12 rows x 1,188; {ok_map}/{n} map blocks = [BO_MAP_LULC, 81,296 + the row's codes, "
           f"EO_MAP]; {ok_ids}/{5 * n} image blocks = TerraMeshTokenDataset's chunks; map ids 10, BO/EO 0; labels on "
           f"every token; positions 0..1187; orders drawn from (seed, stream, row); problems {probs[:3]}")
    # control: the next row's codes
    r0, r1 = int(rows[0]), int(rows[1])
    seq = mb["input_ids"][0]
    k = [i for i, b in enumerate(blocks(seq)) if int(b[0]) == bo_map][0]
    other = torch.tensor([MV.SLOTS["aligned"][0] + int(c) for c in sc.maps.codes(r1)])
    record("Z15b control: another row's codes", not torch.equal(blocks(seq)[k][1:-1], other),
           f"row {r0}'s map block against row {r1}'s codes: {int((blocks(seq)[k][1:-1] == other).sum())}/196 equal")
    try:
        MD.MapArtifact(VAL_MAPS, MINI_ROOT, "aligned")
        record("Z15b control: val maps against the train root", False, "accepted")
    except RuntimeError as e:
        record("Z15b control: val maps against the train root", "index" in str(e), f"refused: {str(e)[:120]}")


def z15c() -> None:
    ds = _train()
    sc = ds.scenes
    stream, rows = ds.members(1)
    mb = ds[1]
    good, probs = 0, []
    for j, r in enumerate(rows.tolist()):
        ids = mb["input_ids"][j]
        chunks = {m: sc.base._load_chunk(m, r) for m in sc.base.present_modalities(r)}
        bl = blocks(ids)
        if len(ids) != 990 or len(bl) != 5:
            probs.append(f"row {r}: {len(ids)} tokens")
            continue
        names = sc.order(r, False, "base")
        if all(torch.equal(b, chunks[n]) for b, n in zip(bl, names)) and int(ids.max()) < 81_188:
            good += 1
    import pandas as pd
    corp = pd.read_parquet(MINI_ROOT / "tok_index.parquet").sort_values("row").corpus.values
    record("Z15c base sequences", stream == "base" and good == len(rows) == 12 and not probs
           and set(corp[rows]) == {"majortom"} and "S1RTC" in sc.base.present_modalities(int(rows[0])),
           f"micro-batch 1: {good}/12 Major TOM rows, 990 tokens, five blocks = TerraMeshTokenDataset's chunks in the drawn "
           f"order, no id >= 81,188 (no map or temporal id); problems {probs[:3]}")
    ctrl = ds[0]["input_ids"][0]
    record("Z15c control: a map row's sequence", len(ctrl) == 1188 and int(ctrl.max()) >= MV.BASE_VOCAB,
           f"{len(ctrl)} tokens, max id {int(ctrl.max())} >= {MV.BASE_VOCAB}: not a base sequence")


def z15d() -> None:
    import pandas as pd
    from eo.data import map_mix_dataset as MD
    from util.trainer_pt import PerModalityLossMixin
    ds = _train()
    corp = pd.read_parquet(MINI_ROOT / "tok_index.parquet").sort_values("row").corpus.values
    specs = [ds.spec(i) for i in range(6)]
    tok = {s: (ds.per[s] * (MD.L_MAP if s == "map" else MD.L_BASE)) for s in ("map", "base")}
    pure = set(corp[ds.pools["map"]]) == {"ssl4eos12"} and set(corp[ds.pools["base"]]) == {"majortom"} \
        and not len(np.intersect1d(ds.pools["map"], ds.pools["base"]))
    seen = np.concatenate([ds.members(2 * k)[1] for k in range(min(200, len(ds.pools["map"]) // ds.per["map"]))])
    norep = len(np.unique(seen)) == len(seen)
    ds2 = _train()
    same = all(torch.equal(ds[i][k], ds2[i][k]) for i in (0, 1, 2, 3, 100, 101) for k in ds[i])
    from torch.utils.data import DataLoader, Subset
    sub = list(range(8))
    dl0 = [b for b in DataLoader(Subset(ds, sub), batch_size=1, collate_fn=ds.collate_fn, num_workers=0)]
    dl2 = [b for b in DataLoader(Subset(ds, sub), batch_size=1, collate_fn=ds.collate_fn, num_workers=2)]
    workers = all(torch.equal(a[k], b[k]) for a, b in zip(dl0, dl2) for k in a)

    class FakeTrainer(PerModalityLossMixin):
        train_dataset = ds
    sampler = FakeTrainer()._get_train_sampler()
    seq = isinstance(sampler, torch.utils.data.SequentialSampler)
    record("Z15d the mix", specs == [("map", 0), ("base", 0), ("map", 1), ("base", 1), ("map", 2), ("base", 2)]
           and tok["map"] == tok["base"] == 11_880 and pure and norep and same and workers and seq
           and ds.token_share() == {"map": 0.5, "base": 0.5},
           f"even indices map, odd base; {tok['map']} = {tok['base']} tokens a micro-batch, so a step of "
           f"{ds.M} is half and half (token share {ds.token_share()}); pools {len(ds.pools['map'])} ssl4eos12 / "
           f"{len(ds.pools['base'])} Major TOM, disjoint; {len(seen)} map rows in the first micro-batches, none repeated; "
           f"a second instance {same}, 2 workers {workers}; the trainer's sampler {type(sampler).__name__}")

    class WorkerKeyed(MD.MapScenes):            # the control: an order keyed on the worker id (CLAUDE.md, the dataloader)
        def order(self, row, with_map, stream, seed=None):
            wi = torch.utils.data.get_worker_info()
            w = wi.id if wi is not None else 0
            mods = self.modalities(row, with_map)
            rng = np.random.default_rng([int(self.seed), w, int(row)])
            return [mods[i] for i in rng.permutation(len(mods))]
    bad = MD.MapMixMicrobatches(WorkerKeyed(ds.scenes.base, ds.scenes.maps, ds.cfg, ds.seed), ds.pools["map"],
                                ds.pools["base"], ds.cfg, ds.M, ds.seed, ds.max_length)
    b0 = [b for b in DataLoader(Subset(bad, sub), batch_size=1, collate_fn=bad.collate_fn, num_workers=0)]
    b2 = [b for b in DataLoader(Subset(bad, sub), batch_size=1, collate_fn=bad.collate_fn, num_workers=2)]
    record("Z15d control: a worker-keyed order", not all(torch.equal(a[k], b[k]) for a, b in zip(b0, b2) for k in a),
           "serves other tensors at 2 workers than at 0")


# ---------------------------------------------------------------- Z15e
def z15e() -> None:
    from eo.data import map_mix_dataset as MD
    from lm_dataset.load_dataset import load_eval_dataset_from_config
    c = _cfg()
    ev = MD.build_eval(c)
    base_ev = load_eval_dataset_from_config(_compose(BASE_ARM))
    single = ev["single"]
    same = all(torch.equal(single[m][k], torch.stack([base_ev[4 * m + i][k] for i in range(4)]))
               for m in (0, 1, 2, 500, len(single) - 1) for k in ("input_ids", "labels", "attention_mask",
                                                                   "position_ids", "modality_ids"))
    record("Z15e single = the base's eval set", same and len(single) * 4 == len(base_ev) == 4416
           and tuple(single[0]["input_ids"].shape) == (4, 990),
           f"{len(single)} micro-batches of 4 = the base arm's {len(base_ev)} held-out rows, tensor for tensor at 990")
    mp, s2 = ev["map"], ev["map_s2"]
    rows = mp.rows
    sc = mp.scenes
    bo_map = MV.BO_MAP["LULC"]
    off = MV.SLOTS["aligned"][0]
    ok_map, ok_s2, probs = 0, 0, []
    pick = list(range(0, len(rows), 551))[:8]
    for i in pick:
        r = int(rows[i])
        m, j = divmod(i, mp.per)
        ids = mp[m]["input_ids"][j]
        lab = mp[m]["labels"][j]
        k = [q for q, b in enumerate(blocks(ids)) if int(b[0]) == bo_map]
        want = torch.tensor([off + int(c) for c in sc.maps.codes(r)])
        if len(k) == 1 and torch.equal(blocks(ids)[k[0]][1:-1], want) and torch.equal(lab, ids) and len(ids) == 1188:
            ok_map += 1
        t = s2[m]
        sid, slab = t["input_ids"][j], t["labels"][j]
        lab_pos = (slab != -100).nonzero().flatten()
        if (len(sid) == 396 and int(sid[198]) == bo_map and torch.equal(sid[199:395], want) and int(sid[395]) == MV.EO_MAP
                and lab_pos.tolist() == list(range(199, 396)) and torch.equal(slab[199:396], sid[199:396])
                and torch.equal(sid[:198], sc.base._load_chunk("S2L2A", r))):
            ok_s2 += 1
        else:
            probs.append(r)
    record("Z15e map and map_s2", ok_map == ok_s2 == len(pick) and len(rows) == 4416 and len(mp) == len(s2) == 276
           and mp.per == s2.per == 16,
           f"{ok_map}/{len(pick)} sampled held-out rows carry their own val map in `map` (1,188 tokens, labels on every "
           f"token); {ok_s2}/{len(pick)} `map_s2` = [S2L2A][BO_MAP_LULC codes EO_MAP], labels on positions 199..395 (the "
           f"196 codes and EO_MAP), the prompt ending at BO_MAP_LULC; problems {probs[:3]}")
    r, r1 = int(rows[pick[1]]), int(rows[pick[1] + 1])
    m, j = divmod(pick[1], s2.per)
    got = ev["map_s2"][m]["input_ids"][j][199:395]
    other = torch.tensor([off + int(c) for c in sc.maps.codes(r1)])
    record("Z15e control: the next held-out row's codes", not torch.equal(got, other),
           f"row {r} against row {r1}'s codes: {int((got == other).sum())}/196 equal")


# ---------------------------------------------------------------- Z15f
def z15f() -> None:
    from eo.train.clevr_init import verify_init
    plan = {"variant": "continue", "conversion": MAP_BASE, "freeze_body": False, "from_stage1": None, "lora": None}
    d = verify_init(plan, MV.TOTAL_VOCAB_SIZE)
    record("Z15f init: the map base", d["layout"] == "map" and d["map_vocab"] == MV.LAYOUT.record()
           and d["extension"]["vocab_after"] == MV.TOTAL_VOCAB_SIZE,
           f"accepted at {MV.TOTAL_VOCAB_SIZE}: layout {d['layout']}, record = map_vocab's, weights sha {d['weights_sha256'][:12]} "
           f"(hash checked)")
    for name, cv, voc in (("the temporal base at 161,665", TEMPORAL_BASE, MV.TOTAL_VOCAB_SIZE),
                          ("the map base at 81,296", MAP_BASE, MV.BASE_VOCAB)):
        try:
            verify_init({**plan, "conversion": cv}, voc, check_hash=False)
            record(f"Z15f control: {name}", False, "accepted")
        except RuntimeError as e:
            record(f"Z15f control: {name}", True, f"refused: {str(e)[:110]}")


# ---------------------------------------------------------------- Z15g
def z15g() -> None:
    from eo.train import preflight as PF
    path = REPO / "conf/pretrain_vision" / f"{OVERLAY}.yaml"
    env = dict(os.environ)

    def run(extra=None, resuming=False, run_dir=None):
        ctx = PF.PreflightContext(gpu_alias="a100", run_dir=run_dir or Path("/data/enric/runs/pretrain/a2_seg/_gate"),
                                  config_path=path, overrides={**MINI, **(extra or {})}, env=env, check_torch=False,
                                  resuming=resuming, allow_existing=resuming)
        return PF.check_map_data(ctx)
    r = run()
    record("Z15g preflight check_map_data", r.ok, r.detail)
    for name, extra in (("the val artifact as train maps", {"r1.train_maps": str(VAL_MAPS)}),
                        ("a schedule past the dataset", {"num_train_steps": 10 ** 7, "stop_steps": 10 ** 7}),
                        ("an odd total_batch_size", {"total_batch_size": 21}),
                        ("another arm", {"r1.arm": "short"})):
        r = run(extra)
        record(f"Z15g control: {name}", not r.ok, f"refused: {r.detail[:120]}")
    # a resume whose record holds another data setting
    import tempfile
    from omegaconf import OmegaConf
    from eo.data import map_mix_dataset as MD
    from eo.data import temporal_dataset as TD
    from eo.terramesh_tok.clean import resolve_token_root
    c = _cfg()
    tm = MD.MapArtifact(MINI_MAPS, MINI_ROOT)
    vm = MD.MapArtifact(VAL_MAPS, resolve_token_root(os.environ["TERRAMESH_TOK_ROOT"], "clean"))
    with tempfile.TemporaryDirectory(dir="/data/enric/runs") as tmp:
        rec = MD.data_settings(c, tm, vm)
        (Path(tmp) / TD.DATA_RECORD).write_text(json.dumps(rec))
        ok_same = run(resuming=True, run_dir=Path(tmp)).ok
        rec["r1"]["eval_seed"] += 1
        (Path(tmp) / TD.DATA_RECORD).write_text(json.dumps(rec))
        r = run(resuming=True, run_dir=Path(tmp))
    record("Z15g resume: the same data", ok_same, "a record of this config's data passes")
    record("Z15g control: a resume into other data", not r.ok, f"refused: {r.detail[:140]}")


# ---------------------------------------------------------------- Z15h, Z15j (read the RCP smoke's artifacts)
ARGS = None
BASE_EVAL_LOSS = 5.102718830108643        # B10cL-tr-x10-NC's logged eval_loss at step 33,000 (bf16, its trainer_state)


def _state(run: Path) -> dict:
    ck = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return json.loads((ck[-1] / "trainer_state.json").read_text()), ck[-1]


def z15h() -> None:
    from eo.finetune.init import _load_state, _weights
    from eo.data import map_mix_dataset as MD
    from eo.data import temporal_dataset as TD
    if ARGS is None or not ARGS.smoke_run:
        record("Z15h", False, "needs --smoke-run <the RCP smoke's run directory, pulled>")
        return
    run = Path(ARGS.smoke_run)
    st, ck = _state(run)
    after, before = _load_state(_weights(ck)), _load_state(_weights(MAP_BASE))
    moved = [k for k in before if k in after and not torch.equal(after[k], before[k])]
    unmoved = [k for k in before if k in after and torch.equal(after[k], before[k])]
    emb = "model.embed_tokens.weight"
    map_rows_moved = bool((after[emb][MV.BASE_VOCAB:] != before[emb][MV.BASE_VOCAB:]).any())
    pi = json.loads((run / "pretrain_init.json").read_text())
    rec = json.loads((run / TD.DATA_RECORD).read_text())
    ext = json.loads((MAP_BASE / "extension.json").read_text())
    h = st["log_history"]
    train = [x for x in h if "loss" in x and "grad_norm" in x]
    ev = {}                                       # every eval dict of a step merged (totals, then per modality)
    for x in h:
        if any(k.startswith("eval_") for k in x):
            ev.setdefault(int(x["step"]), {}).update(x)
    single = {s: x["eval_single_loss"] for s, x in ev.items() if "eval_single_loss" in x}
    mp = {s: x["eval_map_loss_MAP"] for s, x in ev.items() if "eval_map_loss_MAP" in x}
    finite = all(np.isfinite([x["loss"], x["grad_norm"]]).all() for x in train)
    # the record holds the data of the config with g49's artifacts, which are the workstation's byte for byte
    tm = MD.MapArtifact(TRAIN_MAPS, TM / "train_coords", "aligned")
    vm = MD.MapArtifact(VAL_MAPS, TM / "val_clean", "aligned")
    want = MD.data_settings(_compose(OVERLAY), tm, vm)
    changes = TD.data_record_changes(rec, want)
    ok = (not unmoved and len(moved) == len(before) and map_rows_moved and pi["variant"] == "continue"
          and pi["conversion_weights_sha256"] == ext["weights_sha256"] and finite and train
          and 0 in single and abs(single[0] - BASE_EVAL_LOSS) < 2e-3 and 0 in mp and mp[max(mp)] < mp[0]
          and not changes)
    record("Z15h the smoke through pretrain.py (RCP)", ok,
           f"{run.name}, {ck.name}: {len(moved)}/{len(before)} tensors moved from the map base (unmoved {unmoved[:3]}), "
           f"map rows moved {map_rows_moved}; pretrain_init.json continue from {pi['conversion_weights_sha256'][:12]} "
           f"(= the map base); {len(train)} train logs finite {finite}; eval_single_loss at step 0 {single.get(0)} "
           f"against the base's last logged {BASE_EVAL_LOSS:.5f}; map CE {mp}; {TD.DATA_RECORD} = this config's data "
           f"with g49's artifacts ({len(changes)} differences, e.g. {changes[:2]})")
    rec2 = json.loads(json.dumps(rec))
    rec2["r1"]["eval_seed"] += 1
    record("Z15h control: a changed data setting", bool(TD.data_record_changes(rec, rec2)),
           f"check_data_record would refuse: {TD.data_record_changes(rec, rec2)}")


def z15j() -> None:
    if ARGS is None or not ARGS.resume_run:
        record("Z15j", False, "needs --resume-run <the smoke resumed with a longer schedule, pulled> and --refused-log")
        return
    run = Path(ARGS.resume_run)
    st, ck = _state(run)
    h = st["log_history"]
    lr = {int(x["step"]): float(x["learning_rate"]) for x in h if "learning_rate" in x}
    peak = 3e-4
    tail = [lr[k] for k in sorted(lr) if k > 40]
    # the resumed job evaluates at its start (eval_on_start), at step 40: each set's eval there must appear twice, equal
    # to the digit (HF's info lines, "Continuing training from global step", are silenced in these logs)
    twice = {}
    for name in ("eval_single_loss", "eval_map_loss", "eval_map_s2_loss"):
        at40 = [x[name] for x in h if name in x and int(x["step"]) == 40]
        twice[name] = (len(at40) == 2 and at40[0] == at40[1], at40)
    ok = (st["global_step"] == st["max_steps"] == 60 and all(v[0] for v in twice.values()) and tail
          and all(b <= a + 1e-15 for a, b in zip(tail, tail[1:])) and tail[0] <= peak + 1e-12 and ck.name == "checkpoint-60")
    record("Z15j extension resume (RCP)", ok,
           f"{run.name}: resumed from checkpoint-40 to {ck.name} (global step {st['global_step']} of {st['max_steps']}); each "
           f"set's eval at step 40 before and after the resume: { {k: v[1] for k, v in twice.items()} }; logged LR after "
           f"40 { {k: f'{v:.2e}' for k, v in sorted(lr.items()) if k > 40} } (the new schedule: stable, then its decay)")
    ref = Path(ARGS.refused_log).read_text(errors="replace") if ARGS.refused_log else ""
    refused = "records other data" in ref and "map data" in ref
    record("Z15j control: a resume into other data", refused,
           "preflight's check_map_data refused it: " + " | ".join(l.strip() for l in ref.splitlines()
                                                                  if "records other data" in l)[:200])


# ---------------------------------------------------------------- Z15i
def z15i() -> None:
    from eo.data import map_mix_dataset as MD
    from transformers import LlamaConfig, LlamaForCausalLM
    from eo.finetune.init import _load_state, _weights
    ev = MD.build_eval(_cfg())["single"]
    batch = {k: ev[0][k] for k in ("input_ids", "attention_mask", "labels", "position_ids")}   # 4 rows: 2.6 GB of fp32 logits

    def ce(model_dir, bump=0.0):
        cfgd = json.loads((model_dir / "config.json").read_text())
        model = LlamaForCausalLM(LlamaConfig(**{k: v for k, v in cfgd.items() if k not in ("architectures",
                                                                                              "transformers_version")}))
        model.load_state_dict(_load_state(_weights(model_dir)), strict=False)
        model.tie_weights()
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           position_ids=batch["position_ids"]).logits.float()
        if bump:
            logits[..., MV.BASE_VOCAB:] += bump
        lab = batch["labels"][:, 1:]
        return float(torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab.reshape(-1),
                                                       ignore_index=-100))
    a, b = ce(TEMPORAL_BASE), ce(MAP_BASE)
    record("Z15i forgetting zero point", abs(a - b) < 1e-4,
           f"4 held-out rows of `single` (3,960 tokens): temporal base {a:.7f}, map base {b:.7f}, |diff| {abs(a - b):.2e} nats")
    c = ce(MAP_BASE, bump=30.0)
    record("Z15i control: map rows given mass", abs(c - a) > 1e-4, f"map-row logits + 30: {c:.4f} (diff {c - a:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke-run", default=None, help="Z15h: the RCP smoke's run directory, pulled")
    ap.add_argument("--resume-run", default=None, help="Z15j: the smoke resumed with a longer schedule, pulled")
    ap.add_argument("--resume-log", default=None, help="Z15j: that resume's job log")
    ap.add_argument("--refused-log", default=None, help="Z15j: the job log of the resume with a changed data setting")
    a = ap.parse_args()
    global ARGS
    ARGS = a
    torch.set_num_threads(max(1, os.cpu_count() - 0))
    _env()
    checks = {"Z14": z14, "Z15a": z15a, "Z15b": z15b, "Z15c": z15c, "Z15d": z15d, "Z15e": z15e, "Z15f": z15f,
              "Z15g": z15g, "Z15i": z15i, "Z15h": z15h, "Z15j": z15j}
    t0 = time.time()
    for k, fn in checks.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            fn()
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
