#!/usr/bin/env python
"""W36 -- the evaluation path scores a Coords-free (nc) checkpoint on its own layout, and refuses a mismatch.

The init comparison (2026-11-14) trains B10c-tr-x10-NC and B10cL-tr-x10-NC on the nc vocabulary
(eo_vocab_nc: 81,188 ids, v1's image codebooks and modality ids, BO/EO after them, PAD last, no Coords).
eval_checkpoint.py and eo/eval/teacher_forced.py read the layout from the data (`TF.layout_of`) and from
the model's vocab_size (`TF.layout_for_model`), and refuse the two apart (`TF.check_layout`). This gate
holds that on the nc arm's held-out set (val's 4,416 rows), CPU only. Every part has a control that must
fail it (CLAUDE.md rule 2):

  plumbing   the nc arm's held-out dataset serves eo_vocab_nc and B10c-tr's serves eo_vocab; the first
             64 held-out sequences, through the loader's own settings: every id below 81,188, PAD 81,187
             exactly where the attention mask is 0, no modality id 7, every BO/EO an nc delimiter.
             Control: the v1 dataset's first sequences hold ids at or above 81,188 (its delimiters).
  targetlast TargetLast on the nc dataset, each of the six targets on 16 corpus-stratified rows: ids
             below 81,188, the target's chunk last ([its nc BO, body, its nc EO], body modality id the
             target's). A full nc row is 990 = max_length tokens and holds no padding, so the grid's
             one-source (DEM) and target-alone layouts are checked for PAD 81,187 after the real tokens. A
             Coords target is refused. Control: DEM -> S2L2A assembled with v1's PAD (what TargetLast did
             before 2026-11-15) holds ids past an nc model's table.
  drop       DropModalities(nc dataset, [S1RTC]) on rows that carry S1RTC: the chunk [nc BO, body, nc EO]
             gone, everything else in order, padded with 81,187. Controls: drop_chunks with v1's layout
             on the same nc rows raises (v1's S1RTC BO/EO are not there, its body ids are); dropping
             Coords from the nc dataset is refused.
  layout     check_layout: nc data and an 81,188-row model pass; v1 data and an 87,556-row model pass, and
             so does a wider v1 model (the temporal stage appends rows). Controls: nc data with an
             87,556-row model, v1 data with an 81,188-row model, and v1 data with a narrower model are
             refused.
  run_pass   the untrained nc arm built on CPU (eo/generate/conditional.build_model), 8 held-out rows at
             fp32, 4 majortom and 4 ssl4eos12 (majortom has no S1GRD): the per-modality keys are exactly the six image modalities, every CE within 1 nat of
             ln 81,188 = 11.30, and each slot mass within 2x of its codebook's share of the table; the
             slot bounds used fit inside the logits. Control: v1's slot bounds reach past 81,188 rows.

    .venv/bin/python eo/scripts/gate_nc_eval.py        # ~2 min on CPU

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ARM_NC = "eo_terramesh/arm_b10ctrxnc_vanilla"
ARM_V1 = "eo_terramesh/arm_b10ctr_vanilla"
ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
IMAGE = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]


def _compose(name, overrides=None):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides or []))


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, KeyError):
        return True
    return False


def _model(n):
    return SimpleNamespace(config=SimpleNamespace(vocab_size=n))


def _say(label: str, ok: bool) -> bool:
    print(f"    {label:<78} = {ok}")
    return ok


def check() -> bool:
    from eo.data import eo_vocab as V1, eo_vocab_nc as NC
    from eo.data.eval_split import load_eval_rows, load_row_table
    from eo.eval import teacher_forced as TF
    from lm_dataset.sequence_assembly import assemble_sequence

    ok = True
    cfg = _compose(ARM_NC)
    nw = int(cfg.dataloader_num_workers)
    ds, loader = TF.eval_loader(cfg, num_workers=nw)
    ds1, loader1 = TF.eval_loader(_compose(ARM_V1), num_workers=nw)
    nc_delims = {i for m in NC.MODALITIES.values() for i in (m.bo_id, m.eo_id)}

    # ---- plumbing
    ok &= _say("the nc arm's held-out data serve eo_vocab_nc, B10c-tr's eo_vocab",
               TF.layout_of(ds) is NC and TF.layout_of(ds1) is V1)
    seqs, n = [], 0
    for b in loader:
        seqs.append(b)
        n += b["input_ids"].shape[0]
        if n >= 64:
            break
    ids = torch.cat([b["input_ids"] for b in seqs]); att = torch.cat([b["attention_mask"] for b in seqs])
    mods = torch.cat([b["modality_ids"] for b in seqs])
    delim = (mods == 0) & (att == 1)
    p_ok = (int(ids.max()) < NC.TOTAL_VOCAB_SIZE and bool((ids[att == 0] == NC.PAD_ID).all())
            and bool((ids[att == 1] != NC.PAD_ID).all()) and not bool((mods == 7).any())
            and set(ids[delim].unique().tolist()) <= nc_delims)
    ok &= _say(f"{n} nc held-out sequences: ids < 81,188, PAD 81,187 off the mask, nc BO/EO, no id 7", p_ok)
    b1 = next(iter(loader1))
    ok &= _say("control: v1's held-out sequences hold ids >= 81,188 (its delimiters)",
               int(b1["input_ids"].max()) >= NC.TOTAL_VOCAB_SIZE)

    # ---- targetlast
    corpus = load_row_table(ROOT).corpus.values
    eval_rows = load_eval_rows(root_dir=ROOT)
    t_ok = True
    for T in IMAGE:
        rows = TF.stratified_target_rows(ds, eval_rows, corpus, T, 16)
        info, mid = NC.get_modality(T), NC.MODALITY_TO_ID[T]
        for i in range(len(rows)):
            it = TF.TargetLast(ds, rows, T)[i]
            x, a, m = it["input_ids"], it["attention_mask"], it["modality_ids"]
            k = int(a.sum())
            body = x[k - 1 - info.tokens_per_sample:k - 1]
            good = (len(rows) == 16 and int(x.max()) < NC.TOTAL_VOCAB_SIZE and bool((x[k:] == NC.PAD_ID).all())
                    and int(x[k - 1]) == info.eo_id and int(x[k - 2 - info.tokens_per_sample]) == info.bo_id
                    and bool((m[k - 1 - info.tokens_per_sample:k - 1] == mid).all())
                    and bool(((body >= info.codebook_offset) & (body < info.codebook_offset + info.codebook_size)).all()))
            t_ok &= good
            if not good:
                print(f"      {T} row {rows[i]}: not [nc BO, body, nc EO] last with nc PAD")
                break
    ok &= _say("TargetLast, 6 targets x 16 rows: nc ids, the target's nc chunk last", t_ok)
    # A full nc row is 5 chunks x 198 = 990 = max_length, so it holds no padding: the grid's one-source
    # and target-alone layouts do (sources=[DEM] / []).
    g_ok = True
    for T in [t for t in IMAGE if t != "DEM"]:
        rows = TF.stratified_target_rows(ds, eval_rows, corpus, T, 4)
        for src in (["DEM"], []):
            for i in range(len(rows)):
                it = TF.TargetLast(ds, rows, T, sources=src)[i]
                k = int(it["attention_mask"].sum())
                g_ok &= (k < ds.max_length and bool((it["input_ids"][k:] == NC.PAD_ID).all())
                         and int(it["input_ids"].max()) < NC.TOTAL_VOCAB_SIZE)
    ok &= _say("TargetLast one-source and target-alone rows (5 targets x 4 x 2): padded with 81,187", g_ok)
    ok &= _say("a Coords target on the nc dataset is refused",
               _raises(lambda: TF.TargetLast(ds, rows[:1], "Coords")))
    r0 = TF.stratified_target_rows(ds, eval_rows, corpus, "S2L2A", 1)[0]
    order = ["DEM", "S2L2A"]                       # the grid's DEM -> S2L2A layout: 396 of 990 tokens
    old = assemble_sequence(chunks=[ds._load_chunk(m, r0) for m in order],
                            chunk_modality_ids=[NC.MODALITY_TO_ID[m] for m in order],
                            chunk_shufflable=[True] * len(order), max_length=ds.max_length, pad_id=V1.PAD_ID,
                            rng=np.random.default_rng(0), shuffle_image_patches=False)
    ok &= _say("control: the same chunks padded with v1's PAD hold ids past an nc table",
               int(old["input_ids"].max()) >= NC.TOTAL_VOCAB_SIZE)

    # ---- drop
    s1 = [r for r in TF.stratified_target_rows(ds, eval_rows, corpus, "S1RTC", 8)]
    pos = {int(r): i for i, r in enumerate(ds.rows)}
    dm = TF.DropModalities(ds, ["S1RTC"])
    info = NC.get_modality("S1RTC")
    d_ok = True
    for r in s1:
        orig, out = ds[pos[r]], dm[pos[r]]
        x = orig["input_ids"][orig["attention_mask"] == 1]
        s, e = int((x == info.bo_id).nonzero()[0]), int((x == info.eo_id).nonzero()[0])
        want = torch.cat([x[:s], x[e + 1:]])
        y = out["input_ids"]
        k = int(out["attention_mask"].sum())
        d_ok &= (e - s == info.tokens_per_sample + 1 and torch.equal(y[:k], want)
                 and bool((y[k:] == NC.PAD_ID).all()))
    ok &= _say(f"DropModalities([S1RTC]) on {len(s1)} nc rows: its nc chunk gone, the rest in order, nc PAD", d_ok)
    ok &= _say("control: drop_chunks with v1's layout on the same nc rows raises",
               all(_raises(lambda r=r: TF.drop_chunks(ds[pos[r]], ["S1RTC"], V1)) for r in s1))
    ok &= _say("control: dropping Coords from the nc dataset is refused",
               _raises(lambda: TF.DropModalities(ds, ["Coords"])))

    # ---- layout
    passes = [(ds, NC.TOTAL_VOCAB_SIZE), (ds1, V1.TOTAL_VOCAB_SIZE), (ds1, V1.TOTAL_VOCAB_SIZE + 144)]
    refused = [(ds, V1.TOTAL_VOCAB_SIZE), (ds1, NC.TOTAL_VOCAB_SIZE), (ds1, V1.TOTAL_VOCAB_SIZE - 1)]
    ok &= _say("check_layout passes nc/81,188, v1/87,556 and v1/87,700 (temporal width)",
               not any(_raises(lambda d=d, m=m: TF.check_layout(_model(m), d)) for d, m in passes))
    ok &= _say("control: check_layout refuses nc/87,556, v1/81,188 and v1/87,555",
               all(_raises(lambda d=d, m=m: TF.check_layout(_model(m), d)) for d, m in refused))

    # ---- run_pass
    from eo.generate import conditional as G
    model, _ = G.build_model(ARM_NC, None, device="cpu")
    TF.check_layout(model, ds)
    s1g = TF.stratified_target_rows(ds, eval_rows, corpus, "S1GRD", 4)   # ssl4eos12: majortom has no S1GRD
    idx = list(range(4)) + [pos[r] for r in s1g]
    sub = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=4)
    res = TF.run_pass(model, sub, device="cpu", keep_tokens=False, dtype=None)
    pm = res["per_modality"]
    lnv = math.log(NC.TOTAL_VOCAB_SIZE)
    ce_ok = all(abs(pm[m]["ce"] - lnv) < 1.0 for m in pm)
    sl_ok = all(0.5 < pm[m]["slot_mass"] / (NC.get_modality(m).codebook_size / NC.TOTAL_VOCAB_SIZE) < 2.0 for m in pm)
    bounds = TF._slot_bounds(TF.layout_for_model(model))
    ok &= _say(f"untrained nc model, 8 rows of both corpora: modalities {sorted(pm)} = the six image ones",
               sorted(pm) == sorted(IMAGE))
    ok &= _say(f"  CE within 1 nat of ln V = {lnv:.2f} ({min(v['ce'] for v in pm.values()):.2f}"
               f"-{max(v['ce'] for v in pm.values()):.2f}); slot masses within 2x of chance", ce_ok and sl_ok)
    ok &= _say("  the slot bounds used fit inside the 81,188 logits",
               max(hi for _, hi in bounds.values()) <= NC.TOTAL_VOCAB_SIZE)
    ok &= _say("control: v1's slot bounds reach past 81,188 rows",
               max(hi for _, hi in TF._slot_bounds(V1).values()) > NC.TOTAL_VOCAB_SIZE)
    return bool(ok)


if __name__ == "__main__":
    good = check()
    print("W36", "PASS" if good else "FAIL")
    raise SystemExit(0 if good else 1)
