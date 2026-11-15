#!/usr/bin/env python
"""W30 -- eval_checkpoint.py --drop-modalities removes whole chunks, and nothing else.

The meeting of 2026-11-11 asked whether to keep Coords. Its free check scores A10c, B10c and T360 on
their held-out sequences with the Coords chunk taken out at inference; the models retrained without
Coords are later scored the same way, through their Coords-trained twin's config, so that every row
keeps its image modalities' order and the two pair per row. This gate holds the machinery to that, on
A10c's held-out set (val_clean through `multimodal.token_set: clean`), CPU only. Every part has a
control that must fail it (CLAUDE.md rule 2):

  identity  no drop: eval_loader(drop=()) does not wrap at all; TF.DropModalities(ds, ()), through
            the held-out loader's own settings (batch 4, 4 workers), serves all 4,416 sequences bit
            for bit; TargetLast(drop=()) equals TargetLast() on every target-last row (7 targets,
            corpus-stratified, 512 each, S1GRD's 384 from ssl4eos12). Controls: the same wrapper at
            0 workers serves other orders (the comparison sees the per-worker draws); TargetLast
            with Coords dropped differs on every row.
  dropped   Coords dropped, all 4,416 sequences through eval_loader(drop=["Coords"]): the original
            with exactly the Coords chunk removed (found by id: its slot, BO and EO; contiguous,
            [BO, body, EO]), every other token in its order; no Coords id or modality id left;
            shorter by the chunk; attention a prefix, PAD after, labels -100 exactly where the
            attention mask is 0, modality ids re-derived from each id's slot, positions arange, the
            five keys. TargetLast(drop={Coords}): the ordering rule written out here, no Coords, the
            target's chunk last, and the undropped TargetLast with the chunk cut out. A hand-built
            row with an 8-token Coords body (what the tokenizer once emitted at the equator) loses
            its whole chunk too.
  controls  (a) removing only Coords' body (BO and EO left) fails; (b) the undropped sequences fail
            ("Coords present"); (c) dropping DEM instead fails the order comparison; malformed chunks,
            a patch-shuffled base, a dropped target or source, unknown names and a directory holding
            another drop set are refused.

    .venv/bin/python eo/scripts/gate_drop_modalities.py        # ~2 min on CPU

Runs in `.venv`. Nothing here imports terratorch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

ARM = "eo_terramesh/arm_a10c_mor"
ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]


def _compose(name, overrides=None):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        return compose(config_name=name, overrides=list(overrides or []))


def _slot_lut() -> torch.Tensor:
    """Token id -> modality id by codebook slot (0 for BO, EO and PAD): derived here, not by drop_chunks."""
    from eo.data.eo_vocab import MODALITIES, MODALITY_TO_ID, TOTAL_VOCAB_SIZE
    lut = torch.zeros(TOTAL_VOCAB_SIZE, dtype=torch.long)
    for name, info in MODALITIES.items():
        lut[info.codebook_offset:info.codebook_offset + info.codebook_size] = MODALITY_TO_ID[name]
    return lut


def _of(mod: str, ids: torch.Tensor) -> torch.Tensor:
    """Positions whose id belongs to `mod`: its codebook slot, its BO or its EO."""
    from eo.data.eo_vocab import get_modality
    i = get_modality(mod)
    return (((ids >= i.codebook_offset) & (ids < i.codebook_offset + i.codebook_size))
            | (ids == i.bo_id) | (ids == i.eo_id))


def judge(orig: dict, out: dict, lut: torch.Tensor, mod: str = "Coords") -> dict:
    """Named parts of "`out` is `orig` with exactly `mod`'s chunk removed", each True or False."""
    from eo.data.eo_vocab import MODALITY_TO_ID, PAD_ID, get_modality
    from eo.eval.teacher_forced import SEQ_KEYS
    info, mid = get_modality(mod), MODALITY_TO_ID[mod]
    L = orig["input_ids"].shape[0]
    n = int(orig["attention_mask"].sum())
    x = orig["input_ids"][:n]
    blk = _of(mod, x)
    w = torch.nonzero(blk).flatten()
    y, att, k = out["input_ids"], out["attention_mask"], int(out["attention_mask"].sum())
    return {
        "five keys, long, (L,)": tuple(out) == SEQ_KEYS and all(v.dtype == torch.long and tuple(v.shape) == (L,)
                                                                for v in out.values()),
        "chunk [BO, body, EO]": (w.numel() >= 3 and int(w[-1]) - int(w[0]) + 1 == w.numel()
                                 and int(x[w[0]]) == info.bo_id and int(x[w[-1]]) == info.eo_id),
        "no Coords id": not bool(_of(mod, y).any()) and not bool((out["modality_ids"] == mid).any()),
        "order": torch.equal(y[:k], x[~blk]),
        "length": k == n - int(blk.sum()),
        "attention prefix": bool((att[:k] == 1).all()) and bool((att[k:] == 0).all()),
        "PAD after": bool((y[k:] == PAD_ID).all()),
        "labels": torch.equal(out["labels"] == -100, att == 0) and torch.equal(out["labels"][:k], y[:k]),
        "modality ids": (torch.equal(out["modality_ids"], lut[y])
                         and torch.equal(out["modality_ids"][:k], orig["modality_ids"][:n][~blk])),
        "positions arange": torch.equal(out["position_ids"], torch.arange(L)),
    }


def _same(a: dict, b: dict) -> bool:
    return tuple(a) == tuple(b) and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)


def _rows_of(batch: dict) -> list:
    return [{k: v[i] for k, v in batch.items()} for i in range(batch["input_ids"].shape[0])]


def _body_only(item: dict) -> dict:
    """CONTROL (a): Coords' body removed, its BO and EO left, padded again."""
    from eo.data.eo_vocab import MODALITY_TO_ID, PAD_ID
    n = int(item["attention_mask"].sum())
    keep = item["modality_ids"][:n] != MODALITY_TO_ID["Coords"]
    L, k = item["input_ids"].shape[0], int(keep.sum())
    out = {"input_ids": torch.full((L,), PAD_ID), "attention_mask": torch.zeros(L, dtype=torch.long),
           "labels": torch.full((L,), -100), "position_ids": torch.arange(L),
           "modality_ids": torch.zeros(L, dtype=torch.long)}
    out["input_ids"][:k] = item["input_ids"][:n][keep]
    out["attention_mask"][:k] = 1
    out["labels"][:k] = item["labels"][:n][keep]
    out["modality_ids"][:k] = item["modality_ids"][:n][keep]
    return out


def _raises(fn) -> bool:
    try:
        fn()
    except (ValueError, SystemExit):
        return True
    return False


def check() -> bool:
    from eo.data.eo_vocab import IMAGE_MODALITIES, MODALITIES as MODS, MODALITY_TO_ID, PAD_ID, get_modality
    from eo.data.eval_split import load_eval_rows, load_row_table
    from eo.eval import teacher_forced as TF
    from eo.scripts import eval_checkpoint as EC
    from lm_dataset.load_dataset import load_eval_dataset_from_config
    from lm_dataset.sequence_assembly import assemble_sequence

    cfg = _compose(ARM)
    nw = int(cfg.dataloader_num_workers)
    ds, base_loader = TF.eval_loader(cfg, num_workers=nw)
    lut = _slot_lut()
    ok = True

    # The data: what a Coords chunk holds in the stored rows.
    co = np.load(Path(ds.root_dir) / "Coords_tok" / "tokens.npy", mmap_mode="r")
    table = load_row_table(ROOT)
    corpus = table.corpus.values
    eq = np.round(table.center_lat.values * 4) / 4 == 0
    d_ok = (Path(ds.root_dir).name == "val_clean" and len(ds) == 4416 and not ds.shuffle_image_patches
            and nw == 4 and co.shape == (ds.n_rows, 3) and bool((np.asarray(co[:, 2]) == 3).all()))
    print(f"    {ARM.split('/')[1]}: {len(ds)} held-out rows of {Path(ds.root_dir).name}/, no patch shuffle, "
          f"{nw} workers; Coords stored {co.shape}, every body [lat, lon, EOS=3] (3 tokens); "
          f"{int(eq.sum())} rows at the equator, {int(np.isin(np.flatnonzero(eq), ds.rows).sum())} held out = {d_ok}")
    ok &= d_ok

    # ---- identity and the dropped main pass: every held-out sequence, through the loader's settings.
    ds_u, unwrapped = TF.eval_loader(cfg, num_workers=nw, drop=())
    u_ok = unwrapped.dataset is ds_u and base_loader.dataset is ds
    print(f"    no drop: eval_loader serves the dataset itself, unwrapped        = {u_ok}")
    ok &= u_ok
    mk = lambda d, w: torch.utils.data.DataLoader(d, batch_size=4, shuffle=False, num_workers=w)  # noqa: E731
    ds_d, drop_loader = TF.eval_loader(cfg, num_workers=nw, drop=["Coords"])
    wrap_ok = (isinstance(drop_loader.dataset, TF.DropModalities) and drop_loader.dataset.base is ds_d
               and drop_loader.dataset.drop == ("Coords",))
    n = n_id = n_drop = n_a = n_b = n_c = n_w0 = 0
    fail_parts: dict = {}
    first_pos = {"first": 0, "last": 0, "middle": 0}
    for b_base, b_id, b_drop, b_w0 in zip(base_loader, mk(TF.DropModalities(ds, ()), nw), drop_loader,
                                          mk(TF.DropModalities(ds, ()), 0)):
        for orig, same, dropped, w0 in zip(_rows_of(b_base), _rows_of(b_id), _rows_of(b_drop), _rows_of(b_w0)):
            n += 1
            n_id += _same(orig, same)
            n_w0 += not _same(orig, w0)
            parts = judge(orig, dropped, lut)
            n_drop += all(parts.values())
            for name, v in parts.items():
                fail_parts[name] = fail_parts.get(name, 0) + (not v)
            m = int(orig["attention_mask"].sum())
            s = int(torch.nonzero(orig["input_ids"][:m] == get_modality("Coords").bo_id).flatten()[0])
            first_pos["first" if s == 0 else "last" if s == m - 5 else "middle"] += 1
            # CONTROLS on the same rows: each must FAIL.
            n_a += not all(judge(orig, _body_only(orig), lut).values())
            n_b += not judge(orig, orig, lut)["no Coords id"]
            n_c += not judge(orig, TF.drop_chunks(orig, ["DEM"]), lut)["order"]
    id_ok, dr_ok = n_id == n == len(ds), n_drop == n == len(ds) and wrap_ok
    print(f"    no drop: DropModalities(ds, ()) == the held-out loader on {n_id}/{n} sequences "
          f"(batch 4, {nw} workers) = {id_ok}")
    print(f"    Coords dropped: original minus exactly its chunk, all parts, on {n_drop}/{n} sequences = {dr_ok}")
    print(f"      parts checked: {', '.join(fail_parts)}; failures {sum(fail_parts.values())}")
    print(f"      Coords chunk first in {first_pos['first']}, last in {first_pos['last']}, between in "
          f"{first_pos['middle']} of the {n} sequences")
    ok &= id_ok and dr_ok
    c_w0 = 0 < n_w0 < n
    print(f"    control: the same wrapper at 0 workers draws other orders on {n_w0}/{n} rows = {c_w0}")
    c_a, c_b, c_c = n_a == n, n_b == n, n_c == n
    print(f"    control (a): body-only removal (BO/EO left) fails on {n_a}/{n} sequences        = {c_a}")
    print(f"    control (b): the undropped sequences fail 'no Coords id' on {n_b}/{n}            = {c_b}")
    print(f"    control (c): dropping DEM instead fails the order comparison on {n_c}/{n}        = {c_c}")
    ok &= c_w0 and c_a and c_b and c_c

    # ---- target-last: TargetLast(drop=) on every target-last row.
    eval_rows = load_eval_rows(root_dir=ROOT)

    def reference(row, target, drop=()):
        present = ds.present_modalities(row)
        order = [m for m in MODS if m in present and m != target and m not in drop] + [target]
        return assemble_sequence(
            chunks=[ds._load_chunk(m, row) for m in order],
            chunk_modality_ids=[MODALITY_TO_ID[m] for m in order],
            chunk_shufflable=[m in IMAGE_MODALITIES for m in order],
            max_length=ds.max_length, pad_id=PAD_ID,
            rng=np.random.default_rng(0), shuffle_image_patches=False)

    t_n = t_id = t_n2 = t_rule = t_cut = t_last = t_none = t_ctl = 0
    by_corpus = {"majortom": 0, "ssl4eos12": 0}
    for T in TARGETS:
        R = TF.stratified_target_rows(ds, eval_rows, corpus, T, 512)
        tl, tl0 = TF.TargetLast(ds, R, T), TF.TargetLast(ds, R, T, drop=())
        tld = TF.TargetLast(ds, R, T, drop=("Coords",)) if T != "Coords" else None
        ti = get_modality(T)
        for i, r in enumerate(R):
            a = tl[i]
            t_n += 1
            t_id += _same(a, tl0[i])
            if tld is None:
                continue
            by_corpus[corpus[r]] += 1
            d = tld[i]
            t_n2 += 1
            t_rule += _same(d, reference(r, T, drop=("Coords",)))
            t_cut += _same(d, TF.drop_chunks(a, ["Coords"]))
            k = int(d["attention_mask"].sum())
            t_last += (int(d["input_ids"][k - 1]) == ti.eo_id
                       and int(d["input_ids"][k - 2 - ti.tokens_per_sample]) == ti.bo_id
                       and bool((d["modality_ids"][k - 1 - ti.tokens_per_sample:k - 1] == MODALITY_TO_ID[T]).all()))
            t_none += not bool(_of("Coords", d["input_ids"]).any()) and not bool(
                (d["modality_ids"] == MODALITY_TO_ID["Coords"]).any())
            t_ctl += not _same(a, d)
    tl_ok = t_id == t_n and t_rule == t_cut == t_last == t_none == t_n2 and t_n2 > 0
    print(f"    TargetLast(drop=()) == TargetLast() on {t_id}/{t_n} target-last rows (7 targets) = {t_id == t_n}")
    print(f"    TargetLast(drop={{Coords}}) on {t_n2} rows of 6 targets ({by_corpus['majortom']} majortom, "
          f"{by_corpus['ssl4eos12']} ssl4eos12): the rule {t_rule}, no Coords {t_none}, target last {t_last}, "
          f"== the undropped one with the chunk cut out {t_cut} = {tl_ok}")
    ok &= tl_ok
    c_t = t_ctl == t_n2
    print(f"    control: TargetLast with Coords dropped differs from TargetLast() on {t_ctl}/{t_n2} rows = {c_t}")
    ok &= c_t

    # ---- a Coords body of another length (the tokenizer's 8 at the equator, fixed in the stored rows).
    r0 = int(ds.rows[0])
    cinfo = get_modality("Coords")
    body8 = torch.tensor([4344, 5000, 3, 3, 3, 3, 3, 3]) + cinfo.codebook_offset
    chunks = [ds._load_chunk("S2L2A", r0), torch.cat([torch.tensor([cinfo.bo_id]), body8, torch.tensor([cinfo.eo_id])]),
              ds._load_chunk("DEM", r0)]
    syn = assemble_sequence(chunks=chunks, chunk_modality_ids=[MODALITY_TO_ID[m] for m in ("S2L2A", "Coords", "DEM")],
                            chunk_shufflable=[True, False, True], max_length=ds.max_length, pad_id=PAD_ID,
                            rng=np.random.default_rng(0), shuffle_image_patches=False)
    p8 = judge(syn, TF.drop_chunks(syn, ["Coords"]), lut)
    s_ok = all(p8.values()) and int(TF.drop_chunks(syn, ["Coords"])["attention_mask"].sum()) == 2 * 198
    print(f"    a hand-built row with an 8-token Coords body loses all 10 of its chunk's tokens = {s_ok}")
    ok &= s_ok

    # ---- refusals: each must raise.
    item = ds[0]
    m0 = int(item["attention_mask"].sum())
    s0 = int(torch.nonzero(item["input_ids"][:m0] == cinfo.bo_id).flatten()[0])
    no_eo = {k: v.clone() for k, v in item.items()}
    no_eo["input_ids"][s0 + 4] = no_eo["labels"][s0 + 4] = get_modality("DEM").eo_id
    stray = {k: v.clone() for k, v in item.items()}
    j = int(torch.nonzero(stray["modality_ids"] == MODALITY_TO_ID["DEM"]).flatten()[5])
    stray["input_ids"][j] = stray["labels"][j] = cinfo.codebook_offset + 5
    shuffled_pos = {k: v.clone() for k, v in item.items()}
    shuffled_pos["position_ids"][[1, 2]] = shuffled_pos["position_ids"][[2, 1]]
    six = dict(item, extra=item["input_ids"])
    shuf_ds = load_eval_dataset_from_config(_compose(ARM, ["multimodal.shuffle_image_patches=true"]))
    R_dem = TF.stratified_target_rows(ds, eval_rows, corpus, "DEM", 4)
    old = {"precision": "fp32", "modes": {}}
    refusals = {
        "a Coords chunk without its EO": _raises(lambda: TF.drop_chunks(no_eo, ["Coords"])),
        "a Coords id outside its chunk": _raises(lambda: TF.drop_chunks(stray, ["Coords"])),
        "shuffled positions": _raises(lambda: TF.drop_chunks(shuffled_pos, ["Coords"])),
        "a sixth key": _raises(lambda: TF.drop_chunks(six, ["Coords"])),
        "a patch-shuffled base": _raises(lambda: TF.DropModalities(shuf_ds, ["Coords"])),
        "an unknown modality": _raises(lambda: TF.DropModalities(ds, ["Coordz"])),
        "a dropped target": _raises(lambda: TF.TargetLast(ds, R_dem, "Coords", drop=["Coords"])),
        "a dropped named source": _raises(lambda: TF.TargetLast(ds, R_dem, "DEM", sources=["Coords"], drop=["Coords"])),
        "--drop-modalities Coordz": _raises(lambda: EC.parse_drop("Coords,Coordz")),
        "a directory holding no drop, for a drop run": EC.drop_refusal(old, ["Coords"]) is not None,
        "a directory holding Coords, for a run without": EC.drop_refusal({"drop_modalities": ["Coords"]}, []) is not None,
    }
    passes = {
        "parse_drop gives registry order": EC.parse_drop(" Coords , DEM ,Coords") == ["DEM", "Coords"],
        "a fresh directory takes any drop": EC.drop_refusal({}, ["Coords"]) is None,
        "an old directory (no key) takes no drop": EC.drop_refusal(old, []) is None,
        "the same drop set resumes": EC.drop_refusal({"drop_modalities": ["Coords"]}, ["Coords"]) is None,
        "TargetLast(sources=[], drop) is the target alone": _same(
            TF.TargetLast(ds, R_dem, "DEM", sources=[], drop=["Coords"])[0], TF.TargetLast(ds, R_dem, "DEM", sources=[])[0]),
    }
    r_ok, p_ok = all(refusals.values()), all(passes.values())
    print(f"    refused: {', '.join(k for k, v in refusals.items() if v)} = {r_ok}")
    if not r_ok:
        print(f"      NOT refused: {[k for k, v in refusals.items() if not v]}")
    print(f"    accepted: {', '.join(k for k, v in passes.items() if v)} = {p_ok}")
    if not p_ok:
        print(f"      NOT accepted: {[k for k, v in passes.items() if not v]}")
    ok &= r_ok and p_ok
    return bool(ok)


if __name__ == "__main__":
    print("W30  --drop-modalities removes whole chunks  [meeting 2026-11-11]")
    try:
        res = check()
    except Exception as e:                        # noqa: BLE001
        print(f"    RAISED {type(e).__name__}: {e}")
        res = False
    print(f"  {'PASS' if res else 'FAIL'}  W30")
    raise SystemExit(0 if res else 1)
