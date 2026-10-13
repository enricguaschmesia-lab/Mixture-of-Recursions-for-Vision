#!/usr/bin/env python
"""Phase 6 gate (PHASE6_PLAN Step 8, D6.10). `.venv`; every check is paired with a control
that must FIRE, so a pass means something.

    python eo/scripts/verify_phase6.py              # everything whose artifacts exist
    python eo/scripts/verify_phase6.py --only X2 X3

  X0  verify_phase3.py still green (the regression net; slow: run with --with-x0)
  X1  BEN round trip within range of TerraMesh; broken band order / standardization worse
  X2  ben_vocab: no collision, every pretraining id unchanged | overlapping layouts rejected
  X3  extension identity + tie on the saved checkpoint      | untied / shifted-PAD caught
  X4  sequences: blocks decode back, BEN never an input, S1/S2 in their slots, five keys,
      input-set frequencies, epoch re-draws               | BEN-as-input sequence rejected
  X5  splits disjoint at the patch level                     | a duplicated patch caught
  X6  LoRA: identity at init, frozen tensors bit-identical after training
                                                             | masked-grad + AdamW moves rows
  X6h head only (L5): identity at init, only the 21 new rows trainable, frozen tensors bit-identical
      after training                                         | the LoRA structure exposes more
  X7  test controls: wrong input collapses toward the prior  | own inputs via the same path do not
  X8  the S6.b overlay differs from b10_full in exactly ben.loss_on | b10_lora is flagged
      L6's overlay b10_lora_lr1e4 differs from b10_lora in exactly learning_rate | b10_full is flagged
  X9  Step 5b chained generation (artifacts of ben_chain.py, plus a CPU re-check):
      a  the chain's prompt with the real blocks = the dataset's S1+S2 prompt | another patch's S2 differs
      b  a generated body becomes BO body EO in its own slot             | S2 codes in the S1 slot differ
      c  KV cache: cached = uncached logits on real continuations (each generation's json)
                                                                         | shift 1 disagrees
      d  every generated token lies inside its target's codebook slot    | the same tokens one slot up are caught
      e  plumbing on the A100: the real-block chain prompt reproduces eval_ben's stored S1+S2 outputs
                                                                         | another patch's S2 does not
      + average precision equals sklearn's (run in the `mor` env) | a broken AP is caught

Artifacts it reads are named in each check. A check whose artifact does not exist yet is
SKIPPED and reported as such, never passed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eo.ben import labels as L  # noqa: E402
from eo.data import ben_vocab as BV  # noqa: E402
from eo.data import eo_vocab as EV  # noqa: E402

BEN_ROOT = Path(__import__("os").environ.get("BEN_ROOT", "/data/enric/data/BigEarthNet"))
EXT = Path("/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300")
REPORTS = Path("/data/enric/reports/ben")
MOR_PY = "/data/enric/miniforge3/envs/mor/bin/python"

RESULTS = []


def record(check, ok, detail, ctrl_ok=None, ctrl_detail=""):
    RESULTS.append((check, ok, detail, ctrl_ok, ctrl_detail))
    c = "" if ctrl_ok is None else f" | control {'FIRED' if ctrl_ok else 'MISSED'}: {ctrl_detail}"
    print(f"  {'PASS' if ok else ('SKIP' if ok is None else 'FAIL')}  {check}: {detail}{c}", flush=True)


# ---------------------------------------------------------------- X0
def x0():
    r = subprocess.run([sys.executable, str(REPO / "eo/scripts/verify_phase3.py"), "--skip-forward"],
                       capture_output=True, text=True)
    verdict = [ln.strip() for ln in r.stdout.splitlines() if "PHASE 3 GATE" in ln] or ["(no verdict line)"]
    record("X0 verify_phase3 --skip-forward", r.returncode == 0 and "PASSED" in verdict[-1], verdict[-1][:120])


# ---------------------------------------------------------------- X1
def x1():
    f = REPORTS / "step1_roundtrip.json"
    if not f.exists():
        return record("X1 round trip", None, f"{f} missing")
    d = json.loads(f.read_text())
    med = lambda m, k: d[m][k]["rmse_z_ml7"]["median"]
    s2, s2_tm = med("S2", "bilinear"), med("S2", "terramesh_reference")
    s1, s1_tm = med("S1", "S1GRD"), med("S1", "terramesh_reference_S1GRD")
    ok = s2 <= 1.5 * s2_tm and s1 <= 1.5 * s1_tm
    ctrls = {f"S2 {k}": med("S2", k) / s2 for k in ("ctrl_reversed_bands", "ctrl_no_standardization")}
    ctrls.update({f"S1 {k}": med("S1", k) / s1 for k in ("ctrl_swapped_vv_vh", "ctrl_no_standardization")})
    c_ok = all(v >= 2.0 for v in ctrls.values())
    record("X1 round trip", ok, f"rmse_z_ml7 S2 {s2:.3f} (TerraMesh {s2_tm:.3f}), S1GRD {s1:.3f} "
           f"(TerraMesh {s1_tm:.3f}); bilinear {s2:.3f} vs bicubic {med('S2', 'bicubic'):.3f}; "
           f"S1RTC {med('S1', 'S1RTC'):.3f}",
           c_ok, "worsening x" + ", ".join(f"{v:.1f}" for v in ctrls.values()) + " (each must be >= 2)")


# ---------------------------------------------------------------- X2
def x2():
    probs = BV.layout_problems(BV.CLASS_IDS, BV.BO_BEN, BV.EO_BEN, BV.TOTAL_VOCAB_SIZE)
    same = all((BV.MODALITIES[n].codebook_offset, BV.MODALITIES[n].bo_id, BV.MODALITIES[n].eo_id) ==
               (i.codebook_offset, i.bo_id, i.eo_id) for n, i in EV.MODALITIES.items())
    ok = not probs and same and BV.PAD_ID == EV.PAD_ID and BV.BASE == EV.TOTAL_VOCAB_SIZE \
        and len(L.NEW_LABELS) == BV.N_CLASSES
    broken = {
        "overlap (start at PAD)": ([EV.PAD_ID + i for i in range(19)], EV.PAD_ID + 19, EV.PAD_ID + 20, EV.PAD_ID + 21),
        "token:8 (class ids = 8..26)": ([8 + i for i in range(19)], BV.BO_BEN, BV.EO_BEN, BV.TOTAL_VOCAB_SIZE),
        "gap after PAD": ([BV.BASE + 1 + i for i in range(19)], BV.BASE + 20, BV.BASE + 21, BV.BASE + 22),
        "duplicate id": ([BV.BASE] * 19, BV.BO_BEN, BV.EO_BEN, BV.TOTAL_VOCAB_SIZE),
    }
    fired = {k: bool(BV.layout_problems(*v)) for k, v in broken.items()}
    record("X2 ben_vocab", ok, f"{BV.N_NEW} ids {BV.BASE}..{BV.TOTAL_VOCAB_SIZE - 1}, PAD {BV.PAD_ID} unchanged, "
           f"all {len(EV.MODALITIES)} EO slots identical", all(fired.values()),
           ", ".join(f"{k}: {'rejected' if v else 'ACCEPTED'}" for k, v in fired.items()))


# ---------------------------------------------------------------- X3
def x3():
    ej = EXT / "extension.json"
    if not ej.exists():
        return record("X3 extension", None, f"{ej} missing")
    from eo.finetune import extend as X
    from eo.finetune.init import verify_extension, strict_load
    from eo.generate.conditional import build_model
    doc = verify_extension(EXT, BV.TOTAL_VOCAB_SIZE)            # hash, identity, controls
    model, _ = build_model("eo_ben/b10_full", None, device="cpu",
                           overrides=[f"finetune.init_from={EXT}"])
    strict_load(model, EXT)
    try:
        X.assert_tied(model)
        tied = True
    except AssertionError:
        tied = False
    ok = tied and doc["identity"]["passed"] and doc["identity"]["max_abs_diff"] <= doc["identity"]["tol"]
    # Control: untie (a copy of the weight, equal values) -> the check must fire.
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    try:
        X.assert_tied(model)
        untied_caught = False
    except AssertionError:
        untied_caught = True
    c_ok = untied_caught and doc["control_insert_before_pad"]["fired"] and doc["control_untied_resize"]["fired"]
    record("X3 extension", ok, f"saved checkpoint tied, identity max |dlogit| {doc['identity']['max_abs_diff']:.1e} "
           f"on {doc['identity']['positions']} positions", c_ok,
           f"equal-valued untied copy caught: {untied_caught}; shifted PAD |dlogit| "
           f"{doc['control_insert_before_pad']['max_abs_diff']:.2f}")


# ---------------------------------------------------------------- X4
def check_sequence(item, classes, n_in_expected=None):
    """Raise on any violation of the BEN sequence contract. Returns the input sources seen."""
    ids, att, lab = item["input_ids"], item["attention_mask"], item["labels"]
    if set(item) != {"input_ids", "attention_mask", "labels", "position_ids", "modality_ids"}:
        raise AssertionError(f"batch keys {sorted(item)}")
    seq = ids[att.bool()].tolist()
    if BV.BO_BEN not in seq:
        raise AssertionError("no BO_BEN")
    b = seq.index(BV.BO_BEN)
    block, inputs = seq[b:], seq[:b]
    if any(t >= BV.BASE for t in inputs):
        raise AssertionError("a BEN id appears outside the label block (BEN as an input)")
    if BV.decode_label_block(block, strict=True) != sorted(classes):
        raise AssertionError(f"label block {block} does not decode to {classes}")
    if block[-1] != BV.EO_BEN:
        raise AssertionError("label block is not last")
    srcs, i = [], 0
    while i < len(inputs):
        hit = [s for s, m in BV.INPUT_MODALITIES.items() if inputs[i] == BV.get_modality(m).bo_id]
        if not hit:
            raise AssertionError(f"position {i}: {inputs[i]} is not an input BO")
        info = BV.get_modality(BV.INPUT_MODALITIES[hit[0]])
        body = inputs[i + 1:i + 1 + info.tokens_per_sample]
        if inputs[i + 1 + info.tokens_per_sample] != info.eo_id:
            raise AssertionError("input chunk not closed by its EO")
        if not all(info.codebook_offset <= t < info.codebook_offset + info.codebook_size for t in body):
            raise AssertionError(f"{hit[0]} ids outside the {info.name} slot")
        srcs.append(hit[0])
        i += info.tokens_per_sample + 2
    keep = lab != -100
    if keep.sum() != len(block) - 1 or not torch.equal(ids[keep], torch.tensor(block[1:])):
        raise AssertionError("labels are not exactly c_1..c_k, EO_BEN")
    return srcs


def x4(full: bool):
    try:
        from eo.data.ben_dataset import BenTokenDataset, load_index
        idx = load_index()
        ds = BenTokenDataset(role="ft_train", condition="random", index=idx)
        dv = BenTokenDataset(role="ft_valsel", condition="random", index=idx)
    except (RuntimeError, FileNotFoundError) as e:
        return record("X4 sequences", None, f"dataset not ready: {str(e)[:100]}")
    n_bad, conds = 0, {"S1": 0, "S2": 0, "S1S2": 0}
    order_s2_first = 0
    sets = [(ds, range(len(ds)) if full else range(0, len(ds), 7)), (dv, range(len(dv)))]
    n_checked = 0
    for d, it in sets:
        for p in it:
            try:
                s = check_sequence(d[p], d.classes(int(d.rows[p])))
            except AssertionError as e:
                n_bad += 1
                if n_bad < 4:
                    print(f"    row {int(d.rows[p])}: {e}")
                continue
            n_checked += 1
            key = "S1S2" if len(s) == 2 else s[0]
            conds[key] += 1
            order_s2_first += (len(s) == 2 and s[0] == "S2")
    tot = sum(conds.values())
    freq = {k: v / tot for k, v in conds.items()}
    freq_ok = all(abs(v - 1 / 3) < 0.02 for v in freq.values()) and abs(order_s2_first / conds["S1S2"] - 0.5) < 0.03
    # epoch re-draws: about 2/3 of rows change condition between epochs
    e0 = [tuple(ds.inputs_for(p)) for p in range(0, len(ds), 13)]
    ds.set_epoch(1)
    e1 = [tuple(ds.inputs_for(p)) for p in range(0, len(ds), 13)]
    ds.set_epoch(0)
    changed = np.mean([a != b for a, b in zip(e0, e1)])
    # epoch reaches DataLoader workers (non-persistent, forked after set_epoch, as in HF)
    ds.set_epoch(2)
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, list(range(64))), batch_size=8, num_workers=2)
    seen = []
    for b in dl:
        for row in b["input_ids"]:
            seen.append(tuple(s for s, m in BV.INPUT_MODALITIES.items()
                              if BV.get_modality(m).bo_id in row[:1].tolist() + row[198:199].tolist()))
    want = [tuple(ds.inputs_for(p)) for p in range(64)]
    ds.set_epoch(0)
    workers_ok = all(sorted(a) == sorted(b) for a, b in zip(seen, want))
    ok = n_bad == 0 and freq_ok and 0.6 < changed < 0.73 and workers_ok
    # Control: a sequence with the label block FIRST (BEN as an input) must be rejected.
    item = ds[0]
    att = item["attention_mask"].bool()
    seq = item["input_ids"][att].tolist()
    b = seq.index(BV.BO_BEN)
    bad = dict(item)
    bad["input_ids"] = item["input_ids"].clone()
    bad["input_ids"][: len(seq)] = torch.tensor(seq[b:] + seq[:b])
    try:
        check_sequence(bad, ds.classes(int(ds.rows[0])))
        c_ok = False
    except AssertionError:
        c_ok = True
    record("X4 sequences", ok, f"{n_checked} sequences ({'all' if full else '1/7 of'} train + all valsel), "
           f"{n_bad} bad; conditions " + ", ".join(f"{k} {v:.3f}" for k, v in freq.items()) +
           f"; S2-first {order_s2_first / conds['S1S2']:.3f}; epoch 0->1 changes {changed:.3f} of rows; "
           f"workers see the epoch: {workers_ok}; max_length {ds.max_length} (needed {ds.required_length})",
           c_ok, "BEN-as-input sequence rejected" if c_ok else "accepted")


# ---------------------------------------------------------------- X5
def x5():
    import pandas as pd
    idx = pd.read_parquet(BEN_ROOT / "index_v1.parquet")
    roles = ["ft_train", "ft_valsel", "test_a", "test_c"]

    def overlaps(df):
        out = {}
        for i, a in enumerate(roles):
            for b in roles[i + 1:]:
                for col in ("s2", "s1"):
                    n = len(set(df.loc[df[a], col]) & set(df.loc[df[b], col]))
                    if n:
                        out[f"{a}&{b}({col})"] = n
        return out
    ov = overlaps(idx)
    dup = idx.s2.duplicated().sum() + idx.s1.duplicated().sum()
    ok = not ov and dup == 0
    # Control: put one test_a patch into ft_train too.
    bad = idx.copy()
    i = int(np.flatnonzero(bad.test_a.values)[0])
    bad.loc[i, "ft_train"] = True
    c = overlaps(bad)
    record("X5 splits", ok, "roles pairwise disjoint by S2 and S1 name; " +
           ", ".join(f"{r} {int(idx[r].sum())}" for r in roles), bool(c), f"duplicated patch -> {c}")


# ---------------------------------------------------------------- X6
def x6(run_dir):
    from eo.finetune import lora as LR
    from eo.finetune.init import strict_load
    from eo.generate.conditional import build_model
    # (a) masked-gradient + AdamW: the trap the separate Parameter exists for.
    torch.manual_seed(0)
    emb = torch.nn.Embedding(10, 4)
    before = emb.weight.detach().clone()
    mask = torch.zeros(10, 1)
    mask[8:] = 1                                             # only rows 8, 9 "trainable"
    emb.weight.register_hook(lambda g: g * mask)
    opt = torch.optim.AdamW(emb.parameters(), lr=1e-2, weight_decay=0.1)
    for _ in range(5):
        opt.zero_grad()
        emb(torch.tensor([1, 8, 9])).sum().backward()
        opt.step()
    moved = float((emb.weight.detach()[:8] - before[:8]).abs().max())
    # (b) the real thing on B10: identity at init, and the frozen set.
    if not (EXT / "extension.json").exists():
        return record("X6 LoRA", None, "no extended checkpoint")
    base, _ = build_model("eo_ben/b10_lora", None, device="cpu", overrides=[f"finetune.init_from={EXT}"])
    strict_load(base, EXT)
    lm, _ = build_model("eo_ben/b10_lora", None, device="cpu", overrides=[f"finetune.init_from={EXT}"])
    strict_load(lm, EXT)
    rep = LR.apply_lora(lm, 16, 32, BV.BASE)
    x = torch.randint(0, BV.TOTAL_VOCAB_SIZE, (2, 48))
    with torch.no_grad():
        d0 = float((base(input_ids=x).logits - lm(input_ids=x).logits).abs().max())
    frozen_names = [n for n, p in lm.named_parameters() if not p.requires_grad]
    trainable = [n for n, p in lm.named_parameters() if p.requires_grad]
    only_adapters = all(("lora_" in n) or n.endswith("embed_tokens.new") for n in trainable)
    detail = (f"init identity max |dlogit| {d0:.1e}; trainable {rep['trainable_params']:,} of "
              f"{rep['total_params']:,} ({len(trainable)} tensors, adapters + new rows only: {only_adapters})")
    ok = d0 < 1e-4 and only_adapters
    # (c) after training: every frozen tensor bit-identical to the extended base.
    if run_dir is not None:
        cks = sorted(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not cks:
            return record("X6 LoRA", False, f"no checkpoint in {run_dir}")
        ref = {n: p.detach().clone() for n, p in lm.named_parameters() if not p.requires_grad}
        strict_load(lm, cks[-1])
        diff = [n for n in frozen_names if not torch.equal(ref[n], dict(lm.named_parameters())[n].detach())]
        moved_new = float((dict(lm.named_parameters())["model.embed_tokens.new"].detach()
                           - base.model.embed_tokens.weight.detach()[BV.BASE:]).abs().max())
        ok = ok and not diff and moved_new > 0
        detail += (f"; after {cks[-1].name}: {len(frozen_names) - len(diff)}/{len(frozen_names)} frozen "
                   f"tensors bit-identical, new rows moved {moved_new:.2e}")
    record("X6 LoRA", ok, detail, moved > 0, f"masked-grad + AdamW(wd 0.1) moved 'frozen' rows by {moved:.2e}")


def x6h(run_dir):
    """L5's head-only regime: identity at init, only the 21 new rows trainable, and after training
    every frozen tensor bit-identical. Control: X6's masked-grad + AdamW trap, on a head run's
    shape (the old rows must stay put only because they are a separate frozen Parameter)."""
    from eo.finetune import lora as LR
    from eo.finetune.init import strict_load
    from eo.generate.conditional import build_model
    if not (EXT / "extension.json").exists():
        return record("X6h head only", None, "no extended checkpoint")
    base, _ = build_model("eo_ben/b10_head", None, device="cpu", overrides=[f"finetune.init_from={EXT}"])
    strict_load(base, EXT)
    hm, _ = build_model("eo_ben/b10_head", None, device="cpu", overrides=[f"finetune.init_from={EXT}"])
    strict_load(hm, EXT)
    rep = LR.apply_head(hm, BV.BASE)
    x = torch.randint(0, BV.TOTAL_VOCAB_SIZE, (2, 48))
    with torch.no_grad():
        d0 = float((base(input_ids=x).logits - hm(input_ids=x).logits).abs().max())
    trainable = [n for n, p in hm.named_parameters() if p.requires_grad]
    frozen_names = [n for n, p in hm.named_parameters() if not p.requires_grad]
    ok = d0 < 1e-4 and trainable == ["model.embed_tokens.new"] and rep["trainable_params"] == 21 * 576
    detail = f"init identity max |dlogit| {d0:.1e}; trainable {trainable} ({rep['trainable_params']:,})"
    if run_dir is not None:
        cks = sorted(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        if not cks:
            return record("X6h head only", False, f"no checkpoint in {run_dir}")
        ref = {n: p.detach().clone() for n, p in hm.named_parameters() if not p.requires_grad}
        strict_load(hm, cks[-1])
        cur = dict(hm.named_parameters())
        diff = [n for n in frozen_names if not torch.equal(ref[n], cur[n].detach())]
        moved_new = float((cur["model.embed_tokens.new"].detach() - base.model.embed_tokens.weight.detach()[BV.BASE:]).abs().max())
        ok = ok and not diff and moved_new > 0
        detail += (f"; after {cks[-1].name}: {len(frozen_names) - len(diff)}/{len(frozen_names)} frozen "
                   f"tensors bit-identical, new rows moved {moved_new:.2e}")
    # control: the same model with the LoRA structure instead must expose more than the new rows
    lm, _ = build_model("eo_ben/b10_lora", None, device="cpu", overrides=[f"finetune.init_from={EXT}"])
    LR.apply_lora(lm, 16, 32, BV.BASE)
    n_lora = sum(p.requires_grad for p in lm.parameters())
    record("X6h head only", ok, detail, n_lora > 1, f"the LoRA structure exposes {n_lora} trainable tensors, not 1")


# ---------------------------------------------------------------- X7
def x7(tag, ckpt, role):
    from eo.eval import ben as E
    # (a) AP equals sklearn's, on data with ties, computed in the mor env.
    rng = np.random.default_rng(0)
    y = rng.random((400, 19)) < 0.2
    s = np.round(rng.random((400, 19)) + 0.3 * y, 2)          # rounding makes ties
    ours = [E.average_precision(y.ravel(), s.ravel())] + [E.average_precision(y[:, c], s[:, c]) for c in range(19)]
    code = ("import sys,json,numpy as np; from sklearn.metrics import average_precision_score as A;"
            "d=np.load(sys.argv[1]); y,s=d['y'],d['s'];"
            "print(json.dumps([A(y.ravel(),s.ravel())]+[A(y[:,c],s[:,c]) for c in range(y.shape[1])]))")
    tmp = Path("/tmp") / f"x7_ap_{abs(hash(tag)) % 10**8}.npz"
    np.savez(tmp, y=y, s=s)
    r = subprocess.run([MOR_PY, "-c", code, str(tmp)], capture_output=True, text=True)
    sk = json.loads(r.stdout) if r.returncode == 0 else None
    ap_ok = sk is not None and np.allclose(ours, sk, atol=1e-12)
    broken = [float(np.mean(E.f1_scores(y[:, [c]], (s[:, [c]] > 0.5))["micro_f1"])) for c in range(1)]
    ap_ctrl = sk is not None and not np.isclose(broken[0], sk[1])
    record("X7a average precision", ap_ok, f"ours == sklearn on 20 problems with ties (max |d| "
           f"{np.max(np.abs(np.array(ours) - np.array(sk))) if sk else float('nan'):.1e})",
           ap_ctrl, "F1-at-0.5 in place of AP disagrees with sklearn")
    # (b) wrong input collapses toward the prior; own inputs through the SAME path do not.
    d = REPORTS / tag / ckpt / role
    own, wrong = d / "S1S2.npz", d / "S1S2_wrong.npz"
    if not (own.exists() and wrong.exists()):
        return record("X7b wrong-input control", None, f"{own} / {wrong} missing")
    import pandas as pd
    idx = pd.read_parquet(BEN_ROOT / "index_v1.parquet")
    tr = np.stack([np.isin(np.arange(19), L.from_mask(int(m))) for m in idx.labels19[idx.ft_train]])
    o, w = np.load(own), np.load(wrong)
    assert (o["rows"] == w["rows"]).all()
    pri = E.prior_scores(tr, len(o["rows"]))
    m_pri = E.map_scores(o["truth"], pri)["micro_map"]
    m_own = E.map_scores(o["truth"], o["marginal"])["micro_map"]
    m_wr = E.map_scores(w["truth"], w["marginal"])["micro_map"]
    frac = E.collapse_fraction(m_wr, m_own, m_pri)
    # Control: the wrong-input path with the identity permutation must NOT collapse: the
    # "wrong" file's own rows are re-scored with the inputs actually used, i.e. the input
    # patch's truth; that is own-input scoring through the derangement bookkeeping.
    inp_truth = np.stack([np.isin(np.arange(19), L.from_mask(int(idx.labels19.iat[int(r)]))) for r in w["input_rows"]])
    m_ctrl = E.map_scores(inp_truth, w["marginal"])["micro_map"]
    frac_ctrl = E.collapse_fraction(m_ctrl, m_own, m_pri)
    record("X7b wrong-input control", frac < 0.2, f"{tag}/{ckpt}/{role} S1+S2: micro-mAP own {m_own:.4f}, "
           f"wrong input {m_wr:.4f}, prior {m_pri:.4f} -> {frac:.3f} of the gain survives",
           frac_ctrl > 0.8, f"scored against the input patch's own labels: {frac_ctrl:.3f} survives")


def x8():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    def flat(d, p=""):
        o = {}
        for k, v in d.items():
            o.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
        return o
    with initialize_config_dir(config_dir=str(REPO / "conf/pretrain_vision"), version_base=None):
        c = {n: flat(OmegaConf.to_container(compose(f"eo_ben/{n}", overrides=["finetune.init_from=/x"]), resolve=False))
             for n in ("b10_full", "b10_full_li", "b10_lora", "b10_lora_lr1e4")}
    diff = lambda a, b: sorted(k for k in set(c[a]) | set(c[b]) if c[a].get(k) != c[b].get(k))
    d_li, d_lora = diff("b10_full", "b10_full_li"), diff("b10_full", "b10_lora")
    record("X8 S6.b overlay", d_li == ["ben.loss_on"], f"b10_full_li vs b10_full: {d_li}",
           d_lora != ["ben.loss_on"], f"b10_lora vs b10_full differs in {d_lora}")
    d_lr, d_ctl = diff("b10_lora", "b10_lora_lr1e4"), diff("b10_lora", "b10_full")
    record("X8 L6 overlay", d_lr == ["learning_rate"] and c["b10_lora_lr1e4"]["learning_rate"] == c["b10_full"]["learning_rate"],
           f"b10_lora_lr1e4 vs b10_lora: {d_lr}; lr {c['b10_lora_lr1e4']['learning_rate']} = full's {c['b10_full']['learning_rate']}",
           d_ctl != ["learning_rate"], f"b10_full vs b10_lora differs in {d_ctl}")


# ---------------------------------------------------------------- X9
def x9(root: Path):
    from eo.data.ben_dataset import BenTokenDataset, load_index
    from eo.eval import ben as E
    from eo.eval import ben_chain as C
    ds = BenTokenDataset(role="test_a", condition="S1S2", index=load_index())
    rr = ds.rows[:256]
    real = lambda i, r, o: C.cls_prompt({"S1": ds.chunk("S1", int(r)), "S2": ds.chunk("S2", int(o))})
    same = sum(torch.equal(real(i, r, r), ds.prompt(i, "S1S2")) for i, r in enumerate(rr))
    other = sum(torch.equal(real(i, r, o), ds.prompt(i, "S1S2")) for i, (r, o) in enumerate(zip(rr, np.roll(rr, 1))))
    record("X9a chain prompt = dataset S1+S2 prompt", same == len(rr), f"{same}/{len(rr)} rows identical",
           other == 0, f"another patch's S2: {other}/{len(rr)} identical")
    raw = lambda src, r: ds._arrays()[src][int(r)]
    blk = sum(torch.equal(C.block(src, raw(src, r)), ds.chunk(src, int(r))) for r in rr[:64] for src in ("S1", "S2"))
    wrong_slot = sum(torch.equal(C.block("S1", raw("S2", r)), ds.chunk("S2", int(r))) for r in rr[:64])
    record("X9b generated-block builder", blk == 128, f"{blk}/128 blocks = the dataset's chunk",
           wrong_slot == 0, f"S2 codes put in the S1 slot: {wrong_slot}/64 equal")
    gens = sorted(root.glob("gen/*/*_T*.json"))
    if not gens:
        record("X9c-e chained generation", None, f"no generations under {root} yet")
        return
    bad_cache, bad_ctl, bad_slot, ctl_slot = [], [], [], []
    for f in gens:
        d = json.loads(f.read_text())
        tol = E.CACHE_TOL[d["autocast"]]
        if not d["cache_check"] <= tol:
            bad_cache.append(f.name)
        if not d["cache_check_control_shift1"] > 10 * tol:
            bad_ctl.append(f.name)
        tok = np.load(f.with_suffix(".npz"))["tokens"].astype(np.int64)
        size = C.slot(C.DIRECTIONS[d["direction"]][1])[1]
        in_slot = lambda t: bool(t.min() >= 0 and t.max() < size)
        if not in_slot(tok):
            bad_slot.append(f.name)
        if in_slot(tok + size):
            ctl_slot.append(f.name)
    record("X9c KV cache (generation)", not bad_cache, f"{len(gens) - len(bad_cache)}/{len(gens)} within tolerance",
           not bad_ctl, f"shift-1 control disagreed in {len(gens) - len(bad_ctl)}/{len(gens)}")
    record("X9d generated tokens in the target slot", not bad_slot, f"{len(gens) - len(bad_slot)}/{len(gens)} files",
           not ctl_slot, f"one slot up flagged in {len(gens) - len(ctl_slot)}/{len(gens)}")
    pls = sorted(root.glob("cls/*/plumbing.json"))
    if not pls:
        record("X9e plumbing on the A100", None, "no classifier runs yet")
        return
    ok = [json.loads(f.read_text()) for f in pls]
    record("X9e plumbing on the A100",
           all(p["prompts_equal_dataset"] and p["greedy_equal"] and p["marginal_max_abs_diff"] < 1e-4 for p in ok),
           "; ".join(f"{f.parent.name}: max |dmarginal| {p['marginal_max_abs_diff']:.1e}, greedy equal {p['greedy_equal']}"
                     for f, p in zip(pls, ok)),
           all(p["control_marginal_max_abs_diff"] > 0.05 for p in ok),
           "; ".join(f"{f.parent.name}: {p['control_marginal_max_abs_diff']:.3f}" for f, p in zip(pls, ok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--with-x0", action="store_true")
    ap.add_argument("--full-x4", action="store_true", help="every train row, not 1 in 7")
    ap.add_argument("--lora-run", default=None, help="a finished B10 LoRA run dir, for X6's trained half")
    ap.add_argument("--x7", nargs=3, default=None, metavar=("TAG", "CKPT", "ROLE"))
    ap.add_argument("--head-run", default=None, help="a finished B10 head-only run dir, for X6h's trained half")
    ap.add_argument("--chain-root", type=Path, default=REPORTS / "chain", help="ben_chain.py's --out-root (X9)")
    a = ap.parse_args()
    want = lambda x: a.only is None or x in a.only
    print("verify_phase6")
    if a.with_x0 and want("X0"):
        x0()
    if want("X1"):
        x1()
    if want("X2"):
        x2()
    if want("X3"):
        x3()
    if want("X4"):
        x4(a.full_x4)
    if want("X5"):
        x5()
    if want("X6"):
        x6(a.lora_run)
        x6h(a.head_run)
    if want("X7") and a.x7:
        x7(*a.x7)
    if want("X8"):
        x8()
    if want("X9"):
        x9(a.chain_root)
    fails = [r for r in RESULTS if r[1] is False or r[3] is False]
    skips = [r for r in RESULTS if r[1] is None]
    print(f"\n{len(RESULTS) - len(fails) - len(skips)} pass, {len(fails)} fail, {len(skips)} skipped")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
