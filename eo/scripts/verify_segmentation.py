#!/usr/bin/env python
"""A2's segmentation gate (`a2seg` env). Z1-Z3 land with Step 1; Z4-Z12 with Steps 2-3.

Same discipline as verify_phase3.py and verify_temporal.py: every check ships with a deliberately broken
control that proves it can fail. A check whose artifact does not exist yet is SKIPPED, never passed.

    python eo/scripts/verify_segmentation.py                 # CPU checks, then GPU checks on one card
    python eo/scripts/verify_segmentation.py --skip-gpu      # CPU only
    python eo/scripts/verify_segmentation.py --only Z3

GPU checks run Cosmos's JITs, which need bf16 GEMMs: on the workstation only the TITAN V has them (the
GTX Titan X refuses with CUBLAS_STATUS_NOT_SUPPORTED, 2026-11-18). Pick it by name:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python eo/scripts/verify_segmentation.py --expect-gpu "TITAN V"

  Z1  Palette (eo/maps/palette.py): for every task, class -> colour -> class is exact on every scored
      pixel and the void colour is never predicted; the universal palette has 32 colours with white at
      0 and black at 1, and LULC's colours are Step 0's `distinct` LULC palette; 1,000 shuffled training
      assignments are injective over colours 1..31 with void on white
                                     | a palette with two equal colours is refused, and its round
                                       trip is not exact; two classes on one index, and a scored class
                                       on the void colour, are refused
  Z2  Tokenizer and the two scoring paths (TITAN V): Cosmos Tokenize1-DI8x8 at 256 px encodes and
      decodes PASTIS fold 4 identically twice at a fixed batch shape; its region-rule ceiling read by
      the harness (eo/maps/scoring.Confusion) equals torchmetrics' MulticlassJaccardIndex on the same
      predictions; under Step 0's palette and rule (tau 0.15, radius 4) it reproduces Step 0's stored
      fold-4 reading (0.95224, the regression anchor)
                                     | the same maps transposed before the tokenizer lose the ceiling
                                       (below half of it)
      The harness's own path with a perfect (identity) tokenizer: exactly 1.0 for nearest, vote and
      region, at 224 and 256 px, on all three read sets (the pooling to the native grid is the adjoint
      of the nearest-neighbour render)
                                     | Step 0's area pooling reads 0.81 on PASTIS at 224; crops cut
                                       right but stitched back one pixel off lose the 1.0
  Z3  Scorer = torchmetrics' MulticlassJaccardIndex(average="macro", ignore_index) on synthetic maps
      with void, a class predicted but absent, a class absent from both, and a task with no void
                                     | a scorer that counts void (void as a scored class) disagrees
  Z6  Splits (A2 Step 2.2; eo/geobench/sources.py, views.py): GEO-Bench-2's PASTIS and CloudSen12 as
      stored (PASTIS 1,200 / 482 / 496, CloudSen12 4,000 / 535 / 975), ids unique and disjoint; PASTIS
      train inside PASTIS-R folds 1-3, val = fold 4, test = fold 5, by patch id; every patch's dates and
      48 train/val masks equal PASTIS-R's; the view index: rows numbered once, each view in its sample's
      split, 8 dihedral variants of each training crop and 1 elsewhere, CloudSen12's val/test crops the
      3 x 3 plan, PASTIS frames (p1 in July-August, p7 at int(j T / 7)) with dates read at their index
                                     | a duplicated patch (one train row copied into val) is refused; a
                                       view moved to another split is refused; another patch's mask
                                       differs
  Z7  Inputs (A2 Steps 2.3-2.5; eo/geobench/s2.py): bands land in contract order (PASTIS's ten, B01/B09
      imputed; CloudSen12's first twelve of fourteen); one bilinear interpolation 128 -> 224; dihedral
      variants are the 8 distinct elements and the same function cuts inputs and maps; the imputation
      read (eval.json): the true-band path reproduces the stored BEN tokens exactly and TerraMesh's to
      1e-4, and the chosen method's 10-band round trip is within 10% of the true bands' on both; the
      contract read (contract_*.json): rule 2.4 as fixed does not fail the set (it fails only if the JSD
      is above 2 x BEN's 0.084 AND the round trip above 1.5 x TerraMesh's recorded 0.156), and the
      TerraMesh reference re-run there equals the recorded one; the token roots agree with the contract
      read's own encodings on the rows both hold; and
      (GPU) each arm's stored validation map tokens equal the harness's own path (eo.maps.harness.tokens)
                                     | wrong band order, raw DN and x10^4 are each worse than production
                                       by > 1.5x and above the limit; B01/B09 exchanged loses the
                                       imputation; a two-step resample differs; map tokens one view off
                                       differ
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eo.maps import decode as D            # noqa: E402
from eo.maps import palette as P           # noqa: E402
from eo.maps import scoring as S           # noqa: E402

RESULTS: list[tuple[str, str, str]] = []
STEP0_F4 = Path("/data/enric/reports/a2_seg/tokenizer/codes_f4_t1_di8_256.json")


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, "PASS" if ok else "FAIL", detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def skip(name: str, why: str) -> None:
    RESULTS.append((name, "SKIP", why))
    print(f"[SKIP] {name}: {why}", flush=True)


# ---------------------------------------------------------------- Z1

def _synthetic(task: P.Task, rng: np.random.Generator, n: int = 4) -> np.ndarray:
    """Maps holding every class (void included) in blocks plus noise, at the native size."""
    H = task.native
    lab = rng.integers(0, task.n_classes, size=(n, H, H))
    blocks = rng.integers(0, task.n_classes, size=(n, 8, 8)).repeat(H // 8, 1).repeat(H // 8, 2)
    lab[:, : H // 2] = blocks[:, : H // 2]
    return lab


def _round_trip(colours, assign, task, lab) -> np.ndarray:
    x = P.render(torch.as_tensor(lab), assign, task.native, colours)
    sc = D.rgb_scores(x, np.asarray(colours)[list(assign)])
    return D.snap_nearest(sc, task.ignore).numpy()


def z1() -> None:
    rng = np.random.default_rng(0)
    U = P.UNIVERSAL
    ok_shape = U.shape == (32, 3) and np.allclose(U[0], 1) and np.allclose(U[1], 0)
    lulc0, _ = P.step0_distinct(P.TASKS["lulc"])
    ok_lulc = np.allclose(P.task_colours(P.TASKS["lulc"]), lulc0)
    record("Z1 universal palette", ok_shape and ok_lulc,
           f"32 colours, white 0, black 1: {ok_shape}; LULC = Step 0's distinct LULC palette: {ok_lulc}; "
           f"min scored distance " + ", ".join(f"{k} {P.min_distance(t):.3f}" for k, t in P.TASKS.items()))
    for name, task in P.TASKS.items():
        P.validate(U, task.assign, task.ignore, task.n_classes)
        lab = _synthetic(task, rng)
        pred = _round_trip(U, task.assign, task, lab)
        scored = lab != task.ignore if task.ignore is not None else np.ones_like(lab, bool)
        exact = bool((pred[scored] == lab[scored]).all())
        never_void = task.ignore is None or not bool((pred == task.ignore).any())
        record(f"Z1 round trip {name}", exact and never_void,
               f"{int(scored.sum())} scored pixels exact: {exact}; void never predicted: {never_void}; "
               f"assignment {list(task.assign)}")
    g = torch.Generator().manual_seed(0)
    ok = True
    for name, task in P.TASKS.items():
        a = P.shuffled_assignment(1000, task.n_classes, task.ignore, g).numpy()
        sc = [c for c in range(task.n_classes) if c != task.ignore]
        ok &= all(len(set(r[sc])) == len(sc) for r in a)
        ok &= bool((a[:, sc] >= 1).all() and (a[:, sc] <= 31).all())
        if task.ignore is not None:
            ok &= bool((a[:, task.ignore] == 0).all())
    record("Z1 shuffled assignments", ok, "1,000 draws per task: injective, colours 1..31, void on white")

    # controls
    bad = U.copy()
    t = P.TASKS["pastis"]
    bad[t.assign[1]] = bad[t.assign[0]]                  # two classes' colours made equal
    try:
        P.validate(bad, t.assign, t.ignore, t.n_classes)
        refused = False
    except P.PaletteError:
        refused = True
    lab = _synthetic(t, rng)
    pred = _round_trip(bad, t.assign, t, lab)
    scored = lab != t.ignore
    broken = not bool((pred[scored] == lab[scored]).all())
    shared = list(t.assign)
    shared[2] = shared[1]
    on_void = list(t.assign)
    on_void[3] = P.VOID_COLOUR
    refusals = []
    for a in (shared, on_void):
        try:
            P.validate(U, a, t.ignore, t.n_classes)
            refusals.append(False)
        except P.PaletteError:
            refusals.append(True)
    record("Z1 control: two equal colours", refused and broken and all(refusals),
           f"refused: {refused}; its round trip exact: {not broken}; two classes on one index refused: "
           f"{refusals[0]}; a scored class on the void colour refused: {refusals[1]}")


# ---------------------------------------------------------------- Z3

def _scorer_counting_void(gt, pred, n, ignore) -> float:
    return S.miou(gt, pred, n, None)


def z3() -> None:
    rng = np.random.default_rng(1)
    worst, worst_ctrl, n_cases = 0.0, np.inf, 0
    cases = []
    for trial in range(200):
        n = int(rng.integers(2, 21))
        ignore = None if trial % 5 == 0 else int(rng.integers(0, n))
        gt = rng.integers(0, n, size=(3, 37, 41))
        if ignore is not None:
            gt[:, :5] = ignore                                   # a void band in every map
        absent = [c for c in range(n) if c != ignore and rng.random() < 0.2]
        for c in absent:                                         # classes absent from the targets
            gt[gt == c] = (ignore + 1) % n if ignore is not None and (ignore + 1) % n not in absent else gt[gt == c]
        choices = [c for c in range(n) if c != ignore]
        pred = rng.choice(choices, size=gt.shape)                # predicted classes may be absent in gt
        if trial % 7 == 0 and len(choices) > 2:
            pred[pred == choices[-1]] = choices[0]               # a class absent from both
            gt[gt == choices[-1]] = choices[0]
        cases.append((gt, pred, n, ignore))
    for gt, pred, n, ignore in cases:
        a = S.miou(gt, pred, n, ignore)
        b = S.torchmetrics_miou(gt, pred, n, ignore)
        worst = max(worst, abs(a - b))
        n_cases += 1
        if ignore is not None:
            worst_ctrl = min(worst_ctrl, abs(_scorer_counting_void(gt, pred, n, ignore) - b))
    record("Z3 scorer = torchmetrics MulticlassJaccardIndex", worst < 1e-6,
           f"{n_cases} synthetic cases (void bands, absent and predicted-absent classes, no-void tasks): "
           f"largest |difference| {worst:.2e}")
    record("Z3 control: a scorer that counts void", worst_ctrl > 1e-3,
           f"smallest |difference| from torchmetrics over the cases with void: {worst_ctrl:.4f}")


# ---------------------------------------------------------------- Z2

def z2(expect_gpu: str | None) -> None:
    if not torch.cuda.is_available():
        skip("Z2", "no GPU visible")
        return
    name = torch.cuda.get_device_name(0)
    if expect_gpu and expect_gpu not in name:
        raise SystemExit(f"expected {expect_gpu}, CUDA returned {name}")
    from eo.maps import harness as H
    from eo.maps import sets
    from eo.maps.tokenizers import cosmos, run_fixed
    tok = cosmos("t1_di8_256")
    lab, _ = sets.load("pastis", "val")
    x = H.render_crops(lab[:48], "pastis", "universal", tok.input_size, "crop224", "cuda")
    i1 = run_fixed(tok.encode, x, tok.batch)
    i2 = run_fixed(tok.encode, x, tok.batch)
    y1 = run_fixed(tok.decode, i1, tok.batch)
    y2 = run_fixed(tok.decode, i1, tok.batch)
    det = bool(torch.equal(i1, i2)) and bool(torch.equal(y1, y2))
    record("Z2 determinism", det, f"48 maps, batch {tok.batch}: codes identical {bool(torch.equal(i1, i2))}, "
           f"decoded RGB identical {bool(torch.equal(y1, y2))} ({name})")
    # across processes: the token cache is written by one process and decoded by another
    import hashlib
    digest = hashlib.sha256(i1.cpu().numpy().astype(np.int32).tobytes()).hexdigest()
    pin = Path("/data/enric/reports/a2_seg/step1/gate/z2_codes_digest.json")
    pins = json.loads(pin.read_text()) if pin.exists() else {}
    if name in pins:
        record("Z2 determinism across processes", pins[name] == digest,
               f"codes of the 48 maps against the digest pinned on {name}: {digest[:16]} vs {pins[name][:16]}")
    else:
        pin.parent.mkdir(parents=True, exist_ok=True)
        pins[name] = digest
        pin.write_text(json.dumps(pins, indent=1))
        skip("Z2 determinism across processes", f"digest pinned now for {name}; the next run compares")

    rule = {"rule": "region", "tau": 0.15, "radius": 4}
    preds: dict = {}
    res = H.sweep(tok, "pastis", "val", [rule], palette="step0", cache=False, return_pred=preds)
    k = D.rule_key(rule)
    native = res[k]["mIoU"]
    tm = S.torchmetrics_miou(np.asarray(lab), preds[k], 20, 19)
    record("Z2 two scoring paths", abs(native - tm) < 1e-6,
           f"PASTIS fold 4, t1_di8_256, region (0.15, 4), Step 0's palette: harness {native:.6f}, "
           f"torchmetrics {tm:.6f}")
    if STEP0_F4.exists():
        s0 = json.loads(STEP0_F4.read_text())["region_tau0.15_r4"]["mIoU"]
        record("Z2 regression anchor (Step 0)", abs(native - s0) < 2e-3,
               f"harness {native:.5f} against Step 0's {s0:.5f} (|diff| {abs(native - s0):.1e}; tokens "
               "re-encoded at batch 16 with the tail padded, so the last maps may differ by a code)")
    else:
        skip("Z2 regression anchor (Step 0)", f"{STEP0_F4} missing")

    # control: the tokenizer reads the transposed maps; scored against the maps as they are
    labT = np.ascontiguousarray(np.transpose(np.asarray(lab), (0, 2, 1)))
    xT = H.render_crops(labT, "pastis", "step0", tok.input_size, "crop224", "cuda")
    iT = run_fixed(tok.encode, xT, tok.batch)
    conf = S.Confusion(20, 19, "cuda")
    for s in range(0, len(labT), 64):
        sc = H.class_scores(tok, iT[s:s + 64], "pastis", "step0", "crop224")
        conf.update(torch.as_tensor(np.asarray(lab[s:s + 64]), device="cuda").long(),
                    D.apply_rule(sc, 19, rule, "rgb"))
    t_miou = conf.result()["mIoU"]
    record("Z2 control: transposed maps", t_miou < 0.5 * native,
           f"ceiling {t_miou:.4f} against {native:.4f}")


def z2_floor() -> None:
    """The harness's own path (render, cut, scores, pooling to the native grid, stitch, rule) with a
    perfect (identity) tokenizer reads exactly 1.0 for every rule, input size and dataset. Runs on the
    visible card, else CPU."""
    from eo.maps import harness as H
    from eo.maps import tiling
    from eo.maps.tokenizers import Identity
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rules = [{"rule": "nearest"}, {"rule": "vote"}, {"rule": "region", "tau": 0.15, "radius": 4}]
    got = {}
    for size in (224, 256):
        tok = Identity(size, dev)
        for d in ("pastis", "lulc", "cloudsen12"):
            r = H.sweep(tok, d, "val", rules, cache=False, device=dev)
            got.update({f"{d}@{size}:{k}": v["mIoU"] for k, v in r.items()})
    ok = all(v == 1.0 for v in got.values())
    worst = min(got, key=got.get)
    record("Z2 harness floor (identity tokenizer)", ok,
           f"{len(got)} readings (3 rules x 2 sizes x 3 datasets) exactly 1.0: {ok}; lowest {worst} {got[worst]:.6f}")
    # control 1: Step 0's area pooling loses the perfect read where the scale is not a whole multiple
    orig_native = D.to_native
    D.to_native = D.to_native_area
    try:
        m = H.sweep(Identity(224, dev), "pastis", "val", [{"rule": "nearest"}], cache=False,
                    device=dev)["nearest"]["mIoU"]
    finally:
        D.to_native = orig_native
    record("Z2 control: Step 0's area pooling", m < 0.9, f"PASTIS fold 4 at 224, nearest, perfect tokenizer: {m:.4f}")
    # control 2: crops cut at the right offsets but stitched back one pixel off
    orig = tiling.stitch

    def shifted(scores, crops, native):
        return orig(scores, [(max(y - 1, 0), max(x - 1, 0), c) for (y, x, c) in crops], native)
    tiling.stitch = shifted
    try:
        m = H.sweep(Identity(224, dev), "cloudsen12", "val", [{"rule": "nearest"}], cache=False,
                    device=dev)["nearest"]["mIoU"]
    finally:
        tiling.stitch = orig
    record("Z2 control: crops stitched one pixel off", m < 1.0, f"CloudSen12 val, crops stitched at offset -1: {m:.4f}")


# ---------------------------------------------------------------- Z6

STEP2 = Path("/data/enric/reports/a2_seg/step2")


def split_problems(task: str, t) -> list[str]:
    """What is wrong with a split table (PASTIS: patch_id; CloudSen12: id) against the stored counts,
    uniqueness, disjointness and, for PASTIS, PASTIS-R's folds."""
    from eo.geobench import sources as Src
    key = "patch_id" if task == "pastis" else "id"
    out = []
    counts = t.split.value_counts().to_dict()
    if counts != Src.COUNTS[task]:
        out.append(f"counts {counts} != {Src.COUNTS[task]}")
    if not t[key].is_unique:
        out.append(f"{int(t[key].duplicated().sum())} duplicated ids")
    ids = {s: set(t[key][t.split == s]) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if ids[a] & ids[b]:
            out.append(f"{len(ids[a] & ids[b])} ids in both {a} and {b}")
    if task == "pastis":
        folds = Src.pastis_r_fold_ids()
        f = {k: {i for i, v in folds.items() if v in k} for k in ((1, 2, 3), (4,), (5,))}
        if not ids["train"] <= f[(1, 2, 3)]:
            out.append("train not inside folds 1-3")
        if ids["val"] != f[(4,)]:
            out.append("val != fold 4")
        if ids["test"] != f[(5,)]:
            out.append("test != fold 5")
    return out


def view_problems(task: str, idx: dict, t) -> list[str]:
    from eo.geobench import sources as Src, views as Vw
    from eo.maps import tiling
    v, x = idx["views"], idx["inputs"]
    out = []
    if not (v.view.values == np.arange(len(v))).all():
        out.append("views not numbered 0..n-1")
    split_of = t.set_index("idx").split
    if (v.sample_idx.map(split_of) != v.split).any():
        out.append(f"{int((v.sample_idx.map(split_of) != v.split).sum())} views not in their sample's split")
    per = v.groupby(["split", "sample_idx", "y0", "x0"]).k.apply(lambda s: tuple(sorted(s)))
    for s, want in (("train", tuple(range(8))), ("val", (0,)), ("test", (0,))):
        got = set(per.loc[s]) if s in per.index.get_level_values(0) else set()
        if got != {want}:
            out.append(f"{s}: dihedral variants per crop {sorted(got)[:3]} != {want}")
    if task == "cloudsen12":
        plan = sorted((y, x0) for (y, x0, _) in tiling.plan("cloudsen12", 512, "crop224"))
        g = v[v.split != "train"].groupby("sample_idx").apply(lambda d: sorted(zip(d.y0, d.x0)))
        if not all(p == plan for p in g):
            out.append("val/test crops are not the 3 x 3 plan")
        if ((v.y0 < 0) | (v.x0 < 0) | (v.y0 + v["size"] > 512) | (v.x0 + v["size"] > 512)).any():
            out.append("a crop outside its tile")
        n_train = v[v.split == "train"].groupby("sample_idx").size()
        if set(n_train) != {Vw.N_RANDOM_CROPS * 8}:
            out.append("not 9 crops x 8 variants per training tile")
    else:
        if not (x.row.values == np.arange(len(x))).all():
            out.append("inputs not numbered 0..n-1")
        if set(x.groupby("view").size()) != {8}:
            out.append("not 8 frames per view")
        dates = t.set_index("idx").dates
        xv = x.merge(v[["view", "sample_idx"]], on="view")
        bad = sum(int(dates[r.sample_idx][r.t]) != r.date for r in xv.itertuples())
        if bad:
            out.append(f"{bad} input dates differ from the patch's date at t")
        p1 = xv[xv.frame == "p1"]
        if not p1.date.astype(str).str[4:6].isin(["07", "08"]).all():
            out.append("a p1 frame outside July-August")
        p7 = xv[xv.frame.str.startswith("p7")]
        want = p7.apply(lambda r: Src.pastis_frames_p7(len(dates[r.sample_idx]))[int(r.frame[3:])], axis=1)
        if (want.values != p7.t.values).any():
            out.append("p7 frame indices differ from int(j T / 7)")
    return out


def z6() -> None:
    from eo.geobench import sources as Src, views as Vw
    try:
        tp, tc = Src.pastis_table(), Src.cloudsen12_table()
    except Exception as e:                                         # data not on disk yet
        skip("Z6", f"GEO-Bench-2 tortillas unreadable: {e}")
        return
    for task, t in (("pastis", tp), ("cloudsen12", tc)):
        p = split_problems(task, t)
        record(f"Z6 splits {task}", not p, f"{t.split.value_counts().to_dict()}; problems: {p or 'none'}"
               + ("; train inside folds 1-3, val = fold 4, test = fold 5" if task == "pastis" and not p else ""))
        bad = t.copy()
        dup = bad[bad.split == "train"].iloc[:1].assign(split="val")
        import pandas as pd
        bad = pd.concat([bad, dup], ignore_index=True)
        pc = split_problems(task, bad)
        record(f"Z6 control: a duplicated {task} patch", bool(pc), f"refused with {pc}")
    # PASTIS data against PASTIS-R: every date list, 48 train/val masks
    rdates = {}
    meta = json.loads((Src.PASTIS_R / "metadata.geojson").read_text())
    for ft in meta["features"]:
        d = ft["properties"]["dates-S2"]
        d = json.loads(d.replace("'", '"')) if isinstance(d, str) else d
        rdates[int(ft["properties"]["ID_PATCH"])] = [int(d[k]) for k in sorted(d, key=int)]
    nd = sum(list(map(int, d)) != rdates[int(p)] for p, d in zip(tp.patch_id, tp.dates))
    tv = tp[tp.split.isin(["train", "val"])]
    pick = tv.iloc[np.sort(np.random.default_rng(6).choice(len(tv), 48, replace=False))]
    masks = {int(r.patch_id): Src.pastis_sample(r.idx, ("semantic",))["semantic"] for r in pick.itertuples()}
    nm = sum(not np.array_equal(m, np.load(Src.PASTIS_R / "ANNOTATIONS" / f"TARGET_{p}.npy")[0]) for p, m in masks.items())
    record("Z6 PASTIS = PASTIS-R", nd == 0 and nm == 0,
           f"dates differ on {nd} of {len(tp)} patches; masks differ on {nm} of 48 train/val patches"
           + (f" (the full read: {STEP2 / 'splits.json'})" if (STEP2 / "splits.json").exists() else ""))
    ids = list(masks)
    nm_ctrl = sum(not np.array_equal(masks[ids[i]], np.load(Src.PASTIS_R / "ANNOTATIONS" / f"TARGET_{ids[(i + 1) % len(ids)]}.npy")[0])
                  for i in range(len(ids)))
    record("Z6 control: another patch's mask", nm_ctrl == len(ids), f"{nm_ctrl} of {len(ids)} differ")
    for task, t in (("pastis", tp), ("cloudsen12", tc)):
        try:
            idx = Vw.load(task)
        except FileNotFoundError:
            skip(f"Z6 views {task}", "views.parquet not written yet")
            continue
        fresh = Vw.build(task)
        same = all(fresh[k].reset_index(drop=True).equals(idx[k]) for k in ("views", "inputs"))
        p = view_problems(task, idx, t)
        record(f"Z6 views {task}", not p and same,
               f"{len(idx['views'])} views, {len(idx['inputs'])} inputs; stored = rebuilt: {same}; problems: {p or 'none'}")
        bad = {k: d.copy() for k, d in idx.items()}
        j = bad["views"].index[bad["views"].split == "val"][0]
        bad["views"].loc[j, "split"] = "train"
        pc = view_problems(task, bad, t)
        record(f"Z6 control: a {task} view moved to another split", bool(pc), f"refused with {pc[:2]}")


# ---------------------------------------------------------------- Z7

def z7() -> None:
    import torch.nn.functional as F
    from eo.geobench import s2 as G
    from eo.terramesh_tok import contract as C
    # bands: a synthetic frame where each band holds its own number
    code = {b: 100.0 * (i + 1) for i, b in enumerate(C.BAND_ORDER["S2L2A"])}
    x10 = torch.stack([torch.full((2, 8, 8), code[b]) for b in G.PASTIS_BANDS], 1)
    y = G.assemble_pastis(x10, "copy")
    want = [code["B02"] if b == "B01" else code["B8A"] if b == "B09" else code[b] for b in G.ORDER]
    got = y[0, :, 0, 0].tolist()
    x14 = torch.stack([torch.full((2, 8, 8), code[b]) for b in G.ORDER] + [torch.full((2, 8, 8), 7.0)] * 2, 1)
    c12, _ = G.cloudsen12_physical(x14, "cpu")
    ok = got == want and c12[0, :, 0, 0].tolist() == [code[b] for b in G.ORDER]
    record("Z7 band order", ok, f"PASTIS's ten in contract positions, B01 <- B02 and B09 <- B8A under copy: "
           f"{got == want}; CloudSen12's first 12 of 14 in contract order, AOT/WVP dropped: "
           f"{c12[0, :, 0, 0].tolist() == [code[b] for b in G.ORDER]}")
    rev = G.pastis_input(x10, "copy", "cpu", standardize_=False, band_perm=list(range(12))[::-1])
    record("Z7 control: reversed bands", rev[0, :, 0, 0].tolist() != want, "the reversed order is not the contract's")
    # one bilinear interpolation
    fr = torch.rand(3, 10, 128, 128) * 4000
    z = G.pastis_input(fr, "copy", "cpu")
    ref = G.standardize(F.interpolate(G.assemble_pastis(fr, "copy"), size=(224, 224), mode="bilinear", align_corners=False))
    two = G.standardize(F.interpolate(F.interpolate(G.assemble_pastis(fr, "copy"), size=(256, 256), mode="bilinear",
                                                    align_corners=False), size=(224, 224), mode="bilinear", align_corners=False))
    record("Z7 resampling", bool(torch.equal(z, ref)) and z.shape[-1] == C.CROP,
           f"pastis_input = one bilinear 128 -> {C.CROP} then v1 stats, exactly")
    record("Z7 control: two-step resample", float((two - ref).abs().max()) > 1e-3,
           f"128 -> 256 -> 224 differs by up to {float((two - ref).abs().max()):.3f} in z")
    # dihedral: 8 distinct transforms, k=0 the identity, numpy = torch
    a = np.random.default_rng(0).integers(0, 9, (5, 7, 7))
    outs = [G.dihedral(a, k) for k in range(8)]
    distinct = len({o.tobytes() for o in outs}) == 8
    same_t = all(np.array_equal(G.dihedral(torch.as_tensor(a), k).numpy(), outs[k]) for k in range(8))
    record("Z7 dihedral", distinct and np.array_equal(outs[0], a) and same_t,
           f"8 distinct: {distinct}; k=0 identity; numpy = torch: {same_t}; inputs and maps both cut with "
           "eo/geobench/s2.dihedral (tokenize_geobench.build_inputs, tokenize_maps.view_labels)")

    # the imputation read
    ev = STEP2 / "impute" / "eval.json"
    if ev.exists():
        e = json.loads(ev.read_text())
        m = G.CHOSEN_IMPUTATION or "linear"
        det = []
        ok = True
        for s in ("BEN", "TerraMesh"):
            tr, ch = e[s]["true"]["roundtrip_10"]["rmse_z_ml7"]["median"], e[s][m]["roundtrip_10"]["rmse_z_ml7"]["median"]
            # BEN exactly; TerraMesh's stored val_clean tokens were written by another run (tokenize_clean),
            # where ~1 token in 1e4-1e5 flips at an FSQ boundary (CLAUDE.md: cross-card ~1e-4)
            ok &= e[s]["identity_vs_stored"] >= (1.0 if s == "BEN" else 0.9999) and ch <= 1.10 * tr
            det.append(f"{s}: identity {e[s]['identity_vs_stored']:.6f}, {m} {ch:.4f} vs true {tr:.4f}, "
                       f"agreement {e[s][m]['token_agreement_mean']:.3f} (mean fill {e[s]['mean']['token_agreement_mean']:.3f})")
        record(f"Z7 imputation ({m})", ok, "; ".join(det))
        ctl = [(s, e[s]["linear_swapped"]) for s in ("BEN", "TerraMesh")]
        okc = all(c["token_agreement_mean"] < 0.5 * e[s][m]["token_agreement_mean"]
                  and c["roundtrip_10"]["rmse_z_ml7"]["median"] > 1.5 * e[s]["true"]["roundtrip_10"]["rmse_z_ml7"]["median"]
                  for s, c in ctl)
        record("Z7 control: B01/B09 exchanged", okc, "; ".join(
            f"{s}: agreement {c['token_agreement_mean']:.3f}, ml7 {c['roundtrip_10']['rmse_z_ml7']['median']:.4f}" for s, c in ctl))
    else:
        skip("Z7 imputation", f"{ev} missing")

    # the contract read
    for task in ("cloudsen12", "pastis"):
        f = STEP2 / "contract" / f"contract_{task}.json"
        if not f.exists():
            skip(f"Z7 round trip {task}", f"{f} missing")
            continue
        r = json.loads(f.read_text())
        ref = r["references"]
        lim, jlim = ref["rt_limit"], ref["jsd_limit"]
        rt = r[task]["roundtrip"]
        p = rt["production"]["rmse_z_ml7"]["median"]
        j = r[task]["usage"][f"jsd_{task}_vs_terramesh"]
        # rule 2.4, fixed: an input set fails only if BOTH its JSD > 2 x BEN's AND its round trip > 1.5 x TerraMesh's
        fails = j > jlim and p > lim
        rerun_ok = abs(r["terramesh_rerun"]["rmse_z_ml7"]["median"] - ref["terramesh_rt_ml7_median"]) < 0.01
        record(f"Z7 round trip and rule 2.4 {task}", not fails and rerun_ok,
               f"rmse_z_ml7 {p:.4f} (limit {lim:.4f}, above: {p > lim}; in range {rt['production']['rmse_z_in']['median']:.4f} "
               f"on {rt['production']['frac_in_range']['median']:.3f} of pixels); JSD {j:.4f} at n {r[task]['usage']['n']} "
               f"(limit {jlim:.4f}, above: {j > jlim}); fails the rule: {fails}; TerraMesh re-run "
               f"{r['terramesh_rerun']['rmse_z_ml7']['median']:.4f} = recorded {ref['terramesh_rt_ml7_median']:.4f}: {rerun_ok}")
        cs = {k: v["rmse_z_ml7"]["median"] for k, v in rt.items() if k.startswith("ctrl_")}
        record(f"Z7 control: {task} broken inputs", all(v > 1.5 * p and v > lim for v in cs.values()),
               ", ".join(f"{k[5:]} {v:.4f}" for k, v in cs.items()) + f" against production {p:.4f}")
    # token roots against the contract read's own encodings
    from eo.geobench import views as Vw
    for task, ref_f, sel in (("cloudsen12", "cloudsen12_train_first_crop_tokens.npy", None),
                             ("pastis", f"pastis_train_p1_k0_tokens_{G.CHOSEN_IMPUTATION}.npy", None)):
        rp = STEP2 / "contract" / ref_f
        tp = Vw.TOK_ROOT[task] / C.tok_dir_name("S2L2A")
        if not rp.exists() or not (tp / "present.npy").exists():
            skip(f"Z7 token root {task}", "contract tokens or token root missing")
            continue
        idx = Vw.load(task)
        v, x = idx["views"], idx["inputs"]
        if task == "cloudsen12":
            rows = np.array(json.loads((STEP2 / "contract" / "cloudsen12_train_first_crop_views.json").read_text()))
        else:
            xv = x.merge(v, on="view")
            rows = xv[(xv.frame == "p1") & (xv.split == "train") & (xv.k == 0)].sort_values("row").row.values
        pres = np.load(tp / "present.npy")
        toks = np.load(tp / "tokens.npy", mmap_mode="r")
        ref = np.load(rp)
        have = pres[rows]
        if not have.any():
            skip(f"Z7 token root {task}", "no overlapping rows tokenized yet")
            continue
        eq = (np.asarray(toks[rows[have]]) == ref[have]).mean()
        record(f"Z7 token root {task}", eq == 1.0, f"{int(have.sum())} rows: stored tokens = the contract read's encodings: {eq:.6f}")
        shifted = (np.asarray(toks[rows[have]][1:]) == ref[have][:-1]).mean()
        record(f"Z7 control: {task} rows one off", shifted < 0.5, f"agreement {shifted:.4f}")


def z7_maps(expect_gpu: str | None) -> None:
    """Each arm's stored validation map tokens = the harness's own render-and-encode path."""
    if not torch.cuda.is_available():
        skip("Z7 maps", "no GPU visible")
        return
    name = torch.cuda.get_device_name(0)
    if expect_gpu and expect_gpu not in name:
        raise SystemExit(f"expected {expect_gpu}, CUDA returned {name}")
    from eo.geobench import views as Vw
    from eo.maps import frozen, harness as H
    for task, ds in (("pastis", "pastis"), ("cloudsen12", "cloudsen12")):
        v = Vw.load(task)["views"]
        rows = v[v.split == "val"].view.values
        for arm in frozen.ARMS:
            d = Vw.TOK_ROOT[task] / f"MAP_{arm}"
            if not (d / "metadata.json").exists():
                skip(f"Z7 maps {task}/{arm}", "not tokenized yet")
                continue
            meta = json.loads((d / "metadata.json").read_text())
            pres = np.load(d / "present.npy")
            if not pres[rows].all():
                skip(f"Z7 maps {task}/{arm}", "val not fully tokenized")
                continue
            if meta["gpu"] != name:
                skip(f"Z7 maps {task}/{arm}", f"written on {meta['gpu']}, this card is {name}")
                continue
            tok = frozen.load(arm)
            ref = H.tokens(tok, ds, "val", cache=False).reshape(len(rows), -1)
            mine = np.asarray(np.load(d / "tokens.npy", mmap_mode="r")[rows]).astype(np.int64)
            record(f"Z7 maps {task}/{arm}", np.array_equal(mine, ref),
                   f"{len(rows)} val maps: stored = harness, mismatching tokens {(mine != ref).mean():.2e}")
            off = (mine[1:] == ref[:-1]).mean()
            record(f"Z7 control: {task}/{arm} maps one view off", off < 0.9, f"agreement {off:.4f}")
            del tok
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", default=None, help="write the results as JSON")
    a = ap.parse_args()
    checks = {"Z1": z1, "Z3": z3, "Z2floor": z2_floor, "Z6": z6, "Z7": z7}
    if not a.skip_gpu:
        checks["Z2"] = lambda: z2(a.expect_gpu)
        checks["Z7maps"] = lambda: z7_maps(a.expect_gpu)
    t0 = time.time()
    for k, fn in checks.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            fn()
        except SystemExit:
            raise
        except Exception as e:                                      # a crash is a failure
            traceback.print_exc()
            record(k, False, f"raised {type(e).__name__}: {e}")
    n_fail = sum(r[1] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS)} checks: {sum(r[1] == 'PASS' for r in RESULTS)} pass, {n_fail} fail, "
          f"{sum(r[1] == 'SKIP' for r in RESULTS)} skipped ({time.time() - t0:.0f} s)")
    if a.out:
        Path(a.out).write_text(json.dumps([{"check": c, "status": s, "detail": d} for c, s, d in RESULTS], indent=1))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
