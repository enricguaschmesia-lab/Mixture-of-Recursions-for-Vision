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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", default=None, help="write the results as JSON")
    a = ap.parse_args()
    checks = {"Z1": z1, "Z3": z3, "Z2floor": z2_floor}
    if not a.skip_gpu:
        checks["Z2"] = lambda: z2(a.expect_gpu)
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
