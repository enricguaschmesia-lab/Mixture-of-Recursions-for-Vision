"""Is decoding without the DiVAE clamp safe on GENERATED tokens? (2026-10-01)

clamp_test.py showed that on ground-truth tokens the unclamped sampler
(thresholding=False, clip_sample=False) beats the released one at every |z|
and never meaningfully loses. That does not settle D3.11, which decodes the
MODEL's tokens: the clamp may also have been capping off-distribution
generations (arm A's checkpoint-16500 DEM is hundreds of metres off), and
removing it could then make generated scores worse, not better. This is the
check on worklog section 4 item 33.

It re-decodes the D3.11 artifacts of arm A at two checkpoints -- 2000 (near
the held-out optimum) and 16500 (final, memorized) -- with the SAME rows,
grids, shuffle, seed, batch and timesteps as sweep_d311.sh, and compares
against the released-sampler per-scene metrics already stored in each
metrics_<T>.json. Nothing in /data/enric/reports/d311 is written.

PASS CRITERIA, fixed before the first run:
  G0  reproduction  re-decoding the first batch with the RELEASED sampler
                    reproduces the stored per-scene rmse_z_ml7 (|delta| <=
                    1e-4, the artifact's rounding) for ceiling, shuffled and
                    generated, every target and checkpoint. Otherwise the
                    comparison below is not paired.
  G1  no masking    over in-range scenes (|z| < 1), at most 10% of generated
                    scenes get worse by more than 10% (noclip > 1.10 x
                    released), per target and checkpoint. A clamp that was
                    hiding wild generations would show up here.
  G2  control       under noclip the shuffled control still collapses against
                    the ceiling by W7's 1.25x margin on rmse_z_ml7, over the
                    in-range scenes AND over all scenes.
Recorded without a criterion: the generated - ceiling gap under both
samplers, in-range vs all-scenes means, output range, and the same collapse
on raw rmse_z (no multilook).

Outcome (2026-10-01 evening, docs/worklog.md): G0 and G2 pass on all five
targets. G1 FAILS on 7 of 10 (target, checkpoint) pairs: unclamped, 3-44% of
in-range generated scenes get > 10% worse (S1RTC worst), while the ceiling
improves and the shuffled control is unchanged, so the generated - ceiling gap
roughly doubles. Leading reading: the clamp was capping MODEL error
(generations implying values beyond +-1 on scenes whose truth is inside).
Not excluded: decoder artefacts on off-distribution token grids. The test
that separates them is a code -> mean-patch-value lookup.

Runs in `mor`, from the repo root, on the Titan X (shares it; memory-capped):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        python eo/experiments/clamp_generated.py
Writes /data/enric/reports/clamp_test/generated/{gen_<T>.json, summary.json}.
"""
import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
for p in (REPO / "eo" / "experiments", REPO / "eo" / "scripts", REPO / "eo"):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import clamp_test as CT  # noqa: E402
import decode_eo as D  # noqa: E402
from terramesh_tok import contract as C, preprocess as P, tokenizers as T  # noqa: E402

MODS = ["DEM", "S2L2A", "NDVI", "S1RTC", "S1GRD"]
CKPTS = ["checkpoint-2000", "checkpoint-16500"]
GEN_ROOT = pathlib.Path("/data/enric/generations/d311/arm_a")
REP_ROOT = pathlib.Path("/data/enric/reports/d311/arm_a")
W7_MARGIN = 1.25     # verify_phase3.W7_MARGIN (not imported: that module runs in .venv)
SEED, BATCH, STEPS = 0, 4, 50   # sweep_d311.sh's decode protocol


def metrics(out, refs, mod):
    rows = []
    for i in range(len(out)):
        m = D.continuous_metrics(out[i], P.destandardize(torch.from_numpy(out[i]), mod).numpy(),
                                 refs[i], mod)
        m["out_absmax"] = round(float(np.abs(out[i]).max()), 3)
        rows.append(m)
    return rows


def mean(rows, key, sel):
    v = [r[key] for r, s in zip(rows, sel) if s and r.get(key) is not None]
    return round(float(np.mean(v)), 4) if v else None


def run_target(mod, args, out_dir):
    t0 = time.time()
    stored = {ck: json.loads((REP_ROOT / ck / f"{mod}_arm_a_{ck}" / f"metrics_{mod}.json").read_text())
              for ck in CKPTS}
    gdir = {ck: GEN_ROOT / ck / mod / "slot_masked" for ck in CKPTS}
    rows = np.load(gdir[CKPTS[0]] / f"decode_rows_{mod}.npy")
    for ck in CKPTS[1:]:
        assert np.array_equal(rows, np.load(gdir[ck] / f"decode_rows_{mod}.npy")), f"{mod}: rows differ"
    n = min(len(rows), args.limit) if args.limit else len(rows)
    rows = rows[:n]

    # Exactly decode_eo's construction, so ceiling/shuffled are the stored ones.
    art = np.load(D.VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")
    truth_all = np.asarray(art[np.load(gdir[CKPTS[0]] / f"decode_rows_{mod}.npy")]
                           ).astype(np.int64).reshape(-1, C.GRID, C.GRID)
    rng = np.random.default_rng(SEED)
    shuf_all = np.stack([s.ravel()[rng.permutation(C.TOKENS_PER_SAMPLE)].reshape(C.GRID, C.GRID)
                         for s in truth_all])
    sources = {"ceiling": truth_all[:n], "shuffled": shuf_all[:n]}
    for ck in CKPTS:
        sources[ck] = np.load(gdir[ck] / f"decode_grids_{mod}.npy")[:n]

    rasters = D.read_rasters(mod, rows)
    refs = [P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32) for a in rasters]
    zs = np.array([D.scene_z(mod, a) for a in rasters])
    inr = zs < 1.0
    for ck in CKPTS:
        assert [s["in_range"] for s in stored[ck]["per_scene"]["ceiling"][:n]] == inr.tolist()
        for src in ("ceiling", "shuffled"):   # the controls must not depend on the checkpoint
            assert stored[ck]["per_scene"][src] == stored[CKPTS[0]]["per_scene"][src], (mod, ck, src)

    tok = T.build(mod, device=T.DEVICE)
    noclip = CT.scheduler_like(tok, **CT.VARIANTS["noclip"])

    # --- G0: the released sampler reproduces the stored artifact ------------
    g0 = {}
    for name, grids in sources.items():
        ck = name if name in CKPTS else CKPTS[0]
        key = "generated" if name in CKPTS else name
        out = D.decode_batched(tok, grids[:BATCH], BATCH, SEED, T.DEVICE, STEPS).numpy()
        got = [m["rmse_z_ml7"] for m in metrics(out, refs[:BATCH], mod)]
        want = [s["rmse_z_ml7"] for s in stored[ck]["per_scene"][key][:BATCH]]
        g0[name] = round(float(np.max(np.abs(np.array(got) - np.array(want)))), 6)
    print(f"[{mod}] G0 max |delta| {g0}", flush=True)

    # --- unclamped decode of every source ------------------------------------
    nc = {}
    for name, grids in sources.items():
        t1 = time.time()
        out = CT.decode(tok, grids, noclip, BATCH, SEED, STEPS).numpy()
        nc[name] = metrics(out, refs, mod)
        print(f"[{mod}] noclip {name:16s} {n} scenes in {time.time() - t1:.0f}s", flush=True)
    del tok; T.build.cache_clear(); torch.cuda.empty_cache()

    rel = {"ceiling": stored[CKPTS[0]]["per_scene"]["ceiling"][:n],
           "shuffled": stored[CKPTS[0]]["per_scene"]["shuffled"][:n]}
    for ck in CKPTS:
        rel[ck] = stored[ck]["per_scene"]["generated"][:n]

    allsel = np.ones(n, bool)
    summ = {}
    for samp, per in (("released", rel), ("noclip", nc)):
        s = {}
        for name in sources:
            s[name] = {sel_name: {k: mean(per[name], k, sel) for k in ("rmse_z_ml7", "rmse_z")}
                       for sel_name, sel in (("in_range", inr), ("all", allsel), ("out_of_range", ~inr))}
        for sel_name in ("in_range", "all"):
            for k in ("rmse_z_ml7", "rmse_z"):
                c, sh = s["ceiling"][sel_name][k], s["shuffled"][sel_name][k]
                s.setdefault("collapse", {}).setdefault(sel_name, {})[k] = \
                    round(sh / c, 3) if c and sh else None
        summ[samp] = s

    tests = {"G0_reproduction": {"pass": all(v <= 1e-4 for v in g0.values()), **g0}}
    for ck in CKPTS:
        r = np.array([x["rmse_z_ml7"] for x in rel[ck]]); q = np.array([x["rmse_z_ml7"] for x in nc[ck]])
        worse = (q > 1.10 * r) & inr
        ratio = q / r
        tests[f"G1_{ck}"] = {
            "pass": bool(worse.sum() <= 0.10 * inr.sum()),
            "n_in_range": int(inr.sum()), "n_worse_10pct": int(worse.sum()),
            "median_ratio_in_range": round(float(np.median(ratio[inr])), 4),
            "max_ratio_in_range": round(float(ratio[inr].max()), 4),
            "median_ratio_out_of_range": (round(float(np.median(ratio[~inr])), 4)
                                          if (~inr).any() else None),
            "gen_absmax_noclip": float(max(x["out_absmax"] for x in nc[ck])),
        }
    col = summ["noclip"]["collapse"]
    tests["G2_control"] = {"pass": all(col[s]["rmse_z_ml7"] is not None
                                       and col[s]["rmse_z_ml7"] >= W7_MARGIN
                                       for s in ("in_range", "all")),
                           "in_range": col["in_range"]["rmse_z_ml7"], "all": col["all"]["rmse_z_ml7"]}
    tests["ceiling_absmax_noclip"] = float(max(x["out_absmax"] for x in nc["ceiling"]))
    print(f"[{mod}] tests {json.dumps(tests)}", flush=True)

    doc = {"target": mod, "n": n, "rows": rows.tolist(), "z": zs.round(4).tolist(),
           "n_in_range": int(inr.sum()), "summary": summ, "tests": tests,
           "per_scene_noclip": nc,
           "provenance": {"gpu": torch.cuda.get_device_name(0), "seed": SEED, "batch": BATCH,
                          "timesteps": STEPS, "ckpts": CKPTS, "arm": "arm_a",
                          "seconds": round(time.time() - t0)}}
    (out_dir / f"gen_{mod}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mods", nargs="*", default=MODS)
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    ap.add_argument("--mem-fraction", type=float, default=0.45)
    ap.add_argument("--out", default="/data/enric/reports/clamp_test/generated")
    args = ap.parse_args()

    print(f"gpu: {D.assert_gpu(False)}", flush=True)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for mod in args.mods:
        f = out_dir / f"gen_{mod}.json"
        doc = json.loads(f.read_text()) if f.exists() else run_target(mod, args, out_dir)
        summary[mod] = {"tests": doc["tests"], "summary": doc["summary"], "n_in_range": doc["n_in_range"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("CLAMP GENERATED DONE")


if __name__ == "__main__":
    main()
