#!/usr/bin/env python
"""Phase 3 Step 12.P, pre-test P1: does teacher-forced CE agree with decoded generation?

Step 12 reports one decoder-free number per cell, teacher-forced CE of the true
target tokens (PHASE3_PLAN.md section 4.18). P1 checks, on the two 50-epoch arms
where both quantities exist at all 9 checkpoints, that CE is not contradicted by
decoded generation quality (D3.11's rmse_z_ml7, or mIoU for LULC).

THE RULE, as amended 2026-10-04 before this was run (PHASE3_PLAN.md 12.P):

  P1a  across checkpoints, gated on reliability. Per target, split the 64 D3.11
       scenes into two fixed corpus-stratified halves (seed 0) and correlate the
       halves' per-point means over the 18 (arm, checkpoint) points, Spearman-Brown
       corrected. Only if that reliability is >= 0.7 does the original test run:
       Spearman rho(CE, decoded mean) over the 18 points with the expected sign,
       |rho| >= 0.7, above the 95th percentile of 1,000 re-pairings -> agree, else
       disagree. Below 0.7 P1a is uninformative for that target.
  P1b  per scene, ACROSS the 18 points: for each scene, Spearman between its CE and
       its generated error over the 18 (arm, checkpoint) points; averaged over
       scenes. 95% CI by bootstrap over scenes (10,000 resamples). Control: each
       scene's CE permuted across points; its CI must include 0. Expected sign and
       CI excluding 0 -> agree; wrong sign and CI excluding 0 -> disagree; else
       uninformative.
       ⚠ Corrected before any P1 number existed. The amendment first specified a
       CROSS-SECTIONAL Spearman partial correlation (per-scene CE vs generated
       error, controlling for the ceiling error, within each point). Validated on
       synthetic data it is BIASED: under pure confounding (x, y = z + noise) it
       reads +0.048 at corr(x,z) ~ 0.96 and +0.021 at ~0.7 instead of 0, and
       normal scores, within-ceiling-bin permutation nulls (9/40 false agree) and
       cubic rank residualization do not remove it. The ceiling is identical at
       every point (same truth, same decode seed: max spread 0.0), so correlating
       within each scene across points removes scene difficulty by construction:
       synthetic null 0.000 (false agree 5.5%, false disagree 3.5% at 95%), weak
       shared effect agree 39/40, opposite effect disagree 40/40. The
       cross-sectional partial is still computed and reported, never decides.
  Verdict per target: P1a's if reliable, else P1b's. CE stands unless >= 3 of the
  6 targets DISAGREE.

Expected sign: positive for the five continuous targets (CE and rmse_z_ml7 are both
lower-is-better), negative for LULC (mIoU is higher-is-better).

Two stages, two environments of work:
  scene-ce   GPU (.venv). Per-row target-last CE on the D3.11 scenes, for all 18
             points, through eval/teacher_forced's own TargetLast + run_pass, so
             the number is the one target_last[T].ce aggregates. First a CONTROL:
             one point over the full 512 target-last rows must reproduce the stored
             aggregate, and the per-row reduction must reproduce run_pass's own.
  analyze    CPU. P1a, P1b, the controls and the verdict -> p1_result.json.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(python -m eo.train.gpu titanx) \
        python eo/experiments/p1_ce_vs_decoded.py scene-ce
    python eo/experiments/p1_ce_vs_decoded.py analyze

Outputs under /data/enric/reports/p1/. Runs in .venv; never imports terratorch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")
RUNS = Path("/data/enric/runs/pretrain/phase3")
ARM_EVAL = Path("/data/enric/reports/arm_eval")
D311 = Path("/data/enric/reports/d311")
OUT = Path("/data/enric/reports/p1")

ARMS = {"arm_a": ("mor_20260923_140238", "eo_terramesh/arm_a_mor"),
        "arm_b": ("vanilla_20260928_001221", "eo_terramesh/arm_b_vanilla")}
CKPTS = [f"checkpoint-{s}" for s in (2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 16500)]
TARGETS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
METRIC = {T: ("miou" if T == "LULC" else "rmse_z_ml7") for T in TARGETS}
SIGN = {T: (-1 if T == "LULC" else 1) for T in TARGETS}

RELIABLE = 0.7          # Spearman-Brown corrected split-half reliability
RHO_MIN = 0.7           # P1a, as originally specified
N_PERM = 1000           # P1a null
N_BOOT = 10_000         # P1b CI
DISAGREE_LIMIT = 3      # CE stands unless >= this many targets disagree


def d311_file(arm, ck, T):
    return D311 / arm / ck / f"{T}_{arm}_{ck}" / f"metrics_{T}.json"


def d311_rows(T):
    """The 64 D3.11 scenes of target T: identical for every point (checked)."""
    lists = {tuple(s["row"] for s in json.loads(d311_file(a, c, T).read_text())["per_scene"]["generated"])
             for a in ARMS for c in CKPTS}
    assert len(lists) == 1, f"{T}: D3.11 scene lists differ across points"
    return list(next(iter(lists)))


# ------------------------------------------------------------------ scene-ce

def per_row_ce(model, ds, rows, T, num_workers=4):
    """(per-row mean target CE, run_pass's float aggregate, token-weighted per-row mean)."""
    import torch
    from eo.eval import teacher_forced as TF
    dl = torch.utils.data.DataLoader(TF.TargetLast(ds, rows, T), batch_size=4,
                                     num_workers=num_workers)
    res = TF.run_pass(model, dl, keep_tokens=True)
    tok = res["tokens"]
    sel = tok["label_mod"] == TF.MODALITY_TO_ID[T]
    ce = tok["ce"].astype(np.float64)
    n = sel.sum(1)
    assert n.min() > 0, f"{T}: a row has no target tokens"
    row_ce = (ce * sel).sum(1) / n
    return row_ce, res["per_modality"][T]["ce"], float((row_ce * n).sum() / n.sum()), n


def stage_scene_ce(args):
    import torch
    from eo.data.eval_split import load_eval_rows, load_row_table
    from eo.eval import teacher_forced as TF
    from eo.generate import conditional as G

    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=...")
    gpu = torch.cuda.get_device_name(0)
    print(f"scene-ce on {gpu}")
    rows = {T: d311_rows(T) for T in TARGETS}
    (OUT / "scene_ce").mkdir(parents=True, exist_ok=True)

    # CONTROL first: the full 512-row target-last pass at one point must reproduce
    # the stored aggregate (same card the stored arm-B pass used), and the per-row
    # reduction must reproduce run_pass's own aggregate.
    ctl_path = OUT / "control_reproduce.json"
    if not ctl_path.exists():
        arm, ck = "arm_b", "checkpoint-2000"
        run, cfg_name = ARMS[arm]
        torch.manual_seed(42)
        model, cfg = G.build_model(cfg_name, str(RUNS / run / ck))
        ds, _ = TF.eval_loader(cfg, num_workers=4)
        corpus = load_row_table(ROOT).corpus.values
        eval_rows = load_eval_rows(root_dir=ROOT)
        stored = json.loads((ARM_EVAL / arm / ck / "metrics.json").read_text())["target_last"]
        ctl = {"arm": arm, "checkpoint": ck, "gpu": gpu, "stored_gpu":
               json.loads((ARM_EVAL / arm / ck / "metrics.json").read_text())["gpu"], "targets": {}}
        ok = True
        for T in TARGETS:
            r512 = TF.stratified_target_rows(ds, eval_rows, corpus, T, 512)
            assert set(rows[T]) <= set(r512), f"{T}: D3.11 scenes are not inside the target-last rows"
            _, agg, recon, _ = per_row_ce(model, ds, r512, T)
            d_store, d_recon = abs(agg - stored[T]["ce"]), abs(recon - agg)
            ok &= d_store < 2e-3 and d_recon < 2e-3
            ctl["targets"][T] = {"n_rows": len(r512), "stored": stored[T]["ce"], "recomputed": agg,
                                 "per_row_reduction": recon, "abs_diff_stored": d_store,
                                 "abs_diff_reduction": d_recon}
            print(f"  control {T}: stored {stored[T]['ce']:.5f}  recomputed {agg:.5f}  "
                  f"per-row {recon:.5f}  (|d| {d_store:.1e}, {d_recon:.1e})")
        ctl["pass"] = bool(ok)
        ctl_path.write_text(json.dumps(ctl, indent=2) + "\n")
        del model
        torch.cuda.empty_cache()
        if not ok:
            raise SystemExit("CONTROL FAILED: the per-row pass does not reproduce the stored target-last CE")
    elif not json.loads(ctl_path.read_text())["pass"]:
        raise SystemExit(f"{ctl_path} records a failed control; delete it to re-run")

    for arm, (run, cfg_name) in ARMS.items():
        for ck in CKPTS:
            out = OUT / "scene_ce" / f"{arm}_{ck}.npz"
            if out.exists():
                print(f"[skip] {arm} {ck}")
                continue
            torch.manual_seed(42)
            model, cfg = G.build_model(cfg_name, str(RUNS / run / ck))
            ds, _ = TF.eval_loader(cfg, num_workers=4)
            save = {}
            for T in TARGETS:
                row_ce, agg, _, n = per_row_ce(model, ds, rows[T], T)
                save[f"{T}_rows"] = np.asarray(rows[T])
                save[f"{T}_ce"] = row_ce
                save[f"{T}_ntok"] = n
            np.savez(out, **save)
            print(f"{arm} {ck}: " + "  ".join(f"{T} {save[f'{T}_ce'].mean():.3f}" for T in TARGETS))
            del model
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ analyze

def avg_rank(a):
    """Ranks along the last axis with TIES AVERAGED (worklog 2026-09-27/28: ranks that
    break ties by position manufacture correlation). Bootstrap resamples tie by design."""
    a = np.asarray(a, dtype=np.float64)
    shape, n = a.shape, a.shape[-1]
    a2 = a.reshape(-1, n)
    lead = a2.shape[0]
    order = np.argsort(a2, axis=1, kind="mergesort")
    srt = np.take_along_axis(a2, order, 1)
    grp = np.concatenate([np.zeros((lead, 1), int), np.cumsum(np.diff(srt, axis=1) != 0, 1)], 1)
    gid = grp + (np.arange(lead) * n)[:, None]
    pos = np.broadcast_to(np.arange(n, dtype=np.float64), (lead, n))
    sums = np.bincount(gid.ravel(), weights=pos.ravel(), minlength=lead * n)
    cnts = np.bincount(gid.ravel(), minlength=lead * n)
    avg = (sums / np.maximum(cnts, 1))[gid]
    ranks = np.empty_like(avg)
    np.put_along_axis(ranks, order, avg, 1)
    return (ranks + 1).reshape(shape)


def pearson(x, y):
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return np.where(den > 0, (x * y).sum(-1) / np.where(den > 0, den, 1), 0.0)


def spearman(x, y):
    return pearson(avg_rank(x), avg_rank(y))


def partial_spearman(x, y, z):
    rx, ry, rz = avg_rank(x), avg_rank(y), avg_rank(z)
    rxy, rxz, ryz = pearson(rx, ry), pearson(rx, rz), pearson(ry, rz)
    den = np.sqrt(np.clip((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12, None))
    return (rxy - rxz * ryz) / den


def stage_analyze(args):
    from eo.data.eval_split import load_row_table
    corpus = load_row_table(ROOT).corpus.values
    points = [(a, c) for a in ARMS for c in CKPTS]
    result = {"rule": {"reliable": RELIABLE, "rho_min": RHO_MIN, "n_perm": N_PERM, "n_boot": N_BOOT,
                       "disagree_limit": DISAGREE_LIMIT, "seed": 0},
              "control_reproduce": json.loads((OUT / "control_reproduce.json").read_text()),
              "targets": {}}
    rng = np.random.default_rng(0)
    for T in TARGETS:
        rows = np.asarray(d311_rows(T))
        gen = np.zeros((len(points), len(rows)))
        ceil = np.zeros_like(gen)
        ce = np.zeros_like(gen)
        tl_ce = np.zeros(len(points))
        dec = np.zeros(len(points))
        in_range = None
        for i, (a, c) in enumerate(points):
            m = json.loads(d311_file(a, c, T).read_text())
            g, cl = m["per_scene"]["generated"], m["per_scene"]["ceiling"]
            assert [s["row"] for s in g] == list(rows) == [s["row"] for s in cl]
            ir = np.array([s["in_range"] for s in g])
            in_range = ir if in_range is None else in_range
            assert (ir == in_range).all(), f"{T}: in-range set differs across points"
            gen[i] = [s[METRIC[T]] if s[METRIC[T]] is not None else np.nan for s in g]
            ceil[i] = [s[METRIC[T]] if s[METRIC[T]] is not None else np.nan for s in cl]
            dec[i] = m["summary"]["generated"][METRIC[T]]
            tl_ce[i] = json.loads((ARM_EVAL / a / c / "metrics.json").read_text())["target_last"][T]["ce"]
            z = np.load(OUT / "scene_ce" / f"{a}_{c}.npz")
            assert (z[f"{T}_rows"] == rows).all()
            ce[i] = z[f"{T}_ce"]
        keep = in_range & np.isfinite(gen).all(0) & np.isfinite(ceil).all(0)
        rows_k, gen, ceil, ce = rows[keep], gen[:, keep], ceil[:, keep], ce[:, keep]
        n = len(rows_k)
        # the summary is the in-range mean; check we read the same scenes
        assert np.allclose(gen.mean(1), dec, atol=5e-4), f"{T}: per-scene mean != summary"

        # ---- P1a: split-half reliability of the decoded reference, then the original test
        half = np.zeros(n, bool)
        for cname in ("majortom", "ssl4eos12"):
            idx = np.flatnonzero(corpus[rows_k] == cname)
            idx = np.random.default_rng(0).permutation(idx)
            half[idx[::2]] = True
        r_half = float(spearman(gen[:, half].mean(1), gen[:, ~half].mean(1)))
        rel = 2 * r_half / (1 + r_half) if r_half > -1 else -1.0
        rho = float(spearman(tl_ce, dec)) * SIGN[T]
        null = np.array([spearman(tl_ce, rng.permutation(dec)) * SIGN[T] for _ in range(N_PERM)])
        p95 = float(np.quantile(null, 0.95))
        reliable = rel >= RELIABLE
        p1a = ("agree" if (rho >= RHO_MIN and rho > p95) else "disagree") if reliable else "uninformative"

        # ---- P1b: within each scene, across the 18 points; bootstrap over scenes + control
        per_scene = spearman(ce.T, gen.T) * SIGN[T]          # (n,): one rho per scene
        stat = float(per_scene.mean())
        boot_idx = np.random.default_rng(1).integers(0, n, size=(N_BOOT, n))
        bs = per_scene[boot_idx].mean(1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        sh_rng = np.random.default_rng(2)
        ce_sh = np.stack([col[sh_rng.permutation(len(points))] for col in ce.T], 1)
        per_scene_sh = spearman(ce_sh.T, gen.T) * SIGN[T]
        sh_stat = float(per_scene_sh.mean())
        sh_lo, sh_hi = np.quantile(per_scene_sh[boot_idx].mean(1), [0.025, 0.975])
        control_ok = bool(sh_lo <= 0 <= sh_hi)
        if lo > 0:
            p1b = "agree"
        elif hi < 0:
            p1b = "disagree"
        else:
            p1b = "uninformative"
        # Reported, never deciding: the cross-sectional readouts (biased under strong
        # confounding by scene difficulty; see the docstring).
        xs_partial = float(partial_spearman(ce, gen, ceil).mean()) * SIGN[T]
        plain = float(spearman(ce, gen).mean()) * SIGN[T]
        ceil_only = float(spearman(ce, ceil).mean()) * SIGN[T]

        verdict = p1a if reliable else p1b
        result["targets"][T] = {
            "metric": METRIC[T], "expected_sign": SIGN[T], "n_scenes_in_range": n,
            "p1a": {"split_half_r": r_half, "reliability_sb": rel, "reliable": bool(reliable),
                    "rho_signed": rho, "null_p95": p95, "verdict": p1a},
            "p1b": {"within_scene_rho_signed": stat, "ci95": [float(lo), float(hi)], "verdict": p1b,
                    "frac_scenes_positive": float((per_scene > 0).mean()),
                    "shuffle_control": {"stat": sh_stat, "ci95": [float(sh_lo), float(sh_hi)],
                                        "includes_0": control_ok},
                    "reported_only": {"cross_sectional_partial_signed": xs_partial,
                                      "cross_sectional_plain_signed": plain,
                                      "ce_vs_ceiling_signed": ceil_only}},
            "verdict": verdict,
        }
        print(f"{T:6s} n={n:2d}  P1a rel {rel:+.2f} ({'reliable' if reliable else 'unreliable'})  "
              f"rho {rho:+.2f} p95 {p95:+.2f} -> {p1a:13s} | P1b within-scene {stat:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] -> {p1b:13s} (shuffle {sh_stat:+.3f} [{sh_lo:+.3f}, {sh_hi:+.3f}]) "
              f"=> {verdict}")

    v = [t["verdict"] for t in result["targets"].values()]
    controls = all(t["p1b"]["shuffle_control"]["includes_0"] for t in result["targets"].values())
    result["n_disagree"] = v.count("disagree")
    result["n_agree"] = v.count("agree")
    result["shuffle_controls_ok"] = controls
    result["ce_stands"] = v.count("disagree") < DISAGREE_LIMIT
    (OUT / "p1_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nagree {v.count('agree')}  disagree {v.count('disagree')}  uninformative "
          f"{v.count('uninformative')}  | shuffle controls include 0: {controls}")
    print("P1:", "CE STANDS" if result["ce_stands"] else "FALLBACK (>= 3 targets disagree)")
    if not controls:
        print("⚠ a shuffle control excluded 0: the P1b statistic is biased; do not read its verdicts")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["scene-ce", "analyze"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"scene-ce": stage_scene_ce, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
