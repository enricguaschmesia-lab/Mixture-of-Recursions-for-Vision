#!/usr/bin/env python
"""Chained generation on BigEarthNet (Phase 6 Step 5b, D6.11). `.venv`; gen and cls need a GPU.

    python eo/scripts/ben_chain.py gen    --generator base|b10_full|b10_lora [--n 10000] [--out-root R]
    python eo/scripts/ben_chain.py cls    --classifier b10_full|b10_lora     [--n 10000] [--out-root R]
    python eo/scripts/ben_chain.py report [--out-root R]                     (CPU)

ROWS: a fixed seeded subsample of test_a (`--n`, seed 0, eval_ben's --limit mechanism); --n 500
is the pilot. Every unit of work writes its own file and is skipped if that file exists.

gen (one generator; `base` = the extended B10, never trained on BEN; b10_full / b10_lora = each
fine-tune as its own generator):
  * per direction S1toS2 / S2toS1: T = 1 with M = 4 samples; the base also gets T = 0.5 with M = 1;
  * first-leg fidelity on the same rows: CE of the real target given the other modality, and
    given nothing; top-1 accuracy within the slot;
  * a KV-cache check on real continuations, plus its control (shift 1 must disagree), BEFORE any
    generation is kept.
  -> <R>/gen/<generator>/<direction>_<T>.{npz,json}, <R>/gen/<generator>/<direction>_tf.json

cls (one classifier), Step 5's decoding (K = 16, Rao-Blackwellized marginals, greedy set):
  * plumbing check: the chain prompt built with the REAL blocks, on test_a positions 0..127 at
    batch 128 and seed 0, must reproduce eval_ben's stored S1S2 output (--ref-root). Control:
    with another patch's S2 block it must not;
  * conditions (S1toS2 shown; S2toS1 mirrors it):
      S1, S2, S1S2                    real inputs: single, single, ceiling
      S1+S2wrong                      another patch's real S2 (fixed derangement): the floor
      S1+S2gen_base_T1_s{0..3}        the chain, 4 imputations
      S1+S2gen_base_T05               the chain at T = 0.5
      S2gen_base_T1_s0                the generated block alone
      S1+S2gen_base_T1_s0_wrong       another patch's generated block (the control)
      S1+S2gen_self_T1_s{0..3}        the classifier's own generations
  -> <R>/cls/<classifier>/<condition>.{npz,json}   (eval_ben's format)

report: f against f*, paired bootstrap CIs, per-class deltas -> <R>/report.{json,md}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.ben import labels as L  # noqa: E402

P6 = "/data/enric/runs/pretrain/phase6"
GEN_SETTINGS = {"base": [("T1", 1.0, 4), ("T05", 0.5, 1)], "self": [("T1", 1.0, 4)]}
CLS_K, PLUMB_N = 16, 128          # Step 5 decoding; the plumbing check mirrors eval_ben's first batch of 128


def commit():
    return subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def model_spec(a, name):
    return {"base": ("eo_ben/b10_full", None),
            "b10_full": ("eo_ben/b10_full", f"{a.runs_root}/b10full_20261012_214747/checkpoint-1461"),
            "b10_lora": ("eo_ben/b10_lora", f"{a.runs_root}/b10lora_20261012_221451/checkpoint-1461")}[name]


def subsample(n_role, n, seed=0):
    return np.sort(np.random.default_rng(seed).permutation(n_role)[:n])


def setup(a):
    import torch
    from eo.data.ben_dataset import BenTokenDataset, load_index
    gpu = torch.cuda.get_device_name(0)
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V: it stalls within seconds under load (worklog 2026-10-12)")
    ds = BenTokenDataset(role="test_a", condition="S1S2", index=load_index())
    pos = subsample(len(ds), a.n)
    return ds, pos, gpu


def load_model(a, name):
    from eo.finetune.init import build_finetuned
    cfg, ck = model_spec(a, name)
    model, _ = build_finetuned(cfg, ck, device="cuda", overrides=[f"finetune.init_from={a.init_from}"])
    return model.eval()


# ------------------------------------------------------------------ gen
def run_gen(a):
    import torch
    from eo.eval import ben as E
    from eo.eval import ben_chain as C
    ds, pos, gpu = setup(a)
    ac = {"none": None, "fp16": torch.float16}[a.autocast]
    rows = ds.rows[pos]
    out = a.out_root / "gen" / a.generator
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(a, a.generator)
    settings = GEN_SETTINGS["base" if a.generator == "base" else "self"]
    for direction, (given, target) in C.DIRECTIONS.items():
        truth = torch.from_numpy(np.stack([np.asarray(ds._arrays()[target][int(r)], dtype=np.int64) for r in rows]))
        # cache check and its control, on real continuations, before anything is kept
        probe = torch.stack([C.gen_prompt(ds, int(r), given, target) for r in rows[:4]]).cuda()
        cc = C.gen_cache_check(model, probe, truth[:4], target, autocast=ac)
        cc_ctl = C.gen_cache_check(model, probe, truth[:4], target, shift=1, autocast=ac)
        if cc > E.CACHE_TOL[a.autocast] or cc_ctl <= 10 * E.CACHE_TOL[a.autocast]:
            sys.exit(f"{direction}: cache check {cc:.3g} (tol {E.CACHE_TOL[a.autocast]}), control {cc_ctl:.3g}")
        # first-leg fidelity
        tf_file = out / f"{direction}_tf.json"
        if not tf_file.exists():
            t0, acc = time.time(), {"ce_ctx": [], "acc_ctx": [], "ce_none": [], "acc_none": []}
            for b0 in range(0, len(rows), 32):
                br, tb = rows[b0:b0 + 32], truth[b0:b0 + 32]
                pr = torch.stack([C.gen_prompt(ds, int(r), given, target) for r in br]).cuda()
                r1 = C.teacher_forced(model, pr, tb, target, autocast=ac)
                r0 = C.teacher_forced(model, torch.full((len(br), 1), C.slot(target)[2]).cuda(), tb, target, autocast=ac)
                for k, v in (("ce_ctx", r1["ce"]), ("acc_ctx", r1["acc"]), ("ce_none", r0["ce"]), ("acc_none", r0["acc"])):
                    acc[k].append(v)
            tf = {k: float(np.concatenate(v).mean()) for k, v in acc.items()}
            tf.update({"n": int(len(rows)), "seconds": round(time.time() - t0, 1), "generator": a.generator,
                       "direction": direction, "cache_check": cc, "cache_check_control_shift1": cc_ctl})
            tf_file.write_text(json.dumps(tf, indent=2) + "\n")
            print(f"{a.generator} {direction} TF: CE ctx {tf['ce_ctx']:.4f} none {tf['ce_none']:.4f}  "
                  f"acc ctx {tf['acc_ctx']:.4f} none {tf['acc_none']:.4f}", flush=True)
        for tname, temp, m in settings:
            npz = out / f"{direction}_{tname}.npz"
            if npz.exists():
                print(f"[skip] {npz}")
                continue
            t0, toks, mass = time.time(), [], []
            salt = {"S1toS2": 1, "S2toS1": 2}[direction] * 7919 + int(temp * 100)
            for b0 in range(0, len(rows), a.gen_batch):
                pr = torch.stack([C.gen_prompt(ds, int(r), given, target) for r in rows[b0:b0 + a.gen_batch]]).cuda()
                r = C.generate_blocks(model, pr, target, n_samples=m, temperature=temp,
                                      seed=a.seed * 1_000_003 + salt * 100_003 + b0, autocast=ac)
                toks.append(r["tokens"]); mass.append(r["slot_mass"])
                if (b0 // a.gen_batch) % 10 == 0:
                    print(f"  gen {a.generator} {direction} {tname}: {b0 + len(pr)}/{len(rows)} "
                          f"{(b0 + len(pr)) / (time.time() - t0):.1f} patches/s", flush=True)
            tok, ms = np.concatenate(toks), np.concatenate(mass)
            np.savez_compressed(npz, pos=pos, rows=rows, tokens=tok, slot_mass=ms)
            t = truth.numpy()
            doc = {"generator": a.generator, "direction": direction, "temperature": temp, "samples": m,
                   "n": int(len(rows)), "seconds": round(time.time() - t0, 1),
                   "patches_per_s": round(len(rows) / (time.time() - t0), 2), "mean_slot_mass": float(ms.mean()),
                   "token_match_sample0": float((tok[:, 0] == t).mean()),
                   "codes_used_sample0": int(len(np.unique(tok[:, 0]))), "codes_used_truth": int(len(np.unique(t))),
                   "cache_check": cc, "cache_check_control_shift1": cc_ctl, "gpu": gpu, "autocast": a.autocast,
                   "fork_commit": commit()}
            (out / f"{direction}_{tname}.json").write_text(json.dumps(doc, indent=2) + "\n")
            print(f"{a.generator} {direction} {tname}: {doc['seconds']} s, slot mass {doc['mean_slot_mass']:.4f}, "
                  f"token match {doc['token_match_sample0']:.4f}, codes {doc['codes_used_sample0']} vs truth "
                  f"{doc['codes_used_truth']}", flush=True)


# ------------------------------------------------------------------ cls
def conditions(ds, rows, gens):
    """name -> function(i, row) -> prompt. `gens[(gen, direction, T)]` = tokens (N, M, 196)."""
    from eo.eval import ben as E
    from eo.eval import ben_chain as C
    der = rows[E.derangement(len(rows), 1)]       # eval_ben's wrong-input convention (seed + 1)
    real = lambda s, r: ds.chunk(s, int(r))
    conds = {"S1": lambda i, r: C.cls_prompt({"S1": real("S1", r)}),
             "S2": lambda i, r: C.cls_prompt({"S2": real("S2", r)}),
             "S1S2": lambda i, r: C.cls_prompt({"S1": real("S1", r), "S2": real("S2", r)})}
    for direction, (given, target) in C.DIRECTIONS.items():
        conds[f"{given}+{target}wrong"] = (lambda g, t: lambda i, r: C.cls_prompt(
            {g: real(g, r), t: real(t, der[i])}))(given, target)
        for (gen, d, tname), tok in gens.items():
            if d != direction:
                continue
            for s in range(tok.shape[1]):
                conds[f"{given}+{target}gen_{gen}_{tname}_s{s}"] = (lambda g, t, tk, s: lambda i, r: C.cls_prompt(
                    {g: real(g, r), t: C.block(t, tk[i, s])}))(given, target, tok, s)
            if gen == "base" and tname == "T1":
                conds[f"{target}gen_base_T1_s0"] = (lambda t, tk: lambda i, r: C.cls_prompt(
                    {t: C.block(t, tk[i, 0])}))(target, tok)
                inv = E.derangement(len(rows), 1)
                conds[f"{given}+{target}gen_base_T1_s0_wrong"] = (lambda g, t, tk: lambda i, r: C.cls_prompt(
                    {g: real(g, r), t: C.block(t, tk[inv[i], 0])}))(given, target, tok)
    return conds


def run_cls(a):
    import torch
    from eo.eval import ben as E
    from eo.eval import ben_chain as C
    ds, pos, gpu = setup(a)
    ac = {"none": None, "fp16": torch.float16}[a.autocast]
    rows = ds.rows[pos]
    out = a.out_root / "cls" / a.classifier
    out.mkdir(parents=True, exist_ok=True)
    gens = {}
    for gen, tname in [("base", "T1"), ("base", "T05"), ("self", "T1")]:
        src = a.out_root / "gen" / (a.classifier if gen == "self" else gen)
        for direction in C.DIRECTIONS:
            z = np.load(src / f"{direction}_{tname}.npz")
            assert (z["rows"] == rows).all(), f"{src}: generated for other rows"
            gens[(gen, direction, tname)] = z["tokens"]
    model = load_model(a, a.classifier)

    # plumbing check: real blocks through the chain builder == eval_ben's stored S1S2, bit for bit
    pl_file = out / "plumbing.json"
    PLUMB = 8 if a.local_smoke else PLUMB_N          # eval_ben's first batch of 128 needs an A100's memory
    if not pl_file.exists():
        ref = np.load(a.ref_root / a.classifier / "checkpoint-1461" / "test_a" / "S1S2.npz")
        rr = ds.rows[:PLUMB]
        assert (ref["rows"][:PLUMB] == rr).all()
        p_real = torch.stack([C.cls_prompt({"S1": ds.chunk("S1", int(r)), "S2": ds.chunk("S2", int(r))}) for r in rr])
        p_ds = torch.stack([ds.prompt(i, "S1S2") for i in range(PLUMB)])
        p_ctl = torch.stack([C.cls_prompt({"S1": ds.chunk("S1", int(r)), "S2": ds.chunk("S2", int(o))})
                             for r, o in zip(rr, np.roll(rr, 1))])
        res = E.decode_labels(model, p_real.cuda(), k_samples=CLS_K, greedy=True, seed=0, autocast=ac)
        ctl = E.decode_labels(model, p_ctl.cuda(), k_samples=CLS_K, greedy=True, seed=0, autocast=ac)
        pl = {"prompts_equal_dataset": bool(torch.equal(p_real, p_ds)),
              "marginal_max_abs_diff": float(np.abs(res["marginal"] - ref["marginal"][:PLUMB]).max()),
              "greedy_equal": bool((res["greedy"] == ref["greedy"][:PLUMB]).all()),
              "control_marginal_max_abs_diff": float(np.abs(ctl["marginal"] - ref["marginal"][:PLUMB]).max()),
              "n": PLUMB, "batch": PLUMB, "k": CLS_K}
        pl_file.write_text(json.dumps(pl, indent=2) + "\n")
        print("plumbing:", pl, flush=True)
        if not (pl["prompts_equal_dataset"] and pl["greedy_equal"] and pl["marginal_max_abs_diff"] < 1e-4
                and pl["control_marginal_max_abs_diff"] > 0.05):
            if not a.local_smoke:
                sys.exit("plumbing check failed")
            print("WARNING: plumbing check failed; --local-smoke (another card than the reference's A100)")
            pl_file.unlink()

    truth = np.stack([np.isin(np.arange(L.N_CLASSES), ds.classes(int(r))) for r in rows])
    for name, fn in conditions(ds, rows, gens).items():
        npz = out / f"{name}.npz"
        if npz.exists():
            print(f"[skip] {npz}")
            continue
        t0, acc = time.time(), {k: [] for k in ("greedy", "marginal", "sample_freq", "slot_mass")}
        for b0 in range(0, len(rows), a.cls_batch):
            pr = torch.stack([fn(i, rows[i]) for i in range(b0, min(b0 + a.cls_batch, len(rows)))]).cuda()
            r = E.decode_labels(model, pr, k_samples=CLS_K, greedy=True, seed=a.seed * 1_000_003 + b0, autocast=ac)
            for k in acc:
                acc[k].append(r[k])
        res = {k: np.concatenate(v) for k, v in acc.items()}
        np.savez_compressed(npz, rows=rows, truth=truth, **res)
        m = {"condition": name, "n": int(len(rows)), "k": CLS_K, "classifier": a.classifier,
             **{f"greedy_{k}": v for k, v in E.f1_scores(truth, res["greedy"]).items() if k != "per_class_f1"},
             **{k: v for k, v in E.map_scores(truth, res["marginal"]).items() if k != "per_class_ap"},
             "mean_slot_mass": float(res["slot_mass"].mean()), "seconds": round(time.time() - t0, 1),
             "gpu": gpu, "autocast": a.autocast, "fork_commit": commit()}
        (out / f"{name}.json").write_text(json.dumps(m, indent=2) + "\n")
        print(f"{a.classifier} {name}: micro-mAP {m['micro_map']:.4f}  micro-F1 {m['greedy_micro_f1']:.4f}  "
              f"({m['seconds']} s)", flush=True)


# ------------------------------------------------------------------ report
def run_report(a):
    from eo.eval import ben as E
    from eo.eval import ben_chain as C
    R = a.out_root
    mm = lambda y, s: E.average_precision(y.ravel(), s.ravel())

    def boot(y, s1, s0, n=200, seed=0):
        rng = np.random.default_rng(seed)
        d = [mm(y[i], s1[i]) - mm(y[i], s0[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
        return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    doc, md = {}, ["# Step 5b: chained generation on BEN (test_a subsample)", ""]
    for cls_dir in sorted((R / "cls").iterdir()):
        c = cls_dir.name
        ld = lambda n: np.load(cls_dir / f"{n}.npz")
        y = ld("S1S2")["truth"].astype(bool)
        sc = {p.stem: ld(p.stem)["marginal"].astype(np.float64) for p in cls_dir.glob("*.npz")}
        doc[c] = {}
        md += [f"## classifier {c} (n = {len(y)})", "",
               "| direction | generator | single | floor | ceiling | chain s0 | chain mean of 4 [single-sample range] | "
               "f (mean 4) | f* | chain − single [95% CI] | T = 0.5 | generated alone (real alone) | control: other patch's Ŝ |",
               "|" + "---|" * 13]
        for direction, (g, t) in C.DIRECTIONS.items():
            m_single, m_ceil, m_floor = mm(y, sc[g]), mm(y, sc["S1S2"]), mm(y, sc[f"{g}+{t}wrong"])
            for gen in ("base", "self"):
                keys = [k for k in sc if k.startswith(f"{g}+{t}gen_{gen}_T1_s") and not k.endswith("wrong")]
                if not keys:
                    continue
                singles = [mm(y, sc[k]) for k in sorted(keys)]
                mean4 = np.mean([sc[k] for k in sorted(keys)], 0)
                m4 = mm(y, mean4)
                f = (m4 - m_floor) / (m_ceil - m_floor)
                fs = (m_single - m_floor) / (m_ceil - m_floor)
                ci = boot(y, mean4, sc[g])
                t05 = mm(y, sc[f"{g}+{t}gen_base_T05_s0"]) if gen == "base" else None
                alone = mm(y, sc[f"{t}gen_base_T1_s0"]) if gen == "base" else None
                wrong = mm(y, sc[f"{g}+{t}gen_base_T1_s0_wrong"]) if gen == "base" else None
                per_class = [E.average_precision(y[:, k], mean4[:, k]) - E.average_precision(y[:, k], sc[g][:, k])
                             for k in range(L.N_CLASSES)]
                r = {"single": m_single, "floor": m_floor, "ceiling": m_ceil, "chain_s0": singles[0],
                     "chain_singles": singles, "chain_mean4": m4, "f": f, "f_star": fs, "chain_minus_single_ci95": ci,
                     "chain_minus_single": m4 - m_single, "T05": t05, "generated_alone": alone,
                     "real_alone": mm(y, sc[t]), "control_other_patch": wrong,
                     "per_class_ap_delta": dict(zip(L.SHORT, per_class))}
                doc[c][f"{direction}/{gen}"] = r
                fmt = lambda v: "" if v is None else f"{v:.4f}"
                md.append(f"| {direction} | {gen} | {m_single:.4f} | {m_floor:.4f} | {m_ceil:.4f} | {singles[0]:.4f} | "
                          f"{m4:.4f} [{min(singles):.4f}, {max(singles):.4f}] | {f:.3f} | {fs:.3f} | "
                          f"{m4 - m_single:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {fmt(t05)} | "
                          f"{fmt(alone)} ({mm(y, sc[t]):.4f}) | {fmt(wrong)} |")
        md.append("")
    tf_rows = []
    for gdir in sorted((R / "gen").iterdir()):
        for f in sorted(gdir.glob("*_tf.json")):
            d = json.loads(f.read_text())
            g = json.loads((gdir / f.name.replace("_tf", "_T1")).read_text()) if (gdir / f.name.replace("_tf", "_T1")).exists() else {}
            tf_rows.append(f"| {gdir.name} | {d['direction']} | {d['ce_ctx']:.4f} | {d['ce_none']:.4f} | "
                           f"{d['ce_none'] - d['ce_ctx']:+.4f} | {d['acc_ctx']:.4f} | {d['acc_none']:.4f} | "
                           f"{g.get('token_match_sample0', float('nan')):.4f} | {g.get('codes_used_sample0', '')} / "
                           f"{g.get('codes_used_truth', '')} | {g.get('mean_slot_mass', float('nan')):.4f} |")
            doc.setdefault("first_leg", {})[f"{gdir.name}/{d['direction']}"] = {**d, **{k: g.get(k) for k in (
                "token_match_sample0", "codes_used_sample0", "codes_used_truth", "mean_slot_mass")}}
    md += ["## First leg (teacher-forced, the same rows)", "",
           "| generator | direction | CE given the other modality | CE given nothing | gain | top-1 acc (ctx) | top-1 acc (none) | "
           "generated = real token (T=1, s0) | codes used generated / real | slot mass |", "|" + "---|" * 10] + tf_rows
    (R / "report.json").write_text(json.dumps(doc, indent=1) + "\n")
    (R / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "cls", "report"])
    ap.add_argument("--generator", choices=["base", "b10_full", "b10_lora"])
    ap.add_argument("--classifier", choices=["b10_full", "b10_lora"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--cls-batch", type=int, default=128)
    ap.add_argument("--autocast", choices=["none", "fp16"], default="fp16")
    ap.add_argument("--out-root", type=Path, default=Path("/data/enric/reports/ben/chain"))
    ap.add_argument("--ref-root", type=Path, default=Path("/data/enric/reports/ben"),
                    help="eval_ben outputs (<ref>/<classifier>/checkpoint-1461/test_a/S1S2.npz) for the plumbing check")
    ap.add_argument("--runs-root", default=P6)
    ap.add_argument("--init-from", default=f"{P6}/base/b10_ext_ckpt3300")
    ap.add_argument("--local-smoke", action="store_true",
                    help="a smoke on another card: the plumbing check cannot be bit-identical, so it warns. Never for a real run")
    a = ap.parse_args()
    if a.stage == "gen":
        assert a.generator, "--generator"
        run_gen(a)
    elif a.stage == "cls":
        assert a.classifier, "--classifier"
        run_cls(a)
    else:
        run_report(a)


if __name__ == "__main__":
    main()
