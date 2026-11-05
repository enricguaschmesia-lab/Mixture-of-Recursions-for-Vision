#!/usr/bin/env python
"""T360 on four-date locations, before anything temporal is built (A1, Step 0). `.venv`, GPU, fp32.

T360 saw every date of these locations' kind as an independent scene, never two dates in one
sequence, and it never saw a val row. This asks three things of it, in teacher-forced CE on the
dev locations of eo/data/temporal_rows_v1.json:

  CONTEXT       What is another date worth? CE of S2L2A, S1GRD and NDVI at date t, with no date
                tokens, given: nothing; the statics; the other modalities at t; the same modality
                at one other date; the same modality at all other dates; all other dates; all
                other dates plus the other modalities at t. This is "T360 without time".

  LONG OFFSETS  How does T360 attend at offsets it never trained on? One source block before the
                target, the target's position_ids moved to 198 k for k = 1..14. T360's longest
                sequence held 995 tokens; the temporal sequence holds 2,793.

  PATCH OFFSET  Does a date block cost alignment? T360 saw a patch's counterpart at 198 n, or
                198 n + 5 with Coords between. A date block adds 4 (3 without its year) for each
                group crossed, so cross-date pairs would sit at 198 n + d, d up to 21. One source
                block before the target, position_ids moved by d = 0..21.

  COORDS        Is 198 n + 5 an offset, or a cue? The smoke test found d = 5 no better than any other
                d, though a third of T360's block pairs had Coords between them. So the same two
                pairs with real Coords tokens: between source and target (offset 203, sequential),
                and before both (offset 198). If between holds up while d = 5 does not, T360 aligns
                through what it reads, not by offset alone.

  LENGTH        Do far keys hurt? S2L2A at t given the last k context blocks of all_other_same, the
                nearest ones kept in place, for k up to 11. The smoke test's full-length CE was above
                ln V. If CE climbs with k, attention at offsets T360 never trained on is doing
                harm, not just failing to help, and a run at full length starts from there.

The decision the sweep feeds, fixed before it runs (docs/plans/a1_temporal_stage.md, Step 0): the
worst CE rise over d = 3..21 against d = 0, as a share of the gain the source gives over no
context, on either pair. Up to SHARE_MAX: position_ids stay sequential. Above: lattice positions.

Pairs swept: S2L2A -> NDVI at one date (S2L2A is NDVI's best source) and S2L2A at another date
-> S2L2A at t (the relation A1 needs).

CONTROLS, each must hold or fire, and the verdict is refused otherwise:
  reference     the same-date layout reproduces T360's stored fp32 target-last CE, row by row
  shift_all     every position moved by one constant leaves the logits unchanged to round-off,
                because RoPE sees only differences
  delta_zero    explicit positions at d = 0 equal the model's default positions
  half_block    d = 99 must change the CE: the shift reaches attention
  wrong_source  a block from another location must lose nearly all of the source's gain: the
                gain is information, not "any context helps"

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(.venv/bin/python -m eo.train.gpu titanx) \\
      .venv/bin/python -X faulthandler -u eo/experiments/temporal/t360_context.py

One file per unit under --out/units, skipped if present, so a hang costs one unit. Runs in `.venv`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eo.data.eo_vocab import MODALITIES  # noqa: E402
from eo.data.temporal_split import load_temporal_split  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.terramesh_tok.contract import TOKENS_PER_SAMPLE  # noqa: E402

CKPT = "/data/enric/runs/pretrain/phase3/vanilla360t_20261026_040535/checkpoint-17441"
ARM = "eo_terramesh/arm_t360_vanilla"
OUT = Path("/data/enric/reports/temporal/t360_context")
REFERENCE = Path("/data/enric/reports/arm_eval/t360_fp32/checkpoint-17441/target_last_rows.npz")
ROOT = "/data/enric/data/TerraMesh/val"

TARGETS = ("S2L2A", "S1GRD", "NDVI")          # what changes between dates
N_LOC, SEED, DRAWS = 256, 20261105, 10_000
SWEEP = tuple(range(22))                      # d = 0..21: T360 trained at 0 and 5
HALF_BLOCK = 99
LONG_K = tuple(range(1, 15))                  # 198 k, up to 2,772
LENGTH_K = (1, 2, 3, 4, 5, 6, 8, 11)          # context blocks kept, nearest first
READ_FROM = 3                                 # the sweep is read over d >= 3 ...
SHARE_MAX = 0.10                              # ... and fixed here, before it runs
BLOCK = TOKENS_PER_SAMPLE + 2                 # 198: BO, 196 tokens, EO
ALL_SECTIONS = ("controls", "context", "long", "coords", "length", "sweep")


def batch_for(length: int) -> int:
    return 16 if length <= 600 else 8 if length <= 1500 else 4


def pick_locations(doc, n: int):
    """The first n of a seeded permutation of dev, in dev order: (rows (n, 4), groups, ids)."""
    dev = doc["dev"]
    idx = np.sort(np.random.default_rng(SEED).permutation(len(dev["locations"]))[:n])
    return (np.asarray(dev["rows"], dtype=np.int64)[idx], np.asarray(dev["groups"], dtype=np.int64)[idx],
            [dev["locations"][i] for i in idx])


def date_of(i: int, j: int) -> int:
    """The target date for location i, target j: balanced over dates, different per target."""
    return (i + j) % 4


def near(t: int) -> int:
    """A location's date next to t, the one used where a single other date is needed."""
    return t - 1 if t > 0 else 1


def statics(r, t: int):
    """Coords and DEM (the same at every date) and LULC from another date: LULC at t holds t's
    cloud and snow, which is the same-date information the `same_date` layout measures."""
    o = near(t)
    return [("DEM", r[o]), ("LULC", r[o]), ("Coords", r[o])]


def others_at(m: str, r, t: int):
    return [(k, r[t]) for k in TARGETS if k != m]


def date_blocks(r, t: int):
    return [(k, r[o]) for o in range(4) if o != t for k in TARGETS]


def registry_sorted(blocks):
    order = {m: i for i, m in enumerate(MODALITIES)}
    return sorted(blocks, key=lambda b: order[b[0]])


def layouts(m: str, r, t: int):
    """Context blocks per layout. Each is (name, blocks), every date in chronological order."""
    gap = [o for o in range(4) if o != t]
    return [("none", []),
           ("statics", statics(r, t)),
           ("same_date", registry_sorted([(k, r[t]) for k in ("S2L2A", "S1GRD", "DEM", "NDVI", "LULC", "Coords")
                                          if k != m])),
           ("same_mod_all", [(m, r[o]) for o in gap]),
           ("all_other", statics(r, t) + date_blocks(r, t)),
           ("all_other_same", statics(r, t) + date_blocks(r, t) + others_at(m, r, t))]


class Run:
    def __init__(self, model, ds, device, out: Path):
        self.model, self.ds, self.device, self.units = model, ds, device, out / "units"
        self.units.mkdir(parents=True, exist_ok=True)

    def seq(self, blocks, target):
        return torch.cat([self.ds._load_chunk(m, int(r)) for m, r in list(blocks) + [target]])

    @torch.no_grad()
    def target_ce(self, seqs: torch.Tensor, delta: int = 0, shift_all: int = 0, plain: bool = False):
        """Per-sequence mean CE and top-1 of the last block's body. `delta` moves the target's
        position_ids; `plain` leaves positions to the model's default."""
        x = seqs.to(self.device)
        L = x.shape[1]
        kw = {}
        if not plain:
            pos = torch.arange(L, device=self.device)
            pos[L - BLOCK:] += delta
            kw["position_ids"] = (pos + shift_all).unsqueeze(0).expand(len(x), -1)
        lg = self.model(input_ids=x, use_cache=False, logits_to_keep=BLOCK, **kw).logits[:, :TOKENS_PER_SAMPLE].float()
        lab = x[:, -(TOKENS_PER_SAMPLE + 1):-1]
        ce = F.cross_entropy(lg.transpose(1, 2), lab, reduction="none")
        n_above = (lg > lg.gather(-1, lab.unsqueeze(-1))).sum(-1)      # never topk (CLAUDE.md)
        return ce.mean(1).double().cpu().numpy(), n_above.eq(0).double().mean(1).cpu().numpy()

    def run_seqs(self, seqs, **kw):
        ces, tops = [], []
        by_len = {}
        for i, s in enumerate(seqs):
            by_len.setdefault(len(s), []).append(i)
        ce, top = np.empty(len(seqs)), np.empty(len(seqs))
        for L, idx in by_len.items():
            b = batch_for(L)
            for k in range(0, len(idx), b):
                part = idx[k:k + b]
                c, t = self.target_ce(torch.stack([seqs[i] for i in part]), **kw)
                ce[part], top[part] = c, t
        return ce, top

    def unit(self, name: str, fn):
        """(ce, top1) of one unit of work, from disk when it is already there."""
        p = self.units / f"{name}.npz"
        if p.exists():
            z = np.load(p)
            return z["ce"], z["top1"]
        t0 = time.perf_counter()
        ce, top = fn()
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, ce=ce, top1=top)
        os.replace(tmp, p)
        print(f"  {name}: CE {ce.mean():.4f}, top-1 {100 * top.mean():.2f}%, {len(ce)} rows, "
              f"{time.perf_counter() - t0:.0f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        return ce, top


def group_ci(values: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Mean and 95% interval from a bootstrap over spatial groups (nearby locations are not
    independent), DRAWS resamples."""
    u, inv = np.unique(groups, return_inverse=True)
    s, n = np.bincount(inv, values), np.bincount(inv).astype(float)
    pick = np.random.default_rng(seed).integers(0, len(u), (DRAWS, len(u)))
    b = s[pick].sum(1) / n[pick].sum(1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=CKPT)
    ap.add_argument("--arm-config", default=ARM)
    ap.add_argument("--n", type=int, default=N_LOC, help="locations (smoke tests only)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--expect-gpu", default="TITAN X")
    ap.add_argument("--sections", default=",".join(ALL_SECTIONS))
    args = ap.parse_args()
    sections = set(args.sections.split(","))

    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    gpu = torch.cuda.get_device_name(0)
    if args.expect_gpu not in gpu:
        raise SystemExit(f"CUDA sees {gpu!r}, expected {args.expect_gpu!r}")
    model, cfg = G.build_model(args.arm_config, args.checkpoint)
    if (cfg.get("multimodal") or {}).get("token_set") != "clean":
        raise SystemExit("all four dates are pre-2022: the cleaned tokens only (plan, data caveats)")
    if next(model.parameters()).dtype != torch.float32:
        raise SystemExit("fp32 parameters expected")
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    print(f"{gpu}; token root {ds.root_dir}; {args.checkpoint}", flush=True)

    doc = load_temporal_split(ROOT)
    rows, groups, ids = pick_locations(doc, args.n)
    out = Path(args.out)
    run = Run(model, ds, "cuda", out)
    N = len(rows)
    report = {"checkpoint": args.checkpoint, "arm_config": args.arm_config, "gpu": gpu, "precision": "fp32",
              "split": doc["tag"], "n_locations": N, "n_groups": int(len(np.unique(groups))), "seed": SEED,
              "bootstrap": {"draws": DRAWS, "unit": "5 km spatial group"},
              "targets": list(TARGETS), "decision_rule": {"read_from_delta": READ_FROM, "share_max": SHARE_MAX}}
    controls = {}
    ok = {}

    # ------------------------------------------------------------ controls that need no sweep
    if "controls" in sections:
        ref = np.load(REFERENCE)
        rr = ref["S1GRD|rows"][: max(args.n, 32)]
        ref_ce = (ref["S1GRD|ce_sum"] / ref["S1GRD|n"])[: len(rr)]
        ce, _ = run.unit("ctrl_reference", lambda: run.run_seqs([
            run.seq(registry_sorted([(k, r) for k in ("S2L2A", "DEM", "NDVI", "LULC", "Coords")]), ("S1GRD", r))
            for r in rr]))
        d = float(np.abs(ce - ref_ce).max())
        controls["reference"] = {"rows": int(len(rr)), "max_abs_diff": d, "tol": 1e-3}
        ok["reference"] = d < 1e-3

        probe = torch.stack([run.seq([("S2L2A", rows[i, near(date_of(i, 0))])], ("S2L2A", rows[i, date_of(i, 0)]))
                             for i in range(min(4, N))]).to("cuda")
        with torch.no_grad():
            L = probe.shape[1]
            pos = torch.arange(L, device="cuda").unsqueeze(0).expand(len(probe), -1)
            a = model(input_ids=probe, position_ids=pos, use_cache=False).logits.float()
            b = model(input_ids=probe, position_ids=pos + 2000, use_cache=False).logits.float()
            c = model(input_ids=probe, use_cache=False).logits.float()
            moved = pos.clone()
            moved[:, L - BLOCK:] += 3
            e = model(input_ids=probe, position_ids=moved, use_cache=False).logits.float()
        sa = float((a - b).abs().max())
        dz = float((a - c).abs().max())
        real = float((a - e).abs()[:, L - BLOCK:].max())
        controls["shift_all"] = {"constant": 2000, "max_abs_logit_diff": sa, "logit_scale": float(a.abs().mean())}
        controls["delta_zero"] = {"max_abs_logit_diff": dz}
        controls["moved_target_reference_diff"] = real
        ok["shift_all"] = sa < 1e-2 * max(real, 1e-6)      # far below what a real shift of 3 does
        ok["delta_zero"] = dz < 1e-4
        print("controls so far:", json.dumps(controls), flush=True)

    # ------------------------------------------------------------ context table
    if "context" in sections:
        ctx = {}
        for j, m in enumerate(TARGETS):
            names = [n for n, _ in layouts(m, rows[0], 0)]
            per = {}
            for name in names:
                seqs = []
                for i in range(N):
                    t = date_of(i, j)
                    blocks = dict(layouts(m, rows[i], t))[name]
                    seqs.append(run.seq(blocks, (m, rows[i, t])))
                per[name] = run.unit(f"ctx_{m}_{name}", lambda s=seqs: run.run_seqs(s))
            ones = []                          # the same modality at one other date: all three
            for k in (0, 1, 2):
                seqs, dist = [], []
                for i in range(N):
                    t = date_of(i, j)
                    o = [x for x in range(4) if x != t][k]
                    seqs.append(run.seq([(m, rows[i, o])], (m, rows[i, t])))
                    dist.append(abs(o - t))
                c, top = run.unit(f"ctx_{m}_one_other_{k}", lambda s=seqs: run.run_seqs(s))
                ones.append((c, top, np.asarray(dist)))
            base = per["none"][0]
            cell = {}
            for name, (ce, top) in per.items():
                cell[name] = {"ce": group_ci(ce, groups), "top1_pct": 100 * float(top.mean()),
                              "gain_over_none": group_ci(base - ce, groups)}
            # By date distance: 1 and 3 steps are about a season from t on the yearly cycle, 2 is half a year.
            c1, t1, d1 = (np.concatenate([o[k] for o in ones]) for k in range(3))
            g1, b1 = np.tile(groups, 3), np.tile(base, 3)
            cell["one_other_all"] = {"ce": group_ci(c1, g1), "top1_pct": 100 * float(t1.mean()),
                                     "gain_over_none": group_ci(b1 - c1, g1)}
            cell["one_other_by_steps"] = {
                str(d): {"pairs": int((d1 == d).sum()), "ce": group_ci(c1[d1 == d], g1[d1 == d]),
                         "gain_over_none": group_ci((b1 - c1)[d1 == d], g1[d1 == d])} for d in (1, 2, 3)}
            cell["paired_gain_all_other_over_same_date"] = group_ci(per["same_date"][0] - per["all_other_same"][0], groups)
            ctx[m] = cell
        report["context"] = ctx

    # ------------------------------------------------------------ the two pairs
    def pair_seqs(pair, wrong=False):
        seqs = []
        for i in range(N):
            t = date_of(i, 0)
            j = (i + 1) % N if wrong else i                # another location's block
            if pair == "S2L2A>NDVI":
                seqs.append(run.seq([("S2L2A", rows[j, t])], ("NDVI", rows[i, t])))
            else:
                seqs.append(run.seq([("S2L2A", rows[j, near(t)])], ("S2L2A", rows[i, t])))
        return seqs

    def target_alone(pair):
        m = "NDVI" if pair == "S2L2A>NDVI" else "S2L2A"
        return [run.seq([], (m, rows[i, date_of(i, 0)])) for i in range(N)]

    PAIRS = ("S2L2A>NDVI", "S2L2A>S2L2A_other_date")
    if "long" in sections:
        long_out = {}
        for pair in PAIRS:
            seqs = pair_seqs(pair)
            base = run.unit(f"long_{pair}_k1", lambda s=seqs: run.run_seqs(s))[0]
            none = run.unit(f"none_{pair}", lambda: run.run_seqs(target_alone(pair)))[0]
            tab = {}
            for k in LONG_K:
                ce = run.unit(f"long_{pair}_k{k}", lambda s=seqs, k=k: run.run_seqs(s, delta=BLOCK * (k - 1)))[0]
                tab[str(BLOCK * k)] = {"ce": group_ci(ce, groups), "rise_vs_198": group_ci(ce - base, groups)}
            long_out[pair] = {"offsets": tab, "none": group_ci(none, groups), "gain_at_198": group_ci(none - base, groups)}
        report["long_offsets"] = long_out

    if "coords" in sections:
        tab = {}
        for pair in PAIRS:
            def layout(where):
                seqs = []
                for i in range(N):
                    t = date_of(i, 0)
                    src, tgt = (("S2L2A", rows[i, t]), ("NDVI", rows[i, t])) if pair == "S2L2A>NDVI" \
                        else (("S2L2A", rows[i, near(t)]), ("S2L2A", rows[i, t]))
                    c = ("Coords", rows[i, t])
                    seqs.append(run.seq([src, c] if where == "between" else [c, src], tgt))
                return seqs
            base = run.unit(f"long_{pair}_k1", lambda s=pair_seqs(pair): run.run_seqs(s))[0]
            between = run.unit(f"coords_{pair}_between", lambda: run.run_seqs(layout("between")))[0]
            before = run.unit(f"coords_{pair}_before", lambda: run.run_seqs(layout("before")))[0]
            five = run.unit(f"sweep_{pair}_d5", lambda s=pair_seqs(pair): run.run_seqs(s, delta=5))[0]
            tab[pair] = {"adjacent": group_ci(base, groups), "coords_before": group_ci(before, groups),
                         "coords_between": group_ci(between, groups), "moved_5_no_coords": group_ci(five, groups),
                         "between_minus_before": group_ci(between - before, groups)}
        report["coords"] = tab

    if "length" in sections:
        tab = {}
        for k in LENGTH_K:
            seqs = []
            for i in range(N):
                t = date_of(i, 0)
                blocks = date_blocks(rows[i], t) + others_at("S2L2A", rows[i], t)
                seqs.append(run.seq(blocks[-k:], ("S2L2A", rows[i, t])))
            ce, top = run.unit(f"length_S2L2A_k{k}", lambda s=seqs: run.run_seqs(s))
            tab[str(k)] = {"tokens": len(seqs[0]), "ce": group_ci(ce, groups), "top1_pct": 100 * float(top.mean())}
        report["length"] = {"target": "S2L2A", "context": "the last k blocks of all_other_same", "by_k": tab}

    if "sweep" in sections:
        sweep, worst_share, wrong_ok, half_ok = {}, 0.0, True, True
        for pair in PAIRS:
            seqs = pair_seqs(pair)
            none = run.unit(f"none_{pair}", lambda: run.run_seqs(target_alone(pair)))[0]
            ce = {d: run.unit(f"sweep_{pair}_d{d}", lambda s=seqs, d=d: run.run_seqs(s, delta=d))[0]
                  for d in SWEEP + (HALF_BLOCK,)}
            plain = run.unit(f"sweep_{pair}_plain", lambda s=seqs: run.run_seqs(s, plain=True))[0]
            wrong = run.unit(f"sweep_{pair}_wrong", lambda: run.run_seqs(pair_seqs(pair, wrong=True)))[0]
            gain = float((none - ce[0]).mean())
            rises = {d: float((ce[d] - ce[0]).mean()) for d in SWEEP}
            read = {d: r for d, r in rises.items() if d >= READ_FROM}
            d_worst = max(read, key=read.get)
            share = read[d_worst] / gain
            worst_share = max(worst_share, share)
            half = group_ci(ce[HALF_BLOCK] - ce[0], groups)
            wrong_gain = float((none - wrong).mean())
            half_fired = (half["lo"] > 0 or half["hi"] < 0) and abs(half["mean"]) > 1e-3
            wrong_fired = wrong_gain < 0.25 * gain
            half_ok = half_ok and half_fired
            wrong_ok = wrong_ok and wrong_fired
            sweep[pair] = {
                "none": group_ci(none, groups), "ce_delta_0": group_ci(ce[0], groups),
                "gain_over_none": group_ci(none - ce[0], groups),
                "rise_by_delta": {str(d): group_ci(ce[d] - ce[0], groups) for d in SWEEP},
                "worst": {"delta": int(d_worst), "rise": group_ci(ce[d_worst] - ce[0], groups), "share_of_gain": share},
                "half_block": {"delta": HALF_BLOCK, "rise": half, "fired": bool(half_fired)},
                "wrong_source": {"gain_over_none": group_ci(none - wrong, groups), "share_of_gain": wrong_gain / gain,
                                 "fired": bool(wrong_fired)},
                "plain_vs_delta_0_max_abs_ce": float(np.abs(plain - ce[0]).max()),
            }
            ok[f"plain_{pair}"] = sweep[pair]["plain_vs_delta_0_max_abs_ce"] < 1e-4
        report["sweep"] = sweep
        ok["half_block"], ok["wrong_source"] = bool(half_ok), bool(wrong_ok)
        report["worst_share_of_gain"] = worst_share
        report["decision_9"] = "lattice" if worst_share > SHARE_MAX else "sequential"

    controls["passed"] = {k: bool(v) for k, v in ok.items()}
    report["controls"] = controls
    if ok and not all(ok.values()):
        report["refused"] = [k for k, v in ok.items() if not v]
        report.pop("decision_9", None)
    p = out / ("t360_context.json" if sections == set(ALL_SECTIONS) else "partial.json")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=1) + "\n")
    os.replace(tmp, p)
    print(f"wrote {p}\ncontrols: {controls['passed']}\ndecision 9: {report.get('decision_9')}", flush=True)
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
