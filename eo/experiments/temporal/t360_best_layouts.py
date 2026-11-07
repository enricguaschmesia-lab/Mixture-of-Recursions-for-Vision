#!/usr/bin/env python
"""Is Step 0's "same modality at the other dates" really T360's best layout with that information?

Step 9's margin and Step 12's baseline say "T360's best layout with the same information", but Step 0
measured only same-modality context (and full lengths, which fail). A left-out-season prompt for
T360-T also holds statics and the other modalities at other dates. So, same 256 dev locations, same
target dates, no date tokens, no block from the target's date:

  statics_same_mod             statics + the same modality at the other three dates     1,193 tokens
  statics_near_all             statics + all three dynamic modalities at the near date  1,193 tokens
  statics_near_others_same_mod statics + the other two at the near date + same modality 1,589 tokens

CONTROLS: the same code path reproduces the stored same_mod_all unit; the same blocks from another
location must lose nearly all the gain they give over statics.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(.venv/bin/python -m eo.train.gpu titanx) \\
      .venv/bin/python -X faulthandler -u eo/experiments/temporal/t360_best_layouts.py
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import t360_context as C  # noqa: E402  (this folder is on sys.path when run as a script; sets the allocator variable)
import torch  # noqa: E402

from eo.data.temporal_split import load_temporal_split  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402

OUT = Path("/data/enric/reports/temporal/t360_best_layouts")
STORED = Path("/data/enric/reports/temporal/t360_context/units")


def same_mod(m, r, t, src=None):
    s = r if src is None else src
    return [(m, s[o]) for o in range(4) if o != t]


def near_all(r, t):
    return [(k, r[C.near(t)]) for k in C.TARGETS]


def near_others(m, r, t):
    return [(k, r[C.near(t)]) for k in C.TARGETS if k != m]


LAYOUTS = {
    "statics_same_mod": lambda m, r, t: C.statics(r, t) + same_mod(m, r, t),
    "statics_near_all": lambda m, r, t: C.statics(r, t) + near_all(r, t),
    "statics_near_others_same_mod": lambda m, r, t: C.statics(r, t) + near_others(m, r, t) + same_mod(m, r, t),
}


def stored(name):
    return np.load(STORED / f"{name}.npz")["ce"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_LOC)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from eo.generate import conditional as G
    if torch.cuda.device_count() != 1:
        raise SystemExit("pin exactly one GPU")
    gpu = torch.cuda.get_device_name(0)
    if "TITAN X" not in gpu:
        raise SystemExit(f"CUDA sees {gpu!r}, expected the Titan X")
    model, cfg = G.build_model(C.ARM, C.CKPT)
    if (cfg.get("multimodal") or {}).get("token_set") != "clean":
        raise SystemExit("cleaned tokens only")
    if next(model.parameters()).dtype != torch.float32:
        raise SystemExit("fp32 parameters expected")
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    print(f"{gpu}; token root {ds.root_dir}; {C.CKPT}", flush=True)

    doc = load_temporal_split(C.ROOT)
    rows, groups, _ = C.pick_locations(doc, args.n)
    out = Path(args.out)
    run = C.Run(model, ds, "cuda", out)
    N = len(rows)
    full = N == C.N_LOC                     # the stored units cover exactly these 256
    report = {"checkpoint": C.CKPT, "gpu": gpu, "precision": "fp32", "n_locations": N,
              "n_groups": int(len(np.unique(groups))), "bootstrap": {"draws": C.DRAWS, "unit": "5 km spatial group"},
              "targets": {}}
    ok = {}

    def seqs_for(m, j, fn):
        return [run.seq(fn(m, rows[i], C.date_of(i, j)), (m, rows[i, C.date_of(i, j)])) for i in range(N)]

    # Control 1: this code path, the stored layout.
    ref_seqs = seqs_for("S2L2A", 0, lambda m, r, t: same_mod(m, r, t))
    ref = run.unit("ctrl_same_mod_all_S2L2A", lambda s=ref_seqs: run.run_seqs(s))[0]
    if full:
        d = float(np.abs(ref - stored("ctx_S2L2A_same_mod_all")).max())
        report["control_reference"] = {"max_abs_diff": d, "tol": 1e-3}
        ok["reference"] = d < 1e-3
        print(f"reference: max |CE - stored| = {d:.2e}", flush=True)

    for j, m in enumerate(C.TARGETS):
        cell = {}
        for name, fn in LAYOUTS.items():
            ce, top = run.unit(f"{name}_{m}", lambda s=seqs_for(m, j, fn): run.run_seqs(s))
            entry = {"tokens": len(run.seq(fn(m, rows[0], C.date_of(0, j)), (m, rows[0, C.date_of(0, j)]))),
                     "ce": C.group_ci(ce, groups), "top1_pct": 100 * float(top.mean())}
            t = np.array([C.date_of(i, j) for i in range(N)])
            entry["ce_by_target_date"] = {str(x): float(ce[t == x].mean()) for x in range(4)}
            if full:
                base = stored(f"ctx_{m}_same_mod_all")
                entry["minus_same_mod_all"] = C.group_ci(ce - base, groups)
                ones = [stored(f"ctx_{m}_one_other_{k}") for k in range(3)]
                pick = np.array([[x for x in range(4) if x != t[i]].index(C.near(int(t[i]))) for i in range(N)])
                near_one = np.array([ones[pick[i]][i] for i in range(N)])
                entry["minus_one_other_near_date"] = C.group_ci(ce - near_one, groups)
                entry["reference_one_other_near_date"] = C.group_ci(near_one, groups)
            cell[name] = entry
        if full:
            s = stored(f"ctx_{m}_same_mod_all")
            t = np.array([C.date_of(i, j) for i in range(N)])
            cell["same_mod_all_stored"] = {"ce": C.group_ci(s, groups),
                                           "ce_by_target_date": {str(x): float(s[t == x].mean()) for x in range(4)}}
        report["targets"][m] = cell

    # Control 2: the same blocks from another location give (almost) nothing over statics.
    def wrong(m, r, t, i):
        return C.statics(r, t) + same_mod(m, r, t, src=rows[(i + 1) % N])

    wr = [run.seq(wrong("S2L2A", rows[i], C.date_of(i, 0), i), ("S2L2A", rows[i, C.date_of(i, 0)])) for i in range(N)]
    w = run.unit("ctrl_wrong_statics_same_mod_S2L2A", lambda s=wr: run.run_seqs(s))[0]
    right = run.unit("statics_same_mod_S2L2A", lambda: None)[0]
    if full:
        st = stored("ctx_S2L2A_statics")
        gain, wrong_gain = float((st - right).mean()), float((st - w).mean())
        report["control_wrong_location"] = {"gain_over_statics": gain, "wrong_gain_over_statics": wrong_gain}
        ok["wrong_location"] = wrong_gain < 0.25 * gain
    report["controls_passed"] = {k: bool(v) for k, v in ok.items()}

    p = out / "t360_best_layouts.json"
    p.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {p}; controls {report['controls_passed']}", flush=True)
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
