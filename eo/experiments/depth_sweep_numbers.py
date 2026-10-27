"""The numbers behind the depth-sweep deck's slide titles (2026-10-27). `.venv`, CPU, ~3 min.
Reads the stored units through depth_sweep_report's loaders; prints, does not write.
    .venv/bin/python eo/experiments/depth_sweep_numbers.py > /data/enric/reports/depth_sweep/title_numbers_final.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_sweep_report as R  # noqa: E402

U, V = R.load_all()
pp = lambda x: f"{100 * x:+.3f}"
ci = lambda b: f"[{100 * b[1][0]:+.3f}, {100 * b[1][1]:+.3f}]"
sig = lambda b: "*" if b[1][0] > 0 or b[1][1] < 0 else " "

print("== 1. Pooled top-1 per model, depth, epoch (any / a2o / o2o, %)")
for mk, spec in R.MODELS.items():
    for ck in R.CKS:
        ds = [d for d in spec["depths"] if (mk, d, ck) in U]
        if not ds:
            continue
        line = "  ".join(f"d{d} {100 * R.val(U[(mk, d, ck)], 'any'):.2f}/{100 * R.val(U[(mk, d, ck)], 'a2o'):.2f}/"
                         f"{100 * R.val(U[(mk, d, ck)], 'o2o'):.2f}" for d in ds)
        print(f"  {mk:6s} ep{R.EPOCH[ck]:2d}  {line}")
for vk in R.VAN:
    for ck in R.CKS:
        if (vk, ck) in V:
            u = V[(vk, ck)]
            print(f"  {vk:6s} ep{R.EPOCH[ck]:2d}  {100 * R.val(u, 'any'):.2f}/{100 * R.val(u, 'a2o'):.2f}/{100 * R.val(u, 'o2o'):.2f}")

print("\n== 2. Depth effect per epoch: depth d minus depth 1 (pp), any / a2o / o2o; spread over trained depths")
for mk, spec in R.MODELS.items():
    for ck in R.CKS:
        if (mk, 1, ck) not in U:
            continue
        u1 = U[(mk, 1, ck)]
        parts = []
        for d in spec["depths"][1:]:
            if (mk, d, ck) in U:
                u = U[(mk, d, ck)]
                parts.append(f"d{d} " + "/".join(f"{100 * (R.val(u, v) - R.val(u1, v)):+.2f}" for v in R.VIEWS))
        print(f"  {mk:6s} ep{R.EPOCH[ck]:2d}  " + "  ".join(parts))

print("\n== 3. Epoch effect for scale: depth 1 at epoch e minus epoch e-1 (pp, any / a2o / o2o)")
for mk in R.MODELS:
    prev = None
    out = []
    for ck in R.CKS:
        if (mk, 1, ck) in U:
            if prev is not None:
                out.append(f"ep{R.EPOCH[ck]} " + "/".join(f"{100 * (R.val(U[(mk, 1, ck)], v) - R.val(prev, v)):+.2f}"
                                                         for v in R.VIEWS))
            prev = U[(mk, 1, ck)]
    print(f"  {mk:6s} " + "  ".join(out))

print("\n== 4. Deepest trained depth minus depth 1, per epoch, paired 95% (any; o2o); * = excludes 0")
for mk, spec in R.MODELS.items():
    nr = spec["nr"]
    for ck in R.CKS:
        if (mk, 1, ck) in U and (mk, nr, ck) in U:
            a = R.boot_any(f"{mk}_force1", f"{mk}_force{nr}", ck)
            o = R.boot_o2o(f"{mk}_force1", f"{mk}_force{nr}", ck)
            print(f"  {mk:6s} ep{R.EPOCH[ck]:2d}  any {pp(a[0])} {ci(a)}{sig(a)}   o2o {pp(o[0])} {ci(o)}{sig(o)}")
for ck in R.CKS:
    if ("b10c", ck) in V and ("b10c11", ck) in V:
        a = R.boot_any("b10c11", "b10c", ck); o = R.boot_o2o("b10c11", "b10c", ck)
        print(f"  V29-V11 ep{R.EPOCH[ck]:2d}  any {pp(a[0])} {ci(a)}{sig(a)}   o2o {pp(o[0])} {ci(o)}{sig(o)}")

print("\n== 5. Best depth per epoch (any-to-any) and its gain over depth 1 (pp)")
for mk in R.MODELS:
    out = []
    for ck in R.CKS:
        db = R.best_depth(U, mk, ck)
        if db and (mk, 1, ck) in U:
            out.append(f"ep{R.EPOCH[ck]} d{db} {100 * (R.val(U[(mk, db, ck)], 'any') - R.val(U[(mk, 1, ck)], 'any')):+.3f}")
    print(f"  {mk:6s} " + "  ".join(out))

print("\n== 6. MoR at depth d minus vanilla 11 at the same epoch (pp, any / a2o / o2o)")
for mk, spec in R.MODELS.items():
    for ck in R.CKS:
        if ("b10c11", ck) not in V:
            continue
        v = V[("b10c11", ck)]
        parts = [f"d{d} " + "/".join(f"{100 * (R.val(U[(mk, d, ck)], w) - R.val(v, w)):+.2f}" for w in R.VIEWS)
                 for d in spec["depths"] if (mk, d, ck) in U]
        if parts:
            print(f"  {mk:6s} ep{R.EPOCH[ck]:2d}  " + "  ".join(parts))

print("\n== 7. Final-epoch per-target gains, deepest trained depth minus depth 1 (pp): a2o, and o2o per source")
for mk, spec in R.MODELS.items():
    ck, nr = R.CKS[-1], spec["nr"]
    if (mk, 1, ck) not in U or (mk, nr, ck) not in U:
        continue
    hi, lo = U[(mk, nr, ck)], U[(mk, 1, ck)]
    za = np.load(R.ARM / f"{mk}_force1" / f"checkpoint-{ck}" / "grid_rows.npz")
    zb = np.load(R.ARM / f"{mk}_force{nr}" / f"checkpoint-{ck}" / "grid_rows.npz")
    rng = np.random.default_rng(0)
    nsig = []
    for T in R.TARGETS:
        cells = []
        for s in R.SOURCES:
            k = f"{s}->{T}"
            if k in hi["o2o"] and f"{k}|rows" in za:
                p, bt = R._cell_boot(za, zb, k, rng, 2000)
                lo_, hi_ = np.percentile(bt, [2.5, 97.5])
                s_ = lo_ > 0 or hi_ < 0
                if s_:
                    nsig.append((k, round(100 * p, 3)))
                cells.append(f"{R.NICE[s]} {100 * p:+.2f}{'*' if s_ else ''}")
        print(f"  {mk:6s} -> {T:6s} a2o {100 * (hi['a2o'][T]['top1'] - lo['a2o'][T]['top1']):+.2f}   " + ", ".join(cells))
    print(f"  {mk}: {len(nsig)} of 34 one-to-one cells with a 95% interval excluding 0: {nsig}")

print("\n== 8. Final epoch, at matched compute: MoR depth 1 vs V11 (11 layers), MoR depth 3 vs V29 (29 layers)")
for mk in R.MODELS:
    ck = R.CKS[-1]
    for d, vk in ((1, "b10c11"), (3, "b10c")):
        if (mk, d, ck) in U:
            a = R.boot_any(vk, f"{mk}_force{d}", ck); o = R.boot_o2o(vk, f"{mk}_force{d}", ck)
            print(f"  {mk:6s} d{d} - {vk:6s}  any {pp(a[0])} {ci(a)}{sig(a)}   o2o {pp(o[0])} {ci(o)}{sig(o)}")
