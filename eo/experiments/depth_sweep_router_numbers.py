"""The numbers behind the learned-router depth-sweep slides (2026-10-28). `.venv`, CPU.
Reads the stored units through depth_sweep_router_report's loaders; prints, does not write.
    .venv/bin/python eo/experiments/depth_sweep_router_numbers.py > /data/enric/reports/depth_sweep_router/title_numbers.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import depth_sweep_router_report as Q  # noqa: E402

R = Q.R
U = Q.load()
CK = Q.CKS[-1]
pp = lambda b: "    n/a" if b is None else f"{100 * b[0]:+.3f} [{100 * b[1][0]:+.3f}, {100 * b[1][1]:+.3f}]"
nat = lambda b: "    n/a" if b is None else f"{b[0]:+.4f} [{b[1][0]:+.4f}, {b[1][1]:+.4f}]"
sig = lambda b: b is not None and (b[1][0] > 0 or b[1][1] < 0)

print("== 1. Routed depth per target at its predicting positions (all-to-one), epoch 10: mean, mix d1/d2/d3")
for s in Q.SEEDS:
    rd, mix = Q.routed_depth(s, CK), Q.routed_mix(s, CK)
    print(f"  {Q.SEEDS[s]['short']}: " + "  ".join(
        f"{Q.NICE[T]} {rd[T]:.2f} ({'/'.join(f'{100 * x:.0f}' for x in mix[T]) if mix else '-'})" for T in Q.TARGETS))

for metric, fmt in (("top1", pp), ("ce", nat)):
    print(f"\n== 2{'a' if metric == 'top1' else 'b'}. Target only at d, minus as routed ({'pp' if metric == 'top1' else 'CE gain, nats'}), "
          f"epoch 10; * = paired 95% interval excludes 0")
    for T in Q.TARGETS:
        for s in Q.SEEDS:
            bs = {d: Q.boot_a2o(s, ("routed", None), ("target", d), CK, T, metric) for d in Q.DEPTHS}
            best = Q.best_target_depth(U, s, T, CK, metric)
            parts = "  ".join(f"d{d} {fmt(b)}{'*' if sig(b) else ' '}" for d, b in bs.items())
            print(f"  {Q.NICE[T]:6s} {Q.SEEDS[s]['short']} routed {Q.routed_depth(s, CK)[T]:.2f} best {best}:  {parts}")

print("\n== 3. Every position at d, minus as routed, pooled views, epoch 10 (pp); * excludes 0")
for s in Q.SEEDS:
    for view, f in (("any", Q.boot_any), ("a2o", Q.boot_a2o), ("o2o", Q.boot_o2o)):
        bs = {d: f(s, ("routed", None), ("all", d), CK) for d in Q.DEPTHS}
        print(f"  {Q.SEEDS[s]['short']} {view}: " + "  ".join(f"d{d} {pp(b)}{'*' if sig(b) else ' '}" for d, b in bs.items()))

print("\n== 4. Pooled top-1 (%), as routed and every position at d, per epoch (any / a2o / o2o)")
for s in Q.SEEDS:
    for ck in Q.CKS:
        r = U.get((s, "routed", None, ck))
        if r is None:
            continue
        g = lambda u, v: (u["any"]["top1"] if v == "any" else u[f"{v}_pooled"]["top1"]) if (
            "any" in u if v == "any" else f"{v}_pooled" in u) else float("nan")
        parts = [f"routed {100 * g(r, 'any'):.2f}/{100 * g(r, 'a2o'):.2f}/{100 * g(r, 'o2o'):.2f}"]
        for d in Q.DEPTHS:
            u = U.get((s, "all", d, ck))
            if u:
                parts.append(f"d{d} {100 * g(u, 'any'):.2f}/{100 * g(u, 'a2o'):.2f}/{100 * g(u, 'o2o'):.2f}")
        print(f"  {Q.SEEDS[s]['short']} ep{Q.EPOCH[ck]:2d}  " + "  ".join(parts))

print("\n== 5. Over training: best target-only depth vs the router's modal depth, per target (top-1 | CE)")
for s in Q.SEEDS:
    for metric in ("top1", "ce"):
        hits = n = 0
        rows = []
        for ck in Q.CKS:
            mix = Q.routed_mix(s, ck)
            for T in Q.TARGETS:
                bd = Q.best_target_depth(U, s, T, ck, metric)
                if bd is None or mix is None:
                    continue
                modal = int(np.argmax(mix[T])) + 1
                n += 1; hits += bd == modal
                rows.append(f"ep{Q.EPOCH[ck]}:{Q.NICE[T]} {modal}->{bd}")
        print(f"  {Q.SEEDS[s]['short']} {metric}: best == router's modal depth in {hits} of {n}")

print("\n== 6. Do the two seeds agree on the best depth (target only, top-1), epoch 10, where they route differently?")
for T in Q.TARGETS:
    b = {s: Q.best_target_depth(U, s, T, CK) for s in Q.SEEDS}
    m = {s: Q.routed_mix(s, CK) for s in Q.SEEDS}
    md = {s: (int(np.argmax(m[s][T])) + 1) if m[s] else None for s in Q.SEEDS}
    print(f"  {Q.NICE[T]:6s} routed (modal) {md['a10c']} / {md['a10cs2']}, best {b['a10c']} / {b['a10cs2']}")

print("\n== 7. Context depth: every position at d minus target only at d, all-to-one per target, epoch 10 (pp)")
for s in Q.SEEDS:
    for T in Q.TARGETS:
        bs = {d: Q.boot_a2o(s, ("target", d), ("all", d), CK, T) for d in Q.DEPTHS}
        print(f"  {Q.SEEDS[s]['short']} {Q.NICE[T]:6s} " + "  ".join(f"d{d} {pp(b)}{'*' if sig(b) else ' '}" for d, b in bs.items()))
