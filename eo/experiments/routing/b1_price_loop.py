#!/usr/bin/env python
"""B1 (2026-11-15): why the compute target's price has a proportional part (eo/train/compute_target.py).

A toy of the loop, written before any smoke to choose the gains. The router's soft mean depth m responds to the price
through Adam: a constant price moves the router's logits at a constant rate, so m integrates it (B0: the LM loss
barely tells depths apart, so little restores it). Adam's momentum (beta1 0.9) is a 10-update lag on the price, and
the LM loss's shallow pull (B0 R4: with balancing off, ties fall to depth 1) a constant drift:

    m_{t+1} = m_t - k * u_t + drift,     u_t = 0.9 u_{t-1} + 0.1 price_t
    price_t = Kp * e_{t-1} + lam_t,      lam_{t+1} = lam_t + Ki * e_t,      e_t = m_t - target

k, the plant's response (passes per update per unit price), is unknown: B0's R4 suggests ~0.15 (the tied modalities
fell about 1.2 passes over ~200 updates when a force of the balancing's size, ~0.04 in price units, was removed), so
the gains are read over 0.03-1.0. A pure integral price (Kp 0, plain dual ascent) on an integrating plant is a double
integrator: it oscillates or diverges. Read by no rule; it chose the defaults (Kp 0.5, Ki 0.01), and the smokes test
them.

    .venv/bin/python eo/experiments/routing/b1_price_loop.py
    .venv/bin/python eo/experiments/routing/b1_price_loop.py --hard

THE HARD MEASURE (B1 Step 6, 2026-11-16; Enric chose "Hard mean (a)"). The controller now reads the hard (top-1) body
mean, which moves in whole-slot steps: five image slots, each 0.2 of the mean per pass. `sim_hard` is that quantised
plant. Each slot k has a continuous preference y_k in passes (what the soft force moves); its hard depth is y_k
rounded and clipped to 1..3, and the hard mean is the mean of the five. Per update

    y_k <- y_k - k_k * u_t + a * (y*_k - y_k) + noise,    u_t = 0.9 u_{t-1} + 0.1 price_t     (Adam's lag, as above)

with k_k the slot's response to the price (k times a factor 0.7-1.3), a = 0.015 the LM loss's pull toward its own
preferred positions y*_k (shallow: B0's R4, balancing off, sent 8 of 9 tied cells to depth 1 within about 200
updates), and noise 0.01 per update (minibatch). The start is A10c-s2's checkpoint-1650 under the balancing loss
(S2L2A at 1, S1 between 2 and 3, the rest split near 2: hard mean 2.0). The controller reads the hard mean with a
deadband of 0.05: inside it the gap counts as zero.

RULE, fixed 2026-11-16 before the hard toy was run: each (gains, k, seed) run of 200 updates passes if it meets
R-hold-hard's analogue (the time mean of the hard mean over updates 101-200 within 0.05 of 1.8, and no update over
151-200 more than 0.10 away). The replay smoke uses the default gains (proportional 0.5, gain 0.01) unless the gains
x0.3 (0.15, 0.003) pass strictly more of the 10 runs at k = 0.1 and 0.3 (5 seeds each; B0's R4 put k near 0.15).
Everything else it prints is description: the other k, the number of hard-mean changes over 101-200 (a cycle shows as
many), and the price at the end.
"""
import argparse

import numpy as np


def sim(k, Kp, Ki, c=1.8, m0=2.05, drift=-0.004, T=200, beta=0.9):
    m, u, lam, eprev = m0, 0.0, 0.0, 0.0
    ms = []
    for _ in range(T):
        e = m - c
        price = Kp * eprev + lam
        u = beta * u + (1 - beta) * price
        m = m - k * u + drift
        lam += Ki * e
        eprev = e
        ms.append(m)
    err = np.array(ms) - c
    return dict(late=float(np.abs(err[-50:]).mean()), maxdev=float(np.abs(err[100:]).max()), lowest=float(err.min()),
                crossings=int(np.sum(np.diff(np.sign(err[20:])) != 0)))


Y0 = np.array([1.0, 2.5, 2.2, 1.9, 1.6])        # S2L2A, S1, DEM, NDVI, LULC: hard depths 1, 3, 2, 2, 2 (mean 2.0)
Y_PREF = np.array([0.7, 1.3, 1.1, 1.4, 0.9])     # the LM loss's own positions: all shallow without a force (R4)


def sim_hard(k, Kp, Ki, band=0.05, c=1.8, T=200, beta=0.9, a=0.015, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    kk = k * rng.uniform(0.7, 1.3, 5)
    y, u, lam, eprev = Y0.copy(), 0.0, 0.0, 0.0
    H, P = [], []
    for _ in range(T):
        h = np.clip(np.floor(y + 0.5), 1, 3)
        Hm = float(h.mean())
        e = Hm - c
        e = 0.0 if abs(e) <= band else e
        price = Kp * eprev + lam
        u = beta * u + (1 - beta) * price
        y = y - kk * u + a * (Y_PREF - y) + rng.normal(0, noise, 5)
        lam += Ki * e
        eprev = e
        H.append(Hm)
        P.append(price)
    H = np.array(H)
    late, last = H[100:] - c, H[150:] - c
    return dict(mean_gap=float(late.mean()), max_late=float(np.abs(last).max()),
                passes=bool(abs(late.mean()) <= 0.05 and np.abs(last).max() <= 0.10),
                changes=int(np.sum(np.diff(H[100:]) != 0)), price_end=float(P[-1]),
                table=np.clip(np.floor(y + 0.5), 1, 3).astype(int).tolist())


def hard_main():
    gains = {"defaults (0.5, 0.01)": (0.5, 0.01), "x0.3 (0.15, 0.003)": (0.15, 0.003)}
    passes = {}
    print("per (gains, k): runs passing R-hold-hard's analogue of 5 seeds; mean gap over 101-200 and max |e| over "
          "151-200 (seed means); hard-mean changes over 101-200; price at the end; the final table of seed 0 "
          "(S2L2A, S1, DEM, NDVI, LULC)")
    for name, (Kp, Ki) in gains.items():
        for k in (0.03, 0.1, 0.3, 1.0):
            rs = [sim_hard(k, Kp, Ki, seed=s) for s in range(5)]
            n = sum(r["passes"] for r in rs)
            if k in (0.1, 0.3):
                passes[name] = passes.get(name, 0) + n
            print(f"  {name:22s} k={k:<4}: pass {n}/5  gap {np.mean([r['mean_gap'] for r in rs]):+.3f}  max "
                  f"{np.mean([r['max_late'] for r in rs]):.2f}  changes {np.mean([r['changes'] for r in rs]):5.1f}  "
                  f"price {np.mean([r['price_end'] for r in rs]):+.3f}  table {rs[0]['table']}")
    d, x = passes["defaults (0.5, 0.01)"], passes["x0.3 (0.15, 0.003)"]
    pick = "x0.3 (0.15, 0.003)" if x > d else "defaults (0.5, 0.01)"
    print(f"RULE: at k 0.1 and 0.3 the defaults pass {d}/10 and the gains x0.3 {x}/10 -> the replay uses the {pick}")
    return pick


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hard", action="store_true", help="the quantised plant under the hard measure (B1 Step 6)")
    if ap.parse_args().hard:
        hard_main()
    else:
        print("late |e| = mean |m - target| over updates 150-200; maxdev = max |e| from update 100; crossings of the target")
        for Kp, Ki in [(0.0, 0.01), (0.0, 0.003), (0.5, 0.01), (0.7, 0.015), (1.0, 0.02), (2.0, 0.04)]:
            print(f"Kp {Kp:3.1f} Ki {Ki:5.3f}: " + "  ".join(
                f"k={k:<4}: late {r['late']:.3f} maxdev {r['maxdev']:.2f} x{r['crossings']}"
                for k in (0.03, 0.1, 0.3, 1.0) for r in [sim(k, Kp, Ki)]))
