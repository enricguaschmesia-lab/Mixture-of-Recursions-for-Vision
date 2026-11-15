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
"""
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


if __name__ == "__main__":
    print("late |e| = mean |m - target| over updates 150-200; maxdev = max |e| from update 100; crossings of the target")
    for Kp, Ki in [(0.0, 0.01), (0.0, 0.003), (0.5, 0.01), (0.7, 0.015), (1.0, 0.02), (2.0, 0.04)]:
        print(f"Kp {Kp:3.1f} Ki {Ki:5.3f}: " + "  ".join(
            f"k={k:<4}: late {r['late']:.3f} maxdev {r['maxdev']:.2f} x{r['crossings']}"
            for k in (0.03, 0.1, 0.3, 1.0) for r in [sim(k, Kp, Ki)]))
