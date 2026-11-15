#!/usr/bin/env python
"""B1 (2026-11-15): intervention 4's smoke, read by a rule written before it ran: what does the switch cost?

The smoke continues A10cR's checkpoint-1980 under A10c's learned router (balancing 0.1, the plain gate) to 2180, as an
exact continuation (eo/train/warm_start.py: Adam, schedule and data carry on; /data/enric/reports/b1/smoke/ws_replay.sh).
A10cR and A10c share seed 42 and every data key, so the warm run trains on exactly the batches A10cR trained on at the
same steps, and its logged training loss pairs window by window with A10cR's (trainer_state.json, every 10 updates).
At the switch the gate drops from A10cR's top-of-three-uniforms (mean 0.75) to a fresh softmax router's top
probability (about 1/3), so every update shrinks: Enric chose the plain gate, smoked first (2026-11-15).

RULE, fixed 2026-11-15 before the smoke ran:
  The switch costs little if the warm run's held-out eval at 2145 (eval_steps 165) exceeds A10cR's logged eval at
  2145 by at most 0.0111 nats, the 1 + 9 + 1 val cell's own seed gap at 1980 (A10cR - A10cR-s2, wave 1). Otherwise
  it is large, and the gate question reopens (open decision 9) before arms C and D run.
Description, by no rule: the training-loss excess per 10-update window (paired: same batches), its first window and
its mean over 2081-2180; each image modality's hard and soft depth over 1981-2030 and 2131-2180 (where the fresh
router goes); the gate (soft top probability is not logged; the soft mean depth stands in).

    .venv/bin/python eo/experiments/routing/b1_ws_smoke.py --self-test
    .venv/bin/python eo/experiments/routing/b1_ws_smoke.py b1w_cr1980_c /data/enric/runs/pretrain/phase3/b1w_cr1980_c

Writes /data/enric/reports/b1/smoke/<tag>.{json,txt}. Runs in `.venv`, CPU, seconds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_ct_smoke as CS  # noqa: E402

OUT = Path("/data/enric/reports/b1/smoke")
SOURCE_STATE = Path("/data/enric/runs/pretrain/phase3/mor10cr_20261012_095834/checkpoint-3300/trainer_state.json")
EVAL_STEP, GAP = 2145, 0.0111
START, STOP = 1980, 2180
IMG = CS.IMG


def logged(state_path: Path):
    """{step: training loss}, {step: eval loss} from a trainer_state.json's log history (the last entry per step)."""
    hist = json.loads(Path(state_path).read_text())["log_history"]
    tr = {int(h["step"]): float(h["loss"]) for h in hist if "loss" in h and "eval_loss" not in h}
    ev = {int(h["step"]): float(h["eval_loss"]) for h in hist if "eval_loss" in h}
    return tr, ev


def run_state(run_dir: Path) -> Path:
    """The warm run's own log: trainer_state.json at the run's root (pretrain.py's save_state), else its last save."""
    root = Path(run_dir) / "trainer_state.json"
    if root.is_file():
        return root
    cks = sorted(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] / "trainer_state.json"


def read(tr_w, ev_w, tr_s, ev_s, depth_rows) -> dict:
    steps = sorted(s for s in tr_w if START < s <= STOP and s in tr_s)
    exc = {s: tr_w[s] - tr_s[s] for s in steps}
    late = [exc[s] for s in steps if s > 2080]
    out = {"paired_windows": len(steps), "excess": exc,
           "first_window": exc[steps[0]] if steps else None,
           "late_mean": float(np.mean(late)) if late else None}
    if START in ev_w and START in ev_s and ev_w[START] != ev_s[START]:
        # description: the start-up eval of the resumed run is the copy's own (fresh router), before any update
        out["eval_at_switch"] = {"warm": ev_w[START], "source": ev_s[START], "excess": ev_w[START] - ev_s[START]}
    if EVAL_STEP in ev_w and EVAL_STEP in ev_s:
        d = ev_w[EVAL_STEP] - ev_s[EVAL_STEP]
        out["eval"] = {"warm": ev_w[EVAL_STEP], "source": ev_s[EVAL_STEP], "excess": d,
                       "verdict": "costs little" if d <= GAP else "large: the gate question reopens"}
    else:
        out["eval"] = None
    if depth_rows is not None:
        classes, st, counts, soft = depth_rows
        res = {}
        for name, (a, b) in (("first", (1981, 2030)), ("last", (2131, 2180))):
            sel = (st >= a) & (st <= b)
            if not sel.any():
                continue
            mods = {}
            for m in IMG:
                i = classes.index(m)
                c = counts[sel, i, :]
                tot = c.sum(1)
                ok = tot > 0
                if not ok.any():
                    continue
                sh = (c[ok] / tot[ok, None]).mean(0)
                mods[m] = {"shares": sh.round(4).tolist(), "depth": float((sh * np.arange(1, 4)).sum())}
                if soft is not None:
                    mods[m]["soft"] = float((soft[sel, i][ok] / tot[ok]).mean())
            res[name] = mods
        out["depth"] = res
    return out


def text(tag, res) -> str:
    L = [f"{tag}: the switch from A10cR's random router to a learned one at {START}, paired with A10cR at the same steps"]
    e = res["eval"]
    if e:
        L.append(f"  held-out eval at {EVAL_STEP}: warm {e['warm']:.4f}, A10cR {e['source']:.4f}, excess {e['excess']:+.4f} "
                 f"(rule: at most {GAP}) -> {e['verdict']}")
    else:
        L.append(f"  held-out eval at {EVAL_STEP}: missing")
    z = res.get("eval_at_switch")
    if z:
        L.append(f"  held-out eval at the switch ({START}, the fresh router before any update; description): warm "
                 f"{z['warm']:.4f}, A10cR {z['source']:.4f}, excess {z['excess']:+.4f}")
    if res["first_window"] is not None:
        L.append(f"  training loss, warm minus A10cR (same batches), {res['paired_windows']} windows of 10: first "
                 f"{res['first_window']:+.4f}, mean over 2081-{STOP} {res['late_mean']:+.4f}")
        L.append("    " + " ".join(f"{s}:{v:+.3f}" for s, v in res["excess"].items()))
    for name, mods in (res.get("depth") or {}).items():
        L.append(f"  depths, {name} 50 updates: " + "  ".join(
            f"{m} {x['depth']:.2f}" + (f"/{x['soft']:.2f}" if "soft" in x else "") for m, x in mods.items())
                 + "  (hard/soft)")
    return "\n".join(L)


def self_test() -> int:
    tr_s = {s: 5.0 for s in range(1990, 2190, 10)}
    ev_s = {2145: 5.50}
    good = read({s: 5.003 for s in tr_s}, {2145: 5.505}, tr_s, ev_s, None)
    bad = read({s: 5.05 for s in tr_s}, {2145: 5.53}, tr_s, ev_s, None)
    ok = (good["eval"]["verdict"] == "costs little" and bad["eval"]["verdict"].startswith("large")
          and abs(good["first_window"] - 0.003) < 1e-12 and good["paired_windows"] == 20)
    print(f"  self-test: +0.005 at {EVAL_STEP} costs little, +0.03 is large, 20 paired windows -> {'PASS' if ok else 'FAIL'}")
    print("self-test", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag", nargs="?")
    ap.add_argument("run_dir", nargs="?", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.tag and a.run_dir):
        ap.error("tag and run_dir, or --self-test")
    tr_w, ev_w = logged(run_state(a.run_dir))
    tr_s, ev_s = logged(SOURCE_STATE)
    f = a.run_dir / "depth_shares.jsonl"
    rows = None
    if f.is_file():
        classes, st, counts, soft, _, _ = CS.load(f)
        rows = (classes, st, counts, soft)
    res = read(tr_w, ev_w, tr_s, ev_s, rows)
    t = text(a.tag, res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.txt").write_text(t + "\n")
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
