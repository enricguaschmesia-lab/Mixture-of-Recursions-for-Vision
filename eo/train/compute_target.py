# eo/train/compute_target.py
"""Paper B, intervention 1: a compute target in place of the equal-load balancing loss (B1, 2026-11-15).

WHAT IT DOES. `mor.token.balancing: compute` turns the balancing loss off. Instead the training loss gains
`price * m`, where m is the SOFT expected depth of the body positions,

    m = mean over body positions of  sum_d d * p_d        (d = 1..Nr passes, p the router's softmax),

and after every optimizer step the price is reset from the gap between that update's own m and the target:

    e_t       = m_t - target                       (m_t over all micro-batches of update t)
    lam_{t+1} = lam_t + gain * e_t                 (integral part)
    price_{t+1} = proportional * e_t + lam_{t+1}   (the price update t+1 trains under)

Both parts move IN PROPORTION TO THE GAP, never by its sign, and m is a continuous function of the router's
probabilities. B0 found that the balancing loss rings because its loop closes through a hard count (bincount
of the top-1 choice), and a sign update on a measured mean would be a hard count again
(docs/notes/routing/b0_audit_2026-11-10.md; Paper B plan, B2).

WHY A PROPORTIONAL PART TOO (decided 2026-11-15, before any smoke). Under Adam a constant price moves the
router's logits at a constant rate, so the router's soft mean integrates the price: the plant is close to an
integrator (B0: the LM loss barely tells depths apart, so little restores it). A price that is only the
integral of the gap (dual ascent) on an integrating plant is a double integrator, which oscillates. A toy loop
with Adam's momentum lag (eo/experiments/routing/b1_price_loop.py) agreed: with the price as a pure
integral, no gain held the mean within 200 updates for any plant response from 0.03 to 1.0 pass per unit price
per update (late errors 0.3 to 10 passes, oscillating), while proportional 0.5 with gain 0.01 held it within
0.03 for responses 0.1-0.3 and stayed stable from 0.03 to 1.0. `proportional: 0` is plain dual ascent.

BODY POSITIONS are B0's label frame, the classes the per-update depth shares already use (util/trainer_pt.py,
gate W31): position t counts under the modality of the token it predicts (t + 1) when that is a body token.
Positions predicting a BO/EO, padding and the last position are masked (5.5% of positions; every learned run
put them at depth 3, and with them in, the statistic would price a free 0.166 of depth-3 load).

WHAT THE CONTROLLER READS (`measure`, added 2026-11-16; Enric: "Hard mean (a)", B1 plan Step 5, decision 12). B1's
smokes found the router's probabilities stay diffuse, so the soft mean is not the compute: a soft target of 1.8 held
the soft mean to 0.003 while the hard (top-1) body mean, which is what compute follows, settled at 1.38. So:

    measure: soft   e_t = soft mean - target   (as built; the four smoke overlays keep it)
    measure: hard   e_t = hard mean - target   (the update's top-1 depths over its body positions, from the
                                                per-class counts the depth shares already accumulate)

Either way the price still multiplies the SOFT mean in the loss, so the force on the router stays continuous; only
what the controller measures changes. The hard mean moves in whole-modality steps (each image slot is 0.2 of the
mean per pass), so a DEADBAND: when |e_t| <= deadband the gap counts as zero (the integral holds and the
proportional part is zero), and outside it the rule above applies to the full gap. 0 <= deadband < 0.1 (half a
slot); arm B uses 0.05, a quarter of a slot, so a whole-modality 1.8 sits inside it.

A CONFIDENCE TERM (`confidence`, added 2026-11-17; Enric: "Build arms B and D and launch them", with what B1 learnt;
B1 plan Step 7b, option (b) of Step 5). Under the soft measure the router met a 1.8 target from step 0 by spreading its
probabilities while every top-1 choice fell to depth 1 (hard 1.008, Step 6.5), and under the hard measure near-uniform
choices flipped in whole groups and cycled (Step 6.4). Both because the probabilities stay diffuse. So, optionally,
the loss also gains

    confidence * mean over body positions of  H(p) = - sum_d p_d log p_d      (nats; 0 <= H <= log Nr)

which sharpens each position's choice, so that the soft mean the controller holds is the compute. It is part of the
arm (in the state file's identity: a resume may not change it). Optional: absent means 0, nothing is added and the
forward and loss are those of a run without it.

THE PRICE IS TRAINER STATE, not a weight: <checkpoint>/compute_target.json, written at every save and read back
on a resume. A resume without it is refused, because a price restarted at its initial value would be a silent
transient (bal_warmup_step's unsaved counter restarts the same way). A replay from another run's checkpoint gets
the file from its launcher (`write_state(..., origin="replay start ...")`).

Runs in `.venv`; pure Python apart from the file I/O. Its gate is W34 in eo/scripts/verify_phase3.py, its
preflight check `check_compute_target` (eo/train/preflight.py), with controls in eo/train/preflight_controls.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

FILE = "compute_target.json"
MODE = "compute"
KEYS = ("target", "gain", "proportional", "price_init", "measure", "deadband")
#: Keys a block may leave out, with the value their absence means (B1 Step 7b, 2026-11-17).
OPTIONAL = {"confidence": 0.0}
NUMBERS = ("target", "gain", "proportional", "price_init", "deadband", "confidence")
CONFIDENCE_MAX = 10.0       # inclusive: a coefficient on a mean entropy of at most log Nr nats
MEASURES = ("soft", "hard")
DEADBAND_MAX = 0.1          # exclusive: half an image slot (each slot is 0.2 of the body mean per pass)
#: The fields a resume must not change (a schedule key may; these define the arm).
IDENTITY = KEYS + tuple(OPTIONAL)
#: What a version-1 file (2026-11-15, before `measure` and `deadband` existed) was written under.
V1_DEFAULTS = {"measure": "soft", "deadband": 0.0}
#: What a version-1 or -2 file (before `confidence`, 2026-11-17) was written under.
V2_DEFAULTS = {"confidence": 0.0}


def _get(node, key, default=None):
    if node is None:
        return default
    try:
        return node.get(key, default)
    except AttributeError:
        return getattr(node, key, default)


def read_block(cfg) -> Optional[Dict[str, float]]:
    """The compute target's settings, or None when `mor.token.balancing` is not `compute`. Raises ValueError.

    `cfg` is the run's DictConfig or the plain dict preflight composes. The block must hold exactly KEYS, and may
    hold OPTIONAL's keys (absent = their value there).
    """
    mor = _get(cfg, "mor") or {}
    tok = _get(mor, "token") or {}
    if _get(tok, "balancing") != MODE:
        return None
    blk = _get(tok, "compute")
    if blk is None:
        raise ValueError("mor.token.balancing is compute but mor.token.compute is not set "
                         f"(it needs {list(KEYS)})")
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(blk):
            blk = OmegaConf.to_container(blk, resolve=True)
    except ImportError:
        pass
    blk = dict(blk)
    extra, missing = sorted(set(blk) - set(KEYS) - set(OPTIONAL)), [k for k in KEYS if k not in blk]
    if extra or missing:
        raise ValueError(f"mor.token.compute must hold exactly {list(KEYS)}, and optionally {list(OPTIONAL)}: "
                         f"unknown {extra}, missing {missing}")
    for k, v in OPTIONAL.items():
        if blk.get(k) is None:
            blk[k] = v
    out = {}
    for k in NUMBERS:
        v = blk[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"mor.token.compute.{k} must be a finite number, got {v!r}")
        out[k] = float(v)
    if blk["measure"] not in MEASURES:
        raise ValueError(f"mor.token.compute.measure {blk['measure']!r} must be one of {list(MEASURES)}")
    out["measure"] = str(blk["measure"])
    if not 0.0 <= out["deadband"] < DEADBAND_MAX:
        raise ValueError(f"mor.token.compute.deadband {out['deadband']} must lie in [0, {DEADBAND_MAX}) passes "
                         f"(below half an image slot)")
    n_rec = int(_get(_get(cfg, "recursive") or {}, "num_recursion") or 0)
    if n_rec < 2:
        raise ValueError(f"recursive.num_recursion {n_rec}: a compute target needs at least two depths")
    if not 1.0 < out["target"] < float(n_rec):
        raise ValueError(f"mor.token.compute.target {out['target']} must lie strictly between 1 and "
                         f"{n_rec} passes")
    if out["gain"] <= 0:
        raise ValueError(f"mor.token.compute.gain {out['gain']} must be positive (the integral part)")
    if out["proportional"] < 0:
        raise ValueError(f"mor.token.compute.proportional {out['proportional']} must not be negative")
    if not 0.0 <= out["confidence"] <= CONFIDENCE_MAX:
        raise ValueError(f"mor.token.compute.confidence {out['confidence']} must lie in [0, {CONFIDENCE_MAX}] "
                         f"(a coefficient on the body positions' mean router entropy, in nats)")
    return out


def config_problems(cfg, n_gpus: int = 1) -> list:
    """What else a compute-target run needs, as a list of problems (empty = fine). `read_block` must pass first."""
    mor = _get(cfg, "mor") or {}
    tok = _get(mor, "token") or {}
    p = []
    if _get(cfg, "dataset") != "terramesh_multimodal":
        p.append(f"dataset {_get(cfg, 'dataset')!r}: the body mask comes from the per-update depth shares, "
                 "which run on terramesh_multimodal only")
    if not _get(mor, "enable") or _get(mor, "type") != "token":
        p.append("a token-choice MoR (mor.enable true, mor.type token)")
    if _get(mor, "rand_router"):
        p.append("a learned router: mor.rand_router is true, so there is no probability to price")
    if _get(tok, "router_func", "softmax") != "softmax":
        p.append(f"mor.token.router_func {_get(tok, 'router_func')!r}: the soft depth needs a softmax")
    if float(_get(tok, "alpha", 1.0)) != 1.0:
        p.append(f"mor.token.alpha {_get(tok, 'alpha')}: must be 1.0 (CLAUDE.md; and the soft depth assumes it)")
    if int(_get(tok, "bal_warmup_step", 0) or 0) > 0:
        p.append("mor.token.bal_warmup_step > 0: it routes every token to the deepest depth and restarts on resume")
    if _get(cfg, "gradient_checkpointing"):
        p.append("gradient_checkpointing: the MoR layer would be recomputed, and its priced sum with it")
    if int(n_gpus) != 1:
        p.append(f"--gpus {n_gpus}: the price is computed on one process (it is not reduced across ranks)")
    return p


def _val(block: Dict[str, float], key: str):
    """A block's value for `key`, an OPTIONAL key's default when a hand-built block leaves it out."""
    return block[key] if key in block else OPTIONAL[key]


def initial_state(block: Dict[str, float]) -> Dict[str, float]:
    """The controller at step 0: the integral at price_init and no measured gap yet."""
    return {"lam": float(block["price_init"]), "e_last": 0.0}


def price(block: Dict[str, float], state: Dict[str, float]) -> float:
    """The price the next update trains under."""
    return float(block["proportional"]) * float(state["e_last"]) + float(state["lam"])


def gap(block: Dict[str, float], measured: float) -> float:
    """The gap the controller acts on: measured - target, or 0 inside the deadband (|gap| <= deadband)."""
    e = float(measured) - float(block["target"])
    return 0.0 if abs(e) <= float(block["deadband"]) else e


def step(block: Dict[str, float], state: Dict[str, float], measured: float) -> Dict[str, float]:
    """The controller after an update whose body positions averaged `measured` passes, in the block's measure (the
    soft mean, or the hard top-1 mean). Pure: returns a new state.

    THE RULE, in one place (W34s tests it, with a sign update, a stuck price and an ignored band as the controls).
    """
    e = gap(block, measured)
    return {"lam": float(state["lam"]) + float(block["gain"]) * e, "e_last": e}


def hard_body_mean(counts) -> float:
    """The update's hard (top-1) mean depth over its body positions, from per-class depth counts (classes x depths:
    "none", the modalities, "delim", "pad"; util/trainer_pt.py's per-update depth shares). Depths are passes 1..Nr."""
    import numpy as np
    c = counts.detach().cpu().numpy() if hasattr(counts, "detach") else counts
    c = np.asarray(c, dtype=np.float64)
    body = c[1:c.shape[0] - 2]
    n = float(body.sum())
    if not n > 0:
        raise ValueError("no body positions were counted in this update")
    return float((body * np.arange(1, c.shape[1] + 1)).sum() / n)


def write_state(ckpt_dir, block: Dict[str, float], state: Dict[str, float], global_step: int,
                origin: str) -> Path:
    """<ckpt_dir>/compute_target.json: the settings, the controller and the step it belongs to."""
    p = Path(ckpt_dir) / FILE
    doc = {"version": 3, **{k: float(_val(block, k)) for k in NUMBERS}, "measure": str(block["measure"]),
           "lam": float(state["lam"]), "e_last": float(state["e_last"]), "price_next": price(block, state),
           "global_step": int(global_step), "origin": str(origin)}
    p.write_text(json.dumps(doc, indent=1) + "\n")
    return p


def checkpoint_step(ckpt_dir) -> Optional[int]:
    ts = Path(ckpt_dir) / "trainer_state.json"
    try:
        return int(json.loads(ts.read_text())["global_step"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def read_state(ckpt_dir, block: Dict[str, float]) -> Dict[str, float]:
    """The controller saved with a checkpoint. Raises RuntimeError with the reason.

    Refused: no file (the price would restart silently); settings other than the run's (a resume would change the
    arm); a file written for another step than the checkpoint's trainer_state.json (copied from elsewhere); a
    non-finite controller.
    """
    p = Path(ckpt_dir) / FILE
    if not p.is_file():
        raise RuntimeError(f"{p} is missing: a compute-target resume would restart the price at its initial "
                           f"value. A run's own checkpoints always carry it; a replay's copy gets it from its "
                           f"launcher (eo.train.compute_target.write_state)")
    doc = json.loads(p.read_text())
    if int(doc.get("version", 1)) == 1:          # written before `measure` and `deadband`: soft, no band
        doc = {**V1_DEFAULTS, **doc}
    if int(doc.get("version", 1)) <= 2:          # written before `confidence` (2026-11-17): none
        doc = {**V2_DEFAULTS, **doc}

    def differs(k):
        a, b = doc.get(k), _val(block, k)
        if a is None:
            return True
        return a != b if k == "measure" else float(a) != float(b)
    bad = {k: (doc.get(k), _val(block, k)) for k in IDENTITY if differs(k)}
    if bad:
        raise RuntimeError(f"{p} was written under other settings: "
                           + ", ".join(f"{k} {a!r} -> {b!r}" for k, (a, b) in bad.items()))
    want = checkpoint_step(ckpt_dir)
    if want is None or int(doc.get("global_step", -1)) != want:
        raise RuntimeError(f"{p} is for step {doc.get('global_step')}, the checkpoint's trainer_state.json says "
                           f"{want}")
    st = {"lam": doc.get("lam"), "e_last": doc.get("e_last")}
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in st.values()):
        raise RuntimeError(f"{p}: controller {st} is not finite")
    return {k: float(v) for k, v in st.items()}
