# eo/train/depth_table.py
"""Paper B, criterion 3's instrument: training under a fixed per-modality depth table (B1 Step 8, 2026-11-18).

WHAT IT DOES. `mor.token.depth_table` replaces a RANDOM router's depth draw with one drawn from a table. Each position
takes its depth from the row of the modality of the token it PREDICTS (the per-update depth shares' label_mod frame,
gate W31), over the Coords-free cell's five slots, the two S1 modalities one slot:

    mor:
      rand_router: true
      token:
        depth_table:
          rows: {S2L2A: [p1, p2, p3], S1: [..], DEM: [..], NDVI: [..], LULC: [..]}   # probabilities of depths 1..Nr
          gate: draw     # draw: the top of the layer's own Nr uniforms, the random router's gate; one: alpha
          seed: 0        # the depth draw's seed: the no-op pair differs only here

Positions predicting BO, EO or padding, padding itself and the last position take depth Nr, as every learned run put
them (B0 finding 4); they are not part of any slot. Uniform rows are the random router's depth distribution, one-hot
rows a fixed depth; Step 9's rows are learned tables, their permutations, a histogram-matched table and uniform depth 2.
Nothing else in the forward changes: with `gate: draw` the layer still draws the random router's Nr uniforms from the
global generator, so the random stream, the gate and everything downstream of the depth are the random router's.

WHY THE GATE IS A CHOICE (worklog §4, item 77). D990-s42 trained at uniform depth 2 under a committed learned router,
whose gate at alpha = 1 is about 1, while the random router scales each exit by the top of three uniforms (mean 0.75).
`gate: one` gives every exit alpha, so Step 9 can tell a fixed depth from a clean gate.

THE KEY AND THE DRAW. The layer (model/mor_model/token_choice_router.py) builds the table from its own config. Whoever
runs the forward hands it, per batch, each position's slot (`slot_key`) and the draw's seed (`draw_seed`): the trainer
on every forward, training and evaluation (util/trainer_pt.py), and eo/eval/teacher_forced.run_pass for E1. A table
model's forward without them raises: it never falls back to the random draw. One uniform per position comes from a CPU
generator seeded per forward, from (seed, "train", global step, micro-batch, rank) in training and (seed, "eval", batch
index, rank) in evaluation. So the draw is the same on any card, a resume repeats it without saved state, every
evaluation draws the same depths, and two tables at one seed share their uniforms position by position (Step 9's rows
are paired draws). A position's depth is the first d with u < CDF_row[d].

Gate W37 in eo/scripts/verify_phase3.py; preflight checks the block (`read_block`, `config_problems`). Runs in `.venv`.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional

import torch

SLOTS = ("S2L2A", "S1", "DEM", "NDVI", "LULC")
#: Each image modality's slot. Coords has none: a Coords position is refused (the cell is Coords-free).
SLOT_OF_MODALITY = {"S2L2A": 0, "S1GRD": 1, "S1RTC": 1, "DEM": 2, "NDVI": 3, "LULC": 4}
GATES = ("draw", "one")
KEYS = ("rows", "gate", "seed")
ROW_TOL = 1e-6


def _get(node, key, default=None):
    if node is None:
        return default
    try:
        return node.get(key, default)
    except AttributeError:
        return getattr(node, key, default)


def _plain(blk):
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(blk):
            return OmegaConf.to_container(blk, resolve=True)
    except ImportError:
        pass
    return blk


def read_block(cfg) -> Optional[Dict]:
    """The table's settings, or None when `mor.token.depth_table` is absent. Raises ValueError.

    Returns {"rows": {slot: [p_1..p_Nr]}, "gate": str, "seed": int, "n": Nr}: every slot present, each row a probability
    vector over the run's depths. `cfg` is the run's DictConfig or the plain dict preflight composes.
    """
    tok = _get(_get(cfg, "mor") or {}, "token") or {}
    blk = _get(tok, "depth_table")
    if blk is None:
        return None
    blk = _plain(blk)
    if not isinstance(blk, dict):
        raise ValueError(f"mor.token.depth_table must be a mapping with {list(KEYS)}, got {type(blk).__name__}")
    extra, missing = sorted(set(blk) - set(KEYS)), [k for k in KEYS if k not in blk]
    if extra or missing:
        raise ValueError(f"mor.token.depth_table must hold exactly {list(KEYS)}: unknown {extra}, missing {missing}")
    n = int(_get(_get(cfg, "recursive") or {}, "num_recursion") or 0)
    if n < 2:
        raise ValueError(f"recursive.num_recursion {n}: a depth table needs at least two depths")
    rows = _plain(blk["rows"])
    if not isinstance(rows, dict):
        raise ValueError(f"mor.token.depth_table.rows must map the slots {list(SLOTS)} to rows")
    extra, missing = sorted(set(rows) - set(SLOTS)), [s for s in SLOTS if s not in rows]
    if extra or missing:
        raise ValueError(f"mor.token.depth_table.rows must hold exactly the slots {list(SLOTS)} "
                         f"(S1 covers S1GRD and S1RTC): unknown {extra}, missing {missing}")
    out_rows = {}
    for s in SLOTS:
        r = _plain(rows[s])
        if not isinstance(r, (list, tuple)) or len(r) != n:
            raise ValueError(f"mor.token.depth_table.rows.{s} must list {n} probabilities (depths 1..{n}), got {r!r}")
        vals = []
        for v in r:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) < 0:
                raise ValueError(f"mor.token.depth_table.rows.{s}: {r!r} must hold finite probabilities >= 0")
            vals.append(float(v))
        if abs(sum(vals) - 1.0) > ROW_TOL:
            raise ValueError(f"mor.token.depth_table.rows.{s}: {r!r} sums to {sum(vals)!r}, not 1")
        out_rows[s] = vals
    if blk["gate"] not in GATES:
        raise ValueError(f"mor.token.depth_table.gate {blk['gate']!r} must be one of {list(GATES)}")
    seed = blk["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise ValueError(f"mor.token.depth_table.seed {seed!r} must be an integer in [0, 2**31)")
    return {"rows": out_rows, "gate": str(blk["gate"]), "seed": int(seed), "n": n}


def config_problems(cfg, n_gpus: int = 1) -> list:
    """What else a table run needs, as a list of problems (empty = fine). `read_block` must pass first."""
    mor = _get(cfg, "mor") or {}
    tok = _get(mor, "token") or {}
    p = []
    if _get(cfg, "dataset") != "terramesh_multimodal":
        p.append(f"dataset {_get(cfg, 'dataset')!r}: the slots come from EO modality ids (terramesh_multimodal only)")
    if not _get(mor, "enable") or _get(mor, "type") != "token":
        p.append("a token-choice MoR (mor.enable true, mor.type token)")
    if not _get(mor, "rand_router"):
        p.append("a random router (mor.rand_router true): the table replaces the random draw; a learned router has "
                 "its own choice to make")
    if int(_get(tok, "bal_warmup_step", 0) or 0) > 0:
        p.append("mor.token.bal_warmup_step > 0: it routes every token to the deepest depth and restarts on resume")
    active = list(_get(_get(cfg, "multimodal") or {}, "active_modalities") or [])
    if not active:
        p.append("multimodal.active_modalities is empty: the table needs the run's modalities")
    bad = [m for m in active if m not in SLOT_OF_MODALITY]
    if bad:
        p.append(f"active modalities {bad} have no slot (the table covers {sorted(SLOT_OF_MODALITY)}; "
                 "the cell is Coords-free)")
    return p


def cdf(table: Dict) -> torch.Tensor:
    """(len(SLOTS), n) float64 cumulative rows, the last column exactly 1."""
    c = torch.tensor([table["rows"][s] for s in SLOTS], dtype=torch.float64).cumsum(-1)
    c[:, -1] = 1.0
    return c


def slot_key(modality_ids: torch.Tensor) -> torch.Tensor:
    """(B, L) long: the slot of the token each position predicts (position t, token t + 1), -1 where that is not an
    image token (BO, EO, padding) and at the last position. Raises on a modality outside the table (Coords)."""
    from eo.data import eo_vocab
    mids = modality_ids.long()
    lut = torch.full((max(eo_vocab.ID_TO_MODALITY) + 1,), -2, dtype=torch.long)
    lut[0] = -1
    for name, s in SLOT_OF_MODALITY.items():
        lut[eo_vocab.MODALITY_TO_ID[name]] = s
    if int(mids.min()) < 0 or int(mids.max()) >= lut.numel():
        raise ValueError(f"modality ids outside 0..{lut.numel() - 1}: {sorted(torch.unique(mids).tolist())}")
    pred = lut.to(mids.device)[mids[:, 1:]]
    if bool((pred == -2).any()):
        raise ValueError(f"a position predicts a modality with no slot: "
                         f"{sorted(eo_vocab.ID_TO_MODALITY[int(i)] for i in torch.unique(mids[:, 1:][pred == -2]))}")
    out = torch.full_like(mids, -1)
    out[:, :-1] = pred
    return out


def draw_seed(seed: int, *parts) -> int:
    """A 63-bit generator seed from the table's seed and the forward's identity, the same in every process."""
    h = hashlib.blake2b(repr((int(seed),) + tuple(parts)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 63) - 1)


def draw_depths(slot: torch.Tensor, seed: int, table_cdf: torch.Tensor) -> torch.Tensor:
    """(B, L) long depth index 0..n-1 for each position: from its slot's row with one uniform per position from a CPU
    generator seeded with `seed` (the same values on any card), n-1 where the slot is -1."""
    n = table_cdf.shape[-1]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    u = torch.rand(tuple(slot.shape), generator=g, dtype=torch.float64).to(slot.device)
    c = table_cdf.to(slot.device)[slot.clamp(min=0)]                 # (B, L, n)
    d = (u.unsqueeze(-1) >= c).sum(-1).clamp_max(n - 1)              # the first d with u < CDF[d]
    return torch.where(slot >= 0, d, torch.full_like(d, n - 1))


# ---- The record a checkpoint carries (written at every save; the transplant launcher writes its own) -----------------
FILE = "depth_table.json"
TRANSPLANT = "transplant.json"


def identity(table: Dict) -> Dict:
    """What defines a table run's draws: its rows, gate and seed (a resume that changes any of them is refused)."""
    return {"rows": {s: [float(v) for v in table["rows"][s]] for s in SLOTS}, "gate": table["gate"],
            "seed": int(table["seed"])}


def write_record(ckpt_dir, table: Dict, global_step: int, origin: str) -> None:
    import json
    from pathlib import Path
    d = Path(ckpt_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / FILE).write_text(json.dumps({"version": 1, "table": identity(table), "global_step": int(global_step),
                                      "origin": origin}, indent=1) + "\n")


def resume_problem(ckpt_dir, table: Optional[Dict]) -> Optional[str]:
    """Why resuming `ckpt_dir` under `table` (None: no table) would mix draws, or None when it is consistent.

    A table run's checkpoint carries depth_table.json, which must name this table. A random-router checkpoint without
    one may start a table run only through the transplant launcher, whose transplant.json must name this table."""
    import json
    from pathlib import Path
    d = Path(ckpt_dir)
    rec = json.loads((d / FILE).read_text()) if (d / FILE).is_file() else None
    tr = json.loads((d / TRANSPLANT).read_text()) if (d / TRANSPLANT).is_file() else None
    if table is None:
        if rec is not None:
            return f"{d.name} carries {FILE} (a table run's), but this run has no mor.token.depth_table"
        return None
    want = identity(table)
    if rec is not None:
        if rec.get("table") != want:
            return f"{d.name}'s {FILE} names another table ({rec.get('table')}), not this run's ({want})"
        return None
    if tr is None:
        return (f"{d.name} has no {FILE} and no {TRANSPLANT}: a random-router checkpoint starts a table run only "
                f"through the transplant launcher (eo/train/transplant.py)")
    if tr.get("table") != want:
        return f"{d.name}'s {TRANSPLANT} names another table ({tr.get('table')}), not this run's ({want})"
    return None
