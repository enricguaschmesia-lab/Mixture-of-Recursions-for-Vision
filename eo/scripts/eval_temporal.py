#!/usr/bin/env python
"""The temporal stage's readings (A1 Step 7). `.venv`; GPU for `run` (the GTX Titan X, fp32), CPU otherwise.

Reads a model on val_clean's dev locations through eo/eval/temporal.py, in the layouts of
eo/eval/temporal_layouts_<tag>.json, so that the pilot (Step 9) and the final run (Steps 10-12) are
read with one code path, and every reading has its step-0 value from the extended base.

    layouts   write the layout file from the builders below (once per tag; then it is frozen)
    run       score units for some readings on one model; one file per unit, skipped when present
    report    read the units into <out>/report.json, paired with Step 0's stored units
    cache     KV-cached decoding against full passes at fp16 and fp32 on the longest task prompt (gap
              fill), for T360 without date blocks and T360-T with them, and the time of a batch of
              decodes: the decode precision and Step 12's budget rest on it (Step 11 reruns it)

THE MODEL. T360-T at 87,664 ids from --temporal-config (eo_terramesh/arm_t360t_vanilla), loaded as
eval_forgetting.py --model temporal loads it: without --checkpoint, the extended base through
pretrain_init's `continue` hook (T360-T at step 0, which reads as T360 on any input without a date
block: finding 24), else a T360-T checkpoint, strictly.

THE READINGS (units are (layout, target, options); `READINGS` below):
    repro       Step 0's own layouts, no date blocks: T360's stored per-location units, its block
                profile and offset curve, reproduced through this code (on the base: finding 24)
    baselines   T360's candidate layouts for forecast and gap fill, no date blocks, chosen on dev
    tasks       T360-T's four task layouts (Step 6's defaults); tasks_nodate: the same, no date blocks
    probe       the left-out season with the target's date moved (the probe's settings, from the file)
    wrong_all   the same with every date block moved
    bars        two-date layouts (the quarter-pass bar)
    far         the block profile along a full chronological sequence, and the same without date 0
    offset      the offset curve on the cross-date pair, nothing in the gap, no date blocks
    lattice     the cross-date pair with a date block between, and one before each block
    context     statics and the same modality, then with the other two at the next date added
    date        the date task: CE of Y and D, top-1 year, circular day error
    probe_null  the same with the date moved one bin: the probe's null (on untrained rows, their noise)
    readings_nodate  the profile, bars and context layouts without their date blocks (the base: their cost)
    pilot       everything a checkpoint is read with at a full pass: tasks probe bars far offset lattice
                context date
    quarter     what a quarter pass is read with: the bars

LOCATIONS: Step 0's 256 dev locations and target dates (eo/eval/temporal.step0_locations). No test
location is ever built: TemporalLocations refuses them.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$(.venv/bin/python -m eo.train.gpu titanx) \\
      .venv/bin/python -X faulthandler -u eo/scripts/eval_temporal.py run --tag t360t_base --readings repro
    .venv/bin/python eo/scripts/eval_temporal.py report --tag t360t_base

Outputs: /data/enric/reports/temporal/step7/<tag>/<checkpoint or extended_base>/units/*.npz, run.json,
report.json. Runs in `.venv`; never imports terratorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from util.env import load_dotenv  # noqa: E402

load_dotenv()        # HF_HOME and HF_HUB_OFFLINE, as pretrain.py reads them; the shell's values win

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.temporal_dataset import DATE, GROUP_MODALITIES, Block  # noqa: E402
from eo.eval import temporal as ET  # noqa: E402

VAL = Path("/data/enric/data/TerraMesh/val_clean")
OUT = Path("/data/enric/reports/temporal/step7")
STEP0 = Path("/data/enric/reports/temporal")
T360_CKPT = "/data/enric/runs/pretrain/phase3/vanilla360t_20261026_040535/checkpoint-17441"
T360_ARM = "eo_terramesh/arm_t360_vanilla"
TG = GROUP_MODALITIES                      # S2L2A, S1GRD, NDVI
REGISTRY = ("S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords")   # eo_vocab.MODALITIES' order
BLOCK = 198
LONG_K = tuple(range(1, 15))               # t360_context.LONG_K: the target at 198 k
LENGTH_K = (1, 2, 3, 4, 5, 6, 7, 8, 11)    # t360_context.LENGTH_K


# =========================================================================== builders (write v1 once)
def near(t: int) -> int:
    """t360_context.near: a location's date next to t."""
    return t - 1 if t > 0 else 1


def others(t: int) -> List[int]:
    return [u for u in range(4) if u != t]


def xs(m: str) -> List[str]:
    """The other two dynamic modalities, in (S2L2A, S1GRD, NDVI) order."""
    return [k for k in TG if k != m]


def s0_statics(t: int) -> List[Tuple[str, int]]:
    """t360_context.statics: DEM, LULC, Coords, all from the near date's row."""
    n = near(t)
    return [("DEM", n), ("LULC", n), ("Coords", n)]


def statics_t(t: int) -> List[Tuple[str, int]]:
    """The statics T360-T's task prompts hold, in Step 0's order: DEM and Coords from date 0's row,
    LULC from the first context date's (eo/eval/temporal.task_blocks)."""
    return [("DEM", 0), ("LULC", others(t)[0]), ("Coords", 0)]


def _s0(m: str, t: int) -> Dict[str, List[Tuple[str, int]]]:
    """Step 0's context layouts for target m at t (t360_context.layouts, one_other, length, pairs,
    coords; t360_best_layouts.LAYOUTS), the target excluded."""
    st, o = s0_statics(t), others(t)
    dyn = [(k, u) for u in o for k in TG]
    out = {"none": [], "statics": st,
           "same_date": sorted([(k, t) for k in ("S2L2A", "S1GRD", "DEM", "NDVI", "LULC", "Coords") if k != m],
                               key=lambda b: REGISTRY.index(b[0])),
           "same_mod_all": [(m, u) for u in o],
           "all_other": st + dyn,
           "all_other_same": st + dyn + [(k, t) for k in xs(m)],
           "one_near": [(m, near(t))],
           "statics_same_mod": st + [(m, u) for u in o],
           "statics_near_all": st + [(k, near(t)) for k in TG],
           "statics_near_others_same_mod": st + [(k, near(t)) for k in xs(m)] + [(m, u) for u in o]}
    for k in range(3):
        out[f"one_other_{k}"] = [(m, o[k])]
    return out


def _fmt(blocks: List[Tuple[str, int]]) -> str:
    return " ".join(f"{m}@{t}" for m, t in blocks)


def _from_task(blocks) -> List[Tuple[str, int]]:
    return [(b.modality, b.t) for b in blocks]


def build_layouts(tag: str = "v1", created: str = "2026-11-09") -> Dict:
    """Every layout of the readings, written out per target and target date (eo/eval/temporal.py,
    LAYOUTS ARE DATA). Run once per tag; the file it writes is then the reference, and its hash is
    pinned in temporal.FROZEN_LAYOUTS."""
    from eo.eval import temporal as T

    L: Dict[str, Dict] = {}

    def add(name, about, role, targets, dates, fn, **extra):
        rule = dates["rule"]
        ts = [None] if rule == "none" else ([int(dates["t"])] if rule == "fixed" else list(range(4)))
        L[name] = {"about": about, "role": role, "targets": list(targets), "dates": dates,
                   "blocks": {m: {("*" if t is None else str(t)): _fmt(fn(m, t)) for t in ts} for m in targets},
                   **extra}

    step0 = {"rule": "step0", "j": "target"}
    pair = {"rule": "step0", "j": 0}            # t360_context.pair_seqs: date_of(i, 0) for both pairs
    s0_names = ("none", "statics", "same_date", "same_mod_all", "all_other", "all_other_same",
                "one_other_0", "one_other_1", "one_other_2")
    for nm in s0_names:
        add(f"s0.{nm}", f"Step 0's context layout `{nm}` (t360_context.layouts, no date blocks)", "step0", TG, step0,
            lambda m, t, nm=nm: _s0(m, t)[nm] + [(m, t)], step0_unit=f"t360_context/units/ctx_{{M}}_{nm}.npz")
    add("s0.one_near", "the same modality at the near date only (Step 0's one_other at near(t))", "step0", TG, step0,
        lambda m, t: _s0(m, t)["one_near"] + [(m, t)], step0_unit="select:one_other_near")
    for nm in ("statics_same_mod", "statics_near_all", "statics_near_others_same_mod"):
        add(f"s0.{nm}", f"Step 0's best-layout candidate `{nm}` (t360_best_layouts.LAYOUTS)", "step0", TG, step0,
            lambda m, t, nm=nm: _s0(m, t)[nm] + [(m, t)], step0_unit=f"t360_best_layouts/units/{nm}_{{M}}.npz")
    for k in LENGTH_K:
        add(f"s0.length_k{k}", f"S2L2A given the last {k} blocks of all_other_same (t360_context LENGTH)", "step0",
            ("S2L2A",), pair,
            lambda m, t, k=k: ([(x, u) for u in others(t) for x in TG] + [(x, t) for x in xs("S2L2A")])[-k:] + [(m, t)],
            step0_unit=f"t360_context/units/length_S2L2A_k{k}.npz")
    add("s0.pair_ndvi", "S2L2A -> NDVI at one date; with the target moved by 198(k-1): the long offsets", "step0",
        ("NDVI",), pair, lambda m, t: [("S2L2A", t), (m, t)], step0_unit="t360_context/units/long_S2L2A>NDVI_k{k}.npz")
    add("s0.pair_cross", "S2L2A at the near date -> S2L2A at t: the cross-date pair, and the offset curve", "step0",
        ("S2L2A",), pair, lambda m, t: [("S2L2A", near(t)), (m, t)],
        step0_unit="t360_context/units/long_S2L2A>S2L2A_other_date_k{k}.npz")
    add("s0.pair_ndvi_none", "NDVI alone (the pair's no-source reference)", "step0", ("NDVI",), pair,
        lambda m, t: [(m, t)], step0_unit="t360_context/units/none_S2L2A>NDVI.npz")
    add("s0.pair_cross_none", "S2L2A alone (the cross-date pair's no-source reference)", "step0", ("S2L2A",), pair,
        lambda m, t: [(m, t)], step0_unit="t360_context/units/none_S2L2A>S2L2A_other_date.npz")
    for where in ("between", "before"):
        add(f"s0.coords_ndvi_{where}", f"S2L2A -> NDVI with Coords {where} (t360_context COORDS)", "step0", ("NDVI",),
            pair, lambda m, t, w=where: ([("S2L2A", t), ("Coords", t)] if w == "between" else [("Coords", t), ("S2L2A", t)])
            + [(m, t)], step0_unit=f"t360_context/units/coords_S2L2A>NDVI_{where}.npz")
        add(f"s0.coords_cross_{where}", f"the cross-date pair with Coords {where} (t360_context COORDS)", "step0",
            ("S2L2A",), pair,
            lambda m, t, w=where: ([("S2L2A", near(t)), ("Coords", t)] if w == "between"
                                   else [("Coords", t), ("S2L2A", near(t))]) + [(m, t)],
            step0_unit=f"t360_context/units/coords_S2L2A>S2L2A_other_date_{where}.npz")
    add("s0.profile", "one full chronological sequence, no date blocks: Coords, DEM, LULC at date 0, then "
        "S2L2A, S1GRD, NDVI at dates 0-3 (t360_lattice.profile), every block scored", "step0", ("*",),
        {"rule": "none"}, lambda m, t: [("Coords", 0), ("DEM", 0), ("LULC", 0)] + [(k, u) for u in range(4) for k in TG],
        n=64, step0_unit="t360_lattice.json:profile")

    # T360-T's task layouts: Step 6's defaults, task_blocks, frozen here.
    for task, targets, dates in (("left_out", TG, step0), ("forecast", TG, {"rule": "fixed", "t": 3}),
                                 ("gap_fill", ("S2L2A", "NDVI"), step0), ("date", (DATE,), {"rule": "step0", "j": 0})):
        add(f"t.{task}", f"T360-T's {task} task: Step 6's defaults (eo/eval/temporal.task_blocks)", "task", targets,
            dates, lambda m, t, task=task: _from_task(T.task_blocks(task, t, m)), task=task)
    # The pilot's readings in T360-T's grammar: statics first, chronological groups each led by its date block,
    # S2L2A, S1GRD, NDVI in a group, LULC once, the target's group last with its date block first.
    add("t.bar2", "two dates: statics, the near date's group (all three, LULC), then the target's group",
        "reading", TG, step0, lambda m, t: [("Coords", 0), ("DEM", 0), ("Date", near(t))] + [(k, near(t)) for k in TG]
        + [("LULC", near(t)), ("Date", t), (m, t)])

    def ctx(m, t, full_near):
        out = [("Coords", 0), ("DEM", 0)]
        for i, u in enumerate(others(t)):
            out += [("Date", u)] + ([(k, u) for k in TG] if (full_near and u == near(t)) else [(m, u)])
            out += [("LULC", u)] if i == 0 else []
        return out + [("Date", t), (m, t)]
    add("t.ctx_a", "statics and the same modality at the other three dates, chronological groups, LULC in the first",
        "reading", TG, step0, lambda m, t: ctx(m, t, False))
    add("t.ctx_b", "t.ctx_a with the other two modalities added to the near date's group (finding 17)", "reading", TG,
        step0, lambda m, t: ctx(m, t, True))
    full = [("Coords", 0), ("DEM", 0), ("Date", 0), ("LULC", 0)] + [(k, 0) for k in TG] + \
           [b for u in (1, 2, 3) for b in [("Date", u)] + [(k, u) for k in TG]]
    add("t.profile", "s0.profile with a date block first in every group (LULC after date 0's): the block profile",
        "reading", ("*",), {"rule": "none"}, lambda m, t: full)
    add("t.profile_drop0", "t.profile without date 0's group (its date block, S2L2A, S1GRD, NDVI); LULC stays",
        "reading", ("*",), {"rule": "none"}, lambda m, t: [b for b in full if not (b[1] == 0 and b[0] in TG + ("Date",))])
    add("t.lattice", "the cross-date pair with the target's date block between (lattice positions)", "reading",
        ("S2L2A",), pair, lambda m, t: [("S2L2A", near(t)), ("Date", t), (m, t)])
    add("t.lattice_each", "the cross-date pair with a date block before each block", "reading", ("S2L2A",), pair,
        lambda m, t: [("Date", near(t)), ("S2L2A", near(t)), ("Date", t), (m, t)])

    # T360's candidate baselines (no date blocks), inside its working range: no more than about 1,600
    # tokens, same-modality blocks at offsets up to 792, and only blocks T360-T's prompt also holds.
    F = 3
    fc = {"same_mod": lambda m: [(m, u) for u in others(F)],
          "one_near": lambda m: [(m, near(F))],
          "statics_same_mod": lambda m: statics_t(F) + [(m, u) for u in others(F)],
          "statics_near_all": lambda m: statics_t(F) + [(k, near(F)) for k in TG],
          "statics_near_others_same_mod": lambda m: statics_t(F) + [(k, near(F)) for k in xs(m)] + [(m, u) for u in others(F)]}
    for nm, fn in fc.items():
        add(f"c.forecast.{nm}", f"T360's forecast candidate `{nm}`, no date blocks", "candidate", TG,
            {"rule": "fixed", "t": F}, lambda m, t, fn=fn: fn(m) + [(m, t)], task="forecast")
    gc = {"s1": lambda m, t: [("S1GRD", t)],
          "statics_s1": lambda m, t: statics_t(t) + [("S1GRD", t)],
          "same_mod_s1": lambda m, t: [(m, u) for u in others(t)] + [("S1GRD", t)],
          "statics_same_mod_s1": lambda m, t: statics_t(t) + [(m, u) for u in others(t)] + [("S1GRD", t)],
          "statics_s1_same_mod": lambda m, t: statics_t(t) + [("S1GRD", t)] + [(m, u) for u in others(t)],
          "statics_near_all_s1": lambda m, t: statics_t(t) + [(k, near(t)) for k in TG] + [("S1GRD", t)]}
    for nm, fn in gc.items():
        add(f"c.gap_fill.{nm}", f"T360's gap-fill candidate `{nm}`, no date blocks", "candidate", ("S2L2A", "NDVI"),
            step0, lambda m, t, fn=fn: fn(m, t) + [(m, t)], task="gap_fill")
    add("c.left_out.statics_same_mod_lulc_o1", "s0.statics_same_mod with T360-T's statics rows (LULC from the first "
        "context date, as its prompt holds), to measure what LULC's row changes", "candidate", TG, step0,
        lambda m, t: statics_t(t) + [(m, u) for u in others(t)] + [(m, t)], task="left_out")

    return {"tag": tag, "created": created,
            "about": "A1 Step 7: every layout of the temporal readings, per target and target date, the target block "
                     "last. Blocks are MODALITY@date-index of the location's four chronological dates; a date block "
                     "keeps its year. Written by eo/scripts/eval_temporal.py layouts; frozen by its hash "
                     "(eo/eval/temporal.FROZEN_LAYOUTS).",
            "locations": {"rule": "step0", "split": "temporal_rows_v1.json dev", "n": ET.STEP0_N, "seed": ET.STEP0_SEED,
                          "target_dates": "t = (i + j) % 4: location i of the n, j the target's index in (S2L2A, S1GRD, "
                                          "NDVI) or the layout's fixed j; forecast t = 3; the date task j = 0"},
            "probe": {"layout": "t.left_out", "which": "target", "day_shift": 36, "year_shift": 0, "null_day_shift": 1,
                      "about": "the target's day bin moved by 36 of 73 (about half a year), its year kept: a real date "
                               "in the same year; read paired per location, by the target date's quarter. The null moves "
                               "it by one bin (5 days), the same swap of date rows with almost no change of season: on "
                               "untrained rows it measures the rows' noise, on a trained model it should read about 0"},
            "baselines": {"left_out": {m: "s0.statics_same_mod" for m in TG},
                          "forecast": {m: "c.forecast.statics_same_mod" for m in TG},
                          "gap_fill": {m: "c.gap_fill.same_mod_s1" for m in ("S2L2A", "NDVI")},
                          "date": "prior (Step 12)",
                          "rule": "per task and target, the candidate with the lowest mean CE on the n dev locations; "
                                  "within 0.005 nats, the shorter layout",
                          "chosen": "2026-11-09, on the extended base without date blocks (it reads as T360 in this range: "
                                    "Step 0's units reproduced within 1e-4), Step 0's 256 dev locations and target dates. "
                                    "Left-out season: Step 0's bar, kept; LULC from the first context date's row instead "
                                    "of the near date's moves it by +0.004 [-0.000, +0.008] on S2L2A, 0.000 on the others. "
                                    "Forecast: statics_same_mod for all three (6.278, 7.189, 6.883). Gap fill: same_mod_s1 "
                                    "for both (6.151, 6.899); statics cost it 0.054 [0.04, 0.07]."},
            "layouts": L}


# =========================================================================== units
def unit_key(layout: str, target: str, with_dates: bool = True, move: Optional[Dict] = None, delta: int = 0) -> str:
    k = f"{layout}__{'all' if target == '*' else target}"
    k += "" if with_dates else "__nodate"
    if move:
        k += f"__move-{move['which']}{int(move['day_shift']):+d}y{int(move['year_shift']):+d}"
    return k + (f"__d{int(delta)}" if delta else "")


def U(layout, target, with_dates=True, move=None, delta=0) -> Dict:
    return {"layout": layout, "target": target, "with_dates": with_dates, "move": move, "delta": int(delta),
            "key": unit_key(layout, target, with_dates, move, delta)}


def readings(doc: Dict) -> Dict[str, List[Dict]]:
    """The units of each reading (module docstring)."""
    lay = doc["layouts"]
    by_role = lambda role: [n for n, e in lay.items() if e["role"] == role]       # noqa: E731
    R: Dict[str, List[Dict]] = {}
    R["repro"] = [U(n, m) for n in by_role("step0") for m in lay[n]["targets"]] + \
                 [U(n, m, delta=BLOCK * (k - 1)) for n in ("s0.pair_ndvi", "s0.pair_cross") for m in lay[n]["targets"]
                  for k in LONG_K[1:]]
    R["baselines"] = [U(n, m, with_dates=False) for n in by_role("candidate") for m in lay[n]["targets"]]
    tasks = [n for n in by_role("task")]
    R["tasks"] = [U(n, m) for n in tasks for m in lay[n]["targets"]]
    R["tasks_nodate"] = [U(n, m, with_dates=False) for n in tasks if n != "t.date" for m in lay[n]["targets"]]
    p = doc["probe"]
    mv = {"which": p["which"], "day_shift": p["day_shift"], "year_shift": p["year_shift"]}
    R["probe"] = [U(p["layout"], m, move=mv) for m in lay[p["layout"]]["targets"]]
    R["probe_null"] = [U(p["layout"], m, move={**mv, "day_shift": p["null_day_shift"]}) for m in lay[p["layout"]]["targets"]]
    R["wrong_all"] = [U(p["layout"], m, move={**mv, "which": "all"}) for m in lay[p["layout"]]["targets"]]
    R["bars"] = [U("t.bar2", m) for m in TG]
    R["far"] = [U("t.profile", "*"), U("t.profile_drop0", "*")]
    R["offset"] = [U("s0.pair_cross", "S2L2A", delta=BLOCK * (k - 1)) for k in LONG_K] + [U("s0.pair_cross_none", "S2L2A")]
    R["lattice"] = [U("s0.pair_cross", "S2L2A"), U("t.lattice", "S2L2A"), U("t.lattice_each", "S2L2A"),
                    U("s0.pair_cross_none", "S2L2A")]
    R["context"] = [U(n, m) for n in ("t.ctx_a", "t.ctx_b") for m in TG]
    R["date"] = [U("t.date", DATE)]
    pilot = []
    for name in ("tasks", "probe", "bars", "far", "offset", "lattice", "context", "date"):
        pilot += [u for u in R[name] if u["key"] not in {x["key"] for x in pilot}]
    R["pilot"] = pilot
    R["quarter"] = list(R["bars"])           # Step 9 reads only the two-date bars at a quarter pass
    # The readings' layouts without their date blocks: T360's row on T360-T's layouts, so that the date blocks'
    # own cost pairs per location (read on the base, step 0).
    R["readings_nodate"] = ([U(n, "*", with_dates=False) for n in ("t.profile", "t.profile_drop0")]
                            + [U("t.bar2", m, with_dates=False) for m in TG]
                            + [U(n, m, with_dates=False) for n in ("t.ctx_a", "t.ctx_b") for m in TG])
    return R


def unit_spec(doc: Dict, u: Dict, n: int) -> Dict:
    e = doc["layouts"][u["layout"]]
    return {"layout": u["layout"], "entry_sha256": ET.canonical_sha(e), "target": u["target"],
            "with_dates": u["with_dates"], "move": u["move"], "delta": u["delta"], "n": min(int(e.get("n", n)), int(n))}


def build_items(temporal, doc: Dict, u: Dict, locs: np.ndarray):
    """(items, kind, extra) for one unit: scored sequences for a target, whole ones for a profile."""
    e = doc["layouts"][u["layout"]]
    n = min(int(e.get("n", len(locs))), len(locs))
    items, ts_list, t_list = [], [], []
    for i in range(n):
        t = ET.target_date(e, i, u["target"])
        blocks = ET.layout_blocks(e, u["target"], t, with_dates=u["with_dates"])
        ts = ET.task_sequence(temporal, int(locs[i]), blocks)
        if u["move"]:
            ts = ET.move_dates(ts, u["move"]["which"], u["move"]["day_shift"], u["move"]["year_shift"])
        if u["delta"]:
            ts = ET.move_target(ts, u["delta"])
        ts_list.append(ts)
        t_list.append(-1 if t is None else t)
        items.append(ts.full if (e["dates"]["rule"] == "none" or u["target"] == DATE) else ET.scored(ts))
    kind = "blocks" if e["dates"]["rule"] == "none" else ("date" if u["target"] == DATE else "target")
    blocks = ET.layout_blocks(e, u["target"], None, u["with_dates"]) if kind == "blocks" else None
    return items, kind, {"t": np.asarray(t_list), "blocks": blocks, "ts": ts_list}


def batch_for(length: int, budget: int) -> int:
    """Sequences per batch at fp32 for a ~12 GB card: run_pass holds (B, L, V) logits and their
    log-softmax, so the batch shrinks with length. `budget` is tokens per batch."""
    return max(1, min(16, budget // max(1, int(length))))


def run_unit(model, temporal, doc, u, locs, dtype, device, budget) -> Dict[str, np.ndarray]:
    items, kind, ex = build_items(temporal, doc, u, locs)
    L = int(items[0]["input_ids"].shape[0])
    bs = batch_for(L, budget)
    if kind == "target":
        r = ET.score_target(model, items, u["target"], bs, dtype=dtype, device=device)
        out = {"ce": r["ce"], "top1": r["top1"], "n_tokens": r["n"]}
    elif kind == "blocks":
        out = {"block_ce": ET.score_blocks(model, items, ex["blocks"], bs, dtype=dtype, device=device),
               "block_names": np.asarray([f"{b.modality}@{b.t}" for b in ex["blocks"]])}
    else:
        out = ET.score_dates(model, items, bs, dtype=dtype, device=device)
    out.update(locations=np.asarray(locs[: len(items)]), t=ex["t"], length=np.asarray(L), batch=np.asarray(bs))
    return out


# =========================================================================== commands
def cmd_layouts(a) -> int:
    path = ET.layouts_path(a.layout_tag)
    if path.exists() and not a.force:
        sys.exit(f"{path} exists; a layout file is written once (--force only while it is unfrozen)")
    if ET.FROZEN_LAYOUTS.get(a.layout_tag):
        sys.exit(f"{a.layout_tag} is frozen: write a new tag")
    doc = build_layouts(a.layout_tag)
    bad = ET.layout_problems(doc)
    if bad:
        sys.exit(f"the builders made an invalid file: {bad[:5]}")
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {path}: {len(doc['layouts'])} layouts, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0


def _ckpt_name(checkpoint, model: str = "temporal") -> str:
    if model == "t360":
        return Path(T360_CKPT).name
    return "extended_base" if checkpoint is None else Path(checkpoint).name


def cmd_run(a) -> int:
    from eo.scripts.eval_forgetting import build_temporal
    doc = ET.load_layouts(a.layout_tag, allow_unfrozen=a.allow_unfrozen)
    R = readings(doc)
    want = [r for r in a.readings.split(",") if r]
    bad = [r for r in want if r not in R]
    if bad:
        sys.exit(f"unknown readings {bad}; known: {sorted(R)}")
    units, seen = [], set()
    for r in want:
        for u in R[r]:
            if u["key"] not in seen and (not a.only or any(s in u["key"] for s in a.only.split(","))):
                units.append(u)
                seen.add(u["key"])
    device = a.device
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    if device == "cuda":
        if torch.cuda.device_count() != 1:
            sys.exit("pin exactly one GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES by name)")
        if a.expect_gpu not in gpu:
            sys.exit(f"CUDA sees {gpu!r}, expected {a.expect_gpu!r}")
    if "TITAN V" in gpu:
        sys.exit("refusing the TITAN V: fp32 passes and long jobs stall there (CLAUDE.md)")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[a.precision]
    out = a.out_root / a.tag / _ckpt_name(a.checkpoint, a.model)
    (out / "units").mkdir(parents=True, exist_ok=True)
    meta_p = out / "run.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    for key, val in (("precision", a.precision), ("checkpoint", a.checkpoint), ("temporal_config", a.temporal_config),
                     ("n_locations", a.n), ("model", "T360" if a.model == "t360" else "T360-T")):
        if key in meta and meta[key] != val:
            sys.exit(f"{out} was scored with {key}={meta[key]!r}; this run has {val!r}: use another --tag")
    todo = []
    for u in units:
        p = out / "units" / f"{u['key']}.npz"
        if p.exists():
            have = json.loads(str(np.load(p)["spec"]))
            if have != unit_spec(doc, u, a.n):
                sys.exit(f"{p.name} was scored under another spec: {have} (remove it to rescore)")
            continue
        todo.append(u)
    print(f"{len(units)} units, {len(units) - len(todo)} on disk, {len(todo)} to score; {gpu}, {a.precision}", flush=True)
    if not todo:
        return 0
    t0 = time.time()
    if a.model == "t360":
        # T360 itself at its 87,556 ids: Step 0's own model, for layouts without date blocks only.
        from eo.generate import conditional as G
        if a.checkpoint is not None:
            sys.exit("--model t360 reads T360's checkpoint-17441; --checkpoint is for T360-T")
        dated = [u["key"] for u in todo if u["with_dates"] and any(
            b.modality == DATE for t in doc["layouts"][u["layout"]]["blocks"][u["target"]].values()
            for b in ET.parse_blocks(t))]
        if dated:
            sys.exit(f"T360 has no date rows: {dated[:3]}")
        model, cfg = G.build_model(T360_ARM, T360_CKPT, device=device)
        loaded = {"checkpoint": T360_CKPT, "arm_config": T360_ARM}
        want_vocab = 87_556
    else:
        model, cfg, loaded = build_temporal(a.temporal_config, a.checkpoint, device=device)
        want_vocab = 87_664
    if int(model.config.vocab_size) != want_vocab or next(model.parameters()).dtype != torch.float32:
        sys.exit(f"{a.model} at {want_vocab:,} ids with fp32 parameters expected")
    temporal, index = ET.dev_temporal(VAL)
    locs, groups, names = ET.step0_locations(index)
    locs, groups, names = locs[: a.n], groups[: a.n], names[: a.n]
    meta.update(model="T360" if a.model == "t360" else "T360-T", temporal_config=a.temporal_config, checkpoint=a.checkpoint, loaded=loaded, gpu=gpu,
                precision=a.precision, layouts=doc["tag"], layouts_sha256=doc["sha256"], root=str(VAL),
                n_locations=int(a.n), location_names_sha256=ET.canonical_sha(names),
                attn=str(getattr(model.config, "_attn_implementation", "?")),
                fork_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True).stdout.strip())
    meta_p.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"model ready in {time.time() - t0:.0f} s ({meta['attn']}); {len(locs)} locations", flush=True)
    for u in todo:
        t1 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        res = run_unit(model, temporal, doc, u, locs, dtype, device, a.budget)
        sec = time.time() - t1
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
        spec = unit_spec(doc, u, a.n)
        p = out / "units" / f"{u['key']}.npz"
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, spec=json.dumps(spec, sort_keys=True), seconds=sec, peak_gib=peak, gpu=gpu, **res)
        os.replace(tmp, p)
        what = (f"CE {res['ce'].mean():.4f}" if "ce" in res else
                f"blocks {res['block_ce'].mean(0).round(3).tolist()[:4]}..." if "block_ce" in res else
                f"CE Y {res['ce_y'].mean():.3f} D {res['ce_d'].mean():.3f}")
        print(f"  {u['key']}: {what}; L {int(res['length'])}, batch {int(res['batch'])}, {sec:.0f} s, "
              f"peak {peak:.1f} GiB", flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


# =========================================================================== report
REPRO_TOL = 1e-4        # per location, fp32: Step 0's control reproduced stored CE to 1.5e-6
PROFILE_TOL = 2e-4      # a block's mean over 64 locations from fp16 per-token CE
TIE = 0.005             # baselines: within this, the shorter layout


def load_units(out: Path) -> Dict[str, Dict]:
    units = {}
    for p in sorted((out / "units").glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["spec"] = json.loads(str(z["spec"]))
        units[p.stem] = d
    return units


def step0_stored(doc: Dict, u: Dict, t: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Step 0's stored per-location unit for a step0 unit of this file, in Step 0's location order."""
    e = doc["layouts"][u["spec"]["layout"]]
    ref, m = e.get("step0_unit"), u["spec"]["target"]
    if ref is None or ref.endswith(":profile"):
        return None
    if ref == "select:one_other_near":
        ones = [np.load(STEP0 / "t360_context/units" / f"ctx_{m}_one_other_{k}.npz") for k in range(3)]
        pick = [others(int(x)).index(near(int(x))) for x in t]
        return {key: np.array([ones[pick[i]][key][i] for i in range(len(t))]) for key in ("ce", "top1")}
    k = int(u["spec"]["delta"]) // BLOCK + 1
    z = np.load(STEP0 / ref.replace("{M}", m).replace("{k}", str(k)))
    return {"ce": z["ce"], "top1": z["top1"]}


def _ci(v, g) -> Dict[str, float]:
    return ET.group_ci(v, g)


def _r(x, nd=4):
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.{nd}g}")          # significant digits: differences near 1e-6 stay visible
    return x


def cmd_report(a) -> int:
    from eo.data.temporal_index import load_temporal_index
    doc = ET.load_layouts(a.layout_tag, allow_unfrozen=a.allow_unfrozen)
    out = a.out_root / a.tag / _ckpt_name(a.checkpoint, getattr(a, "model", "temporal"))
    units = load_units(out)
    if not units:
        sys.exit(f"no units under {out}")
    index = load_temporal_index(VAL)
    locs_all, groups_all, _ = ET.step0_locations(index)
    L = doc["layouts"]
    rep: Dict = {"tag": a.tag, "checkpoint": a.checkpoint, "run": json.loads((out / "run.json").read_text()),
                 "layouts_sha256": doc["sha256"], "n_units": len(units)}
    stale = [k for k, u in units.items() if u["spec"]["entry_sha256"] != ET.canonical_sha(L[u["spec"]["layout"]])]
    if stale:
        sys.exit(f"units scored under another layout entry: {stale[:5]}")

    def get(layout, target, with_dates=True, move=None, delta=0):
        return units.get(unit_key(layout, target, with_dates, move, delta))

    def groups_of(u):
        n = len(u["locations"])
        if not np.array_equal(u["locations"], locs_all[:n]):
            raise RuntimeError("a unit's locations are not Step 0's, in Step 0's order")
        return groups_all[:n]

    def day_bins(u):
        return np.array([index.date_of(int(index.locations[int(l)][int(t)]))[1] for l, t in zip(u["locations"], u["t"])])

    # ---- timing
    rep["seconds"] = {k: _r(float(u["seconds"]), 1) for k, u in units.items()}
    by_reading = {}
    for name, us in readings(doc).items():
        have = [units[u["key"]] for u in us if u["key"] in units]
        if have:
            by_reading[name] = {"units": len(have), "of": len(us), "seconds": round(sum(float(u["seconds"]) for u in have), 1),
                                "tokens": int(sum(int(u["length"]) * len(u["locations"]) for u in have))}
    rep["time_by_reading"] = by_reading

    # ---- Step 0 reproduced through this code (no date blocks)
    repro, worst = {}, 0.0
    for key, u in units.items():
        sp = u["spec"]
        if L[sp["layout"]]["role"] != "step0" or "ce" not in u or not sp["with_dates"] or sp["move"]:
            continue
        ref = step0_stored(doc, u, u["t"])
        if ref is None:
            continue
        n = len(u["ce"])
        d = u["ce"] - ref["ce"][:n]
        worst = max(worst, float(np.abs(d).max()))
        repro[key] = {"n": n, "ce": float(u["ce"].mean()), "stored": float(ref["ce"][:n].mean()),
                      "max_abs_diff": float(np.abs(d).max()), "mean_diff": float(d.mean()),
                      "top1_tokens_differing": int(np.abs(np.round((u["top1"] - ref["top1"][:n]) * 196)).sum())}
    prof = get("s0.profile", "*")
    prof_rep = None
    if prof is not None:
        stored = json.loads((STEP0 / "t360_lattice.json").read_text())["profile"]
        names = [str(x) for x in prof["block_names"]]
        sb = {b["block"]: b["ce"]["mean"] for b in stored["by_block"]}
        want = {"Coords@0": "Coords", "DEM@0": "DEM", "LULC@0": "LULC"}
        if len(prof["locations"]) == stored["n"]:
            diffs = {nm: float(prof["block_ce"][:, i].mean() - sb[want.get(nm, nm)]) for i, nm in enumerate(names)}
            prof_rep = {"n": stored["n"], "by_block": {nm: [float(prof["block_ce"][:, i].mean()), sb[want.get(nm, nm)]]
                                                       for i, nm in enumerate(names)},
                        "max_abs_diff": max(abs(v) for v in diffs.values())}
    if repro or prof_rep:
        rep["repro"] = {"tolerance": REPRO_TOL, "units": len(repro), "max_abs_diff": worst,
                        "passed": bool(worst <= REPRO_TOL and (prof_rep is None or prof_rep["max_abs_diff"] <= PROFILE_TOL)),
                        "profile_tolerance": PROFILE_TOL, "profile": prof_rep, "by_unit": repro}

    # ---- T360's candidate baselines (no date blocks), chosen on dev by the file's rule
    cands: Dict[Tuple[str, str], List] = {}
    for n_, e in L.items():
        if e["role"] != "candidate":
            continue
        for m in e["targets"]:
            u = get(n_, m, with_dates=False)
            if u is not None:
                cands.setdefault((e["task"], m), []).append((n_, u))
    base_rep = {}
    for (task, m), lst in sorted(cands.items()):
        g = groups_of(lst[0][1])
        rows = sorted(((float(u["ce"].mean()), int(u["length"]), n_, u) for n_, u in lst), key=lambda x: x[0])
        best = rows[0]
        close = [r for r in rows if r[0] - best[0] <= TIE]
        pick = min(close, key=lambda r: r[1])
        entry = {"candidates": {r[2]: {"tokens": r[1], "ce": _ci(r[3]["ce"], g), "top1": float(r[3]["top1"].mean())}
                                for r in rows}, "chosen": pick[2]}
        if len(rows) > 1:
            other = rows[1] if pick is rows[0] else rows[0]
            entry["chosen_minus_next"] = {"next": other[2], "diff": _ci(pick[3]["ce"] - other[3]["ce"], g)}
        if task == "left_out":
            s = get("s0.statics_same_mod", m)
            ref = np.load(STEP0 / f"t360_best_layouts/units/statics_same_mod_{m}.npz")["ce"]
            n = len(lst[0][1]["ce"])
            entry["lulc_o1_minus_step0_lulc_near"] = _ci(lst[0][1]["ce"] - ref[:n], g)
            if s is not None:
                entry["step0_statics_same_mod_here"] = float(s["ce"].mean())
        base_rep[f"{task}/{m}"] = entry
    if base_rep:
        rep["baselines"] = base_rep

    # ---- T360-T's tasks, with and without date blocks
    tasks = {}
    for n_ in ("t.left_out", "t.forecast", "t.gap_fill"):
        for m in L[n_]["targets"]:
            u = get(n_, m)
            if u is None:
                continue
            g = groups_of(u)
            cell = {"tokens": int(u["length"]), "ce": _ci(u["ce"], g), "top1": float(u["top1"].mean())}
            v = get(n_, m, with_dates=False)
            if v is not None:
                cell["nodate"] = {"tokens": int(v["length"]), "ce": _ci(v["ce"], g)}
                cell["dates_minus_nodate"] = _ci(u["ce"] - v["ce"], g)
            chosen = (doc.get("baselines") or {}).get(L[n_]["task"]) or {}
            b = get(chosen.get(m), m, with_dates=False) if chosen.get(m, "").startswith("c.") else None
            if L[n_]["task"] == "left_out":
                ref = np.load(STEP0 / f"t360_best_layouts/units/statics_same_mod_{m}.npz")["ce"][: len(u["ce"])]
                cell["minus_t360_best_step0"] = _ci(u["ce"] - ref, g)
            elif b is not None:
                cell["minus_t360_baseline"] = {"baseline": chosen[m], "diff": _ci(u["ce"] - b["ce"], g)}
            tasks[f"{L[n_]['task']}/{m}"] = cell
    if tasks:
        rep["tasks"] = tasks

    # ---- the quarter-pass bars: two dates, against T360's best on the same date pair (Step 0's units)
    bars = {}
    for m in TG:
        u = get("t.bar2", m)
        if u is None:
            continue
        g = groups_of(u)
        n = len(u["ce"])
        if m == "NDVI":
            ref = step0_stored(doc, {"spec": {"layout": "s0.one_near", "target": m, "delta": 0}}, u["t"])["ce"]
            refname = "s0.one_near"
        else:
            ref = np.load(STEP0 / f"t360_best_layouts/units/statics_near_all_{m}.npz")["ce"][:n]
            refname = "s0.statics_near_all"
        bars[m] = {"tokens": int(u["length"]), "ce": _ci(u["ce"], g), "t360_layout": refname,
                   "t360": float(ref.mean()), "minus_t360": _ci(u["ce"] - ref, g)}
        v = get("t.bar2", m, with_dates=False)
        if v is not None:
            bars[m]["nodate"] = _ci(v["ce"], g)
            bars[m]["dates_minus_nodate"] = _ci(u["ce"] - v["ce"], g)
    if bars:
        rep["bars_quarter_pass"] = bars

    # ---- the date probe and the wrong date: paired against the right date, by the target date's quarter
    p = doc["probe"]
    for name, which, shift in (("probe", p["which"], p["day_shift"]), ("probe_null", p["which"], p["null_day_shift"]),
                               ("wrong_all", "all", p["day_shift"])):
        mv = {"which": which, "day_shift": shift, "year_shift": p["year_shift"]}
        sec = {}
        for m in L[p["layout"]]["targets"]:
            u, w = get(p["layout"], m), get(p["layout"], m, move=mv)
            if u is None or w is None:
                continue
            g, q = groups_of(u), ET.quarter(day_bins(u))
            d = w["ce"] - u["ce"]
            sec[m] = {"moved_minus_right": _ci(d, g), "abs_max": float(np.abs(d).max()),
                      "by_quarter": {ET.QUARTERS[k]: {"n": int((q == k).sum()), "diff": _ci(d[q == k], g[q == k])}
                                     for k in range(4) if (q == k).sum() > 1}}
        if sec:
            rep[name] = {"settings": mv, "layout": p["layout"], "targets": sec}

    # ---- far attention: the block profile, and the last date without the oldest group
    fp, fd = get("t.profile", "*"), get("t.profile_drop0", "*")
    if fp is not None:
        g = groups_of(fp)
        names = [str(x) for x in fp["block_names"]]
        far = {"n": len(fp["locations"]), "by_block": {nm: _ci(fp["block_ce"][:, i], g) for i, nm in enumerate(names)}}
        if fd is not None:
            dn = [str(x) for x in fd["block_names"]]
            far["last_date_without_oldest_minus_with"] = {
                m: _ci(fd["block_ce"][:, dn.index(f"{m}@3")] - fp["block_ce"][:, names.index(f"{m}@3")], g) for m in TG}
        fn = get("t.profile", "*", with_dates=False)
        if fn is not None:
            nn = [str(x) for x in fn["block_names"]]
            far["dates_minus_nodate"] = {nm: _ci(fp["block_ce"][:, i] - fn["block_ce"][:, nn.index(nm)], g)
                                         for i, nm in enumerate(names) if nm in nn}
            img = [i for i, nm in enumerate(names) if not nm.startswith(("Date", "Coords"))]
            imn = [nn.index(names[i]) for i in img]
            far["image_mean"] = {"dates": float(fp["block_ce"][:, img].mean()), "nodate": float(fn["block_ce"][:, imn].mean())}
        rep["far"] = far

    # ---- the offset curve (no date blocks) and the cost of a date block on the cross-date pair
    adj, none = get("s0.pair_cross", "S2L2A"), get("s0.pair_cross_none", "S2L2A")
    if adj is not None:
        g = groups_of(adj)
        curve = {}
        for k in LONG_K:
            u = get("s0.pair_cross", "S2L2A", delta=BLOCK * (k - 1))
            if u is not None:
                curve[str(BLOCK * k)] = {"ce": _ci(u["ce"], g), "rise_vs_198": _ci(u["ce"] - adj["ce"], g)}
        rep["offset"] = {"pair": "S2L2A at the near date -> S2L2A at t", "by_offset": curve,
                         "none": _ci(none["ce"], g) if none is not None else None}
        if none is not None:
            gain = float((none["ce"] - adj["ce"]).mean())
            lat = {"adjacent": _ci(adj["ce"], g), "none": _ci(none["ce"], g), "gain": gain}
            for n_ in ("t.lattice", "t.lattice_each"):
                u = get(n_, "S2L2A")
                if u is not None:
                    rise = _ci(u["ce"] - adj["ce"], g)
                    lat[n_] = {"ce": _ci(u["ce"], g), "rise": rise, "share_of_gain": rise["mean"] / gain}
            rep["lattice"] = lat

    # ---- more context helps?
    ctxr = {}
    for m in TG:
        A, B = get("t.ctx_a", m), get("t.ctx_b", m)
        if A is not None and B is not None:
            g = groups_of(A)
            ctxr[m] = {"a": _ci(A["ce"], g), "b": _ci(B["ce"], g), "b_minus_a": _ci(B["ce"] - A["ce"], g),
                       "tokens": [int(A["length"]), int(B["length"])]}
            An, Bn = get("t.ctx_a", m, with_dates=False), get("t.ctx_b", m, with_dates=False)
            if An is not None and Bn is not None:
                ctxr[m]["nodate"] = {"a": _ci(An["ce"], g), "b": _ci(Bn["ce"], g), "b_minus_a": _ci(Bn["ce"] - An["ce"], g)}
    if ctxr:
        rep["more_context"] = ctxr

    # ---- the date task
    u = get("t.date", DATE)
    if u is not None:
        g, q = groups_of(u), ET.quarter(u["day"])
        dm = ET.date_measures(u)
        sec = {"ce_y": _ci(u["ce_y"], g), "ce_d": _ci(u["ce_d"], g), "mass_years": float(u["mass_years"].mean()),
               "mass_days": float(u["mass_days"].mean()), **{k: _ci(v, g) for k, v in dm.items()},
               "chance": {"top1_year": 1 / 12, "day_err_uniform_days": float(ET.BIN_DAYS * np.mean(
                   ET.circular_bins(np.arange(73)[:, None], np.arange(73)[None, :])))},
               "by_quarter": {ET.QUARTERS[k]: {"n": int((q == k).sum()), "ce_d": float(u["ce_d"][q == k].mean()),
                                               "day_err_mode": float(dm["day_err_mode"][q == k].mean())}
                              for k in range(4) if (q == k).sum()}}
        rep["date"] = sec

    path = out / "report.json"
    path.write_text(json.dumps(_r(rep, 6), indent=1) + "\n")
    print(f"wrote {path}")
    for key in ("repro", "time_by_reading"):
        if key in rep:
            print(key, json.dumps(_r({k: v for k, v in rep[key].items() if k != "by_unit"}, 3))[:1500])
    return 0


# =========================================================================== cache check and decode time


def cmd_cache(a) -> int:
    """KV-cached decoding against full passes at the task prompts' length, fp16 and fp32, on the GTX
    Titan X: T360 itself on the gap-fill prompt without date blocks (its default positions), and the
    extended base (or a T360-T checkpoint) on the same prompt with them (lattice positions), each over
    the whole 197-token continuation (body and EO); then the time of a batch of decodes."""
    from eo.eval.ben import CACHE_TOL, CACHE_ULPS
    from eo.generate import cached as GC
    from eo.generate import conditional as G
    from eo.scripts.eval_forgetting import build_temporal
    gpu = torch.cuda.get_device_name(0)
    if torch.cuda.device_count() != 1 or a.expect_gpu not in gpu:
        sys.exit(f"pin exactly one GPU, the {a.expect_gpu} (CUDA sees {gpu!r})")
    doc = ET.load_layouts(a.layout_tag, allow_unfrozen=a.allow_unfrozen)
    e = doc["layouts"]["t.gap_fill"]
    temporal, index = ET.dev_temporal(VAL)
    locs, _, _ = ET.step0_locations(index)
    out = a.out_root / a.tag / _ckpt_name(a.checkpoint) / "cache"
    out.mkdir(parents=True, exist_ok=True)

    def prompts(n, with_dates, offset=0):
        ts = [ET.task_sequence(temporal, int(locs[offset + i]),
                               ET.layout_blocks(e, "S2L2A", ET.target_date(e, offset + i, "S2L2A"), with_dates))
              for i in range(n)]
        ids = torch.stack([x.prompt["input_ids"] for x in ts]).cuda()
        pos = torch.stack([x.prompt["position_ids"] for x in ts]).cuda()
        s, end = ts[0].target[0] + 1, ts[0].target[1]
        end = end if a.steps is None else min(end, s + a.steps)
        cont = torch.stack([x.full["input_ids"][s:end] for x in ts]).cuda()
        return ids, pos, cont

    rep = {"gpu": gpu, "layout": "t.gap_fill, target S2L2A", "checks": {}, "timing": {}}
    rule = {"fp32": CACHE_TOL["none"], "fp16": CACHE_TOL["fp16"], "ulps": CACHE_ULPS}
    rep["rule"] = (f"fp32: max |cached - full| <= {rule['fp32']:g}; fp16: <= {rule['fp16']:g} or <= {CACHE_ULPS} fp16 "
                   f"steps at the largest |logit| (eo/eval/ben.py); every logit finite")
    models = [("t360", None), ("t360t", a.checkpoint)]
    for tag, ck in models:
        if tag == "t360":
            model, _ = G.build_model(T360_ARM, T360_CKPT)
            ids, pos, cont = prompts(a.batch, with_dates=False)
            kw = {"position_ids": None}
        else:
            model, _, _ = build_temporal(a.temporal_config, ck, device="cuda")
            ids, pos, cont = prompts(a.batch, with_dates=True)
            kw = {"position_ids": pos}
        for prec, dt in (("fp32", None), ("fp16", torch.float16)):
            t0 = time.time()
            r = GC.cache_check(model, ids, cont, autocast=dt, **kw)
            r["seconds"] = round(time.time() - t0, 1)
            lim = rule[prec]
            r["passed"] = bool(r["finite"] and (r["max_abs_logit_diff"] <= lim or (prec == "fp16" and r["max_ulp_diff"] <= CACHE_ULPS)))
            rep["checks"][f"{tag}/{prec}"] = {k: v for k, v in r.items() if k != "per_step"}
            rep["checks"][f"{tag}/{prec}"]["per_step_max"] = max(r["per_step"]) if r["finite"] else None
            print(f"{tag} {prec}: prompt {r['prompt_length']}, {r['steps']} steps x {r['batch']}: max |d| "
                  f"{r['max_abs_logit_diff']:.2e} ({r['max_ulp_diff']:.1f} steps at |logit| {r['max_abs_logit']:.1f}), "
                  f"finite {r['finite']}, passed {r['passed']}, {r['seconds']} s", flush=True)
        # Controls, 8 steps each: the lattice prompt decoded on default positions; any prompt decoded one position off.
        c_kw = {"position_ids": pos}
        ctl = GC.cache_check(model, ids, cont[:, :8], decode_offset=1, **c_kw)
        rep["checks"][f"{tag}/control_offset_1"] = {k: v for k, v in ctl.items() if k != "per_step"}
        print(f"{tag} control, decoded one position off: {ctl['max_abs_logit_diff']:.2f}", flush=True)
        if tag == "t360t":
            ctl = GC.cache_check(model, ids, cont[:, :8], decode_positions=False, **c_kw)
            rep["checks"][f"{tag}/control_default_positions"] = {k: v for k, v in ctl.items() if k != "per_step"}
            print(f"{tag} control, default positions behind date blocks: {ctl['max_abs_logit_diff']:.2f}", flush=True)
            # Decode time: a batch of whole target blocks, both samplers' settings as Step 12's (t1).
            bids, bpos, _ = prompts(a.time_batch, with_dates=True)
            for prec, dt in (("fp32", None), ("fp16", torch.float16)):
                torch.cuda.synchronize()
                t0 = time.time()
                ctx = torch.autocast("cuda", dtype=dt) if dt is not None else GC._NullCtx()
                with ctx:
                    x, mass = GC.sample(model, bids, "S2L2A", temperature=1.0, top_p=1.0, seed=1000, position_ids=bpos)
                torch.cuda.synchronize()
                sec = time.time() - t0
                rep["timing"][prec] = {"batch": a.time_batch, "prompt_length": int(bids.shape[1]), "tokens": int(x.shape[1]),
                                       "seconds": round(sec, 1), "seconds_per_block": round(sec / a.time_batch, 2),
                                       "slot_mass": float(mass.mean()), "finite": bool(torch.isfinite(mass).all())}
                print(f"decode {prec}: {a.time_batch} blocks of {x.shape[1]} tokens after {bids.shape[1]}-token prompts in "
                      f"{sec:.1f} s ({sec / a.time_batch:.2f} s a block), slot mass {float(mass.mean()):.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    path = out / "cache_check.json"
    path.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"wrote {path}")
    return 0 if all(v.get("passed", True) for v in rep["checks"].values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("layouts")
    lp.add_argument("--layout-tag", default=ET.LAYOUTS_TAG)
    lp.add_argument("--force", action="store_true")
    for name in ("run", "report", "cache"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True, help="the model's reading, e.g. t360t_base")
        p.add_argument("--checkpoint", default=None, help="a T360-T checkpoint; none: the extended base")
        p.add_argument("--out-root", type=Path, default=OUT)
        p.add_argument("--layout-tag", default=ET.LAYOUTS_TAG)
        p.add_argument("--allow-unfrozen", action="store_true", help="while the layout file is being drawn up")
        p.add_argument("--n", type=int, default=ET.STEP0_N, help="Step 0's first n locations (smoke tests only)")
    sub.choices["report"].add_argument("--model", choices=("temporal", "t360"), default="temporal")
    rp = sub.choices["run"]
    rp.add_argument("--readings", required=True)
    rp.add_argument("--only", default="", help="comma-separated substrings of unit keys (development)")
    rp.add_argument("--temporal-config", default="eo_terramesh/arm_t360t_vanilla")
    rp.add_argument("--model", choices=("temporal", "t360"), default="temporal",
                    help="temporal: T360-T (default); t360: T360 itself at 87,556 ids, layouts without date blocks only")
    rp.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    rp.add_argument("--device", default="cuda")
    rp.add_argument("--expect-gpu", default="TITAN X")
    rp.add_argument("--budget", type=int, default=2800, help="tokens per batch")
    cp = sub.choices["cache"]
    cp.add_argument("--temporal-config", default="eo_terramesh/arm_t360t_vanilla")
    cp.add_argument("--expect-gpu", default="TITAN X")
    cp.add_argument("--batch", type=int, default=2, help="prompts per cache check")
    cp.add_argument("--time-batch", type=int, default=8, help="prompts per timed decode")
    cp.add_argument("--steps", type=int, default=None, help="cached steps per check (default: the whole target, 197)")
    a = ap.parse_args()
    if a.cmd == "cache":
        return cmd_cache(a)
    if a.cmd == "layouts":
        return cmd_layouts(a)
    if a.cmd == "run":
        return cmd_run(a)
    return cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
