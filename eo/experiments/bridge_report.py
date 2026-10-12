#!/usr/bin/env python
"""The bridge (next steps §3, 2026-10-11): the old A10/B10 at checkpoint-3300 on the cleaned tokens.

Three evaluations of the same checkpoint, the same held-out rows in the same modality orders:

  <arm>           original tokens, TITAN V   (arm_eval/<arm>, Step 11)
  <arm>_a100      original tokens, A100      (the card control)
  <arm>_tok224c   cleaned tokens,  A100      (the bridge)

  card delta    = <arm>_a100 - <arm>            the noise floor of comparing across cards
  token delta   = <arm>_tok224c - <arm>_a100    what the cleaned tokens change, same card

Per modality, teacher-forced CE: all held-out rows, the offset rows (where S2L2A/NDVI tokens
changed) and the clean rows. A clean row's sequence is identical in both token sets, so its
token delta must be ~0 on one card: that is the check that the pairing is right.
Paired per row, with a bootstrap 95% CI over rows. Also the A10 - B10 gap on each token set,
the grid's cells, and (A10) the mean depth per modality.

    python eo/experiments/bridge_report.py        # .venv, CPU
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.data.eo_vocab import MODALITY_TO_ID  # noqa: E402
from eo.terramesh_tok.clean import load_labels  # noqa: E402

EVAL = Path("/data/enric/reports/arm_eval")
CLEAN = Path("/data/enric/data/TerraMesh/val_clean")
OUT = Path("/data/enric/reports/bridge")
CK = "checkpoint-3300"
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"]


def load(tag):
    d = EVAL / tag / CK
    z = np.load(d / "tokens_router.npz")
    return {"rows": np.load(d / "eval_rows.npy"), "ce": z["ce"].astype(np.float64),
            "mod": z["label_mod"], "depth": z["depth"] if "depth" in z.files else None,
            "metrics": json.loads((d / "metrics.json").read_text()),
            "grid": json.loads((d / "grid.json").read_text())["cells"] if (d / "grid.json").exists() else None}


def per_row(e, mid, what="ce"):
    """Per-row mean of `what` over the row's tokens labelled modality `mid` (NaN if none)."""
    x = e[what] if what == "ce" else e["depth"].astype(np.float64)
    m = e["mod"] == mid
    n = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (x * m).sum(1) / np.maximum(n, 1), np.nan)


def boot(x, key, n_boot=10_000):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0}
    rng = np.random.default_rng([0, zlib.crc32(key.encode())])
    m = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "max_abs": float(np.abs(x).max())}


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore", "Mean of empty slice")
    OUT.mkdir(parents=True, exist_ok=True)
    lab = load_labels(CLEAN).set_index("row").offset
    res, lines = {}, []
    E = {}
    for arm in ("b10", "a10"):
        E[arm] = {k: load(t) for k, t in (("titanv", arm), ("a100", f"{arm}_a100"), ("clean", f"{arm}_tok224c"))}
        rows = E[arm]["titanv"]["rows"]
        for k in ("a100", "clean"):
            if not np.array_equal(E[arm][k]["rows"], rows) or not np.array_equal(E[arm][k]["mod"], E[arm]["titanv"]["mod"]):
                raise SystemExit(f"{arm} {k}: rows or modality layout differ from the TITAN V pass")
        off = np.array([bool(lab[int(r)]) for r in rows])
        res[arm] = {"n_rows": int(len(rows)), "n_offset_rows": int(off.sum()), "modalities": {}}
        lines.append(f"\n=== {arm.upper()} checkpoint-3300: {len(rows)} held-out rows, {int(off.sum())} offset "
                     f"({off.mean():.1%})")
        lines.append(f"{'modality':8} {'CE TITAN V':>10} {'A100':>8} {'clean':>8}   "
                     f"{'card d (all)':>22}   {'token d, offset rows':>26}   {'token d, clean rows':>22}")
        for name in MODS:
            mid = MODALITY_TO_ID[name]
            ce = {k: per_row(E[arm][k], mid) for k in E[arm]}
            card = ce["a100"] - ce["titanv"]
            tok = ce["clean"] - ce["a100"]
            r = {"ce": {k: float(np.nanmean(v)) for k, v in ce.items()},
                 "card_delta_all": boot(card, f"{arm}|{name}|card"),
                 "token_delta_all": boot(tok, f"{arm}|{name}|tok"),
                 "token_delta_offset": boot(np.where(off, tok, np.nan), f"{arm}|{name}|tok_off"),
                 "token_delta_clean": boot(np.where(~off, tok, np.nan), f"{arm}|{name}|tok_cln"),
                 "ce_offset_rows": {k: float(np.nanmean(np.where(off, v, np.nan))) for k, v in ce.items()},
                 "ce_clean_rows": {k: float(np.nanmean(np.where(~off, v, np.nan))) for k, v in ce.items()}}
            if E[arm]["titanv"]["depth"] is not None:
                dp = {k: per_row(E[arm][k], mid, "depth") for k in E[arm]}
                r["depth_offset_rows"] = {k: float(np.nanmean(np.where(off, v, np.nan))) for k, v in dp.items()}
                r["depth_clean_rows"] = {k: float(np.nanmean(np.where(~off, v, np.nan))) for k, v in dp.items()}
            res[arm]["modalities"][name] = r
            f = lambda b: f"{b['mean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]" if b.get("n") else "-"
            lines.append(f"{name:8} {r['ce']['titanv']:10.4f} {r['ce']['a100']:8.4f} {r['ce']['clean']:8.4f}   "
                         f"{f(r['card_delta_all']):>22}   {f(r['token_delta_offset']):>26}   "
                         f"{f(r['token_delta_clean']):>22}")
            if "depth_offset_rows" in r:
                lines.append(f"{'':8} depth offset rows: orig {r['depth_offset_rows']['a100']:.3f} -> clean "
                             f"{r['depth_offset_rows']['clean']:.3f}; clean rows {r['depth_clean_rows']['a100']:.3f} -> "
                             f"{r['depth_clean_rows']['clean']:.3f}")
        # The grid, per cell, paired by row from grid_rows.npz: the token delta on offset rows,
        # and on clean rows (must be ~0, as above).
        z0 = np.load(EVAL / f"{arm}_a100" / CK / "grid_rows.npz")
        z1 = np.load(EVAL / f"{arm}_tok224c" / CK / "grid_rows.npz")
        cells = {}
        for key in sorted(k[:-len("|rows")] for k in z0.files if k.endswith("|rows")):
            if f"{key}|rows" not in z1.files or not np.array_equal(z0[f"{key}|rows"], z1[f"{key}|rows"]):
                raise SystemExit(f"{arm} grid {key}: rows differ between token sets")
            r_ = z0[f"{key}|rows"]
            c0 = z0[f"{key}|ce_sum"] / np.maximum(z0[f"{key}|n"], 1)
            c1 = z1[f"{key}|ce_sum"] / np.maximum(z1[f"{key}|n"], 1)
            o = np.array([bool(lab[int(r)]) for r in r_])
            d = c1 - c0
            cells[key] = {"n": int(len(r_)), "n_offset": int(o.sum()), "ce_orig": float(c0.mean()),
                          "ce_clean": float(c1.mean()),
                          "delta_offset": boot(np.where(o, d, np.nan), f"{arm}|{key}|off"),
                          "delta_clean_max_abs": float(np.abs(d[~o]).max()) if (~o).any() else None}
        res[arm]["grid_cells"] = cells
        clean_max = max(v["delta_clean_max_abs"] or 0 for v in cells.values())
        top = sorted(((c, v) for c, v in cells.items() if v["delta_offset"].get("n")),
                     key=lambda kv: -abs(kv[1]["delta_offset"]["mean"]))[:10]
        lines.append(f"grid ({len(cells)} cells), token delta on offset rows, largest |mean| first "
                     f"(clean rows: max |delta| {clean_max:.1e}):")
        for c, v in top:
            b = v["delta_offset"]
            lines.append(f"   {c:15} n_off {b['n']:3d}  {b['mean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]")

    # The retrained model (B10c / A10c), when evaluated: against the old model, both on the
    # cleaned tokens, same rows and modality orders. On offset rows this says whether the
    # corrected targets are intrinsically harder (retrained no better) or only unfamiliar to a
    # model trained with the offset (retrained better); on clean rows it is the run-to-run
    # difference of two trainings that differ only in the 24% offset rows' tokens.
    for arm in ("b10", "a10"):
        tag = f"{arm}c"
        if not (EVAL / tag / CK / "tokens_router.npz").exists():
            continue
        new = load(tag)
        old = E[arm]["clean"]
        if not np.array_equal(new["rows"], old["rows"]) or not np.array_equal(new["mod"], old["mod"]):
            raise SystemExit(f"{tag}: rows or modality layout differ from {arm}_tok224c")
        off = np.array([bool(lab[int(r)]) for r in new["rows"]])
        res[arm]["retrained"] = {}
        lines.append(f"\n=== {tag.upper()} (retrained on the cleaned tokens) - {arm.upper()} (old), both on the "
                     f"cleaned tokens, checkpoint-3300, paired by row")
        lines.append(f"{'modality':8} {'old':>8} {'retrained':>9}   {'offset rows':>26}   {'clean rows':>26}")
        for name in MODS:
            mid = MODALITY_TO_ID[name]
            d = per_row(new, mid) - per_row(old, mid)
            r = {"ce_old": float(np.nanmean(per_row(old, mid))), "ce_retrained": float(np.nanmean(per_row(new, mid))),
                 "delta_offset": boot(np.where(off, d, np.nan), f"{tag}|{name}|off"),
                 "delta_clean": boot(np.where(~off, d, np.nan), f"{tag}|{name}|cln")}
            res[arm]["retrained"][name] = r
            f = lambda b: f"{b['mean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}]" if b.get("n") else "-"
            lines.append(f"{name:8} {r['ce_old']:8.4f} {r['ce_retrained']:9.4f}   {f(r['delta_offset']):>26}   "
                         f"{f(r['delta_clean']):>26}")

    # The A10 - B10 gap per token set (paired by row, same card).
    res["gap"] = {}
    lines.append("\n=== A10 - B10, paired by row, A100")
    for name in MODS:
        mid = MODALITY_TO_ID[name]
        g = {k: per_row(E["a10"][k], mid) - per_row(E["b10"][k], mid) for k in ("a100", "clean")}
        res["gap"][name] = {k: boot(v, f"gap|{name}|{k}") for k, v in g.items()}
        b = res["gap"][name]
        lines.append(f"{name:8} original {b['a100']['mean']:+.4f} [{b['a100']['lo']:+.4f},{b['a100']['hi']:+.4f}]   "
                     f"clean {b['clean']['mean']:+.4f} [{b['clean']['lo']:+.4f},{b['clean']['hi']:+.4f}]")

    (OUT / "bridge_report.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "bridge_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {OUT}/bridge_report.{{json,txt}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
