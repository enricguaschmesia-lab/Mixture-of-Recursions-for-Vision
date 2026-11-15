#!/usr/bin/env python
"""Paper A, A0 (wave 1, 2026-11-11): the readings that A0.1 (Coords) and A0.2 (the base) are decided on, from
eval_checkpoint.py's fp32 outputs on the GTX Titan X. Rules: Paper A plan, A0.1 and A0.2 (fixed 2026-11-11, before
the runs); step plan docs/plans/wave1_base_and_2x2.md.

  screen <tag> [--ck checkpoint-3300]
      A0.2's screen for a model trained on the train split: E1 (CE over every body token of the 4,416 held-out rows,
      Coords included, as T360's 5.0361) against the screen, at most 5.136 (T360 + 0.10). Paired per token with T360
      (t360_fp32, checkpoint-17441) and B10c (b10c_fp32, checkpoint-3300): all three must have scored the same
      sequences (rows, labels, label and input modalities), checked first; CONTROL: B10 on the original tokens
      (arm_eval/b10) must fail that check. Per modality and pooled, with a bootstrap over rows (2,000); the
      target-last pass per target; and, if the tag has a grid, the 34 one-source cells pooled, which must reproduce
      T360's 5.638 and B10c's 6.190 from their own grids before the tag's is read.
  coords <nc-tag> <ref-tag>
      A0.1's pair: a model trained without Coords, scored on the reference's held-out sequences with the Coords
      block removed (eval_checkpoint.py --drop-modalities Coords), against the reference scored on its own
      sequences. Positions shift where Coords is removed, so the pairing is per row on the image tokens: each row's
      image-token labels, in order, must be identical (CONTROL: B10 on the original tokens fails it). The image-token
      CE difference, pooled over the six image modalities and per modality, with a row bootstrap (2,000), and the
      rule's first half: worse by at most 0.006 nats. Target-last per image target; the grid's one-source cells
      without a Coords source, ∅ and "all others", if both have grids.

    .venv/bin/python eo/experiments/base/a0_read.py screen b10ctr_fp32
    .venv/bin/python eo/experiments/base/a0_read.py coords a10cnc_fp32 a10c_fp32

Writes /data/enric/reports/wave1/a0/<sub>_<tag>.{json,txt}. Runs in `.venv` (numpy only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.data.eo_vocab import MODALITIES, MODALITY_TO_ID  # noqa: E402

AE = Path("/data/enric/reports/arm_eval")
OUT = Path("/data/enric/reports/wave1/a0")
T360 = AE / "t360_fp32" / "checkpoint-17441"
B10C = AE / "b10c_fp32" / "checkpoint-3300"
CONTROL = AE / "b10" / "checkpoint-3300"           # B10, ORIGINAL tokens: its S2L2A/NDVI labels differ on offset rows
SCREEN = 5.136                                     # T360's 5.0361 + 0.10 (Paper A plan, A0.2)
COORDS_MARGIN = 0.006                              # A10c's seed gap (Paper A plan, A0.1)
IMAGE = [m for m in MODALITIES if m != "Coords"]
NBOOT = 2000
_W = {}


def weights(n):
    if n not in _W:
        _W[n] = np.random.default_rng(0).multinomial(n, np.full(n, 1.0 / n), size=NBOOT)
    return _W[n]


def tokens(d):
    z = np.load(d / "tokens_router.npz")
    return {k: z[k] for k in ("ce", "correct", "label_mod", "input_mod", "label")}


def meta(d):
    m = json.loads((d / "metrics.json").read_text())
    assert m["precision"] == "fp32" and m["token_set"] == "clean" and m["n_eval_rows"] == 4416, (d, m["precision"])
    assert "GTX TITAN X" in m["gpu"], (d, m["gpu"])
    return m


def same_sequences(a, b):
    ra, rb = np.load(a / "eval_rows.npy"), np.load(b / "eval_rows.npy")
    if ra.shape != rb.shape or (ra != rb).any():
        return False, "eval_rows differ"
    ta, tb = tokens(a), tokens(b)
    for k in ("label", "label_mod", "input_mod"):
        n = int((ta[k] != tb[k]).sum()) if ta[k].shape == tb[k].shape else -1
        if n:
            return False, f"{k}: {'shapes differ' if n < 0 else f'{n:,} tokens differ'}"
    return True, f"identical ({len(ra):,} rows, {int((ta['label_mod'] > 0).sum()):,} scored tokens)"


def per_row(t, names):
    ce = t["ce"].astype(np.float64)
    out = {}
    for name in names:
        if name == "all":
            m = t["label_mod"] > 0
        elif name == "image":
            m = (t["label_mod"] > 0) & (t["label_mod"] != MODALITY_TO_ID["Coords"])
        else:
            m = t["label_mod"] == MODALITY_TO_ID[name]
        out[name] = {"n": m.sum(1), "ce": (ce * m).sum(1), "top1": (t["correct"] & m).sum(1)}
    return out


def mean(r, key="ce"):
    return float(r[key].sum() / max(r["n"].sum(), 1))


def paired(a, b, key="ce"):
    """Token-weighted mean of a - b over the same rows, with a bootstrap 95% CI over rows."""
    assert (a["n"] == b["n"]).all(), "different token counts per row: not paired"
    keep = a["n"] > 0
    d, n = (a[key] - b[key]).astype(np.float64)[keep], a["n"][keep]
    W = weights(len(d))
    boots = (W @ d) / (W @ n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.sum() / n.sum()), "lo": float(lo), "hi": float(hi), "rows": int(keep.sum())}


def tl_rows(d):
    f = d / "target_last_rows.npz"
    if not f.exists():
        return None
    z = np.load(f)
    targets = sorted({k.split("|")[0] for k in z.files}, key=list(MODALITIES).index)
    return {T: {"rows": z[f"{T}|rows"], "n": z[f"{T}|n"], "ce": z[f"{T}|ce_sum"], "top1": z[f"{T}|top1"]}
            for T in targets}


def grid_one_source(d, exclude_source=()):
    """{cell: ce_row_mean} of a grid's one-source cells (no "none"/"all", no holes), or None."""
    f = d / "grid.json"
    if not f.exists():
        return None
    cells = json.loads(f.read_text())["cells"]
    return {k: v["ce_row_mean"] for k, v in cells.items()
            if not v.get("hole") and v["source"] not in ("none", "all") + tuple(exclude_source)}


def grid_cells(d, keys):
    f = d / "grid.json"
    if not f.exists():
        return None
    cells = json.loads(f.read_text())["cells"]
    return {k: cells[k]["ce_row_mean"] for k in keys if k in cells and not cells[k].get("hole")}


def screen(tag, ck):
    D = AE / tag / ck
    m = {"tag": meta(D), "t360": meta(T360), "b10c": meta(B10C)}
    doc = {"tag": tag, "dir": str(D), "screen": SCREEN}
    lines = [f"A0.2 screen: {tag} {ck} (fp32, {m['tag']['gpu']})"]
    pair = {"t360": same_sequences(D, T360), "b10c": same_sequences(D, B10C)}
    ctl = same_sequences(CONTROL, B10C)
    doc["pairing"] = {k: v[1] for k, v in pair.items()}
    doc["pairing_control"] = ctl[1]
    lines += [f"  pairing vs {k}: {'PASS' if v[0] else 'FAIL'} {v[1]}" for k, v in pair.items()]
    lines.append(f"  control, B10 (original tokens) vs B10c: {'FAILS as it must' if not ctl[0] else 'PASSES: CHECK IS BLIND'} {ctl[1]}")
    if not all(v[0] for v in pair.values()) or ctl[0]:
        raise SystemExit("\n".join(lines) + "\npairing check failed (or its control passed): not a paired comparison")
    names = list(MODALITIES) + ["image", "all"]
    r = {k: per_row(tokens(p), names) for k, p in (("tag", D), ("t360", T360), ("b10c", B10C))}
    e1 = mean(r["tag"]["all"])
    ref = m["tag"]["modes"]["router"]["ce_all_body_tokens"]
    assert abs(e1 - ref) < 2e-3, (e1, ref)
    doc["e1"] = {"tag": e1, "t360": mean(r["t360"]["all"]), "b10c": mean(r["b10c"]["all"]), "metrics_json": ref}
    doc["verdict_screen"] = "PASS" if e1 <= SCREEN else "FAIL"
    gap_t, gap_b = paired(r["tag"]["all"], r["t360"]["all"]), paired(r["tag"]["all"], r["b10c"]["all"])
    closed = (doc["e1"]["b10c"] - e1) / (doc["e1"]["b10c"] - doc["e1"]["t360"])
    doc["vs"] = {"t360": gap_t, "b10c": gap_b, "share_of_t360_gap_closed": closed}
    lines.append(f"  E1 {e1:.4f} (T360 {doc['e1']['t360']:.4f}, B10c {doc['e1']['b10c']:.4f}); screen <= {SCREEN}: "
                 f"{doc['verdict_screen']}")
    lines.append(f"  - T360 {gap_t['diff']:+.4f} [{gap_t['lo']:+.4f}, {gap_t['hi']:+.4f}]; - B10c {gap_b['diff']:+.4f} "
                 f"[{gap_b['lo']:+.4f}, {gap_b['hi']:+.4f}]; closes {100 * closed:.1f}% of B10c's gap to T360")
    doc["per_modality"] = {}
    lines.append("  per modality: CE (tag, T360, B10c) and tag - T360 [95%]")
    for n in list(MODALITIES) + ["image"]:
        v = {"tag": mean(r["tag"][n]), "t360": mean(r["t360"][n]), "b10c": mean(r["b10c"][n]),
             "vs_t360": paired(r["tag"][n], r["t360"][n]), "vs_b10c": paired(r["tag"][n], r["b10c"][n])}
        doc["per_modality"][n] = v
        lines.append(f"    {n:7s} {v['tag']:.4f} {v['t360']:.4f} {v['b10c']:.4f}  {v['vs_t360']['diff']:+.4f} "
                     f"[{v['vs_t360']['lo']:+.4f}, {v['vs_t360']['hi']:+.4f}]")
    tl = {k: tl_rows(p) for k, p in (("tag", D), ("t360", T360), ("b10c", B10C))}
    if tl["tag"]:
        doc["target_last"] = {}
        lines.append("  target-last: CE (tag, T360, B10c) and tag - T360 [95%]")
        for T in tl["b10c"]:
            for k in ("t360", "tag"):
                assert (tl[k][T]["rows"] == tl["b10c"][T]["rows"]).all() and (tl[k][T]["n"] == tl["b10c"][T]["n"]).all(), (k, T)
            v = {k: mean(tl[k][T]) for k in tl} | {"vs_t360": paired(tl["tag"][T], tl["t360"][T])}
            doc["target_last"][T] = v
            lines.append(f"    {T:7s} {v['tag']:.4f} {v['t360']:.4f} {v['b10c']:.4f}  {v['vs_t360']['diff']:+.4f} "
                         f"[{v['vs_t360']['lo']:+.4f}, {v['vs_t360']['hi']:+.4f}]")
    g = {k: grid_one_source(p) for k, p in (("tag", D), ("t360", T360), ("b10c", B10C))}
    if g["tag"]:
        pooled = {k: float(np.mean(list(v.values()))) for k, v in g.items()}
        repro = abs(pooled["t360"] - 5.638413114317431) < 1e-6 and abs(pooled["b10c"] - 6.189540261010014) < 1e-6
        doc["grid"] = {"pooled_34": pooled, "n_cells": {k: len(v) for k, v in g.items()},
                       "reproduces_t360_and_b10c": repro}
        lines.append(f"  grid, 34 one-source cells pooled: {pooled['tag']:.4f} (T360 {pooled['t360']:.4f}, B10c "
                     f"{pooled['b10c']:.4f}); T360's 5.638 and B10c's 6.190 reproduced: {repro}")
    else:
        lines.append("  grid: not scored yet")
    return doc, lines


def image_rows(t):
    """Per row: the image-token labels in order (the pairing key) and per-modality CE sums and counts."""
    keep = (t["label_mod"] > 0) & (t["label_mod"] != MODALITY_TO_ID["Coords"])
    labs = [t["label"][i][keep[i]] for i in range(len(keep))]
    return labs, per_row(t, IMAGE + ["image"])


def coords(nc, ref, ck):
    A, B = AE / nc / ck, AE / ref / ck
    ma, mb = meta(A), meta(B)
    drop_a, drop_b = ma.get("drop_modalities") or [], mb.get("drop_modalities") or []
    assert "Coords" in drop_a and "Coords" not in drop_b, (drop_a, drop_b)
    doc = {"nc": nc, "ref": ref, "margin": COORDS_MARGIN}
    lines = [f"A0.1 pair: {nc} (Coords dropped from {ma['arm_config']}'s sequences) vs {ref} ({mb['arm_config']}), {ck}, fp32"]
    ra, rb = np.load(A / "eval_rows.npy"), np.load(B / "eval_rows.npy")
    ta, tb, tc = tokens(A), tokens(B), tokens(CONTROL)
    la, pa = image_rows(ta)
    lb, pb = image_rows(tb)
    lc, _ = image_rows(tc)
    same = (ra == rb).all() and all(np.array_equal(x, y) for x, y in zip(la, lb))
    ctl = (np.load(CONTROL / "eval_rows.npy") == rb).all() and all(np.array_equal(x, y) for x, y in zip(lc, lb))
    doc["pairing"] = bool(same)
    doc["pairing_control_passes"] = bool(ctl)
    lines.append(f"  pairing: every row's image-token labels identical, in order = {same}; control (B10, original "
                 f"tokens) {'fails as it must' if not ctl else 'PASSES: CHECK IS BLIND'}")
    if not same or ctl:
        raise SystemExit("\n".join(lines) + "\nnot a paired comparison")
    coords_left = int((ta["label_mod"] == MODALITY_TO_ID["Coords"]).sum())
    assert coords_left == 0, f"{coords_left} Coords tokens scored in {nc}"
    doc["image"] = {n: {"nc": mean(pa[n]), "ref": mean(pb[n]), "d": paired(pa[n], pb[n])} for n in IMAGE + ["image"]}
    d = doc["image"]["image"]["d"]
    doc["verdict_ce"] = "holds" if d["diff"] <= COORDS_MARGIN else "fails"
    lines.append(f"  image tokens pooled: {doc['image']['image']['nc']:.4f} vs {doc['image']['image']['ref']:.4f}, "
                 f"diff {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]; the rule's CE half (worse by at most "
                 f"{COORDS_MARGIN}): {doc['verdict_ce']}")
    for n in IMAGE:
        v = doc["image"][n]
        lines.append(f"    {n:7s} {v['nc']:.4f} vs {v['ref']:.4f}  {v['d']['diff']:+.4f} [{v['d']['lo']:+.4f}, {v['d']['hi']:+.4f}]")
    tla, tlb = tl_rows(A), tl_rows(B)
    if tla and tlb:
        doc["target_last"] = {}
        lines.append("  target-last (Coords out of the context for the no-Coords model)")
        for T in IMAGE:
            if T not in tla or T not in tlb:
                continue
            assert (tla[T]["rows"] == tlb[T]["rows"]).all() and (tla[T]["n"] == tlb[T]["n"]).all(), T
            v = {"nc": mean(tla[T]), "ref": mean(tlb[T]), "d": paired(tla[T], tlb[T])}
            doc["target_last"][T] = v
            lines.append(f"    {T:7s} {v['nc']:.4f} vs {v['ref']:.4f}  {v['d']['diff']:+.4f} [{v['d']['lo']:+.4f}, {v['d']['hi']:+.4f}]")
    srcs = [s for s in IMAGE] + ["none", "all"]
    keys = [f"{s}->{t}" for t in IMAGE for s in srcs if s != t]
    ga, gb = grid_cells(A, keys), grid_cells(B, keys)
    if ga and gb:
        common = sorted(set(ga) & set(gb))
        one = [k for k in common if k.split("->")[0] in IMAGE]
        doc["grid"] = {"cells": {k: {"nc": ga[k], "ref": gb[k], "d": ga[k] - gb[k]} for k in common},
                       "one_source_pooled": {"n": len(one), "nc": float(np.mean([ga[k] for k in one])),
                                             "ref": float(np.mean([gb[k] for k in one]))}}
        p = doc["grid"]["one_source_pooled"]
        lines.append(f"  grid: {len(one)} one-source cells without a Coords source pooled {p['nc']:.4f} vs {p['ref']:.4f} "
                     f"({p['nc'] - p['ref']:+.4f}); ∅ and 'all others' per target in the json")
    else:
        lines.append("  grid: not scored for both yet")
    return doc, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen")
    s.add_argument("tag")
    s.add_argument("--ck", default="checkpoint-3300")
    c = sub.add_parser("coords")
    c.add_argument("nc")
    c.add_argument("ref")
    c.add_argument("--ck", default="checkpoint-3300")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    doc, lines = screen(a.tag, a.ck) if a.cmd == "screen" else coords(a.nc, a.ref, a.ck)
    name = f"{a.cmd}_{a.tag if a.cmd == 'screen' else a.nc}"
    (OUT / f"{name}.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
