#!/usr/bin/env python
"""A2 Step 1, agent A: tabulate the harness's JSONs under /data/enric/reports/a2_seg/step1/A/ (formatting
only; every number is read from the JSONs that step1_A_run.py wrote through eo.maps.harness).

    python eo/experiments/segmentation/step1_A_table.py [--md out.md] [--json out.json]

Tables: (1) each candidate's rule, how it was chosen, and each dataset's own optimum on train_sub;
(2) the read, candidate x dataset (ceiling with the chosen rule, nearest and vote beside, boundary and
interior accuracy, code statistics, the val sweep's own optimum and the gap); (3) CloudSen12's native256
read beside crop224; (4) the controls; (5) the prune against K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

OUT = Path("/data/enric/reports/a2_seg/step1/A")
CANDS = ["t1_di16_224", "t1_di16_256", "t1_di8_224", "t1_di8_256", "c01_di16_256", "c01_di8_256"]
DS = ["pastis", "cloudsen12", "lulc"]
NEED = {"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}


def j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=4):
    return "–" if x is None else f"{x:.{n}f}"


def rk(rule: dict | None) -> str:
    if rule is None:
        return "–"
    if rule["rule"] == "region":
        return f"region τ{rule['tau']:g} r{rule['radius']}"
    return rule["rule"]


def key_h(k: str) -> str:
    return k.replace("region_tau", "τ").replace("_r", " r")


def near(v: dict) -> dict:
    return v["also"].get("nearest", v["ceiling"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    L, rows = [], []

    L.append("### Candidates: the rule, chosen on train_sub")
    L.append("| candidate | px | tokens | codebook | rule | how chosen | train_sub mean mIoU | pastis optimum | "
             "cloudsen12 optimum | lulc optimum | bar passes |")
    L.append("|" + "---|" * 11)
    for c in CANDS:
        sel, rd = j(OUT / f"select_{c}.json"), j(OUT / f"read_{c}.json")
        if sel is None:
            L.append(f"| {c} | not selected yet |" + " |" * 9)
            continue
        fin = sel.get("extended", sel["default"])
        how = "select_rule, default grid"
        if "extension" in sel:
            how = f"on edge {sel['extension']}: re-selected on taus {sel['extended_grid']['taus']}, radii " \
                  f"{sel['extended_grid']['radii']}"
        elif "radius_lo_note" in sel:
            how += " (radius 1: the step below is nearest, already read)"
        tok = sel["tokenizer"]
        po = fin["per_dataset_optimum"]
        L.append(f"| {c} | {tok['input_size']} | {tok['tokens']} | {tok['n_codes']:,} | {rk(sel['chosen'])} | {how} | "
                 f"{fin['mean'][sel['chosen_key']]:.4f} | "
                 + " | ".join(f"{key_h(po[d]['rule'])} {po[d]['mIoU']:.4f} (rule {fin['by_dataset'][d][sel['chosen_key']]:.4f})"
                              for d in DS)
                 + f" | {rd['bar']['passes'] if rd else '–'} |")

    L.append("")
    L.append("### The read (val), with the rule chosen on train_sub")
    L.append("| candidate | dataset | ceiling | nearest | vote | boundary | interior | distinct codes | entropy (bits) | "
             "unseen token share | val's own optimum (gap) | bar |")
    L.append("|" + "---|" * 12)
    for c in CANDS:
        sel, rd, vs = j(OUT / f"select_{c}.json"), j(OUT / f"read_{c}.json"), j(OUT / f"valsweep_{c}.json")
        if sel is None or rd is None:
            continue
        for d in DS:
            r = rd["read"][d]
            ce, nr = r["ceiling"], near(r)
            s = vs["summary"][d] if vs else None
            po = sel.get("extended", sel["default"])["per_dataset_optimum"][d]
            row = {"cand": c, "dataset": d, "rule": sel["chosen"], "tokens": r["tokens_per_map"],
                   "n_codes": r["codes"]["n_codes"], "ceiling": ce["mIoU"], "nearest": nr["mIoU"],
                   "vote": s["vote"] if s else None, "boundary": ce["boundary_acc"], "interior": ce["interior_acc"],
                   "pixel": ce["pixel_acc"], "distinct": r["codes"]["read_distinct"],
                   "entropy": r["codes"]["read_entropy_bits"], "unseen": r["codes"]["unseen_token_share"],
                   "ref_distinct": r["codes"].get("ref_distinct"), "train_opt": po,
                   "val_opt": {"rule": s["val_optimum"], "mIoU": s["val_optimum_mIoU"], "gap": s["gap"]} if s else None,
                   "passes_dataset": ce["mIoU"] >= NEED[d], "json": str(OUT / f"read_{c}.json")}
            rows.append(row)
            L.append(f"| {c} | {d} | **{f(row['ceiling'])}** | {f(row['nearest'])} | {f(row['vote'])} | "
                     f"{f(row['boundary'], 3)} | {f(row['interior'], 4)} | {row['distinct']:,} | {f(row['entropy'], 2)} | "
                     f"{f(row['unseen'], 5)} | "
                     + (f"{key_h(s['val_optimum'])} {f(s['val_optimum_mIoU'])} (+{s['gap']:.4f})" if s else "–")
                     + f" | {'≥' if row['passes_dataset'] else '<'} {NEED[d]} |")

    L.append("")
    L.append("### CloudSen12 val, native256 crops beside crop224 (256-px candidates; never instead)")
    L.append("| candidate | crop224 ceiling | native256 ceiling | native256 nearest | native256 boundary | "
             "native256 distinct | native256 unseen |")
    L.append("|---|---|---|---|---|---|---|")
    for c in CANDS:
        rd, nv = j(OUT / f"read_{c}.json"), j(OUT / f"native256_{c}.json")
        if nv is None:
            continue
        r = nv["read"]
        L.append(f"| {c} | {f(rd['read']['cloudsen12']['ceiling']['mIoU']) if rd else '–'} | {f(r['ceiling']['mIoU'])} | "
                 f"{f(near(r)['mIoU'])} | {f(r['ceiling']['boundary_acc'], 3)} | {r['codes']['read_distinct']:,} | "
                 f"{f(r['codes']['unseen_token_share'], 5)} |")

    L.append("")
    L.append("### Controls")
    for c in CANDS:
        det, tr, vs = j(OUT / f"det_{c}.json"), j(OUT / f"transposed_{c}.json"), j(OUT / f"valsweep_{c}.json")
        if det:
            ds = det["datasets"]
            L.append(f"- det {c}: encode twice / decode twice identical, and equal to the harness cache written by "
                     f"another process: {det['all_identical']} ("
                     + ", ".join(f"{d} {v['crops']} crops {v['codes_identical'] and v['decoded_identical']}/"
                                 f"{v.get('codes_equal_cache')}" for d, v in ds.items())
                     + "); control, one 32x32 block of crop 0 repainted: "
                     + ", ".join(f"{d} {v['control_block']['crop0_codes_changed']} codes of crop 0 change, "
                                 f"{v['control_block']['other_crops_codes_changed']} elsewhere" for d, v in ds.items())
                     + "; batch 8 instead of 16: "
                     + ", ".join(f"{d} {v['control_batch8']['codes_differing']}/{v['control_batch8']['of']} differ"
                                 for d, v in ds.items())
                     + f"; FSQ index <-> digits = the encoder's own normalised codes: {det['fsq_mapping_ok']}")
        if tr:
            fl = j(OUT / f"transposed_floor_identity_{c.rsplit('_', 1)[1]}.json")
            L.append(f"- transposed {c} (rule {rk(tr['rule'])}): " + ", ".join(
                f"{d} {v['ceiling_transposed']:.4f} vs {v['ceiling_read']:.4f} (x{v['ratio']:.3f}; nearest "
                f"{v['result']['also'].get('nearest', v['result']['ceiling'])['mIoU']:.4f}"
                + (f"; a perfect tokenizer fed the same transposed renders: "
                   f"{fl['by_dataset'][d]['ceiling_transposed']:.4f}" if fl else "") + ")"
                for d, v in tr["by_dataset"].items()))
        if vs:
            L.append(f"- val sweep {c}: chosen rule's val numbers = the read's (another process): {vs.get('equals_read')}; "
                     f"val's shared optimum {key_h(vs['val_shared_optimum']['rule'])} mean {vs['val_shared_optimum']['mean']:.4f} "
                     f"vs the chosen rule's {vs['val_shared_optimum']['chosen_mean']:.4f} (gap {vs['val_shared_optimum']['gap']:.4f}); "
                     f"Step 0's region τ0.15 r4 on val: "
                     + ", ".join(f"{d} {vs['summary'][d]['region_tau0.15_r4']:.4f}" for d in DS))

    # Step 0 against now, t1_di8_256: palette, pooling, rule (the orchestrator's anchors and gate Z2)
    P0 = Path("/data/enric/reports/a2_seg/step1/phase0")
    v1 = j(P0 / "anchor_t1_di8_256_universal_region015r4_harness_v1_areapool.json")
    v2 = j(P0 / "anchor_t1_di8_256_universal_region015r4.json")
    gate = j(Path("/data/enric/reports/a2_seg/step1/gate/gate_final.json"))
    rd = j(OUT / f"read_t1_di8_256.json")
    if v1 and v2 and rd:
        z2 = next((g["detail"] for g in (gate or []) if g["check"] == "Z2 two scoring paths"), None)
        L.append("")
        L.append("### Step 0 against now (t1_di8_256, region τ0.15 r4 unless marked)")
        L.append(f"Step 0: PASTIS fold 4 0.952 (per-dataset `distinct` palette, area pooling); the harness "
                 f"reproduces it (gate Z2: {z2}).")
        L.append("| dataset | universal palette, v1 area pooling | universal, v2 adjoint pooling | Δ pooling | "
                 f"chosen rule {rk(rd['rule'])}, v2 | Δ rule |")
        L.append("|---|---|---|---|---|---|")
        for d in DS:
            a1, a2 = v1["read"][d]["ceiling"]["mIoU"], v2["read"][d]["ceiling"]["mIoU"]
            c = rd["read"][d]["ceiling"]["mIoU"]
            L.append(f"| {d} | {a1:.4f} | {a2:.4f} | {a2 - a1:+.4f} | {c:.4f} | {c - a2:+.4f} |")

    prunes = sorted(OUT.glob("prune_*.json"))
    if prunes:
        L.append("")
        L.append("### The lossy prune (kept codes = the K most frequent over the three train_subs; others -> nearest kept "
                 "on the normalised FSQ grid; the unpruned tokenizer's rule)")
        L.append("| candidate | K | kept mass on train_sub | dataset | val tokens remapped | ceiling | Δ vs unpruned | "
                 "nearest | boundary | distinct | entropy | unseen | bar |")
        L.append("|" + "---|" * 13)
        for p in prunes:
            r = j(p)
            base = j(OUT / f"read_{r['cand']}.json")
            for d in DS:
                v = r["read"][d]
                b = base["read"][d]["ceiling"]["mIoU"] if base else None
                L.append(f"| {r['cand']} | {r['tag']} ({r['kept']:,}) | {r['kept_mass_train_sub']:.4f} | {d} | "
                         f"{r['tokens_remapped_share'][d]['val']:.4f} | {v['ceiling']['mIoU']:.4f} | "
                         f"{(v['ceiling']['mIoU'] - b) if b is not None else float('nan'):+.4f} | {near(v)['mIoU']:.4f} | "
                         f"{v['ceiling']['boundary_acc']:.3f} | {v['codes']['read_distinct']:,} | "
                         f"{v['codes']['read_entropy_bits']:.2f} | {v['codes']['unseen_token_share']:.5f} | "
                         f"{r['bar']['passes']} |")
    text = "\n".join(L)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
