#!/usr/bin/env python
"""B3's 5-layer vanilla (B10c-5L-tr-NC, 2026-11-15), read against the Coords-free 1 + 3 + 1 train cell.

The Paper B plan's 2 x 2 (its order, item 4) keeps vanilla references "beside and optional": they say what unique
layers buy in each cell, and whether A10cR ~ B10c-11L holds for the small block. On val's 1 + 9 + 1 block a random
router ended level with an 11-layer vanilla holding the same unique layers. Here the vanilla holds the 1 + 3 + 1
block's 5 unique layers, applied once (68,133,312 parameters, A10cR-b3's); the random router applies them 5, 8 or 11
times, 8 on average.

Every unit is eval_checkpoint.py's main pass on the TITAN V at fp16, with Coords dropped, on the same 4,416 held-out
rows (the vanilla through its Coords twin, arm_b10c5tr_vanilla, whose sequences gate W29 holds equal to
arm_a10crb3_random's):
  - the vanilla, arm_eval/b10c5trnc_e1tv;
  - the cell's routed passes: a10crb3trnc_e1tv and a10crb3trs2nc_e1tv (random router, seeds 42 and 43),
    a10cb3trnc_e1tv and a10cb3trs2nc_e1tv (learned, seeds 42 and 43), and the width check a10crw384trnc_e1tv;
  - the two random backbones forced to depth 1, 2 and 3 everywhere (B0 Step 6's tvforce units,
    wave1/landscape/<tag>/<tag>_tvforce<d>/).
Each pair is read per token on pooled image tokens with a0_read's pairing: rows, labels and modalities must be
identical, checked first. CONTROLS: B10 on the original tokens must fail that check, and the vanilla paired with
itself must read 0 with a zero-width interval. 95% row bootstrap (2,000 draws), at 1980 and 3300, per modality
beside it. "Random, both seeds" pairs the vanilla with the mean of the two seeds' per-row sums. The yardstick is the
cell's own seed gap (0.0103 at 1980, 0.0069 at 3300; B3's rule). No rule was fixed for this reading: it is
description.

E8's pair, read for the record (the plan's "E8's 5x rule is read on them"): B10c-11L-tr, the 11-layer vanilla on train,
scored on the TITAN V at fp16 WITH Coords, as it trained (arm_eval/b10c11tr_e1tv, its own config), minus the 5-layer
vanilla, paired per row on the image tokens as A0.1 pairs a Coords model with its Coords-free twin (a0_read's
image_rows: eval rows equal and every row's image-token labels identical, in order; CONTROL: B10 on the original tokens
must fail it). Beside it, the same 11-layer model minus the random backbones forced to depth 3, which apply their 5
unique layers 11 times: unique layers against recursion at equal layer applications. Not like with like in Coords:
the 11-layer model has Coords in its context, and A0.1 found Coords-free models 0.0075 (B10c) and 0.0121 (A10c) lower
on image tokens at val's 10 epochs, so these differences read the 11-layer model's advantage small by about that much.

    .venv/bin/python eo/experiments/routing/b3_vanilla5.py

Writes /data/enric/reports/wave1/vanilla5/read.{txt,json}. Runs in `.venv`, CPU.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

AE = Path("/data/enric/reports/arm_eval")
LS = Path("/data/enric/reports/wave1/landscape")
OUT = Path("/data/enric/reports/wave1/vanilla5")
CKS = ("checkpoint-1980", "checkpoint-3300")
VAN = "b10c5trnc_e1tv"
ROUTED = {"random s42": "a10crb3trnc_e1tv", "random s43": "a10crb3trs2nc_e1tv",
          "learned s42": "a10cb3trnc_e1tv", "learned s43": "a10cb3trs2nc_e1tv",
          "width check": "a10crw384trnc_e1tv"}
FORCED = {"random s42": "a10crb3trnc", "random s43": "a10crb3trs2nc"}
E8 = "b10c11tr_e1tv"   # B10c-11L-tr: 11 layers, train, Coords present (as it trained), TITAN V fp16
SEED_GAP = {"checkpoint-1980": 0.0103, "checkpoint-3300": 0.0069}   # the Coords-free cell's own (B3 note, section 1)


def _a0():
    """a0_read's loaders and pairing (eo/experiments/base/a0_read.py; eo/experiments is not a package)."""
    f = Path(__file__).resolve().parents[1] / "base" / "a0_read.py"
    spec = importlib.util.spec_from_file_location("a0_read", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unit_ok(d: Path, drop=("Coords",)) -> dict:
    m = json.loads((d / "metrics.json").read_text())
    if (m["gpu"] != "NVIDIA TITAN V" or m.get("precision", "fp16") != "fp16"
            or list(m.get("drop_modalities") or []) != list(drop)
            or m["token_set"] != "clean" or m["n_eval_rows"] != 4416):
        raise SystemExit(f"{d}: {m['gpu']}, {m.get('precision')}, drop {m.get('drop_modalities')}, "
                         f"{m['token_set']}, {m['n_eval_rows']} rows: not the TITAN V fp16 protocol with drop {list(drop)}")
    return m


def e8_pair(a0, ck: str, img: list, fmt) -> tuple[dict, list]:
    """B10c-11L-tr (Coords present) minus the 5-layer vanilla and minus the forced-depth-3 backbones, per row on the
    image tokens (A0.1's pairing: a Coords model against Coords-free scoring of the same rows)."""
    import numpy as np
    E, V = AE / E8 / ck, AE / VAN / ck
    unit_ok(E, drop=())
    te, tc = a0.tokens(E), a0.tokens(a0.CONTROL)
    if int((te["label_mod"] == a0.MODALITY_TO_ID["Coords"]).sum()) == 0:
        raise SystemExit(f"{E}: no Coords token scored, but it was scored with Coords present")
    le, pe = a0.image_rows(te)
    lc, _ = a0.image_rows(tc)
    rows_e = np.load(E / "eval_rows.npy")
    ctl = (np.load(a0.CONTROL / "eval_rows.npy") == rows_e).all() and all(np.array_equal(x, y) for x, y in zip(lc, le))
    if ctl:
        raise SystemExit("CONTROL PASSED (B10, original tokens, has the 11-layer model's image labels): the check is blind")
    others = {"5-layer vanilla": V}
    for k, tag in FORCED.items():
        others[f"{k} forced d3"] = LS / tag / f"{tag}_tvforce3" / ck
    D, lines = {"deep_image_ce": a0.mean(pe["image"]), "pairs": {}}, []
    rows_mean = None
    for k, p in others.items():
        unit_ok(p)
        t = a0.tokens(p)
        if int((t["label_mod"] == a0.MODALITY_TO_ID["Coords"]).sum()):
            raise SystemExit(f"{p}: Coords tokens scored in a Coords-dropped unit")
        lo_, po = a0.image_rows(t)
        same = (np.load(p / "eval_rows.npy") == rows_e).all() and all(np.array_equal(x, y) for x, y in zip(lo_, le))
        if not same:
            raise SystemExit(f"{E} vs {p}: the rows' image-token labels differ: not paired")
        d = a0.paired(pe["image"], po["image"])
        D["pairs"][k] = {"other_image_ce": a0.mean(po["image"]), "image": d,
                         "per_modality": {n: a0.paired(pe[n], po[n]) for n in img},
                         "gain_x_seed_gap": -d["diff"] / SEED_GAP[ck], "gain_x_0006": -d["diff"] / 0.006}
        if k.endswith("forced d3"):
            rows_mean = po if rows_mean is None else {n: {"n": po[n]["n"], "ce": (po[n]["ce"] + rows_mean[n]["ce"]) / 2,
                                                         "top1": (po[n]["top1"] + rows_mean[n]["top1"]) / 2} for n in po}
    d = a0.paired(pe["image"], rows_mean["image"])
    D["pairs"]["random forced d3, both seeds"] = {"other_image_ce": a0.mean(rows_mean["image"]), "image": d,
                                                  "per_modality": {n: a0.paired(pe[n], rows_mean[n]) for n in img}}
    lines.append(f"  {ck}, E8's pair (for the record): B10c-11L-tr, Coords present, image tokens "
                 f"{D['deep_image_ce']:.4f}; pairing per row on image tokens identical with every unit, control (B10, "
                 f"original tokens) fails as it must. Not like with like in Coords: see the docstring")
    for k, v in D["pairs"].items():
        g = v.get("gain_x_seed_gap")
        tail = f"; the 11-layer gain {g:.2f}x the cell's seed gap, {v['gain_x_0006']:.2f}x 0.006 (E8's 5x)" if g is not None else ""
        lines.append(f"    B10c-11L-tr - {k:28s} {fmt(v['image'])}   (other {v['other_image_ce']:.4f}{tail})")
    lines.append("      per modality, B10c-11L-tr - 5-layer vanilla: " + "; ".join(
        f"{n} {fmt(x)}" for n, x in D["pairs"]["5-layer vanilla"]["per_modality"].items()))
    return D, lines


def main() -> int:
    a0 = _a0()
    img = [m for m in a0.MODALITIES if m != "Coords"]
    names = img + ["image", "all"]
    fmt = lambda p: f"{p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}]"  # noqa: E731
    doc = {"vanilla": VAN, "routed": ROUTED, "forced": FORCED, "seed_gap": SEED_GAP,
           "measure": "pooled image-token CE, vanilla - other, paired per token, 95% row bootstrap "
                      f"({a0.NBOOT} draws)", "by_ck": {}}
    lines = ["B3's 5-layer vanilla (B10c-5L-tr-NC: the 1 + 3 + 1 block's 5 unique layers, applied once) against the "
             "Coords-free 1 + 3 + 1 train cell; pooled image-token CE, vanilla - other, paired per token, 95% row "
             f"bootstrap ({a0.NBOOT}); TITAN V fp16, Coords dropped. Negative: the vanilla is better."]
    for ck in CKS:
        V = AE / VAN / ck
        mv = unit_ok(V)
        c_ok, c_why = a0.same_sequences(a0.CONTROL, V)
        if c_ok:
            raise SystemExit(f"CONTROL PASSED (B10, original tokens, vs {V}: {c_why}): the check is blind")
        rv = a0.per_row(a0.tokens(V), names)
        z = a0.paired(rv["image"], rv["image"])
        if z["diff"] != 0 or z["lo"] != 0 or z["hi"] != 0:
            raise SystemExit(f"CHECK FAILED: the vanilla paired with itself reads {z}")
        e1 = a0.mean(rv["all"])
        if abs(e1 - mv["modes"]["router"]["ce_all_body_tokens"]) > 2e-3:
            raise SystemExit(f"E1 from tokens {e1:.6f} vs metrics.json {mv['modes']['router']['ce_all_body_tokens']:.6f}")
        units = {k: AE / t / ck for k, t in ROUTED.items()}
        for k, tag in FORCED.items():
            for d in (1, 2, 3):
                units[f"{k} forced d{d}"] = LS / tag / f"{tag}_tvforce{d}" / ck
        R, why0 = {}, None
        for k, p in units.items():
            unit_ok(p)
            ok, why = a0.same_sequences(V, p)
            if not ok:
                raise SystemExit(f"{V} vs {p}: {why}: not the same sequences")
            why0 = why
            R[k] = a0.per_row(a0.tokens(p), names)
        R["random, both seeds"] = {n: {"n": R["random s42"][n]["n"],
                                       "ce": (R["random s42"][n]["ce"] + R["random s43"][n]["ce"]) / 2,
                                       "top1": (R["random s42"][n]["top1"] + R["random s43"][n]["top1"]) / 2}
                                   for n in names}
        D = {"vanilla_image_ce": a0.mean(rv["image"]), "vanilla_e1": e1, "pairing": why0, "control": c_why, "pairs": {}}
        lines += [f"  {ck}: vanilla image tokens {D['vanilla_image_ce']:.4f} (E1 {e1:.4f}); pairing {why0} with every "
                  f"unit; control (B10, original tokens) fails as it must ({c_why}); self-pair 0 [0, 0]; the cell's "
                  f"seed gap {SEED_GAP[ck]:.4f}"]
        order = ["random s42", "random s43", "random, both seeds", "learned s42", "learned s43", "width check"] + \
                [f"{k} forced d{d}" for k in FORCED for d in (1, 2, 3)]
        for k in order:
            p = a0.paired(rv["image"], R[k]["image"])
            pm = {n: a0.paired(rv[n], R[k][n]) for n in img}
            D["pairs"][k] = {"other_image_ce": a0.mean(R[k]["image"]), "image": p, "per_modality": pm,
                             "x_seed_gap": abs(p["diff"]) / SEED_GAP[ck]}
            lines.append(f"    vanilla - {k:22s} {fmt(p)}   (other {a0.mean(R[k]['image']):.4f}; "
                         f"|diff| {abs(p['diff']) / SEED_GAP[ck]:.2f}x the seed gap)")
        for k in ("random, both seeds", "learned s42", "learned s43", "width check"):
            lines.append(f"      per modality, vanilla - {k}: " + "; ".join(
                f"{n} {fmt(v)}" for n, v in D["pairs"][k]["per_modality"].items()))
        doc["by_ck"][ck] = D
    for ck in CKS:
        if (AE / E8 / ck / "metrics.json").exists():
            D8, l8 = e8_pair(a0, ck, img, fmt)
            doc["by_ck"][ck]["e8"] = D8
            lines += l8
        else:
            lines.append(f"  {ck}, E8's pair: {E8} not scored yet")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(json.dumps(doc, indent=1) + "\n")
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
