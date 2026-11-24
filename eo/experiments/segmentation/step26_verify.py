#!/usr/bin/env python
"""A2 Step 2.6: verify a LULC map-token artifact written by eo/scripts/tokenize_train_lulc_maps.py against its raw
tars (`a2seg` env, GPU: the artifact's own card, since codes are compared bit for bit).

Each check has a deliberately broken control that must fail it (CLAUDE.md rule 2):

  A  integrity. tokens.npy is (n_rows, tokens) uint16; present is exactly the done shards' rows; every present code is
     below the arm's codebook; each done shard's tokens and stems sha256, recomputed from the artifact and the index,
     equal its done record; absent rows are zero (--full only: it reads the whole array).
       control A: the expected present mask with one row flipped is refused (2026-11-22; first written as the mask
       moved by one row, which cannot differ on an artifact whose every row is present, as val's).
  B  the harness. Sampled rows re-read from their tar and re-encoded through the harness's own path
     (eo.maps.harness.render_crops + run_fixed, 64 maps a chunk, as harness.tokens encodes Step 1's sets) equal the
     artifact's codes bit for bit.
       controls B1-B3: the artifact's codes one row off; the labels transposed; LULC's colours rotated by one
       (class c -> colour c + 1). Each must agree on less than half the codes.
  C  the round trip. The artifact's codes decoded by the frozen decoder, the nearest rule on the native grid
     (harness.class_scores + decode.apply_rule), scored against the rows' own labels: pooled macro mIoU with no data
     ignored (eo.maps.scoring.Confusion) at least 0.90. The arm's LULC val ceiling is 0.9538 (Step 1.3).
       control C: the same decoded maps scored against the next row's labels must read below 0.5.

    python eo/experiments/segmentation/step26_verify.py --artifact /data/enric/data/TerraMesh/maps/train/MAP_aligned \\
        --raw /data/enric/data/TerraMesh/train_lulc_raw --per-shard 128 --tag validation
    python eo/experiments/segmentation/step26_verify.py --artifact .../maps/val/MAP_aligned \\
        --raw /data/enric/data/TerraMesh/val --per-shard 16 --full --tag full

Writes /data/enric/reports/a2_seg/step26/verify_<split>_<tag>.json. Exit 1 if any check fails or any control passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from eo.maps import decode as D, frozen, harness as H, palette as P, scoring as S  # noqa: E402
from eo.maps.tokenizers import run_fixed  # noqa: E402
from eo.terramesh_tok import contract as C  # noqa: E402
from eo.terramesh_tok import train_split as TS  # noqa: E402

REPORTS = Path("/data/enric/reports/a2_seg/step26")
TASK = "lulc"
CEILING_BAR = 0.90           # the arm's LULC val ceiling is 0.9538 (nearest rule, Step 1.3)
CONTROL_MIOU = 0.5
CONTROL_AGREE = 0.9          # a control must leave at least a tenth of the codes different (any < 1 fails B)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_rows(raw: Path, shard: str, stems_wanted: set) -> dict:
    """{stem: (224, 224) uint8} for the wanted stems of one LULC tar."""
    got = {}
    for stem, arr in TS._iter_tar(raw, "LULC", shard):
        if stem in stems_wanted:
            got[stem] = np.asarray(TS.crop(arr)).reshape(C.CROP, C.CROP).copy()
    missing = stems_wanted - set(got)
    if missing:
        raise RuntimeError(f"{shard}: {len(missing)} stems not in the tar")
    return got


def harness_codes(tok, labels: np.ndarray, assign=None, chunk: int = 64) -> np.ndarray:
    """The harness's encoding of label maps: render_crops (universal palette, crop224) + run_fixed, 64 maps a chunk.
    `assign` replaces LULC's assignment (control B3 only)."""
    import torch
    out = []
    for s in range(0, len(labels), chunk):
        if assign is None:
            x = H.render_crops(labels[s:s + chunk], TASK, "universal", tok.input_size, "crop224", "cuda")
        else:
            lab = torch.as_tensor(np.ascontiguousarray(labels[s:s + chunk]), device="cuda").long()
            x = P.render(lab, assign, tok.input_size, P.UNIVERSAL)
        out.append(run_fixed(tok.encode, x, tok.batch).reshape(len(labels[s:s + chunk]), -1).int().cpu().numpy())
    return np.concatenate(out)


def roundtrip(tok, codes: np.ndarray, labels: np.ndarray, chunk: int = 32) -> dict:
    """Pooled nearest-rule mIoU of decoded codes against labels (no data ignored)."""
    import torch
    task = P.TASKS[TASK]
    g = tok.frozen["tokens"]
    side = int(round(g ** 0.5))
    conf = S.Confusion(task.n_classes, task.ignore, "cuda")
    with torch.no_grad():
        for s in range(0, len(labels), chunk):
            ib = torch.as_tensor(codes[s:s + chunk].astype(np.int64), device="cuda").view(-1, side, side)
            sc = H.class_scores(tok, ib, TASK, "universal", "crop224")
            pr = D.apply_rule(sc, task.ignore, {"rule": "nearest"}, tok.output)
            conf.update(torch.as_tensor(labels[s:s + chunk], device="cuda").long(), pr)
    r = conf.result()
    return {"mIoU": r["mIoU"], "pixel_acc": r["pixel_acc"], "boundary_acc": r["boundary_acc"],
            "interior_acc": r["interior_acc"], "n_maps": int(len(labels))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True, help="the root holding LULC/<shard>.tar")
    ap.add_argument("--per-shard", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma-separated shards (default: every done shard)")
    ap.add_argument("--full", action="store_true", help="also read the whole array (absent rows zero)")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--tag", default="check")
    a = ap.parse_args()
    import torch

    art = a.artifact
    card = json.loads((art / "card.json").read_text())
    gpu = torch.cuda.get_device_name(0)
    if gpu != card["gpu"]:
        raise SystemExit(f"the artifact was written on {card['gpu']}; run this on that card (got {gpu})")
    TS.strict_fp32()
    idx = pd.read_parquet(card["index"])
    if hashlib.sha256(Path(card["index"]).read_bytes()).hexdigest() != card["index_sha256"]:
        raise SystemExit(f"{card['index']} changed since the artifact was written")
    tok = frozen.load(card["arm"])
    T, n_codes = tok.frozen["tokens"], tok.frozen["n_codes"]
    tokens = np.load(art / "tokens.npy", mmap_mode="r")
    present = np.load(art / "present.npy")
    done = {p.stem: json.loads(p.read_text()) for p in sorted((art / "done").glob("*.json"))}
    shards = a.only.split(",") if a.only else sorted(done)
    meta_path = art / "metadata.json"
    rep = {"artifact": str(art), "card": card, "gpu": gpu, "frozen": tok.frozen, "shards": shards,
           "metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest() if meta_path.exists() else None,
           "per_shard": a.per_shard, "seed": a.seed, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
    fails = []

    # ---- A: integrity -------------------------------------------------------------------------------------------
    t0 = time.time()
    ok_shape = tokens.shape == (len(idx), T) and tokens.dtype == np.uint16
    want_present = idx.source_shard.isin(list(done)).values
    ok_present = bool((present == want_present).all())
    # the control: the expected mask with one row flipped (a row lost or gained) must be refused. (A mask moved by one
    # row, the first version, cannot differ on an artifact whose every row is present, as val's is: 2026-11-22.)
    ctrl_present = want_present.copy()
    ctrl_present[len(ctrl_present) // 2] = not ctrl_present[len(ctrl_present) // 2]
    ctrl_fires = not bool((present == ctrl_present).all())
    sha_bad, stems_bad, range_bad = [], [], []
    for s, rec in done.items():
        r0, r1 = rec["row_first"], rec["row_last"]
        rows = idx[idx.source_shard == s]
        if not (rows.row.min() == r0 and rows.row.max() == r1 and len(rows) == rec["rows"]):
            stems_bad.append(s)
            continue
        blk = np.asarray(tokens[r0:r1 + 1]) if rec["contiguous"] else np.asarray(tokens[np.sort(rows.row.values)])
        if sha256_bytes(np.ascontiguousarray(blk).tobytes()) != rec["tokens_sha256"]:
            sha_bad.append(s)
        if sha256_bytes("\n".join(rows.sort_values("row").stem.tolist()).encode()) != rec["stems_sha256"]:
            stems_bad.append(s)
        if int(blk.max()) >= n_codes:
            range_bad.append(s)
    absent_nonzero = None
    if a.full:
        absent_nonzero = 0
        for c0 in range(0, len(idx), 500_000):
            blk = np.asarray(tokens[c0:c0 + 500_000])
            absent_nonzero += int((blk[~present[c0:c0 + 500_000]] != 0).any(1).sum())
    A = {"shape_ok": ok_shape, "present_ok": ok_present, "n_present": int(present.sum()),
         "present_expected": int(want_present.sum()), "tokens_sha_bad": sha_bad, "stems_bad": stems_bad,
         "range_bad": range_bad, "absent_rows_nonzero": absent_nonzero,
         "control_present_flipped_refused": ctrl_fires, "seconds": round(time.time() - t0, 1)}
    A["pass"] = bool(ok_shape and ok_present and not sha_bad and not stems_bad and not range_bad
                     and (absent_nonzero in (None, 0)) and ctrl_fires)
    rep["checks"]["A_integrity"] = A
    if not A["pass"]:
        fails.append("A")
    print(f"A integrity: {'PASS' if A['pass'] else 'FAIL'} {A}", flush=True)

    # ---- sample rows and re-read their labels -------------------------------------------------------------------
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    sample_rows, sample_lab = [], []
    for k, s in enumerate(shards):
        rows = idx[idx.source_shard == s].sort_values("row")
        pick = np.sort(rng.choice(len(rows), size=min(a.per_shard, len(rows)), replace=False))
        # each picked row and the row after it (for control C's neighbour labels), inside the shard
        pick = np.unique(np.concatenate([pick, np.minimum(pick + 1, len(rows) - 1)]))
        sub = rows.iloc[pick]
        got = read_rows(a.raw, s, set(sub.stem))
        sample_rows.append(sub.row.values)
        sample_lab.append(np.stack([got[st] for st in sub.stem]))
    rows_all = np.concatenate(sample_rows)
    lab_all = np.concatenate(sample_lab)
    art_codes = np.asarray(tokens[rows_all]).astype(np.int64)
    print(f"sampled {len(rows_all)} rows from {len(shards)} shards in {time.time() - t0:.0f} s", flush=True)

    # ---- B: the harness's own encoding --------------------------------------------------------------------------
    t0 = time.time()
    hc = harness_codes(tok, lab_all)
    eq = hc == art_codes
    rot = tuple([0] + [c + 1 if c < 9 else 1 for c in range(1, 10)])
    P.validate(P.UNIVERSAL, rot, P.TASKS[TASK].ignore, P.TASKS[TASK].n_classes)
    b1 = float((hc[1:] == art_codes[:-1]).mean())
    b2 = float((harness_codes(tok, np.ascontiguousarray(lab_all.transpose(0, 2, 1))) == art_codes).mean())
    b3 = float((harness_codes(tok, lab_all, assign=rot) == art_codes).mean())
    B = {"rows": int(len(rows_all)), "codes_equal": float(eq.mean()), "rows_all_equal": float(eq.all(1).mean()),
         "control_B1_one_row_off_agreement": b1, "control_B2_transposed_agreement": b2,
         "control_B3_colours_rotated_agreement": b3, "seconds": round(time.time() - t0, 1)}
    B["pass"] = bool(eq.all() and b1 < CONTROL_AGREE and b2 < CONTROL_AGREE and b3 < CONTROL_AGREE)
    rep["checks"]["B_harness"] = B
    if not B["pass"]:
        fails.append("B")
    print(f"B harness: {'PASS' if B['pass'] else 'FAIL'} {B}", flush=True)

    # ---- C: round trip -----------------------------------------------------------------------------------------
    t0 = time.time()
    rt = roundtrip(tok, art_codes, lab_all)
    # control: each decoded map against the labels of the next sampled row of its shard (rolled within the sample)
    nxt = np.concatenate([np.roll(np.arange(len(r)), -1) + off for r, off in
                          zip(sample_rows, np.cumsum([0] + [len(r) for r in sample_rows[:-1]]))])
    rt_ctrl = roundtrip(tok, art_codes, lab_all[nxt])
    Cc = {"mIoU": rt["mIoU"], "pixel_acc": rt["pixel_acc"], "bar": CEILING_BAR,
          "control_C_next_row_mIoU": rt_ctrl["mIoU"], "control_bar": CONTROL_MIOU, "maps": rt["n_maps"],
          "seconds": round(time.time() - t0, 1)}
    Cc["pass"] = bool(rt["mIoU"] >= CEILING_BAR and rt_ctrl["mIoU"] < CONTROL_MIOU)
    rep["checks"]["C_roundtrip"] = Cc
    if not Cc["pass"]:
        fails.append("C")
    print(f"C round trip: {'PASS' if Cc['pass'] else 'FAIL'} {Cc}", flush=True)

    # ---- code statistics (description) --------------------------------------------------------------------------
    used = np.unique(art_codes)
    rep["codes_in_sample"] = {"distinct": int(len(used)), "of": n_codes,
                              "entropy_bits": float(-(lambda p: (p * np.log2(p)).sum())(
                                  np.bincount(art_codes.reshape(-1), minlength=n_codes)[np.bincount(
                                      art_codes.reshape(-1), minlength=n_codes) > 0] / art_codes.size))}
    rep["failed"] = fails
    rep["pass"] = not fails
    rep["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORTS.mkdir(parents=True, exist_ok=True)
    outp = REPORTS / f"verify_{card['split']}_{a.tag}.json"
    with open(str(outp) + ".tmp", "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    Path(str(outp) + ".tmp").replace(outp)
    print(f"{'PASS' if not fails else 'FAIL ' + ','.join(fails)} -> {outp}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
