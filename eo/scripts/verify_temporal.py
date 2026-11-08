#!/usr/bin/env python
"""The temporal stage's gate (`.venv`, CPU).

Same discipline as verify_phase3.py and verify_phase6.py: every check ships with a deliberately broken
control that proves it can fail, and a check lands with the step that produces what it reads. A check
whose artifact does not exist yet is SKIPPED and reported as such, never passed.

    python eo/scripts/verify_temporal.py
    python eo/scripts/verify_temporal.py --only Y1

  Y1  temporal_vocab: every id below the date rows maps to itself and every date id is used once;
      a date block round-trips for every date, with and without its year, from dates.npy's uint8;
      UNTRAINED_YEARS describes both dates.npy
                                     | date rows before BEN's, on PAD, past a gap, a repeated id and a
                                       missing bin are rejected; date blocks out of range or out of
                                       order are refused; the wrong untrained years are flagged
  Y9  The extended base: extension.json records a passed fp32 identity and both controls fired; its
      weights hash to the record and carry the current layout; the table is 87,664 rows and every
      tensor of the source checkpoint is unchanged, bit for bit; lm_head is tied, so not stored
                                     | a failed identity, an unfired control, another size, a changed
                                       weights file and one moved row are all refused
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eo.data import ben_vocab as BV  # noqa: E402
from eo.data import eo_vocab as EV  # noqa: E402
from eo.data import temporal_vocab as TV  # noqa: E402
from eo.data.temporal_index import N_DOY_BINS, N_YEARS, load_temporal_index  # noqa: E402

ROOTS = {"val_clean": Path("/data/enric/data/TerraMesh/val_clean"),
         "train": Path("/data/enric/data/TerraMesh/train_coords")}
EXT = Path("/data/enric/runs/pretrain/temporal/base/t360_ext_temporal_ckpt17441")
TABLE = "model.embed_tokens.weight"

RESULTS = []


def record(check, ok, detail, ctrl_ok=None, ctrl_detail=""):
    RESULTS.append((check, ok, ctrl_ok))
    print(f"{'PASS' if ok else 'FAIL'}  {check}: {detail}")
    if ctrl_ok is not None:
        print(f"      control {'fired' if ctrl_ok else 'DID NOT FIRE'}: {ctrl_detail}")


def skip(check, why):
    RESULTS.append((check, None, None))
    print(f"SKIP  {check}: {why}")


def refused(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except (ValueError, RuntimeError):
        return True
    return False


# ---------------------------------------------------------------- Y1
def y1():
    probs = TV.layout_problems(TV.YEAR_IDS, TV.DAY_IDS, TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE)
    same = all(TV.MODALITIES[n] == i for n, i in BV.MODALITIES.items()) \
        and all(TV.MODALITIES[n].bo_id == i.bo_id and TV.MODALITIES[n].codebook_offset == i.codebook_offset
                for n, i in EV.MODALITIES.items())
    ids_same = all(TV.MODALITY_TO_ID[n] == i for n, i in BV.MODALITY_TO_ID.items()) and TV.MODALITY_TO_ID["Date"] == 9
    used = TV.YEAR_IDS + TV.DAY_IDS + [TV.BO_DATE, TV.EO_DATE]
    ok = (not probs and same and ids_same and TV.PAD_ID == EV.PAD_ID and TV.BASE == BV.TOTAL_VOCAB_SIZE
          and sorted(used) == list(range(TV.BASE, TV.TOTAL_VOCAB_SIZE)))

    y, d = TV.YEAR_IDS, TV.DAY_IDS
    broken = {
        "date rows before BEN's": ([EV.TOTAL_VOCAB_SIZE + i for i in range(N_YEARS)],
                                   [EV.TOTAL_VOCAB_SIZE + N_YEARS + i for i in range(N_DOY_BINS)],
                                   EV.TOTAL_VOCAB_SIZE + 85, EV.TOTAL_VOCAB_SIZE + 86, TV.TOTAL_VOCAB_SIZE),
        "start at PAD": ([EV.PAD_ID + i for i in range(N_YEARS)], d, TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE),
        "gap after BEN's": ([i + 1 for i in y], [i + 1 for i in d], TV.BO_DATE + 1, TV.EO_DATE + 1,
                            TV.TOTAL_VOCAB_SIZE + 1),
        "repeated id": ([y[0]] * N_YEARS, d, TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE),
        "a day bin missing": (y, d[:-1], TV.BO_DATE, TV.EO_DATE, TV.TOTAL_VOCAB_SIZE),
    }
    fired = {k: bool(TV.layout_problems(*v)) for k, v in broken.items()}
    record("Y1 layout", ok, f"{TV.N_NEW} date ids {TV.BASE}..{TV.TOTAL_VOCAB_SIZE - 1}, PAD {TV.PAD_ID} and every "
           f"id below {TV.BASE} unchanged, Date is modality {TV.MODALITY_TO_ID['Date']}", all(fired.values()),
           ", ".join(f"{k}: {'rejected' if v else 'ACCEPTED'}" for k, v in fired.items()))

    # Every date, both forms, through dates.npy's type. Offsets added to a uint8 would wrap.
    n, good = 0, True
    for yi in np.arange(N_YEARS, dtype=np.uint8):
        for di in np.arange(N_DOY_BINS, dtype=np.uint8):
            for with_year in (True, False):
                n += 1
                block = TV.date_block(yi, di, with_year=with_year)
                good &= all(type(t) is int for t in block) and block[0] == TV.BO_DATE and block[-1] == TV.EO_DATE
                good &= TV.decode_date_block(block) == (int(yi) if with_year else None, int(di))
    bad_in = [("year 12", lambda: TV.date_block(N_YEARS, 0)), ("day bin 73", lambda: TV.date_block(0, N_DOY_BINS)),
              ("year after day", lambda: TV.decode_date_block([TV.BO_DATE, TV.DAY_BASE, TV.YEAR_BASE, TV.EO_DATE])),
              ("no EO_Date", lambda: TV.decode_date_block([TV.BO_DATE, TV.YEAR_BASE, TV.DAY_BASE])),
              ("a BEN id inside", lambda: TV.decode_date_block([TV.BO_DATE, BV.BASE, TV.DAY_BASE, TV.EO_DATE])),
              ("empty body", lambda: TV.decode_date_block([TV.BO_DATE, TV.EO_DATE]))]
    fired = {k: refused(f) for k, f in bad_in}
    record("Y1 date blocks", bool(good), f"{n} blocks round-trip from uint8", all(fired.values()),
           ", ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in fired.items()))

    seen, sizes = set(), {}
    for name, root in ROOTS.items():
        dates = np.asarray(load_temporal_index(root).dates)
        years, counts = np.unique(dates[:, 0], return_counts=True)
        seen |= set(int(v) for v in years)
        sizes[name] = len(dates)
    problems = TV.coverage_problems(seen)
    wrong = {"2020 listed as untrained": TV.coverage_problems(seen, (2015, 2020, 2025, 2026)),
             "2026 not listed": TV.coverage_problems(seen, (2015, 2025))}
    n_ids = len(TV.UNTRAINED_IDS)
    ok = not problems and n_ids == 24 and all(BV.BASE <= i < TV.BASE + N_YEARS for i in TV.UNTRAINED_IDS)
    record("Y1 untrained rows", ok, f"{n_ids} ids: BEN's 21 and years {TV.UNTRAINED_YEARS}, none of which has a row "
           f"in {sizes}", all(bool(v) for v in wrong.values()),
           ", ".join(f"{k}: {'flagged' if v else 'NOT FLAGGED'}" for k, v in wrong.items()))


# ---------------------------------------------------------------- Y9
def y9(ext: Path = EXT):
    if not (ext / "extension.json").is_file():
        skip("Y9 extended base", f"{ext} has no extension.json: Step 3's extension has not run")
        return
    from safetensors import safe_open

    from eo.finetune.init import verify_extension

    doc = verify_extension(ext, TV.TOTAL_VOCAB_SIZE, check_hash=True)
    ident, c1 = doc["identity"], doc["control_insert_before_pad"]
    ok = (doc["layout"] == "temporal" and doc["temporal_vocab"] == TV.layout_record() and doc["tied"]
          and ident["precision"] == "fp32" and ident["max_abs_diff"] <= ident["tol"] < c1["max_abs_diff"])
    record("Y9 record", ok, f"identity max |dlogit| {ident['max_abs_diff']:.3e} on {ident['positions']} positions, "
           f"insert before PAD {c1['max_abs_diff']:.2f}, device {doc['gpu']}, sha {doc['weights_sha256'][:12]}")

    source = Path(doc["source_checkpoint"]) / "model.safetensors"
    with safe_open(str(source), "pt") as old, safe_open(str(ext / doc["weights_file"]), "pt") as new:
        names_old, names_new = set(old.keys()), set(new.keys())
        moved = [k for k in sorted(names_old - {TABLE}) if not torch.equal(old.get_tensor(k), new.get_tensor(k))]
        a, b = old.get_tensor(TABLE), new.get_tensor(TABLE)
    n_old = a.shape[0]
    same_rows = bool(torch.equal(a, b[:n_old]))
    fresh = b[n_old:]
    expected = a.float().mean(0, keepdim=True)
    near_mean = float((fresh - expected).abs().max() / a.float().std(0).max())
    ok = (names_old == names_new and not moved and same_rows and b.shape == (TV.TOTAL_VOCAB_SIZE, a.shape[1])
          and "lm_head.weight" not in names_new and near_mean < 0.1)
    doctored = b.clone()
    doctored[1234, 7] += 1e-3
    shifted = torch.cat([b[:EV.PAD_ID], fresh, b[EV.PAD_ID:n_old]], 0)
    fired = {"one moved row": not torch.equal(a, doctored[:n_old]), "rows before PAD": not torch.equal(a, shifted[:n_old])}
    record("Y9 weights", ok, f"{len(names_new)} tensors, {len(moved)} moved; table {tuple(b.shape)}, rows below {n_old} "
           f"bit-identical, {len(fresh)} new rows at the mean (largest move {near_mean:.3f} of a std), lm_head not stored",
           all(fired.values()), ", ".join(f"{k}: {'seen' if v else 'MISSED'}" for k, v in fired.items()))

    tmp = tempfile.TemporaryDirectory(prefix="y9_")

    def fake(mutate, change_weights=False):
        d = Path(tempfile.mkdtemp(dir=tmp.name))
        w = doc["weights_file"]
        (d / w).symlink_to(ext / w)
        fake_doc = json.loads(json.dumps(doc))
        mutate(fake_doc)
        if change_weights:
            fake_doc["weights_sha256"] = "0" * 64
        (d / "extension.json").write_text(json.dumps(fake_doc))
        return d

    def set_key(*path, value):
        def go(x):
            for k in path[:-1]:
                x = x[k]
            x[path[-1]] = value
        return go

    cases = {"identity failed": (fake(set_key("identity", "passed", value=False)), {}),
             "control 1 did not fire": (fake(set_key("control_insert_before_pad", "fired", value=False)), {}),
             "control 2 did not fire": (fake(set_key("control_untied_resize", "fired", value=False)), {}),
             "another size": (fake(set_key("extension", "vocab_after", value=BV.TOTAL_VOCAB_SIZE)), {}),
             "weights changed": (fake(lambda x: None, change_weights=True), {"check_hash": True})}
    fired = {k: refused(verify_extension, d, TV.TOTAL_VOCAB_SIZE, **{"check_hash": False, **kw})
             for k, (d, kw) in cases.items()}
    tmp.cleanup()
    record("Y9 refusals", refused(verify_extension, ext, BV.TOTAL_VOCAB_SIZE, check_hash=False),
           "the base is refused at BEN's size", all(fired.values()),
           ", ".join(f"{k}: {'refused' if v else 'ACCEPTED'}" for k, v in fired.items()))


CHECKS = {"Y1": y1, "Y9": y9}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(CHECKS), default=sorted(CHECKS))
    args = ap.parse_args()
    for name in args.only:
        CHECKS[name]()
    failed = [c for c, ok, ctrl in RESULTS if ok is False or ctrl is False]
    skipped = [c for c, ok, _ in RESULTS if ok is None]
    print(f"\n{len(RESULTS) - len(failed) - len(skipped)} passed, {len(failed)} failed, {len(skipped)} skipped"
          + (f": {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
