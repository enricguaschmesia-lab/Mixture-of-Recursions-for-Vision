#!/usr/bin/env python
"""Phase 2 Step 6 -- the verification gate (D2.6).

Every check ships with a deliberately-broken control that proves the check can
fail (CLAUDE.md rule 2). A check that cannot fail is not evidence, and this
phase produced a concrete example: an assert added during Step 3 to guard the
Coords slot turned out to be vacuous by construction, and only writing its
control revealed that. See V8.

V0-V6 are the checks PHASE2_PLAN.md section 6 specifies. V0-V5 are imported from
verify_step7.py rather than reimplemented -- that script is Phase 1's D1.8 gate
and its checks are exactly the ones Phase 2 needs, so duplicating them would
create two things to keep in step.

V7-V10 are ADDITIONAL, each closing a hole that a specific Step 1-5 finding
exposed. They are not in the plan because the plan was written before the
findings existed:

  V7  crop consistency        Step 1. Two token sets now coexist on disk and
                              the plan (section 4.1) promised a load-time crop
                              assert that the structural directory naming did
                              not actually replace.
  V8  vocabulary covers data  Step 3. The vocabulary's import-time asserts
                              CANNOT catch a wrong codebook size -- sizes are
                              inputs to the derivation, so any sizes are
                              self-consistent. Verified by mutation.
  V9  sequence budget         Step 4. Coords was silently absent from every
                              sequence for weeks and no check noticed, because
                              V1-V5 all validate whatever active_modalities
                              happens to contain.
  V10 shipped config          Step 5. The config is the thing an actual run
                              consumes; nothing checked it.

Runs in the repo .venv, NOT the `mor` env (it imports the dataloader and the
model, never terratorch).

    HF_HOME=/data/enric/hf ./.venv/bin/python eo/scripts/verify_phase2.py
    ./.venv/bin/python eo/scripts/verify_phase2.py --skip-forward   # no GPU
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# V6 forks dataloader workers after the tokenizer has been touched; the warning
# is noise, and the parallelism it disables is not used here.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch

import eo.scripts.verify_step7 as s7  # noqa: E402  (sets up sys.path itself)

EO_CONFIG = REPO / "conf/pretrain_vision/eo_terramesh/terramesh_mor_token.yaml"


def _deps():
    from eo.mor_data import eo_vocab
    from eo.mor_data.terramesh_token_dataset import (
        TerraMeshTokenDataset, assert_artifact_crop)
    from eo.terramesh_tok import contract
    return eo_vocab, TerraMeshTokenDataset, assert_artifact_crop, contract


def _fake_artifact(tmp: Path, modality: str, meta: dict, contract) -> Path:
    """A metadata.json-only artifact directory, for controls."""
    d = tmp / contract.tok_dir_name(modality)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


# --------------------------------------------------------------------------
# V3 (extended) -- presence, AND the active modality set itself
# --------------------------------------------------------------------------
def v3_presence_and_modality_set() -> bool:
    """The plan's V3, plus the hole Step 4 exposed.

    V1-V5 all validate whatever `active_modalities` contains. That makes them
    structurally blind to an OMITTED modality: when Coords was missing, every
    sequence was a self-consistent 990 tokens and every check passed. So the
    active set has to be asserted against the ratified list directly, not
    inferred from the sequences being internally consistent.
    """
    eo_vocab, DS, _, _ = _deps()
    base_ok = s7.v3_presence()          # S1GRD/S1RTC complementarity

    ratified = {"S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"}
    ds = DS(modality_order="random")
    active = set(ds.active_modalities)
    set_ok = active == ratified
    print(f"    default active set == ratified set (Meeting 2) = {set_ok}")
    if not set_ok:
        print(f"      missing: {sorted(ratified - active)}  extra: {sorted(active - ratified)}")

    # Coords present on EVERY row, and no row can produce an empty sequence.
    coords_all = bool(ds._present["Coords"].all())
    any_present = np.stack([ds._present[m] for m in ds.active_modalities]).any(0)
    never_empty = bool(any_present.all())
    print(f"    Coords present on all {ds.n_rows} rows = {coords_all}")
    print(f"    no row produces an empty sequence      = {never_empty}")

    # Every sequence actually carries a Coords chunk.
    rng = np.random.default_rng(3)
    cid = eo_vocab.MODALITY_TO_ID["Coords"]
    carried = all(int((ds[int(r)]["modality_ids"] == cid).sum())
                  == eo_vocab.get_modality("Coords").tokens_per_sample
                  for r in rng.choice(len(ds), size=50, replace=False))
    print(f"    50 sampled sequences each carry Coords = {carried}")

    # CONTROL: an active set missing Coords must be rejected by this check,
    # even though every other check would still pass on it.
    ctrl_ds = DS(active_modalities=[m for m in ds.active_modalities if m != "Coords"])
    ctrl_detected = set(ctrl_ds.active_modalities) != ratified and ctrl_ds.required_length == 990
    print(f"    control: image-only set detected ({ctrl_ds.required_length} tokens, "
          f"was the real bug) = {ctrl_detected}")

    return base_ok and set_ok and coords_all and never_empty and carried and ctrl_detected


# --------------------------------------------------------------------------
# V6 -- determinism at a fixed worker count
# --------------------------------------------------------------------------
def v6_worker_determinism(n_batches: int = 3, bs: int = 4) -> bool:
    """Modality ORDER depends on dataloader_num_workers; token CONTENT does not.

    `_get_rng` seeds on (seed, worker_id, idx), inherited from the CLEVR dataset.
    So a run is bit-reproducible only at a FIXED worker count. This is a
    property to pin down, not a bug to fix -- but it has to be stated and
    re-checked, because it silently invalidates a "reproducible" claim.
    """
    from torch.utils.data import DataLoader
    _, DS, _, _ = _deps()

    def run(nw):
        dl = DataLoader(DS(modality_order="random", seed=42),
                        batch_size=bs, num_workers=nw, shuffle=False)
        out = []
        for i, b in enumerate(dl):
            if i >= n_batches:
                break
            out.append(b)
        return out

    a1, a2, b = run(0), run(0), run(2)

    same_run = all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a1, a2))
    print(f"    same worker count, two runs -> bit-identical = {same_run}")

    content_same = all(
        sorted(x["input_ids"][r].tolist()) == sorted(y["input_ids"][r].tolist())
        for x, y in zip(a1, b) for r in range(x["input_ids"].shape[0]))
    order_differs = any(not torch.equal(x["modality_ids"], y["modality_ids"])
                        for x, y in zip(a1, b))
    print(f"    workers 0 vs 2 -> token multiset per row unchanged = {content_same}")
    print(f"    workers 0 vs 2 -> modality ORDER stream differs    = {order_differs}")

    # CONTROL: the order stream must be the thing that moved. If nothing
    # differed, this check would be asserting a property it never exercised.
    print(f"    control: worker-count change is observable = {order_differs}")
    return same_run and content_same and order_differs


# --------------------------------------------------------------------------
# V7 -- crop consistency (ADDITIONAL: Step 1)
# --------------------------------------------------------------------------
def v7_crop() -> bool:
    """Every loaded artifact was tokenized at the contract crop.

    PHASE2_PLAN section 4.1: "make the dataloader assert metadata.json's crop
    against contract.CROP at load. The assert is the real safety mechanism; the
    directory naming is just hygiene." The naming was done in Step 1; the assert
    was not, until Step 6.

    A crop mismatch is silent: ids are in range, the sequence assembles, the
    loss is finite. It is simply a different scene geometry from the rest of the
    batch.
    """
    eo_vocab, DS, assert_crop, contract = _deps()
    ds = DS(modality_order="random")

    crops = assert_crop(ds.root_dir, ds.active_modalities)
    image_ok = all(v == contract.CROP for k, v in crops.items() if k != "Coords")
    coords_ok = crops.get("Coords", "missing") is None
    print(f"    six image modalities at crop {contract.CROP} = {image_ok}")
    print(f"    Coords records crop=null (crop-independent) = {coords_ok}")

    # Metadata must agree with the contract's derived geometry too.
    geom_ok = True
    for m in ds.active_modalities:
        meta = json.loads((Path(ds.root_dir) / contract.tok_dir_name(m) / "metadata.json").read_text())
        if meta.get("crop") is None:
            continue
        if int(meta["tokens_per_sample"]) != contract.TOKENS_PER_SAMPLE or \
           list(meta["grid"]) != [contract.GRID, contract.GRID]:
            geom_ok = False
    print(f"    grid/tokens_per_sample agree with contract    = {geom_ok}")

    tmp = Path(tempfile.mkdtemp())
    try:
        # CONTROL 1: an artifact from the OTHER crop must be rejected.
        _fake_artifact(tmp, "DEM", {"crop": {"crop": 256, "offset": 4},
                                    "tokens_per_sample": 256, "grid": [16, 16]}, contract)
        try:
            assert_crop(tmp, ["DEM"]); c1 = False
        except RuntimeError as e:
            c1 = "256" in str(e)
        print(f"    control: a 256-crop artifact is rejected     = {c1}")

        # CONTROL 2: metadata with no crop field at all must be rejected, NOT
        # skipped -- "no crop recorded" must never read as "any crop is fine".
        _fake_artifact(tmp, "NDVI", {"tokens_per_sample": 196}, contract)
        try:
            assert_crop(tmp, ["NDVI"]); c2 = False
        except RuntimeError as e:
            c2 = "records no crop" in str(e)
        print(f"    control: artifact with no crop field rejected= {c2}")

        # CONTROL 3: a hand-edited crop field that disagrees with the token
        # count must be caught (internally inconsistent metadata).
        _fake_artifact(tmp, "LULC", {"crop": {"crop": contract.CROP},
                                     "tokens_per_sample": 256}, contract)
        try:
            assert_crop(tmp, ["LULC"]); c3 = False
        except RuntimeError as e:
            c3 = "internally inconsistent" in str(e)
        print(f"    control: crop/token-count mismatch rejected  = {c3}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return image_ok and coords_ok and geom_ok and c1 and c2 and c3


# --------------------------------------------------------------------------
# V8 -- the vocabulary covers the data (ADDITIONAL: Step 3)
# --------------------------------------------------------------------------
def v8_vocab_covers_artifacts() -> bool:
    """Slot sizes vs what the tokenizers can actually emit.

    THE POINT OF THIS CHECK. eo_vocab's import-time asserts verify that
    build_vocab()'s DERIVATION is self-consistent. They cannot validate the
    codebook SIZES, because the sizes are inputs: BO/EO begins wherever the
    codebooks end and PAD follows, so ANY set of sizes yields a consistent
    layout. Setting Coords back to the buggy 6,365 trips none of them.

    That blind spot is asserted here as a negative control, so that if someone
    later adds an import-time assert believing it covers this, the negative
    control fails and tells them it does not.
    """
    eo_vocab, DS, _, contract = _deps()
    ds = DS(modality_order="random")

    seen = eo_vocab.assert_artifact_fits(ds.root_dir, ds.active_modalities)
    fits = all(seen[m] < eo_vocab.get_modality(m).codebook_size for m in seen)
    print(f"    {len(seen)} artifacts' max ids fit their slots = {fits}")

    # The Coords slot must cover the TOKENIZER's range, not just observed ids.
    meta = json.loads((Path(ds.root_dir) / contract.tok_dir_name("Coords") / "metadata.json").read_text())
    bound = meta.get("tokenizer_id_bound")
    slot = eo_vocab.get_modality("Coords").codebook_size
    bound_ok = bound is not None and slot >= int(bound)
    print(f"    Coords slot {slot} >= tokenizer_id_bound {bound} = {bound_ok}")
    print(f"      (observed max is only {meta.get('observed_id_max')} -- checking "
          f"that instead is what let the slot be one id short)")

    tmp = Path(tempfile.mkdtemp())
    try:
        # CONTROL 1: a tokenizer bound wider than the slot must be rejected.
        _fake_artifact(tmp, "Coords", {"crop": None, "observed_id_max": 6364,
                                       "tokenizer_id_bound": slot + 1}, contract)
        try:
            eo_vocab.assert_artifact_fits(tmp, ["Coords"]); c1 = False
        except RuntimeError as e:
            c1 = "tokenizer can emit ids up to" in str(e)
        print(f"    control: tokenizer bound > slot rejected     = {c1}")

        # CONTROL 2: an observed id past the slot must be rejected too.
        _fake_artifact(tmp, "Coords", {"crop": None, "observed_id_max": slot}, contract)
        try:
            eo_vocab.assert_artifact_fits(tmp, ["Coords"]); c2 = False
        except RuntimeError as e:
            c2 = "silently" in str(e)
        print(f"    control: observed id past slot rejected      = {c2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # NEGATIVE CONTROL: confirm the documented blind spot is still the truth.
    # A wrong slot size must NOT be caught by import-time asserts -- if this
    # starts failing, the asserts changed and this comment is stale.
    import types
    src = (REPO / "eo/mor_data/eo_vocab.py").read_text().replace(
        "COORDS_CODEBOOK = 6366", "COORDS_CODEBOOK = 6365")
    mod = types.ModuleType("eo_vocab_mutant"); mod.__file__ = str(REPO / "eo/mor_data/eo_vocab.py")
    sys.modules["eo_vocab_mutant"] = mod
    try:
        exec(compile(src, "eo_vocab_mutant", "exec"), mod.__dict__)
        blind = True      # imported clean: the blind spot is real, as documented
    except AssertionError:
        blind = False     # an import-time assert caught it -- docs are now stale
    finally:
        sys.modules.pop("eo_vocab_mutant", None)
    print(f"    negative control: a wrong slot size is NOT caught by the")
    print(f"      import-time asserts, exactly as documented = {blind}")

    return fits and bound_ok and c1 and c2 and blind


# --------------------------------------------------------------------------
# V9 -- sequence budget and Coords placement (ADDITIONAL: Step 4)
# --------------------------------------------------------------------------
def v9_sequence_budget(n: int = 300) -> bool:
    """995 on every row, and Coords in the shuffle rather than pinned."""
    eo_vocab, DS, _, _ = _deps()
    ds = DS(modality_order="random")

    lens = np.unique(ds._seq_lens)
    single = len(lens) == 1 and int(lens[0]) == 995
    print(f"    every row is 995 tokens (distinct lengths: {lens.tolist()}) = {single}")
    print(f"      -> padding is zero and batch shapes are constant")

    # max_length below the budget must be refused, not silently truncated.
    try:
        DS(max_length=990); refused = False          # the old image-only length
    except ValueError:
        refused = True
    print(f"    max_length=990 refused rather than truncating = {refused}")

    def coords_positions(order_mode, k):
        pos = np.zeros(7, dtype=int)
        d = DS(modality_order=order_mode)
        for i in range(k):
            mids = d[i]["modality_ids"]
            first = {}
            for t, m in enumerate(mids.tolist()):
                if m and m not in first:
                    first[m] = t
            order = [eo_vocab.ID_TO_MODALITY[m] for m, _ in sorted(first.items(), key=lambda kv: kv[1])]
            pos[order.index("Coords")] += 1
        return pos

    p_rand = coords_positions("random", n)
    occupied = int((p_rand > 0).sum())
    max_share = p_rand.max() / p_rand.sum()
    shuffled = occupied >= 4 and max_share < 0.60
    print(f"    Coords chunk index over {n} samples: {p_rand[:6].tolist()}")
    print(f"      occupies {occupied} positions, max share {max_share:.1%} -> shuffled = {shuffled}")

    # CONTROL: modality_order='fixed' IS the pinned layout this check must
    # reject -- Coords lands in the same slot every time.
    p_fixed = coords_positions("fixed", 40)
    pinned_detected = (p_fixed > 0).sum() == 1
    print(f"    control: fixed order pins Coords to one slot, detected = {pinned_detected}")

    return single and refused and shuffled and pinned_detected


# --------------------------------------------------------------------------
# V10 -- the shipped EO config (ADDITIONAL: Step 5)
# --------------------------------------------------------------------------
def v10_config() -> bool:
    """The config an actual run consumes must satisfy 5.1, 5.2 and 5.3."""
    from omegaconf import OmegaConf
    from lm_dataset.modality_registry import assert_vocab_size, get_id_to_modality, is_eo
    eo_vocab, _, _, contract = _deps()

    if not EO_CONFIG.exists():
        print(f"    FAIL: {EO_CONFIG} does not exist")
        return False
    cfg = OmegaConf.load(EO_CONFIG)

    eo_path = is_eo(cfg)
    vocab_ok = assert_vocab_size(cfg) == eo_vocab.TOTAL_VOCAB_SIZE
    print(f"    config is on the EO registry path             = {eo_path}")
    print(f"    vocab_size == registry ({eo_vocab.TOTAL_VOCAB_SIZE})           = {vocab_ok}")

    names = set(get_id_to_modality(cfg).values())
    names_ok = names == {"S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC", "Coords"}
    no_clevr = not any("tok_" in n or n in ("caption", "scene_desc") for n in names)
    print(f"    per-modality loss keys are EO names (5.2)     = {names_ok and no_clevr}")

    ve_off = cfg.vision_eval.enable is False
    grid_ok = int(cfg.vision_eval.patch_grid_size) == contract.GRID
    print(f"    vision_eval disabled on the EO path (5.3)     = {ve_off}")
    print(f"    patch_grid_size records contract GRID {contract.GRID}       = {grid_ok}")

    ml_ok = int(cfg.max_length) >= 995
    print(f"    max_length {int(cfg.max_length)} >= the 995 budget            = {ml_ok}")

    # CONTROL 1: CLEVR's vocab_size on the EO path must be rejected.
    bad = OmegaConf.merge(cfg, {"model_config": {"vocab_size": 242271}})
    try:
        assert_vocab_size(bad); c1 = False
    except ValueError:
        c1 = True
    print(f"    control: CLEVR's 242271 rejected here         = {c1}")

    # CONTROL 2: enabling vision_eval on the EO path must refuse to construct.
    from util.callback import MultimodalVisionEvalCallback
    try:
        MultimodalVisionEvalCallback(OmegaConf.merge(cfg, {"vision_eval": {"enable": True}}))
        c2 = False
    except ValueError:
        c2 = True
    print(f"    control: enabling vision_eval refuses to build= {c2}")

    return eo_path and vocab_ok and names_ok and no_clevr and ve_off and grid_ok and ml_ok and c1 and c2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-forward", action="store_true",
                    help="skip V5 (the only check needing a GPU)")
    args = ap.parse_args()

    checks = [
        ("V0   CLEVR bit-identity across shared-code changes", s7.v0_check),
        ("V1   sequence round-trips to the 224 tokens.npy rows", s7.v1_roundtrip),
        ("V2   every id inside its own modality slot", s7.v2_range),
        ("V3   presence + the active modality set itself", v3_presence_and_modality_set),
        ("V4   mask / label / position invariants", s7.v4_masks),
    ]
    if not args.skip_forward:
        checks.append(("V5   forward pass; untrained loss ~ ln(V)", s7.v5_forward))
    checks += [
        ("V6   determinism at a fixed worker count", v6_worker_determinism),
        ("V7   crop consistency            [added: Step 1]", v7_crop),
        ("V8   vocabulary covers the data  [added: Step 3]", v8_vocab_covers_artifacts),
        ("V9   sequence budget + placement [added: Step 4]", v9_sequence_budget),
        ("V10  the shipped EO config       [added: Step 5]", v10_config),
    ]

    results = {}
    for title, fn in checks:
        print(f"\n{title}")
        try:
            results[title] = bool(fn())
        except Exception:
            import traceback
            traceback.print_exc()
            results[title] = False

    print("\n" + "=" * 70)
    for title, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {title}")
    failed = [t for t, ok in results.items() if not ok]
    print("=" * 70)
    print("PHASE 2 GATE PASSED -- all checks pass and all controls fire"
          if not failed else f"{len(failed)} CHECK(S) FAILED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
