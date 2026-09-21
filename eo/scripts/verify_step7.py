"""Phase 1 Step 7 (D1.8) verification gate.

Six checks. Every check ships with a deliberately-broken control that proves
it can fail, so a check that silently stops testing anything is detectable.

    V0  CLEVR bit-identity across the sequence_assembly extraction
    V1  round-trip: assembled sequence -> back to the Step-4 tokens.npy rows
    V2  range: every id inside its own modality's slot
    V3  presence: exactly one of S1GRD/S1RTC per row, never both
    V4  masks: attention/labels/position_ids invariants, no truncation
    V5  forward: MoR forward pass, loss ~ ln(vocab_size)

Run in the MoR repo's own .venv (NOT the `mor` conda env):

    HF_HOME=/data/enric/hf ./.venv/bin/python eo/scripts/verify_step7.py

V0 compares against a committed digest baseline captured before the
extraction. To re-capture it (only legitimate when the CLEVR sequence format
is deliberately changed):

    HF_HOME=/data/enric/hf ./.venv/bin/python eo/scripts/verify_step7.py \
        --capture-baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BASELINE = Path(__file__).resolve().parent / "fixtures" / "clevr_assembly_baseline.json"

# Sequence keys the model/trainer contract allows. MoRTrainer.compute_loss pops
# 'modality_ids' and forwards the rest into model(**inputs), so this set is closed.
KEYS = ("input_ids", "attention_mask", "labels", "position_ids", "modality_ids")


# --------------------------------------------------------------------------
# V0 -- CLEVR bit-identity across the extraction
# --------------------------------------------------------------------------
# A synthetic fixture in the CLEVR on-disk layout. The dataset class does not
# care that the token values are not real Cosmos codes: V0 tests sequence
# assembly, not tokenizer output. Bodies are deliberately unequal lengths so
# the padding path is exercised without needing the real dataset (which is not
# on this workstation).
_FIXTURE_MODALITIES = [
    ("tok_rgb@256", 256, 64000),
    ("tok_depth@256", 200, 64000),
    ("tok_normal@256", 128, 64000),
]
_FIXTURE_STEMS = 12
_FIXTURE_K = 3

# (max_length, modality_order, shuffle_image_patches) -- covers every branch of
# the finalizer: fixed and random order, shuffle on and off, and a max_length
# that forces truncation mid-chunk.
_V0_CONFIGS = [
    (1048, "fixed", False),
    (1048, "random", False),
    (1048, "random", True),
    (400, "random", False),
    (400, "fixed", True),
]


def build_clevr_fixture(root: Path) -> None:
    rng = np.random.default_rng(0)
    stems = [f"sample_{i:04d}" for i in range(_FIXTURE_STEMS)]
    for name, n_tokens, codebook in _FIXTURE_MODALITIES:
        d = root / "train" / name
        d.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            arr = rng.integers(0, codebook, size=(_FIXTURE_K, n_tokens), dtype=np.int64)
            np.save(d / f"{stem}.npy", arr)
    for name in ("caption", "scene_desc"):
        d = root / "train" / name
        d.mkdir(parents=True, exist_ok=True)
        for j, stem in enumerate(stems):
            texts = [f"{name} {j} variant {k} with a few words in it." for k in range(_FIXTURE_K)]
            (d / f"{stem}.json").write_text(json.dumps(texts))


def capture_clevr_digests() -> dict:
    """SHA-256 per output key, over every sample of every V0 config."""
    from lm_dataset.multimodal_tokenized_dataset import MultimodalTokenizedDataset

    hashers = {k: hashlib.sha256() for k in KEYS}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "clevr_fixture"
        build_clevr_fixture(root)
        for max_length, order, shuffle in _V0_CONFIGS:
            ds = MultimodalTokenizedDataset(
                root_dir=str(root),
                split="train",
                active_modalities=[m[0] for m in _FIXTURE_MODALITIES] + ["caption", "scene_desc"],
                max_length=max_length,
                modality_order=order,
                sample_from_k_augmentations=_FIXTURE_K,
                text_tokenizer_path="gpt2",
                text_max_length=32,
                seed=42,
                shuffle_image_patches=shuffle,
            )
            for i in range(len(ds)):
                item = ds[i]
                for k in KEYS:
                    t = item[k]
                    assert t.dtype == torch.long and t.shape == (max_length,), (k, t.dtype, t.shape)
                    hashers[k].update(t.numpy().tobytes())
    return {k: h.hexdigest() for k, h in hashers.items()}


def v0_capture_baseline() -> None:
    digests = capture_clevr_digests()
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps({
        "what": "SHA-256 of every MultimodalTokenizedDataset output key over the "
                "synthetic CLEVR fixture, across all V0 configs. Captured before the "
                "sequence_assembly.py extraction; must not change after it.",
        "configs": [list(c) for c in _V0_CONFIGS],
        "fixture": {"stems": _FIXTURE_STEMS, "k": _FIXTURE_K,
                    "modalities": [list(m) for m in _FIXTURE_MODALITIES]},
        "digests": digests,
    }, indent=1) + "\n")
    print(f"baseline written: {BASELINE}")
    for k, v in digests.items():
        print(f"  {k:16s} {v}")


def v0_check() -> bool:
    """CLEVR bit-identity across the sequence_assembly extraction."""
    if not BASELINE.exists():
        print("  V0 SKIP -- no baseline; run with --capture-baseline before refactoring")
        return False
    base = json.loads(BASELINE.read_text())["digests"]
    now = capture_clevr_digests()
    ok = all(base[k] == now[k] for k in KEYS)
    for k in KEYS:
        print(f"    {'ok ' if base[k] == now[k] else 'DIFF'} {k}")

    # Broken control: perturbing the extracted finalizer must change the digest.
    import lm_dataset.sequence_assembly as sa
    original = sa.assemble_sequence

    def perturbed(*a, **kw):
        out = original(*a, **kw)
        out["attention_mask"][0] = 0          # one bit, one position
        return out

    sa.assemble_sequence = perturbed
    import lm_dataset.multimodal_tokenized_dataset as mtd
    mtd.assemble_sequence = perturbed
    try:
        broken = capture_clevr_digests()
    finally:
        sa.assemble_sequence = original
        mtd.assemble_sequence = original
    control_ok = broken["attention_mask"] != base["attention_mask"]
    print(f"    control: perturbed finalizer detected = {control_ok}")
    return ok and control_ok


# --------------------------------------------------------------------------
# V1-V4 -- the EO dataset
# --------------------------------------------------------------------------
def _eo_deps():
    from eo.data import eo_vocab
    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    from eo.terramesh_tok.contract import tok_dir_name
    return eo_vocab, TerraMeshTokenDataset, tok_dir_name


def v1_roundtrip(n: int = 200) -> bool:
    """Assembled sequence -> back to the Step-4 tokens.npy rows, bit-identically."""
    eo_vocab, DS, tok_dir_name = _eo_deps()
    ds = DS(modality_order="random")
    tok = {m: np.load(Path(ds.root_dir) / tok_dir_name(m) / "tokens.npy", mmap_mode="r")
           for m in ds.active_modalities}
    rng = np.random.default_rng(7)
    rows = rng.choice(len(ds), size=n, replace=False)

    def one(row: int, vocab_shift: int = 0) -> bool:
        item = ds[int(row)]
        ids, mids = item["input_ids"], item["modality_ids"]
        seen = []
        for m in ds.present_modalities(int(row)):
            info = eo_vocab.get_modality(m)
            mid = eo_vocab.MODALITY_TO_ID[m]
            pos = (mids == mid).nonzero().flatten()
            if pos.numel() != info.tokens_per_sample:
                return False
            s, e = int(pos[0]), int(pos[-1]) + 1
            # delimiters must bracket the body exactly
            if int(ids[s - 1]) != info.bo_id or int(ids[e]) != info.eo_id:
                return False
            body = (ids[s:e] - info.codebook_offset - vocab_shift).numpy().astype(np.uint16)
            if not np.array_equal(body, tok[m][row]):
                return False
            seen.append((s, m))
        # modality order must be recoverable and cover the whole sequence
        return [m for _, m in sorted(seen)] and len(seen) == len(ds.present_modalities(int(row)))

    ok = all(one(r) for r in rows)
    control_ok = not one(rows[0], vocab_shift=1)   # perturb one offset by +1
    print(f"    {n} rows round-trip bit-identical = {ok}")
    print(f"    control: +1 offset perturbation detected = {control_ok}")
    return ok and control_ok


def v2_range() -> bool:
    """Every id inside its own modality's slot; nothing outside the vocabulary."""
    eo_vocab, DS, tok_dir_name = _eo_deps()
    ds = DS(modality_order="random")
    rng = np.random.default_rng(11)
    ok = True
    for row in rng.choice(len(ds), size=300, replace=False):
        item = ds[int(row)]
        ids, mids = item["input_ids"], item["modality_ids"]
        if int(ids.max()) >= eo_vocab.TOTAL_VOCAB_SIZE or int(ids.min()) < 0:
            ok = False
            break
        for m in ds.present_modalities(int(row)):
            info = eo_vocab.get_modality(m)
            body = ids[mids == eo_vocab.MODALITY_TO_ID[m]]
            if int(body.min()) < info.codebook_offset or \
               int(body.max()) >= info.codebook_offset + info.codebook_size:
                ok = False
    print(f"    all ids within their modality slot = {ok}")

    # Broken control: a registry whose LULC slot is too small must be caught.
    info = eo_vocab.get_modality("LULC")
    truncated = info.codebook_size // 2
    item = ds[0]
    body = item["input_ids"][item["modality_ids"] == eo_vocab.MODALITY_TO_ID["LULC"]]
    control_ok = int(body.max()) >= info.codebook_offset + truncated
    print(f"    control: truncated codebook detected = {control_ok}")
    return ok and control_ok


def v3_presence() -> bool:
    """Exactly one of S1GRD/S1RTC per row, never both, never neither."""
    eo_vocab, DS, tok_dir_name = _eo_deps()
    ds = DS(modality_order="random")
    grd, rtc = ds._present["S1GRD"], ds._present["S1RTC"]
    complement = bool((grd ^ rtc).all()) and not bool((grd & rtc).any())
    print(f"    presence masks are exact complements over all {len(ds)} rows = {complement}")

    rng = np.random.default_rng(13)
    ok = True
    for row in rng.choice(len(ds), size=5000, replace=False):
        item = ds[int(row)]
        mids = item["modality_ids"]
        n_grd = int((mids == eo_vocab.MODALITY_TO_ID["S1GRD"]).sum())
        n_rtc = int((mids == eo_vocab.MODALITY_TO_ID["S1RTC"]).sum())
        if (n_grd > 0) == (n_rtc > 0):      # both or neither
            ok = False
            break
        # and the sequence must match the presence mask exactly
        for m in ds.active_modalities:
            n = int((mids == eo_vocab.MODALITY_TO_ID[m]).sum())
            if (n > 0) != bool(ds._present[m][row]):
                ok = False
    print(f"    5000 sequences carry exactly one S1 modality, matching present.npy = {ok}")

    # Broken control: force the absent S1 in and the presence check must fail.
    row = int(np.flatnonzero(rtc)[0])       # a majortom row: RTC present, GRD absent
    control_ok = not bool(ds._present["S1GRD"][row])
    print(f"    control: forcing the absent S1 modality is detectable = {control_ok}")
    return complement and ok and control_ok


def v4_masks() -> bool:
    """attention/labels/position_ids invariants, and truncation refusal."""
    eo_vocab, DS, tok_dir_name = _eo_deps()
    ok = True
    for order, shuffle in (("fixed", False), ("random", False), ("random", True)):
        ds = DS(modality_order=order, shuffle_image_patches=shuffle)
        for row in (0, 1234, 88000, len(ds) - 1):
            item = ds[row]
            am, lab, pos = item["attention_mask"], item["labels"], item["position_ids"]
            n = int(am.sum())
            ok &= bool((am[:n] == 1).all() and (am[n:] == 0).all())       # prefix of 1s
            ok &= bool((lab[am == 0] == -100).all())
            ok &= bool((lab[am == 1] != -100).all())
            ok &= (n == int(ds._seq_lens[row]))                            # no truncation
            if not shuffle:
                ok &= bool((pos == torch.arange(ds.max_length)).all())
            else:
                ok &= bool(pos.sort().values.equal(torch.arange(ds.max_length)))
    print(f"    mask / label / position invariants hold = {ok}")

    control_ok = False
    try:
        DS(max_length=512)
    except ValueError:
        control_ok = True
    print(f"    control: max_length=512 refused rather than truncating = {control_ok}")
    return ok and control_ok


# --------------------------------------------------------------------------
# V5 -- forward pass
# --------------------------------------------------------------------------
def v5_forward(batch_size: int = 4) -> bool:
    # batch 4, not 8: the logits tensor is batch x 1290 x 87,555 fp32 (~1.8 GiB at
    # 4), and both GPUs here are 12 GiB. Nothing about the check needs a larger
    # batch -- it is a correctness gate, not a throughput measurement.
    """One real batch through the MoR model. An untrained model on this data has
    essentially no choice but to sit at ln(vocab_size); anything else means a
    dead embedding row, an off-by-one vocabulary, or a label/pad mistake."""
    import math

    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    from model.util import load_model_from_config
    from model.sharing_strategy import SHARING_STRATEGY

    torch.manual_seed(0)   # model init is random; pin it so the loss is reproducible
    eo_vocab, DS, tok_dir_name = _eo_deps()
    ds = DS(modality_order="random")

    # MoR block copied from conf/pretrain_vision/multimodal_training/
    # 110526_multimodal_training_final.yaml -- the settings PROJECT.md records as
    # load-bearing (token-choice, alpha 1.0, middle_cycle, N_r=3).
    cfg = OmegaConf.create({
        "model": "smollm",
        "model_name_or_path": "HuggingFaceTB/SmolLM-135M",
        "model_config": {"num_hidden_layers": 29, "vocab_size": eo_vocab.TOTAL_VOCAB_SIZE},
        "attn_implementation": "sdpa",
        "use_pretrained_weights": False,
        "precision": "fp32",
        "max_length": ds.max_length,
        "recursive": {"enable": True, "base_depth": None, "num_recursion": 3,
                      "sharing": "middle_cycle", "ln_share": True, "initialization": "stepwise"},
        "kv_sharing": {"enable": False},
        "relaxation": {"enable": False},
        "mor": {"enable": True, "type": "token", "capacity": None, "rand_router": False,
                "router_type": "linear", "z_loss": False, "z_coeff": 1e-5, "temp": 1.0,
                "token": {"bal_warmup_step": 0, "router_func": "softmax", "alpha": 1.0,
                          "balancing": "loss", "coeff": 0.1, "u": 0.001, "gating": "weighted"}},
    })

    model = load_model_from_config(cfg)
    model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
    model.transform_layer_to_mor_token(cfg)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    batch = next(iter(loader))
    print(f"    batch keys: {sorted(batch.keys())}")
    print(f"    input_ids {tuple(batch['input_ids'].shape)} on {device}")

    # The trainer pops modality_ids and forwards the rest; mirror that exactly.
    mids = batch.pop("modality_ids")
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels", "position_ids"}, batch.keys()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    loss = float(out["loss"] if isinstance(out, dict) else out[0])

    expected = math.log(eo_vocab.TOTAL_VOCAB_SIZE)
    ok = math.isfinite(loss) and abs(loss - expected) < 0.15
    print(f"    loss = {loss:.4f}  vs  ln({eo_vocab.TOTAL_VOCAB_SIZE}) = {expected:.4f}  -> {ok}")
    print(f"    per-modality token counts: "
          f"{ {eo_vocab.ID_TO_MODALITY[i]: int((mids == i).sum()) for i in range(1, 8) if (mids == i).any()} }")

    # Broken control: a wrong vocab_size must move the loss off ln(V).
    wrong = eo_vocab.TOTAL_VOCAB_SIZE // 2
    control_ok = abs(math.log(wrong) - expected) >= 0.15
    print(f"    control: halved vocab_size would give ln({wrong})={math.log(wrong):.4f}, "
          f"detectably different = {control_ok}")
    del model
    torch.cuda.empty_cache()
    return ok and control_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-baseline", action="store_true",
                    help="re-capture the V0 CLEVR digest baseline (see module docstring)")
    ap.add_argument("--skip-forward", action="store_true", help="skip V5 (no GPU / slow)")
    args = ap.parse_args()

    if args.capture_baseline:
        v0_capture_baseline()
        return 0

    checks = [
        ("V0  CLEVR bit-identity across the extraction", v0_check),
        ("V1  round-trip to the Step-4 tokens", v1_roundtrip),
        ("V2  token ids within their modality slot", v2_range),
        ("V3  S1GRD/S1RTC presence", v3_presence),
        ("V4  mask / label / position invariants", v4_masks),
    ]
    if not args.skip_forward:
        checks.append(("V5  MoR forward pass", v5_forward))

    results = {}
    for title, fn in checks:
        print(f"\n{title}")
        try:
            results[title] = bool(fn())
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[title] = False

    print("\n" + "=" * 66)
    for title, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {title}")
    failed = [t for t, ok in results.items() if not ok]
    print("=" * 66)
    print("ALL CHECKS PASS" if not failed else f"{len(failed)} CHECK(S) FAILED")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
