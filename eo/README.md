# `eo/` — TerraMesh data pipeline for MoR

Everything needed to turn IBM/ESA **TerraMesh** samples into discrete tokens for this MoR
fork. Established and verified in Phase 1 Steps 3–4.

## ⚠ Most of this code does NOT run in the repo's `.venv`

The repo pins `transformers==4.52.4` exactly; `terratorch` (which provides the TerraMind
tokenizers) requires a newer floor. The two cannot coexist, deliberately. `uv sync` will
not install this directory's dependencies and is not meant to.

**The exception is `mor_data/`** (and `scripts/verify_step7.py`), which runs in the
`.venv` *by design* — it is the training-side dataloader. It imports numpy, torch and
pyarrow but never terratorch; `terramesh_tok/contract.py` is pure constants and is
importable from either environment, which is what lets the dataloader use the contract
as its single source of truth without dragging terratorch into the training env.

    # for anything under eo/
    source /data/enric/miniforge3/etc/profile.d/conda.sh && conda activate mor
    export HF_HOME=/data/enric/hf

    # for pretrain.py / model/ / lm_dataset/
    source .venv/bin/activate

## Layout

    mor_data/             training-side dataloader (runs in .venv, NOT the mor env)
      eo_vocab.py           PROVISIONAL unified vocabulary -- Phase 2 owns the real one
      terramesh_token_dataset.py   map-style Dataset over the *_tok arrays
    terramesh_tok/        the preprocessing contract, as code -- SINGLE SOURCE OF TRUTH
      contract.py           crop, standardization stats, codebooks, flatten order,
                            output-directory naming (TOK_DIR_SUFFIX/tok_dir_name)
      io.py                 reading TerraMesh WebDataset/Zarr shards
      preprocess.py         prepare(): raw array -> encoder-ready tensor
      tokenizers.py         build / encode / decode
    scripts/
      tokenize_terramesh.py   batch tokenization CLI (resumable, atomic, provenance)
      make_gate_reference.py  regenerates the gate's single-sample reference per crop
      bringup_224.py          Step 1.4 bring-up + off-centre-crop control (rule 2)
      gate_equivalence.py     reproducibility gate: determinism + padding invariance
      verify_step4.py         hand-off verification incl. end-to-end row alignment
      tokenize_coords.py      optional coords modality
      verify_step7.py         dataloader gate (runs in .venv) -- V0..V5, see below
    experiments/          the Step 3 scripts that established the contract (archival)

Import anything from `terramesh_tok` rather than re-deriving a constant. The scripts put
`eo/` on `sys.path` themselves, so they run from any working directory.

## Usage

    python eo/scripts/make_gate_reference.py                        # once per crop change
    python eo/scripts/bringup_224.py                                # rule 2: verify before scaling
    python eo/scripts/gate_equivalence.py                           # before any full run
    python eo/scripts/tokenize_terramesh.py --modality DEM          # one modality
    python eo/scripts/tokenize_terramesh.py --modality DEM --dry-run
    python eo/scripts/verify_step4.py                               # after

    # dataloader gate -- note the DIFFERENT interpreter (.venv, not the mor env)
    HF_HOME=/data/enric/hf ./.venv/bin/python eo/scripts/verify_step7.py

Output goes to `/data/enric/data/TerraMesh/val/<MOD>_tok<CROP>/` — **never under `/home`**:

    tok_index.parquet     canonical row order for all 89,088 samples (at the val/ root)
    <MOD>_tok224/
      tokens.npy          (89088, 196) uint16 -- SAME row order for every modality
      present.npy         (89088,) bool -- S1RTC/S1GRD cover complementary halves
      shards/*.npy        per source tar; the resumable unit
      metadata.json       tokenizer revision, stats used, crop, contract_git_rev, ...

**The directory name carries the crop.** Two token sets coexist — Phase 1's
`<MOD>_tok256` and the ratified `<MOD>_tok224` — and training against the wrong one
is the highest-consequence silent failure available here. The name is derived from
`contract.TOK_DIR_SUFFIX`, so there is no flag to forget and no way to write 224
arrays into a 256-named directory. Never hardcode `_tok`; call `contract.tok_dir_name(mod)`.

`Coords_tok/` is the exception: coords are tokenized from the scene centre, which is
invariant to the crop, so it keeps an unsuffixed name, is never re-tokenized when the
crop changes, and records `"crop": null, "crop_independent": true` in its metadata. A
loader asserting `crop` must read null as "applies at every crop", not as missing.

Joining modalities is `arr[i]`: all six share one row order, so there is no runtime
lookup. Always consult `present.npy` — absent rows are zero-filled and zero is a valid
token.

## Using the dataloader

    from eo.mor_data.terramesh_token_dataset import TerraMeshTokenDataset
    ds = TerraMeshTokenDataset(modality_order='random')     # 89,088 samples

or through the training config: `dataset: terramesh_multimodal`, rooted at
`TERRAMESH_TOK_ROOT`. It yields exactly `input_ids, attention_mask, labels,
position_ids, modality_ids` — a **closed set**, because `MoRTrainer.compute_loss` pops
`modality_ids` and forwards the rest straight into `model(**inputs)`.

Sequence assembly (pad / mask / truncate / position) is shared with the CLEVR dataset
via `lm_dataset/sequence_assembly.py`, so the two paths cannot drift; `verify_step7.py`
V0 holds CLEVR's output bit-identical across that extraction.

## Three things that will bite you

1. **LULC is one-hot over 10 channels**, not class indices. Indices give in-range,
   meaningless tokens.
2. **Tokens are bit-reproducible only at a fixed batch shape** (batch 32 with tail
   padding is the recorded contract). Batch size selects a different cuBLAS kernel and
   flips ~1 token in 8,192 at FSQ bin boundaries.
3. **Poor reconstruction is not evidence of a bad contract.** The DiVAE decoders clamp
   their output to [-1,1] in standardized space, so extreme scenes cannot be
   reconstructed no matter how correct the preprocessing. The encoder is unaffected.

Full evidence for all three is in the private docs repo, `notes/tokenizer_bringup.md`.

Two more, on the dataloader side:

4. **`eo_vocab.py` is provisional.** Its token ids are a placeholder so Step 7 had
   something to load against. Phase 2 designs the real vocabulary; do not train anything
   you intend to keep against these ids.
5. **Modality *order* depends on `dataloader_num_workers`.** `_get_rng` seeds on
   `(seed, worker_id, idx)` — inherited from the CLEVR dataset — so a run is only
   bit-reproducible at a fixed worker count. Token *content* is unaffected.
