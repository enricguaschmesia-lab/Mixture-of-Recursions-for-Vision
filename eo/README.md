# `eo/` — TerraMesh data pipeline for MoR

Everything needed to turn IBM/ESA **TerraMesh** samples into discrete tokens for this MoR
fork. Established and verified in Phase 1 Steps 3–4.

## ⚠ This code does NOT run in the repo's `.venv`

The repo pins `transformers==4.52.4` exactly; `terratorch` (which provides the TerraMind
tokenizers) requires a newer floor. The two cannot coexist, deliberately. `uv sync` will
not install this directory's dependencies and is not meant to.

    # for anything under eo/
    source /data/enric/miniforge3/etc/profile.d/conda.sh && conda activate mor
    export HF_HOME=/data/enric/hf

    # for pretrain.py / model/ / lm_dataset/
    source .venv/bin/activate

## Layout

    terramesh_tok/        the preprocessing contract, as code -- SINGLE SOURCE OF TRUTH
      contract.py           crop, standardization stats, codebooks, flatten order
      io.py                 reading TerraMesh WebDataset/Zarr shards
      preprocess.py         prepare(): raw array -> encoder-ready tensor
      tokenizers.py         build / encode / decode
    scripts/
      tokenize_terramesh.py   batch tokenization CLI (resumable, atomic, provenance)
      gate_equivalence.py     reproducibility gate: determinism + padding invariance
      verify_step4.py         hand-off verification incl. end-to-end row alignment
      tokenize_coords.py      optional coords modality
    experiments/          the Step 3 scripts that established the contract (archival)

Import anything from `terramesh_tok` rather than re-deriving a constant. The scripts put
`eo/` on `sys.path` themselves, so they run from any working directory.

## Usage

    python eo/scripts/tokenize_terramesh.py --modality DEM          # one modality
    python eo/scripts/tokenize_terramesh.py --modality DEM --dry-run
    python eo/scripts/gate_equivalence.py                           # before any full run
    python eo/scripts/verify_step4.py                               # after

Output goes to `/data/enric/data/TerraMesh/val/<MOD>_tok/` — **never under `/home`**:

    tok_index.parquet     canonical row order for all 89,088 samples (at the val/ root)
    <MOD>_tok/
      tokens.npy          (89088, 256) uint16 -- SAME row order for every modality
      present.npy         (89088,) bool -- S1RTC/S1GRD cover complementary halves
      shards/*.npy        per source tar; the resumable unit
      metadata.json       tokenizer revision, stats used, crop, contract_git_rev, ...

Joining modalities is `arr[i]`: all six share one row order, so there is no runtime
lookup. Always consult `present.npy` — absent rows are zero-filled and zero is a valid
token.

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
