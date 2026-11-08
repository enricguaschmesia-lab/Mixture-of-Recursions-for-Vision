# `eo/`: next-token decoders over Earth-observation tokens

Everything this fork adds to Mixture-of-Recursions for Earth observation: turning IBM/ESA
**TerraMesh** samples into discrete tokens with TerraMind's released tokenizers, training a
decoder (vanilla or recursive) on them, evaluating it, and fine-tuning it on downstream tasks.
TerraMesh is the dataset; TerraMind is the model trained on it.

## Two Python environments, and they conflict

The repo pins `transformers==4.52.4` exactly; `terratorch` (which provides the tokenizers) needs a
newer floor. `uv sync` will not install terratorch and is not meant to. **The split is by
dependency, not by directory:**

| Env | For | Activate |
|---|---|---|
| `mor` (conda, 3.11, terratorch) | anything that touches a tokenizer or decodes to pixels: `terramesh_tok/`, tokenization, `scripts/decode_eo.py` | `source /data/enric/miniforge3/etc/profile.d/conda.sh && conda activate mor`, then `export HF_HOME=/data/enric/hf` |
| `.venv` (uv, 3.10) | the model and its data path: `pretrain.py`, `model/`, `lm_dataset/`, `data/`, `train/`, `eval/`, `generate/`, `finetune/`, the gates | `source .venv/bin/activate` |

`terramesh_tok/contract.py` is pure constants, so it imports from either env. That is what lets
the dataloader use the contract as its single source of truth without pulling terratorch into
training.

## Layout

| Path | What |
|---|---|
| `terramesh_tok/` | The preprocessing contract, as code. **Single source of truth: import it, never re-derive a constant.** `contract.py` (crop, stats, codebooks, directory naming), `io.py`, `preprocess.py`, `tokenizers.py`, `clean.py` (the S2 BOA offset), `train_split.py` |
| `data/` | The EO vocabulary (`eo_vocab.py`, final), `TerraMeshTokenDataset`, the held-out split (`eval_split.py`, `eval_rows_v1.json`), the dev/test split of the four-date val locations (`temporal_split.py`, `temporal_rows_v1.json`), their location and date index (`temporal_index.py`), the date rows (`temporal_vocab.py`), and BEN's vocabulary and dataset |
| `train/` | Launch machinery: `gpu.py` (resolve a GPU by name), `preflight.py` and `preflight_controls.py`, `distributed.py`, and the pretrained-body inits `clevr_init.py`, `smollm_init.py` |
| `eval/` | `teacher_forced.py` (the per-checkpoint pass, target-last layout, depth intervention), `curves.py`, `ben.py` (constrained label decoding), `ben_chain.py` |
| `generate/` | Conditional generation with a KV cache. Decoding ids to pixels is a separate process in `mor` |
| `finetune/` | Vocabulary extension (identity-checked), hand-rolled LoRA, init from a checkpoint |
| `routing/` | Recursion-depth telemetry for the MoR arm, including the router-only pass (`capture_router` in `telemetry.py`), and routing statistics (`stats.py`: exact permutation, group bootstrap, the equal-compute lattice) |
| `ben/` | BigEarthNet v1: labels, layout, patch index and splits; `pack.py` and `preprocess.py` (`mor`) |
| `scripts/` | Thin CLI entry points (below). The rule: real code lives in a package, and scripts stay thin |
| `experiments/` | One-off analyses grouped by topic, kept as the record behind each result: see [`experiments/README.md`](experiments/README.md) |

Comments that cite `PHASE*_PLAN.md`, `worklog` or `docs/notes/…` refer to the project's private
planning repository.

## Scripts by purpose

| Purpose | Scripts |
|---|---|
| Train | **`train_eo.sh`** (the only supported entry point), `train_chain.sh` (runs back to back), `convert_clevr_init.py`, `convert_smollm_init.py`, `extend_checkpoint.py` |
| Tokenize | `tokenize_terramesh.py` (val), `tokenize_train.py` and `run_tokenize_train.sh` (train, streamed), `tokenize_coords.py`, `label_s2_offset.py`, `tokenize_clean.py` (`val_clean`), `build_ben_index.py`, `pack_ben.py`, `tokenize_ben.py` |
| Evaluate | `eval_checkpoint.py` (per-checkpoint pass; `--grid` for the source × target grid, `--force-depth` and `--force-scope` for depth sweeps), `arm_report.py`, `grid_report.py`, `routing_telemetry.py`, `capture_router.py` (the router alone, on held-out, training or probe rows), `patch_stats.py`, `wrong_scene_control.py` |
| Downstream | `eval_ben.py`, `eval_forgetting.py`, `ben_report.py`, `ben_chain.py`, `ben_teacher.py` |
| Generate and decode | `generate_eo.py`, `prepare_decode.py`, `decode_eo.py` (`mor`), `sweep_d311.sh` |
| Held-out split | `build_eval_split.py`, and `build_temporal_split.py` for the four-date locations |
| Temporal index | `build_temporal_index.py`: `build` where the metadata is, `check` where the tokens are |
| Gates | `verify_phase3.py` (the current gate; runs `verify_phase2.py` as its first check), `verify_phase6.py` (downstream), `verify_temporal.py` (the temporal stage), `verify_step4.py`, `verify_step5.py`, `verify_step7.py` (artifacts and dataloader), `gate_equivalence.py`, `make_gate_reference.py`, `bringup_224.py` (tokenization) |

Every gate check is paired with a deliberately broken control that must fire.

    ./.venv/bin/python eo/scripts/verify_phase3.py                  # the full gate
    ./.venv/bin/python eo/scripts/verify_phase3.py --skip-forward   # everything that needs no GPU
    ./.venv/bin/python eo/scripts/verify_phase6.py

## Starting a training run

**`eo/scripts/train_eo.sh` is the only supported entry point.** Not `scripts/pretrain.sh`, which
launches through accelerate/deepspeed.

    bash eo/scripts/train_eo.sh --gpu titanx --arm b10c --config eo_terramesh/arm_b10c_vanilla --detach
    bash eo/scripts/train_eo.sh --arm smoke stop_steps=30 total_batch_size=4 wandb=false

It loads `.env`, pins `CUDA_DEVICE_ORDER=PCI_BUS_ID`, resolves the GPU **by name**
(`titanv`, `titanx`, `a100`), asserts that CUDA returned that card, runs `eo.train.preflight`, and
refuses to start if any check fails. `--detach` puts it in tmux with a log under
`/data/enric/logs/`. `--resume` validates what it resumes. `--gpus N` is opt-in DDP on one node of
identical cards.

⚠ **The two GPU index orderings are inverted on this workstation:** `nvidia-smi` uses PCI bus
order (0 = GTX Titan X) while CUDA defaults to fastest-first (0 = TITAN V). Picking the wrong card
is a ~2.4× slowdown with no error, which is why nothing here takes a raw index.

Arms are thin Hydra overlays under `conf/pretrain_vision/eo_terramesh/`:

| Config | Model |
|---|---|
| `arm_b10c_vanilla` | B10c: 29-layer vanilla, 10 epochs on the val split |
| `arm_t360_vanilla` | T360: the SmolLM-360M shape on the train split |
| `arm_a10c_mor`, `arm_a10cs2_mor` | MoR, learned router, seeds 42 and 43 |
| `arm_a10cr_random`, `arm_a10cr5_random` | MoR, random router, 3 and 5 recursions |
| `arm_b10c11_vanilla` | 11-layer vanilla, MoR's unique layers |
| `arm_b10c{v,s,l,ls}_*`, `arm_b10cl_lora` | Pretrained-body inits (CLEVR, SmolLM, shuffled controls, LoRA) |
| `arm_{a,b}_*`, `arm_{a10,b10}_*` | The earlier 50-epoch and 10-epoch arms, on the original token set |

## Fine-tuning a downstream task

A fine-tune starts through the same entry point, from a checkpoint extended by the task's
vocabulary rows:

    bash eo/scripts/train_eo.sh --arm b10full --config eo_ben/b10_full --gpu titanx \
        --init-from /data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300 --detach

`--init-from` is not a resume: a strict load, a fresh optimizer and scheduler, and preflight
verifies the extension's identity artifact first. Regimes (`finetune.regime`, one overlay each under
`conf/pretrain_vision/eo_ben/`):

| Regime | What trains |
|---|---|
| `full` | Every weight |
| `lora` | Rank-16 adapters plus the new rows |
| `head` | Only the new rows |
| `scratch` | Random init, nothing loaded |

## Tokenizing

    python eo/scripts/make_gate_reference.py                        # once per crop change
    python eo/scripts/bringup_224.py                                # verify before scaling
    python eo/scripts/gate_equivalence.py                           # before any full run
    python eo/scripts/tokenize_terramesh.py --modality DEM          # one modality
    python eo/scripts/verify_step4.py                               # after

Output goes to `/data/enric/data/TerraMesh/val/<MOD>_tok<CROP>/`, **never under `/home`**:

    tok_index.parquet     canonical row order for all 89,088 samples (at the val/ root)
    <MOD>_tok224/
      tokens.npy          (89088, 196) uint16, the SAME row order for every modality
      present.npy         (89088,) bool; S1RTC and S1GRD cover complementary rows
      shards/*.npy        per source tar; the resumable unit
      metadata.json       tokenizer revision, stats used, crop, contract_git_rev, ...

**The directory name carries the crop.** Two token sets coexist, `<MOD>_tok256` and the ratified
`<MOD>_tok224`, and training against the wrong one is the worst silent failure available here. The
name derives from `contract.TOK_DIR_SUFFIX`: never hardcode `_tok`, call `contract.tok_dir_name(mod)`.
`Coords_tok/` is the exception: coords come from the scene centre, invariant to the crop, so its
metadata records `"crop": null, "crop_independent": true`. Always consult `present.npy`: absent rows
are zero-filled, and zero is a valid token.

## The dataloader

    from eo.data.terramesh_token_dataset import TerraMeshTokenDataset
    ds = TerraMeshTokenDataset(modality_order='random')     # 89,088 samples, 995 tokens

Or through the training config: `dataset: terramesh_multimodal`, rooted at `TERRAMESH_TOK_ROOT`.
`multimodal.token_set: clean` selects the cleaned S2L2A/NDVI tokens. The presence masks pick exactly
one of S1GRD/S1RTC per row, so every sequence is 5 × (196 + BO + EO) + (3 + BO + EO) = **995
tokens**: padding is zero and batch shapes are stable.

The dataset yields exactly `input_ids, attention_mask, labels, position_ids, modality_ids`. This is
a **closed set**: `MoRTrainer.compute_loss` pops `modality_ids` and forwards the rest straight into
`model(**inputs)`, so a sixth key becomes an unexpected kwarg.

Sequence assembly is shared with the CLEVR dataset through `lm_dataset/sequence_assembly.py`, so the
two paths cannot drift; `verify_step7.py` V0 holds CLEVR's output bit-identical.

## Things that will bite you

1. **LULC is one-hot over 10 channels,** not class indices. Indices give in-range, meaningless tokens.
2. **Tokens are bit-reproducible only at a fixed batch shape** (batch 32 with tail padding is the
   recorded contract). Batch size selects a different cuBLAS kernel and flips ~1 token in 8,192 at
   FSQ bin boundaries.
3. **Poor reconstruction is not evidence of a bad contract.** The DiVAE decoders clamp output to
   [-1, 1] in standardized space, so extreme scenes cannot be reconstructed however correct the
   preprocessing. The encoder is unaffected.
4. **`eo_vocab.py` is a one-way door.** `TOTAL_VOCAB_SIZE = 87,556`, `PAD_ID = 87,555`. Anything
   trained against these ids is invalidated if the layout changes, and resizing the last codebook
   slot shifts every BO/EO and PAD id. New rows are appended after it, as `ben_vocab.py` does. Its
   import-time asserts check the derivation, not the codebook sizes: those are guarded by
   `contract.CODEBOOK` and `assert_artifact_fits()`.
5. **Modality order depends on `dataloader_num_workers`:** `_get_rng` seeds on
   `(seed, worker_id, idx)`, so a run is bit-reproducible only at a fixed worker count. Token content
   is unaffected.
6. **Importing terratorch sets `torch.set_float32_matmul_precision("medium")`.** On an A100 that
   flips 0.4–4.9% of tokens: re-apply `"highest"` after importing it on any Ampere+ card
   (`train_split.strict_fp32`).
