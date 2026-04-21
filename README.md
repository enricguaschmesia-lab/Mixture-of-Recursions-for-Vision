# Mixture-of-Recursions for Vision

Adapting [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) (MoR, NeurIPS 2025) — a decoder-only transformer with per-token adaptive recursion depth — to vision and multimodal image generation on [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/).

Course project for EPFL CS-503 (Visual Intelligence). Built on top of the [original MoR codebase](https://github.com/raymin0223/mixture_of_recursions).

## What's in here

- **`model/mor_model/`** — MoR Llama backbone with expert-choice and token-choice routers (from upstream).
- **`model/sharing_strategy/llama.py`** — Middle-Cycle parameter sharing across recursions.
- **`model/kv_caches/`** — Recursion-wise KV caching.
- **`lm_dataset/multimodal_tokenized_dataset.py`** — CLEVR loader for pre-tokenized RGB / depth / normals / text, with a unified multimodal vocab.
- **`conf/pretrain_vision/`** — Hydra configs for the vision runs (unimodal RGB PoC, multimodal, vanilla baseline).
- **`pretrain.py`** — single training entry point.

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and a CUDA-capable GPU (tested on RTX 5000 Ada, CUDA 12.1).

```bash
uv sync
```

That installs torch with CUDA 12.1 wheels, transformers 4.52.4 (pinned — MoR uses private APIs), and everything else. No conda, no manual torch step.

Pre-fetch the SmolLM tokenizer into the local HF cache the trainer expects (`./hf_cache`):

```bash
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM-135M',
                  allow_patterns=['*.json', 'tokenizer*', '*.txt'],
                  cache_dir='./hf_cache')
"
```

## Data

Pre-tokenized CLEVR (RGB / depth / normals via Cosmos DI-16×16, text via GPT-2) laid out as `<root>/<split>/<modality>/<stem>.{npy,json}`.

Download from: https://drive.google.com/file/d/1QRFqoGKMFYlgxYfPr9O7PeOadzXAbJI8/view

```bash
tar -xzvf clevr_cs503.tar.gz
mv clevr_cs503 clevr_dataset   # rename if the archive extracts as clevr_cs503
mv clevr_dataset data/
```

The default path is `<repo>/data/clevr_dataset`. Override with:

```bash
export CLEVR_ROOT=/path/to/clevr_dataset
```

## Run

Copy `.env.example` to `.env` and set `WANDB_ENTITY` (and optionally `WANDB_PROJECT`, `CLEVR_ROOT`). `pretrain.py` auto-loads `.env` at startup, and the configs read W&B settings via `${oc.env:...}` — no need to edit YAML per machine.

```bash
cp .env.example .env
$EDITOR .env   # set WANDB_ENTITY
```

One-off overrides still work the ordinary way:

```bash
WANDB_MODE=offline uv run bash scripts/pretrain.sh accelerate offline 0 smoke_50steps
```

Smoke test (1 GPU, 50 steps, RGB only):

```bash
uv run bash scripts/pretrain.sh accelerate online 0 smoke_50steps
```

Full Milestone I unimodal RGB run (135M, R=3, middle_cycle, expert-choice router):

```bash
uv run bash scripts/pretrain.sh accelerate online 0,1 \
  250720_pretrain_smollm-135m_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001
```

Vanilla fixed-depth baseline (for RQ3 compute comparison):

```bash
uv run bash scripts/pretrain.sh accelerate online 0 \
  250720_pretrain_smollm-135m_vanilla_lr3e-3
```

## Citation

```
@misc{bae2025mixtureofrecursionslearningdynamicrecursive,
  title  = {Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation},
  author = {Sangmin Bae and Yujin Kim and Reza Bayat and Sungnyun Kim and Jiyoun Ha and Tal Schuster and Adam Fisch and Hrayr Harutyunyan and Ziwei Ji and Aaron Courville and Se-Young Yun},
  year   = {2025},
  eprint = {2507.10524},
  archivePrefix = {arXiv},
}
```
