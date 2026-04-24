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
### CLEVR
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

### COCO

Follow these steps to download, extract, and tokenize the COCO dataset for the MoR model.

Create the raw data directory and download the COCO 2017 Train and Val splits. 
*(Estimated time: ~18 mins for train, ~2 mins for val)*

```bash
# Create the directory and navigate into it
mkdir -p data/coco_dataset/raw
cd data/coco_dataset/raw

wget http://images.cocodataset.org/zips/train2017.zip
unzip -n train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip -n val2017.zip

# Return to the project root
cd ../../../
```

Then be sure to have the Cosmos tokenizer.
```bash
huggingface-cli download nvidia/Cosmos-0.1-Tokenizer-DI16x16 \
  --local-dir pretrained_ckpts/Cosmos-0.1-Tokenizer-DI16x16
```

Run the preparation script to crop the images to 256x256 and generate the .npy token files.
(Estimated time: ~6 mins)
```bash
cd lm_dataset
uv run python prepare_coco.py
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

If you want to plot during training intermediate plotting download the tokenizer. 

```bash
huggingface-cli download nvidia/Cosmos-0.1-Tokenizer-DI16x16 \
  --local-dir pretrained_ckpts/Cosmos-0.1-Tokenizer-DI16x16
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

## Inference

`infer.py` is the single entry point for sampling from a trained MoR checkpoint. It builds a prompt in the unified multimodal vocab, runs `model.generate`, decodes tokens back to an image via the Cosmos tokenizer, and (for expert-choice MoR) saves a per-token recursion-depth overlay.

The default config `conf/infer/mor_19500_rgb.yaml` points at our pretrained RGB checkpoint on the Hub (`gbasi18/MoR-Vision-19500-RGB`), so you can run it with no local training:

```bash
uv run python infer.py
```

Common overrides (Hydra CLI):

```bash
# Use a local checkpoint instead of the Hub one
uv run python infer.py infer.checkpoint=results/pretrain/<run_name>

# Image prefix completion — feed the first 128 tokens of a test image and let the model finish it
uv run python infer.py \
  infer.prompt_npy=data/clevr_dataset/test/tok_rgb@256/02645.npy \
  infer.prefix_len=128

# Text-conditional generation (caption -> RGB)
uv run python infer.py infer.prompt_text="a red metal cube next to a blue rubber sphere"

# Diverse sampling instead of greedy
uv run python infer.py infer.temperature=0.9 infer.top_p=0.95 infer.seed=42

# Swap the config file entirely
uv run python infer.py --config-name=infer/my_run
```

Outputs (decoded PNG and, for expert-choice MoR, the `*_depth_overlay.png` heatmap) are written to `results/infer/`. To skip the extra forward pass used for the overlay, set `infer.save_depth_overlay=false`.

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

