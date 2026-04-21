# Setup — MoR for Vision

## Hardware Assumptions

Tested on:
- NVIDIA RTX 5000 Ada (compute capability 8.9) — bf16 works natively
- CUDA driver 13.0, toolkit wheels 12.1 (backwards-compatible)
- Ubuntu 20.04 or similar

**If your GPU is pre-Ampere (V100, T4, RTX 20xx):** change `precision: bf16`
to `precision: fp16` in your config YAML. bf16 isn't supported on those GPUs.

## Install

```bash
# 1. Create Python 3.10 env
conda create -n mor-vision python=3.10 -y
conda activate mor-vision

# 2. Install torch FIRST, explicitly with CUDA 12.1 wheels
#    (this is critical — do not let pip pick a torch version)
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Verify torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.3.1+cu121 True

# 4. Install the rest
pip install -r requirements.txt

# 5. Sanity-check torch survived
python -c "import torch; print(torch.__version__)"
# Must still print: 2.3.1+cu121
# If it changed, something force-upgraded torch. Reinstall with --no-deps.
```

## Model Weights / Config

The MoR codebase loads SmolLM tokenizers at module import time, so you need
local copies of both even if you only use one:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceTB/SmolLM-135M', allow_patterns=['*.json','tokenizer*','*.txt'])
snapshot_download('HuggingFaceTB/SmolLM2-135M', allow_patterns=['*.json','tokenizer*','*.txt'])
"
```

Then symlink the cached models into the repo's `hf_cache/`:

```bash
mkdir -p hf_cache/hub
ln -sf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M hf_cache/hub/
ln -sf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M hf_cache/hub/
```

This is needed because `pretrain.py` sets `HF_HOME=./hf_cache/` before
transformers imports, so the library looks there for models.

## Dataset

Expected layout under `<root_dir>/{split}/{modality}/{stem}.{ext}`: