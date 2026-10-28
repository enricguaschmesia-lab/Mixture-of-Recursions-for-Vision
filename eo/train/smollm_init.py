# eo/train/smollm_init.py
"""The LLM source for the pretrained-body init: SmolLM-135M (`.venv`, CPU).

B10c's architecture IS SmolLM-135M's: `terramesh_mor_token.yaml` builds it from
`AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")` with 29 layers instead of 30 and
the EO vocabulary. So SmolLM-135M, a LLaMA-architecture next-token predictor trained on
English text and code (SmolLM-Corpus, 600B tokens), is the LLM whose body fits B10c with
nothing reshaped. Every body tensor of 29 of its 30 layers has B10c's name and shape.

What carries over (2026-10-28, docs/notes/smollm_init_2026-10-28.md):
  * 29 of the 30 layers, and the final norm, bit for bit (stored fp32, values bf16-exact).
    One layer must go. It is the one whose removal costs SmolLM the least next-token loss
    on English text (the ShortGPT / Gromov et al. 2024 criterion, measured 2026-10-28):
    DROPPED_LAYER = 5, +0.06 nats. Keeping the first 29 instead (dropping the last) would
    cost +1.45, ~24x more. Layers 6..29 move down by one index.
  * NOT the embedding (tied to lm_head): 49,152 BPE text rows mean nothing to TerraMind
    codes. The EO table is B10c's own seeded build, as for the CLEVR arms.

The SHUFFLED control permutes every body tensor (clevr_init.body_state), as for CLEVR.

The checks reuse clevr_init's tensor checks. The function check's reference is
`transformers.LlamaForCausalLM`, the released class, loaded from the pinned files with
the dropped layer cut out of its module list. Our model is built by the repo's own
`model/base_model/modeling_llama.py`, so the check also proves the two classes agree.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import torch

from eo.train import clevr_init as C

SMOLLM_REPO = "HuggingFaceTB/SmolLM-135M"
SMOLLM_REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
# sha256 of model.safetensors at SMOLLM_REVISION; equals Hugging Face's LFS sha256 (2026-10-28).
SMOLLM_WEIGHTS_SHA256 = "c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093"
SOURCE_LAYERS = 30
#: The cheapest single layer to remove, by SmolLM's next-token loss on English text
#: (eo/experiments/smollm_layer_drop.py, 2026-10-28). On 16,384 tokens of prose, from 3.348 nats
#: with all 30 layers: layer 5 +0.060 (the least), layers 4/6/8/10 +0.08-0.09, the last layer
#: (29) +1.45, layer 11 +1.58, layer 0 +7.2. On 4,096 tokens of LICENSE + README: layer 5
#: +0.165, second to layer 15's +0.156; the last layer +1.33.
DROPPED_LAYER = 5
#: SmolLM's config must equal B10c's in every architecture key but these two.
ARCH_KEYS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
             "hidden_act", "rms_norm_eps", "rope_theta", "rope_scaling", "attention_bias",
             "mlp_bias", "tie_word_embeddings", "head_dim")

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def load_smollm(src: Path) -> Dict:
    """The pinned SmolLM-135M: state (fp32 storage), config. Refuses anything else."""
    from safetensors.torch import load_file
    wf = src / "model.safetensors"
    got = C.sha256(wf)
    if got != SMOLLM_WEIGHTS_SHA256:
        raise RuntimeError(f"{wf} hashes to {got}, not the pinned {SMOLLM_REPO}@{SMOLLM_REVISION[:8]}")
    config = json.loads((src / "config.json").read_text())
    if int(config["num_hidden_layers"]) != SOURCE_LAYERS:
        raise RuntimeError(f"{src}: {config['num_hidden_layers']} layers, expected {SOURCE_LAYERS}")
    state = load_file(str(wf), device="cpu")
    if C.HEAD_KEY in state:
        raise RuntimeError("SmolLM-135M ships no lm_head.weight (tied); this file has one")
    return {"state": state, "config": config, "weights_sha256": got}


def check_architecture(src_config: Dict, model_config) -> Dict:
    """SmolLM's architecture keys equal the built B10c model's; only depth and vocab differ."""
    diff = {}
    for k in ARCH_KEYS:
        a = src_config.get(k)
        b = getattr(model_config, k, None)
        if k == "head_dim" and a is None:           # SmolLM's config predates the key
            a = src_config["hidden_size"] // src_config["num_attention_heads"]
        if a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b)):
            diff[k] = (a, b)
    return {"keys": list(ARCH_KEYS), "differ": {k: list(v) for k, v in diff.items()},
            "layers": [src_config["num_hidden_layers"], model_config.num_hidden_layers],
            "vocab": [src_config["vocab_size"], model_config.vocab_size], "passed": not diff}


def layer_map(dropped: int, n_source: int = SOURCE_LAYERS) -> List[int]:
    """Target layer i <- source layer layer_map[i]: every layer but `dropped`, in order."""
    if not 0 <= dropped < n_source:
        raise ValueError(f"dropped layer {dropped} not in 0..{n_source - 1}")
    return [i for i in range(n_source) if i != dropped]


def mapped_state(state: Dict[str, torch.Tensor], dropped: int) -> Dict[str, torch.Tensor]:
    """SmolLM's body under B10c's names: layers renumbered past the dropped one, the final norm
    as is, the embedding left out. The tensors are the source's own (no copy, no cast)."""
    src_of = {t: s for t, s in enumerate(layer_map(dropped))}
    by_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    out = {}
    for k, t in state.items():
        m = _LAYER.match(k)
        if m:
            by_layer.setdefault(int(m[1]), {})[m[2]] = t
        elif k not in C.EMBED_KEYS:
            out[k] = t
    if sorted(by_layer) != list(range(SOURCE_LAYERS)):
        raise RuntimeError(f"source layers {sorted(by_layer)[:3]}..: expected 0..{SOURCE_LAYERS - 1}")
    for t, s in src_of.items():
        for rest, tensor in by_layer[s].items():
            out[f"model.layers.{t}.{rest}"] = tensor
    return out


def reference(src: Path, dropped: int):
    """SmolLM-135M in transformers' own LlamaForCausalLM, fp32 on CPU, with layer `dropped` cut
    out of its module list: the function check's reference. Independent of mapped_state."""
    import transformers
    m = transformers.LlamaForCausalLM.from_pretrained(
        str(src), torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True)
    if dropped is not None:
        keep = layer_map(dropped)
        m.model.layers = torch.nn.ModuleList([m.model.layers[i] for i in keep])
        m.config.num_hidden_layers = len(keep)
    return m.eval()


@torch.no_grad()
def text_loss(model, ids: torch.Tensor) -> float:
    """Next-token loss of a full LM (its own text embedding and head) on `ids`, (B, T)."""
    return float(model(input_ids=ids, labels=ids, use_cache=False).loss)


def text_ids(src: Path, paths, seq_len: int = 1024, n_seq: int = 4) -> torch.Tensor:
    """`n_seq` x `seq_len` SmolLM tokens of the given text files, concatenated in order."""
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(str(src), local_files_only=True)
    ids = torch.cat([torch.tensor(tok(Path(p).read_text())["input_ids"]) for p in paths])
    n = min(n_seq, ids.numel() // seq_len)
    if n < 1:
        raise RuntimeError(f"{[str(p) for p in paths]}: fewer than {seq_len} tokens")
    return ids[: n * seq_len].view(n, seq_len)
