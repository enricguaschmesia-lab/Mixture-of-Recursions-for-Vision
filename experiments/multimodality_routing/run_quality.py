"""Quality / difficulty experiments.

E7: per-modality teacher-forced cross-entropy (does prediction loss match routing depth?)
E8: forced-depth ablation (monkey-patch router) — measure per-modality CE at each fixed depth
E9: cross-modal generation quality — caption->rgb, rgb->depth token accuracy / PSNR
"""

import os, json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
try:
    from torch.distributed.tensor import DTensor  # noqa
except ImportError:
    class DTensor: pass
import transformers.modeling_utils
transformers.modeling_utils.DTensor = DTensor  # type: ignore

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from omegaconf import open_dict
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from transformers import AutoTokenizer

from util.env import load_dotenv
load_dotenv()
from paths import HF_CACHE_DIR
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

from model.util import load_model_from_config, load_checkpoint
from model.sharing_strategy import SHARING_STRATEGY
from util.config import preprocess_config
from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES, PAD_ID

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
OUT_DIR = PROJECT_DIR / "experiments/multimodality_routing/out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_TEST = PROJECT_DIR / "data/clevr_dataset/test"

N_SAMPLES = 12
AUG_IDX = 0

# ---------------- load model
if GlobalHydra().is_initialized():
    GlobalHydra().clear()
with initialize_config_dir(config_dir=str(PROJECT_DIR / "conf"), version_base=None):
    cfg = compose(config_name="infer/vanilla_10000_multimodality")
with open_dict(cfg):
    cfg.wandb = False
    cfg.wandb_entity = ""
    cfg.wandb_project = "exp"
    cfg.wandb_run_name = "exp"
    cfg.resume_from_checkpoint = False
cfg = preprocess_config(cfg)
DEVICE = cfg.infer.device if torch.cuda.is_available() else "cpu"

print(f"Building model on {DEVICE}…")
model = load_model_from_config(cfg)
if cfg.recursive.get("enable"):
    model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
if cfg.mor.get("enable") and cfg.mor.type == "token":
    model.transform_layer_to_mor_token(cfg)
model = load_checkpoint(model, cfg.infer.checkpoint)
model.to(DEVICE).eval()
NUM_RECURSION = cfg.recursive.num_recursion
TEXT_MAX_LEN = cfg.multimodal.get("text_max_length", 64)
GPT2_TOK = AutoTokenizer.from_pretrained("gpt2")
print(f"Ready. num_recursion={NUM_RECURSION}")


# ---------------- helpers (subset of run_exp.py)
_BO_TO_MOD = {info.bo_id: name for name, info in MODALITIES.items()}
_EO_IDS = {info.eo_id for info in MODALITIES.values()}


def chunk_text(modality, text, close=True):
    info = MODALITIES[modality]
    ids = GPT2_TOK(text, truncation=True, max_length=TEXT_MAX_LEN, return_tensors="pt")["input_ids"][0].long()
    parts = [torch.tensor([info.bo_id]), ids + info.codebook_offset]
    if close: parts.append(torch.tensor([info.eo_id]))
    return torch.cat(parts)


def chunk_tokens(modality, source, aug_idx=0, close=True):
    info = MODALITIES[modality]
    arr = np.load(source) if isinstance(source, (str, Path)) else np.asarray(source)
    if arr.ndim == 2: arr = arr[aug_idx]
    body = torch.from_numpy(arr.flatten()).long() + info.codebook_offset
    parts = [torch.tensor([info.bo_id]), body]
    if close: parts.append(torch.tensor([info.eo_id]))
    return torch.cat(parts)


def segment_by_modality(input_ids):
    if input_ids.dim() > 1: input_ids = input_ids[0]
    segs, s, m = [], None, None
    for i, t in enumerate(input_ids.tolist()):
        if t in _BO_TO_MOD: s, m = i + 1, _BO_TO_MOD[t]
        elif t in _EO_IDS and m is not None:
            segs.append((s, i, m)); s, m = None, None
    return segs


def load_sample(sample_id):
    return dict(
        id=sample_id,
        rgb=np.load(DATA_TEST / "tok_rgb@256" / f"{sample_id}.npy")[AUG_IDX],
        depth=np.load(DATA_TEST / "tok_depth@256" / f"{sample_id}.npy")[AUG_IDX],
        normal=np.load(DATA_TEST / "tok_normal@256" / f"{sample_id}.npy")[AUG_IDX],
        caption=json.load(open(DATA_TEST / "caption" / f"{sample_id}.json"))[AUG_IDX],
    )


samples = [load_sample(f"{i:05d}") for i in range(N_SAMPLES)]


# ---------------- find the MoR module so we can monkey-patch it
mor_modules = [m for m in model.modules() if getattr(m, "mor", False)]
assert len(mor_modules) == 1, f"expected 1 MoR module, got {len(mor_modules)}"
mor_mod = mor_modules[0]
print(f"Found MoR module: {type(mor_mod).__name__}")


# We override the router decision *and* the gating weights by patching the forward pre-step:
# the cleanest hook point is to replace `self.mor_router` output by adding a forward hook
# that always returns a constant logits vector that argmaxes to a chosen expert.
# Simpler: monkey-patch torch.topk inside the router context — but topk is global.
# Cleanest: write a wrapper around mor_mod.forward that pre-sets a class attribute used by
# our patched router_probs (forced_expert). We instead monkey-patch the router's forward
# to return logits that yield the desired argmax.

class _ConstRouter(torch.nn.Module):
    """Replace `mor_router` so that argmax always equals `expert`."""
    def __init__(self, base_router, expert: int, num_recursion: int):
        super().__init__()
        self.base = base_router
        self.expert = expert
        self.num_recursion = num_recursion

    def forward(self, x):
        bs, sl, _ = x.shape
        # very peaked logits at chosen expert -> softmax≈one-hot at that expert
        logits = torch.full((bs, sl, self.num_recursion), -10.0, device=x.device, dtype=x.dtype)
        logits[..., self.expert] = 10.0
        return logits


def force_expert(expert: int | None):
    """Replace mor_router with a constant one (expert in [0..Nr-1]) or restore original."""
    if not hasattr(mor_mod, "_orig_router"):
        mor_mod._orig_router = mor_mod.mor_router
    if expert is None:
        mor_mod.mor_router = mor_mod._orig_router
    else:
        mor_mod.mor_router = _ConstRouter(mor_mod._orig_router, expert, NUM_RECURSION).to(DEVICE)


# ---------------- E7 / E8: per-modality teacher-forced CE under each routing mode
print("\n[E7/E8] Per-modality CE under {router, depth=1, depth=2, depth=3}…")
modes = {"router": None, "d1": 0, "d2": 1, "d3": 2}
mods = ["caption", "tok_rgb@256", "tok_depth@256", "tok_normal@256"]
# results[mode][modality] = list of mean CE values
results = {mode: {m: [] for m in mods} for mode in modes}

for s in samples:
    chunks = [
        chunk_text("caption", s["caption"]),
        chunk_tokens("tok_rgb@256", s["rgb"]),
        chunk_tokens("tok_depth@256", s["depth"]),
        chunk_tokens("tok_normal@256", s["normal"]),
    ]
    seq = torch.cat(chunks).unsqueeze(0).to(DEVICE)
    labels = seq.clone()
    segs = segment_by_modality(seq)

    for mode_name, expert in modes.items():
        force_expert(expert)
        with torch.no_grad():
            out = model(input_ids=seq, labels=None, use_cache=False)
        logits = out.logits  # (1, T, V)
        # shift for next-token prediction: predict token t from logits[:, t-1, :]
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(shift_labels.shape).squeeze(0).cpu().numpy()  # (T-1,)

        # For each modality body, get CE on the positions where the label is a body token of that modality
        for st, en, m in segs:
            # positions in labels are [1..T-1], shift index = pos - 1
            # body labels are at positions st..en-1 (inclusive)
            idxs = np.arange(max(st - 1, 0), en - 1)
            if idxs.size > 0:
                results[mode_name][m].append(float(ce[idxs].mean()))
    force_expert(None)  # restore


# ---------------- plot E7/E8
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
mode_names = list(modes.keys())
colors = {"router": "#1f77b4", "d1": "#440154", "d2": "#21918c", "d3": "#fde725"}

# bar: per modality, group by mode
x = np.arange(len(mods))
w = 0.2
for i, mn in enumerate(mode_names):
    means = [np.mean(results[mn][m]) for m in mods]
    stds = [np.std(results[mn][m]) for m in mods]
    axes[0].bar(x + (i - 1.5) * w, means, width=w, yerr=stds, capsize=3,
                label=mn, color=colors[mn])
axes[0].set_xticks(x)
axes[0].set_xticklabels(mods, rotation=15)
axes[0].set_ylabel("teacher-forced CE (nats)")
axes[0].set_title("Per-modality CE under router vs forced fixed-depth")
axes[0].legend(fontsize=9)
axes[0].set_yscale("log")

# Delta plot: forced-depth CE minus router CE (gain/loss from the router)
for i, mn in enumerate(["d1", "d2", "d3"]):
    diffs = [np.mean(results[mn][m]) - np.mean(results["router"][m]) for m in mods]
    axes[1].bar(x + (i - 1) * w, diffs, width=w,
                label=f"{mn} − router", color=colors[mn])
axes[1].axhline(0, c="k", lw=0.6)
axes[1].set_xticks(x)
axes[1].set_xticklabels(mods, rotation=15)
axes[1].set_ylabel("ΔCE vs router (nats)  [higher = worse]")
axes[1].set_title("Cost of forcing every token to a fixed depth")
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "E7_forced_depth_ce.png", dpi=120)
plt.close()

with open(OUT_DIR / "E7_forced_depth_ce.json", "w") as f:
    json.dump({mn: {m: results[mn][m] for m in mods} for mn in mode_names}, f, indent=2)

print("  per-modality mean CE:")
for mn in mode_names:
    row = " | ".join(f"{m}={np.mean(results[mn][m]):.3f}" for m in mods)
    print(f"    {mn:8s}  {row}")
print("  saved E7_forced_depth_ce.png + json")
print("DONE.")
