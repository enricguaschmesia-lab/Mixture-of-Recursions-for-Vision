"""Exploratory experiments on multimodality MoR routing.

Loads Vanilla-10000-Multimodality once, runs a battery of routing/quality probes
across several CLEVR test samples, and writes figures + JSON to OUT_DIR.

Run from project root:
    python experiments/multimodality_routing/run_exp.py
"""

import os
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# DTensor workaround (single-GPU + transformers 4.52.x)
try:
    from torch.distributed.tensor import DTensor  # noqa: F401
except ImportError:
    class DTensor:  # type: ignore
        pass
import transformers.modeling_utils
transformers.modeling_utils.DTensor = DTensor  # type: ignore[attr-defined]

import numpy as np
import torch
import matplotlib.pyplot as plt

from omegaconf import open_dict
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from util.env import load_dotenv
load_dotenv()
from paths import HF_CACHE_DIR
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

from model.util import load_model_from_config, load_checkpoint
from model.sharing_strategy import SHARING_STRATEGY
from util.config import preprocess_config
from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES, PAD_ID
from visualization.decode import _resolve_cosmos_decoder_jit


# ---------------------------------------------------------------- paths / config
PROJECT_DIR = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_DIR)
OUT_DIR = PROJECT_DIR / "experiments/multimodality_routing/out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG_NAME = "infer/vanilla_10000_multimodality"
DATA_TEST = PROJECT_DIR / "data/clevr_dataset/test"

# Experiment knobs
N_SAMPLES = 16            # how many CLEVR samples to aggregate
AUG_IDX = 0
N_QUALITY_SAMPLES = 4     # samples used for slow quality / generation tasks
TEMPERATURE = 0.9


# ---------------------------------------------------------------- model load
if GlobalHydra().is_initialized():
    GlobalHydra().clear()
with initialize_config_dir(config_dir=str(PROJECT_DIR / "conf"), version_base=None):
    cfg = compose(config_name=CFG_NAME)
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
if "mor" in cfg and cfg.mor.get("enable"):
    if cfg.mor.type == "expert":
        model.transform_layer_to_mor_expert(cfg)
    elif cfg.mor.type == "token":
        model.transform_layer_to_mor_token(cfg)
print(f"Loading checkpoint: {cfg.infer.checkpoint}")
model = load_checkpoint(model, cfg.infer.checkpoint)
model.to(DEVICE).eval()
NUM_RECURSION = cfg.recursive.num_recursion
PATCH_GRID = cfg.infer.patch_grid_size
COSMOS_PATH = cfg.infer.cosmos_model_path
TEXT_MAX_LEN = cfg.multimodal.get("text_max_length", 64)
GPT2_TOK = AutoTokenizer.from_pretrained("gpt2")
print(f"Ready. layers={model.config.num_hidden_layers}  num_recursion={NUM_RECURSION}")


# ---------------------------------------------------------------- helpers
_BO_TO_MOD = {info.bo_id: name for name, info in MODALITIES.items()}
_EO_TO_MOD = {info.eo_id: name for name, info in MODALITIES.items()}
_EO_IDS = set(_EO_TO_MOD)

_cosmos = None
def _get_cosmos():
    global _cosmos
    if _cosmos is None:
        from cosmos_tokenizer.image_lib import ImageTokenizer
        _cosmos = ImageTokenizer(
            checkpoint_dec=_resolve_cosmos_decoder_jit(COSMOS_PATH),
            device="cuda", dtype="bfloat16",
        )
    return _cosmos


def decode_image_tokens(raw: np.ndarray):
    raw = raw.flatten()[: PATCH_GRID * PATCH_GRID]
    if raw.size != PATCH_GRID * PATCH_GRID:
        return None
    t = _get_cosmos()
    idx = torch.from_numpy(raw.astype(np.int64)).reshape(1, PATCH_GRID, PATCH_GRID).to("cuda")
    with torch.no_grad():
        d = t.decode(idx).float().cpu().numpy()[0]
    d = np.clip((d + 1.0) / 2.0, 0.0, 1.0)
    return (d.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)


def chunk_text(modality, text, close=True):
    info = MODALITIES[modality]
    ids = GPT2_TOK(text, truncation=True, max_length=TEXT_MAX_LEN, return_tensors="pt")["input_ids"][0].long()
    parts = [torch.tensor([info.bo_id]), ids + info.codebook_offset]
    if close:
        parts.append(torch.tensor([info.eo_id]))
    return torch.cat(parts)


def chunk_tokens(modality, source, aug_idx=0, prefix_len=None, close=True):
    info = MODALITIES[modality]
    arr = np.load(source) if isinstance(source, (str, Path)) else np.asarray(source)
    if arr.ndim == 2:
        arr = arr[aug_idx]
    body = torch.from_numpy(arr.flatten()).long() + info.codebook_offset
    if prefix_len is not None:
        body = body[:prefix_len]
        close = False
    parts = [torch.tensor([info.bo_id]), body]
    if close:
        parts.append(torch.tensor([info.eo_id]))
    return torch.cat(parts)


def segment_by_modality(input_ids):
    if input_ids.dim() > 1:
        input_ids = input_ids[0]
    segs, s, m = [], None, None
    for i, t in enumerate(input_ids.tolist()):
        if t in _BO_TO_MOD:
            s, m = i + 1, _BO_TO_MOD[t]
        elif t in _EO_IDS and m is not None:
            segs.append((s, i, m))
            s, m = None, None
    if m is not None:
        segs.append((s, int(input_ids.shape[0]), m))
    return segs


def collect_routing(input_ids):
    """Run model once with no cache; capture per-MoR-layer expert indices.

    Returns depth (L, T) in [1, Nr] and segments list.
    """
    captured = []
    def hook(_m, _i, output):
        tei = getattr(output, "token_expert_indices", None)
        if tei is not None:
            captured.append(tei.detach().cpu())
    handles = [m.register_forward_hook(hook) for m in model.modules() if getattr(m, "mor", False)]
    try:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            model(input_ids=input_ids.to(DEVICE), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    depth = torch.stack([t[0] for t in captured], dim=0).long() + 1  # (L, T)
    return depth, segment_by_modality(input_ids)


class _EoStop(StoppingCriteria):
    def __init__(self, eo_id): self.eo_id = eo_id
    def __call__(self, input_ids, scores, **kw):
        return bool((input_ids[:, -1] == self.eo_id).all())


def generate(chunks, target_modality, do_sample=False, temperature=1.0, seed=0, max_new_tokens=None):
    info = MODALITIES[target_modality]
    prompt = torch.cat([*chunks, torch.tensor([info.bo_id])]).unsqueeze(0).to(DEVICE)
    if max_new_tokens is None:
        max_new_tokens = 257 if info.data_type == "tokens" else (TEXT_MAX_LEN + 1)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=max_new_tokens,
            use_cache=False,
            stopping_criteria=StoppingCriteriaList([_EoStop(info.eo_id)]),
            do_sample=do_sample or temperature != 1.0,
            temperature=temperature,
            pad_token_id=PAD_ID,
        )
    return out[0, prompt.shape[1]:].cpu()


def strip_specials(tokens, info):
    body = tokens[(tokens != info.bo_id) & (tokens != info.eo_id) & (tokens != PAD_ID)]
    return (body - info.codebook_offset).cpu().numpy()


# ---------------------------------------------------------------- data loading
def load_sample(sample_id):
    return {
        "id": sample_id,
        "rgb": np.load(DATA_TEST / "tok_rgb@256" / f"{sample_id}.npy")[AUG_IDX],
        "depth": np.load(DATA_TEST / "tok_depth@256" / f"{sample_id}.npy")[AUG_IDX],
        "normal": np.load(DATA_TEST / "tok_normal@256" / f"{sample_id}.npy")[AUG_IDX],
        "caption": json.load(open(DATA_TEST / "caption" / f"{sample_id}.json"))[AUG_IDX],
    }


sample_ids = [f"{i:05d}" for i in range(N_SAMPLES)]
samples = [load_sample(sid) for sid in sample_ids]
print(f"Loaded {len(samples)} samples")


# ---------------------------------------------------------------- E1: per-modality depth distribution (aggregated)
# Build the 4-modality sequence for each sample, collect routing, aggregate.
print("\n[E1] Per-modality routing distribution across samples…")
all_depth_by_mod = {m: [] for m in ("caption", "tok_rgb@256", "tok_depth@256", "tok_normal@256")}
# Also collect per-(layer, modality) depth counts
mod_layer_dist = {m: None for m in all_depth_by_mod}  # will be (L, Nr) int

for s in samples:
    chunks = [
        chunk_text("caption", s["caption"]),
        chunk_tokens("tok_rgb@256", s["rgb"]),
        chunk_tokens("tok_depth@256", s["depth"]),
        chunk_tokens("tok_normal@256", s["normal"]),
    ]
    seq = torch.cat(chunks)
    depth, segs = collect_routing(seq)
    L = depth.shape[0]
    for st, en, m in segs:
        d = depth[:, st:en]  # (L, n)
        all_depth_by_mod[m].append(d.numpy())
        # bincount per layer
        if mod_layer_dist[m] is None:
            mod_layer_dist[m] = np.zeros((L, NUM_RECURSION), dtype=np.int64)
        for li in range(L):
            c = np.bincount(d[li].numpy() - 1, minlength=NUM_RECURSION)
            mod_layer_dist[m][li] += c

# Save plot E1a: per-modality average depth ± std across samples
mod_stats = {}
for m, ds in all_depth_by_mod.items():
    flat = np.concatenate([d.ravel() for d in ds])
    mod_stats[m] = {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "frac_d1": float((flat == 1).mean()),
        "frac_d2": float((flat == 2).mean()),
        "frac_d3": float((flat == NUM_RECURSION).mean()),
        "n_tokens": int(flat.size),
    }
print(json.dumps(mod_stats, indent=2))

# E1a: bar with overall fractions per modality
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
mods = list(all_depth_by_mod.keys())
fracs = np.array([[mod_stats[m][f"frac_d{k}"] for k in (1, 2, 3)] for m in mods])
bot = np.zeros(len(mods))
colors = ["#440154", "#21918c", "#fde725"]
for k in range(NUM_RECURSION):
    axes[0].bar(mods, fracs[:, k], bottom=bot, color=colors[k], label=f"depth {k+1}")
    bot = bot + fracs[:, k]
axes[0].set_ylabel("token fraction")
axes[0].set_title(f"Depth distribution per modality (N={N_SAMPLES} samples)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].tick_params(axis="x", rotation=20)

# E1b: per-layer fraction at max depth, per modality
for m in mods:
    arr = mod_layer_dist[m].astype(float)
    arr = arr / arr.sum(axis=1, keepdims=True)
    axes[1].plot(arr[:, NUM_RECURSION - 1], marker="o", label=m)
axes[1].set_xlabel("MoR layer index")
axes[1].set_ylabel(f"fraction of tokens at max depth ({NUM_RECURSION})")
axes[1].set_title("Per-layer 'hard-token' fraction by modality")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "E1_per_modality_distribution.png", dpi=120)
plt.close()
print(f"  saved E1_per_modality_distribution.png")


# ---------------------------------------------------------------- E2: spatial routing stability for images
print("\n[E2] Spatial routing maps stability across samples…")
spatial_maps = {m: [] for m in ("tok_rgb@256", "tok_depth@256", "tok_normal@256")}
for s in samples:
    # Run each image modality alone for a clean spatial map (no other-modality context)
    for m, arr in (("tok_rgb@256", s["rgb"]), ("tok_depth@256", s["depth"]), ("tok_normal@256", s["normal"])):
        depth, segs = collect_routing(torch.cat([chunk_tokens(m, arr)]))
        st, en = next((a, b) for a, b, mm in segs if mm == m)
        body = depth[:, st:en].float().mean(dim=0).numpy()  # (256,)
        spatial_maps[m].append(body.reshape(PATCH_GRID, PATCH_GRID))

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for row, m in enumerate(spatial_maps):
    maps = np.stack(spatial_maps[m])  # (N, H, W)
    mean_map = maps.mean(axis=0)
    std_map = maps.std(axis=0)
    im0 = axes[row, 0].imshow(mean_map, cmap="viridis", vmin=1, vmax=NUM_RECURSION)
    axes[row, 0].set_title(f"{m}\nmean depth (avg over N={N_SAMPLES})", fontsize=9)
    im1 = axes[row, 1].imshow(std_map, cmap="magma", vmin=0)
    axes[row, 1].set_title(f"{m}\nstd of depth (per-cell variation)", fontsize=9)
    # Position-mean over flattened idx
    flat_mean = mean_map.flatten()
    axes[row, 2].plot(flat_mean, ".-", lw=1)
    axes[row, 2].set_title(f"{m}\nflattened mean depth vs position", fontsize=9)
    axes[row, 2].set_xlabel("token position 0..255")
    axes[row, 2].set_ylim(0.95, NUM_RECURSION + 0.05)
    fig.colorbar(im0, ax=axes[row, 0], fraction=0.046)
    fig.colorbar(im1, ax=axes[row, 1], fraction=0.046)
plt.tight_layout()
plt.savefig(OUT_DIR / "E2_spatial_stability.png", dpi=120)
plt.close()
print(f"  saved E2_spatial_stability.png")


# ---------------------------------------------------------------- E3: context effect on target routing (aggregate)
print("\n[E3] Context effect on target routing depth (aggregated)…")
TARGETS = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "caption"]
context_keys = ["none", "caption", "rgb", "caption+rgb"]
agg = {tgt: {ck: [] for ck in context_keys} for tgt in TARGETS}

for s in samples[: max(8, N_SAMPLES // 2)]:
    rgb_chunk = chunk_tokens("tok_rgb@256", s["rgb"])
    cap_chunk = chunk_text("caption", s["caption"])
    for tgt in TARGETS:
        if tgt == "tok_rgb@256":
            tgt_chunk = chunk_tokens(tgt, s["rgb"])
        elif tgt == "tok_depth@256":
            tgt_chunk = chunk_tokens(tgt, s["depth"])
        elif tgt == "tok_normal@256":
            tgt_chunk = chunk_tokens(tgt, s["normal"])
        else:
            tgt_chunk = chunk_text(tgt, s["caption"])
        ctxs = {
            "none": [],
            "caption": [cap_chunk] if tgt != "caption" else [],
            "rgb": [rgb_chunk] if tgt != "tok_rgb@256" else [],
            "caption+rgb": [cap_chunk, rgb_chunk] if tgt not in ("caption", "tok_rgb@256") else [],
        }
        for ck, ctx in ctxs.items():
            if ck != "none" and not ctx:
                continue  # skip degenerate (e.g. caption->caption)
            seq = torch.cat(ctx + [tgt_chunk])
            depth, segs = collect_routing(seq)
            st, en = next((a, b) for a, b, mm in segs if mm == tgt)
            agg[tgt][ck].append(depth[:, st:en].float().mean().item())

fig, axes = plt.subplots(1, len(TARGETS), figsize=(4.2 * len(TARGETS), 4))
for ax, tgt in zip(axes, TARGETS):
    keys = [k for k in context_keys if agg[tgt][k]]
    means = [np.mean(agg[tgt][k]) for k in keys]
    stds = [np.std(agg[tgt][k]) for k in keys]
    ax.bar(keys, means, yerr=stds, capsize=4, color="#1f77b4", alpha=0.85)
    base = np.mean(agg[tgt]["none"]) if agg[tgt]["none"] else 0.0
    ax.axhline(base, ls="--", c="grey", alpha=0.6, label="none")
    ax.set_ylim(0.95, NUM_RECURSION + 0.05)
    ax.set_title(f"target = {tgt}", fontsize=10)
    ax.set_ylabel("avg exit depth on target body")
    ax.tick_params(axis="x", rotation=20)
plt.suptitle("Routing depth on target as a function of available context", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "E3_context_effect.png", dpi=120)
plt.close()
# Also save raw numbers
with open(OUT_DIR / "E3_context_effect.json", "w") as f:
    json.dump({t: {k: [float(x) for x in v] for k, v in d.items()} for t, d in agg.items()}, f, indent=2)
print(f"  saved E3_context_effect.png + json")


# ---------------------------------------------------------------- E4: Generation vs ground-truth routing
print("\n[E4] Generation vs teacher-forced routing on the same target…")
# Compare per-modality depth distribution when target tokens are GT vs model-generated.
quality_samples = samples[: N_QUALITY_SAMPLES]
gen_vs_tf = []
for s in quality_samples:
    for tgt, arr in (("tok_rgb@256", s["rgb"]), ("tok_depth@256", s["depth"]), ("tok_normal@256", s["normal"])):
        # TF: GT body, collect routing in target slice
        d_tf, segs = collect_routing(torch.cat([chunk_tokens(tgt, arr)]))
        st, en = next((a, b) for a, b, mm in segs if mm == tgt)
        d_tf_mean = d_tf[:, st:en].float().mean().item()

        # Generation: model produces the body unconditionally, then routing-collect
        gen = generate([], target_modality=tgt, do_sample=True, temperature=TEMPERATURE, seed=int(s["id"]))
        # Strip specials, re-wrap (the generated body already ends in EO if seen) and route the produced body
        body = strip_specials(gen, MODALITIES[tgt])
        if body.size < PATCH_GRID * PATCH_GRID:
            # pad if generation stopped early
            body = np.pad(body, (0, PATCH_GRID * PATCH_GRID - body.size))
        body = body[: PATCH_GRID * PATCH_GRID]
        d_gen, segs2 = collect_routing(torch.cat([chunk_tokens(tgt, body)]))
        st2, en2 = next((a, b) for a, b, mm in segs2 if mm == tgt)
        d_gen_mean = d_gen[:, st2:en2].float().mean().item()
        gen_vs_tf.append({"id": s["id"], "modality": tgt, "tf": d_tf_mean, "gen": d_gen_mean})

# Plot
fig, ax = plt.subplots(figsize=(7, 4))
mods = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
tf_vals = [np.mean([r["tf"] for r in gen_vs_tf if r["modality"] == m]) for m in mods]
gen_vals = [np.mean([r["gen"] for r in gen_vs_tf if r["modality"] == m]) for m in mods]
x = np.arange(len(mods))
ax.bar(x - 0.18, tf_vals, width=0.35, label="teacher-forced (GT)", color="#1f77b4")
ax.bar(x + 0.18, gen_vals, width=0.35, label="model-generated", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(mods, rotation=15)
ax.set_ylim(0.95, NUM_RECURSION + 0.05)
ax.set_ylabel("avg exit depth")
ax.set_title("Routing depth: GT-token rollouts vs unconditional generations")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "E4_gen_vs_tf.png", dpi=120)
plt.close()
with open(OUT_DIR / "E4_gen_vs_tf.json", "w") as f:
    json.dump(gen_vs_tf, f, indent=2)
print(f"  saved E4_gen_vs_tf.png + json")


# ---------------------------------------------------------------- E5: position-in-modality routing
print("\n[E5] Position-in-modality routing curves…")
# For each modality, compute avg depth (averaged over layers + samples) at each within-body position.
pos_curves = {}
for s in samples:
    chunks = [
        chunk_text("caption", s["caption"]),
        chunk_tokens("tok_rgb@256", s["rgb"]),
        chunk_tokens("tok_depth@256", s["depth"]),
        chunk_tokens("tok_normal@256", s["normal"]),
    ]
    depth, segs = collect_routing(torch.cat(chunks))
    for st, en, m in segs:
        body_avg = depth[:, st:en].float().mean(dim=0).numpy()  # (n,)
        pos_curves.setdefault(m, []).append(body_avg)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
# Image modalities — plot together because they share the same length 256
for m in ("tok_rgb@256", "tok_depth@256", "tok_normal@256"):
    arr = np.stack(pos_curves[m])
    mu = arr.mean(0)
    sd = arr.std(0)
    axes[0].plot(mu, label=m)
    axes[0].fill_between(np.arange(len(mu)), mu - sd, mu + sd, alpha=0.15)
axes[0].set_ylim(0.95, NUM_RECURSION + 0.05)
axes[0].set_xlabel("position within image body (0..255)")
axes[0].set_ylabel("avg exit depth")
axes[0].set_title("Image modalities — position vs depth")
axes[0].legend(fontsize=8)

# Caption — variable length, pool to max length seen
cap = pos_curves["caption"]
maxL = max(c.size for c in cap)
mat = np.full((len(cap), maxL), np.nan)
for i, c in enumerate(cap):
    mat[i, : c.size] = c
mu = np.nanmean(mat, axis=0)
n = np.sum(~np.isnan(mat), axis=0)
axes[1].plot(mu, "o-", label="caption")
axes[1].fill_between(np.arange(maxL), mu - np.nanstd(mat, 0), mu + np.nanstd(mat, 0), alpha=0.2)
axes[1].plot(n / n.max() * NUM_RECURSION, "--", c="grey", label="sample support (rescaled)")
axes[1].set_ylim(0.95, NUM_RECURSION + 0.05)
axes[1].set_xlabel("position within caption body")
axes[1].set_title("Caption — position vs depth")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "E5_position_curves.png", dpi=120)
plt.close()
print(f"  saved E5_position_curves.png")


# ---------------------------------------------------------------- E6: per-MoR-layer modality fingerprint
print("\n[E6] Per-MoR-layer × per-modality fingerprint heatmap…")
# mod_layer_dist[m] is (L, Nr) counts; convert to fraction-at-max-depth and mean depth
L = next(iter(mod_layer_dist.values())).shape[0]
heat_mean = np.zeros((len(mods + ["caption"]), L))
heat_max = np.zeros((len(mods + ["caption"]), L))
mod_order = ["caption", "tok_rgb@256", "tok_depth@256", "tok_normal@256"]
for r, m in enumerate(mod_order):
    arr = mod_layer_dist[m].astype(float)
    p = arr / arr.sum(axis=1, keepdims=True)
    heat_mean[r] = (p * np.arange(1, NUM_RECURSION + 1)).sum(axis=1)
    heat_max[r] = p[:, -1]

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
im0 = axes[0].imshow(heat_mean, aspect="auto", cmap="viridis", vmin=1, vmax=NUM_RECURSION)
axes[0].set_yticks(range(len(mod_order)))
axes[0].set_yticklabels(mod_order)
axes[0].set_xlabel("MoR layer index")
axes[0].set_title("Mean exit depth per (modality, layer)")
fig.colorbar(im0, ax=axes[0], fraction=0.04)

im1 = axes[1].imshow(heat_max, aspect="auto", cmap="magma", vmin=0, vmax=1)
axes[1].set_yticks(range(len(mod_order)))
axes[1].set_yticklabels(mod_order)
axes[1].set_xlabel("MoR layer index")
axes[1].set_title(f"Fraction of tokens at max depth ({NUM_RECURSION})")
fig.colorbar(im1, ax=axes[1], fraction=0.04)
plt.tight_layout()
plt.savefig(OUT_DIR / "E6_layer_modality_fingerprint.png", dpi=120)
plt.close()
print(f"  saved E6_layer_modality_fingerprint.png")


# ---------------------------------------------------------------- summary JSON
summary = {
    "checkpoint": cfg.infer.checkpoint,
    "n_samples": N_SAMPLES,
    "num_recursion": NUM_RECURSION,
    "num_mor_layers": int(L),
    "per_modality_stats": mod_stats,
    "context_effect_mean_depth": {t: {k: float(np.mean(v)) if v else None for k, v in d.items()} for t, d in agg.items()},
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved summary to {OUT_DIR / 'summary.json'}")
print("DONE.")
