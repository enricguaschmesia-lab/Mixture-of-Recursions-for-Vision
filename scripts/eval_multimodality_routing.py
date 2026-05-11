"""Multimodality routing & quality evaluation for token-choice MoR.

Loads any inference config that points to a multimodality (caption + tok_rgb +
tok_depth + tok_normal) checkpoint and runs a focused battery of probes built
from the patterns we found in `notebooks/260507_multimodality_inference.ipynb`:

    1. Per-modality depth distribution  (the headline result: each modality
       lives in its own compute lane — caption≈2, rgb≈1.4, depth≈2.7, normal≈1.1).
    2. Spatial routing maps for image modalities, averaged across N samples
       (stability check + reveals where the router spends compute spatially).
    3. Context-effect: does conditioning context change routing on the target?
       (Answer for the released ckpt: almost no — router ≈ modality identity.)
    4. Generation vs teacher-forced: does the router behave the same on
       model-rolled-out tokens? (Yes, within ~0.1.)
    5. Forced-depth ablation: replace the router with a constant choice (d=1,
       2, 3) while preserving the trained gating weights, and measure
       teacher-forced cross-entropy per modality. The router consistently
       beats every fixed depth, including d=3 (max compute) — evidence that
       within-modality token-level routing is doing real work.

Outputs:
    <out>/E1_modality_distribution.png  + summary.json["per_modality_stats"]
    <out>/E2_spatial_maps.png
    <out>/E3_context_effect.png         + E3_context_effect.json
    <out>/E4_gen_vs_tf.png               + E4_gen_vs_tf.json
    <out>/E5_forced_depth_ce.png         + E5_forced_depth_ce.json
    <out>/summary.json

Usage:
    python scripts/eval_multimodality_routing.py --config infer/vanilla_10000_multimodality
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# transformers 4.52 + single-GPU DTensor workaround (same trick as infer.py)
try:
    from torch.distributed.tensor import DTensor  # noqa: F401
except ImportError:
    class DTensor:  # type: ignore
        pass
import transformers.modeling_utils
transformers.modeling_utils.DTensor = DTensor  # type: ignore[attr-defined]

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


# ============================================================ Args & config
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="infer/vanilla_10000_multimodality",
                   help="Hydra config name under conf/ (e.g. infer/vanilla_10000_multimodality)")
    p.add_argument("--data-dir", default="data/clevr_dataset/test")
    p.add_argument("--n-samples", type=int, default=16,
                   help="CLEVR test samples for aggregated routing stats")
    p.add_argument("--n-quality-samples", type=int, default=8,
                   help="Samples used for the forced-depth CE ablation (each forward is ~Nr× slower)")
    p.add_argument("--n-gen-samples", type=int, default=3,
                   help="Samples used for the (slow) gen-vs-TF routing comparison")
    p.add_argument("--aug-idx", type=int, default=0)
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: results/eval/<config-leaf>)")
    return p.parse_args()


# ============================================================ Setup
def build_model(cfg_name: str):
    """Compose the Hydra config and instantiate the trained model."""
    project_dir = Path(__file__).resolve().parents[1]
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    with initialize_config_dir(config_dir=str(project_dir / "conf"), version_base=None):
        cfg = compose(config_name=cfg_name)
    with open_dict(cfg):
        cfg.wandb = False
        cfg.wandb_entity = ""
        cfg.wandb_project = "eval"
        cfg.wandb_run_name = "eval"
        cfg.resume_from_checkpoint = False
    cfg = preprocess_config(cfg)

    device = cfg.infer.device if torch.cuda.is_available() else "cpu"
    model = load_model_from_config(cfg)
    if cfg.recursive.get("enable"):
        model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
        elif cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
    model = load_checkpoint(model, cfg.infer.checkpoint)
    model.to(device).eval()
    return cfg, model, device


# ============================================================ Helpers
class _EoStop(StoppingCriteria):
    def __init__(self, eo_id):
        self.eo_id = eo_id
    def __call__(self, input_ids, scores, **kw):
        return bool((input_ids[:, -1] == self.eo_id).all())


class Modality:
    """Tiny convenience namespace over MODALITIES + tokenizers."""
    def __init__(self, text_tokenizer: AutoTokenizer, text_max_len: int):
        self.tok = text_tokenizer
        self.text_max_len = text_max_len
        self.bo_to_mod = {info.bo_id: name for name, info in MODALITIES.items()}
        self.eo_to_mod = {info.eo_id: name for name, info in MODALITIES.items()}
        self.eo_ids = set(self.eo_to_mod)

    def chunk_text(self, modality: str, text: str, close: bool = True) -> torch.Tensor:
        info = MODALITIES[modality]
        ids = self.tok(text, truncation=True, max_length=self.text_max_len,
                       return_tensors="pt")["input_ids"][0].long()
        parts = [torch.tensor([info.bo_id]), ids + info.codebook_offset]
        if close: parts.append(torch.tensor([info.eo_id]))
        return torch.cat(parts)

    def chunk_tokens(self, modality: str, source, aug_idx: int = 0, close: bool = True) -> torch.Tensor:
        info = MODALITIES[modality]
        arr = np.load(source) if isinstance(source, (str, Path)) else np.asarray(source)
        if arr.ndim == 2: arr = arr[aug_idx]
        body = torch.from_numpy(arr.flatten()).long() + info.codebook_offset
        parts = [torch.tensor([info.bo_id]), body]
        if close: parts.append(torch.tensor([info.eo_id]))
        return torch.cat(parts)

    def segment(self, input_ids: torch.Tensor) -> list[tuple[int, int, str]]:
        """Walk a token sequence and return (body_start, body_end_excl, modality)."""
        if input_ids.dim() > 1: input_ids = input_ids[0]
        segs, s, m = [], None, None
        for i, t in enumerate(input_ids.tolist()):
            if t in self.bo_to_mod:
                s, m = i + 1, self.bo_to_mod[t]
            elif t in self.eo_ids and m is not None:
                segs.append((s, i, m))
                s, m = None, None
        if m is not None:
            segs.append((s, int(input_ids.shape[0]), m))
        return segs


def collect_routing(model, mor_mod, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    """One no-cache forward; returns per-MoR-module depth choices, shape (L, T) in [1, Nr].
    For middle_cycle there is a single MoR module so L=1, but the same code handles
    multi-module variants (cycle / sequence) transparently."""
    captured: list[torch.Tensor] = []
    def hook(_m, _i, output):
        tei = getattr(output, "token_expert_indices", None)
        if tei is not None:
            captured.append(tei.detach().cpu())
    handles = [m.register_forward_hook(hook) for m in model.modules() if getattr(m, "mor", False)]
    try:
        if input_ids.dim() == 1: input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            model(input_ids=input_ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if not captured:
        raise RuntimeError("No MoR layers fired — pass a token-choice MoR checkpoint.")
    return torch.stack([t[0] for t in captured], dim=0).long() + 1  # (L, T)


def load_sample(data_dir: Path, sid: str, aug_idx: int):
    return dict(
        id=sid,
        rgb=np.load(data_dir / "tok_rgb@256" / f"{sid}.npy")[aug_idx],
        depth=np.load(data_dir / "tok_depth@256" / f"{sid}.npy")[aug_idx],
        normal=np.load(data_dir / "tok_normal@256" / f"{sid}.npy")[aug_idx],
        caption=json.load(open(data_dir / "caption" / f"{sid}.json"))[aug_idx],
    )


# ============================================================ Forced-depth ablation
# Idea: keep the trained router AND its softmax weights, but post-process the
# argmax to force a chosen expert. We monkey-patch `torch.topk` only inside the
# MoR module's forward — preserving the natural gating weight that the router
# would have given the forced expert, instead of slamming it to 1.0.
class _TopkOverride:
    """Context manager: while inside `with _TopkOverride(expert=k):`, every
    call to torch.topk with k=1 along dim=-1 returns indices=k everywhere."""
    def __init__(self, expert: int):
        self.expert = expert
        self._orig = None
    def __enter__(self):
        self._orig = torch.topk
        expert = self.expert
        orig = self._orig
        def patched(input, k, dim=-1, largest=True, sorted=True, *, out=None):
            if k == 1 and dim in (-1, input.ndim - 1):
                # Return values gathered at `expert` and indices = expert everywhere
                idx_shape = list(input.shape)
                idx_shape[-1] = 1
                idx = torch.full(idx_shape, expert, dtype=torch.long, device=input.device)
                vals = torch.gather(input, -1, idx)
                return torch.return_types.topk((vals, idx))
            return orig(input, k, dim=dim, largest=largest, sorted=sorted)
        torch.topk = patched
        return self
    def __exit__(self, *a):
        torch.topk = self._orig


def teacher_forced_ce_per_modality(model, mod: Modality, seq: torch.Tensor,
                                    device: str, segs) -> dict[str, float]:
    """One TF forward, returns mean CE per modality body slice."""
    with torch.no_grad():
        out = model(input_ids=seq.to(device), use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = seq[:, 1:]
    ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1).to(logits.device),
        reduction="none",
    ).reshape(labels.shape).squeeze(0).cpu().numpy()
    res = {}
    for st, en, m in segs:
        idxs = np.arange(max(st - 1, 0), en - 1)
        if idxs.size > 0:
            res[m] = float(ce[idxs].mean())
    return res


# ============================================================ Experiments
def exp_modality_distribution(model, mod, samples, num_recursion, device, out_dir):
    """E1: per-modality depth distribution across N samples."""
    print("\n[E1] per-modality depth distribution…")
    mor_mod = next(m for m in model.modules() if getattr(m, "mor", False))
    bag = {m: [] for m in ("caption", "tok_rgb@256", "tok_depth@256", "tok_normal@256")}
    for s in samples:
        chunks = [
            mod.chunk_text("caption", s["caption"]),
            mod.chunk_tokens("tok_rgb@256", s["rgb"]),
            mod.chunk_tokens("tok_depth@256", s["depth"]),
            mod.chunk_tokens("tok_normal@256", s["normal"]),
        ]
        seq = torch.cat(chunks)
        d = collect_routing(model, mor_mod, seq, device)
        for st, en, m in mod.segment(seq):
            bag[m].append(d[:, st:en].numpy())
    stats = {}
    for m, ds in bag.items():
        flat = np.concatenate([d.ravel() for d in ds])
        stats[m] = {
            "mean": float(flat.mean()), "std": float(flat.std()),
            "n_tokens": int(flat.size),
            **{f"frac_d{k}": float((flat == k).mean()) for k in range(1, num_recursion + 1)},
        }

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    mods = list(bag.keys())
    fracs = np.array([[stats[m][f"frac_d{k}"] for k in range(1, num_recursion + 1)] for m in mods])
    bot = np.zeros(len(mods))
    palette = plt.get_cmap("viridis")(np.linspace(0, 0.9, num_recursion))
    for k in range(num_recursion):
        ax.bar(mods, fracs[:, k], bottom=bot, color=palette[k], label=f"depth {k + 1}",
               edgecolor="white")
        bot = bot + fracs[:, k]
    for i, m in enumerate(mods):
        ax.text(i, 1.02, f"μ={stats[m]['mean']:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of tokens")
    ax.set_title(f"Per-modality routing depth (N={len(samples)} samples)")
    ax.legend(loc="lower right", fontsize=8)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(out_dir / "E1_modality_distribution.png", dpi=120)
    plt.close()
    return stats


def exp_spatial_maps(model, mod, samples, num_recursion, patch_grid, device, out_dir):
    """E2: spatial routing maps for image modalities, averaged across samples."""
    print("\n[E2] spatial routing maps…")
    mor_mod = next(m for m in model.modules() if getattr(m, "mor", False))
    image_mods = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
    maps = {m: [] for m in image_mods}
    for s in samples:
        for m, arr in zip(image_mods, [s["rgb"], s["depth"], s["normal"]]):
            d = collect_routing(model, mor_mod, mod.chunk_tokens(m, arr), device)
            segs = mod.segment(mod.chunk_tokens(m, arr))
            st, en = next((a, b) for a, b, mm in segs if mm == m)
            maps[m].append(d[:, st:en].float().mean(dim=0).numpy().reshape(patch_grid, patch_grid))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))
    for col, m in enumerate(image_mods):
        arr = np.stack(maps[m])
        mu, sd = arr.mean(0), arr.std(0)
        im0 = axes[0, col].imshow(mu, cmap="viridis", vmin=1, vmax=num_recursion)
        axes[0, col].set_title(f"{m}\nmean depth (μ={mu.mean():.2f})", fontsize=10)
        axes[0, col].axis("off")
        im1 = axes[1, col].imshow(sd, cmap="magma", vmin=0)
        axes[1, col].set_title(f"{m}\nper-cell std (across samples)", fontsize=10)
        axes[1, col].axis("off")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046)
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046)
    plt.suptitle(f"Spatial routing stability across {len(samples)} samples", y=1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "E2_spatial_maps.png", dpi=120)
    plt.close()


def exp_context_effect(model, mod, samples, num_recursion, device, out_dir):
    """E3: how does adding caption / rgb context shift target-body routing depth?"""
    print("\n[E3] context effect on target routing…")
    mor_mod = next(m for m in model.modules() if getattr(m, "mor", False))
    TARGETS = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "caption"]
    keys = ["none", "caption", "rgb", "caption+rgb"]
    agg = {t: {k: [] for k in keys} for t in TARGETS}

    for s in samples:
        cap = mod.chunk_text("caption", s["caption"])
        rgb = mod.chunk_tokens("tok_rgb@256", s["rgb"])
        chunks_for = {
            "tok_rgb@256":   mod.chunk_tokens("tok_rgb@256", s["rgb"]),
            "tok_depth@256": mod.chunk_tokens("tok_depth@256", s["depth"]),
            "tok_normal@256": mod.chunk_tokens("tok_normal@256", s["normal"]),
            "caption":       mod.chunk_text("caption", s["caption"]),
        }
        for tgt in TARGETS:
            tgt_chunk = chunks_for[tgt]
            ctxs = {
                "none": [],
                "caption": [cap] if tgt != "caption" else [],
                "rgb": [rgb] if tgt != "tok_rgb@256" else [],
                "caption+rgb": [cap, rgb] if tgt not in ("caption", "tok_rgb@256") else [],
            }
            for ck, ctx in ctxs.items():
                if ck != "none" and not ctx: continue
                seq = torch.cat(ctx + [tgt_chunk])
                d = collect_routing(model, mor_mod, seq, device)
                segs = mod.segment(seq)
                st, en = next((a, b) for a, b, mm in segs if mm == tgt)
                agg[tgt][ck].append(d[:, st:en].float().mean().item())

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 3.8))
    for ax, tgt in zip(axes, TARGETS):
        ks = [k for k in keys if agg[tgt][k]]
        means = [np.mean(agg[tgt][k]) for k in ks]
        stds = [np.std(agg[tgt][k]) for k in ks]
        ax.bar(ks, means, yerr=stds, capsize=4, color="#1f77b4", alpha=0.85)
        ax.set_ylim(0.95, num_recursion + 0.05)
        ax.set_title(f"target = {tgt}", fontsize=10)
        ax.set_ylabel("avg exit depth")
        ax.tick_params(axis="x", rotation=15)
    plt.suptitle("Routing depth on target as a function of context", y=1.03)
    plt.tight_layout()
    plt.savefig(out_dir / "E3_context_effect.png", dpi=120)
    plt.close()
    with open(out_dir / "E3_context_effect.json", "w") as f:
        json.dump({t: {k: [float(x) for x in v] for k, v in d.items()} for t, d in agg.items()}, f, indent=2)
    return {t: {k: float(np.mean(v)) if v else None for k, v in d.items()} for t, d in agg.items()}


def exp_gen_vs_tf(model, mod, samples, num_recursion, patch_grid, device, out_dir,
                  temperature=0.9):
    """E4: routing on GT (teacher-forced) vs on model-generated body, for image modalities."""
    print("\n[E4] generation vs teacher-forced routing…")
    mor_mod = next(m for m in model.modules() if getattr(m, "mor", False))
    rows = []
    for s in samples:
        for tgt, arr in (("tok_rgb@256", s["rgb"]), ("tok_depth@256", s["depth"]),
                         ("tok_normal@256", s["normal"])):
            # TF
            seq_tf = mod.chunk_tokens(tgt, arr)
            d_tf = collect_routing(model, mor_mod, seq_tf, device)
            segs = mod.segment(seq_tf)
            st, en = next((a, b) for a, b, mm in segs if mm == tgt)
            tf_mean = d_tf[:, st:en].float().mean().item()

            # Gen
            info = MODALITIES[tgt]
            prompt = torch.tensor([info.bo_id]).unsqueeze(0).to(device)
            torch.manual_seed(int(s["id"]))
            with torch.no_grad():
                gen = model.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=257,
                    use_cache=False,
                    stopping_criteria=StoppingCriteriaList([_EoStop(info.eo_id)]),
                    do_sample=True, temperature=temperature, pad_token_id=PAD_ID,
                )
            body = gen[0, 1:].cpu()
            body = body[(body != info.bo_id) & (body != info.eo_id) & (body != PAD_ID)]
            body = (body - info.codebook_offset).numpy()
            if body.size < patch_grid * patch_grid:
                body = np.pad(body, (0, patch_grid * patch_grid - body.size))
            body = body[: patch_grid * patch_grid]

            d_gen = collect_routing(model, mor_mod, mod.chunk_tokens(tgt, body), device)
            segs2 = mod.segment(mod.chunk_tokens(tgt, body))
            st2, en2 = next((a, b) for a, b, mm in segs2 if mm == tgt)
            gen_mean = d_gen[:, st2:en2].float().mean().item()
            rows.append({"id": s["id"], "modality": tgt, "tf": tf_mean, "gen": gen_mean})

    fig, ax = plt.subplots(figsize=(7, 4))
    mods = ["tok_rgb@256", "tok_depth@256", "tok_normal@256"]
    tf_v = [np.mean([r["tf"] for r in rows if r["modality"] == m]) for m in mods]
    gn_v = [np.mean([r["gen"] for r in rows if r["modality"] == m]) for m in mods]
    x = np.arange(len(mods))
    ax.bar(x - 0.18, tf_v, width=0.35, label="teacher-forced (GT body)", color="#1f77b4")
    ax.bar(x + 0.18, gn_v, width=0.35, label="model-generated body",   color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(mods, rotation=15)
    ax.set_ylim(0.95, num_recursion + 0.05)
    ax.set_ylabel("avg exit depth")
    ax.set_title(f"GT vs generated routing  (N={len(samples)}, T={temperature})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "E4_gen_vs_tf.png", dpi=120)
    plt.close()
    with open(out_dir / "E4_gen_vs_tf.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def exp_forced_depth_ce(model, mod, samples, num_recursion, device, out_dir):
    """E5: forced-depth ablation. Trained router vs each fixed depth ∈ [1..Nr].
    We override only the argmax (via torch.topk patch); the trained softmax
    weight at the chosen expert is preserved so we ablate routing alone, not
    routing+gating."""
    print("\n[E5] forced-depth CE ablation…")
    modes = {"router": None, **{f"d{k+1}": k for k in range(num_recursion)}}
    mods = ["caption", "tok_rgb@256", "tok_depth@256", "tok_normal@256"]
    res = {mn: {m: [] for m in mods} for mn in modes}
    for s in samples:
        chunks = [
            mod.chunk_text("caption", s["caption"]),
            mod.chunk_tokens("tok_rgb@256", s["rgb"]),
            mod.chunk_tokens("tok_depth@256", s["depth"]),
            mod.chunk_tokens("tok_normal@256", s["normal"]),
        ]
        seq = torch.cat(chunks).unsqueeze(0)
        segs = mod.segment(seq)
        for mn, expert in modes.items():
            ctx = _TopkOverride(expert) if expert is not None else _NullCtx()
            with ctx:
                ce_per_mod = teacher_forced_ce_per_modality(model, mod, seq, device, segs)
            for m, v in ce_per_mod.items():
                res[mn][m].append(v)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    mode_names = list(modes.keys())
    palette = {"router": "#1f77b4", **{f"d{k+1}": plt.get_cmap("viridis")(0.15 + 0.7 * k / max(num_recursion - 1, 1))
                                       for k in range(num_recursion)}}
    w = 0.8 / len(mode_names)
    x = np.arange(len(mods))
    for i, mn in enumerate(mode_names):
        means = [np.mean(res[mn][m]) for m in mods]
        stds = [np.std(res[mn][m]) for m in mods]
        axes[0].bar(x + (i - (len(mode_names) - 1) / 2) * w, means, width=w, yerr=stds,
                    capsize=3, label=mn, color=palette[mn])
    axes[0].set_xticks(x); axes[0].set_xticklabels(mods, rotation=15)
    axes[0].set_ylabel("teacher-forced CE (nats)")
    axes[0].set_title("Per-modality CE under {router, fixed-depth}")
    axes[0].legend(fontsize=9)
    axes[0].set_yscale("log")

    for i, mn in enumerate([f"d{k+1}" for k in range(num_recursion)]):
        diffs = [np.mean(res[mn][m]) - np.mean(res["router"][m]) for m in mods]
        axes[1].bar(x + (i - (num_recursion - 1) / 2) * w * 1.2, diffs, width=w,
                    label=f"{mn} − router", color=palette[mn])
    axes[1].axhline(0, c="k", lw=0.6)
    axes[1].set_xticks(x); axes[1].set_xticklabels(mods, rotation=15)
    axes[1].set_ylabel("ΔCE vs router (higher = worse)")
    axes[1].set_title("Cost of forcing every token to a fixed depth\n(only argmax is overridden; gating weights preserved)")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "E5_forced_depth_ce.png", dpi=120)
    plt.close()
    with open(out_dir / "E5_forced_depth_ce.json", "w") as f:
        json.dump(res, f, indent=2)
    return {mn: {m: float(np.mean(res[mn][m])) for m in mods} for mn in mode_names}


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ============================================================ Main
def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    os.chdir(project_dir)
    data_dir = Path(args.data_dir).resolve()

    out_dir = Path(args.out_dir) if args.out_dir else (
        project_dir / "results/eval" / Path(args.config).name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    cfg, model, device = build_model(args.config)
    num_recursion = cfg.recursive.num_recursion
    patch_grid = cfg.infer.patch_grid_size
    text_max_len = cfg.multimodal.get("text_max_length", 64)
    mod = Modality(AutoTokenizer.from_pretrained("gpt2"), text_max_len)
    print(f"Model ready. num_recursion={num_recursion}  patch_grid={patch_grid}")

    sids = [f"{i:05d}" for i in range(args.n_samples)]
    samples = [load_sample(data_dir, sid, args.aug_idx) for sid in sids]

    summary = {
        "config": args.config,
        "checkpoint": cfg.infer.checkpoint,
        "num_recursion": num_recursion,
        "n_samples": args.n_samples,
    }

    summary["per_modality_stats"] = exp_modality_distribution(
        model, mod, samples, num_recursion, device, out_dir,
    )
    exp_spatial_maps(model, mod, samples, num_recursion, patch_grid, device, out_dir)
    summary["context_effect_mean_depth"] = exp_context_effect(
        model, mod, samples[: max(args.n_quality_samples, 4)], num_recursion, device, out_dir,
    )
    summary["gen_vs_tf"] = exp_gen_vs_tf(
        model, mod, samples[: args.n_gen_samples], num_recursion, patch_grid, device, out_dir,
    )
    summary["forced_depth_mean_ce"] = exp_forced_depth_ce(
        model, mod, samples[: args.n_quality_samples], num_recursion, device, out_dir,
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll experiments done. Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
