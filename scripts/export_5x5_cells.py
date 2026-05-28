"""Export the 5x5 cross-modal grid as 25 clean per-cell PNGs + metadata for Manim.

Mirrors the rendering logic from notebooks/260518_5x5_visualisation.ipynb, but
each cell is rendered to its own borderless square PNG (no subplot titles,
no axes), suitable for compositing in a Manim scene.

Outputs (default `outputs/manim_5x5/`):
  cells/{src}__to__{tgt}.png      — 25 cell images (src, tgt in modality order)
  legend.png                      — depth-colormap legend strip
  meta.json                       — modality order, sample id, num_recursions,
                                    per-cell flags (is_gt)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# DTensor workaround — matches notebook/infer.py for transformers 4.52.4.
try:
    from torch.distributed.tensor import DTensor  # type: ignore
except ImportError:
    class DTensor:  # type: ignore
        pass

import transformers.modeling_utils as _tmu
_tmu.DTensor = DTensor

import numpy as np
import numpy.ma as ma
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex, to_rgba
from matplotlib.patches import Rectangle

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


MODS = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "caption", "scene_desc"]
SHORT_NAME = {
    "tok_rgb@256": "rgb", "tok_depth@256": "depth", "tok_normal@256": "normal",
    "caption": "caption", "scene_desc": "scene_desc",
}

CLEVR_COLOR_MAP = {
    "gray": "#7f7f7f", "red": "#d62728", "blue": "#1f77b4",
    "green": "#2ca02c", "brown": "#8c564b", "purple": "#9467bd",
    "cyan": "#17becf", "yellow": "#e9d700",
}
_SCENE_OBJ_RE = re.compile(
    r"Object\s+\d+\s*-\s*Position:\s*x=(\d+)\s*y=(\d+)\s*"
    r"Shape:\s*(\w+)\s*Color:\s*(\w+)\s*Material:\s*(\w+)",
    re.IGNORECASE,
)

OVERLAY_ALPHA = 0.22
CAPTION_FONTSIZE = 11
CAPTION_FONTFAMILY = "monospace"


# ---------------------------------------------------------------- model setup
def build_model(config_name: str):
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    with initialize_config_dir(config_dir=str(PROJECT_DIR / "conf"), version_base=None):
        cfg = compose(config_name=config_name)
    with open_dict(cfg):
        cfg.wandb = False
        cfg.wandb_entity = ""
        cfg.wandb_project = "inference"
        cfg.wandb_run_name = "inference"
        cfg.resume_from_checkpoint = False
    cfg = preprocess_config(cfg)

    device = cfg.infer.device if torch.cuda.is_available() else "cpu"
    print(f"Building model on {device} …")
    model = load_model_from_config(cfg)
    if cfg.recursive.get("enable"):
        model, _ = SHARING_STRATEGY[cfg.model](cfg, model)
    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)
    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
    print(f"Loading checkpoint: {cfg.infer.checkpoint}")
    model = load_checkpoint(model, cfg.infer.checkpoint)
    model.to(device).eval()
    return model, cfg, device


# ---------------------------------------------------------------- chunk + gen
def chunk_text(modality, text, gpt2, text_max_len, *, close=True):
    info = MODALITIES[modality]
    ids = gpt2(text, truncation=True, max_length=text_max_len, return_tensors="pt")[
        "input_ids"
    ][0].long()
    parts = [torch.tensor([info.bo_id]), ids + info.codebook_offset]
    if close:
        parts.append(torch.tensor([info.eo_id]))
    return torch.cat(parts)


def chunk_tokens(modality, source, *, aug_idx=0, close=True):
    info = MODALITIES[modality]
    arr = np.load(source) if isinstance(source, (str, Path)) else np.asarray(source)
    if arr.ndim == 2:
        arr = arr[aug_idx]
    body = torch.from_numpy(arr.flatten()).long() + info.codebook_offset
    parts = [torch.tensor([info.bo_id]), body]
    if close:
        parts.append(torch.tensor([info.eo_id]))
    return torch.cat(parts)


class _EoModStopping(StoppingCriteria):
    def __init__(self, eo_id):
        self.eo_id = eo_id
    def __call__(self, input_ids, scores, **kw):
        return bool((input_ids[:, -1] == self.eo_id).all())


def generate(model, cfg, device, text_max_len, chunks, target_modality,
             do_sample=True, temperature=0.9, top_p=0.95, seed=0):
    info = MODALITIES[target_modality]
    prompt = torch.cat([*chunks, torch.tensor([info.bo_id])]).unsqueeze(0).to(device)
    max_new = 257 if info.data_type == "tokens" else (text_max_len + 1)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=max_new,
            use_cache=bool(cfg.infer.use_cache),
            stopping_criteria=StoppingCriteriaList([_EoModStopping(info.eo_id)]),
            do_sample=do_sample, temperature=temperature, top_p=top_p,
            pad_token_id=PAD_ID,
        )
    return out[0, prompt.shape[1]:].cpu()


# ---------------------------------------------------------------- routing capture
def _find_mor_layer(m):
    for sub in m.modules():
        if type(sub).__name__ == "MoRLlamaDecoderLayer":
            return sub
    return None


class RoutingRecorder:
    def __init__(self, layer):
        self.layer = layer; self.chunks = []; self._h = None
    def __enter__(self):
        self.chunks.clear()
        def hook(_m, _i, out):
            idx = getattr(out, "token_expert_indices", None)
            if idx is not None:
                self.chunks.append(idx.detach().cpu()[0])
        self._h = self.layer.register_forward_hook(hook); return self
    def __exit__(self, *a):
        self._h.remove()
    def all_depths(self):
        if not self.chunks:
            return torch.empty(0, dtype=torch.long)
        non_caching = len(self.chunks) >= 2 and self.chunks[1].numel() > 1
        if non_caching:
            return max(self.chunks, key=lambda c: c.numel())
        return torch.cat(self.chunks)


def generate_with_routing(model, cfg, device, text_max_len,
                          mor_layer, chunks, target_modality, **kw):
    prompt_len = sum(c.numel() for c in chunks) + 1
    with RoutingRecorder(mor_layer) as rec:
        out = generate(model, cfg, device, text_max_len,
                       chunks, target_modality, **kw)
    depths_all = rec.all_depths()
    n_gen = out.numel()
    take = max(0, min(n_gen, depths_all.numel() - prompt_len))
    depths = torch.full((n_gen,), -1, dtype=torch.long)
    if take > 0:
        depths[:take] = depths_all[prompt_len:prompt_len + take]
    return out, depths


# ---------------------------------------------------------------- helpers
def _strip_specials(tokens, info):
    body = tokens[(tokens != info.bo_id) & (tokens != info.eo_id) & (tokens != PAD_ID)]
    return (body - info.codebook_offset).cpu().numpy()


def body_and_depths(out, depths, modality):
    info = MODALITIES[modality]
    mask = (out != info.bo_id) & (out != info.eo_id) & (out != PAD_ID)
    return out[mask], depths[mask]


def _text_color_for(bg_hex):
    r, g, b, _ = to_rgba(bg_hex)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"


# ---------------------------------------------------------------- cell renderers
def _new_square_fig(size_in=4.0, dpi=200):
    fig = plt.figure(figsize=(size_in, size_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def render_image_cell(decode_fn, payload, modality, depth_cmap, depth_norm,
                      patch_grid, depths=None, *, size=4.0, dpi=200):
    fig, ax = _new_square_fig(size, dpi)
    if isinstance(payload, torch.Tensor):
        raw = _strip_specials(payload, MODALITIES[modality])
    else:
        raw = np.asarray(payload).flatten()
    img = decode_fn(raw[: patch_grid * patch_grid])
    ax.imshow(img)
    if depths is not None and isinstance(payload, torch.Tensor):
        _, body_d = body_and_depths(payload, depths, modality)
        n_need = patch_grid * patch_grid
        arr = body_d.numpy().astype(np.int64).copy()
        if arr.size < n_need:
            arr = np.concatenate([arr, np.full(n_need - arr.size, -1, dtype=arr.dtype)])
        grid = arr[:n_need].reshape(patch_grid, patch_grid)
        H, W = img.shape[:2]
        masked = ma.masked_where(grid < 0, grid)
        ax.imshow(masked, cmap=depth_cmap, norm=depth_norm,
                  interpolation="nearest", alpha=OVERLAY_ALPHA,
                  extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    return fig


def _scene_desc_text(payload):
    if isinstance(payload, torch.Tensor):
        raw = _strip_specials(payload, MODALITIES["scene_desc"]).tolist()
        return AutoTokenizer.from_pretrained("gpt2").decode(raw, skip_special_tokens=True)
    return payload


def render_scene_desc_cell(payload, depth_colors, depth_cmap, depth_norm,
                           gpt2, depths=None, *, size=4.0, dpi=200, box_size=10):
    fig, ax = _new_square_fig(size, dpi)
    ax.set_xlim(0, 100); ax.set_ylim(100, 0); ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    if depths is not None and isinstance(payload, torch.Tensor):
        info = MODALITIES["scene_desc"]
        body_t, body_d = body_and_depths(payload, depths, "scene_desc")
        raw_ids = (body_t - info.codebook_offset).tolist()
        pieces = [gpt2.decode([rid]) for rid in raw_ids]
        dlist = body_d.tolist()
        offsets = [0]
        for p in pieces:
            offsets.append(offsets[-1] + len(p))
        full_text = "".join(pieces)
        objs = []
        for m in _SCENE_OBJ_RE.finditer(full_text):
            s, e = m.span()
            tok_d = [dlist[i] for i in range(len(pieces))
                     if dlist[i] >= 0 and not (offsets[i + 1] <= s or offsets[i] >= e)]
            mean_d = int(round(sum(tok_d) / len(tok_d))) if tok_d else -1
            objs.append({"x": int(m.group(1)), "y": int(m.group(2)),
                         "shape": m.group(3).lower(), "color": m.group(4).lower(),
                         "material": m.group(5).lower(), "mean_depth": mean_d})
    else:
        text = _scene_desc_text(payload) if not isinstance(payload, str) else payload
        objs = [{"x": int(m.group(1)), "y": int(m.group(2)),
                 "shape": m.group(3).lower(), "color": m.group(4).lower(),
                 "material": m.group(5).lower(), "mean_depth": -1}
                for m in _SCENE_OBJ_RE.finditer(text)]
        dlist = []

    for o in objs:
        face = CLEVR_COLOR_MAP.get(o["color"], "#000000")
        inner_edge = "white" if o["material"] == "metal" else "black"
        if o["mean_depth"] >= 0:
            outer = depth_colors[o["mean_depth"]]
            ax.add_patch(Rectangle(
                (o["x"] - box_size / 2 - 1.4, o["y"] - box_size / 2 - 1.4),
                box_size + 2.8, box_size + 2.8,
                facecolor="none", edgecolor=outer, linewidth=2.5))
        ax.add_patch(Rectangle(
            (o["x"] - box_size / 2, o["y"] - box_size / 2),
            box_size, box_size, facecolor=face, edgecolor=inner_edge, linewidth=1.0))

    if dlist:
        inset = ax.inset_axes([0.02, 0.02, 0.96, 0.06])
        arr = np.array(dlist, dtype=np.int64).reshape(1, -1)
        masked = ma.masked_where(arr < 0, arr)
        inset.imshow(masked, cmap=depth_cmap, norm=depth_norm,
                     aspect="auto", interpolation="nearest")
        inset.set_xticks([]); inset.set_yticks([])
        for s in inset.spines.values():
            s.set_edgecolor("black"); s.set_linewidth(0.5)
    return fig


def render_caption_cell(payload, depth_colors, gpt2, depths=None,
                        *, size=4.0, dpi=200):
    fig, ax = _new_square_fig(size, dpi)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_facecolor("white")

    if depths is None or not isinstance(payload, torch.Tensor):
        if isinstance(payload, torch.Tensor):
            raw = _strip_specials(payload, MODALITIES["caption"]).tolist()
            text = gpt2.decode(raw, skip_special_tokens=True)
        else:
            text = payload
        ax.text(0.04, 0.5, textwrap.fill(text, width=22),
                fontsize=CAPTION_FONTSIZE, family=CAPTION_FONTFAMILY, va="center")
        return fig

    info = MODALITIES["caption"]
    body_t, body_d = body_and_depths(payload, depths, "caption")
    raw_ids = (body_t - info.codebook_offset).tolist()
    pieces = [gpt2.decode([rid]) for rid in raw_ids]
    dlist = body_d.tolist()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    x, y = 0.04, 0.94
    line_h = 0.10
    for piece, d in zip(pieces, dlist):
        if piece == "":
            continue
        bg = depth_colors[d] if d >= 0 else "#dddddd"
        fg = _text_color_for(bg) if d >= 0 else "black"
        t = ax.text(x, y, piece, fontsize=CAPTION_FONTSIZE, va="top", ha="left",
                    color=fg, family=CAPTION_FONTFAMILY,
                    bbox=dict(facecolor=bg, edgecolor="none", pad=0.8),
                    transform=ax.transAxes)
        bb = t.get_window_extent(renderer=renderer).transformed(inv)
        if bb.x1 > 0.96 and x > 0.04:
            t.remove()
            x = 0.04; y -= line_h
            if y < 0.04:
                break
            t = ax.text(x, y, piece, fontsize=CAPTION_FONTSIZE, va="top", ha="left",
                        color=fg, family=CAPTION_FONTFAMILY,
                        bbox=dict(facecolor=bg, edgecolor="none", pad=0.8),
                        transform=ax.transAxes)
            bb = t.get_window_extent(renderer=renderer).transformed(inv)
        x = bb.x1 + 0.005
    return fig


def render_cell(payload, modality, *, decode_fn, depth_colors, depth_cmap,
                depth_norm, gpt2, patch_grid, depths=None):
    if modality in {"tok_rgb@256", "tok_depth@256", "tok_normal@256"}:
        return render_image_cell(decode_fn, payload, modality, depth_cmap,
                                 depth_norm, patch_grid, depths=depths)
    if modality == "scene_desc":
        return render_scene_desc_cell(payload, depth_colors, depth_cmap,
                                      depth_norm, gpt2, depths=depths)
    return render_caption_cell(payload, depth_colors, gpt2, depths=depths)


def render_legend(num_recur, depth_colors, out_path, *, dpi=200):
    fig, ax = plt.subplots(figsize=(6, 0.8), dpi=dpi)
    ax.set_axis_off()
    for k in range(num_recur):
        ax.add_patch(Rectangle((k, 0), 1, 1, facecolor=depth_colors[k],
                               edgecolor="black", linewidth=0.8))
        ax.text(k + 0.5, 0.5, f"{k + 1} recursion{'s' if k else ''}",
                ha="center", va="center",
                color=_text_color_for(depth_colors[k]),
                fontsize=11, family="monospace")
    ax.set_xlim(0, num_recur); ax.set_ylim(0, 1); ax.set_aspect("equal")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- cosmos decoder
def make_decode_fn(cfg, device):
    patch_grid = cfg.infer.patch_grid_size
    cosmos_path = cfg.infer.cosmos_model_path
    tokenizer = None
    def decode(raw_tokens):
        nonlocal tokenizer
        if raw_tokens.size != patch_grid * patch_grid:
            return np.zeros((patch_grid * 8, patch_grid * 8, 3), dtype=np.uint8)
        if device != "cuda":
            raise RuntimeError("Cosmos decode requires CUDA")
        if tokenizer is None:
            from cosmos_tokenizer.image_lib import ImageTokenizer
            decoder_jit = _resolve_cosmos_decoder_jit(cosmos_path)
            tokenizer = ImageTokenizer(checkpoint_dec=decoder_jit,
                                       device="cuda", dtype="bfloat16")
        indices = (torch.from_numpy(raw_tokens.astype(np.int64))
                   .reshape(1, patch_grid, patch_grid).to("cuda"))
        with torch.no_grad():
            decoded = tokenizer.decode(indices).float().cpu().numpy()[0]
        decoded = np.clip((decoded + 1.0) / 2.0, 0.0, 1.0)
        return (decoded.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return decode, patch_grid


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="infer/mor_5000_multimodality_3r.yaml")
    p.add_argument("--sample-id", default="00007")
    p.add_argument("--aug-idx", type=int, default=0)
    p.add_argument("--data-dir", default="data/clevr_dataset/test")
    p.add_argument("--out-dir", default="outputs/manim_5x5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--cell-size-in", type=float, default=4.0)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    out_dir = PROJECT_DIR / args.out_dir
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)

    model, cfg, device = build_model(args.config)
    text_max_len = cfg.multimodal.get("text_max_length", 64)
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    decode_fn, patch_grid = make_decode_fn(cfg, device)

    mor_layer = _find_mor_layer(model)
    if mor_layer is None:
        raise RuntimeError("No MoRLlamaDecoderLayer; this checkpoint is not MoR-routed.")
    num_recur = int(cfg.recursive.num_recursion)

    _viridis = mpl.colormaps["viridis"]
    depth_colors = [to_hex(_viridis(i / max(1, num_recur - 1))) for i in range(num_recur)]
    depth_cmap = ListedColormap(depth_colors)
    depth_norm = BoundaryNorm(
        boundaries=[i - 0.5 for i in range(num_recur + 1)], ncolors=num_recur)

    data_test = PROJECT_DIR / args.data_dir
    ref_rgb = np.load(data_test / "tok_rgb@256" / f"{args.sample_id}.npy")[args.aug_idx]
    ref_depth = np.load(data_test / "tok_depth@256" / f"{args.sample_id}.npy")[args.aug_idx]
    ref_normal = np.load(data_test / "tok_normal@256" / f"{args.sample_id}.npy")[args.aug_idx]
    with open(data_test / "caption" / f"{args.sample_id}.json") as f:
        ref_captions = json.load(f)
    with open(data_test / "scene_desc" / f"{args.sample_id}.json") as f:
        ref_scene_descs = json.load(f)

    gt = {
        "tok_rgb@256": ref_rgb, "tok_depth@256": ref_depth,
        "tok_normal@256": ref_normal,
        "caption": ref_captions[args.aug_idx],
        "scene_desc": ref_scene_descs[args.aug_idx],
    }

    def build_input_chunk(modality):
        payload = gt[modality]
        if MODALITIES[modality].data_type == "tokens":
            return chunk_tokens(modality, payload)
        return chunk_text(modality, payload, gpt2, text_max_len)

    cells_meta = {}
    for i, src in enumerate(MODS):
        for j, tgt in enumerate(MODS):
            name = f"{SHORT_NAME[src]}__to__{SHORT_NAME[tgt]}"
            out_path = out_dir / "cells" / f"{name}.png"
            is_gt = (i == j)
            print(f"[{i},{j}] {name}  {'(GT)' if is_gt else ''}")
            if is_gt:
                fig = render_cell(gt[tgt], tgt,
                                  decode_fn=decode_fn, depth_colors=depth_colors,
                                  depth_cmap=depth_cmap, depth_norm=depth_norm,
                                  gpt2=gpt2, patch_grid=patch_grid)
            else:
                chunks = [build_input_chunk(src)]
                out_t, depths = generate_with_routing(
                    model, cfg, device, text_max_len,
                    mor_layer, chunks, tgt,
                    do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, seed=args.seed)
                fig = render_cell(out_t, tgt,
                                  decode_fn=decode_fn, depth_colors=depth_colors,
                                  depth_cmap=depth_cmap, depth_norm=depth_norm,
                                  gpt2=gpt2, patch_grid=patch_grid, depths=depths)
            fig.set_size_inches(args.cell_size_in, args.cell_size_in)
            fig.savefig(out_path, dpi=args.dpi, bbox_inches=None,
                        pad_inches=0, facecolor="white")
            plt.close(fig)
            cells_meta[name] = {
                "src": src, "tgt": tgt, "row": i, "col": j,
                "is_gt": is_gt, "file": f"cells/{name}.png",
            }

    render_legend(num_recur, depth_colors, out_dir / "legend.png", dpi=args.dpi)

    meta = {
        "modalities": MODS,
        "short_name": SHORT_NAME,
        "num_recursions": num_recur,
        "depth_colors": depth_colors,
        "sample_id": args.sample_id,
        "aug_idx": args.aug_idx,
        "cells": cells_meta,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Wrote {len(cells_meta)} cells + legend + meta.json to {out_dir}")


if __name__ == "__main__":
    main()
