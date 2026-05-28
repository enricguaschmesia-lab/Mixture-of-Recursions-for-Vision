"""
Pass 1: generate scene_desc text completions from RGB tokens for the first N
augmentation-0 samples of the CLEVR test split.

Writes one JSON line per sample to <out-dir>/predictions.jsonl:
    {"id": "00000",
     "generated_text": "Object 1 - Position: ...",
     "n_new_tokens": 127,
     "stopped_on_eo": true,
     "gen_seconds": 0.83}

Usage (from repo root):
    PYTHONPATH=. uv run python scripts/generate_scene_desc.py --config infer/scene_desc_random --data-dir /home/gianfranco/projects/2025/Visual_Intelligence_Project/Dataset/clevr_com_304/test --n-samples 500 --out-dir results/eval/scene_desc/random_router
    
    
    
    uv run python scripts/generate_scene_desc.py \\
        --config infer/scene_desc_mor_token \\
        --data-dir /home/gianfranco/projects/2025/Visual_Intelligence_Project/Dataset/clevr_com_304/test \\
        --n-samples 500 \\
        --out-dir results/eval/scene_desc/mor_token
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# DTensor workaround (same as infer.py / pretrain.py) — must happen before
# any transformers import that touches modeling_utils.
try:
    from torch.distributed.tensor import DTensor  # noqa: F401
except ImportError:
    class DTensor:  # type: ignore
        pass
import transformers.modeling_utils
transformers.modeling_utils.DTensor = DTensor  # type: ignore[attr-defined]

import numpy as np
import torch
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="infer/scene_desc_mor_token",
                   help="Hydra config name under conf/")
    p.add_argument("--data-dir", required=True,
                   help="Test split root, e.g. .../clevr_com_304/test")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--aug-idx", type=int, default=0,
                   help="Augmentation index to use for both rgb and (later) GT scoring")
    p.add_argument("--out-dir", required=True,
                   help="Output dir (predictions.jsonl is written there)")
    p.add_argument("--start", type=int, default=0,
                   help="First sample id (inclusive), useful for resume/shard")
    return p.parse_args()


class _EoStop(StoppingCriteria):
    """Stop as soon as every sequence in the batch ends with eo_id."""
    def __init__(self, eo_id: int):
        self.eo_id = eo_id
    def __call__(self, input_ids, scores, **kw):
        return bool((input_ids[:, -1] == self.eo_id).all())


def build_model(cfg_name: str):
    """Compose Hydra config and instantiate the trained model."""
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
    print(f"Building model on {device}...")
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
    return cfg, model, device


def build_prompt(rgb_tokens: np.ndarray) -> torch.Tensor:
    """<BO_rgb> rgb_body <EO_rgb> <BO_scene_desc>  (1, L)."""
    rgb_info = MODALITIES["tok_rgb@256"]
    sd_info = MODALITIES["scene_desc"]
    body = torch.from_numpy(rgb_tokens.flatten()).long() + rgb_info.codebook_offset
    prompt = torch.cat([
        torch.tensor([rgb_info.bo_id], dtype=torch.long),
        body,
        torch.tensor([rgb_info.eo_id], dtype=torch.long),
        torch.tensor([sd_info.bo_id], dtype=torch.long),
    ])
    return prompt.unsqueeze(0)


def decode_scene_desc(new_tokens: torch.Tensor, gpt2_tok) -> tuple[str, bool, int]:
    """Strip specials, subtract codebook_offset, decode through GPT-2.

    Returns (text, stopped_on_eo, n_body_tokens).
    """
    sd_info = MODALITIES["scene_desc"]
    flat = new_tokens.cpu()
    stopped_on_eo = bool((flat == sd_info.eo_id).any())
    # Body = everything that isn't BO/EO/PAD. <BO_scene_desc> isn't in `new_tokens`
    # since it was part of the prompt; <EO_scene_desc> may or may not be present.
    mask = (flat != sd_info.bo_id) & (flat != sd_info.eo_id) & (flat != PAD_ID)
    body = flat[mask]
    n_body = int(body.numel())
    if n_body == 0:
        return "", stopped_on_eo, 0
    ids = (body - sd_info.codebook_offset).tolist()
    # Defensive: any negative or out-of-range id is a sign the model emitted a
    # non-text token (image codebook, BO/EO of another modality, etc.). Drop them.
    ids = [i for i in ids if 0 <= i < sd_info.codebook_size]
    text = gpt2_tok.decode(ids, skip_special_tokens=True)
    return text, stopped_on_eo, n_body


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    rgb_dir = data_dir / "tok_rgb@256"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Expected RGB token dir at {rgb_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"

    cfg, model, device = build_model(args.config)
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    sd_info = MODALITIES["scene_desc"]
    stopping = StoppingCriteriaList([_EoStop(sd_info.eo_id)])
    max_new_tokens = int(cfg.infer.max_new_tokens)
    use_cache = bool(cfg.infer.use_cache)
    do_sample = bool(cfg.infer.do_sample) or cfg.infer.temperature != 1.0 \
                or cfg.infer.top_k != 0 or cfg.infer.top_p != 1.0

    ids = [f"{i:05d}" for i in range(args.start, args.start + args.n_samples)]
    print(f"Generating {len(ids)} samples → {out_path}")
    print(f"  greedy={not do_sample}  use_cache={use_cache}  max_new_tokens={max_new_tokens}")

    t_start = time.time()
    n_done = 0
    n_eo = 0
    with open(out_path, "w") as f_out:
        for sid in ids:
            rgb_path = rgb_dir / f"{sid}.npy"
            if not rgb_path.is_file():
                print(f"  [skip] {sid}: no rgb tokens at {rgb_path}")
                continue
            arr = np.load(rgb_path)
            if arr.ndim == 2:
                if args.aug_idx >= arr.shape[0]:
                    print(f"  [skip] {sid}: aug_idx={args.aug_idx} out of range "
                          f"(have {arr.shape[0]})")
                    continue
                rgb_tokens = arr[args.aug_idx]
            else:
                rgb_tokens = arr  # single-aug layout

            prompt = build_prompt(rgb_tokens).to(device)
            prompt_len = prompt.shape[1]

            gen_kwargs = dict(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                stopping_criteria=stopping,
                do_sample=do_sample,
                pad_token_id=PAD_ID,
            )
            # Sampling kwargs are only valid when do_sample=True; transformers
            # 4.52 warns "not a valid generation flag" otherwise.
            if do_sample:
                gen_kwargs["temperature"] = cfg.infer.temperature
                gen_kwargs["top_k"] = cfg.infer.top_k
                gen_kwargs["top_p"] = cfg.infer.top_p

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**gen_kwargs)
            dt = time.time() - t0
            new_tokens = out[0, prompt_len:].cpu()
            text, stopped_on_eo, n_body = decode_scene_desc(new_tokens, gpt2_tok)

            record = {
                "id": sid,
                "aug_idx": args.aug_idx,
                "prompt_len": int(prompt_len),
                "n_new_tokens": int(new_tokens.numel()),
                "n_body_tokens": int(n_body),
                "stopped_on_eo": bool(stopped_on_eo),
                "gen_seconds": round(dt, 3),
                "generated_text": text,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            n_done += 1
            n_eo += int(stopped_on_eo)
            if n_done % 25 == 0:
                rate = n_done / (time.time() - t_start)
                eta = (len(ids) - n_done) / max(rate, 1e-6)
                print(f"  [{n_done}/{len(ids)}] {sid}  "
                      f"{n_body}tok  {dt:.2f}s  "
                      f"eo_rate={n_eo/n_done:.2%}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min")

    total = time.time() - t_start
    print(f"Done. {n_done} samples in {total/60:.1f} min  "
          f"({n_done/max(total,1e-6):.2f}/s, eo_rate={n_eo/max(n_done,1):.2%})")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()