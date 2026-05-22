"""Reproduce results/milestone2/metrics/summary_pivot.{md,csv,tex}.

Evaluates four checkpoints on 500 CLEVR test scenes (aug_idx=0) over three
directions:
  - RGB within-modality (teacher-forced perplexity)
  - RGB → caption          (CLIPScore + BLEU-4)
  - Caption → RGB          (Pixel-MSE + SSIM)

Per-sample results are written to results/milestone2/metrics/per_sample/ and
re-used on subsequent runs (safe to interrupt and resume).

Run:
    uv run python scripts/reproduce_metrics.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

# torch.distributed.tensor shim for transformers internals
try:
    from torch.distributed.tensor import DTensor  # noqa: F401
except ImportError:
    class DTensor:  # type: ignore
        pass
    import transformers.modeling_utils
    transformers.modeling_utils.DTensor = DTensor

from util.env import load_dotenv
load_dotenv()
from paths import HF_CACHE_DIR
os.environ.setdefault('HF_HOME', HF_CACHE_DIR)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from omegaconf import open_dict
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES, PAD_ID
from model.sharing_strategy import SHARING_STRATEGY
from model.util import load_checkpoint, load_model_from_config
from scripts._routing_lib import Modality, load_sample
from util.config import preprocess_config
from visualization.decode import _resolve_cosmos_decoder_jit

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
PATCH_GRID   = 16
N_SAMPLES    = 500
COSMOS_PATH  = 'nvidia/Cosmos-0.1-Tokenizer-DI16x16'

DATA_TEST       = PROJECT_DIR / 'data' / 'clevr_dataset' / 'test'
METRICS_DIR     = PROJECT_DIR / 'results_' / 'milestone2' / 'metrics'
PER_SAMPLE_DIR  = METRICS_DIR / 'per_sample'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = [
    {'label': 'baseline_vanilla',   'config': 'infer/baseline_multimodal_vanilla',       'display': 'Vanilla'},
    {'label': 'random_router_5000', 'config': 'infer/random_router_5000_multimodality', 'display': 'Random Nr=3'},
    {'label': 'mor_5000_3r',        'config': 'infer/mor_5000_multimodality_3r',        'display': 'MoR Nr=3'},
    {'label': 'mor_5000_4r',        'config': 'infer/mor_5000_multimodality_4r',        'display': 'MoR Nr=4'},
]

DIRECTION_METRICS = {
    'tok_rgb@256_within': [('tf_perplexity', 'PPL ↓',        True)],
    'rgb_caption':        [('clip_score',    'CLIPScore ↑',  False),
                           ('bleu4',         'BLEU-4 ↑',     False)],
    'caption_rgb':        [('pixel_mse',     'Pixel-MSE ↓',  True),
                           ('ssim',          'SSIM ↑',       False)],
}
DIRECTION_LABELS = {
    'tok_rgb@256_within': 'RGB within (25%→75%)',
    'rgb_caption':        'RGB → Caption',
    'caption_rgb':        'Caption → RGB',
}
ORDERED_DIRS = list(DIRECTION_METRICS.keys())

# ──────────────────────────────────────────────────────────────────────────────
# Model + Cosmos + CLIP
# ──────────────────────────────────────────────────────────────────────────────
def load_model(cfg_name: str):
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()
    with initialize_config_dir(config_dir=str(PROJECT_DIR / 'conf'), version_base=None):
        cfg = compose(config_name=cfg_name)
    with open_dict(cfg):
        cfg.wandb = False
        cfg.wandb_entity = ''
        cfg.wandb_project = 'reproduce'
        cfg.wandb_run_name = 'reproduce'
        cfg.resume_from_checkpoint = False
    cfg = preprocess_config(cfg)
    m = load_model_from_config(cfg)
    if cfg.recursive.get('enable'):
        m, _ = SHARING_STRATEGY[cfg.model](cfg, m)
    if cfg.get('mor') and cfg.mor.get('enable'):
        if cfg.mor.type == 'token':
            m.transform_layer_to_mor_token(cfg)
        else:
            m.transform_layer_to_mor_expert(cfg)
    m = load_checkpoint(m, cfg.infer.checkpoint)
    return m.to(DEVICE).eval()


_cosmos = None
def get_cosmos():
    global _cosmos
    if _cosmos is None:
        from cosmos_tokenizer.image_lib import ImageTokenizer
        jit = _resolve_cosmos_decoder_jit(COSMOS_PATH)
        _cosmos = ImageTokenizer(checkpoint_dec=jit, device='cuda', dtype='bfloat16')
    return _cosmos


def decode_image(toks: np.ndarray) -> np.ndarray | None:
    need = PATCH_GRID * PATCH_GRID
    if toks.size < need:
        toks = np.pad(toks, (0, need - toks.size))
    try:
        idx = torch.from_numpy(toks[:need].astype(np.int64)).reshape(1, PATCH_GRID, PATCH_GRID).to('cuda')
        with torch.no_grad():
            dec = get_cosmos().decode(idx).float().cpu().numpy()[0]
        dec = np.clip((dec + 1.0) / 2.0, 0.0, 1.0)
        return (dec.transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    except Exception:
        return None


_clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cpu').eval()
_clip_proc  = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')


def clip_score(img: np.ndarray | None, text: str) -> float | None:
    if img is None or not text:
        return None
    inputs = _clip_proc(text=[text], images=Image.fromarray(img),
                        return_tensors='pt', padding=True).to('cpu')
    with torch.no_grad():
        return float(_clip_model(**inputs).logits_per_image[0, 0])


def bleu4(hyp: str, ref: str) -> float:
    return float(sentence_bleu([ref.split()], hyp.split(),
                               smoothing_function=SmoothingFunction().method1))


def pixel_mse(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean())


def compute_ssim(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(ssim_fn(a, b, channel_axis=-1))


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────
mod = Modality(AutoTokenizer.from_pretrained('gpt2'), text_max_len=64)
_gpt2_tok = AutoTokenizer.from_pretrained('gpt2')


class _EoStop(StoppingCriteria):
    def __init__(self, eo_id): self.eo_id = eo_id
    def __call__(self, input_ids, _scores, **_kw):
        return bool((input_ids[:, -1] == self.eo_id).all())


def generate(model, tgt_mod: str, ctx: torch.Tensor, *,
             do_sample: bool = False, seed: int | None = None
             ) -> tuple[torch.Tensor, bool]:
    info = MODALITIES[tgt_mod]
    prompt = torch.cat([ctx, torch.tensor([info.bo_id])]).unsqueeze(0).to(DEVICE)
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=257 if info.data_type == 'tokens' else 128,
            use_cache=False,
            stopping_criteria=StoppingCriteriaList([_EoStop(info.eo_id)]),
            do_sample=do_sample,
            temperature=0.7 if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=PAD_ID,
        )
    new_ids = out[0, prompt.shape[1] - 1:].cpu()
    other_bos = {v.bo_id for k, v in MODALITIES.items() if k != tgt_mod}
    failure = (info.eo_id not in new_ids.tolist()) or any(t.item() in other_bos for t in new_ids[1:])
    return new_ids, failure


def gen_to_tokens(gen_ids: torch.Tensor, tgt_mod: str) -> np.ndarray:
    info = MODALITIES[tgt_mod]
    body = gen_ids[(gen_ids != info.bo_id) & (gen_ids != info.eo_id) & (gen_ids != PAD_ID)]
    return (body - info.codebook_offset).numpy()


def gen_to_text(gen_ids: torch.Tensor, tgt_mod: str) -> str:
    info = MODALITIES[tgt_mod]
    body = gen_ids[(gen_ids != info.bo_id) & (gen_ids != info.eo_id) & (gen_ids != PAD_ID)]
    return _gpt2_tok.decode((body - info.codebook_offset).tolist(), skip_special_tokens=True)


def tf_perplexity(model, seq: torch.Tensor, suffix_start: int) -> float:
    with torch.no_grad():
        logits = model(input_ids=seq.to(DEVICE), use_cache=False).logits[:, :-1, :].float()
    labels = seq[:, 1:].to(DEVICE)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                         reduction='none').reshape(1, -1).squeeze(0)
    suffix_ce = ce[suffix_start - 1:]
    return float('nan') if suffix_ce.numel() == 0 else float(torch.exp(suffix_ce.mean()))


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample evaluation
# ──────────────────────────────────────────────────────────────────────────────
def eval_sample(model, label: str, sid: str) -> list[dict]:
    s = load_sample(DATA_TEST, sid, 0)
    rgb_arr = s.get('rgb')
    cap     = s.get('caption')
    if rgb_arr is None:
        return []
    rows: list[dict] = []
    base = {'checkpoint': label, 'sid': sid, 'aug_idx': 0}

    # 1. RGB within-modality TF-perplexity
    chunk   = mod.chunk_tokens('tok_rgb@256', rgb_arr, aug_idx=0, close=True)
    segs    = mod.segment(chunk)
    body_st = next((a for a, _, m in segs if m == 'tok_rgb@256'), None)
    ppl = float('nan')
    if body_st is not None:
        try:
            ppl = tf_perplexity(model, chunk.unsqueeze(0), body_st + chunk.shape[0] // 4)
        except Exception:
            pass
    rows.append({**base, 'direction': 'tok_rgb@256_within',
                 'tf_perplexity': None if np.isnan(ppl) else ppl,
                 'generation_failure': False})

    # 2. RGB → caption (sampled)
    if cap is not None:
        ctx = mod.chunk_tokens('tok_rgb@256', rgb_arr, aug_idx=0, close=True)
        try:
            gen_ids, fail = generate(model, 'caption', ctx, do_sample=True, seed=int(sid))
            text = gen_to_text(gen_ids, 'caption') if not fail else ''
            cs   = clip_score(decode_image(rgb_arr), text) if not fail else None
            bl   = bleu4(text, cap) if not fail else None
        except Exception:
            fail, cs, bl = True, None, None
        rows.append({**base, 'direction': 'rgb_caption',
                     'clip_score': cs, 'bleu4': bl, 'generation_failure': fail})

    # 3. Caption → RGB (greedy)
    if cap is not None:
        ctx = mod.chunk_text('caption', cap)
        try:
            gen_ids, fail = generate(model, 'tok_rgb@256', ctx, do_sample=False)
            gen_img = decode_image(gen_to_tokens(gen_ids, 'tok_rgb@256')) if not fail else None
            gt_img  = decode_image(rgb_arr)
            mse     = pixel_mse(gen_img, gt_img)
            sim     = compute_ssim(gen_img, gt_img)
        except Exception:
            fail, mse, sim = True, None, None
        rows.append({**base, 'direction': 'caption_rgb',
                     'pixel_mse': mse, 'ssim': sim, 'generation_failure': fail})
    return rows


def write_atomic(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows))
    tmp.rename(path)


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
def main():
    sids = [p.stem for p in sorted((DATA_TEST / 'tok_rgb@256').glob('*.npy'))[:N_SAMPLES]]
    all_rows: list[dict] = []

    for ckpt in CHECKPOINTS:
        label = ckpt['label']
        print(f'\n=== {ckpt["display"]} ===')
        todo: list[str] = []
        for sid in sids:
            p = PER_SAMPLE_DIR / f'{label}_{sid}_a0.json'
            if p.exists():
                try:
                    all_rows.extend(json.loads(p.read_text()))
                    continue
                except Exception:
                    pass
            todo.append(sid)
        print(f'  {len(sids) - len(todo)} cached, {len(todo)} to compute')
        if not todo:
            continue
        model = load_model(ckpt['config'])
        for sid in tqdm(todo, desc=f'  {label}', unit='sample'):
            rows = eval_sample(model, label, sid)
            if rows:
                write_atomic(rows, PER_SAMPLE_DIR / f'{label}_{sid}_a0.json')
            all_rows.extend(rows)
        del model
        torch.cuda.empty_cache()

    write_pivot(all_rows)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation → summary_pivot.{md,csv,tex}
# ──────────────────────────────────────────────────────────────────────────────
def write_pivot(rows: list[dict]) -> None:
    agg  = defaultdict(lambda: defaultdict(list))
    fail = defaultdict(lambda: {'total': 0, 'failed': 0})
    for r in rows:
        k = (r['checkpoint'], r['direction'])
        fail[k]['total'] += 1
        if r.get('generation_failure'):
            fail[k]['failed'] += 1
            continue
        for f, *_ in sum(DIRECTION_METRICS.values(), []):
            v = r.get(f)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[k][f].append(v)

    models = [c['label'] for c in CHECKPOINTS]
    labels = {c['label']: c['display'] for c in CHECKPOINTS}

    # pivot[direction][model] = [(metric_label, "mean ± std", is_best), ...]
    pivot: dict[str, dict[str, list[tuple[str, str, bool]]]] = {}
    for d in ORDERED_DIRS:
        pivot[d] = {m: [] for m in models}
        for field, mlabel, lower in DIRECTION_METRICS[d]:
            stats, means = {}, {}
            for m in models:
                vals = np.array(agg[(m, d)].get(field, []), dtype=float)
                if vals.size == 0:
                    stats[m] = (None, None)
                else:
                    stats[m] = (float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
                    means[m] = stats[m][0]
            best = (min if lower else max)(means, key=means.get) if means else None
            for m in models:
                mean, std = stats[m]
                txt = '—' if mean is None else f'{mean:.3f} ± {std:.3f}'
                pivot[d][m].append((mlabel, txt, m == best))

    # Markdown
    md = ['| Direction | Metric | ' + ' | '.join(labels[m] for m in models) + ' |',
          '|---|---|' + '---|' * len(models)]
    for d in ORDERED_DIRS:
        for i, (mlabel, _, _) in enumerate(pivot[d][models[0]]):
            cells = []
            for m in models:
                _, txt, best = pivot[d][m][i]
                cells.append(f'**{txt}**' if best and txt != '—' else txt)
            dcell = DIRECTION_LABELS[d] if i == 0 else ''
            md.append(f'| {dcell} | {mlabel} | ' + ' | '.join(cells) + ' |')
    (METRICS_DIR / 'summary_pivot.md').write_text('\n'.join(md) + '\n')
    print(f'\nWrote {METRICS_DIR / "summary_pivot.md"}')

    # LaTeX
    tex = [f'% Pivot tables — N≤{N_SAMPLES}; mean $\\pm$ std; best per row in \\textbf{{bold}}\n']
    for d in ORDERED_DIRS:
        n = len(pivot[d][models[0]])
        tex.append('\\begin{table}[ht]\\centering')
        tex.append(f'\\caption{{{DIRECTION_LABELS[d]} — mean $\\pm$ std}}')
        tex.append('\\begin{tabular}{l' + 'c' * (len(models) * n) + '}')
        tex.append('\\toprule')
        tex.append(' & ' + ' & '.join(f'\\multicolumn{{{n}}}{{c}}{{{labels[m]}}}' for m in models) + ' \\\\')
        sub = ['']
        for _ in models:
            for mlabel, _, _ in pivot[d][models[0]]:
                sub.append(mlabel.replace('↓', '$\\downarrow$').replace('↑', '$\\uparrow$'))
        tex.append(' & '.join(sub) + ' \\\\')
        tex.append('\\midrule')
        row = [DIRECTION_LABELS[d]]
        for m in models:
            for _, txt, best in pivot[d][m]:
                s = txt.replace('±', '$\\pm$')
                row.append(f'\\textbf{{{s}}}' if best and txt != '—' else s)
        tex.append(' & '.join(row) + ' \\\\')
        tex.append('\\bottomrule\n\\end{tabular}\\end{table}\n')
    (METRICS_DIR / 'summary_pivot.tex').write_text('\n'.join(tex))
    print(f'Wrote {METRICS_DIR / "summary_pivot.tex"}')

    # CSV (long format)
    with open(METRICS_DIR / 'summary_pivot.csv', 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['direction', 'metric', 'model', 'mean', 'std',
                    'is_best', 'n_samples', 'failure_rate'])
        for d in ORDERED_DIRS:
            for i, (field, _, _) in enumerate(DIRECTION_METRICS[d]):
                for m in models:
                    vals = np.array(agg[(m, d)].get(field, []), dtype=float)
                    fc   = fail[(m, d)]
                    fr   = fc['failed'] / max(fc['total'], 1)
                    best = pivot[d][m][i][2]
                    if vals.size == 0:
                        w.writerow([d, field, m, '', '', int(best), fc['total'], f'{fr:.4f}'])
                    else:
                        std = vals.std(ddof=1) if vals.size > 1 else 0.0
                        w.writerow([d, field, m, f'{vals.mean():.4f}', f'{std:.4f}',
                                    int(best), fc['total'], f'{fr:.4f}'])
    print(f'Wrote {METRICS_DIR / "summary_pivot.csv"}')


if __name__ == '__main__':
    main()
