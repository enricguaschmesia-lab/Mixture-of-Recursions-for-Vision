"""A2's generated-map evaluation: generate map tokens, decode them, score GEO-Bench-2's mIoU (A2 Step 3.3).

TWO HALVES IN TWO ENVIRONMENTS, joined by a file:
  generate (`.venv`)  the model writes each view's map, KV-cached and constrained to the arm's slot, from the
                      prompt eo/data/seg_dataset.py builds (inputs, then BO_MAP_<task>, lattice positions).
                      Greedy, or K samples at temperature T. Writes codes (N, K, G) uint16 + a record.
  score (`a2seg`)     the frozen map tokenizer (eo/maps/frozen.py, hashes checked) decodes each code grid; the
                      harness turns the output into class scores on the native grid (eo/maps/harness.
                      class_scores: CloudSen12's 3 x 3 crops stitched, scores averaged where they overlap),
                      the arm's rule (nearest) picks a class per pixel, and K samples vote per pixel (BEN's
                      K-sample majority; ties to the lower class); eo/maps/scoring.Confusion reads
                      GEO-Bench-2's macro Jaccard (PASTIS void ignored, CloudSen12 nothing ignored).
  Decoding is a tokenizer's job, so it never runs in `.venv` (CLAUDE.md); generation needs transformers 4.52.4,
  so it never runs in `a2seg`.

CONTROLS (plan Step 3.3), each a dataset variant scored the same way:
  prior           no input: the class prior the model writes from BO_MAP_<task> alone
  wrong           another sample's input under a fixed derangement: should collapse toward the prior (Z9)
  ceiling         the stored TRUE map tokens through the same decode and scorer: Step 1's ceiling, the bound
  shuffled_dates, one_date   PASTIS-7's time controls

⚠ Never a test split: the dataset refuses it, and so does the scorer (eo/maps/sets.load).
⚠ No vocabulary-sized topk or sort (CLAUDE.md, the TITAN V): greedy is argmax over the slot, sampling is
multinomial over the slot. Run generation on the GTX Titan X.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

CACHE_ULPS = 4             # eo/eval/ben.py's tolerance: cached = uncached within 4 rounding steps (set at fp16/bf16)
FLOOR_FACTOR = 2.0         # ... or within twice the measured GEMM-shape round-off, whichever is larger (fp32)


# ====================================================================================== generation (.venv)
@torch.no_grad()
def generate(model, prompts: torch.Tensor, positions: torch.Tensor, slot: Tuple[int, int], n_tokens: int,
             greedy: bool = True, temperature: float = 1.0, seed: int = 0,
             autocast: Optional[torch.dtype] = None, cache_offset: int = 0,
             step_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, L) prompts ending in BO_MAP_<task>, their (B, L) positions -> (B, n_tokens) raw codes in the slot,
    and the mean unconstrained softmax mass the model put on the slot.

    Constrained: each step reads logits[off : off + size] only, so every emitted id is the arm's (Z8). Step t's
    token sits at the prompt's last position + 1 + t (eo/generate/cached.py: positions must be passed under the
    lattice), or at step_positions[t] when given (the dataset's `step_positions`: a shifted or nested map layout,
    seg_dataset MAP LAYOUTS). `cache_offset` exists for gate Z8's control only: it shifts every cached step's position."""
    off, size = slot
    ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else _null()
    g = torch.Generator(device=prompts.device).manual_seed(int(seed))
    toks, mass = [], torch.zeros(len(prompts), device=prompts.device)
    with ctx:
        out = model(input_ids=prompts, position_ids=positions, use_cache=True, logits_to_keep=1)
        cache, logits = out.past_key_values, out.logits[:, -1, :].float()
        for t in range(n_tokens):
            mass += torch.softmax(logits, -1)[:, off:off + size].sum(-1)
            sl = logits[:, off:off + size]
            if greedy:
                x = sl.argmax(-1)
            else:
                x = torch.multinomial(torch.softmax(sl / float(temperature), -1), 1, generator=g).squeeze(1)
            toks.append(x)
            if t == n_tokens - 1:
                break
            pos = (positions[:, -1:] + 1 + t if step_positions is None
                   else step_positions[t].to(positions.device).view(1, 1).expand(len(prompts), 1)) + int(cache_offset)
            o = model(input_ids=(x + off).unsqueeze(1), past_key_values=cache, position_ids=pos, use_cache=True)
            cache, logits = o.past_key_values, o.logits[:, -1, :].float()
    return torch.stack(toks, 1), mass / n_tokens


@torch.no_grad()
def cache_check(model, prompts: torch.Tensor, positions: torch.Tensor, slot: Tuple[int, int], n_steps: int,
                autocast: Optional[torch.dtype] = None, cache_offset: int = 0,
                step_positions: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Max |logit difference| over the slot between a cached decode and full recomputation, after n_steps
    greedy tokens (fed back both ways); that in rounding steps of the compute dtype (eo/eval/ben.py); and the
    tolerance max(CACHE_ULPS steps, FLOOR_FACTOR x the round-off floor measured on the same prompts). The
    4-step rule alone was set at fp16 and bf16, where a step is 0.03-0.25 at these logits; at fp32 a step is
    ~4e-6, below what a 29-layer pass picks up from a change of GEMM shape."""
    from eo.eval.ben import rounding_step
    off, size = slot
    codes, _ = generate(model, prompts, positions, slot, n_steps, greedy=True, autocast=autocast,
                        cache_offset=cache_offset, step_positions=step_positions)
    ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else _null()
    diff, mag = [], []
    with ctx:
        out = model(input_ids=prompts, position_ids=positions, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        seq, pos = prompts, positions
        for t in range(n_steps):
            ids = (codes[:, t] + off).unsqueeze(1)
            p = (positions[:, -1:] + 1 + t if step_positions is None
                 else step_positions[t].to(positions.device).view(1, 1).expand(len(prompts), 1))
            o = model(input_ids=ids, past_key_values=cache, position_ids=p + int(cache_offset), use_cache=True)
            cache = o.past_key_values
            seq, pos = torch.cat([seq, ids], 1), torch.cat([pos, p], 1)
            full = model(input_ids=seq, position_ids=pos, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            a, b = o.logits[:, -1, off:off + size].float(), full[:, off:off + size].float()
            diff.append((a - b).abs().max())
            mag.append(torch.maximum(a.abs(), b.abs()).max())
        # The round-off floor: the prompt's last logits recomputed with one trailing pad token. Causal attention
        # leaves them unchanged mathematically; only the GEMM shapes change (A2 Step 3: at fp32 the cached decode
        # differs from recomputation by this much, 8-30 fp32 rounding steps, on the Titan X and on CPU).
        a0 = model(input_ids=prompts, position_ids=positions, logits_to_keep=1).logits[:, -1, off:off + size].float()
        pad = torch.full_like(prompts[:, :1], int(getattr(model.config, "pad_token_id", None) or 0))
        b0 = model(input_ids=torch.cat([prompts, pad], 1),
                   position_ids=torch.cat([positions, positions[:, -1:] + 1], 1)).logits[:, -2, off:off + size].float()
        floor = float((a0 - b0).abs().max())
    d, m = torch.stack(diff).max(), torch.stack(mag).max()
    step = rounding_step(m, autocast or torch.float32)
    tol = max(float(CACHE_ULPS * step), FLOOR_FACTOR * floor)
    return {"max_abs_logit_diff": float(d), "max_abs_logit": float(m), "max_ulp_diff": float(d / step),
            "floor": floor, "tol": tol, "within": bool(float(d) <= tol), "steps": n_steps,
            "batch": int(prompts.shape[0]), "codes_in_slot": bool(((codes >= 0) & (codes < size)).all())}


def generate_views(model, ds, views: Sequence[int], variant: str, k: int, greedy: bool, temperature: float,
                   seed: int, batch: int, device, autocast=None, log_every: int = 0) -> Dict[str, np.ndarray]:
    """Every view's map: (N, K, G) codes and (N, K) slot mass. Prompts of one dataset have one length."""
    from eo.data import map_vocab as MV
    slot = ds.lay.slot(ds.spec.arm)
    n_tok = MV.spec(ds.spec.arm)[1]
    sp = ds.step_positions(variant).to(device)          # the layout's code positions (sequential by default)
    codes = np.zeros((len(views), k, n_tok), np.uint16)
    mass = np.zeros((len(views), k), np.float32)
    t0 = time.time()
    for s in range(0, len(views), batch):
        vs = list(views[s:s + batch])
        pr = [ds.prompt(v, variant) for v in vs]
        ids = torch.stack([p[0] for p in pr]).to(device)
        pos = torch.stack([p[1] for p in pr]).to(device)
        for j in range(k):
            c, m = generate(model, ids, pos, slot, n_tok, greedy=greedy, temperature=temperature,
                            seed=seed * 1_000_003 + j * 7919 + s, autocast=autocast, step_positions=sp)
            codes[s:s + len(vs), j] = ds.to_raster(c.cpu().numpy()).astype(np.uint16)   # stored in raster order
            mass[s:s + len(vs), j] = m.cpu().numpy()
        if log_every and (s // batch) % log_every == 0:
            print(f"  {s + len(vs)}/{len(views)} views ({time.time() - t0:.0f} s)", flush=True)
    return {"codes": codes, "slot_mass": mass}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ====================================================================================== scoring (a2seg)
def gt_for_views(task: str, views) -> Tuple[np.ndarray, np.ndarray]:
    """(labels (S, H, W) of the views' samples in first-seen order, sample index of each view). Refuses test."""
    from eo.maps import sets
    split = set(views.split)
    if split != {"val"}:
        raise PermissionError(f"the scorer reads the val split only (Steps 3-5); got {sorted(split)}")
    labels, ids = sets.load(task, "val")
    where = {(str(i) if task == "cloudsen12" else int(i)): n for n, i in enumerate(ids)}
    samples = list(dict.fromkeys(views.sample_id.tolist()))
    missing = [s for s in samples if (str(s) if task == "cloudsen12" else int(s)) not in where]
    if missing:
        raise RuntimeError(f"{len(missing)} samples are not in eo/maps' {task} val set, e.g. {missing[:3]}")
    take = [where[str(s) if task == "cloudsen12" else int(s)] for s in samples]
    s_of = {s: n for n, s in enumerate(samples)}
    return np.asarray(labels[take]), np.asarray([s_of[s] for s in views.sample_id.tolist()])


def crop_order_problems(task: str, views) -> list:
    """CloudSen12: each tile's views are its 9 crops in eo/maps/tiling.plan's order (the order stitch expects)."""
    from eo.maps import palette as P, tiling
    if task != "cloudsen12":
        return [] if (views.groupby("sample_id").size() == 1).all() else ["a PASTIS sample has more than one view"]
    plan = [(y, x) for (y, x, _) in tiling.plan("cloudsen12", P.TASKS["cloudsen12"].native, "crop224")]
    out = []
    for sid, g in views.groupby("sample_id", sort=False):
        got = list(zip(g.y0.tolist(), g.x0.tolist()))
        if got != plan:
            out.append(f"{sid}: crops {got[:3]}... not the plan")
    return out


def vote(preds: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(K, N, H, W) class maps -> (N, H, W): the per-pixel majority, ties to the lower class."""
    if preds.shape[0] == 1:
        return preds[0]
    oh = torch.nn.functional.one_hot(preds.long(), n_classes).sum(0)          # (N, H, W, C)
    return oh.argmax(-1)


def score(task: str, arm: str, codes: np.ndarray, views, device: str = "cuda", maps_per_chunk: Optional[int] = None,
          tok=None, return_pred: bool = False) -> Dict:
    """codes (V, K, G) for these val views (rows aligned) -> GEO-Bench-2's mIoU at native resolution, with K
    samples voted per pixel, plus the predicted class shares (the prior control reads them)."""
    from eo.data import map_vocab as MV
    from eo.maps import decode as D, frozen, harness as H, palette as P, scoring as S
    problems = crop_order_problems(task, views)
    if problems:
        raise RuntimeError("; ".join(problems[:3]))
    t = P.TASKS[task]
    side = MV.spec(arm)[2]
    if codes.ndim == 2:
        codes = codes[:, None]
    V, K, G = codes.shape
    if G != side * side or V != len(views):
        raise ValueError(f"codes {codes.shape} for {len(views)} views of arm {arm} ({side} x {side})")
    if tok is None:
        tok = frozen.load(arm, device)
    rule = frozen.record(arm)["rule"]
    labels, s_idx = gt_for_views(task, views)
    n_crops = len(views) // len(labels)
    if n_crops * len(labels) != len(views):
        raise RuntimeError("views are not a whole number of crops a sample")
    conf = S.Confusion(t.n_classes, t.ignore, device)
    hist = torch.zeros(t.n_classes, dtype=torch.float64)
    per = maps_per_chunk or {"pastis": 64, "cloudsen12": 4}[task]
    preds_out = []
    with torch.no_grad():
        for s in range(0, len(labels), per):
            e = min(s + per, len(labels))
            gt = torch.as_tensor(np.ascontiguousarray(labels[s:e]), device=device).long()
            ks = []
            for j in range(K):
                ib = torch.as_tensor(codes[s * n_crops:e * n_crops, j].astype(np.int64), device=device)
                sc = H.class_scores(tok, ib.view(-1, side, side), task, "universal", "crop224")
                ks.append(D.apply_rule(sc, t.ignore, rule, tok.output))
            pr = vote(torch.stack(ks), t.n_classes)
            conf.update(gt, pr)
            hist += torch.bincount(pr.flatten().cpu(), minlength=t.n_classes).double()
            if return_pred:
                preds_out.append(pr.to(torch.uint8).cpu().numpy())
    res = conf.result()
    share = (hist / hist.sum()).numpy()
    scored = [c for c in range(t.n_classes) if c != t.ignore]
    p = share[scored] / max(share[scored].sum(), 1e-12)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    out = {"task": task, "arm": arm, "views": int(V), "samples": int(len(labels)), "K": int(K), "rule": rule,
           "result": res, "pred_share": share.tolist(), "pred_entropy_nats": ent,
           "pred_entropy_max_nats": float(np.log(len(scored))), "pred_top_share": float(p.max())}
    if return_pred:
        out["pred"] = np.concatenate(preds_out)
    return out


def write_codes(path: Path, codes: Dict[str, np.ndarray], record: Dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "codes.npy", codes["codes"])
    np.save(path / "slot_mass.npy", codes["slot_mass"])
    (path / "record.json").write_text(json.dumps(record, indent=1, default=str) + "\n")
