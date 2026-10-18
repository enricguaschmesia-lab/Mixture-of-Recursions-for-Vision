"""BigEarthNet label generation and its metrics (Phase 6 Step 5, D6.7). `.venv`.

GENERATION. The prompt is the patch's inputs in a fixed order, then BO_BEN; the model writes
`c_1 ... c_k EO_BEN`. Decoding is CONSTRAINED to the format the block has: at each step only
classes above the last one emitted, or EO_BEN, are allowed (Phase 3's slot-masked decoding,
plus the canonical order). The unconstrained mass on that allowed set is recorded at the first
step (`slot_mass`), so the constraint is never doing all the work unseen.

SCORES FOR mAP (PHASE6_PLAN §4.4 option (a), worklog §4 item 29). A set gives F1 exactly but
no per-class score. The per-class MARGINAL P(c in set | inputs) is computed from K sampled
label blocks with a Rao-Blackwellized estimator instead of a 0/1 count:

    P(c in S) = sum_t E[ p_t(c | h_t) ]      (p_t = 0 once a class >= c or EO_BEN was emitted)

because c can be emitted at step t only while every earlier token is below it, and given the
history the chance it is emitted then is exactly the model's p_t(c | h_t). Each sampled path
therefore contributes sum_t p_t(c | h_t) over the steps where c is still reachable: unbiased,
and continuous even at K = 1, so ties (which make AP depend on tie-breaking) are rare.

KV CACHE. Vanilla only (B10's class is HF Llama with DynamicCache). `cache_check` compares
cached against uncached next-token logits on real prompts before any number is trusted; the
recursive classes still raise (Phase 3 S3.c is not done).

METRICS. Average precision is implemented to sklearn's definition (`average_precision_score`:
step-wise, ties grouped at distinct thresholds) because `.venv` has no sklearn; gate X7
cross-checks it against sklearn in the `mor` env. SMARTIES reports MICRO mAP only.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from eo.data import ben_vocab as V

#: max |cached - uncached| next-token logit difference accepted per autocast mode (eval_ben, ben_chain)
CACHE_TOL = {"none": 1e-3, "fp16": 5e-2, "bf16": 2e-1}


def derangement(n: int, seed: int) -> np.ndarray:
    """A fixed permutation with no fixed point: patch i gets the inputs of patch d[i] != i."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    fixed = np.flatnonzero(p == np.arange(n))
    for i in fixed:                      # swap each fixed point with its neighbour
        j = (i + 1) % n
        p[i], p[j] = p[j], p[i]
    assert not (p == np.arange(n)).any()
    return p


ALLOWED = V.CLASS_IDS + [V.EO_BEN]            # the 20 ids the block can contain after BO_BEN
_EO_POS = V.N_CLASSES                        # EO_BEN's column in the 20-way restricted logits


# ------------------------------------------------------------------ generation

def _restricted(logits: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
    """(B, V) full logits -> (B, 20) log-probs over [classes..., EO_BEN], masked to c > last."""
    sub = logits[:, ALLOWED].float()
    cls = torch.arange(V.N_CLASSES, device=logits.device)
    bad = cls.unsqueeze(0) <= last.unsqueeze(1)                       # (B, 19)
    bad = torch.cat([bad, torch.zeros_like(bad[:, :1])], 1)           # EO_BEN always allowed
    return torch.log_softmax(sub.masked_fill(bad, float("-inf")), -1)


def _expand_cache(cache, k: int):
    from transformers import DynamicCache
    if k == 1:
        return cache
    new = DynamicCache()
    for layer, (kk, vv) in enumerate(zip(cache.key_cache, cache.value_cache)):
        new.update(kk.repeat_interleave(k, 0), vv.repeat_interleave(k, 0), layer)
    return new


@torch.no_grad()
def decode_labels(model, prompts: torch.Tensor, *, k_samples: int = 0, greedy: bool = True,
                  seed: int = 0, autocast: Optional[torch.dtype] = None,
                  return_samples: bool = False) -> Dict[str, np.ndarray]:
    """prompts: (B, L) equal-length prompts ending in BO_BEN, on the model's device.

    Returns greedy multi-hot sets (B, 19), RB marginals (B, 19) averaged over k_samples paths,
    the first-step slot mass (B,), and the greedy path length. return_samples adds every sampled
    set, (B, k_samples, 19) bool: the teacher blocks of L11b (`sample_freq` is their mean).
    """
    dev = prompts.device
    B = prompts.shape[0]
    ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else _null()
    with ctx:
        # logits_to_keep=1: only the last position's 87,577-way logits are used; the full
        # (B, L, V) tensor would cost ~25% more compute and GBs of memory per batch.
        out = model(input_ids=prompts, use_cache=True, logits_to_keep=1)
    first = out.logits[:, -1, :].float()
    slot_mass = torch.softmax(first, -1)[:, ALLOWED].sum(-1)
    res = {"slot_mass": slot_mass.cpu().numpy()}

    def run(cache, first_logits, n, sample: bool, gen: Optional[torch.Generator]):
        last = torch.full((n,), -1, dtype=torch.long, device=dev)
        stopped = torch.zeros(n, dtype=torch.bool, device=dev)
        chosen = torch.zeros(n, V.N_CLASSES, dtype=torch.bool, device=dev)
        rb = torch.zeros(n, V.N_CLASSES, dtype=torch.float32, device=dev)
        logits = first_logits
        steps = 0
        for _ in range(V.N_CLASSES + 1):
            lp = _restricted(logits, last)
            p = lp.exp()
            rb += p[:, :V.N_CLASSES] * (~stopped).unsqueeze(1)
            if sample:
                x = torch.multinomial(p, 1, generator=gen).squeeze(1)
            else:
                x = lp.argmax(-1)
            x = torch.where(stopped, torch.full_like(x, _EO_POS), x)
            is_cls = (x < V.N_CLASSES) & ~stopped
            chosen[is_cls, x[is_cls]] = True
            last = torch.where(is_cls, x, last)
            stopped |= x == _EO_POS
            steps += 1
            if bool(stopped.all()):
                break
            ids = torch.tensor(ALLOWED, device=dev)[x].unsqueeze(1)
            with ctx:
                o = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            logits = o.logits[:, -1, :].float()
        return chosen, rb, steps

    if greedy:
        ch, _, steps = run(_copy_cache(out.past_key_values), first, B, False, None)
        res["greedy"] = ch.cpu().numpy()
        res["greedy_steps"] = steps
    if k_samples:
        g = torch.Generator(device=dev).manual_seed(seed)
        cache = _expand_cache(out.past_key_values, k_samples)
        ch, rb, _ = run(cache, first.repeat_interleave(k_samples, 0), B * k_samples, True, g)
        res["marginal"] = rb.view(B, k_samples, -1).mean(1).cpu().numpy()
        res["sample_freq"] = ch.view(B, k_samples, -1).float().mean(1).cpu().numpy()
        if return_samples:
            res["samples"] = ch.view(B, k_samples, -1).cpu().numpy()
    return res


def _copy_cache(cache):
    from transformers import DynamicCache
    new = DynamicCache()
    for layer, (kk, vv) in enumerate(zip(cache.key_cache, cache.value_cache)):
        new.update(kk.clone(), vv.clone(), layer)
    return new


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@torch.no_grad()
def cache_check(model, prompts: torch.Tensor, n_steps: int = 3,
                autocast: Optional[torch.dtype] = None) -> Dict[str, float]:
    """Max |logit difference| between cached and uncached decoding, on the allowed ids.

    Feeds a fixed continuation (the first n_steps class ids) both ways.
    """
    ctx = torch.autocast("cuda", dtype=autocast) if autocast is not None else _null()
    cont = torch.tensor(V.CLASS_IDS[:n_steps], device=prompts.device)
    worst = 0.0
    with ctx:
        out = model(input_ids=prompts, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        seq = prompts
        for t in range(n_steps):
            ids = cont[t].expand(prompts.shape[0], 1)
            o = model(input_ids=ids, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            seq = torch.cat([seq, ids], 1)
            full = model(input_ids=seq, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            d = (o.logits[:, -1, ALLOWED].float() - full[:, ALLOWED].float()).abs().max()
            worst = max(worst, float(d))
    return {"max_abs_logit_diff": worst, "steps": n_steps, "batch": int(prompts.shape[0])}


# ------------------------------------------------------------------ metrics

def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """sklearn.metrics.average_precision_score for one binary problem (ties grouped)."""
    y = np.asarray(y).astype(bool).ravel()
    s = np.asarray(s, dtype=np.float64).ravel()
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    distinct = np.where(np.diff(s))[0]
    thr = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[thr]
    fps = 1 + thr - tps
    prec = tps / (tps + fps)
    rec = tps / tps[-1]
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def f1_scores(y: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp = (y & pred).sum(0).astype(float)
    fp = (~y & pred).sum(0).astype(float)
    fn = (y & ~pred).sum(0).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    # sample-averaged F1 (the "per-patch" view); empty-vs-empty counts as 1
    tps, fps, fns = (y & pred).sum(1), (~y & pred).sum(1), (y & ~pred).sum(1)
    den = 2 * tps + fps + fns
    samp = np.where(den > 0, 2 * tps / np.maximum(den, 1), 1.0)
    return {"micro_f1": float(micro), "macro_f1": float(np.nanmean(per)),
            "sample_f1": float(samp.mean()), "per_class_f1": per.tolist(),
            "exact_match": float((y == pred).all(1).mean())}


def map_scores(y: np.ndarray, s: np.ndarray) -> Dict[str, object]:
    y = np.asarray(y, bool)
    per = [average_precision(y[:, c], s[:, c]) for c in range(y.shape[1])]
    return {"micro_map": average_precision(y.ravel(), s.ravel()),
            "macro_map": float(np.nanmean(per)), "per_class_ap": per}


def prior_scores(train_y: np.ndarray, n: int) -> np.ndarray:
    """The class-prior predictor: every patch gets the training label frequencies."""
    return np.broadcast_to(np.asarray(train_y, float).mean(0), (n, train_y.shape[1])).copy()


def best_threshold(y: np.ndarray, s: np.ndarray, grid: Sequence[float] = None) -> float:
    """The threshold maximizing micro-F1 of (s >= t) on (y, s). Chosen on VAL, never test."""
    grid = np.linspace(0.02, 0.98, 49) if grid is None else grid
    return float(max(grid, key=lambda t: f1_scores(y, s >= t)["micro_f1"]))


def collapse_fraction(m_wrong: float, m_own: float, m_prior: float) -> float:
    """Share of the gain over the prior that survives when inputs come from another patch.

    ~0 = collapsed to the prior (the model reads its inputs); ~1 = unchanged (it does not).
    """
    return float((m_wrong - m_prior) / (m_own - m_prior)) if m_own != m_prior else float("nan")
