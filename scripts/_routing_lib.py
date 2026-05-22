"""Shared routing helpers for Mixture-of-Recursions experiments.

Importable without CLI side effects. Re-used by all U1–U4 notebooks and by
scripts/eval_multimodality_routing.py (which lifts its core helpers from here).

Lifted from scripts/eval_multimodality_routing.py on 2026-05-19.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from lm_dataset.multimodal_vocab_shared_caption_scene_desc import MODALITIES, PAD_ID  # noqa: F401


# ------------------------------------------------------------------ Modality
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
        if close:
            parts.append(torch.tensor([info.eo_id]))
        return torch.cat(parts)

    def chunk_tokens(self, modality: str, source, aug_idx: int = 0,
                     close: bool = True) -> torch.Tensor:
        info = MODALITIES[modality]
        arr = np.load(source) if isinstance(source, (str, Path)) else np.asarray(source)
        if arr.ndim == 2:
            arr = arr[aug_idx]
        body = torch.from_numpy(arr.flatten()).long() + info.codebook_offset
        parts = [torch.tensor([info.bo_id]), body]
        if close:
            parts.append(torch.tensor([info.eo_id]))
        return torch.cat(parts)

    def segment(self, input_ids: torch.Tensor) -> list[tuple[int, int, str]]:
        """Walk a token sequence; return (body_start, body_end_excl, modality)."""
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
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


# --------------------------------------------------------------- collect_routing
def collect_routing(model, mor_mod, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    """One no-cache forward; returns per-MoR-module depth choices, shape (L, T) in [1, Nr].

    For middle_cycle there is a single MoR module so L=1, but the same code handles
    multi-module variants (cycle / sequence) transparently.
    """
    captured: list[torch.Tensor] = []

    def hook(_m, _i, output):
        tei = getattr(output, "token_expert_indices", None)
        if tei is not None:
            captured.append(tei.detach().cpu())

    handles = [m.register_forward_hook(hook) for m in model.modules() if getattr(m, "mor", False)]
    try:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            model(input_ids=input_ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if not captured:
        raise RuntimeError("No MoR layers fired — pass a token-choice MoR checkpoint.")
    return torch.stack([t[0] for t in captured], dim=0).long() + 1  # (L, T)


# --------------------------------------------------------------- TopkOverride
class TopkOverride:
    """Context manager: while inside `with TopkOverride(expert=k):`, every call
    to torch.topk with k=1 along dim=-1 returns indices=expert everywhere.

    This overrides only the argmax; the trained softmax weight at the chosen
    expert is preserved so we ablate routing, not routing+gating.
    """

    def __init__(self, expert: int):
        self.expert = expert
        self._orig = None

    def __enter__(self):
        self._orig = torch.topk
        expert = self.expert
        orig = self._orig

        def patched(input, k, dim=-1, largest=True, sorted=True, *, out=None):
            if k == 1 and dim in (-1, input.ndim - 1):
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


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ------------------------------------------------------- teacher_forced_ce_per_modality
def teacher_forced_ce_per_modality(
    model, mod: Modality, seq: torch.Tensor, device: str, segs
) -> dict[str, float]:
    """One TF forward; returns mean CE per modality body slice."""
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


# --------------------------------------------------------------- load_sample
def load_sample(data_dir: Path, sid: str, aug_idx: int) -> dict:
    """Load one CLEVR sample.  Keys: id, aug_idx, rgb, depth, normal, caption
    (+ scene_desc when present).  Short keys (rgb/depth/normal) keep backward
    compatibility with eval_multimodality_routing.py experiment functions."""
    data_dir = Path(data_dir)
    sample: dict = {"id": sid, "aug_idx": aug_idx}
    for short, mod in (("rgb", "tok_rgb@256"), ("depth", "tok_depth@256"), ("normal", "tok_normal@256")):
        p = data_dir / mod / f"{sid}.npy"
        if p.exists():
            sample[short] = np.load(p)[aug_idx]
    cap_p = data_dir / "caption" / f"{sid}.json"
    if cap_p.exists():
        sample["caption"] = json.load(open(cap_p, encoding='utf-8'))[aug_idx]
    sd_p = data_dir / "scene_desc" / f"{sid}.json"
    if sd_p.exists():
        sample["scene_desc"] = json.load(open(sd_p, encoding='utf-8'))[aug_idx]
    return sample


# ----------------------------------------------------------------- Sample lists
# Seeded with numpy.random.default_rng(42). Explicit + committed so all notebooks
# and re-runs use identical splits.  Format: (scene_id, aug_idx).

SAMPLES_U1 = [
    ('01803', 3),
    ('02504', 9),
    ('02224', 7),
    ('00447', 3),
    ('00446', 9),
    ('01717', 4),
    ('02472', 5),
    ('03221', 4),
    ('01157', 1),
    ('02976', 9),
    ('02787', 0),
    ('04545', 1),
    ('02729', 2),
    ('03860', 2),
    ('03716', 1),
    ('01896', 0),
    ('03829', 6),
    ('03392', 9),
    ('03888', 1),
    ('01098', 2),
    ('04126', 5),
    ('04004', 9),
    ('00371', 6),
    ('03593', 7),
    ('00171', 5),
    ('00799', 4),
    ('00828', 1),
    ('04694', 8),
    ('02730', 8),
    ('00445', 0),
    ('01716', 7),
    ('02019', 0),
    ('00923', 7),
    ('03785', 8),
    ('04913', 1),
    ('02080', 4),
    ('03280', 1),
    ('04407', 5),
    ('04292', 9),
    ('01393', 0),
    ('01585', 3),
    ('00829', 9),
    ('02485', 3),
    ('04236', 1),
    ('00625', 4),
    ('00329', 5),
    ('03421', 6),
    ('03908', 3),
    ('02255', 9),
    ('00288', 5),
    ('01191', 2),
    ('02128', 9),
    ('01504', 9),
    ('03466', 4),
    ('01366', 0),
    ('03296', 5),
    ('03936', 5),
    ('00684', 1),
    ('03031', 6),
    ('02286', 8),
    ('03612', 1),
    ('03541', 7),
    ('02298', 1),
    ('04066', 7),
    ('01411', 4),
    ('00527', 0),
    ('00212', 9),
    ('01787', 1),
    ('00687', 5),
    ('03129', 1),
    ('03057', 9),
    ('01526', 7),
    ('04941', 8),
    ('02319', 4),
    ('02478', 4),
    ('03346', 6),
    ('02030', 7),
    ('04329', 8),
    ('00968', 0),
    ('03107', 6),
    ('01782', 1),
    ('01610', 2),
    ('02157', 8),
    ('03621', 2),
    ('02654', 7),
    ('01661', 4),
    ('01060', 2),
    ('02163', 3),
    ('00437', 5),
    ('02759', 7),
    ('00452', 6),
    ('00979', 9),
    ('02795', 8),
    ('02633', 7),
    ('02152', 6),
    ('01360', 2),
    ('00475', 4),
    ('00428', 1),
    ('04042', 3),
    ('04151', 9),
    ('02832', 4),
    ('01694', 1),
    ('04704', 6),
    ('03462', 9),
    ('02416', 8),
    ('01454', 2),
    ('03568', 4),
    ('00036', 5),
    ('02141', 2),
    ('03753', 2),
    ('00802', 2),
    ('04211', 9),
    ('01998', 7),
    ('04046', 8),
    ('03263', 8),
    ('04793', 9),
    ('02075', 1),
    ('03268', 8),
    ('04923', 0),
    ('02680', 4),
    ('02532', 5),
    ('03352', 3),
    ('04220', 1),
    ('00749', 1),
    ('04484', 8),
    ('01766', 3),
    ('03289', 3),
    ('02170', 9),
    ('03143', 1),
    ('01339', 9),
    ('01106', 9),
    ('04589', 5),
    ('02160', 1),
    ('03319', 1),
    ('00112', 2),
    ('03922', 5),
    ('01595', 1),
    ('03670', 4),
    ('04470', 1),
    ('03834', 5),
    ('00784', 0),
    ('03771', 7),
    ('02761', 0),
    ('03342', 1),
    ('02412', 1),
    ('01173', 8),
    ('03287', 0),
    ('03972', 4),
    ('04809', 5),
    ('01942', 5),
    ('00413', 6),
    ('00563', 8),
    ('04705', 3),
    ('00183', 8),
    ('03664', 3),
    ('00589', 9),
    ('03946', 5),
    ('03470', 7),
    ('04118', 8),
    ('02495', 7),
    ('03836', 8),
    ('02470', 5),
    ('03875', 0),
    ('02314', 4),
    ('01207', 1),
    ('04978', 8),
    ('00390', 2),
    ('02108', 2),
    ('04908', 2),
    ('02754', 6),
    ('02176', 5),
    ('04396', 9),
    ('04081', 4),
    ('04855', 2),
    ('02265', 0),
    ('03769', 0),
    ('00308', 3),
    ('02784', 9),
    ('00758', 5),
    ('00880', 8),
    ('02017', 1),
    ('02134', 0),
    ('00617', 8),
    ('03454', 1),
    ('03146', 0),
    ('01711', 4),
    ('03076', 9),
    ('03412', 6),
    ('01500', 0),
    ('03775', 8),
    ('01806', 8),
    ('03686', 8),
    ('01900', 9),
    ('04948', 2),
    ('00944', 9),
    ('04069', 3),
    ('00152', 8),
    ('03119', 5),
    ('04672', 7),
    ('00633', 6),
]

SAMPLES_U2_U3 = [
    ('01892', 3),
    ('02961', 4),
    ('01466', 0),
    ('03994', 3),
    ('02976', 5),
    ('01806', 9),
    ('04780', 7),
    ('03624', 7),
    ('00813', 5),
    ('00656', 1),
    ('01615', 3),
    ('01004', 9),
    ('02640', 0),
    ('03164', 8),
    ('00473', 6),
    ('03021', 2),
    ('03799', 7),
    ('03916', 8),
    ('02221', 0),
    ('01228', 1),
    ('03604', 6),
    ('03819', 9),
    ('03094', 0),
    ('02352', 6),
    ('01162', 1),
    ('02891', 6),
    ('03970', 6),
    ('01666', 0),
    ('01310', 6),
    ('04095', 8),
    ('03647', 8),
    ('04276', 0),
    ('03451', 2),
    ('01145', 2),
    ('04725', 4),
    ('02140', 1),
    ('00902', 8),
    ('02134', 7),
    ('04735', 8),
    ('02489', 1),
    ('04052', 0),
    ('03611', 9),
    ('03634', 2),
    ('00359', 6),
    ('02053', 9),
    ('03985', 0),
    ('00659', 0),
    ('04754', 0),
    ('02605', 1),
    ('01585', 0),
    ('02740', 8),
    ('00450', 6),
    ('04784', 9),
    ('04661', 8),
    ('03906', 9),
    ('03373', 2),
    ('03385', 2),
    ('02378', 8),
    ('02649', 2),
    ('04651', 0),
    ('02028', 6),
    ('00687', 8),
    ('04567', 9),
    ('00684', 9),
    ('03853', 1),
    ('02949', 7),
    ('00876', 8),
    ('01811', 6),
    ('04201', 8),
    ('01624', 8),
    ('01872', 4),
    ('02630', 0),
    ('04975', 4),
    ('02524', 2),
    ('03138', 1),
    ('00250', 1),
    ('00844', 6),
    ('03055', 5),
    ('04211', 5),
    ('02278', 7),
    ('00300', 5),
    ('04688', 6),
    ('04529', 0),
    ('02500', 8),
    ('00420', 4),
    ('02966', 1),
    ('00174', 8),
    ('03809', 7),
    ('03607', 0),
    ('03315', 1),
    ('04223', 0),
    ('03232', 2),
    ('03148', 0),
    ('01002', 7),
    ('02273', 5),
    ('01266', 9),
    ('03144', 7),
    ('01429', 6),
    ('02422', 5),
    ('04747', 0),
]
