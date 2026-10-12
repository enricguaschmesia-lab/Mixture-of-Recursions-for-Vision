"""Init-from-checkpoint for the BEN fine-tunes, and the loader for their checkpoints (`.venv`).

⚠ THIS IS NOT A RESUME (PHASE6_PLAN §4.9). A resume restores optimizer, scheduler and step,
and `preflight.check_resume_checkpoint` refuses a partial model load, which loading across
a vocabulary change would be. Here:
  * the model is BUILT at the extended vocabulary (config `model_config.vocab_size` 87,577,
    asserted against the ben registry at startup);
  * the EXTENDED checkpoint (`eo/scripts/extend_checkpoint.py`) is loaded STRICTLY: no
    unexpected key, and the only missing key allowed is `lm_head.weight` when it is tied;
  * optimizer and scheduler start fresh, by design (HF builds them; nothing is restored);
  * its identity artifact (`extension.json`) must exist and record a passed identity check
    with both controls fired, and the weights must hash to what it recorded.
Then the regime is applied: `full` (everything trains) or `lora` (eo/finetune/lora.py).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import torch

from eo.finetune import extend as X
from eo.finetune import lora as LR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _weights(d: Path) -> Path:
    for n in ("model.safetensors", "pytorch_model.bin"):
        if (d / n).exists():
            return d / n
    raise FileNotFoundError(f"no model.safetensors / pytorch_model.bin in {d}")


def _load_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def verify_extension(init_dir: Path, vocab_size: int, check_hash: bool = True) -> Dict:
    """The identity artifact's verdict. Raises with the reason if it does not hold."""
    ej = init_dir / "extension.json"
    if not ej.is_file():
        raise RuntimeError(f"{ej} missing: {init_dir} was not produced by extend_checkpoint.py "
                           f"(or its identity check never passed)")
    doc = json.loads(ej.read_text())
    if not doc.get("identity", {}).get("passed"):
        raise RuntimeError(f"{ej}: identity check did not pass")
    if not (doc.get("control_insert_before_pad", {}).get("fired")
            and doc.get("control_untied_resize", {}).get("fired")):
        raise RuntimeError(f"{ej}: a control did not fire, so the identity check proves nothing")
    if int(doc["extension"]["vocab_after"]) != int(vocab_size):
        raise RuntimeError(f"{ej}: extended to {doc['extension']['vocab_after']}, "
                           f"config wants {vocab_size}")
    wf = init_dir / doc["weights_file"]
    if check_hash and _sha256(wf) != doc["weights_sha256"]:
        raise RuntimeError(f"{wf} does not hash to what extension.json recorded: it changed "
                           f"after the identity check")
    return doc


def strict_load(model, ckpt_dir: Path) -> Dict:
    state = _load_state(_weights(ckpt_dir))
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = getattr(model.config, "tie_word_embeddings", False)
    allowed = {"lm_head.weight"} if tied else set()
    bad_missing = sorted(set(missing) - allowed)
    if unexpected or bad_missing:
        raise RuntimeError(f"partial load from {ckpt_dir}: missing {bad_missing[:5]} "
                           f"({len(bad_missing)}), unexpected {list(unexpected)[:5]} "
                           f"({len(unexpected)}). Refusing: this is exactly the silent "
                           f"partial load the resume checks exist for.")
    return {"tensors": len(state), "missing_allowed": sorted(set(missing) & allowed)}


def init_from(model, cfg) -> Dict:
    """pretrain.py's hook: strict-load the extended checkpoint, then apply the regime."""
    from eo.data import ben_vocab as BV
    ft = cfg.finetune
    init_dir = Path(str(ft.init_from))
    vocab = int(model.config.vocab_size)
    if vocab != BV.TOTAL_VOCAB_SIZE:
        raise RuntimeError(f"model built at vocab {vocab}, BEN registry is {BV.TOTAL_VOCAB_SIZE}")
    ext = verify_extension(init_dir, vocab)
    load = strict_load(model, init_dir)
    X.assert_tied(model)
    report = {"init_from": str(init_dir), "extension_source": ext["source_checkpoint"],
              "weights_sha256": ext["weights_sha256"], "load": load,
              "regime": str(ft.regime)}
    if ft.regime == "lora":
        if cfg.recursive.get("enable"):
            raise NotImplementedError(
                "LoRA on a recursive/MoR model: the shared block's adapter must be shared "
                "across recursion steps and verified as the same tensor (PHASE6_PLAN §4.6). "
                "Only the vanilla arm (B10) is wired so far.")
        report["lora"] = LR.apply_lora(model, int(ft.lora.r), float(ft.lora.alpha), BV.BASE)
    elif ft.regime == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"finetune.regime must be 'full' or 'lora', got {ft.regime!r}")
    report["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    report["total_params"] = int(sum(p.numel() for p in model.parameters()))
    return report


def build_finetuned(config_name: str, checkpoint: Optional[str], device: str = "cuda",
                    overrides: Optional[Sequence[str]] = None):
    """A fine-tuned (or extended-base, `checkpoint=None` -> finetune.init_from) model, for eval.

    Built exactly as training built it: compose the BEN config, build at 87,577, apply the
    regime's structure (LoRA wraps BEFORE loading, so the keys match), then strict-load.
    """
    from eo.generate.conditional import build_model
    model, cfg = build_model(config_name, None, device="cpu", overrides=overrides)
    ft = cfg.finetune
    if checkpoint is None:
        verify_extension(Path(str(ft.init_from)), int(model.config.vocab_size))
        strict_load(model, Path(str(ft.init_from)))
        X.assert_tied(model)
    else:
        if ft.regime == "lora":
            from eo.data import ben_vocab as BV
            LR.apply_lora(model, int(ft.lora.r), float(ft.lora.alpha), BV.BASE)
        strict_load(model, Path(checkpoint))
        if ft.regime != "lora":
            X.assert_tied(model)
    model.to(device).eval()
    return model, cfg
