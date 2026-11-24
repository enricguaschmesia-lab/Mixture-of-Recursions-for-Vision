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

A second kind of base (A2 Step 5.2, 2026-11-23): a CONTINUED base, a training checkpoint of a run
that continued from a verified extended base (`pretrain_init` variant `continue`, every parameter
training: A2's R1). It has no `extension.json`; its record is `continued.json`, written by
`write_continued` (CLI eo/scripts/package_continued.py), which holds the base's own extension.json,
the run's pretrain_init.json, the run, step and vocabulary size, and the weights' SHA-256.
`verify_extension` accepts either record, never both.
"""
from __future__ import annotations

import hashlib
import json
import shutil
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


CONTINUED = "continued.json"      # a continued base's record (A2 Step 5.2; write_continued)


def _check_extension_doc(doc: Dict, vocab_size: int, where) -> None:
    """An extension.json's verdict without its weights: identity passed, both controls fired, the vocabulary."""
    if not doc.get("identity", {}).get("passed"):
        raise RuntimeError(f"{where}: identity check did not pass")
    if not (doc.get("control_insert_before_pad", {}).get("fired")
            and doc.get("control_untied_resize", {}).get("fired")):
        raise RuntimeError(f"{where}: a control did not fire, so the identity check proves nothing")
    if int(doc["extension"]["vocab_after"]) != int(vocab_size):
        raise RuntimeError(f"{where}: extended to {doc['extension']['vocab_after']}, "
                           f"config wants {vocab_size}")


def verify_extension(init_dir: Path, vocab_size: int, check_hash: bool = True) -> Dict:
    """The identity artifact's verdict. Raises with the reason if it does not hold.

    A directory with `continued.json` and no `extension.json` is a continued base (verify_continued); its
    returned record carries `kind: "continued"`, `weights_sha256` and `source_checkpoint` like an extension's."""
    ej = init_dir / "extension.json"
    if ej.is_file() and (init_dir / CONTINUED).is_file():
        raise RuntimeError(f"{init_dir} holds both extension.json and {CONTINUED}: which base is it?")
    if not ej.is_file() and (init_dir / CONTINUED).is_file():
        return verify_continued(init_dir, vocab_size, check_hash)
    if not ej.is_file():
        raise RuntimeError(f"{ej} missing: {init_dir} was not produced by extend_checkpoint.py "
                           f"(or its identity check never passed), and has no {CONTINUED} either")
    doc = json.loads(ej.read_text())
    _check_extension_doc(doc, vocab_size, ej)
    wf = init_dir / doc["weights_file"]
    if check_hash and _sha256(wf) != doc["weights_sha256"]:
        raise RuntimeError(f"{wf} does not hash to what extension.json recorded: it changed "
                           f"after the identity check")
    return doc


def verify_continued(init_dir: Path, vocab_size: int, check_hash: bool = True) -> Dict:
    """A continued base's verdict (A2 Step 5.2). Raises with the reason if it does not hold.

    * `continued.json` exists, of kind "continued", version 1, at the config's vocabulary, as is the directory's
      config.json;
    * the base it continued from passed its own identity check (its extension.json, held in the record: identity
      passed, both controls fired) at the SAME vocabulary, so the continuation never resized the table;
    * the run continued from exactly that base: its pretrain_init.json (held in the record) is variant `continue`,
      without LoRA or a frozen body, and names the base's weights hash;
    * the weights hash to what the record holds (`check_hash`).
    The base's path is not checked: it differs between sites, and the hash is the link."""
    cj = init_dir / CONTINUED
    if not cj.is_file():
        raise RuntimeError(f"{cj} missing: {init_dir} is not a packaged continued base "
                           f"(eo/scripts/package_continued.py)")
    doc = json.loads(cj.read_text())
    if doc.get("kind") != "continued" or doc.get("version") != 1:
        raise RuntimeError(f"{cj}: not a continued-base record (kind {doc.get('kind')!r}, "
                           f"version {doc.get('version')!r})")
    if int(doc["vocab_size"]) != int(vocab_size):
        raise RuntimeError(f"{cj}: a base at {doc['vocab_size']} ids, config wants {vocab_size}")
    cfgj = init_dir / "config.json"
    if cfgj.is_file() and int(json.loads(cfgj.read_text()).get("vocab_size", -1)) != int(vocab_size):
        raise RuntimeError(f"{cfgj}: vocab_size {json.loads(cfgj.read_text()).get('vocab_size')}, "
                           f"config wants {vocab_size}")
    base = doc["base"]
    _check_extension_doc(base["extension"], vocab_size, f"{cj}: the base's extension.json")
    if base["extension"]["weights_sha256"] != base["weights_sha256"]:
        raise RuntimeError(f"{cj}: the base's weights hash {base['weights_sha256'][:12]} is not its "
                           f"extension.json's {base['extension']['weights_sha256'][:12]}")
    pi = doc["pretrain_init"]
    if pi.get("variant") != "continue" or pi.get("lora") is not None or pi.get("freeze_body"):
        raise RuntimeError(f"{cj}: the run was not a full continuation (variant {pi.get('variant')!r}, "
                           f"lora {pi.get('lora')!r}, freeze_body {pi.get('freeze_body')!r})")
    if pi.get("conversion_weights_sha256") != base["weights_sha256"]:
        raise RuntimeError(f"{cj}: the run continued from weights {str(pi.get('conversion_weights_sha256'))[:12]}, "
                           f"not the verified base's {base['weights_sha256'][:12]}")
    wf = init_dir / doc["weights_file"]
    if check_hash and _sha256(wf) != doc["weights_sha256"]:
        raise RuntimeError(f"{wf} does not hash to what {CONTINUED} recorded: it changed after packaging")
    return doc


def write_continued(checkpoint: Path, base: Path, out: Path, admission: Optional[Path] = None,
                    note: str = "") -> Dict:
    """Package a training checkpoint of a `continue` run as a fine-tune base: its weights and config.json copied
    into `out` beside a `continued.json`, then verified back. Refuses unless the base passes verify_extension here
    (weights hashed) at the checkpoint's vocabulary, the run's pretrain_init.json names that base's weights, the
    checkpoint is its trainer state's step, and `out` does not exist. `admission`: a json recording why this
    checkpoint is the base (kept by path and SHA-256, with its top-level scalars)."""
    import datetime
    import socket
    checkpoint, base, out = Path(checkpoint).resolve(), Path(base).resolve(), Path(out)
    run_dir = checkpoint.parent
    if out.exists():
        raise FileExistsError(f"{out} exists: a packaged base is never rewritten")
    pi = json.loads((run_dir / "pretrain_init.json").read_text())
    cfg = json.loads((checkpoint / "config.json").read_text())
    vocab = int(cfg["vocab_size"])
    ext = verify_extension(base, vocab, check_hash=True)
    if ext.get("kind") == "continued":
        raise RuntimeError(f"{base} is itself a continued base: package from the extension it continued from")
    ts = json.loads((checkpoint / "trainer_state.json").read_text())
    step = int(ts["global_step"])
    if checkpoint.name != f"checkpoint-{step}":
        raise RuntimeError(f"{checkpoint}: its trainer state is at step {step}")
    wf = _weights(checkpoint)
    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise FileExistsError(f"{tmp} exists: a previous attempt stopped there; look at it and remove it")
    tmp.mkdir(parents=True)
    shutil.copy2(wf, tmp / wf.name)
    shutil.copy2(checkpoint / "config.json", tmp / "config.json")
    sha = _sha256(tmp / wf.name)
    if sha != _sha256(wf):
        raise RuntimeError(f"the copy of {wf} does not hash to its source")
    doc = {"kind": "continued", "version": 1,
           "what": "a training checkpoint of a run that continued from a verified extended base, packaged as a "
                   "fine-tune base (eo/finetune/init.py write_continued)",
           "run": run_dir.name, "step": step, "max_steps": ts.get("max_steps"),
           "source_checkpoint": str(checkpoint), "vocab_size": vocab,
           "weights_file": wf.name, "weights_sha256": sha,
           "config_sha256": _sha256(tmp / "config.json"),
           "trainer_state_sha256": _sha256(checkpoint / "trainer_state.json"),
           "base": {"path": str(base), "weights_sha256": ext["weights_sha256"], "extension": ext},
           "pretrain_init": pi, "note": note,
           "packaged": {"at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                        "host": socket.gethostname()}}
    if admission is not None:
        adm = json.loads(Path(admission).read_text())
        doc["admission"] = {"record": str(Path(admission).resolve()), "sha256": _sha256(Path(admission)),
                            **{k: v for k, v in adm.items() if isinstance(v, (int, float, str, bool))}}
    (tmp / CONTINUED).write_text(json.dumps(doc, indent=2) + "\n")
    verify_continued(tmp, vocab, check_hash=True)
    tmp.rename(out)
    return verify_extension(out, vocab, check_hash=True)


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


def frozen_rows(cfg) -> int:
    """How many leading vocabulary rows LoRA and head freeze: the pretraining vocabulary. BEN: ben_vocab.BASE
    (87,556). A2's segmentation (dataset geobench_seg): the temporal base the map rows were appended to,
    derived from the model's size (eo/data/map_vocab.py), so the map rows, and only they, train."""
    if cfg.get("dataset") == "geobench_seg":
        from eo.data import map_vocab as MV
        return MV.for_total(int(cfg.model_config.vocab_size)).base_vocab
    from eo.data import ben_vocab as BV
    return BV.BASE


def expected_vocab(cfg) -> int:
    if cfg.get("dataset") == "geobench_seg":
        from eo.data import map_vocab as MV
        return MV.TOTAL_VOCAB_SIZE
    from eo.data import ben_vocab as BV
    return BV.TOTAL_VOCAB_SIZE


def init_from(model, cfg) -> Dict:
    """pretrain.py's hook: strict-load the extended checkpoint, then apply the regime."""
    ft = cfg.finetune
    vocab = int(model.config.vocab_size)
    if vocab != expected_vocab(cfg):
        raise RuntimeError(f"model built at vocab {vocab}, the {cfg.get('dataset')} registry is {expected_vocab(cfg)}")
    n_old = frozen_rows(cfg)
    if ft.regime == "scratch":
        # L4's reference: the same architecture, randomly initialized, nothing loaded.
        if ft.get("init_from"):
            raise ValueError("finetune.regime=scratch takes no init_from")
        X.assert_tied(model)
        for p in model.parameters():
            p.requires_grad_(True)
        n = int(sum(p.numel() for p in model.parameters()))
        return {"init_from": None, "regime": "scratch", "trainable_params": n, "total_params": n}
    init_dir = Path(str(ft.init_from))
    ext = verify_extension(init_dir, vocab)
    load = strict_load(model, init_dir)
    X.assert_tied(model)
    report = {"init_from": str(init_dir), "extension_source": ext["source_checkpoint"],
              "weights_sha256": ext["weights_sha256"], "load": load,
              "regime": str(ft.regime), "init_kind": ext.get("kind", "extension")}
    if ft.regime == "lora":
        if cfg.recursive.get("enable"):
            raise NotImplementedError(
                "LoRA on a recursive/MoR model: the shared block's adapter must be shared "
                "across recursion steps and verified as the same tensor (PHASE6_PLAN §4.6). "
                "Only the vanilla arm (B10) is wired so far.")
        report["lora"] = LR.apply_lora(model, int(ft.lora.r), float(ft.lora.alpha), n_old)
    elif ft.regime == "head":
        report["head"] = LR.apply_head(model, n_old)
    elif ft.regime == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(f"finetune.regime must be 'full', 'lora', 'head' or 'scratch', got {ft.regime!r}")
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
        n_old = frozen_rows(cfg)
        if ft.regime == "lora":
            LR.apply_lora(model, int(ft.lora.r), float(ft.lora.alpha), n_old)
        elif ft.regime == "head":
            LR.apply_head(model, n_old)
        strict_load(model, Path(checkpoint))
        if ft.regime not in ("lora", "head"):
            X.assert_tied(model)
    model.to(device).eval()
    return model, cfg
