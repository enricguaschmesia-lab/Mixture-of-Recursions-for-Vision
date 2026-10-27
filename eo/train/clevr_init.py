# eo/train/clevr_init.py
"""Initialise a TerraMesh vanilla model from the CLEVR vanilla checkpoint (`.venv`).

The source is the CS-503 multimodal vanilla, `Mahlia/baseline-multimodal-vanilla` at
revision `a92e8b2a`: SmolLM-135M at 29 layers over the 242,271-id CLEVR vocabulary
(`lm_dataset/multimodal_vocab_shared_caption_scene_desc.py`), saved at step 5,000 of a
19,500-step WSD schedule with the LR still on its 3e-3 plateau. Its transformer body has
exactly B10c's shape (102,663,360 parameters); only the vocabulary differs.

What carries over (decided 2026-10-27, docs/notes/clevr_init_2026-10-27.md):
  * the BODY, i.e. the 29 layers and the final norm (262 tensors), copied bit for bit.
    The checkpoint is bf16 and B10c's parameters are fp32, so the cast is exact.
  * NOT the embedding (tied to `lm_head`). CLEVR row k is Cosmos DI16x16 code k, and EO row
    k is a TerraMind FSQ code k. The two share a number, not a meaning, so mapping them
    would be a confident wrong prior. The EO table is the one B10c's own build makes,
    seeded, so the CLEVR arm and its control get the same table bit for bit.
  * The CLEVR RGB rows (ids 0..63,999, plus tok_rgb@256's BO/EO) are kept in a separate
    file for later use. Nothing else of the CLEVR table is kept.

The SHUFFLED control is the same body with each tensor's entries permuted independently,
seeded. It keeps every tensor's values and scale but none of the learned structure. So it
separates "CLEVR taught the body something" from "trained weights are better scaled than
an init".

⚠ The scale gap is real and deliberate. CLEVR's embedding has std 0.107 (row norm 2.46;
the RGB rows 2.96); a fresh N(0, 0.02) row has norm 0.48. The body was trained on a
residual stream that starts ~5-6x larger than the one it now gets. The run plan's embedding-only
warm-up is what lets the new table grow into it; the converter does not rescale anything.

Every check here comes with a control that must FAIL it (CLAUDE.md rule 2). The controls
are built in `eo/scripts/convert_clevr_init.py`, which writes `conversion.json` only when
every check passed and every control fired.

The training side is at the bottom: `pretrain_init` (pretrain.py's hook), `freeze_body`
(stage 1 trains the embedding table alone) and `check_handoff` (stage 2 starts from a stage 1
whose body is still the conversion's, bit for bit). Their controls are in
`eo/train/preflight_controls.py` and in `eo/scripts/verify_phase3.py` W19 and W20.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

CLEVR_REPO = "Mahlia/baseline-multimodal-vanilla"
CLEVR_REVISION = "a92e8b2a07d7cdd2932baf4dc874324013207a81"
# sha256 of pytorch_model.bin at CLEVR_REVISION; equals Hugging Face's LFS etag (2026-10-27).
CLEVR_WEIGHTS_SHA256 = "45c7e968f965b3b04857d0b851cd9be0b385cc08468676eb293987894ac4d224"
CLEVR_RGB_MODALITY = "tok_rgb@256"

EMBED_KEY = "model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"
EMBED_KEYS = (EMBED_KEY, HEAD_KEY)

VARIANTS = ("clevr", "shuffled")
FUNCTION_TOL = 1e-5        # identical tensors through identical fp32 CPU code: expected 0
EMBED_STD_RTOL = 0.02      # the fresh table's std against initializer_range
SHUFFLED_SAME_MAX = 0.05   # a permuted body may keep a few entries in place by chance, not more


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def clevr_vocab():
    """The CLEVR registry the checkpoint was trained against (the shared-caption one).

    `lm_dataset/multimodal_vocab.py` is the other CLEVR registry, at 292,539 ids; only the
    shared-caption layout gives the checkpoint's 242,271. `load_clevr` asserts it.
    """
    from lm_dataset import multimodal_vocab_shared_caption_scene_desc as CV
    return CV


def load_clevr(src: Path) -> Dict:
    """The pinned CLEVR checkpoint: state (bf16), config and trainer state. Refuses anything else."""
    wf = src / "pytorch_model.bin"
    got = sha256(wf)
    if got != CLEVR_WEIGHTS_SHA256:
        raise RuntimeError(f"{wf} hashes to {got}, not the pinned {CLEVR_REPO}@{CLEVR_REVISION[:8]}")
    config = json.loads((src / "config.json").read_text())
    CV = clevr_vocab()
    if int(config["vocab_size"]) != CV.TOTAL_VOCAB_SIZE:
        raise RuntimeError(f"checkpoint vocab {config['vocab_size']} != shared-caption CLEVR "
                           f"registry {CV.TOTAL_VOCAB_SIZE}")
    # A pickle: weights_only, never a full unpickle of a downloaded file.
    state = torch.load(wf, map_location="cpu", weights_only=True)
    if HEAD_KEY in state and not torch.equal(state[HEAD_KEY], state[EMBED_KEY]):
        raise RuntimeError("CLEVR lm_head differs from its embedding: the checkpoint is not tied")
    ts = src / "trainer_state.json"
    trainer = json.loads(ts.read_text()) if ts.is_file() else {}
    last = (trainer.get("log_history") or [{}])[-1]
    return {"state": state, "config": config, "weights_sha256": got,
            "global_step": trainer.get("global_step"), "max_steps": trainer.get("max_steps"),
            "lr_at_save": last.get("learning_rate"), "loss_at_save": last.get("loss")}


def body_keys(state: Dict[str, torch.Tensor]):
    return sorted(k for k in state if k not in EMBED_KEYS)


def clevr_reference(src: Path, state: Dict[str, torch.Tensor]):
    """The CLEVR model itself, fp32 on CPU, built with the class B10c uses: the function check's reference."""
    from transformers import AutoConfig
    from model.util import MODEL_CLS
    config = AutoConfig.from_pretrained(str(src), local_files_only=True)
    m = MODEL_CLS["smollm"]._from_config(config, attn_implementation="sdpa", torch_dtype=torch.float32)
    m.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return m.eval()


def body_state(state: Dict[str, torch.Tensor], variant: str, shuffle_seed: int = 0) -> Dict[str, torch.Tensor]:
    """The body to install: CLEVR's as saved, or each tensor permuted independently."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    keys = body_keys(state)
    if variant == "clevr":
        return {k: state[k] for k in keys}
    g = torch.Generator().manual_seed(shuffle_seed)
    out = {}
    for k in keys:              # sorted, so the permutations do not depend on dict order
        t = state[k]
        out[k] = t.flatten()[torch.randperm(t.numel(), generator=g)].view_as(t).clone()
    return out


@torch.no_grad()
def install_body(model, body: Dict[str, torch.Tensor]) -> int:
    """Copy `body` into `model` in place. The key sets must match exactly, embedding aside."""
    sd = model.state_dict()
    want = {k for k in sd if k not in EMBED_KEYS}
    if set(body) != want:
        raise RuntimeError(f"body key mismatch: missing {sorted(want - set(body))[:5]}, "
                           f"unexpected {sorted(set(body) - want)[:5]}")
    for k, t in body.items():
        if sd[k].shape != t.shape:
            raise RuntimeError(f"{k}: model {tuple(sd[k].shape)} vs source {tuple(t.shape)}")
        sd[k].copy_(t.to(sd[k].dtype))      # state_dict tensors share storage with the parameters
    return len(body)


def rgb_rows(state: Dict[str, torch.Tensor]):
    """CLEVR's RGB rows, bf16 as trained. Tied, so they are input embeddings AND output rows."""
    m = clevr_vocab().get_modality(CLEVR_RGB_MODALITY)
    e = state[EMBED_KEY]
    tensors = {"code_rows": e[m.codebook_offset: m.codebook_offset + m.codebook_size].clone(),
               "bo_row": e[m.bo_id].clone(), "eo_row": e[m.eo_id].clone()}
    meta = {"source_repo": CLEVR_REPO, "source_revision": CLEVR_REVISION,
            "source_weights_sha256": CLEVR_WEIGHTS_SHA256, "modality": CLEVR_RGB_MODALITY,
            "codebook_offset": str(m.codebook_offset), "codebook_size": str(m.codebook_size),
            "bo_id": str(m.bo_id), "eo_id": str(m.eo_id),
            "tokenizer": "Cosmos DI16x16 (CLEVR, 256x256 -> 16x16 tokens)", "tied": "true"}
    return tensors, meta


# --- checks -----------------------------------------------------------------------------
# Each returns a dict with `passed`. The CLI pairs each with a control that must fail it.

def check_body_identity(model, state: Dict[str, torch.Tensor]) -> Dict:
    """Every body tensor equals CLEVR's, exactly."""
    sd = model.state_dict()
    bad, worst = [], 0.0
    keys = body_keys(state)
    for k in keys:
        a, b = sd.get(k), state[k]
        if a is None or a.shape != b.shape:
            bad.append(k)
            worst = float("inf")
            continue
        if not torch.equal(a.float(), b.float()):
            bad.append(k)
            worst = max(worst, float((a.float() - b.float()).abs().max()))
    return {"tensors": len(keys), "unequal": len(bad), "first_unequal": bad[:3],
            "max_abs_diff": worst, "passed": not bad}


def check_body_permutation(model, state: Dict[str, torch.Tensor]) -> Dict:
    """Every body tensor is a permutation of CLEVR's, and the body as a whole has moved.

    A tensor holding a single repeated value cannot move under a permutation, so it is
    counted but not required to.
    """
    sd = model.state_dict()
    not_perm, unmoved, constant, same, total = [], [], 0, 0, 0
    for k in body_keys(state):
        a, b = sd[k].float().flatten(), state[k].float().flatten()
        if a.shape != b.shape or not torch.equal(a.sort().values, b.sort().values):
            not_perm.append(k)
            continue
        if b.unique().numel() == 1:
            constant += 1
        elif torch.equal(a, b):
            unmoved.append(k)
        same += int((a == b).sum())
        total += a.numel()
    frac = same / max(total, 1)
    return {"not_a_permutation": len(not_perm), "first_not_a_permutation": not_perm[:3],
            "unmoved": len(unmoved), "constant_tensors": constant,
            "fraction_entries_in_place": frac,
            "passed": not not_perm and not unmoved and frac < SHUFFLED_SAME_MAX}


@torch.no_grad()
def last_hidden(model, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    return model.model(inputs_embeds=x, position_ids=position_ids, use_cache=False).last_hidden_state


def check_function(reference, model, x: torch.Tensor, position_ids: torch.Tensor) -> Dict:
    """Same inputs_embeds through both bodies: the final hidden states must agree.

    This is what the tensor check cannot see: a config difference (rope_theta, norm eps,
    head layout) leaves every tensor equal and changes the function.
    """
    a, b = last_hidden(reference, x, position_ids), last_hidden(model, x, position_ids)
    d = float((a - b).abs().max())
    return {"tokens": int(x.shape[0] * x.shape[1]), "max_abs_diff": d,
            "hidden_scale": float(a.abs().mean()), "tol": FUNCTION_TOL, "passed": d <= FUNCTION_TOL}


def check_fresh_embedding(table: torch.Tensor, reference: torch.Tensor, init_std: float) -> Dict:
    """The table is the seeded B10c build's, bit for bit, at the init's scale."""
    e = table.detach()
    equal = e.shape == reference.shape and torch.equal(e, reference)
    std = float(e.float().std())
    return {"shape": list(e.shape), "equal_to_seeded_build": bool(equal), "std": std,
            "init_std": init_std,
            "passed": bool(equal) and abs(std / init_std - 1) < EMBED_STD_RTOL}


def overlap_layout(fresh: torch.Tensor, state: Dict[str, torch.Tensor], n_rows: int) -> torch.Tensor:
    """THE REJECTED OPTION, built only as a control: the first EO slot (S2L2A) takes the
    CLEVR RGB rows of the same local ids, i.e. "S2 ids 0..n = RGB ids 0..n"."""
    rgb, _ = rgb_rows(state)
    w = fresh.clone()
    w[:n_rows] = rgb["code_rows"][:n_rows].to(w.dtype)
    return w


# --- the consumer's half ----------------------------------------------------------------

def verify_conversion(init_dir: Path, variant: str, vocab_size: int, check_hash: bool = True) -> Dict:
    """conversion.json's verdict. A training hook calls this before loading; raises with the reason."""
    cj = init_dir / "conversion.json"
    if not cj.is_file():
        raise RuntimeError(f"{cj} missing: {init_dir} was not written by convert_clevr_init.py, "
                           f"or its checks did not pass")
    doc = json.loads(cj.read_text())
    if doc.get("variant") != variant:
        raise RuntimeError(f"{cj}: variant {doc.get('variant')!r}, the run wants {variant!r}")
    failed = [n for n, c in doc.get("checks", {}).items() if not c.get("passed")]
    silent = [n for n, c in doc.get("controls", {}).items() if not c.get("fired")]
    if failed or silent or not doc.get("checks") or not doc.get("controls"):
        raise RuntimeError(f"{cj}: checks failed {failed}, controls that did not fire {silent}")
    if int(doc["vocab_size"]) != int(vocab_size):
        raise RuntimeError(f"{cj}: converted at vocab {doc['vocab_size']}, the run builds {vocab_size}")
    wf = init_dir / doc["weights_file"]
    if check_hash and sha256(wf) != doc["weights_sha256"]:
        raise RuntimeError(f"{wf} does not hash to what conversion.json recorded")
    return doc


def save_rgb_rows(out: Path, state: Dict[str, torch.Tensor]) -> Dict:
    from safetensors.torch import save_file
    tensors, meta = rgb_rows(state)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "clevr_rgb_rows.safetensors"
    save_file(tensors, str(f), metadata=meta)
    return {"file": f.name, "sha256": sha256(f),
            "shapes": {k: list(v.shape) for k, v in tensors.items()},
            "dtype": str(tensors["code_rows"].dtype), **meta}


def load_rgb_rows(path: Path, expect_sha256: Optional[str] = None) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    if expect_sha256 is not None and sha256(path) != expect_sha256:
        raise RuntimeError(f"{path} does not hash to {expect_sha256}")
    return load_file(str(path), device="cpu")


# --------------------------------------------------------------------------
# Training side (2026-10-27): pretrain.py's hook, the stage-1 freeze and the handoff.
# --------------------------------------------------------------------------
#
# A run names its init in one config block (conf/pretrain_vision/eo_terramesh/arm_b10c{v,s}_*):
#
#   pretrain_init:
#     conversion: <dir>         the verified converted init; provenance in both stages
#     variant: clevr|shuffled   must be the conversion's
#     freeze_body: true|false   true = stage 1: only the embedding table trains
#     from_stage1: <ckpt>|null  stage 2: stage 1's FINAL checkpoint, handoff checked
#
# Stage 1 is (true, null) and stage 2 is (false, <ckpt>). B10cV0, the arm without stage 1,
# would be (false, null). (true, <ckpt>) is refused.
# ⚠ Stage 2's configs say `from_stage1: ???`. A forgotten override must not turn into
# B10cV0, and OmegaConf's `.get()` returns None for a MISSING value (measured 2026-10-27).
# So read_block() converts the block and refuses "???" by name; never read it with `.get`.

BLOCK_KEYS = ("conversion", "variant", "freeze_body", "from_stage1")
RECORD = "pretrain_init.json"          # written by pretrain.py into the run directory
#: The fields that must agree between a run's record and its config on --resume, and
#: between stage 1's record and stage 2's config at the handoff.
RECORD_IDENTITY = ("conversion_weights_sha256", "variant", "freeze_body", "from_stage1")
_OC_ENV = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


def read_block(block, env: Optional[Mapping[str, str]] = None) -> Dict:
    """The `pretrain_init` block as {conversion, variant, freeze_body, from_stage1}. Raises ValueError.

    `block` is the run's DictConfig node (pretrain.py), or the plain unresolved dict that
    preflight composes, in which case `${oc.env:KEY,default}` is resolved against `env`.
    """
    from omegaconf import OmegaConf
    if OmegaConf.is_config(block):
        block = OmegaConf.to_container(block, resolve=True)      # MISSING stays "???"
    extra = sorted(set(block) - set(BLOCK_KEYS))
    if extra:
        raise ValueError(f"pretrain_init has unknown keys {extra} (known: {list(BLOCK_KEYS)})")
    out = {}
    for k in BLOCK_KEYS:
        if k not in block:
            raise ValueError(f"pretrain_init.{k} is not set")
        v = block[k]
        if v == "???":
            raise ValueError(f"pretrain_init.{k} is ??? (mandatory): pass "
                             f"pretrain_init.{k}=<value> on the command line"
                             + (" -- stage 1's final checkpoint" if k == "from_stage1" else ""))
        if isinstance(v, str) and env is not None:
            def sub(m):
                if m[1] in env:
                    return env[m[1]]
                if m[2] is None:
                    raise ValueError(f"pretrain_init.{k}: ${{oc.env:{m[1]}}} is unset")
                return m[2]
            v = _OC_ENV.sub(sub, v)
        out[k] = v
    if out["variant"] not in VARIANTS:
        raise ValueError(f"pretrain_init.variant {out['variant']!r} is not one of {VARIANTS}")
    if not isinstance(out["freeze_body"], bool):
        raise ValueError(f"pretrain_init.freeze_body must be true or false, got {out['freeze_body']!r}")
    if out["freeze_body"] and out["from_stage1"] is not None:
        raise ValueError("pretrain_init: freeze_body with from_stage1. Stage 1 freezes the body and "
                         "starts from the conversion; from_stage1 is stage 2's")
    out["conversion"] = Path(str(out["conversion"]))
    out["from_stage1"] = None if out["from_stage1"] is None else Path(str(out["from_stage1"]))
    return out


def check_arm(cfg) -> None:
    """Vanilla only, and not on top of a fine-tune's init. Raises ValueError."""
    if (cfg.get("recursive") or {}).get("enable") or (cfg.get("mor") or {}).get("enable"):
        raise ValueError("pretrain_init is for the vanilla arm only: the CLEVR body has 29 "
                         "distinct layers, and a recursive arm would tie them")
    if (cfg.get("finetune") or {}).get("init_from"):
        raise ValueError("pretrain_init and finetune.init_from do not combine")


def _state(d: Path) -> Dict[str, torch.Tensor]:
    from eo.finetune.init import _load_state, _weights
    return _load_state(_weights(d))


def freeze_body(model) -> int:
    """Stage 1: only the embedding table (tied to lm_head) trains. Returns the trainable count.

    Must run before the Trainer exists. HF builds the optimizer from the parameters that
    require grad at that moment, so a frozen tensor gets no Adam state and no weight decay.
    """
    for n, p in model.named_parameters():          # tied: lm_head.weight is not listed again
        p.requires_grad_(n == EMBED_KEY)
    n = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    want = int(model.config.vocab_size) * int(model.config.hidden_size)
    if n != want:
        raise RuntimeError(f"stage-1 freeze left {n:,} trainable parameters, expected the "
                           f"embedding table alone ({want:,})")
    return n


def check_handoff(stage1_ckpt: Path, conversion: Path, variant: str, conv_doc: Dict) -> Dict:
    """Stage 2 starts from stage 1's final weights. Raises RuntimeError unless:

      * the stage-1 run's record names this conversion (by weights hash) and this variant,
        with the body frozen and no from_stage1 of its own;
      * the checkpoint is that run's last (trainer_state: global_step == max_steps > 0);
      * its body equals the conversion's bit for bit, with the same key set (the freeze held);
      * its embedding table is finite and differs from the conversion's fresh one (stage 1
        trained it).
    """
    rec_path = stage1_ckpt.parent / RECORD
    if not rec_path.is_file():
        raise RuntimeError(f"{rec_path} missing: {stage1_ckpt} is not from a pretrain_init run")
    rec = json.loads(rec_path.read_text())
    want = {"conversion_weights_sha256": conv_doc["weights_sha256"], "variant": variant,
            "freeze_body": True, "from_stage1": None}
    bad = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
    if bad:
        raise RuntimeError(f"{rec_path} is not a stage 1 of this init: {bad} (want {want})")
    ts_path = stage1_ckpt / "trainer_state.json"
    if not ts_path.is_file():
        raise RuntimeError(f"{ts_path} missing")
    ts = json.loads(ts_path.read_text())
    step, last = int(ts.get("global_step") or 0), int(ts.get("max_steps") or -1)
    if step <= 0 or step != last:
        raise RuntimeError(f"{stage1_ckpt.name} is step {step} of {last}: stage 2 starts from "
                           f"stage 1's final checkpoint")
    s1, s0 = _state(stage1_ckpt), _state(conversion)
    if set(s1) != set(s0):
        raise RuntimeError(f"{stage1_ckpt}: key set differs from the conversion's "
                           f"({sorted(set(s1) ^ set(s0))[:4]}...)")
    body = [k for k in s0 if k not in EMBED_KEYS]
    moved = [k for k in body if not torch.equal(s1[k], s0[k])]
    if moved:
        dmax = max(float((s1[k].float() - s0[k].float()).abs().max()) for k in moved)
        raise RuntimeError(f"{stage1_ckpt}: {len(moved)} of {len(body)} body tensors differ from "
                           f"the conversion (max |d| {dmax:.3g}), e.g. {moved[0]}: the freeze did not hold")
    t1, t0 = s1[EMBED_KEY].float(), s0[EMBED_KEY].float()
    if not bool(torch.isfinite(t1).all()):
        raise RuntimeError(f"{stage1_ckpt}: the embedding table has non-finite entries")
    d = (t1 - t0).abs()
    if float(d.max()) == 0.0:
        raise RuntimeError(f"{stage1_ckpt}: the embedding table is the conversion's, unchanged: "
                           f"stage 1 did not train it")
    return {"stage1_checkpoint": str(stage1_ckpt), "global_step": step,
            "body_tensors_identical": len(body), "table_max_abs_change": float(d.max()),
            "table_rows_changed": int((d.amax(dim=1) > 0).sum()),
            "table_row_norm": [float(t0.norm(dim=1).mean()), float(t1.norm(dim=1).mean())]}


def pretrain_init(model, cfg) -> Dict:
    """pretrain.py's hook: load the verified init, then freeze the body for stage 1.

    Not a resume: HF builds a fresh optimizer and scheduler. On --resume this runs first and
    HF then loads the run's own checkpoint over it; the freeze (set here, before the
    optimizer exists) is what makes the saved optimizer's single param group fit again.
    """
    from eo.finetune.extend import assert_tied
    from eo.finetune.init import strict_load
    plan = read_block(cfg.pretrain_init)
    check_arm(cfg)
    vocab = int(model.config.vocab_size)
    doc = verify_conversion(plan["conversion"], plan["variant"], vocab)
    handoff, src = None, plan["conversion"]
    if plan["from_stage1"] is not None:
        handoff = check_handoff(plan["from_stage1"], plan["conversion"], plan["variant"], doc)
        src = plan["from_stage1"]
    load = strict_load(model, src)
    assert_tied(model)
    # strict_load proves the key set. This proves the values arrived, and that the model's
    # parameters are fp32 like the file (a bf16 model would round the body on load).
    sd, state = model.state_dict(), _state(src)
    off = [k for k, t in state.items() if sd[k].dtype != t.dtype or not torch.equal(sd[k], t)]
    if off:
        raise RuntimeError(f"after loading {src}: {len(off)} tensors differ from the file "
                           f"(model dtype {sd[off[0]].dtype}, file {state[off[0]].dtype}), e.g. {off[0]}")
    if plan["freeze_body"]:
        trainable = freeze_body(model)
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = int(sum(p.numel() for p in model.parameters()))
    return {"conversion": str(plan["conversion"]), "conversion_weights_sha256": doc["weights_sha256"],
            "variant": plan["variant"], "freeze_body": plan["freeze_body"],
            "from_stage1": None if plan["from_stage1"] is None else str(plan["from_stage1"]),
            "loaded_from": str(src), "load": load, "handoff": handoff,
            "source": {k: doc["source"].get(k) for k in ("repo", "revision", "weights_sha256", "global_step")},
            "trainable_params": trainable,
            "total_params": int(sum(p.numel() for p in model.parameters()))}


def check_record(run_dir: Path, report: Dict) -> Optional[Dict]:
    """The run directory's existing record, if any, must name the same init. Raises RuntimeError.

    A resume re-runs the hook and then loads the run's checkpoint over it. Resuming a stage-1
    run under the stage-2 config, or a CLEVR run under its shuffled control, would load
    cleanly and mean nothing.
    """
    p = Path(run_dir) / RECORD
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    bad = {k: (rec.get(k), report.get(k)) for k in RECORD_IDENTITY if rec.get(k) != report.get(k)}
    if bad:
        raise RuntimeError(f"{p} records another init than this run's config: "
                           f"{ {k: f'{a!r} -> {b!r}' for k, (a, b) in bad.items()} }")
    return rec


def write_record(run_dir: Path, report: Dict) -> Path:
    """Write pretrain_init.json, or check the one a resumed run already has."""
    p = Path(run_dir) / RECORD
    if check_record(run_dir, report) is None:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2))
    return p
