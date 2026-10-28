#!/usr/bin/env python
"""Turn SmolLM-135M into TerraMesh init checkpoints (`.venv`, CPU only). The LLM counterpart of
convert_clevr_init.py (2026-10-28; design: eo/train/smollm_init.py).

    CUDA_VISIBLE_DEVICES= .venv/bin/python -I eo/scripts/convert_smollm_init.py \\
        --source /data/enric/weights/smollm_135m/1d461723eec654e65efdc40cf49301c89c0c92f4 \\
        --arm-config eo_terramesh/arm_b10c_vanilla \\
        --out /data/enric/weights/smollm_init/b10c_v1

Writes, under --out:
  smollm_body/           SmolLM's body (29 of 30 layers) + B10c's seeded fresh embedding,
                         model.safetensors + conversion.json
  smollm_shuffled_body/  the same, every body tensor permuted (the control arm's init)

Each check is paired with a control that must FAIL it:
  architecture      SmolLM's config == the built B10c model's, but depth and vocab
  body identity     smollm body == SmolLM's layers renumbered past the dropped one
                                       <- the shuffled body; an untrained body;
                                          another layer dropped
  function          final hidden states == transformers' LlamaForCausalLM with the layer cut
                                       <- the shuffled body; another layer dropped;
                                          identical tensors at rope_theta x10 (what the
                                          tensor check misses); the reference with all 30 layers
  source is an LM   text loss of SmolLM, full and with the layer cut, far below uniform
                                       <- the shuffled body inside SmolLM
  body permutation  shuffled body == the smollm body, permuted
                                       <- the smollm body (unmoved); an untrained body
  fresh embedding   == B10c's build at --seed
                                       <- a build at another seed; SmolLM's text rows
                                          in the first EO slots
  strict load       the saved dir loads strictly into a B10c build at another seed, and every
                    check above holds again on what was loaded
                                       <- the saved state minus one body tensor
Nothing is written unless every in-memory check passes and every control fires.
conversion.json is written last: `clevr_init.verify_conversion` refuses a directory without it.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/enric/hf")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from eo.data import eo_vocab as EV  # noqa: E402
from eo.finetune import extend as X  # noqa: E402
from eo.finetune.init import strict_load  # noqa: E402
from eo.scripts.convert_clevr_init import build, fired, rope_control  # noqa: E402
from eo.train import clevr_init as C  # noqa: E402
from eo.train import smollm_init as S  # noqa: E402

VARIANTS = C.SOURCES["smollm"]
PRETRAINED, SHUFFLED = VARIANTS
DROPPED = "model.layers.13.mlp.down_proj.weight"   # the strict-load control's missing tensor
TEXT = ("LICENSE", "README.md")                     # public English text in this repo
LM_LOSS_MAX = 4.0         # SmolLM on English prose: ~2.4-3.4 nats measured; uniform is ln 49,152 = 10.8
CUT_LOSS_MAX_RISE = 0.5   # the cut model may lose a little, not its competence


def git_state():
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--",
                            "eo/train/clevr_init.py", "eo/train/smollm_init.py",
                            "eo/scripts/convert_smollm_init.py", "eo/scripts/convert_clevr_init.py"],
                           capture_output=True, text=True).stdout.strip()
    return commit, bool(dirty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_b10c_vanilla")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42, help="B10c's seed: the fresh embedding's build")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--drop-layer", type=int, default=S.DROPPED_LAYER)
    a = ap.parse_args()
    if a.drop_layer is None:
        sys.exit("no layer to drop: pass --drop-layer or set smollm_init.DROPPED_LAYER")
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"{a.out} exists and is not empty")
    other = 29 if a.drop_layer != 29 else 28        # the control's wrong layer

    print(f"loading {a.source}")
    src = S.load_smollm(a.source)
    state = src["state"]
    body = S.mapped_state(state, a.drop_layer)

    target, cfg = build(a.arm_config, a.seed)
    if cfg.recursive.get("enable") or cfg.get("mor", {}).get("enable"):
        sys.exit(f"{a.arm_config} is not a vanilla arm: the body only fits 29 unshared layers")
    X.assert_tied(target)
    vocab = int(target.config.vocab_size)
    if vocab != EV.TOTAL_VOCAB_SIZE:
        sys.exit(f"built at vocab {vocab}, eo_vocab says {EV.TOTAL_VOCAB_SIZE}")
    arch = S.check_architecture(src["config"], target.config)
    if not arch["passed"]:
        sys.exit(f"SmolLM's architecture differs from {a.arm_config}: {arch['differ']}")
    if len(body) != len(C.body_keys(target.state_dict())):
        sys.exit(f"mapped body has {len(body)} tensors, the model {len(C.body_keys(target.state_dict()))}")
    fresh = target.model.embed_tokens.weight.detach().clone()
    init_std = float(target.config.initializer_range)

    again, _ = build(a.arm_config, a.seed)
    reproducible = C.check_fresh_embedding(again.model.embed_tokens.weight, fresh, init_std)
    del again
    scratch, _ = build(a.arm_config, a.seed + 1)      # an untrained body, and a table at another seed
    ctl_other_seed = fired(C.check_fresh_embedding(scratch.model.embed_tokens.weight, fresh, init_std))
    text_rows = fresh.clone()
    n_text = state[C.EMBED_KEY].shape[0]
    text_rows[:n_text] = state[C.EMBED_KEY].to(text_rows.dtype)
    ctl_text_rows = fired(C.check_fresh_embedding(text_rows, fresh, init_std))
    del text_rows

    models = {}
    for v in VARIANTS:
        m = copy.deepcopy(target)
        C.install_body(m, C.body_state(body, v, a.shuffle_seed))
        X.assert_tied(m)
        models[v] = m.eval()
    wrong = copy.deepcopy(target)
    C.install_body(wrong, S.mapped_state(state, other))
    wrong.eval()
    del target

    ref = S.reference(a.source, a.drop_layer)
    ids = S.text_ids(a.source, [REPO / p for p in TEXT], seq_len=1024, n_seq=2)
    with torch.no_grad():
        x = ref.model.embed_tokens(ids)               # SmolLM's own embedding of real text
    pos = torch.arange(ids.shape[1]).unsqueeze(0).expand(ids.shape[0], -1)

    checks = {v: {"architecture": arch} for v in VARIANTS}
    controls = {v: {} for v in VARIANTS}

    print(f"body identity / function ({PRETRAINED}), layer {a.drop_layer} dropped")
    checks[PRETRAINED]["body_identity"] = C.check_body_identity(models[PRETRAINED], body)
    controls[PRETRAINED]["body_identity:shuffled_body"] = fired(C.check_body_identity(models[SHUFFLED], body))
    controls[PRETRAINED]["body_identity:untrained_body"] = fired(C.check_body_identity(scratch, body))
    controls[PRETRAINED][f"body_identity:layer_{other}_dropped"] = fired(C.check_body_identity(wrong, body))
    checks[PRETRAINED]["function"] = C.check_function(ref, models[PRETRAINED], x, pos)
    controls[PRETRAINED]["function:shuffled_body"] = fired(C.check_function(ref, models[SHUFFLED], x, pos))
    controls[PRETRAINED][f"function:layer_{other}_dropped"] = fired(C.check_function(ref, wrong, x, pos))
    rope = rope_control(models[PRETRAINED])
    rc = fired(C.check_function(ref, rope, x, pos))
    rc["tensor_check_passed_anyway"] = C.check_body_identity(rope, body)["passed"]
    controls[PRETRAINED]["function:rope_theta_x10"] = rc
    del rope, wrong
    full = S.reference(a.source, None)
    controls[PRETRAINED]["function:reference_all_30_layers"] = fired(C.check_function(full, models[PRETRAINED], x, pos))

    print("source is an LM (text loss)")
    lm_ids = S.text_ids(a.source, [REPO / p for p in TEXT], seq_len=1024, n_seq=4)
    loss_full, loss_cut = S.text_loss(full, lm_ids), S.text_loss(ref, lm_ids)
    shuf_lm = copy.deepcopy(ref)
    C.install_body(shuf_lm, C.body_state(body, SHUFFLED, a.shuffle_seed))
    loss_shuf = S.text_loss(shuf_lm, lm_ids)
    del shuf_lm, full
    lm = {"text": list(TEXT), "tokens": int(lm_ids.numel()), "loss_all_30_layers": loss_full,
          f"loss_layer_{a.drop_layer}_dropped": loss_cut, "uniform": float(torch.log(torch.tensor(float(n_text)))),
          "passed": loss_full < LM_LOSS_MAX and loss_cut < loss_full + CUT_LOSS_MAX_RISE}
    lm_ctl = {"loss_shuffled_body": loss_shuf, "fired": not (loss_shuf < LM_LOSS_MAX)}
    for v in VARIANTS:
        checks[v]["source_is_an_lm"] = lm
        controls[v]["source_is_an_lm:shuffled_body"] = lm_ctl
    print(f"  text loss: all 30 layers {loss_full:.3f}, layer {a.drop_layer} cut {loss_cut:.3f}, "
          f"shuffled body {loss_shuf:.3f} (uniform {lm['uniform']:.2f})")

    print(f"body permutation ({SHUFFLED})")
    checks[SHUFFLED]["body_permutation"] = C.check_body_permutation(models[SHUFFLED], body)
    controls[SHUFFLED]["body_permutation:smollm_body"] = fired(C.check_body_permutation(models[PRETRAINED], body))
    controls[SHUFFLED]["body_permutation:untrained_body"] = fired(C.check_body_permutation(scratch, body))
    del scratch

    for v in VARIANTS:
        checks[v]["fresh_embedding"] = C.check_fresh_embedding(models[v].model.embed_tokens.weight, fresh, init_std)
        checks[v]["seeded_build_reproducible"] = reproducible
        controls[v]["fresh_embedding:other_seed"] = ctl_other_seed
        controls[v]["fresh_embedding:smollm_text_rows"] = ctl_text_rows

    def gate(stage):
        bad = [f"{v}.{n}" for v in VARIANTS for n, c in checks[v].items() if not c["passed"]]
        silent = [f"{v}.{n}" for v in VARIANTS for n, c in controls[v].items() if not c["fired"]]
        for v in VARIANTS:
            for n, c in checks[v].items():
                print(f"  {v:15s} check   {n:42s} {'PASS' if c['passed'] else 'FAIL'}")
            for n, c in controls[v].items():
                print(f"  {v:15s} control {n:42s} {'fired' if c['fired'] else 'DID NOT FIRE'}")
        if bad or silent:
            sys.exit(f"{stage}: checks failed {bad}, controls silent {silent}; no conversion.json written")

    gate("in-memory checks")

    # --- write, then load back strictly --------------------------------------------------
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = {v: a.out / f"{v}_body" for v in VARIANTS}
    for v, d in dirs.items():
        models[v].save_pretrained(d, safe_serialization=True)

    print("strict load of the saved weights")
    loader, _ = build(a.arm_config, a.seed + 2)     # another table: a skipped tensor would show
    for v, d in dirs.items():
        lm_ = copy.deepcopy(loader)
        load = strict_load(lm_, d)
        X.assert_tied(lm_)
        b = (C.check_body_identity(lm_, body) if v == PRETRAINED
             else C.check_body_permutation(lm_, body))
        if v == SHUFFLED:      # the permutation must be the one checked in memory, not merely a permutation
            b["equal_to_in_memory"] = all(
                torch.equal(lm_.state_dict()[k], models[v].state_dict()[k]) for k in C.body_keys(body))
            b["passed"] = b["passed"] and b["equal_to_in_memory"]
        emb = C.check_fresh_embedding(lm_.model.embed_tokens.weight, fresh, init_std)
        checks[v]["strict_load"] = {**load, "body": b, "embedding": emb,
                                    "passed": b["passed"] and emb["passed"]}
        del lm_

    broken = a.out / "_control_missing_tensor"
    broken.mkdir()
    from safetensors.torch import save_file
    sd = models[PRETRAINED].state_dict()
    save_file({k: t.contiguous().clone() for k, t in sd.items() if k not in (C.HEAD_KEY, DROPPED)},
              str(broken / "model.safetensors"), metadata={"format": "pt"})
    try:
        strict_load(copy.deepcopy(loader), broken)
        ctl_missing = {"refused": False, "fired": False}
    except RuntimeError as err:
        ctl_missing = {"refused": True, "dropped": DROPPED, "message": str(err)[:200], "fired": True}
    shutil.rmtree(broken)
    for v in VARIANTS:
        controls[v]["strict_load:missing_tensor"] = ctl_missing
    del loader

    gate("strict load")

    # --- conversion.json, last ------------------------------------------------------------
    commit, dirty = git_state()
    se = state[C.EMBED_KEY].float()
    for v, d in dirs.items():
        wf = d / "model.safetensors"
        doc = {
            "variant": v,
            "source": {"repo": S.SMOLLM_REPO, "revision": S.SMOLLM_REVISION, "path": str(a.source),
                       "weights_sha256": src["weights_sha256"], "global_step": None,
                       "name_or_path": src["config"].get("_name_or_path")},
            "arm_config": a.arm_config,
            "token_set": (cfg.get("multimodal") or {}).get("token_set", "original"),
            "vocab_size": vocab,
            "dropped_layer": a.drop_layer,
            "layer_map": S.layer_map(a.drop_layer),
            "carried_over": f"body (29 of 30 layers, layer {a.drop_layer} dropped, + final norm), " + (
                "as trained" if v == PRETRAINED else f"each tensor permuted, shuffle seed {a.shuffle_seed}"),
            "embedding": {"how": "B10c's own build (load_model_from_config), seeded; nothing from SmolLM",
                          "seed": a.seed, "init_std": init_std,
                          "std": float(fresh.std()), "row_norm_mean": float(fresh.norm(dim=1).mean()),
                          "smollm_std": float(se.std()), "smollm_row_norm_mean": float(se.norm(dim=1).mean())},
            "shuffle_seed": a.shuffle_seed if v == SHUFFLED else None,
            "checks": checks[v], "controls": controls[v],
            "weights_file": wf.name, "weights_sha256": C.sha256(wf),
            "fork_commit": commit, "code_uncommitted": dirty,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
        }
        (d / "conversion.json").write_text(json.dumps(doc, indent=2) + "\n")
        C.verify_conversion(d, v, vocab)
        print(f"saved {d} ({doc['weights_sha256'][:12]})")


if __name__ == "__main__":
    main()
