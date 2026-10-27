#!/usr/bin/env python
"""Turn the CLEVR vanilla checkpoint into TerraMesh init checkpoints (`.venv`, CPU only).

    CUDA_VISIBLE_DEVICES= .venv/bin/python -I eo/scripts/convert_clevr_init.py \\
        --source /data/enric/weights/clevr_vanilla_mahlia/a92e8b2a07d7cdd2932baf4dc874324013207a81 \\
        --arm-config eo_terramesh/arm_b10c_vanilla \\
        --out /data/enric/weights/clevr_init/b10c_v1

Writes, under --out (design and reasons: eo/train/clevr_init.py):
  clevr_body/      CLEVR's body + B10c's seeded fresh embedding, model.safetensors + conversion.json
  shuffled_body/   the same, every body tensor permuted (the control arm's init)
  clevr_rgb_rows/  CLEVR's 64,000 RGB rows + their BO/EO rows, bf16, + rgb_rows.json

Each check is paired with a control that must FAIL it:
  body identity     clevr body == CLEVR's, exactly       <- the shuffled body; an untrained body
  body permutation  shuffled body == CLEVR's, permuted   <- the CLEVR body (unmoved); an untrained body
  function          final hidden states == CLEVR model's <- the shuffled body; identical tensors at
                                                             rope_theta 1e5 (what the tensor check misses)
  fresh embedding   == B10c's build at --seed            <- a build at another seed; the overlap layout
                                                             (S2L2A ids take CLEVR's RGB rows)
  strict load       the saved dir loads strictly into a B10c build at another seed, and every
                    check above holds again on what was loaded
                                                         <- the saved state minus one body tensor
  RGB rows          file == CLEVR rows 0..63,999 + BO/EO <- the same slice shifted by one row
Nothing is written unless every in-memory check passes and every control fires.
conversion.json is written last, only once the strict load holds too:
`clevr_init.verify_conversion` refuses a directory without it.
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
from eo.generate import conditional as G  # noqa: E402
from eo.terramesh_tok.contract import CODEBOOK  # noqa: E402
from eo.train import clevr_init as C  # noqa: E402
from util.seeding import set_global_seed  # noqa: E402

SEQ_LEN = 520            # two CLEVR chunks (258 each) and then some
DROPPED = "model.layers.13.mlp.down_proj.weight"   # the strict-load control's missing tensor


def build(arm_config: str, seed: int):
    """B10c's model as pretrain.py builds it: seed, then load_model_from_config."""
    set_global_seed(seed)
    m, cfg = G.build_model(arm_config, None, device="cpu")
    return m.float().eval(), cfg


def fired(r: dict) -> dict:
    return {**r, "fired": not r["passed"]}


def rope_control(model):
    """Identical tensors in a model whose only difference is rope_theta (1e4 -> 1e5)."""
    cfg = copy.deepcopy(model.config)
    cfg.rope_theta = float(cfg.rope_theta) * 10
    m = type(model)._from_config(cfg, attn_implementation="sdpa", torch_dtype=torch.float32)
    m.load_state_dict(model.state_dict(), strict=True)
    return m.eval()


def git_state():
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--",
                            "eo/train/clevr_init.py", "eo/scripts/convert_clevr_init.py"],
                           capture_output=True, text=True).stdout.strip()
    return commit, bool(dirty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_b10c_vanilla")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42, help="B10c's seed: the fresh embedding's build")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"{a.out} exists and is not empty")

    print(f"loading {a.source}")
    src = C.load_clevr(a.source)
    state = src["state"]
    ref = C.clevr_reference(a.source, state)

    target, cfg = build(a.arm_config, a.seed)
    if cfg.recursive.get("enable") or cfg.get("mor", {}).get("enable"):
        sys.exit(f"{a.arm_config} is not a vanilla arm: the CLEVR body only fits the 29 unshared layers")
    X.assert_tied(target)
    vocab = int(target.config.vocab_size)
    if vocab != EV.TOTAL_VOCAB_SIZE:
        sys.exit(f"built at vocab {vocab}, eo_vocab says {EV.TOTAL_VOCAB_SIZE}")
    fresh = target.model.embed_tokens.weight.detach().clone()
    init_std = float(target.config.initializer_range)

    again, _ = build(a.arm_config, a.seed)
    reproducible = C.check_fresh_embedding(again.model.embed_tokens.weight, fresh, init_std)
    del again
    scratch, _ = build(a.arm_config, a.seed + 1)      # an untrained body, and a table at another seed
    ctl_other_seed = fired(C.check_fresh_embedding(scratch.model.embed_tokens.weight, fresh, init_std))
    ctl_overlap = fired(C.check_fresh_embedding(
        C.overlap_layout(fresh, state, CODEBOOK["S2L2A"]), fresh, init_std))

    models = {}
    for v in C.VARIANTS:
        m = copy.deepcopy(target)
        C.install_body(m, C.body_state(state, v, a.shuffle_seed))
        X.assert_tied(m)
        models[v] = m.eval()
    del target

    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, ref.config.vocab_size, (2, SEQ_LEN), generator=g)
    with torch.no_grad():
        x = ref.model.embed_tokens(ids)
    pos = torch.arange(SEQ_LEN).unsqueeze(0).expand(2, -1)

    checks = {v: {} for v in C.VARIANTS}
    controls = {v: {} for v in C.VARIANTS}

    print("body identity / function (clevr)")
    checks["clevr"]["body_identity"] = C.check_body_identity(models["clevr"], state)
    controls["clevr"]["body_identity:shuffled_body"] = fired(C.check_body_identity(models["shuffled"], state))
    controls["clevr"]["body_identity:untrained_body"] = fired(C.check_body_identity(scratch, state))
    checks["clevr"]["function"] = C.check_function(ref, models["clevr"], x, pos)
    controls["clevr"]["function:shuffled_body"] = fired(C.check_function(ref, models["shuffled"], x, pos))
    rope = rope_control(models["clevr"])
    rc = fired(C.check_function(ref, rope, x, pos))
    rc["tensor_check_passed_anyway"] = C.check_body_identity(rope, state)["passed"]
    controls["clevr"]["function:rope_theta_x10"] = rc
    del rope

    print("body permutation (shuffled)")
    checks["shuffled"]["body_permutation"] = C.check_body_permutation(models["shuffled"], state)
    controls["shuffled"]["body_permutation:clevr_body"] = fired(C.check_body_permutation(models["clevr"], state))
    controls["shuffled"]["body_permutation:untrained_body"] = fired(C.check_body_permutation(scratch, state))
    del scratch

    for v in C.VARIANTS:
        checks[v]["fresh_embedding"] = C.check_fresh_embedding(models[v].model.embed_tokens.weight, fresh, init_std)
        checks[v]["seeded_build_reproducible"] = reproducible
        controls[v]["fresh_embedding:other_seed"] = ctl_other_seed
        controls[v]["fresh_embedding:overlap_layout"] = ctl_overlap

    def gate(stage):
        bad = [f"{v}.{n}" for v in C.VARIANTS for n, c in checks[v].items() if not c["passed"]]
        silent = [f"{v}.{n}" for v in C.VARIANTS for n, c in controls[v].items() if not c["fired"]]
        for v in C.VARIANTS:
            for n, c in checks[v].items():
                print(f"  {v:8s} check   {n:42s} {'PASS' if c['passed'] else 'FAIL'}")
            for n, c in controls[v].items():
                print(f"  {v:8s} control {n:42s} {'fired' if c['fired'] else 'DID NOT FIRE'}")
        if bad or silent:
            sys.exit(f"{stage}: checks failed {bad}, controls silent {silent}; no conversion.json written")

    gate("in-memory checks")

    # --- write, then load back strictly --------------------------------------------------
    a.out.mkdir(parents=True, exist_ok=True)
    dirs = {v: a.out / f"{v}_body" for v in C.VARIANTS}
    for v, d in dirs.items():
        models[v].save_pretrained(d, safe_serialization=True)

    print("strict load of the saved weights")
    loader, _ = build(a.arm_config, a.seed + 2)     # another table: a skipped tensor would show
    for v, d in dirs.items():
        lm = copy.deepcopy(loader)
        load = strict_load(lm, d)
        X.assert_tied(lm)
        body = (C.check_body_identity(lm, state) if v == "clevr"
                else C.check_body_permutation(lm, state))
        if v == "shuffled":     # the permutation must be the one checked in memory, not merely a permutation
            body["equal_to_in_memory"] = all(
                torch.equal(lm.state_dict()[k], models[v].state_dict()[k]) for k in C.body_keys(state))
            body["passed"] = body["passed"] and body["equal_to_in_memory"]
        emb = C.check_fresh_embedding(lm.model.embed_tokens.weight, fresh, init_std)
        checks[v]["strict_load"] = {**load, "body": body, "embedding": emb,
                                    "passed": body["passed"] and emb["passed"]}
        del lm

    broken = a.out / "_control_missing_tensor"
    broken.mkdir()
    from safetensors.torch import save_file
    sd = models["clevr"].state_dict()
    save_file({k: t.contiguous().clone() for k, t in sd.items() if k not in (C.HEAD_KEY, DROPPED)},
              str(broken / "model.safetensors"), metadata={"format": "pt"})
    try:
        strict_load(copy.deepcopy(loader), broken)
        ctl_missing = {"refused": False, "fired": False}
    except RuntimeError as err:
        ctl_missing = {"refused": True, "dropped": DROPPED, "message": str(err)[:200], "fired": True}
    shutil.rmtree(broken)
    for v in C.VARIANTS:
        controls[v]["strict_load:missing_tensor"] = ctl_missing
    del loader

    gate("strict load")

    # --- the RGB rows ---------------------------------------------------------------------
    print("RGB rows")
    rgb_dir = a.out / "clevr_rgb_rows"
    rgb = C.save_rgb_rows(rgb_dir, state)
    back = C.load_rgb_rows(rgb_dir / rgb["file"], rgb["sha256"])
    e = state[C.EMBED_KEY]
    off, n = int(rgb["codebook_offset"]), int(rgb["codebook_size"])
    rgb_check = {"passed": bool(torch.equal(back["code_rows"], e[off:off + n])
                                and torch.equal(back["bo_row"], e[int(rgb["bo_id"])])
                                and torch.equal(back["eo_row"], e[int(rgb["eo_id"])]))}
    rgb_ctl = {"fired": not torch.equal(back["code_rows"], e[off + 1:off + n + 1])}
    rgb_doc = {**rgb, "check_equal_to_source_rows": rgb_check, "control_shifted_by_one": rgb_ctl,
               "row_norm_mean": float(back["code_rows"].float().norm(dim=1).mean()),
               "created": dt.datetime.now().isoformat(timespec="seconds")}
    (rgb_dir / "rgb_rows.json").write_text(json.dumps(rgb_doc, indent=2) + "\n")
    if not (rgb_check["passed"] and rgb_ctl["fired"]):
        sys.exit("RGB rows check failed (or its control did not fire)")

    # --- conversion.json, last ------------------------------------------------------------
    commit, dirty = git_state()
    ce = e.float()
    for v, d in dirs.items():
        wf = d / "model.safetensors"
        doc = {
            "variant": v,
            "source": {"repo": C.CLEVR_REPO, "revision": C.CLEVR_REVISION, "path": str(a.source),
                       "weights_sha256": src["weights_sha256"], "global_step": src["global_step"],
                       "max_steps": src["max_steps"], "lr_at_save": src["lr_at_save"],
                       "loss_at_save": src["loss_at_save"]},
            "arm_config": a.arm_config,
            "token_set": (cfg.get("multimodal") or {}).get("token_set", "original"),
            "vocab_size": vocab,
            "carried_over": "body (29 layers + final norm), " + (
                "as trained" if v == "clevr" else f"each tensor permuted, shuffle seed {a.shuffle_seed}"),
            "embedding": {"how": "B10c's own build (load_model_from_config), seeded; nothing from CLEVR",
                          "seed": a.seed, "init_std": init_std,
                          "std": float(fresh.std()), "row_norm_mean": float(fresh.norm(dim=1).mean()),
                          "clevr_std": float(ce.std()), "clevr_row_norm_mean": float(ce.norm(dim=1).mean())},
            "shuffle_seed": a.shuffle_seed if v == "shuffled" else None,
            "checks": checks[v], "controls": controls[v],
            "rgb_rows": str(rgb_dir / rgb["file"]),
            "weights_file": wf.name, "weights_sha256": C.sha256(wf),
            "fork_commit": commit, "code_uncommitted": dirty,
            "created": dt.datetime.now().isoformat(timespec="seconds"),
        }
        (d / "conversion.json").write_text(json.dumps(doc, indent=2) + "\n")
        C.verify_conversion(d, v, vocab)
        print(f"saved {d} ({doc['weights_sha256'][:12]})")
    print(f"saved {rgb_dir / rgb['file']} ({rgb['sha256'][:12]})")


if __name__ == "__main__":
    main()
