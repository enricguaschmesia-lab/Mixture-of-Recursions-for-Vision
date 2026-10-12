"""Where does BEN fine-tuning's forgetting go? (Phase 6 Step 6, a diagnostic; `.venv`, CPU)

eval_forgetting.py shows TerraMesh held-out CE rising after the BEN fine-tunes, and the per-
modality slot mass falling for S1GRD and Coords. Two very different causes:

  * LEAKAGE: the model now puts probability on the 21 new BEN ids in TerraMesh contexts. This is
    removable at inference by masking those ids (slot-masked generation already does);
  * DRIFT: the distribution among the right modality's own codes got worse.

Per modality, on a stratified subset of held-out rows (both corpora, so S1GRD is present):
  ce          as eval_forgetting
  ben_mass    softmax mass on ids >= 87,556
  ce_noben    CE with the BEN ids masked out (renormalized over the 87,556 pretraining ids)
If ce_noben returns to the base value, the forgetting is leakage.

    python eo/experiments/ben_forgetting_leak.py --rows 48 --out /data/enric/reports/ben/forgetting/leak.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eo.data import ben_vocab as BV  # noqa: E402
from eo.data.eo_vocab import ID_TO_MODALITY  # noqa: E402
from eo.eval import teacher_forced as TF  # noqa: E402
from eo.finetune.init import build_finetuned  # noqa: E402
from eo.generate import conditional as G  # noqa: E402

EXT = "/data/enric/runs/pretrain/phase6/base/b10_ext_ckpt3300"
RUNS = "/data/enric/runs/pretrain/phase6"


@torch.no_grad()
def measure(model, batches):
    acc = {}
    for b in batches:
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                       position_ids=b["position_ids"]).logits[:, :-1].float()
        lab, mid = b["labels"][:, 1:], b["modality_ids"][:, 1:]
        valid = lab.ne(-100) & mid.gt(0)
        lp = F.log_softmax(logits, -1)
        lp_old = F.log_softmax(logits[..., :BV.BASE], -1)
        safe = lab.clamp_min(0)
        ce = -lp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        ce_old = -lp_old.gather(-1, safe.clamp_max(BV.BASE - 1).unsqueeze(-1)).squeeze(-1)
        ben = lp[..., BV.BASE:].exp().sum(-1)
        for m_id, name in ID_TO_MODALITY.items():
            sel = valid & mid.eq(m_id)
            if sel.any():
                a = acc.setdefault(name, [0, 0.0, 0.0, 0.0])
                a[0] += int(sel.sum()); a[1] += float(ce[sel].sum())
                a[2] += float(ce_old[sel].sum()); a[3] += float(ben[sel].sum())
    return {n: {"n": v[0], "ce": v[1] / v[0], "ce_noben": v[2] / v[0], "ben_mass": v[3] / v[0]}
            for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    _, cfg = G.build_model("eo_terramesh/arm_b10_vanilla", None, device="cpu")
    ds, _ = TF.eval_loader(cfg, num_workers=0)
    from eo.data.eval_split import load_row_table
    corpus = load_row_table(ds.root_dir).corpus.values
    pos = []
    for c in ("majortom", "ssl4eos12"):
        cand = [i for i, r in enumerate(ds.rows) if corpus[r] == c]
        pos += cand[:: max(1, len(cand) // (a.rows // 2))][: a.rows // 2]
    items = [ds[i] for i in pos]
    batches = [{k: torch.stack([it[k] for it in items[j:j + 4]]) for k in items[0]} for j in range(0, len(items), 4)]
    out = {"rows": [int(ds.rows[i]) for i in pos]}
    ov = [f"finetune.init_from={EXT}"]
    specs = [("extended_base", "eo_ben/b10_full", None),
             ("b10_full/checkpoint-1461", "eo_ben/b10_full", f"{RUNS}/b10full_20261012_214747/checkpoint-1461"),
             ("b10_lora/checkpoint-1461", "eo_ben/b10_lora", f"{RUNS}/b10lora_20261012_221451/checkpoint-1461")]
    for name, cfg_name, ck in specs:
        model, _ = build_finetuned(cfg_name, ck, device="cpu", overrides=ov)
        model.float()
        out[name] = measure(model, batches)
        print(name, "  ".join(f"{m}: ce {v['ce']:.3f} noBEN {v['ce_noben']:.3f} BENmass {v['ben_mass']:.4f}"
                              for m, v in out[name].items()), flush=True)
        del model
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
