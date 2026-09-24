#!/usr/bin/env python
"""Conditional cross-modal generation from an EO checkpoint (Phase 3 D3.7).

Thin CLI. The design, the fixed prompt order and the off-slot reasoning all
live in eo/generate/conditional.py -- read that first.

    python eo/scripts/generate_eo.py --arm-config eo_terramesh/arm_a_mor \
        --checkpoint /data/enric/runs/pretrain/phase3/<run>/checkpoint-28000 \
        --target LULC --n-scenes 64 --out /data/enric/generations/<tag>

⚠ --checkpoint may be omitted. That gives an UNTRAINED model, whose off-slot
rate must land at chance (82.5% image / 95.0% LULC / 92.7% Coords) -- the
control that proves the metric measures anything at all.

⚠ --identity builds no model at all: it writes the GROUND-TRUTH target bodies
in the generation format, as if a model had produced them. Step 7's decode
path must then return ceiling == generated EXACTLY; anything else is a wiring
bug between here and the decoder. Build every new decode path against it.

⚠ Scenes are drawn corpus-STRATIFIED from the held-out split. The eval rows are
sorted majortom-first and majortom never carries S1GRD, so an unstratified head
would silently contain no S1GRD scene at all.

Runs in the repo .venv, not the `mor` env. Decoding ids to pixels is Step 7 and
a separate process.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from eo.data.eo_vocab import get_modality  # noqa: E402
from eo.data.eval_split import load_eval_rows, load_row_table  # noqa: E402
from eo.data.terramesh_token_dataset import TerraMeshTokenDataset  # noqa: E402
from eo.generate import conditional as C  # noqa: E402

ROOT = os.environ.get("TERRAMESH_TOK_ROOT", "/data/enric/data/TerraMesh/val")


def stratified_rows(n, target, dataset):
    """Corpus-stratified held-out rows that actually carry `target`."""
    rows = load_eval_rows(root_dir=ROOT)
    corpus = load_row_table(ROOT).corpus.values
    ok = [r for r in rows if target in dataset.present_modalities(r)]
    mt = [r for r in ok if corpus[r] == "majortom"]
    ss = [r for r in ok if corpus[r] == "ssl4eos12"]
    half = n // 2
    picked = mt[:half] + ss[: n - half]
    if len(picked) < n:                       # one corpus cannot supply its half
        picked = ok[:n]
    return sorted(int(r) for r in picked)


def write_identity(args, ds, rows) -> int:
    """The identity control: ground-truth bodies saved as the generation.

    Saved under slot_masked/ because that is the mode the decode path reads,
    and the rows are the same stratified_rows() the real generations use, so
    its ceiling is the ceiling of every run over the same --n-scenes.
    """
    if args.checkpoint:
        print("ERROR: --identity uses no model; drop --checkpoint", file=sys.stderr)
        return 1
    # Terminated with <EO_target> like a well-behaved generation, so the
    # control also exercises prepare_decode's EO stripping.
    eo = torch.tensor([get_modality(args.target).eo_id], dtype=torch.long)
    truths = [torch.cat([C.build_prompt(ds, r, args.target)["truth"].long(), eo]) for r in rows]
    out = Path(args.out or f"/data/enric/generations/identity_control_{args.target}") / "slot_masked"
    C.save(out, args.target, rows, truths,
           {"note": f"identity control: ground-truth {args.target} tokens fed back in as if generated"},
           {"synthetic": True, "checkpoint": None, "n_scenes": len(rows)})
    print(f"identity      : {len(truths)} ground-truth bodies\n  -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-config", default="eo_terramesh/arm_a_mor")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--target", default="LULC")
    ap.add_argument("--n-scenes", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: /data/enric/generations/<arm>_<target>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-cache", action="store_true", help="KV cache (S3.c, unvalidated)")
    ap.add_argument("--slot-masked-only", action="store_true")
    ap.add_argument("--identity", action="store_true",
                    help="write ground truth as if generated; no model, no GPU")
    args = ap.parse_args()

    ds = TerraMeshTokenDataset(root_dir=ROOT, max_length=1048, modality_order="fixed")
    rows = stratified_rows(args.n_scenes, args.target, ds)
    print(f"target        : {args.target}")
    print(f"scenes        : {len(rows)} held-out rows (corpus-stratified)")
    if args.identity:
        return write_identity(args, ds, rows)
    print(f"checkpoint    : {args.checkpoint or 'NONE -- untrained control'}")

    model, cfg = C.build_model(args.arm_config, args.checkpoint, device=args.device)
    built = [C.build_prompt(ds, r, args.target) for r in rows]
    prompts = [b["prompt"] for b in built]
    truths = [b["truth"] for b in built]
    print(f"context order : {built[0]['context']}")
    print(f"prompt length : {len(prompts[0])} tokens\n")

    modes = [("slot_masked", True)] if args.slot_masked_only else [
        ("unconstrained", False), ("slot_masked", True)]
    all_stats = {}
    for name, masked in modes:
        gen, t0 = [], time.perf_counter()
        for i in range(0, len(prompts), args.batch_size):
            gen += C.generate(
                model, prompts[i:i + args.batch_size], args.target, device=args.device,
                do_sample=not args.greedy, temperature=args.temperature, top_k=args.top_k,
                slot_masked=masked, use_cache=args.use_cache, seed=args.seed,
            )
        el = time.perf_counter() - t0
        st = C.score(gen, truths, args.target)
        st["seconds"] = round(el, 1)
        st["seconds_per_scene"] = round(el / max(len(gen), 1), 2)
        all_stats[name] = st
        print(f"[{name}] {el:.1f}s ({el/max(len(gen),1):.2f}s/scene)")
        print(f"  off-slot rate      {st['off_slot_rate']:.4f}   (chance {st['off_slot_chance']:.4f})")
        print(f"  stopped at exactly {st['stopped_at_exact_length']}/{st['n_scenes']} scenes"
              f"   never emitted EO: {st['never_emitted_eo']}")
        print(f"  codes used         {st['codes_used']}/{st['codebook_size']}"
              f"   (ground truth {st['codes_used_truth']})")
        print(f"  entropy nats       {st['entropy_nats']:.3f}   (ground truth {st['entropy_nats_truth']:.3f})")
        print(f"  token accuracy     {st['token_accuracy']:.4f}")

        out = Path(args.out or f"/data/enric/generations/{Path(args.arm_config).name}_{args.target}") / name
        C.save(out, args.target, rows, gen, st, {
            "arm_config": args.arm_config, "checkpoint": args.checkpoint,
            "mode": name, "seed": args.seed, "greedy": args.greedy,
            "use_cache": args.use_cache, "n_scenes": len(rows),
            "context_order": built[0]["context"],
        })
        print(f"  -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
