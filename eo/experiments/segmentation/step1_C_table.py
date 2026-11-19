"""Tabulate agent C's candidates from the harness's own JSONs (A2 Step 1). It reads results; it scores nothing.

    python -m eo.experiments.segmentation.step1_C_table <read_dir> [<read_dir> ...]

Each <read_dir> is a run's read/ directory, written by step1_C_read: read_<dataset>.json (harness.score via
read_candidate), selection.json (harness.select_rule), summary_controls.json (the controls), and optionally
read_cloudsen12_native256.json. Prints, per candidate: Step 1.2's numbers per dataset, the bar, the controls, and
the freezing record (weights, sha256, palette, rule, input size, tokens, loader).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATASETS = ("pastis", "cloudsen12", "lulc")
BAR = {"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def one(read_dir: Path) -> None:
    reads = {d: load(read_dir / f"read_{d}.json") for d in DATASETS}
    if not all(reads.values()):
        print(f"## {read_dir}: read incomplete ({[d for d, r in reads.items() if not r]} missing)\n")
        return
    tok = reads["pastis"]["tokenizer"]
    sel = load(read_dir / "selection.json")
    print(f"## {Path(tok['weights']).name}  ({read_dir})")
    print(f"- weights `{tok['weights']}`, sha256 `{tok['sha256']['weights']}`; trained step {tok['meta'].get('step')}, "
          f"monitor mean {tok['meta'].get('mean'):.4f}")
    print(f"- input {tok['input_size']} px, grid {tok['grid'][0]} x {tok['grid'][1]} = {tok['tokens']} tokens, "
          f"FSQ {tok['fsq_levels']} = {tok['n_codes']} codes, {tok['params']:,} parameters, output {tok['output']}, "
          f"batch {tok['batch']}, read on {tok['gpu']}")
    print(f"- palette sha256 `{reads['pastis']['palette_sha256']}`, harness {reads['pastis']['harness']}, "
          f"rule {json.dumps(reads['pastis']['rule'])}")
    if sel:
        print(f"- rule chosen by harness.select_rule on train_sub: mean {json.dumps({k: round(v, 4) for k, v in sel['mean'].items()})}"
              f" -> {sel['chosen_key']}")
    print(f"- loader: `{tok['loader']}`")
    print()
    print("| dataset | ceiling (mIoU) | nearest | boundary acc | interior acc | pixel acc | distinct codes | entropy (bits) "
          "| unseen-token share | tokens/map | bar | JSON |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in DATASETS:
        r = reads[d]
        c, k = r["ceiling"], r["codes"]
        near = r["also"].get("nearest", {}).get("mIoU", c["mIoU"])
        print(f"| {d} | {c['mIoU']:.4f} | {near:.4f} | {c['boundary_acc']:.4f} | {c['interior_acc']:.4f} | "
              f"{c['pixel_acc']:.4f} | {k['read_distinct']} / {k['n_codes']} | {k['read_entropy_bits']:.3f} | "
              f"{k.get('unseen_token_share', float('nan')):.2e} | {r['tokens_per_map']} | "
              f"{'pass' if c['mIoU'] >= BAR[d] else 'FAIL'} (>= {BAR[d]}) | `{read_dir / f'read_{d}.json'}` |")
    n256 = load(read_dir / "read_cloudsen12_native256.json")
    if n256:
        print(f"| cloudsen12 native256 (beside) | {n256['ceiling']['mIoU']:.4f} | | {n256['ceiling']['boundary_acc']:.4f} | "
              f"| | {n256['codes']['read_distinct']} | {n256['codes']['read_entropy_bits']:.3f} | "
              f"{n256['codes'].get('unseen_token_share', float('nan')):.2e} | {n256['tokens_per_map']} | | "
              f"`{read_dir / 'read_cloudsen12_native256.json'}` |")
    print()
    ctl = load(read_dir / "summary_controls.json")
    if ctl:
        det = ctl["determinism"]
        print("- controls (`" + str(read_dir / "summary_controls.json") + "`):")
        print("  - determinism at batch " + str(tok["batch"]) + ": " + "; ".join(
            f"{d} {v['inputs']} inputs codes identical {v['codes_identical']}, decode identical "
            f"{v['decoded_logits_identical']}, at batch 7 {v['codes_differing_at_batch_7']} codes differ"
            + (f", identical to the harness cache {v['identical_to_harness_cache']}" if 'identical_to_harness_cache' in v else "")
            for d, v in det.items()))
        print("  - transposed inputs, scored against the maps as they are: " + ", ".join(
            f"{d} {ctl['transposed'][d]:.4f}" for d in DATASETS))
        print("  - untrained tokenizer of the same design: " + ", ".join(
            f"{d} {ctl['untrained'][d]['mIoU']:.4f}" for d in DATASETS))
        print("  - shuffled training assignments equal to the fixed read assignment: " + ", ".join(
            f"{d} {v['equal_to_fixed']} of {v['draws']} (p = {v['p_equal_exact']:.1e})" for d, v in ctl["assignment"].items()))
        pdg = ctl.get("palette_diag")
        if pdg:
            print("  - monitor read under random injective assignments (fixed, then 3 random): " + "; ".join(
                f"{d} {v['fixed']:.4f} / " + " ".join(f"{x['mIoU']:.4f}" for x in v["random"]) for d, v in pdg.items()))
    print()


def curves_all(png: Path, run_dirs: list[Path]) -> None:
    """One figure: each run's monitor mIoU (EMA weights, the harness's nearest read) per dataset and their mean,
    against steps and against maps seen (steps x maps per step), from each run's history.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for rd in run_dirs:
        hist = json.loads((rd / "history.json").read_text())[1:]
        cfg = json.loads((rd / "config.json").read_text())
        per_step = sum(int(kv.split(":")[1]) for kv in cfg["args"]["mix"].split(","))
        label = f"{rd.name} ({cfg['gpu'].replace('NVIDIA ', '')[:14]}, {per_step}/step)"
        s = [h["step"] for h in hist]
        for j, key in enumerate(("pastis", "cloudsen12", "lulc", "mean")):
            y = [h["mean"] if key == "mean" else h["monitor"][key]["mIoU"] for h in hist]
            ax[0, j].plot(s, y, marker=".", label=label)
            ax[1, j].plot([v * per_step / 1e6 for v in s], y, marker=".", label=label)
    for j, key in enumerate(("pastis", "cloudsen12", "lulc", "mean")):
        for i in range(2):
            ax[i, j].set_title(f"monitor {key}")
            ax[i, j].set_xlabel("step" if i == 0 else "maps seen (millions)")
            ax[i, j].grid(alpha=0.3)
            if key != "mean":
                ax[i, j].axhline({"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}[key], color="grey", lw=0.6)
            ax[i, j].set_ylim(0.9 if key != "pastis" else 0.85, 1.0)
    ax[0, 3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    print("figure", png)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--curves-all":
        curves_all(Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]])
        return
    for p in sys.argv[1:]:
        one(Path(p))


if __name__ == "__main__":
    main()
