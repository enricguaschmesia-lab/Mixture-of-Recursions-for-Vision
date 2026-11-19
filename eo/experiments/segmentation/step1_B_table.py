"""Agent B's candidates in one table (A2 Step 1.2 reporting helper): every read through eo.maps.harness, from the
read JSONs step1_B_read wrote (the A100's read_best.json and the TITAN V's read_best_titanv.json).

    python -m eo.experiments.segmentation.step1_B_table [--root /data/enric/reports/a2_seg/step1/B/rcp] [run ...]
Per candidate and read set: the ceiling (mIoU) under the rule select_rule chose on train_sub, nearest beside, boundary
accuracy, distinct codes, code entropy (bits), the unseen-token share, tokens per map and codebook size; the bar; the
TITAN V's ceilings beside the A100's; the weights' sha256 and the monitor step best.pt was chosen at.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DS = ("pastis", "cloudsen12", "lulc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/enric/reports/a2_seg/step1/B/rcp")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()
    root = Path(a.root)
    runs = a.runs or sorted(p.name for p in root.iterdir() if (p / "read_best.json").exists())
    print("| candidate | tokens | codes | rule | PASTIS f4 | CloudSen12 val | LULC read | bar | boundary acc p/c/l | "
          "distinct codes p/c/l | entropy bits p/c/l | unseen % p/c/l | TITAN V p/c/l | CS native256 | best step |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        d = root / run
        r = json.loads((d / "read_best.json").read_text())
        tv = json.loads((d / "read_best_titanv.json").read_text()) if (d / "read_best_titanv.json").exists() else None
        rd, t = r["read"], r["tokenizer"]
        cur = json.loads((d / "curves.json").read_text()) if (d / "curves.json").exists() else {}
        done = json.loads((d / "done.json").read_text()) if (d / "done.json").exists() else {}
        best_step = done.get("best", {}).get("step")
        f = lambda k, g: "/".join(g(rd[x]) for x in DS)
        tvs = "/".join(f"{tv['read'][x]['ceiling']['mIoU']:.4f}" for x in DS) if tv else "-"
        n256 = f"{r['cloudsen12_native256']['ceiling']['mIoU']:.4f}" if "cloudsen12_native256" in r else "-"
        print(f"| {run} | {t['tokens']} | {t['n_codes']} | {r['rule']['rule']} | "
              + " | ".join(f"{rd[x]['ceiling']['mIoU']:.4f}" for x in DS)
              + f" | {'pass' if r['bar']['passes'] else 'fail'} | "
              + f(0, lambda v: f"{v['ceiling']['boundary_acc']:.3f}") + " | "
              + f(0, lambda v: str(v['codes']['read_distinct'])) + " | "
              + f(0, lambda v: f"{v['codes']['read_entropy_bits']:.2f}") + " | "
              + f(0, lambda v: f"{100 * v['codes']['unseen_token_share']:.3f}") + f" | {tvs} | {n256} | {best_step} |")
    print()
    print("rule 3 (coordinator, 2026-11-18): a candidate's ceiling is the mean of its three ceilings; a tie is two "
          "means under 0.1 point apart (then the smaller codebook, then the lower mean entropy)")
    print("| candidate | tokens | codes | passes | mean ceiling | mean entropy (bits) | PASTIS unseen % |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for run in runs:
        r = json.loads((root / run / "read_best.json").read_text())
        rd = r["read"]
        rows.append((run, r["tokenizer"]["tokens"], r["tokenizer"]["n_codes"], r["bar"]["passes"],
                     sum(rd[x]["ceiling"]["mIoU"] for x in DS) / 3,
                     sum(rd[x]["codes"]["read_entropy_bits"] for x in DS) / 3,
                     100 * rd["pastis"]["codes"]["unseen_token_share"]))
    for row in sorted(rows, key=lambda v: -v[4]):
        print(f"| {row[0]} | {row[1]} | {row[2]} | {'pass' if row[3] else 'fail'} | {row[4]:.4f} | {row[5]:.2f} | "
              f"{row[6]:.2f} |")
    print()
    for run in runs:
        d = root / run
        r = json.loads((d / "read_best.json").read_text())
        t = r["tokenizer"]
        print(f"{run}: weights {d / 'best.pt'} sha256 {t['sha256']['weights']}; palette sha256 {r['palette_sha256']}; "
              f"A100 read {d / 'read_best.json'}; TITAN V read "
              f"{d / 'read_best_titanv.json' if (d / 'read_best_titanv.json').exists() else '-'}")


if __name__ == "__main__":
    main()
