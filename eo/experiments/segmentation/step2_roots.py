"""A2 Step 2: the token roots' record (a2seg env, CPU): per task and artifact, rows present per split,
codes used, file sizes and sha256 of tokens.npy / present.npy / the index parquets.

    python eo/experiments/segmentation/step2_roots.py --out /data/enric/reports/a2_seg/step2/roots.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from eo.geobench import views as V  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for task, root in V.TOK_ROOT.items():
        idx = V.load(task)
        v, x = idx["views"], idx["inputs"]
        split_of_row = x.view.map(v.set_index("view").split).values
        r = {"root": str(root), "index": {k: {"rows": int(len(idx[k])), "sha256": sha(root / f"{k}.parquet")}
                                          for k in ("views", "inputs")}}
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            t, pr = d / "tokens.npy", d / "present.npy"
            if not t.exists():
                continue
            pres = np.load(pr)
            tok = np.load(t, mmap_mode="r")
            splits = split_of_row if d.name.startswith("S2L2A") else v.split.values
            e = {"shape": list(tok.shape), "dtype": str(tok.dtype), "bytes": t.stat().st_size,
                 "present": int(pres.sum()),
                 "present_by_split": {s: f"{int(pres[splits == s].sum())}/{int((splits == s).sum())}"
                                      for s in ("train", "val", "test")},
                 "sha256_tokens": sha(t), "sha256_present": sha(pr)}
            if d.name.startswith("S2L2A") and task == "pastis":
                e["present_by_frame"] = {f: f"{int(pres[x.frame.values == f].sum())}/{int((x.frame.values == f).sum())}"
                                         for f in ["p1"] + V.FRAMES_P7}
            meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
            e["gpu"] = meta.get("gpu")
            e["codes_used"] = meta.get("codes_used")
            res.setdefault(task, r)[d.name] = e
            print(task, d.name, e["present_by_split"], f"{e['bytes'] / 1e6:.0f} MB", e["sha256_tokens"][:16], e["gpu"], flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
