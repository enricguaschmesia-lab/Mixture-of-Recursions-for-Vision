#!/usr/bin/env python
"""A2 Step 7.0g, beside (descriptive, 2026-11-27, about 20:20): the stack's fold-4 mIoU against the number of votes, per
seed and pooled over E784's two seeds, from the stored class maps of 7.0f's and 7.0g's stacks. n samples per orientation
over the 8 orientations (8n votes); plain vote and frequency-corrected vote at alpha 0.5 (the alpha both seeds' cross-fits
chose at 64 votes; not tuned here). `.venv`, CPU.

    python eo/experiments/segmentation/step70g_votes_curve.py
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "/home/enric/code/Mixture-of-Recursions-for-Vision")
sys.path.insert(0, "/home/enric/code/Mixture-of-Recursions-for-Vision/eo/experiments/segmentation")
import step70e_read as RD, step70e_votes as VT
from step53_verdict import miou
prior = json.loads(RD.PRIOR.read_text())
w05 = VT.class_weights(prior, 0.5, 20, 19)
runs = {"s42": "p7fstack_20261127_080904", "s43": "p7gstack_20261127_133608"}
data = {}
for s, name in runs.items():
    maps, vj, _ = RD.load_maps(name)
    data[s] = (maps, vj)
sids = [vj["sample_id"][8 * i] for i in range(len(vj["k"]) // 8)]
assert sids == [data["s42"][1]["sample_id"][8 * i] for i in range(len(sids))]
labels = RD.labels_for(sids)
n = len(sids)
def fn_for(seeds, per):
    def fn(i):
        out = []
        for s in seeds:
            maps = data[s][0]
            out += [VT.to_k0(np.asarray(maps[8 * i + k][:per]), k) for k in range(8)]
        return np.concatenate(out)
    return fn
res = {}
t0 = time.time()
for seeds in (("s42",), ("s43",)):
    for per in (1, 2, 4, 8):
        fn = fn_for(seeds, per)
        p = miou(RD.cms_of(fn, n, labels).sum(0), 19) * 100
        c = miou(RD.cms_of(fn, n, labels, w05).sum(0), 19) * 100
        res[f"{'+'.join(seeds)} {8 * per}"] = (p, c)
        print(f"{'+'.join(seeds):8s} votes {8 * per * len(seeds):4d}: plain {p:6.2f}  corrected(a=0.5) {c:6.2f}  ({time.time() - t0:.0f} s)", flush=True)
for per in (4, 8):
    fn = fn_for(("s42", "s43"), per)
    p = miou(RD.cms_of(fn, n, labels).sum(0), 19) * 100
    c = miou(RD.cms_of(fn, n, labels, w05).sum(0), 19) * 100
    print(f"s42+s43  votes {16 * per:4d}: plain {p:6.2f}  corrected(a=0.5) {c:6.2f}  ({time.time() - t0:.0f} s)", flush=True)
