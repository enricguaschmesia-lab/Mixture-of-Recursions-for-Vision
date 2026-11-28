#!/usr/bin/env python
"""A2 Step 7.0e tests 2 and 3: each sample's native class map, and the votes (docs/plans/a2_segmentation.md, Step 7,
"7.0e"; `a2seg` env, the GTX Titan X).

    python eo/experiments/segmentation/step70e_votes.py decode --gen <gen dir> --out <dir> [--root <seg root>]
    python eo/experiments/segmentation/step70e_votes.py truth  --root /data/enric/data/geobench2_tta --out <dir>

  decode  codes.npy (V, K, G) of a val generation -> <out>/classmaps.npy (V, K, H, W) uint8: sample j of view v as
          eo/eval/seg.score makes it (the frozen arm's class scores, its rule), in the VIEW's frame; <out>/views.json
          (view, sample_id, k per row) and decode.json. For a CloudSen12 run a "view" row is a tile (its 9 crops
          stitched, as the scorer stitches them).
  truth   the TRUE map tokens of every view under --root, decoded the same way (K = 1): test 3's ceiling check.

The votes (pure; the read and the gate call them):
  plain_vote     eval/seg.vote itself: one-hot sum over the stack, argmax, ties to the lower class;
  weighted_vote  argmax_c w_c V_c, ties to the lower class (w = 1 is plain_vote);
  to_k0          a class map in view frame k brought back to k = 0 (the inverse of eo/geobench/s2.dihedral).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------- the votes (pure)
def to_k0(y: np.ndarray, k: int) -> np.ndarray:
    """Inverse of s2.dihedral(x, k) on the last two axes: dihedral = rot90(x, k % 4), then a flip of the last axis if
    k >= 4; so x = rot90(y, -(k % 4)) for k < 4 and rot90(flip(y), -(k % 4)) for k >= 4."""
    if not 0 <= k < 8:
        raise ValueError(k)
    z = y[..., ::-1] if k >= 4 else y
    return np.ascontiguousarray(np.rot90(z, -(k % 4), axes=(-2, -1)))


def plain_vote(stack, n_classes: int):
    """(M, N, H, W) class maps -> (N, H, W): eo/eval/seg.vote, on whatever device the stack is."""
    from eo.eval import seg as SEG
    return SEG.vote(stack, n_classes)


def weighted_vote(stack, n_classes: int, w):
    """(M, N, H, W) -> (N, H, W): argmax_c w_c * (votes for c); a class with no vote never wins; ties to the lower
    class. w = None or all ones is plain_vote, computed by plain_vote."""
    import torch
    if w is None or np.allclose(np.asarray(w, dtype=np.float64), 1.0):
        return plain_vote(stack, n_classes)
    oh = torch.nn.functional.one_hot(stack.long(), n_classes).sum(0).double()       # (N, H, W, C)
    wt = torch.as_tensor(np.asarray(w, dtype=np.float64), device=oh.device)
    sc = oh * wt
    sc[oh == 0] = -1.0                                                               # unwritten classes never win
    return sc.argmax(-1)


def class_weights(prior: dict, alpha: float, n_classes: int, ignore: int) -> np.ndarray:
    """w_c = f_c ** -alpha from step70e's fixed class prior (shares over the scored classes); w_ignore = 1."""
    w = np.ones(n_classes, dtype=np.float64)
    for name, i in prior["class_index"].items():
        if i == ignore:
            continue
        w[i] = float(prior["share"][name]) ** (-float(alpha)) if alpha else 1.0
    return w


# ---------------------------------------------------------------- decoding (a2seg, GPU)
def _views_for(rec: dict, root):
    from eo.data import seg_dataset as SD
    kw = {} if root is None else {"root": Path(root)}
    ds = SD.SegDataset(SD.SegSpec(rec["task"], rec["arm"], rec["frames"], rec["split"], **kw))
    return ds.views.set_index("view").loc[rec["views"]].reset_index()


def decode_codes(task: str, arm: str, codes: np.ndarray, views, out: Path, tok=None, device: str = "cuda",
                 per: int = None) -> dict:
    """codes (V, K, G) of these views -> out/classmaps.npy (S, K, H, W) uint8, S = samples (PASTIS: views)."""
    import torch
    from eo.data import map_vocab as MV
    from eo.eval import seg as SEG
    from eo.maps import decode as D, frozen, harness as H, palette as P
    t = P.TASKS[task]
    side = MV.spec(arm)[2]
    if codes.ndim == 2:
        codes = codes[:, None]
    V, K, G = codes.shape
    if G != side * side or V != len(views):
        raise ValueError(f"codes {codes.shape} for {len(views)} views of arm {arm}")
    if task == "pastis":
        n_crops, S_ = 1, V
    else:
        problems = SEG.crop_order_problems(task, views)
        if problems:
            raise RuntimeError("; ".join(problems[:3]))
        n_crops = 9
        S_ = V // n_crops
    tok = tok or frozen.load(arm, device)
    rule = frozen.record(arm)["rule"]
    native = t.native
    out.mkdir(parents=True, exist_ok=True)
    cm = np.lib.format.open_memmap(out / "classmaps.npy", mode="w+", dtype=np.uint8, shape=(S_, K, native, native))
    per = per or {"pastis": 64, "cloudsen12": 4}[task]
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, S_, per):
            e = min(s + per, S_)
            for j in range(K):
                ib = torch.as_tensor(codes[s * n_crops:e * n_crops, j].astype(np.int64), device=device)
                sc = H.class_scores(tok, ib.view(-1, side, side), task, P.arm_palette(arm), "crop224")
                cm[s:e, j] = D.apply_rule(sc, t.ignore, rule, tok.output).to(torch.uint8).cpu().numpy()
            cm.flush()
    rows = views.iloc[::n_crops] if n_crops > 1 else views
    meta = {"task": task, "arm": arm, "rule": rule, "K": int(K), "samples": int(S_), "native": int(native),
            "n_classes": int(t.n_classes), "ignore": t.ignore, "seconds": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    (out / "views.json").write_text(json.dumps({"view": [int(x) for x in rows.view],
                                                "sample_id": [str(x) for x in rows.sample_id],
                                                "k": [int(x) for x in rows.k]}) + "\n")
    (out / "decode.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["decode", "truth"])
    ap.add_argument("--gen", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=None, help="the seg root the views live under (test 3: geobench2_tta)")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--arm", default="aligned", help="truth: the map arm (A2 Step 7.0f: c4375_224)")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and a.expect_gpu and a.expect_gpu not in torch.cuda.get_device_name(0):
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {torch.cuda.get_device_name(0)}")
    out = Path(a.out)
    if (out / "decode.json").exists():
        print(f"skip {out}: decoded already", flush=True)
        return
    if a.cmd == "decode":
        g = Path(a.gen)
        rec = json.loads((g / "record.json").read_text())
        if rec["split"] != "val":
            raise PermissionError(f"{g}: split {rec['split']}")
        views = _views_for(rec, a.root)
        meta = decode_codes(rec["task"], rec["arm"], np.load(g / "codes.npy"), views, out, device=dev)
        meta["gen"] = str(g)
    else:
        from eo.data import seg_dataset as SD
        ds = SD.SegDataset(SD.SegSpec("pastis", a.arm, "p7", "val", root=Path(a.root)))
        views = ds.views
        codes = np.asarray(ds.maps)[views.view.values]
        meta = decode_codes("pastis", a.arm, codes, views, out, device=dev)
        meta["truth_of"] = str(Path(a.root) / "pastis" / "tok" / f"MAP_{a.arm}")
    (out / "decode.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    print(json.dumps({k: meta[k] for k in ("task", "arm", "K", "samples", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
