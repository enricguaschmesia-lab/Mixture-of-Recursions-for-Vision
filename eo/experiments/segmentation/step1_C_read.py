"""Read agent C's map tokenizers through the harness, with the controls (A2 Step 1, long slot).

Everything scored here goes through `eo.maps.harness` (select_rule, read_candidate, score, sweep); this
script only orchestrates and writes the JSONs.

    python -m eo.experiments.segmentation.step1_C_read --tok <tok_best.pt> --out <dir> --all
      --select     the rule on the training maps (harness.select_rule: nearest or vote, on train_sub)
      --read       Step 1.2's read on the three read sets (harness.read_candidate) and the bar
      --controls   determinism (in process, and against the harness's token cache written by another
                   process), transposed maps, an untrained tokenizer of the same design, the shuffled
                   training assignments against the fixed read assignment
    python -m eo.experiments.segmentation.step1_C_read --curves <run_dir>    training curves (figure + table)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from eo.maps import harness as H
from eo.maps import palette as P
from eo.maps import sets
from eo.maps.tokenizers import MapTokenizer, run_fixed
from eo.experiments.segmentation import step1_C_tok as T

DATASETS = sets.DATASETS


class Transposed(MapTokenizer):
    """The control tokenizer: encodes every input transposed (H <-> W), decodes as the inner one. Read
    through the harness, its maps are scored against the maps as they are, so it must lose the ceiling."""

    def __init__(self, inner: MapTokenizer):
        self.inner = inner
        self.name = inner.name + "_TRANSPOSED"
        self.input_size, self.n_codes, self.output, self.batch = inner.input_size, inner.n_codes, inner.output, inner.batch

    def encode(self, x):
        return self.inner.encode(x.transpose(-1, -2).contiguous())

    def decode(self, idx):
        return self.inner.decode(idx)

    def record(self):
        r = dict(self.inner.record())
        r["control"] = "transposed inputs"
        r["sha256"] = {"transposed_of": r["sha256"]}
        return r


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype(np.int32).tobytes()).hexdigest()


def determinism(tok: T.StepCTokenizer) -> dict:
    """Encode (and decode) the same inputs twice at the fixed batch shape; and compare this process's
    codes with the harness's token cache of PASTIS fold 4, when another process wrote it."""
    out = {}
    for d, n in (("pastis", 482), ("lulc", 96), ("cloudsen12", 8)):
        lab, _ = sets.load(d, "val")
        x = H.render_crops(np.asarray(lab[:n]), d, "universal", tok.input_size, "crop224", "cuda")
        i1 = run_fixed(tok.encode, x, tok.batch)
        i2 = run_fixed(tok.encode, x, tok.batch)
        y1 = run_fixed(tok.decode, i1[:64], tok.batch)
        y2 = run_fixed(tok.decode, i1[:64], tok.batch)
        out[d] = {"inputs": int(len(x)), "codes_identical": bool(torch.equal(i1, i2)),
                  "decoded_logits_identical": bool(torch.equal(y1, y2)), "codes_sha256": digest(i1.cpu().numpy())}
        if d == "pastis":
            path = H.CACHE / f"{tok.name}__{H._weights_tag(tok)}" / \
                f"pastis_val_{H._set_sha('pastis', 'val')}_universal_crop224_b{tok.batch}.npy"
            if path.exists():
                cached = np.load(path)
                out[d]["harness_cache"] = str(path)
                out[d]["identical_to_harness_cache"] = bool(np.array_equal(cached, i1.cpu().numpy()))
            else:
                out[d]["harness_cache"] = None
        # the control: a batch shape other than the fixed one may pick other kernels; report, not assert
        i3 = run_fixed(tok.encode, x, 7)
        out[d]["codes_differing_at_batch_7"] = int((i3 != i1).sum())
    return out


def transposed_control(tok, rule) -> dict:
    tt = Transposed(tok)
    return {d: H.sweep(tt, d, "val", [rule], cache=False)[_k(rule)]["mIoU"] for d in DATASETS}


def untrained_control(cfg, rule, seed: int = 0) -> dict:
    ut = T.untrained(cfg, seed)
    res = {}
    for d in DATASETS:
        r = H.score(H.Candidate(ut, rule), d, "val", cache=False)
        res[d] = {"mIoU": r["ceiling"]["mIoU"], "pixel_acc": r["ceiling"]["pixel_acc"],
                  "read_distinct": r["codes"]["read_distinct"], "read_entropy_bits": r["codes"]["read_entropy_bits"],
                  "classes_scored": r["ceiling"]["classes_scored"]}
    res["record"] = ut.record()
    return res


def assignment_control(n_draws: int = 100000) -> dict:
    """The training path draws P.shuffled_assignment per map (step1_C_train.Sampler.draw); the read path
    renders with the task's fixed assignment (harness.render_crops -> task_palette -> P.TASKS[d].assign).
    How often does a training draw equal the fixed read assignment?"""
    g = torch.Generator().manual_seed(0)
    out = {}
    for d in DATASETS:
        t = P.TASKS[d]
        a = P.shuffled_assignment(n_draws, t.n_classes, t.ignore, generator=g)
        fixed = torch.as_tensor(t.assign)
        sc = [c for c in range(t.n_classes) if c != t.ignore]
        eq = (a[:, sc] == fixed[sc]).all(1)
        per_class = (a[:, sc] == fixed[sc]).float().mean(0)
        n_scored = len(sc)
        p_exact = 1.0
        for j in range(n_scored):
            p_exact /= (31 - j)
        out[d] = {"draws": n_draws, "equal_to_fixed": int(eq.sum()), "p_equal_exact": p_exact,
                  "mean_share_of_classes_on_their_fixed_colour": float(per_class.mean()),
                  "fixed_assign": list(t.assign)}
    return out


def palette_diag(tok, rule, n_trials: int = 3, split: str = "monitor", seed: int = 123) -> dict:
    """A diagnostic beside the controls, on the monitor maps only: the same read through the harness's
    own path with the task's fixed assignment replaced by random injective ones (P.TASKS[d] swapped for
    the duration, then restored). A tokenizer trained on shuffled palettes should read the fixed
    assignment like any other."""
    import dataclasses
    g = torch.Generator().manual_seed(seed)
    out = {}
    for d in DATASETS:
        t0 = P.TASKS[d]
        res = {"split": split, "fixed": H.sweep(tok, d, split, [rule], cache=False)[_k(rule)]["mIoU"], "random": []}
        for _ in range(n_trials):
            a = P.shuffled_assignment(1, t0.n_classes, t0.ignore, generator=g)[0].tolist()
            P.TASKS[d] = dataclasses.replace(t0, assign=tuple(a))
            try:
                m = H.sweep(tok, d, split, [rule], cache=False)[_k(rule)]["mIoU"]
            finally:
                P.TASKS[d] = t0
            res["random"].append({"assign": a, "mIoU": m})
        out[d] = res
    assert all(P.TASKS[d].assign == P.palette_record()["tasks"][d]["assign"] or
               list(P.TASKS[d].assign) == P.palette_record()["tasks"][d]["assign"] for d in DATASETS)
    return out


def _k(rule):
    from eo.maps import decode as D
    return D.rule_key(rule)


def curves(run_dir: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = json.loads((run_dir / "history.json").read_text())
    logs = {}
    for line in (run_dir / "train_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        logs[r["step"]] = r                                          # a resume rewrites a few steps: keep the last
    steps = sorted(logs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in DATASETS:
        ax[0].plot(steps, [logs[s]["loss"][d] for s in steps], label=d, lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("training NLL at the native grid (log)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    hs = [h["step"] for h in hist]
    for d in DATASETS:
        ax[1].plot(hs, [h["monitor"][d]["mIoU"] for h in hist], marker=".", label=d)
    ax[1].plot(hs, [h["mean"] for h in hist], color="k", lw=2, label="mean")
    for h in hist:
        if h.get("cooldown_started"):
            ax[1].axvline(h["step"], color="grey", ls="--", lw=1)
    ax[1].set_ylim(max(0.0, min(h["mean"] for h in hist[1:]) - 0.05) if len(hist) > 1 else 0, 1.0)
    ax[1].axhline(0.95, color="grey", lw=0.5)
    ax[1].axhline(0.85, color="grey", lw=0.5)
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("monitor mIoU (harness, nearest, EMA weights)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    png = run_dir / "curves.png"
    fig.savefig(png, dpi=110)
    table = [{"step": h["step"], "time_h": round(h["time_h"], 2), "lr": h["lr"],
              "train_loss": ({d: round(logs[max(s for s in steps if s <= h["step"])]["loss"][d], 4) for d in DATASETS}
                             if any(s <= h["step"] for s in steps) else None),
              **{d: round(h["monitor"][d]["mIoU"], 4) for d in DATASETS}, "mean": round(h["mean"], 4),
              "codes": {d: h["codes"][d]["distinct"] for d in DATASETS}} for h in hist]
    (run_dir / "curves_table.json").write_text(json.dumps(table, indent=1))
    return {"figure": str(png), "table": table}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--read", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rule", default=None, help="JSON rule (skips --select)")
    ap.add_argument("--native256", action="store_true",
                    help="also CloudSen12 val in crop_mode native256 (256-px tokenizers; beside, never instead)")
    ap.add_argument("--curves", default=None, help="a run directory")
    ap.add_argument("--expect-gpu", default="TITAN X")
    a = ap.parse_args()
    if a.curves:
        r = curves(Path(a.curves))
        print(json.dumps(r["table"][-3:], indent=1))
        print("figure", r["figure"])
        return
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    torch.set_num_threads(2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = T.load(a.tok)
    res = {"tokenizer": tok.record(), "gpu": name, "palette_sha256": P.palette_record()["sha256"],
           "harness": H.HARNESS_VERSION, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    rule = json.loads(a.rule) if a.rule else None
    t0 = time.time()
    if a.select or a.all:
        if rule is None:
            res["selection"] = H.select_rule(tok)
            rule = res["selection"]["chosen"]
            (out / "selection.json").write_text(json.dumps(res["selection"], indent=1))
            print("selection", json.dumps(res["selection"]["mean"]), "->", rule, f"({time.time() - t0:.0f} s)", flush=True)
    rule = rule or {"rule": "nearest"}
    res["rule"] = rule
    if a.read or a.all:
        read = H.read_candidate(tok, rule)
        for d, r in read.items():
            (out / f"read_{d}.json").write_text(json.dumps(r, indent=1))
            print(d, "ceiling", round(r["ceiling"]["mIoU"], 4), "boundary", round(r["ceiling"]["boundary_acc"], 4),
                  "codes", r["codes"]["read_distinct"], "entropy", round(r["codes"]["read_entropy_bits"], 3),
                  "unseen", r["codes"].get("unseen_token_share"), f"({time.time() - t0:.0f} s)", flush=True)
        res["read"] = {d: {"json": str(out / f"read_{d}.json"), "mIoU": r["ceiling"]["mIoU"],
                           "boundary_acc": r["ceiling"]["boundary_acc"],
                           "also": {k: v["mIoU"] for k, v in r["also"].items()}} for d, r in read.items()}
        res["bar"] = H.bar(read)
        print("bar", res["bar"], flush=True)
    if a.native256:
        if tok.input_size != 256:
            raise SystemExit("native256 is the alternative read of a 256-px tokenizer")
        r = H.score(H.Candidate(tok, rule, crop_mode="native256"), "cloudsen12", "val", also=[{"rule": "nearest"}])
        (out / "read_cloudsen12_native256.json").write_text(json.dumps(r, indent=1))
        res["native256"] = {"json": str(out / "read_cloudsen12_native256.json"), "mIoU": r["ceiling"]["mIoU"],
                            "boundary_acc": r["ceiling"]["boundary_acc"]}
        print("cloudsen12 native256 ceiling", round(r["ceiling"]["mIoU"], 4), "boundary",
              round(r["ceiling"]["boundary_acc"], 4), f"({time.time() - t0:.0f} s)", flush=True)
    if a.controls or a.all:
        res["determinism"] = determinism(tok)
        print("determinism", json.dumps(res["determinism"]), flush=True)
        res["transposed"] = transposed_control(tok, rule)
        print("transposed", res["transposed"], flush=True)
        res["untrained"] = untrained_control(tok.cfg, rule)
        print("untrained", {d: res["untrained"][d]["mIoU"] for d in DATASETS}, flush=True)
        res["assignment"] = assignment_control()
        print("assignment", json.dumps(res["assignment"]), flush=True)
        res["palette_diag"] = palette_diag(tok, rule)
        print("palette diagnostic (monitor)", json.dumps({d: [v["fixed"]] + [r["mIoU"] for r in v["random"]]
                                                          for d, v in res["palette_diag"].items()}), flush=True)
    res["seconds"] = time.time() - t0
    parts = [p for p in ("select", "read", "controls", "native256") if getattr(a, p) or (a.all and p != "native256")]
    path = out / f"summary_{'_'.join(parts)}.json"
    path.write_text(json.dumps(res, indent=1))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
