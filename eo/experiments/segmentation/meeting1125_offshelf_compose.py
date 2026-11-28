#!/usr/bin/env python
"""Meeting of 2026-11-25, topic 1 (off-the-shelf Cosmos): what an as-released Cosmos would cost on the maps R0 writes.

Question. No model has ever generated maps through an as-released Cosmos. Its true-token ceilings are known (A2
Step 1, agent A). This check measures the other half of "how much would the tokenizer itself lose", at the quality of
the maps our decoder actually writes: R0's K = 8 generated maps (the frozen aligned tokenizer, Step 5.2's arm) are
decoded, re-rendered with the universal palette, passed through another tokenizer's round trip (encode, decode, its
Step 1 rule), voted, and scored against the truth.

Read rule (written before any number of this script):
  - delta = composed - direct, in fold-4 / val mIoU, per run, and its mean over the three seeds of each task.
  - It estimates the representational cost a tokenizer adds at R0's map quality, assuming the model's errors are
    unchanged. It is NOT a prediction of a model trained on that tokenizer: what such a model learns (code spread,
    alignment, rows) is not in it. It is one component, read beside A2 Step 4.0's follow-up 2 (the learnability cost of
    a Cosmos code assignment at a fixed ceiling).
  - Two orders: "per_sample" (each of the K samples through the round trip, then the vote: what a pipeline on that
    tokenizer would vote) and "voted" (the vote first, one round trip). PASTIS reads both; CloudSen12 reads "voted"
    only unless --per-sample-cloudsen12, for time.
Controls, each able to fail:
  R  the direct path (aligned decode, nearest, vote) reproduces each run's recorded K = 8 mIoU to 1e-6;
  T  the same round-trip code fed the TRUE val maps reproduces the tokenizer's recorded val ceiling (agent A's read, or
     the frozen record) to 1e-4 (codes depend on the card and the executor; A read on this TITAN V);
  F  the composed maps scored against the next view's labels fall to the prior level (below 0.05 PASTIS, 0.30
     CloudSen12): the score reads the maps' content;
  S  (reported, --tok aligned) the aligned tokenizer's own round trip on its own maps: the re-encoding artefact floor.

Runs in the `a2seg` env. Cosmos's JITs need bf16, so the TITAN V (under the power watchdog,
/data/enric/reports/a2_seg/queues/step1_titanv_unit.sh); the fp32 tokenizers (aligned, cosmos16_224) also run on the
GTX Titan X. Val only: eo/eval/seg and eo/maps/sets refuse a test split.

    python eo/experiments/segmentation/meeting1125_offshelf_compose.py --tok t1_di16_224 --task pastis \
        --dirs <gen dir> ... --out /data/enric/reports/a2_seg/meeting_2026-11-25/offshelf/compose/t1_di16_224_pastis.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eo.data import map_vocab as MV          # noqa: E402
from eo.data import seg_dataset as SD        # noqa: E402
from eo.eval import seg as SEG               # noqa: E402
from eo.maps import decode as D              # noqa: E402
from eo.maps import frozen                   # noqa: E402
from eo.maps import harness as H             # noqa: E402
from eo.maps import palette as P             # noqa: E402
from eo.maps import scoring as S             # noqa: E402
from eo.maps import sets                     # noqa: E402
from eo.maps.tokenizers import cosmos, run_fixed  # noqa: E402

A_READS = Path("/data/enric/reports/a2_seg/step1/A")
CHUNK = {"pastis": 64, "cloudsen12": 4}


def load_tok(name: str, dev: str):
    """(tokenizer, rule, recorded val ceilings {task: mIoU}, provenance)."""
    if name in frozen.ALL:
        tok = frozen.load(name, dev)
        rec = frozen.record(name)
        return tok, rec["rule"], rec["ceilings"], f"frozen {frozen.ALL[name]}"
    read = json.loads((A_READS / f"read_{name}.json").read_text())
    tok = cosmos(name, dev)
    ceil = {d: v["ceiling"]["mIoU"] for d, v in read["read"].items()}
    return tok, read["rule"], ceil, f"as released, agent A's read {A_READS / f'read_{name}.json'} (rule from {read.get('rule_from')})"


def round_trip(tok, rule: dict, maps: np.ndarray, task: str, dev: str) -> np.ndarray:
    """Native class maps (S, H, W) -> the tokenizer's reading of them (S, H, W) uint8: render with the task's universal
    assignment, cut as the test reads it, encode at the fixed batch shape, decode, class scores on the native grid,
    the rule. The same calls as eo/maps/harness.tokens + sweep, on maps given in memory."""
    t = P.TASKS[task]
    out = []
    with torch.no_grad():
        for s in range(0, len(maps), CHUNK[task]):
            x = H.render_crops(maps[s:s + CHUNK[task]], task, "universal", tok.input_size, "crop224", dev)
            idx = run_fixed(tok.encode, x, tok.batch).long()
            need_votes = rule["rule"] == "vote"
            res = H.class_scores(tok, idx, task, "universal", "crop224", with_votes=need_votes)
            sc, votes = res if need_votes else (res, None)
            out.append(D.apply_rule(sc, t.ignore, rule, tok.output, votes).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def direct_samples(aligned, codes: np.ndarray, views, task: str, dev: str) -> np.ndarray:
    """(V, K, G) aligned codes -> (K, S, H, W) uint8 class maps: eo/eval/seg.score's decode and nearest rule, per
    sample, before the vote."""
    t = P.TASKS[task]
    side = MV.spec("aligned")[2]
    labels, _ = SEG.gt_for_views(task, views)
    n_crops = len(views) // len(labels)
    rule = frozen.record("aligned")["rule"]
    K = codes.shape[1]
    out = np.zeros((K, len(labels), *labels.shape[1:]), np.uint8)
    with torch.no_grad():
        for s in range(0, len(labels), CHUNK[task]):
            e = min(s + CHUNK[task], len(labels))
            for j in range(K):
                ib = torch.as_tensor(codes[s * n_crops:e * n_crops, j].astype(np.int64), device=dev)
                sc = H.class_scores(aligned, ib.view(-1, side, side), task, P.arm_palette("aligned"), "crop224")
                out[j, s:e] = D.apply_rule(sc, t.ignore, rule, aligned.output).to(torch.uint8).cpu().numpy()
    return out


def vote(preds: np.ndarray, task: str, dev: str) -> np.ndarray:
    t = P.TASKS[task]
    out = []
    for s in range(0, preds.shape[1], CHUNK[task] * 4):
        out.append(SEG.vote(torch.as_tensor(preds[:, s:s + CHUNK[task] * 4], device=dev).long(),
                            t.n_classes).to(torch.uint8).cpu().numpy())
    return np.concatenate(out)


def score(pred: np.ndarray, labels: np.ndarray, task: str, dev: str) -> dict:
    t = P.TASKS[task]
    conf = S.Confusion(t.n_classes, t.ignore, dev)
    for s in range(0, len(labels), CHUNK[task] * 4):
        conf.update(torch.as_tensor(np.ascontiguousarray(labels[s:s + CHUNK[task] * 4]), device=dev).long(),
                    torch.as_tensor(pred[s:s + CHUNK[task] * 4], device=dev).long())
    return conf.result()


def brief(r: dict) -> dict:
    keep = ("mIoU", "pixel_acc", "boundary_acc", "interior_acc", "iou_per_class")
    return {k: r[k] for k in keep if k in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True, help="t1_di16_224, t1_di8_224 (as released), or a frozen arm (aligned, "
                                                   "cosmos16_224, c4375_224)")
    ap.add_argument("--task", required=True, choices=("pastis", "cloudsen12"))
    ap.add_argument("--dirs", nargs="*", default=[], help="R0's full_k8 generation directories of this task")
    ap.add_argument("--pred-dirs", nargs="*", default=[],
                    help="Route C's head directories (pred.npy: one native class map per sample, in the scorer's "
                         "sample order): the same round trip on the maps of a better model (added before any of its "
                         "numbers, 2026-11-25 12:40)")
    ap.add_argument("--per-sample", action="store_true", help="also the per-sample order (default for PASTIS)")
    ap.add_argument("--skip-truth", action="store_true")
    ap.add_argument("--expect-gpu", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in gpu:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {gpu}")
    t0 = time.time()
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = {"tok": a.tok, "task": a.task, "gpu": gpu, "started": time.strftime("%F %T"), "runs": {}}
    tok, rule, ceilings, prov = load_tok(a.tok, dev)
    aligned = frozen.load("aligned", dev) if a.tok != "aligned" else tok
    res.update({"rule": rule, "provenance": prov, "recorded_val_ceiling": ceilings.get(a.task),
                "tokenizer_record": tok.record()})
    per_sample = a.per_sample or a.task == "pastis"

    # Control T: the round trip fed the true val maps reproduces the recorded ceiling.
    if not a.skip_truth:
        labels_all, _ = sets.load(a.task, "val")
        labels_all = np.asarray(labels_all)
        ts = time.time()
        truth_pred = round_trip(tok, rule, labels_all, a.task, dev)
        rt = score(truth_pred, labels_all, a.task, dev)
        res["control_T"] = {"mIoU": rt["mIoU"], "recorded": ceilings.get(a.task),
                            "abs_diff": abs(rt["mIoU"] - ceilings.get(a.task, np.nan)),
                            "pass": bool(abs(rt["mIoU"] - ceilings.get(a.task, np.nan)) <= 1e-4),
                            "result": brief(rt), "seconds": time.time() - ts}
        print(f"[T] truth round trip {a.tok} {a.task}: {rt['mIoU']:.6f} against recorded {ceilings.get(a.task)} "
              f"({time.time() - ts:.0f} s)", flush=True)
        out_path.write_text(json.dumps(res, indent=1, default=float))

    for d in map(Path, a.dirs):
        rs = time.time()
        rec = json.loads((d / "record.json").read_text())
        if rec["split"] != "val" or rec["task"] != a.task or rec["arm"] != "aligned":
            raise SystemExit(f"{d}: {rec['task']} {rec['arm']} {rec['split']}")
        codes = np.load(d / "codes.npy")
        recorded = json.loads((d / "score.json").read_text())["result"]["mIoU"]
        ds = SD.SegDataset(SD.SegSpec(rec["task"], rec["arm"], rec["frames"], rec["split"]))
        views = ds.views.set_index("view").loc[rec["views"]].reset_index()
        labels, s_idx = SEG.gt_for_views(a.task, views)
        samples = direct_samples(aligned, codes, views, a.task, dev)          # (K, S, H, W)
        direct = vote(samples, a.task, dev)
        r_direct = score(direct, labels, a.task, dev)
        run = {"dir": str(d), "checkpoint": rec.get("checkpoint"), "K": int(codes.shape[1]),
               "recorded_K8": recorded, "direct": brief(r_direct),
               "control_R": {"abs_diff": abs(r_direct["mIoU"] - recorded),
                             "pass": bool(abs(r_direct["mIoU"] - recorded) <= 1e-6)}}
        print(f"[R] {d.parent.name}: direct {r_direct['mIoU']:.6f} against recorded {recorded:.6f}", flush=True)
        # the voted order
        comp_v = round_trip(tok, rule, direct, a.task, dev)
        r_v = score(comp_v, labels, a.task, dev)
        r_f = score(np.roll(comp_v, 1, axis=0), labels, a.task, dev)
        run["voted"] = {"composed": brief(r_v), "delta": r_v["mIoU"] - r_direct["mIoU"],
                        "control_F_next_view": r_f["mIoU"],
                        "control_F_pass": bool(r_f["mIoU"] < {"pastis": 0.05, "cloudsen12": 0.30}[a.task])}
        print(f"    voted: composed {r_v['mIoU']:.4f} (delta {r_v['mIoU'] - r_direct['mIoU']:+.4f}); "
              f"F next view {r_f['mIoU']:.4f}", flush=True)
        if per_sample:
            comp = np.stack([round_trip(tok, rule, samples[j], a.task, dev) for j in range(samples.shape[0])])
            comp_vote = vote(comp, a.task, dev)
            r_p = score(comp_vote, labels, a.task, dev)
            r_p1 = score(comp[0], labels, a.task, dev)
            r_d1 = score(samples[0], labels, a.task, dev)
            run["per_sample"] = {"composed": brief(r_p), "delta": r_p["mIoU"] - r_direct["mIoU"],
                                 "K1_direct": r_d1["mIoU"], "K1_composed": r_p1["mIoU"],
                                 "K1_delta": r_p1["mIoU"] - r_d1["mIoU"]}
            print(f"    per sample: composed {r_p['mIoU']:.4f} (delta {r_p['mIoU'] - r_direct['mIoU']:+.4f}); "
                  f"K=1 {r_d1['mIoU']:.4f} -> {r_p1['mIoU']:.4f}", flush=True)
        run["seconds"] = time.time() - rs
        res["runs"][d.parent.name] = run
        out_path.write_text(json.dumps(res, indent=1, default=float))

    for d in map(Path, a.pred_dirs):
        rs = time.time()
        rec = json.loads((d / "record.json").read_text())
        if rec["split"] != "val" or rec["task"] != a.task or rec.get("route") != "C":
            raise SystemExit(f"{d}: {rec['task']} {rec.get('route')} {rec['split']}")
        recorded = json.loads((d / "score.json").read_text())["result"]["mIoU"]
        ds = SD.SegDataset(SD.SegSpec(rec["task"], "aligned", rec["frames"], rec["split"]))
        views = ds.views.set_index("view").loc[rec["views"]].reset_index()
        labels, _ = SEG.gt_for_views(a.task, views)
        pred = np.load(d / "pred.npy")
        if pred.shape != labels.shape:
            raise SystemExit(f"{d}: pred {pred.shape} against labels {labels.shape}")
        r_direct = score(pred, labels, a.task, dev)
        comp = round_trip(tok, rule, pred, a.task, dev)
        r_c = score(comp, labels, a.task, dev)
        r_f = score(np.roll(comp, 1, axis=0), labels, a.task, dev)
        res.setdefault("route_c", {})[d.parent.name] = {
            "dir": str(d), "recorded": recorded, "direct": brief(r_direct),
            "control_R": {"abs_diff": abs(r_direct["mIoU"] - recorded), "pass": bool(abs(r_direct["mIoU"] - recorded) <= 1e-6)},
            "composed": brief(r_c), "delta": r_c["mIoU"] - r_direct["mIoU"], "control_F_next_view": r_f["mIoU"],
            "control_F_pass": bool(r_f["mIoU"] < {"pastis": 0.05, "cloudsen12": 0.30}[a.task]), "seconds": time.time() - rs}
        print(f"[C] {d.parent.name}: direct {r_direct['mIoU']:.6f} (recorded {recorded:.6f}); composed {r_c['mIoU']:.4f} "
              f"(delta {r_c['mIoU'] - r_direct['mIoU']:+.4f}); F next view {r_f['mIoU']:.4f}", flush=True)
        out_path.write_text(json.dumps(res, indent=1, default=float))
    if "route_c" in res:
        rc = list(res["route_c"].values())
        res["summary_route_c"] = {"n": len(rc), "direct_mean": float(np.mean([r["direct"]["mIoU"] for r in rc])),
                                  "composed_mean": float(np.mean([r["composed"]["mIoU"] for r in rc])),
                                  "delta_mean": float(np.mean([r["delta"] for r in rc])),
                                  "delta_per_run": [r["delta"] for r in rc],
                                  "controls_R": all(r["control_R"]["pass"] for r in rc),
                                  "controls_F": all(r["control_F_pass"] for r in rc)}
        print(json.dumps(res["summary_route_c"], indent=1), flush=True)
    runs = list(res["runs"].values())
    if not runs:
        res["seconds"], res["finished"] = time.time() - t0, time.strftime("%F %T")
        out_path.write_text(json.dumps(res, indent=1, default=float))
        return
    summ = {"n_runs": len(runs), "direct_mean": float(np.mean([r["direct"]["mIoU"] for r in runs])),
            "voted_composed_mean": float(np.mean([r["voted"]["composed"]["mIoU"] for r in runs])),
            "voted_delta_mean": float(np.mean([r["voted"]["delta"] for r in runs])),
            "voted_delta_per_run": [r["voted"]["delta"] for r in runs],
            "controls_R": all(r["control_R"]["pass"] for r in runs),
            "controls_F": all(r["voted"]["control_F_pass"] for r in runs)}
    if per_sample:
        summ.update({"per_sample_composed_mean": float(np.mean([r["per_sample"]["composed"]["mIoU"] for r in runs])),
                     "per_sample_delta_mean": float(np.mean([r["per_sample"]["delta"] for r in runs])),
                     "per_sample_delta_per_run": [r["per_sample"]["delta"] for r in runs]})
    res["summary"] = summ
    res["seconds"] = time.time() - t0
    res["finished"] = time.strftime("%F %T")
    out_path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
