#!/usr/bin/env python
"""A2 Step 1.3's test-fold ceilings (worklog §4 item 85; Enric's yes of 2026-11-24): the frozen pick's TRUE-token
ceiling on PASTIS fold 5 (GEO-Bench-2's test fold) and CloudSen12 test, read ONCE (`a2seg`, GTX Titan X, fp32).

Rule 1.3 (docs/plans/a2_segmentation.md): "Test-fold ceilings are read once, after the pick, and reported beside every
score." Nothing a model generated is read here. Only the stored true map tokens of the test views are read:
Step 2.5's MAP_aligned, encoded by the frozen tokenizer from the test labels. They are decoded and scored exactly as
the val ceilings are, by eo.eval.seg.score: the frozen tokenizer and its rule, pooling to the native grid,
CloudSen12's 3 x 3 tiling, and GEO-Bench-2's mIoU.

The committed guards stay closed: eo/maps/sets.py never extracts a test set, and eo.eval.seg.gt_for_views reads val
only. This script opens the test split for this one read alone:
  - the test views come through SegDataset(..., allow_test=True), the dataset's own opt-in;
  - the test labels are read here, in memory, from the sources eo/maps/build_sets.py reads for val: PASTIS-R's
    TARGET_<id>.npy channel 0 for fold 5, and the CloudSen12 tortilla's 'test' masks;
  - eo.eval.seg.score runs with gt_for_views replaced, in this process only, by the same lookup over those labels.

Two stages:
  checks  (val only; writes checks.json):
          V1 this script's label path reproduces eo/maps/sets.load(task, "val") bit for bit;
          V2 the score through it equals the guarded scorer's, and the frozen record's val ceiling within 1e-6;
          V3 control: the true tokens one view off score far below the ceiling.
  test    (needs checks.json, all passing; writes ceiling_<task>.json, then ceiling.json and ceiling.md):
          T0 the test labels equal what Step 2.5 encoded (PASTIS: eo.geobench.sources' semantic map of every test
             view's sample; CloudSen12: the same tortilla reader, checked on 25 tiles through sources);
          T1 the ceiling, once per task. A task whose ceiling_<task>.json exists is never read again.

    python eo/experiments/segmentation/step13_test_ceiling.py checks --expect-gpu "GTX TITAN X"
    python eo/experiments/segmentation/step13_test_ceiling.py test --expect-gpu "GTX TITAN X"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

OUT = Path("/data/enric/reports/a2_seg/step13_test")
PICK = Path("/data/enric/reports/a2_seg/step4/pick.json")
PASTIS_R = Path("/data/enric/data/PASTIS/PASTIS-R")
ARM = "aligned"
FRAMES = {"pastis": "p1", "cloudsen12": "single"}
TEST_SPLIT = {"pastis": 5, "cloudsen12": "test"}          # PASTIS-R fold 5 = GEO-Bench-2's test fold
VAL_SPLIT = {"pastis": 4, "cloudsen12": "validation"}
TOL = 1e-6


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ labels, as eo/maps/build_sets.py reads val
def pastis_labels(fold: int):
    meta = json.loads((PASTIS_R / "metadata.geojson").read_text())
    fold_of = {int(f["properties"]["ID_PATCH"]): int(f["properties"]["Fold"]) for f in meta["features"]}
    ids = np.array(sorted(i for i, f in fold_of.items() if f == fold), dtype=np.int64)
    lab = np.stack([np.load(PASTIS_R / "ANNOTATIONS" / f"TARGET_{i}.npy")[0] for i in ids])
    return np.ascontiguousarray(lab.astype(np.uint8)), ids


def cloudsen12_labels(tsplit: str):
    import rasterio
    from eo.geobench import sources as S
    df = S._cloudsen12_taco()
    counts = df["tortilla:data_split"].value_counts().to_dict()
    assert counts == {"train": 4000, "validation": 535, "test": 975}, counts
    sub = df[df["tortilla:data_split"] == tsplit].reset_index(drop=True)
    labs, ids = [], []
    for i in range(len(sub)):
        row = sub.read(i)
        with rasterio.open(row.read(1)) as src:
            m = src.read()
        assert m.shape == (1, 512, 512), m.shape
        labs.append(m[0])
        ids.append(sub.iloc[i]["tortilla:id"])
    lab = np.stack(labs)
    assert lab.max() <= 3, lab.max()
    return np.ascontiguousarray(lab.astype(np.uint8)), np.array(ids)


def labels(task: str, split) -> tuple:
    return pastis_labels(split) if task == "pastis" else cloudsen12_labels(split)


def gt_lookup(task: str, lab: np.ndarray, ids: np.ndarray, split: str):
    """eo.eval.seg.gt_for_views's lookup over the given set (same order: the views' samples, first seen)."""
    key = (lambda s: str(s)) if task == "cloudsen12" else (lambda s: int(s))
    where = {key(i): n for n, i in enumerate(ids)}

    def gt(task_: str, views):
        assert task_ == task and set(views.split) == {split}, (task_, sorted(set(views.split)))
        samples = list(dict.fromkeys(views.sample_id.tolist()))
        missing = [s for s in samples if key(s) not in where]
        if missing:
            raise RuntimeError(f"{len(missing)} samples are not in the {task} {split} labels, e.g. {missing[:3]}")
        take = [where[key(s)] for s in samples]
        s_of = {s: n for n, s in enumerate(samples)}
        return np.asarray(lab[take]), np.asarray([s_of[s] for s in views.sample_id.tolist()])
    return gt


def score(task: str, split: str, gt=None, roll: int = 0) -> dict:
    """The ceiling: SEG.score on the stored true tokens of the split's views. gt replaces gt_for_views here only."""
    from eo.data import seg_dataset as SD
    from eo.eval import seg as SEG
    ds = SD.SegDataset(SD.SegSpec(task, ARM, FRAMES[task], split), allow_test=(split == "test"))
    views = ds.views
    codes = np.asarray(ds.maps[views.view.values])[:, None]
    if roll:
        codes = np.roll(codes, roll, axis=0)
    orig = SEG.gt_for_views
    if gt is not None:
        SEG.gt_for_views = gt
    try:
        t0 = time.time()
        res = SEG.score(task, ARM, codes, views, device="cuda")
    finally:
        SEG.gt_for_views = orig
    return {"split": split, "views": int(len(views)), "samples": int(res["samples"]), "codes_sha256": sha(codes),
            "roll": roll, "result": res["result"], "pred_share": res["pred_share"], "rule": res["rule"],
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ stages
def env(expect_gpu: str) -> dict:
    import torch
    from eo.maps import frozen
    name = torch.cuda.get_device_name(0)
    if expect_gpu and expect_gpu not in name:
        raise SystemExit(f"expected {expect_gpu}, CUDA returned {name}")
    pick = json.loads(PICK.read_text())
    assert pick["arm"] == ARM, pick["arm"]
    sums = frozen.verify(ARM)                       # re-hash the frozen tokenizer's files
    code = REPO / "CODE_COMMIT"
    return {"gpu": name, "torch": torch.__version__, "pick_arm": pick["arm"], "pick_candidate": pick["candidate"],
            "frozen_files_checked": len(sums), "frozen_dir": frozen.ALL[ARM] if hasattr(frozen, "ALL") else frozen.ARMS[ARM],
            "recorded_val_ceilings": frozen.record(ARM)["ceilings"],
            "code_commit": code.read_text().strip() if code.exists() else None,
            "script_sha256": file_sha(Path(__file__))}


def checks(a) -> int:
    from eo.maps import frozen, sets
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"stage": "checks (val only)", **env(a.expect_gpu), "tasks": {}}
    recorded = frozen.record(ARM)["ceilings"]
    ok_all = True
    for task in ("pastis", "cloudsen12"):
        lab, ids = labels(task, VAL_SPLIT[task])
        ref_lab, ref_ids = sets.load(task, "val")
        v1 = bool(np.array_equal(lab, np.asarray(ref_lab)) and np.array_equal(ids.astype(str), np.asarray(ref_ids).astype(str)))
        guarded = score(task, "val")
        mine = score(task, "val", gt=gt_lookup(task, lab, ids, "val"))
        m_g, m_m = guarded["result"]["mIoU"], mine["result"]["mIoU"]
        v2 = bool(abs(m_m - m_g) <= TOL and abs(m_g - recorded[task]) <= TOL)
        ctl = score(task, "val", gt=gt_lookup(task, lab, ids, "val"), roll=1)
        v3 = bool(ctl["result"]["mIoU"] < m_g - 0.10)
        rec["tasks"][task] = {
            "V1 labels = sets.load(val)": v1, "labels_sha256": sha(lab), "n": int(len(ids)),
            "V2 score": v2, "guarded_mIoU": m_g, "this_path_mIoU": m_m, "recorded_ceiling": recorded[task],
            "V3 control (true tokens one view off) fires": v3, "control_mIoU": ctl["result"]["mIoU"],
            "guarded": guarded, "this_path": mine, "control": ctl}
        ok_all &= v1 and v2 and v3
        print(f"{task} val: V1 labels {'equal' if v1 else 'DIFFER'}; V2 guarded {m_g:.10f} this path {m_m:.10f} "
              f"recorded {recorded[task]:.10f} -> {'PASS' if v2 else 'FAIL'}; V3 one view off "
              f"{ctl['result']['mIoU']:.4f} -> {'fires' if v3 else 'DOES NOT FIRE'}", flush=True)
    rec["pass"] = bool(ok_all)
    (OUT / "checks.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"checks {'PASS' if ok_all else 'FAIL'} -> {OUT / 'checks.json'}", flush=True)
    return 0 if ok_all else 1


def test(a) -> int:
    from eo.data import seg_dataset as SD
    from eo.geobench import sources as S
    ck = json.loads((OUT / "checks.json").read_text())
    if not ck.get("pass"):
        raise SystemExit("checks.json does not pass: the test split stays closed")
    e = env(a.expect_gpu)
    if ck["script_sha256"] != e["script_sha256"]:
        raise SystemExit("this script changed since its checks ran: rerun the checks first")
    for task in ("pastis", "cloudsen12"):
        out = OUT / f"ceiling_{task}.json"
        if out.exists():
            print(f"{task}: {out} exists; a test ceiling is read once", flush=True)
            continue
        lab, ids = labels(task, TEST_SPLIT[task])
        views = SD.SegDataset(SD.SegSpec(task, ARM, FRAMES[task], "test"), allow_test=True).views
        # T0: the labels scored are the ones Step 2.5 encoded
        where = {(str(i) if task == "cloudsen12" else int(i)): n for n, i in enumerate(ids)}
        smp = views.drop_duplicates("sample_id")
        if task == "pastis":
            chk = smp
            src = lambda r: S.pastis_sample(int(r.sample_idx), ("semantic",))["semantic"]
        else:
            chk = smp.iloc[np.linspace(0, len(smp) - 1, 25).round().astype(int)]
            src = lambda r: S.cloudsen12_sample(int(r.sample_idx))[1]
        bad = [r.sample_id for r in chk.itertuples()
               if not np.array_equal(np.asarray(src(r), np.uint8), lab[where[str(r.sample_id) if task == "cloudsen12"
                                                                          else int(r.sample_id)]])]
        t0 = {"samples_checked": int(len(chk)), "differ": [str(b) for b in bad[:10]], "pass": not bad}
        if bad:
            raise SystemExit(f"{task}: {len(bad)} test labels differ from what Step 2.5 encoded, e.g. {bad[:3]}")
        if len(views.sample_id.unique()) != len(ids):
            raise SystemExit(f"{task}: {views.sample_id.nunique()} test samples in the views, {len(ids)} labels")
        r = score(task, "test", gt=gt_lookup(task, lab, ids, "test"))
        rec = {"what": "A2 Step 1.3's test-fold ceiling of the frozen pick (item 85), read once", "task": task,
               "test_split": str(TEST_SPLIT[task]), "arm": ARM, **e, "labels_sha256": sha(lab), "n_samples": int(len(ids)),
               "T0 labels = Step 2.5's source": t0, "ceiling": r,
               "written": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        out.write_text(json.dumps(rec, indent=1, default=str) + "\n")
        print(f"{task} TEST ceiling: mIoU {r['result']['mIoU']:.4f} (pixel acc {r['result']['pixel_acc']:.4f}; "
              f"{r['samples']} samples, {r['views']} views; T0 {t0['samples_checked']} checked) -> {out}", flush=True)
    res = {t: json.loads((OUT / f"ceiling_{t}.json").read_text()) for t in ("pastis", "cloudsen12")}
    val = ck["recorded_val_ceilings"]
    summary = {"arm": ARM, "candidate": e["pick_candidate"],
               "test": {t: res[t]["ceiling"]["result"]["mIoU"] for t in res},
               "val": {t: val[t] for t in res}, "code_commit": e["code_commit"], "gpu": e["gpu"]}
    (OUT / "ceiling.json").write_text(json.dumps(summary, indent=1) + "\n")
    md = ["# A2 Step 1.3: the frozen pick's test-fold ceilings (item 85)", "",
          f"Arm `{ARM}` ({e['pick_candidate']}), true tokens of every test view, decoded and scored as the val ceilings "
          f"are (`eo.eval.seg.score`), {e['gpu']}, code {e['code_commit']}. Read once.", "",
          "| set | ceiling (mIoU) | pixel accuracy | samples | the same tokenizer on val |", "|---|---|---|---|---|"]
    for t, name in (("pastis", "PASTIS fold 5 (test)"), ("cloudsen12", "CloudSen12 test")):
        rr = res[t]["ceiling"]
        md.append(f"| {name} | {rr['result']['mIoU']:.4f} | {rr['result']['pixel_acc']:.4f} | {rr['samples']} | "
                  f"{val[t]:.4f} |")
    (OUT / "ceiling.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("checks", "test"))
    ap.add_argument("--expect-gpu", default="GTX TITAN X")
    a = ap.parse_args()
    sys.exit(checks(a) if a.stage == "checks" else test(a))


if __name__ == "__main__":
    main()
