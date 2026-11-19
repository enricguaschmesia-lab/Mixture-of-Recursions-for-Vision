"""A2 Step 1, Phase 3: apply rule 1.3 to every candidate read on the validation maps, and write the table.

Scans the agents' reads (each a `harness.score` dict per dataset: A's read_*.json and prune_*.json, B's
rcp/<run>/read_best.json, C's [rcp/]<run>/read/read_<dataset>.json), puts each candidate in its token band,
checks it against the bar, and picks per band the highest ceiling among those that pass.

The rule (docs/plans/a2_segmentation.md, Step 1.3; fixed, never tuned here):
  - bar: PASTIS fold 4 >= 0.85, CloudSen12 val >= 0.95, LULC >= 0.95 (harness.bar);
  - short band: <= 256 tokens; long band: 784-1,024 tokens;
  - the highest-ceiling candidate that passes, per band; ties go to the smaller codebook, then the lower code
    entropy.
Two readings the plan leaves open, fixed here before the close comparisons were known (2026-11-18 21:40,
worklog): "the ceiling" of a candidate over three read sets is the unweighted mean of its three ceilings
(each is reported beside it); a tie is two means less than 0.5 point apart (Enric, 2026-11-19, worklog item 80;
0.1 point until then). Code entropy for the tie-break is the mean over the three read sets of bits per token.
Cosmos-0.1 rows are reference rows (handover: "only as a reference row") and never shortlisted.

The orchestrator's re-scores (O/rescore_*.json, step1_O_rescore.py) are matched to their candidates by the
weights' sha256 (or the Cosmos key and size) and their largest difference from the agent's read is reported.

    python -m eo.experiments.segmentation.step1_O_shortlist [--freeze]

Writes O/shortlist.json and O/shortlist.md under /data/enric/reports/a2_seg/step1. With --freeze, copies each
shortlisted candidate's weights, its loader module and eo/maps/*.py into frozen/<band>_<name>/ with
SHA256SUMS, read-only (the record for freezing; the pick is Step 4.0's).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import shutil
import stat
from pathlib import Path

import numpy as np

ROOT = Path("/data/enric/reports/a2_seg/step1")
REPO = Path(__file__).resolve().parents[3]
DATASETS = ("pastis", "cloudsen12", "lulc")
BAR = {"pastis": 0.85, "cloudsen12": 0.95, "lulc": 0.95}        # = eo.maps.harness.bar's
TIE = 0.005                                                     # 0.5 point (Enric, 2026-11-19; was 0.1 point)
RCP_OUT = "/a2seg/out/"                                         # $L/a2seg/out/<agent>/<run>/... on lts4 scratch


def band(tokens: int) -> str | None:
    if tokens <= 256:
        return "short"
    if 784 <= tokens <= 1024:
        return "long"
    return None


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _row(name: str, agent: str, per: dict, paths: dict, other: dict | None = None, extra: dict | None = None):
    """One candidate from its per-dataset score dicts. `other`: the same reads on another card."""
    tok = per["pastis"]["tokenizer"]
    rules = {json.dumps(per[d]["rule"], sort_keys=True) for d in DATASETS}
    assert len(rules) == 1, f"{name}: one rule for all three read sets, got {rules}"
    harness = {per[d].get("harness") for d in DATASETS}
    assert harness == {"step1-v2"}, f"{name}: read with harness {harness}"
    assert all(per[d]["split"] == "val" for d in DATASETS), name
    r = {"name": name, "agent": agent, "tokens": int(per["pastis"]["tokens_per_map"]),
         "n_codes": int(per["pastis"]["codes"]["n_codes"]), "rule": per["pastis"]["rule"],
         "palette_sha256": per["pastis"]["palette_sha256"], "crop_mode": per["cloudsen12"]["crop_mode"],
         "tokenizer": tok, "json": paths, "extra": extra or {}}
    for d in DATASETS:
        c, k = per[d]["ceiling"], per[d]["codes"]
        r[d] = {"mIoU": c["mIoU"], "boundary_acc": c["boundary_acc"], "distinct": k["read_distinct"],
                "entropy_bits": k["read_entropy_bits"], "unseen_token_share": k["unseen_token_share"],
                "unseen_code_share": k["unseen_code_share"], "top1000_code_mass": k["top1000_code_mass"],
                "nearest": per[d].get("also", {}).get("nearest", {}).get("mIoU")}
    r["mean"] = float(np.mean([r[d]["mIoU"] for d in DATASETS]))
    r["entropy"] = float(np.mean([r[d]["entropy_bits"] for d in DATASETS]))
    r["passes"] = all(r[d]["mIoU"] >= BAR[d] for d in DATASETS)
    r["band"] = band(r["tokens"])
    r["reference_only"] = name.startswith("A:c01_")
    if other:
        r["other_card"] = {"gpu": other["gpu"],
                           "max_abs_diff": max(abs(other[d] - r[d]["mIoU"]) for d in DATASETS)}
    return r


def collect() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "A/read_*.json"))) + sorted(glob.glob(str(ROOT / "A/prune_*.json"))):
        j = json.loads(Path(f).read_text())
        if "read" not in j or not all(d in j["read"] for d in DATASETS):
            continue
        stem = Path(f).stem.replace("read_", "")
        rows.append(_row(f"A:{stem}", "A", j["read"], {"read": f}, extra={"gpu": j.get("gpu")}))
    for f in sorted(glob.glob(str(ROOT / "B/rcp/*/read_best.json"))):
        j = json.loads(Path(f).read_text())
        if "read" not in j:
            continue
        run = Path(f).parent.name
        other = None
        tv = Path(f).with_name("read_best_titanv.json")
        if tv.exists():
            t = json.loads(tv.read_text())
            if "read" in t:
                other = {"gpu": t.get("gpu"), **{d: t["read"][d]["ceiling"]["mIoU"] for d in DATASETS}}
        rows.append(_row(f"B:{run}", "B", j["read"], {"read": f}, other, extra={"gpu": j.get("gpu")}))
    for rd in sorted(glob.glob(str(ROOT / "C/rcp/*/read"))) + sorted(glob.glob(str(ROOT / "C/*/read"))):
        rd = Path(rd)
        if not all((rd / f"read_{d}.json").exists() for d in DATASETS) or "smoke" in rd.parent.name:
            continue                                            # a smoke test of the read path is no candidate
        per = {d: json.loads((rd / f"read_{d}.json").read_text()) for d in DATASETS}
        other = None
        a1 = rd.with_name("read_a100")
        if all((a1 / f"read_{d}.json").exists() for d in DATASETS):
            other = {"gpu": "A100", **{d: json.loads((a1 / f"read_{d}.json").read_text())["ceiling"]["mIoU"]
                                       for d in DATASETS}}
        rows.append(_row(f"C:{rd.parent.name}", "C", per, {d: str(rd / f"read_{d}.json") for d in DATASETS},
                         other, extra={"gpu": per["pastis"]["tokenizer"].get("gpu")}))
    return rows


def attach_rescores(rows: list[dict]):
    for f in sorted(glob.glob(str(ROOT / "O/rescore_*.json"))):
        j = json.loads(Path(f).read_text())
        if not all(d in j.get("read", {}) for d in DATASETS):
            continue
        tok = j["tokenizer"]
        w = tok.get("sha256", {}).get("weights")
        for r in rows:
            rt = r["tokenizer"]
            same = (w and rt.get("sha256", {}).get("weights") == w) or \
                   (tok.get("kind") == "cosmos_di" and rt.get("kind") == "cosmos_di" and
                    tok.get("key") == rt.get("key") and tok.get("input_size") == rt.get("input_size") and
                    r["n_codes"] == j["read"]["pastis"]["codes"]["n_codes"])
            if same and json.dumps(j["rule"], sort_keys=True) == json.dumps(r["rule"], sort_keys=True):
                diff = {d: j["read"][d]["ceiling"]["mIoU"] - r[d]["mIoU"] for d in DATASETS}
                codes = {d: (j["read"][d]["codes"]["read_distinct"], r[d]["distinct"]) for d in DATASETS}
                r.setdefault("rescores", []).append({"json": f, "gpu": j.get("gpu"), "diff": diff,
                                                     "max_abs_diff": max(abs(v) for v in diff.values()),
                                                     "distinct_codes_mine_theirs": codes})


def apply_rule(rows: list[dict]) -> dict:
    out = {}
    for b in ("short", "long"):
        cand = [r for r in rows if r["band"] == b and r["passes"] and not r["reference_only"]]
        if not cand:
            out[b] = None
            continue
        best = max(r["mean"] for r in cand)
        tied = [r for r in cand if best - r["mean"] < TIE]
        win = sorted(tied, key=lambda r: (r["n_codes"], r["entropy"]))[0]
        near = [{"name": r["name"], "mean": r["mean"], "n_codes": r["n_codes"], "behind_points": 100 * (best - r["mean"])}
                for r in cand if r is not win and best - r["mean"] < 0.005]
        out[b] = {"winner": win["name"], "mean": win["mean"], "tied": [r["name"] for r in tied],
                  "passing": len(cand), "within_0.5pt": near}
    if out["short"] is None and out["long"] is None:
        out["fallback"] = "Cosmos Tokenize1-DI8x8 at 256 px with the region rule, alone"
    return out


def table(rows: list[dict], picks: dict) -> str:
    win = {p["winner"] for p in picks.values() if isinstance(p, dict)}
    lines = ["| band | candidate | tokens | codes | rule | PASTIS f4 | CloudSen12 val | LULC | mean | bar | "
             "boundary acc (P/C/L) | distinct codes (P/C/L) | entropy bits (P/C/L) | unseen tokens % (P/C/L) | "
             "other card max abs diff | re-score max abs diff |",
             "|" + "---|" * 16]
    order = {"short": 0, "long": 1, None: 2}
    for r in sorted(rows, key=lambda r: (order[r["band"]], -r["mean"])):
        rule = r["rule"]["rule"] + (f" {r['rule']['tau']}/{r['rule']['radius']}" if r["rule"]["rule"] == "region" else "")
        f3 = lambda k, fmt: "/".join(fmt.format(r[d][k]) for d in DATASETS)
        name = f"**{r['name']}**" if r["name"] in win else r["name"]
        if r["reference_only"]:
            name += " (ref)"
        oc = f"{r['other_card']['max_abs_diff']:.1e}" if "other_card" in r else "–"
        rs = ", ".join(f"{x['max_abs_diff']:.1e} ({x['gpu'].replace('NVIDIA ', '')})" for x in r.get("rescores", [])) or "–"
        unseen = "/".join("{:.2f}".format(100 * r[d]["unseen_token_share"]) for d in DATASETS)
        lines.append(f"| {r['band'] or '–'} | {name} | {r['tokens']} | {r['n_codes']:,} | {rule} | "
                     f"{r['pastis']['mIoU']:.4f} | {r['cloudsen12']['mIoU']:.4f} | {r['lulc']['mIoU']:.4f} | "
                     f"{r['mean']:.4f} | {'pass' if r['passes'] else 'fail'} | {f3('boundary_acc', '{:.3f}')} | "
                     f"{f3('distinct', '{:,}')} | {f3('entropy_bits', '{:.2f}')} | {unseen} | {oc} | {rs} |")
    return "\n".join(lines)


def freeze(r: dict, b: str) -> dict:
    """Copy the candidate's weights, its loader module and eo/maps/*.py to frozen/<band>_<name>/ (read-only)."""
    from eo.maps.palette import palette_record
    tok = r["tokenizer"]
    dst = ROOT / "frozen" / f"{b}_{r['name'].replace(':', '_')}"
    if dst.exists():
        raise SystemExit(f"{dst} exists: a frozen record is never overwritten")
    (dst / "code" / "eo" / "maps").mkdir(parents=True)
    files = {}
    weights = tok.get("weights")
    wlist = list(weights.values()) if isinstance(weights, dict) else [weights]
    local = []
    for w in wlist:
        src = Path(w)
        if not src.exists() and RCP_OUT in str(src):         # an A100 read: the pulled, sha-checked copy
            agent, rest = str(src).split(RCP_OUT, 1)[1].split("/", 1)
            src = ROOT / agent.upper() / "rcp" / rest
        if not isinstance(weights, dict):
            assert sha256(src) == tok["sha256"]["weights"], f"{src}: not the weights that were read"
        tgt = dst / "weights" / src.name
        tgt.parent.mkdir(exist_ok=True)
        shutil.copy2(src, tgt)
        files[str(tgt.relative_to(dst))] = sha256(tgt)
        local.append(str(src))
    mod = tok["loader"].split("(", 1)[0].rsplit(".", 1)[0] if tok.get("loader") else None
    loader_mod = None
    if mod:                                                     # the module the loader string imports
        cand = REPO / (mod.replace(".", "/") + ".py")
        loader_mod = str(cand) if cand.exists() else None
    if loader_mod and tok.get("code_sha256"):
        assert sha256(loader_mod) == tok["code_sha256"], f"{loader_mod} changed since the read"
    mods = [Path(loader_mod)] if loader_mod else []
    mods += sorted((REPO / "eo/maps").glob("*.py"))
    for m in mods:
        rel = Path(m).resolve().relative_to(REPO)
        tgt = dst / "code" / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(m, tgt)
        files[str(tgt.relative_to(dst))] = sha256(tgt)
    pal = palette_record()
    assert pal["sha256"] == r["palette_sha256"], "the palette changed since the read"
    frozen_loader = None
    if mod and not isinstance(weights, dict):
        frozen_loader = f"{mod}.load('{dst / 'weights' / Path(local[0]).name}', 'cuda')"
    rec = {"band": b, "candidate": r["name"], "agent": r["agent"], "tokens": r["tokens"], "n_codes": r["n_codes"],
           "input_size": tok.get("input_size"), "grid": tok.get("grid"), "fsq_levels": tok.get("fsq_levels"),
           "output": tok.get("output"), "rule": r["rule"], "palette": "eo.maps.palette.UNIVERSAL",
           "palette_sha256": pal["sha256"], "crop_mode_cloudsen12": r["crop_mode"],
           "weights_as_read": weights, "weights_local": local, "weights_sha256": tok.get("sha256"),
           "loader_as_read": tok.get("loader"), "loader_frozen": frozen_loader,
           "loader_code": loader_mod, "loader_code_sha256": sha256(loader_mod) if loader_mod else None,
           "tokenizer_record": tok,
           "read_on": r["extra"].get("gpu"), "other_card": r.get("other_card"), "rescores": r.get("rescores"),
           "ceilings": {d: r[d]["mIoU"] for d in DATASETS}, "mean": r["mean"], "reads": r["json"],
           "harness": "step1-v2", "frozen_files_sha256": files}
    (dst / "record.json").write_text(json.dumps(rec, indent=1))
    (dst / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in sorted(files.items())))
    for p in sorted(dst.rglob("*"), reverse=True):
        p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "O"))
    a = ap.parse_args()
    rows = collect()
    attach_rescores(rows)
    picks = apply_rule(rows)
    md = table(rows, picks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"rule": {"bar": BAR, "bands": {"short": "<= 256", "long": "784-1024"}, "ceiling": "mean of the three",
                    "tie": f"means less than {TIE} apart -> smaller codebook -> lower mean entropy"},
           "picks": picks, "candidates": rows}
    if a.freeze:
        res["frozen"] = {b: freeze(next(r for r in rows if r["name"] == p["winner"]), b)
                         for b, p in picks.items() if isinstance(p, dict)}
    (out / "shortlist.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "shortlist.md").write_text(md + "\n\n" + json.dumps(picks, indent=1) + "\n")
    print(md)
    print(json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
