#!/usr/bin/env python
"""A2 Step 4.0, protocol item 10: freeze the picked map tokenizer, its palette assignment and its decoding rule, with
hashes, in /data/enric/reports/a2_seg/step4/pick.json (`a2seg`; no GPU). Run only after step4_read.py picked an arm and
its guard did not fire (or after the guard's R1 comparison picked one).

  * the tokenizer: the arm's frozen copy under /data/enric/reports/a2_seg/step1/frozen/<dir>/, every file of its
    SHA256SUMS re-hashed now (eo/maps/frozen.verify), SHA256SUMS's own hash, record.json's hash and its weights and
    loader hashes, and the repo's loader module compared with the frozen copy;
  * the palette: eo.maps.palette.palette_record()'s hash (the colours and every task's assignment, Step 1's own hash),
    which must equal the frozen record's palette_sha256; beside it UNIVERSAL (32 colours) and each task's fixed
    class -> colour assignment, hashed as canonical JSON and checked against palette.validate (gate Z1's refusals);
  * the decoding rule: the record's rule (nearest), hashed the same way;
  * the read it rests on: select.json and read.json, hashed.

    python eo/experiments/segmentation/step4_freeze.py --arm <picked arm> [--root /data/enric/reports/a2_seg/step4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eo.maps import frozen, palette as P    # noqa: E402


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p: Path) -> str:
    return frozen._sha(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(frozen.ARMS))
    ap.add_argument("--root", default="/data/enric/reports/a2_seg/step4")
    a = ap.parse_args()
    R = Path(a.root)
    read = json.loads((R / "read.json").read_text())
    if read["pick"]["pick"] != a.arm:
        raise SystemExit(f"read.json picked {read['pick']['pick']}, not {a.arm}")
    if read["pick"]["guard_fires"] and not (R / "guard" / "pick_after_r1.json").exists():
        raise SystemExit("the guard fired: the arms are compared again after a short R1 mid-training before freezing")
    d = frozen.FROZEN / frozen.ARMS[a.arm]
    rec = frozen.record(a.arm)
    sums = frozen.verify(a.arm)                       # raises on any mismatch
    mod = rec["loader_frozen"].split(".load(", 1)[0]
    repo_loader = REPO / (mod.replace(".", "/") + ".py")
    frozen_loader_rel = "code/" + mod.replace(".", "/") + ".py"
    colours = [[round(float(v), 12) for v in row] for row in P.UNIVERSAL]
    tasks = {}
    for name, t in P.TASKS.items():
        P.validate(P.UNIVERSAL, t.assign, t.ignore, t.n_classes)
        body = {"assign": list(t.assign), "ignore": t.ignore, "n_classes": t.n_classes,
                "lulc_matches": {str(k): v for k, v in t.lulc_matches.items()}}
        tasks[name] = {**body, "sha256": sha_text(canon(body))}
    pal_rec = P.palette_record()                      # Step 1's hash: colours + every task's assignment
    if pal_rec["sha256"] != rec.get("palette_sha256"):
        raise SystemExit(f"the palette changed since the tokenizer was frozen: {pal_rec['sha256']} != "
                         f"{rec.get('palette_sha256')}")
    rule = rec["rule"]
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "what": "A2 Step 4.0's pick: the map tokenizer, palette assignment and decoding rule every later A2 step uses "
                "(Steps 4.1, 5, 6, 7; R1). Frozen by the protocol pre-registered 2026-11-20 (docs/plans/a2_segmentation.md).",
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "arm": a.arm, "candidate": rec.get("candidate"), "tokens": rec["tokens"], "n_codes": rec["n_codes"],
        "grid": rec.get("grid"), "input_size": rec.get("input_size"),
        "tokenizer": {"dir": str(d), "SHA256SUMS_sha256": sha_file(d / "SHA256SUMS"), "files_rehashed": sums,
                      "record_json_sha256": sha_file(d / "record.json"), "weights_sha256": rec.get("weights_sha256"),
                      "loader_frozen": rec["loader_frozen"], "loader_frozen_sha256": sums.get(frozen_loader_rel),
                      "repo_loader": str(repo_loader),
                      "repo_loader_sha256": sha_file(repo_loader) if repo_loader.exists() else None,
                      "repo_loader_equals_frozen": repo_loader.exists() and sha_file(repo_loader) == sums.get(frozen_loader_rel)},
        "palette": {"palette_record_sha256": pal_rec["sha256"], "equals_frozen_record": True,
                    "palette_record_rule": "eo.maps.palette.palette_record(): sha256 of json.dumps({colours rounded to "
                                           "6 places, every task's assign/ignore/classes/lulc_matches}, sort_keys=True)",
                    "universal": colours, "universal_sha256": sha_text(canon(colours)), "tasks": tasks,
                    "module": "eo/maps/palette.py", "module_sha256": sha_file(REPO / "eo/maps/palette.py")},
        "decoding_rule": {"rule": rule, "sha256": sha_text(canon(rule)), "module": "eo/maps/decode.py",
                          "module_sha256": sha_file(REPO / "eo/maps/decode.py")},
        "read": {"select_json_sha256": sha_file(R / "select.json"), "read_json_sha256": sha_file(R / "read.json"),
                 "pick": read["pick"], "k8_mIoU": {k: v["mIoU"]["k8"] for k, v in read["arms"].items()}},
        "fork_commit": commit,
    }
    (R / "pick.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"froze {a.arm} ({rec.get('candidate')}, {rec['tokens']} tokens, {rec['n_codes']} codes): {len(sums)} files "
          f"re-hashed, palette {out['palette']['universal_sha256'][:12]}, rule {rule} {out['decoding_rule']['sha256'][:12]} "
          f"-> {R / 'pick.json'}")


if __name__ == "__main__":
    main()
