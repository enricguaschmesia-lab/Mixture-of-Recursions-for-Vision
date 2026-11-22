"""The map tokenizers frozen by A2 Step 1, loaded only after their hashes check (Step 2.5 on).

    ARMS      Step 4.0's three arms: aligned (b224_64000_long, 196 tokens), short (b256_15360_long,
              256 tokens), long (c1000_256L, 1,024 tokens). The two superseded 0.1-point picks are not here.
    load(arm) every file in the arm's SHA256SUMS re-hashed, the repo's loader module equal to the frozen
              copy (else the frozen copy is imported), then the record's `loader_frozen` called

Every arm decodes with the nearest rule over the universal palette (record.json).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

FROZEN = Path("/data/enric/reports/a2_seg/step1/frozen")
ARMS = {"aligned": "aligned_B_b224_64000_long", "short": "short_B_b256_15360_long", "long": "long_C_c1000_256L"}
#: Step 4.0's follow-up tokenizers (Enric, 2026-11-20, evening): frozen like the arms, kept apart so that every gate that
#: iterates ARMS (verify_segmentation Z7: PASTIS and CloudSen12 tokens per arm) is unchanged. long_C_c4375_224 and
#: long_C_c1000_224a were frozen by eo/experiments/segmentation/step4f_freeze.py (Step 1's own freeze()).
FOLLOWUP = {"b256_64000": "short_B_b256_64000", "c4375_224": "long_C_c4375_224", "c1000_224a": "long_C_c1000_224a",
            #: follow-up 2 (Enric, 2026-11-22): Cosmos DI16x16 fine-tuned with a logit head (agent D, the second run),
            #: frozen by eo/experiments/segmentation/step4c_freeze.py
            "cosmos16_224": "aligned_D_d224_cosmos16_lg"}
ALL = {**ARMS, **FOLLOWUP}
REPO = Path(__file__).resolve().parents[2]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def record(arm: str) -> dict:
    return json.loads((FROZEN / ALL[arm] / "record.json").read_text())


def verify(arm: str) -> dict:
    """Re-hash every file in SHA256SUMS. Returns {path: sha}; raises on any mismatch."""
    d = FROZEN / ALL[arm]
    out = {}
    for line in (d / "SHA256SUMS").read_text().splitlines():
        sha, rel = line.split(maxsplit=1)
        got = _sha(d / rel)
        if got != sha:
            raise RuntimeError(f"{arm}: {rel} sha256 {got} != frozen {sha}")
        out[rel] = sha
    return out


def load(arm: str, device: str = "cuda"):
    """The frozen tokenizer of `arm`, after verify(). The loader module is the repo's if it is
    byte-identical to the frozen copy, else the frozen copy itself."""
    rec = record(arm)
    sums = verify(arm)
    d = FROZEN / ALL[arm]
    loader = rec["loader_frozen"]                      # "eo.experiments.segmentation.step1_X_tok.load('<path>', 'cuda')"
    mod_name, rest = loader.split(".load(", 1)
    weights = rest.split("'")[1]
    rel = "code/" + mod_name.replace(".", "/") + ".py"
    repo_file = REPO / (mod_name.replace(".", "/") + ".py")
    if repo_file.exists() and _sha(repo_file) == sums[rel]:
        mod = importlib.import_module(mod_name)
        source = str(repo_file)
    else:
        spec = importlib.util.spec_from_file_location(f"frozen_{arm}", d / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        source = str(d / rel)
    tok = mod.load(weights, device)
    tok.frozen = {"arm": arm, "dir": str(d), "loader_source": source, "weights": weights,
                  "weights_sha256": sums[[k for k in sums if k.startswith("weights/")][0]],
                  "tokens": rec["tokens"], "n_codes": rec["n_codes"], "input_size": rec["input_size"],
                  "rule": rec["rule"], "palette_sha256": rec["palette_sha256"]}
    assert tok.input_size == rec["input_size"] and tok.n_codes == rec["n_codes"], (arm, tok.input_size, tok.n_codes)
    return tok
