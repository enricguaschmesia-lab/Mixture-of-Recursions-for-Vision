#!/usr/bin/env python
"""A2 Step 1, agent A: off-the-shelf Cosmos discrete tokenizers as map tokenizers, read through the
harness (eo/maps). One unit of work per call; each unit writes one JSON at its end, so a stall costs one
unit. Every number is the harness's (`select_rule`, `read_candidate`, `score`, `sweep`); this file only
chooses what to read and wraps tokenizers.

    python -X faulthandler -u eo/experiments/segmentation/step1_A_run.py <stage> --cand t1_di16_224 --out x.json

Runs in the a2seg env on the TITAN V (Cosmos's JITs need bf16 GEMMs), through
/data/enric/reports/a2_seg/queues/step1_titanv_unit.sh; the driver is step1_A_queue.py.

Stages
  select      harness.select_rule(tok) on each dataset's train_sub. If the chosen region rule sits on the
              grid's edge (tau 0.05 or 0.3, radius 6), the grid is extended one step past that edge (tau
              0.025 or 0.35, radius 8) and select_rule runs again on the extended grid. Radius 1 has no step
              below it: radius 0 is exactly the nearest rule (no non-confident pixel has a confident pixel
              within 0 px, so every pixel keeps its nearest class), which every selection already reads.
  det         determinism at the fixed batch shape (16): the same crops encoded twice and decoded twice in
              one process, and against the harness's token cache written by an earlier process. Controls:
              one changed 32 x 32 block changes that crop's codes and no other crop's; batch 8 instead of 16.
              Also Cosmos's FSQ index <-> digit mapping, checked on real codes: the encoder's own normalised
              codes (its second output) against the digits of its indices.
  read        harness.read_candidate(tok, rule) with the rule `select` chose (nearest beside); harness.bar.
  valsweep    harness.sweep(tok, d, "val", rules) over the selection's rules: the rule each read set would
              pick for itself. A control only: never the candidate's rule.
  native256   (256-px candidates) harness.score(Candidate(tok, rule, crop_mode="native256"), "cloudsen12", "val").
  transposed  control: the tokenizer encodes the transposed renders (x -> x^T before the encoder), scored
              against the maps as they are, with the candidate's rule (code_stats=False, cache=False).
  prune       the lossy prune (BRIEF_A, optional): keep the K most frequent codes over the train_sub tokens
              of the three datasets (pooled counts; ties by smaller index), send every other code to its
              nearest kept code on Cosmos's FSQ grid, and read it like any candidate with the unpruned
              tokenizer's rule. "Nearest": squared Euclidean distance between the codes' normalised FSQ
              values (the decoder's input, each digit d of level L at (d - L//2) / (L//2)), computed exactly
              in integer units of 1/4; ties go to the more frequent kept code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from eo.maps import decode as D                       # noqa: E402
from eo.maps import harness as H                      # noqa: E402
from eo.maps import sets                              # noqa: E402
from eo.maps.tokenizers import MapTokenizer, cosmos, deterministic, run_fixed   # noqa: E402

OUT = Path("/data/enric/reports/a2_seg/step1/A")
HB = OUT / "heartbeat"
NEAREST = {"rule": "nearest"}
TAUS, RADII = D.region_grid.__defaults__           # the harness's default grid, read from it
TAU_LO, TAU_HI, R_HI = 0.025, 0.35, 8              # one step past each edge (the grid's own end steps)
FSQ_LEVELS = np.array([8, 8, 8, 5, 5, 5])
FSQ_BASIS = np.cumprod([1] + list(FSQ_LEVELS[:-1]))  # cosmos_tokenizer FSQuantizer._basis
_HB_FILE: Path | None = None


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def heartbeat():
    if _HB_FILE is not None:
        _HB_FILE.write_text(f"{os.getpid()}\n")


# ------------------------------------------------------------------ wrappers

class Progress(MapTokenizer):
    """Transparent: the same name, record, batch and outputs as the tokenizer it wraps (so the same token
    cache and the same numbers). It only counts encode/decode calls, prints the rate, and touches the
    heartbeat file the driver's stall watchdog reads (the queue's power watchdog cannot see a stall while
    another job keeps the card busy)."""

    def __init__(self, tok: MapTokenizer, every: int = 100):
        self.tok, self.every = tok, every
        self.name, self.input_size, self.n_codes = tok.name, tok.input_size, tok.n_codes
        self.output, self.batch = tok.output, tok.batch
        self.n = {"enc": 0, "dec": 0}
        self.t = {"enc": time.time(), "dec": time.time()}

    def _tick(self, kind: str, rows: int):
        self.n[kind] += 1
        heartbeat()
        if self.n[kind] % self.every == 0:
            torch.cuda.synchronize()
            now = time.time()
            rate = self.every * rows / max(now - self.t[kind], 1e-9)
            self.t[kind] = now
            log(f"{self.name} {kind} {self.n[kind]} calls ({self.n[kind] * rows} rows), {rate:.1f} rows/s, "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak")

    def encode(self, x):
        y = self.tok.encode(x)
        self._tick("enc", len(x))
        return y

    def decode(self, idx):
        y = self.tok.decode(idx)
        self._tick("dec", len(idx))
        return y

    def record(self):
        return self.tok.record()


class Transposed(MapTokenizer):
    """Control: the tokenizer sees the transposed render (encode(x^T)); its decoded map is scored against
    the maps as they are. Never cached (the caller passes cache=False)."""

    def __init__(self, tok: MapTokenizer):
        self.tok = tok
        self.name = f"{tok.name}_CONTROL_transposed_input"
        self.input_size, self.n_codes, self.output, self.batch = tok.input_size, tok.n_codes, tok.output, tok.batch

    def encode(self, x):
        return self.tok.encode(x.transpose(-1, -2).contiguous())

    def decode(self, idx):
        return self.tok.decode(idx)

    def record(self):
        r = dict(self.tok.record())
        r["control"] = "transposed input: codes = encode(x.transpose(-1, -2))"
        r["sha256"] = {**r["sha256"], "control": "transposed_input"}
        return r


def fsq_digits(idx: np.ndarray) -> np.ndarray:
    """Cosmos's FSQ index -> digits (..., 6): (index // basis) % levels (FSQuantizer.indices_to_codes)."""
    return (np.asarray(idx, dtype=np.int64)[..., None] // FSQ_BASIS) % FSQ_LEVELS


def fsq_normalised(idx: np.ndarray) -> np.ndarray:
    """index -> the normalised code the decoder reads: (d - L//2) / (L//2) per digit."""
    h = FSQ_LEVELS // 2
    return (fsq_digits(idx) - h) / h


def fsq_units(idx: np.ndarray) -> np.ndarray:
    """index -> the normalised code in integer units of 1/4 (exact): L=8 -> d-4, L=5 -> 2(d-2)."""
    h = FSQ_LEVELS // 2
    return (fsq_digits(idx) - h) * (4 // h)


def prune_table(counts: np.ndarray, K, device: str = "cuda"):
    """Kept codes (most frequent first) and the code -> kept code table (n_codes,)."""
    n = len(counts)
    order = np.lexsort((np.arange(n), -counts))          # count desc, then index asc
    seen = int((counts > 0).sum())
    k = seen if K == "seen" else min(int(K), seen)
    kept = order[:k].astype(np.int64)
    assert (counts[kept] > 0).all()
    U = torch.as_tensor(fsq_units(np.arange(n)), dtype=torch.int32, device=device)       # (n, 6)
    Uk = U[torch.as_tensor(kept, device=device)]                                           # (k, 6)
    rank = torch.arange(k, device=device, dtype=torch.int64)
    lut = torch.empty(n, dtype=torch.int64, device=device)
    for s in range(0, n, 256):
        u = U[s:s + 256]
        d2 = torch.zeros(len(u), k, dtype=torch.int64, device=device)
        for j in range(U.shape[1]):
            d2 += (u[:, None, j] - Uk[None, :, j]).long() ** 2
        key = (d2 << 20) + rank[None]                       # exact tie-break: the more frequent kept code
        lut[s:s + 256] = torch.as_tensor(kept, device=device)[key.argmin(1)]
    lut_np = lut.cpu().numpy()
    assert (lut_np[kept] == kept).all(), "a kept code does not map to itself"
    assert np.isin(lut_np, kept).all()
    # brute-force check on 2,000 random codes, in float on the normalised grid
    rng = np.random.default_rng(0)
    probe = rng.choice(n, 2000, replace=False)
    V, Vk = fsq_normalised(probe), fsq_normalised(kept)
    d = ((V[:, None] - Vk[None]) ** 2).sum(-1)
    best = d.min(1)
    got = ((fsq_normalised(lut_np[probe]) - V) ** 2).sum(-1)
    assert np.allclose(got, best, atol=1e-12), "the table is not the nearest kept code"
    return kept, lut_np, seen


class Pruned(MapTokenizer):
    """A Cosmos tokenizer whose codes are restricted to `kept`: encode, then map each code to its nearest
    kept code (`prune_table`). Decoding is the unpruned decoder."""

    def __init__(self, tok, tag: str, kept: np.ndarray, lut: np.ndarray, kept_path: str, reference: str):
        self.tok, self.tag = tok, tag
        self.name = f"{tok.name}_prune{tag}"
        self.input_size, self.output, self.batch = tok.input_size, tok.output, tok.batch
        self.n_codes = int(len(kept))
        self.kept_sha = hashlib.sha256(np.ascontiguousarray(kept, dtype="<i8").tobytes()).hexdigest()
        self.kept_path, self.reference = kept_path, reference
        self.lut = torch.as_tensor(lut, dtype=torch.long, device=tok.device)

    def encode(self, x):
        return self.lut[self.tok.encode(x)]

    def decode(self, idx):
        return self.tok.decode(idx)

    def record(self):
        r = dict(self.tok.record())
        r["prune"] = {"K": self.tag, "kept": self.n_codes, "kept_codes_sha256": self.kept_sha,
                      "kept_codes_path": self.kept_path, "kept_order": "most frequent first (int64 LE)",
                      "reference": self.reference,
                      "nearest": "squared Euclidean distance between normalised FSQ codes ((d - L//2)/(L//2) "
                                 "per digit, levels 8,8,8,5,5,5, basis cumprod([1,8,8,8,5,5])), exact in "
                                 "integer units of 1/4; ties to the more frequent kept code"}
        r["sha256"] = {**r["sha256"], "kept_codes": self.kept_sha}
        r["n_codes"] = self.n_codes
        r["loader"] = (f"eo.experiments.segmentation.step1_A_run.load_pruned('{self.tok.name}', "
                       f"'{self.kept_path}')")
        return r


def load_pruned(cand: str, kept_path: str, device: str = "cuda") -> Pruned:
    """Rebuild a pruned tokenizer from its kept-code list (most frequent first)."""
    tok = cosmos(cand, device)
    kept = np.load(kept_path)
    counts = np.zeros(tok.n_codes, dtype=np.int64)
    counts[kept] = np.arange(len(kept), 0, -1)               # rank order reproduces the same table
    kept2, lut, _ = prune_table(counts, len(kept), device)
    assert np.array_equal(kept2, kept)
    meta = json.loads(Path(kept_path).with_suffix(".json").read_text())
    return Pruned(tok, meta["tag"], kept, lut, kept_path, meta["reference"])


# ------------------------------------------------------------------ stages

def on_edge(rule: dict) -> list[str]:
    if rule["rule"] != "region":
        return []
    e = []
    if float(rule["tau"]) == max(TAUS):
        e.append("tau_hi")
    if float(rule["tau"]) == min(TAUS):
        e.append("tau_lo")
    if int(rule["radius"]) == max(RADII):
        e.append("radius_hi")
    return e


def stage_select(tok, a) -> dict:
    t0 = time.time()
    log("select_rule on the default grid", len(TAUS), "taus x", len(RADII), "radii + nearest + vote")
    sel = H.select_rule(tok)
    log("default grid: chosen", sel["chosen"], "mean", round(sel["mean"][sel["chosen_key"]], 5),
        "per dataset", {d: v["rule"] for d, v in sel["per_dataset_optimum"].items()})
    out = {"default_grid": {"taus": list(TAUS), "radii": list(RADII)}, "default": sel,
           "seconds_default": time.time() - t0}
    final, taus, radii = sel, list(TAUS), list(RADII)
    edge = on_edge(sel["chosen"])
    if sel["chosen"]["rule"] == "region" and int(sel["chosen"]["radius"]) == min(RADII):
        out["radius_lo_note"] = "radius 1 chosen: the step below (radius 0) is the nearest rule, already read"
    if edge:
        if "tau_hi" in edge:
            taus.append(TAU_HI)
        if "tau_lo" in edge:
            taus.insert(0, TAU_LO)
        if "radius_hi" in edge:
            radii.append(R_HI)
        rules = [NEAREST, {"rule": "vote"}] + D.region_grid(tuple(taus), tuple(radii))
        log("chosen rule on the edge", edge, "-> extended grid", taus, radii)
        t1 = time.time()
        final = H.select_rule(tok, rules=rules)
        out.update(extended=final, extension=edge, extended_grid={"taus": taus, "radii": radii},
                   seconds_extended=time.time() - t1)
        still = []
        c = final["chosen"]
        if c["rule"] == "region":
            if float(c["tau"]) in (max(taus), min(taus)) and float(c["tau"]) not in (max(TAUS), min(TAUS)):
                still.append("tau")
            if int(c["radius"]) == max(radii) and int(c["radius"]) != max(RADII):
                still.append("radius")
        out["extended_chosen_on_new_edge"] = still
        log("extended grid: chosen", final["chosen"], "mean", round(final["mean"][final["chosen_key"]], 5))
    out["chosen"] = final["chosen"]
    out["chosen_key"] = final["chosen_key"]
    out["grid_used"] = {"taus": taus, "radii": radii}
    out["rules"] = [NEAREST, {"rule": "vote"}] + D.region_grid(tuple(taus), tuple(radii))
    out["seconds"] = time.time() - t0
    return out


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing {path}: run its stage first")
    return json.loads(path.read_text())


def chosen_rule(cand: str) -> tuple[dict, dict]:
    sel = _load(OUT / f"select_{cand}.json")
    return sel["chosen"], sel


def stage_read(tok, a) -> dict:
    rule, _ = chosen_rule(a.cand)
    log("read_candidate with", rule)
    res = H.read_candidate(tok, rule)
    out = {"rule": rule, "rule_from": str(OUT / f"select_{a.cand}.json"), "read": res, "bar": H.bar(res)}
    for d, v in res.items():
        log(d, "ceiling", round(v["ceiling"]["mIoU"], 5), "nearest", round(v["also"].get("nearest", v["ceiling"])["mIoU"], 5),
            "boundary", round(v["ceiling"]["boundary_acc"], 4), "codes", v["codes"]["read_distinct"],
            "unseen", round(v["codes"]["unseen_token_share"], 5))
    log("bar", out["bar"])
    return out


def _order(res: dict, rules: list[dict]) -> list[str]:
    """select_rule's own order (mIoU to 6 decimals, then smaller radius, then smaller tau)."""
    return [D.rule_key(r) for r in sorted(rules, key=lambda r: (-round(res[D.rule_key(r)]["mIoU"], 6),
                                                                   r.get("radius", 0), r.get("tau", 0)))]


def stage_valsweep(tok, a) -> dict:
    rule, sel = chosen_rule(a.cand)
    rules = sel["rules"]
    if not any(D.rule_key(r) == "region_tau0.15_r4" for r in rules):
        rules = rules + [{"rule": "region", "tau": 0.15, "radius": 4}]
    k = D.rule_key(rule)
    out = {"chosen_on_train_sub": rule, "rules": rules, "by_dataset": {}, "summary": {}}
    for d in sets.DATASETS:
        log("sweep val", d, len(rules), "rules")
        res = H.sweep(tok, d, "val", rules)
        out["by_dataset"][d] = res
        best = _order(res, rules)[0]
        out["summary"][d] = {"chosen": k, "chosen_mIoU": res[k]["mIoU"], "val_optimum": best,
                             "val_optimum_mIoU": res[best]["mIoU"], "gap": res[best]["mIoU"] - res[k]["mIoU"],
                             "nearest": res["nearest"]["mIoU"], "vote": res["vote"]["mIoU"],
                             "region_tau0.15_r4": res["region_tau0.15_r4"]["mIoU"]}
        log(d, out["summary"][d])
    keys = [D.rule_key(r) for r in rules]
    mean = {kk: float(np.mean([out["by_dataset"][d][kk]["mIoU"] for d in sets.DATASETS])) for kk in keys}
    shared = sorted(rules, key=lambda r: (-round(mean[D.rule_key(r)], 6), r.get("radius", 0), r.get("tau", 0)))[0]
    out["val_shared_optimum"] = {"rule": D.rule_key(shared), "mean": mean[D.rule_key(shared)],
                                 "chosen_mean": mean[k], "gap": mean[D.rule_key(shared)] - mean[k]}
    out["mean"] = mean
    # the read (another process) must give the chosen rule the same numbers
    rp = OUT / f"read_{a.cand}.json"
    if rp.exists():
        read = json.loads(rp.read_text())["read"]
        out["equals_read"] = {d: read[d]["ceiling"]["mIoU"] == out["by_dataset"][d][k]["mIoU"] for d in sets.DATASETS}
        log("chosen rule equals the read's ceiling:", out["equals_read"])
    return out


def stage_native256(tok, a) -> dict:
    if tok.input_size != 256:
        raise SystemExit("native256 is for 256-px tokenizers")
    rule, _ = chosen_rule(a.cand)
    log("score cloudsen12 val native256 with", rule)
    r = H.score(H.Candidate(tok, rule, crop_mode="native256"), "cloudsen12", "val", also=[NEAREST])
    log("native256 ceiling", round(r["ceiling"]["mIoU"], 5), "nearest",
        round(r["also"].get("nearest", r["ceiling"])["mIoU"], 5))     # also is empty when the rule is nearest
    return {"rule": rule, "read": r}


def stage_transposed(tok, a) -> dict:
    rule, _ = chosen_rule(a.cand)
    read = _load(OUT / f"read_{a.cand}.json")["read"]
    ctl = Progress(Transposed(tok.tok if isinstance(tok, Progress) else tok))
    out = {"rule": rule, "by_dataset": {}}
    for d in sets.DATASETS:
        log("transposed control", d)
        r = H.score(H.Candidate(ctl, rule), d, "val", cache=False, code_stats=False, also=[NEAREST])
        c, ref = r["ceiling"]["mIoU"], read[d]["ceiling"]["mIoU"]
        out["by_dataset"][d] = {"result": r, "ceiling_transposed": c, "ceiling_read": ref, "ratio": c / ref,
                                "collapses_below_half": c < 0.5 * ref}
        log(d, "transposed", round(c, 4), "read", round(ref, 4), "ratio", round(c / ref, 3))
    return out


def stage_transposed_floor(a) -> dict:
    """The transposed control's reference, on the CPU and without Cosmos: the harness's perfect tokenizer
    (eo.maps.tokenizers.Identity, which reads exactly 1.0 untransposed: gate Z2) fed the transposed renders
    and scored against the maps as they are. It is what the transposition alone leaves of a perfect
    ceiling (the maps' own symmetry and class priors): a tokenizer whose output follows its input falls to
    about this level times its own ceiling. --cand names the input size, e.g. identity_224."""
    from eo.maps.tokenizers import Identity
    size = int(a.cand.rsplit("_", 1)[1])
    torch.set_num_threads(2)
    ctl = Transposed(Identity(size, device=a.device))
    out = {"input_size": size, "device": a.device, "rule": NEAREST, "by_dataset": {}}
    for d in sets.DATASETS:
        t0 = time.time()
        r = H.score(H.Candidate(ctl, NEAREST), d, "val", cache=False, code_stats=False, device=a.device)
        out["by_dataset"][d] = {"result": r, "ceiling_transposed": r["ceiling"]["mIoU"], "seconds": time.time() - t0}
        log(d, "perfect tokenizer, transposed input:", round(r["ceiling"]["mIoU"], 4),
            "pixel", round(r["ceiling"]["pixel_acc"], 4), f"{time.time() - t0:.0f} s")
    return out


def stage_det(tok, a) -> dict:
    base = tok.tok if isinstance(tok, Progress) else tok
    out = {"batch": base.batch, "datasets": {}}
    n_maps = {"pastis": 32, "cloudsen12": 16, "lulc": 32}            # 32, 144, 32 crops: whole batches
    for d in sets.DATASETS:
        lab, _ = sets.load(d, "train_sub")
        x = H.render_crops(np.asarray(lab[:n_maps[d]]), d, "universal", base.input_size, "crop224", "cuda")
        i1 = run_fixed(base.encode, x, base.batch)
        i2 = run_fixed(base.encode, x, base.batch)
        y1 = run_fixed(base.decode, i1, base.batch)
        y2 = run_fixed(base.decode, i1, base.batch)
        r = {"crops": len(x), "codes_identical": bool(torch.equal(i1, i2)),
             "decoded_identical": bool(torch.equal(y1, y2)),
             "codes_sha256": hashlib.sha256(i1.cpu().numpy().astype(np.int32).tobytes()).hexdigest()}
        # across processes: the harness's cache for this set, written by the select process
        cpath = H.CACHE / f"{base.name}__{H._weights_tag(base)}" / \
            f"{d}_train_sub_{H._set_sha(d, 'train_sub')}_universal_crop224_b{base.batch}.npy"
        if cpath.exists():
            cached = np.load(cpath, mmap_mode="r")[:len(x)]
            r["cache_path"] = str(cpath)
            r["codes_equal_cache"] = bool(np.array_equal(cached, i1.cpu().numpy().astype(np.int32)))
        # control 1: one 32 x 32 block of crop 0 painted another universal colour
        xc = x.clone()
        xc[0, :, :32, :32] = torch.tensor([1.0, 0.0, 0.0], device=x.device)[:, None, None]
        ic = run_fixed(base.encode, xc, base.batch)
        r["control_block"] = {"crop0_codes_changed": int((ic[0] != i1[0]).sum()),
                              "other_crops_codes_changed": int((ic[1:] != i1[1:]).sum()),
                              "detected": bool(not torch.equal(ic, i1))}
        # control 2: batch shape 8 instead of 16
        i8 = run_fixed(base.encode, x, 8)
        r["control_batch8"] = {"codes_differing": int((i8 != i1).sum()), "of": int(i1.numel()),
                               "share": float((i8 != i1).float().mean())}
        # Cosmos's FSQ mapping on real codes: the encoder's normalised codes vs the digits of its indices
        with deterministic(), torch.no_grad():
            enc = base.enc((x[:base.batch] * 2 - 1).to(base.device, torch.bfloat16))
        idx = enc[0].long().cpu().numpy()
        codes = enc[1].float().cpu().numpy()                           # (B, 6, h, w)
        mine = np.moveaxis(fsq_normalised(idx), -1, 1)
        back = (fsq_digits(idx) * FSQ_BASIS).sum(-1)
        r["fsq_check"] = {"encoder_outputs": len(enc), "codes_shape": list(codes.shape),
                          "normalised_equal": bool(np.array_equal(mine.astype(np.float32), codes)),
                          "max_abs_diff": float(np.abs(mine - codes).max()),
                          "index_roundtrip": bool(np.array_equal(back, idx)),
                          "distinct_values_per_digit": [sorted(set(np.round(codes[:, j].ravel(), 4).tolist()))
                                                        for j in range(codes.shape[1])]}
        out["datasets"][d] = r
        log(d, {k: v for k, v in r.items() if k not in ("fsq_check",)}, "fsq", r["fsq_check"]["normalised_equal"],
            r["fsq_check"]["index_roundtrip"])
    out["all_identical"] = all(v["codes_identical"] and v["decoded_identical"] and v.get("codes_equal_cache", True)
                               for v in out["datasets"].values())
    out["controls_detected"] = all(v["control_block"]["detected"] for v in out["datasets"].values())
    out["fsq_mapping_ok"] = all(v["fsq_check"]["normalised_equal"] and v["fsq_check"]["index_roundtrip"]
                                for v in out["datasets"].values())
    return out


def stage_prune(tok, a) -> dict:
    base = tok.tok if isinstance(tok, Progress) else tok
    rule, _ = chosen_rule(a.cand)
    counts = np.zeros(base.n_codes, dtype=np.int64)
    per = {}
    for d in sets.DATASETS:
        t = H.tokens(base, d, "train_sub")                          # the harness's cache
        c = np.bincount(t.ravel().astype(np.int64), minlength=base.n_codes)
        per[d] = {"tokens": int(t.size), "distinct": int((c > 0).sum())}
        counts += c
    tag = "seen" if a.K == "seen" else f"K{int(a.K)}"
    kept, lut, seen = prune_table(counts, a.K)
    kept_path = OUT / "prune" / f"kept_{a.cand}_{tag}.npy"
    kept_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(kept_path, kept)
    ref = "train_sub tokens of pastis, cloudsen12, lulc pooled (eo.maps.harness.tokens, crop224, batch 16)"
    kept_path.with_suffix(".json").write_text(json.dumps({"tag": tag, "reference": ref, "K": a.K,
                                                          "kept": int(len(kept)), "seen": seen}, indent=1))
    mass = float(counts[kept].sum() / counts.sum())
    moved = int((lut != np.arange(len(lut))).sum())
    log(f"prune {tag}: kept {len(kept)} of {seen} seen codes; kept mass on train_sub {mass:.5f}")
    # share of each set's (unpruned, harness-cached) tokens whose code the prune replaces
    is_kept = np.zeros(base.n_codes, dtype=bool)
    is_kept[kept] = True
    remapped = {}
    for d in sets.DATASETS:
        remapped[d] = {s: float(1 - is_kept[H.tokens(base, d, s).ravel().astype(np.int64)].mean())
                       for s in ("train_sub", "val")}
    log("tokens remapped by the prune", remapped)
    ptok = Progress(Pruned(base, tag, kept, lut, str(kept_path), ref))
    res = H.read_candidate(ptok, rule)
    out = {"rule": rule, "K": a.K, "tag": tag, "kept": int(len(kept)), "seen_train_sub": seen,
           "train_sub_per_dataset": per, "kept_mass_train_sub": mass, "codes_remapped": moved,
           "tokens_remapped_share": remapped,
           "kept_path": str(kept_path), "kept_sha256": ptok.tok.kept_sha, "read": res, "bar": H.bar(res)}
    for d, v in res.items():
        log(d, "ceiling", round(v["ceiling"]["mIoU"], 5), "codes", v["codes"]["read_distinct"],
            "entropy", round(v["codes"]["read_entropy_bits"], 3))
    return out


STAGES = {"select": stage_select, "det": stage_det, "read": stage_read, "valsweep": stage_valsweep,
          "native256": stage_native256, "transposed": stage_transposed, "prune": stage_prune,
          "transposed_floor": None}                          # handled in main(): CPU, no Cosmos


def main():
    global _HB_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--cand", required=True)
    ap.add_argument("--K", default=None, help="prune: 'seen' or an integer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-gpu", default="TITAN V")
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--device", default="cuda", help="transposed_floor only (it runs on the CPU)")
    a = ap.parse_args()
    if a.heartbeat:
        _HB_FILE = Path(a.heartbeat)
        _HB_FILE.parent.mkdir(parents=True, exist_ok=True)
        heartbeat()
    if a.stage == "transposed_floor":                       # no Cosmos, no GPU
        t0 = time.time()
        res = stage_transposed_floor(a)
        res.update({"stage": a.stage, "cand": a.cand, "harness": H.HARNESS_VERSION, "wall_seconds": time.time() - t0,
                    "finished": time.strftime("%F %T"), "code": str(Path(__file__).resolve())})
        Path(a.out).write_text(json.dumps(res, indent=1))
        log("wrote", a.out)
        return
    name = torch.cuda.get_device_name(0)
    if a.expect_gpu and a.expect_gpu not in name:
        raise SystemExit(f"expected {a.expect_gpu}, CUDA returned {name}")
    t0 = time.time()
    log(a.stage, a.cand, a.K or "", "on", name, "pid", os.getpid())
    tok = Progress(cosmos(a.cand))
    res = STAGES[a.stage](tok, a)
    res.update({"stage": a.stage, "cand": a.cand, "gpu": name, "tokenizer": tok.record(),
                "harness": H.HARNESS_VERSION, "codes_version": H.CODES_VERSION,
                "wall_seconds": time.time() - t0, "finished": time.strftime("%F %T"),
                "code": str(Path(__file__).resolve())})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(a.out)
    log("wrote", a.out, f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
