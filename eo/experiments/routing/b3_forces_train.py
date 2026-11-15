#!/usr/bin/env python
"""B3's train-split control, prediction 2: on data seen once, does the held-out minus training gate force stay near its
epoch-1 value? On val it grew from +0.02 at epoch 1 to +0.26 (A10c) and +0.42 (A10c-s2) at the end, with the held-out
minus training loss gap (r +0.79 over 16 (run, checkpoint) pairs; B0's supplementary checks,
/data/enric/reports/b0/questions/q1_overfit.py). The prediction, written 2026-11-08 (Paper B plan, B3): it stays near
+0.02 instead of climbing to +0.26-0.42.

The force is B0 Step 2's, computed by `b0_forces.one_unit` unchanged: dL/dw, the LM loss's derivative with respect to
the chosen depth's gate. q1 reads the mean of dL/dw over the image modalities not at a single depth (gate < 0.9).

Rows.
  heldout  B0's: 256 of the 4,416 held-out rows, stratified by corpus (seed 20261107), A10c-tr's config.
  seen     rows A10c-tr trained on, 256 per checkpoint c, seen in the 330 updates up to c (the most recently seen, where
           memorising would show most), stratified by corpus.
  unseen   256 rows never seen in the run, the same at every checkpoint: same distribution, never trained on, so seen
           against unseen isolates memorising from the held-out split's own difference.
Training rows come from the 219 shards tokenized on the workstation (train_tok/; the rest are on g49 only).

Which rows A10c-tr saw. One process: HF's RandomSampler, replaced by accelerate's SeedableRandomSampler, seeded with
torch's initial seed (set_seed(42)) at epoch 0, so the k-th scene trained on is torch.randperm(8,929,841, generator
seeded 42)[k] in the root's row order (tok_index.parquet, which is train_metadata.parquet's), 256 an update. GATE, before
anything is selected: the ssl4eos12 rows this predicts in every update must equal the S1GRD positions A10c-tr logged
(depth_shares.jsonl) / 196 at all 3,300 updates, and seeds 43 and 0 must not (2026-11-15: 3,300 of 3,300; 182; 174).
Each row is served in the modality order it was trained in: the dataset's RNG is keyed on (seed, worker, row); the row is
its global id, and the worker is the one torch dispatched its micro-batch to (the per-device batch of 4, round-robin
over 4 workers; the logs cannot check this part).

The mini root (OUT/root) holds only the selected rows, in the format `train_split.consolidate` writes, so
TerraMeshTokenDataset reads it as a partial train root. CONTROLS on it: 16 rows of shard 780 equal the one-shard root
train_tok_partial's tokens bit for bit in every modality, and the same rows shifted by one do not.

Rule, fixed 2026-11-15 before any unit ran: the prediction HOLDS if held-out minus seen is below +0.10 at 3300 (under
half of val's smaller end value, +0.26); it FAILS if it reaches +0.26; otherwise PARTLY.

    .venv/bin/python eo/experiments/routing/b3_forces_train.py select
    .venv/bin/python eo/experiments/routing/b3_forces_train.py build
    CUDA_VISIBLE_DEVICES=<titanx> .venv/bin/python eo/experiments/routing/b3_forces_train.py unit heldout|seen|unseen CK
    .venv/bin/python eo/experiments/routing/b3_forces_train.py read

Writes under /data/enric/reports/wave1/forces_train/. Runs in `.venv`; units on a GPU (about 95 s each on the Titan X).
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

OUT = Path("/data/enric/reports/wave1/forces_train")
ROOT = OUT / "root"
TT = Path("/data/enric/data/TerraMesh/train_tok")
PARTIAL = Path("/data/enric/data/TerraMesh/train_tok_partial")
COORDS = Path("/data/enric/data/TerraMesh/train_coords/Coords_tok")
VAL_META = Path("/data/enric/data/TerraMesh/val_clean")
RUN = Path("/data/enric/runs/pretrain/phase3/mor10ctr_20261111_200306")
CFG = "eo_terramesh/arm_a10ctr_mor"
# MB, the micro-batch torch hands to one worker, is the per-device batch: 4 (with 64 accumulation steps for the update
# of 256), read again from the run's trainer_state.json at selection. A first version assumed 64 and dispatched
# three in four seen rows to the wrong worker, hence another modality order (found at the width check's resume,
# whose preflight line reads "per-device 4"; 2026-11-15).
N, B, MB, WORKERS, STEPS, SEED = 8_929_841, 256, 4, 4, 3300, 42
CKS = [330, 660, 990, 1320, 1980, 2640, 2970, 3300]
MODS = ["S2L2A", "S1GRD", "S1RTC", "DEM", "NDVI", "LULC"]
OUT_MODS = {"majortom": ["S2L2A", "S1RTC", "DEM", "NDVI", "LULC"], "ssl4eos12": ["S2L2A", "S1GRD", "DEM", "NDVI", "LULC"]}
N_ROWS, SEL_SEED = 256, 20261115
HOLDS, FAILS = 0.10, 0.26


def tok_dir(m: str) -> str:
    from eo.terramesh_tok import contract as C
    return C.tok_dir_name(m)


def order(seed: int) -> np.ndarray:
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(N, generator=g).numpy()[: STEPS * B]


def select() -> int:
    from eo.experiments.routing import b3_depth_shares as DS
    idx = pd.read_parquet(TT / "tok_index.parquet")
    meta = pd.read_parquet("/data/enric/data/TerraMesh/train_meta/train_metadata.parquet", columns=["zarr"])
    assert len(idx) == N and np.array_equal(idx.row.values, np.arange(N))
    assert (idx.stem.values == meta.zarr.str.replace(".zarr.zip", "", regex=False).values).all()
    ssl = (idx.corpus.values == "ssl4eos12")
    classes, steps, counts = DS.load(RUN / "depth_shares.jsonl")
    obs = counts[:, classes.index("S1GRD"), :].sum(1)
    assert np.all(obs % 196 == 0)
    obs = (obs / 196).astype(int)
    gate = {}
    for s in (SEED, 43, 0):
        pred = ssl[order(s)].reshape(STEPS, B).sum(1)
        gate[s] = int((pred[steps - 1] == obs).sum())
    print(f"gate: updates whose ssl4eos12 count matches A10c-tr's logged S1GRD positions: seed 42 {gate[42]} of "
          f"{len(obs)}, controls seed 43 {gate[43]}, seed 0 {gate[0]}")
    if not (gate[42] == len(obs) == STEPS and gate[43] < STEPS // 2 and gate[0] < STEPS // 2):
        print("GATE FAILED: the reconstructed order is not A10c-tr's; nothing selected")
        return 1
    ts = json.loads((RUN / "checkpoint-3300" / "trainer_state.json").read_text())
    if int(ts["train_batch_size"]) != MB:
        print(f"the run's per-device batch is {ts['train_batch_size']}, not {MB}: the worker map would be wrong")
        return 1
    perm = order(SEED)
    k = np.arange(STEPS * B)
    update = np.zeros(N, dtype=np.int32)
    update[perm] = k // B + 1
    worker = np.full(N, -1, dtype=np.int8)
    worker[perm] = (k // MB) % WORKERS
    # local shards with every modality their corpus has
    have = {m: {f[:-4] for f in os.listdir(TT / tok_dir(m) / "shards") if f.endswith(".npy") and not f.endswith(".nan.npy")}
            for m in MODS}
    shards = idx.source_shard.str.replace(".tar", "", regex=False).values
    local_sh = {s for s in set(have["S2L2A"]) if all(s in have[m] for m in OUT_MODS["ssl4eos12" if s.startswith("ssl4eo") else "majortom"])}
    local = np.isin(shards, list(local_sh))
    print(f"local shards {len(local_sh)} ({sum(s.startswith('ssl4eo') for s in local_sh)} ssl4eos12), rows {int(local.sum())}, "
          f"of them seen {int((local & (update > 0)).sum())}")
    rng = np.random.default_rng(SEL_SEED)

    def strat(cand: np.ndarray) -> np.ndarray:
        pool = np.where(cand)[0]
        keep = []
        for c in ("majortom", "ssl4eos12"):
            hv = pool[idx.corpus.values[pool] == c]
            keep.append(rng.choice(hv, size=int(round(N_ROWS * len(hv) / len(pool))), replace=False))
        return np.sort(np.concatenate(keep))

    sets = {"unseen": strat(local & (update == 0))}
    for c in CKS:
        sets[f"seen_{c}"] = strat(local & (update > c - 330) & (update <= c))
    # the build control: 16 rows of shard 780, which train_tok_partial holds
    p_idx = pd.read_parquet(PARTIAL / "tok_index.parquet")
    first = int(np.where(idx.stem.values == p_idx.stem.iloc[0])[0][0])
    sets["check780"] = np.arange(first, first + 16)
    for name, r in sets.items():
        ns = int((idx.corpus.values[r] == "ssl4eos12").sum())
        upd = update[r]
        print(f"  {name:10s} {len(r)} rows ({ns} ssl4eos12), updates {upd.min()}-{upd.max()}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "selection.npz", update=update, worker=worker, **{k: v for k, v in sets.items()})
    (OUT / "selection.json").write_text(json.dumps({"gate": gate, "local_shards": len(local_sh),
                                                    "sets": {k: v.tolist() for k, v in sets.items()}}) + "\n")
    print(f"-> {OUT / 'selection.npz'}")
    return 0


def build() -> int:
    S = np.load(OUT / "selection.npz")
    idx = pd.read_parquet(TT / "tok_index.parquet")
    names = [k for k in S.files if k.startswith("seen_") or k in ("unseen", "check780")]
    glob_rows = np.unique(np.concatenate([S[k] for k in names]))
    n = len(glob_rows)
    sub = idx.iloc[glob_rows].reset_index(drop=True)
    shards = sub.source_shard.str.replace(".tar", "", regex=False).values
    first_of = {s: int(g.row.values[0]) for s, g in idx[idx.source_shard.isin(set(sub.source_shard))].groupby("source_shard")}
    pos_in = np.array([glob_rows[i] - first_of[sub.source_shard.values[i]] for i in range(n)])
    if ROOT.exists():
        raise SystemExit(f"{ROOT} exists: remove it to rebuild")
    ROOT.mkdir(parents=True)
    out_idx = pd.DataFrame({"row": np.arange(n), "stem": sub.stem.values, "source_shard": sub.source_shard.values,
                            "corpus": sub.corpus.values})
    out_idx.to_parquet(ROOT / "tok_index.parquet", index=False)
    meta_mods = {}
    for m in MODS:
        tokens = np.zeros((n, 196), dtype=np.uint16)
        present = np.zeros(n, dtype=bool)
        for s in sorted(set(shards)):
            f = TT / tok_dir(m) / "shards" / f"{s}.npy"
            if not f.exists():
                continue
            a = np.load(f, mmap_mode="r")
            sel = np.where(shards == s)[0]
            tokens[sel] = a[pos_in[sel]]
            present[sel] = True
        d = ROOT / tok_dir(m)
        d.mkdir()
        np.save(d / "tokens.npy", tokens)
        np.save(d / "present.npy", present)
        tmpl = json.loads((VAL_META / tok_dir(m) / "metadata.json").read_text())
        tm = tokens[present]
        meta = {**tmpl, "split": "train", "n_samples_total_rows": n, "n_samples_present": int(present.sum()),
                "token_min": int(tm.min()) if len(tm) else None, "token_max": int(tm.max()) if len(tm) else None,
                "source": "b3_forces_train.py build: rows of train_tok's local shards (A10c-tr's seen and unseen rows)"}
        (d / "metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
        meta_mods[m] = {"present": int(present.sum()), "token_max": meta["token_max"]}
    ct, cp = np.load(COORDS / "tokens.npy", mmap_mode="r"), np.load(COORDS / "present.npy", mmap_mode="r")
    d = ROOT / "Coords_tok"
    d.mkdir()
    np.save(d / "tokens.npy", np.asarray(ct[glob_rows]))
    np.save(d / "present.npy", np.asarray(cp[glob_rows]))
    cmeta = json.loads((COORDS / "metadata.json").read_text())
    cmeta.update({"n_samples_total_rows": n, "source": "b3_forces_train.py build: rows of train_coords"})
    (d / "metadata.json").write_text(json.dumps(cmeta, indent=1) + "\n")
    build = {"split": "train", "rows": n, "shards": len(set(shards)), "partial": True, "offset_free": True,
             "modalities": meta_mods,
             "note": "B3 forces_train mini root (2026-11-15): A10c-tr's seen and never-seen rows, and 16 rows of shard "
                     "780 for the build control; offset_free as the full train root's build.json records (C3 passed)"}
    (ROOT / "build.json").write_text(json.dumps(build, indent=1) + "\n")
    np.savez(OUT / "root_map.npz", global_row=glob_rows, worker=S["worker"][glob_rows], update=S["update"][glob_rows])
    # CONTROL: the shard-780 rows equal train_tok_partial's, bit for bit; shifted by one they do not
    p_idx = pd.read_parquet(PARTIAL / "tok_index.parquet")
    ok, shift_differs = True, True
    mine = {s: i for i, s in enumerate(out_idx.stem.values)}
    rows_p = np.arange(16)
    rows_m = np.array([mine[s] for s in p_idx.stem.values[:16]])
    for m in MODS + ["Coords"]:
        dm = ROOT / ("Coords_tok" if m == "Coords" else tok_dir(m))
        dp = PARTIAL / ("Coords_tok" if m == "Coords" else tok_dir(m))
        a, b = np.load(dm / "tokens.npy")[rows_m], np.load(dp / "tokens.npy")[rows_p]
        pa, pb = np.load(dm / "present.npy")[rows_m], np.load(dp / "present.npy")[rows_p]
        same = np.array_equal(a, b) and np.array_equal(pa, pb)
        ok &= same
        if pb.any():
            shift_differs &= not np.array_equal(np.load(dm / "tokens.npy")[rows_m], np.load(dp / "tokens.npy")[rows_p + 1])
        print(f"  control {m:6s}: 16 shard-780 rows equal train_tok_partial's = {same}")
    print(f"  control: shifted by one row they differ = {shift_differs}")
    print(f"mini root {ROOT}: {n} rows, {len(set(shards))} shards")
    return 0 if (ok and shift_differs) else 1


def _mini_dataset():
    os.environ["TERRAMESH_TRAIN_TOK_ROOT"] = str(ROOT)
    from lm_dataset.load_dataset import load_dataset_from_config
    from eo.generate import conditional as G   # noqa: F401  (Hydra composition helpers below)
    from hydra import compose, initialize_config_dir
    conf = Path(__file__).resolve().parents[3] / "conf" / "pretrain_vision"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        cfg = compose(config_name=CFG, overrides=["++multimodal.allow_partial_train_root=true"])
    ds = load_dataset_from_config(cfg)
    assert Path(ds.root_dir) == ROOT, ds.root_dir
    mp = np.load(OUT / "root_map.npz")
    gl, wk = mp["global_row"], mp["worker"]

    def exact_rng(self, row: int) -> np.random.Generator:
        w = int(wk[row])
        return np.random.default_rng(seed=(self.seed, w if w >= 0 else 0, int(gl[row])))

    ds._get_rng = types.MethodType(exact_rng, ds)
    return ds, gl


def unit(which: str, ck: int) -> int:
    from eo.experiments.routing import b0_forces as BF
    out_root = OUT / which
    if which == "heldout":
        BF.one_unit("a10ctr", str(RUN), f"checkpoint-{ck}", CFG, N_ROWS, str(out_root), 4, "heldout")
        return 0
    S = np.load(OUT / "selection.npz")
    want = S["unseen" if which == "unseen" else f"seen_{ck}"]
    ds, gl = _mini_dataset()
    pos_of = {int(g): i for i, g in enumerate(gl)}
    rows = np.array([pos_of[int(g)] for g in want])
    table = pd.read_parquet(ROOT / "tok_index.parquet")
    saved = (BF.TF.eval_loader, BF.load_row_table, BF.sample_rows, BF.S.eval_groups)
    BF.TF.eval_loader = lambda cfg, num_workers=4: (ds, None)
    BF.load_row_table = lambda root: table
    BF.sample_rows = lambda pool, corpus, n: rows
    BF.S.eval_groups = lambda r: np.arange(len(r))          # training rows: each row its own bootstrap unit
    try:
        BF.one_unit("a10ctr", str(RUN), f"checkpoint-{ck}", CFG, len(rows), str(out_root), 4, "heldout")
    finally:
        BF.TF.eval_loader, BF.load_row_table, BF.sample_rows, BF.S.eval_groups = saved
    f = out_root / "a10ctr" / f"checkpoint-{ck}.json"
    rep = json.loads(f.read_text())
    rep.update({"rows_from": f"train_{which}", "root": str(ROOT), "global_rows": [int(g) for g in want],
                "row_bootstrap": "per row (eval_groups replaced)"})
    f.write_text(json.dumps(rep, indent=1) + "\n")
    return 0


def q1_mean(u: dict, mods=None) -> float:
    v = [r["dL_dw"] for m, r in u["mods"].items() if m != "Coords" and r["gate"] < 0.9 and (mods is None or m in mods)]
    return float(np.mean(v)) if v else float("nan")


def boot_mean(npz: Path, rep: dict, units: np.ndarray, mods: list[str], draws=2000, seed=0) -> np.ndarray:
    """Bootstrap draws of the q1 mean: rows resampled by unit, each modality's dL/dw = sum dw / sum n."""
    z = np.load(npz)
    dw, n = z["dw"], z["n"]                       # (rows, 7)
    J = [MODS.index(m) for m in mods]
    u_ids, inv = np.unique(units, return_inverse=True)
    sdw = np.stack([np.bincount(inv, weights=dw[:, j], minlength=len(u_ids)) for j in J], 1)
    sn = np.stack([np.bincount(inv, weights=n[:, j], minlength=len(u_ids)) for j in J], 1)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for b in range(draws):
        w = np.bincount(rng.integers(0, len(u_ids), len(u_ids)), minlength=len(u_ids))
        out[b] = np.mean((w @ sdw) / (w @ sn))
    return out


def read() -> int:
    from eo.routing import stats as S
    lines, res = [], {}

    def P(*a):
        t = " ".join(str(x) for x in a)
        lines.append(t)
        print(t, flush=True)

    P("== Val (B0's stored units, q1's reading): mean dL/dw over image modalities with gate < 0.9")
    for t in ("a10c", "a10cs2"):
        row = []
        for c in CKS:
            try:
                h = json.loads(Path(f"/data/enric/reports/b0/forces/{t}/checkpoint-{c}.json").read_text())
                r = json.loads(Path(f"/data/enric/reports/b0/forces_train/{t}/checkpoint-{c}.json").read_text())
                row.append(f"{c} {q1_mean(h) - q1_mean(r):+.3f}")
            except FileNotFoundError:
                row.append(f"{c} -")
        P(f"  {t:7s} held-out minus training: " + ", ".join(row))
    P("\n== A10c-tr (train split, 1 + 9 + 1, Coords): held-out, seen in the 330 updates up to c, never seen")
    res["a10ctr"] = {}
    for c in CKS:
        f = {w: OUT / w / "a10ctr" / f"checkpoint-{c}.json" for w in ("heldout", "seen", "unseen")}
        if not all(x.exists() for x in f.values()):
            P(f"  {c}: units missing")
            continue
        u = {w: json.loads(x.read_text()) for w, x in f.items()}
        mods = [m for m in MODS if all(m in u[w]["mods"] and u[w]["mods"][m]["gate"] < 0.9 for w in u)]
        q = {w: q1_mean(u[w]) for w in u}
        qb = {w: q1_mean(u[w], mods) for w in u}
        units = {"heldout": S.eval_groups(np.load(str(f["heldout"]).replace(".json", ".npz"))["rows"])}
        for w in ("seen", "unseen"):
            units[w] = np.arange(len(np.load(str(f[w]).replace(".json", ".npz"))["rows"]))
        bd = {w: boot_mean(Path(str(f[w]).replace(".json", ".npz")), u[w], units[w], mods, seed=i)
              for i, w in enumerate(u)}
        d_hs, d_hu, d_su = bd["heldout"] - bd["seen"], bd["heldout"] - bd["unseen"], bd["seen"] - bd["unseen"]
        ci = lambda x: np.percentile(x, [2.5, 97.5])
        res["a10ctr"][c] = dict(q1=q, both_mods=mods, q1_both=qb, ho_minus_seen=qb["heldout"] - qb["seen"],
                                ho_minus_seen_ci=ci(d_hs).tolist(), ho_minus_unseen=qb["heldout"] - qb["unseen"],
                                ho_minus_unseen_ci=ci(d_hu).tolist(), seen_minus_unseen=qb["seen"] - qb["unseen"],
                                seen_minus_unseen_ci=ci(d_su).tolist())
        P(f"  {c:4d} q1 (each unit's modalities): held-out {q['heldout']:+.3f}, seen {q['seen']:+.3f}, unseen "
          f"{q['unseen']:+.3f}; held-out minus seen {q['heldout'] - q['seen']:+.3f}")
        P(f"       on the modalities below 0.9 in all three ({', '.join(mods)}): held-out minus seen "
          f"{qb['heldout'] - qb['seen']:+.3f} [{ci(d_hs)[0]:+.3f}, {ci(d_hs)[1]:+.3f}], held-out minus unseen "
          f"{qb['heldout'] - qb['unseen']:+.3f} [{ci(d_hu)[0]:+.3f}, {ci(d_hu)[1]:+.3f}], seen minus unseen "
          f"{qb['seen'] - qb['unseen']:+.3f} [{ci(d_su)[0]:+.3f}, {ci(d_su)[1]:+.3f}]")
    if 3300 in res["a10ctr"]:
        v = res["a10ctr"][3300]["q1"]["heldout"] - res["a10ctr"][3300]["q1"]["seen"]
        verdict = "HOLDS" if v < HOLDS else ("FAILS" if v >= FAILS else "PARTLY")
        res["verdict"] = dict(ho_minus_seen_3300=v, verdict=verdict)
        P(f"\nRule (fixed before the units): held-out minus seen at 3300 {v:+.3f} -> the prediction {verdict} "
          f"(holds below +{HOLDS}, fails from +{FAILS})")
    (OUT / "read.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (OUT / "read.txt").write_text("\n".join(lines) + "\n")
    P(f"-> {OUT / 'read.txt'}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if a == ["select"]:
        return select()
    if a == ["build"]:
        return build()
    if len(a) == 3 and a[0] == "unit" and a[1] in ("heldout", "seen", "unseen"):
        return unit(a[1], int(a[2]))
    if a == ["read"]:
        return read()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
