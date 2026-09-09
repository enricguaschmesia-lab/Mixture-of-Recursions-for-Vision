"""Step 3.5 round-trip + 3.6 control experiment + 3.7 S1 alignment + norm A/B.

Metrics to stdout; figures to /data/enric/figures/step3/.
"""
import os, json, warnings, numpy as np, torch, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_fn

from terramesh_tok import contract as C, io, preprocess as P, tokenizers as T

FIG = "/data/enric/figures/step3"
os.makedirs(FIG, exist_ok=True)
DEV, TIMESTEPS, SEED = T.DEVICE, 50, 0
META = pd.read_parquet("/data/enric/data/TerraMesh/val_metadata.parquet")
MT = META[META.tar.str.startswith("majortom")]
SS = META[META.tar.str.startswith("ssl4eos12")]


def pick(df, rows):
    return [(r.tar, r.zarr[: -len(".zarr.zip")], r) for r in rows]


SAMPLES = pick(MT, [MT[MT.center_lat.abs() < 8].iloc[0],
                    MT[MT.center_lat.abs() > 55].iloc[0],
                    MT.sort_values("cloud_cover", ascending=False).iloc[0]])
SAMPLES_SS = pick(SS, [SS.iloc[0], SS.iloc[400], SS.iloc[3000]])

print("=== Samples (Step 3.5, deliberately diverse) ===")
for tag, S in (("majortom", SAMPLES), ("ssl4eos12", SAMPLES_SS)):
    for tar, stem, p in S:
        print(f"  [{tag}] {stem:26s} lat={p.center_lat:8.3f} "
              f"lon={p.center_lon:9.3f} cloud={p.cloud_cover:.3f}")
print()


def cont_metrics(ref, rec):
    err = rec - ref
    rmse = float(np.sqrt((err ** 2).mean()))
    rng = float(ref.max() - ref.min())
    psnr = float(20 * np.log10(rng / rmse)) if rmse > 0 and rng > 0 else float("nan")
    s = float(np.mean([ssim_fn(ref[c], rec[c],
                               data_range=float(ref[c].max() - ref[c].min()) or 1.0)
                       for c in range(ref.shape[0])]))
    return dict(rmse=rmse, mae=float(np.abs(err).mean()), psnr=psnr, ssim=s)


def lulc_metrics(ref, rec, n=C.LULC_N_CLASSES):
    ious = []
    for k in range(n):
        u = np.logical_or(ref == k, rec == k).sum()
        if u > 0:
            ious.append(np.logical_and(ref == k, rec == k).sum() / u)
    return dict(pixel_acc=float((ref == rec).mean()), mIoU=float(np.mean(ious)),
                n_classes_present=len(ious))


def decode(tok, g):
    gen = torch.Generator().manual_seed(SEED)  # CPU: pipeline makes noise on CPU
    return tok.decode_tokens(g, timesteps=TIMESTEPS, generator=gen)


def phys(mod, x, arr=None):
    """standardized tensor -> physical-unit numpy (C,H,W)."""
    return P.destandardize(x.float(), mod).cpu().numpy()


def rt(mod, tok, arr, **kw):
    """Full round trip. Returns (tokens_flat, ref_phys, rec_phys)."""
    x, _ = P.prepare(arr, mod, **kw)
    g = T.encode(tok, x)
    rec = decode(tok, g)
    if mod == "LULC":
        o, c = C.CROP_OFF, C.CROP
        ref = arr[0, 0][o:o + c, o:o + c].astype(np.int64)
        return P.flatten_tokens(g), ref, rec.argmax(1)[0].cpu().numpy()
    st = kw.get("stats", "v1")
    if not kw.get("standardize", True):
        ref, pred = x[0].cpu().numpy(), rec[0].float().cpu().numpy()
    else:
        ref, pred = phys(mod, x[0]), phys(mod, rec[0])
    if kw.get("reverse_bands"):
        ref, pred = ref[::-1].copy(), pred[::-1].copy()
    return P.flatten_tokens(g), ref, pred


# ============================================================ 3.5
print(f"=== 3.5 Round-trip, correct contract (timesteps={TIMESTEPS}, seed={SEED}) ===")
results, tokens_out, figs = {}, {}, {}
for mod in ["DEM", "NDVI", "LULC", "S1RTC", "S1GRD", "S2L2A"]:
    tok = T.build(mod)
    S = SAMPLES_SS if mod == "S1GRD" else SAMPLES
    rows = []
    for tar, stem, p in S:
        try:
            _, arr = io.read_sample(mod, tar, stem)
        except (KeyError, FileNotFoundError):
            continue
        f, ref, rec = rt(mod, tok, arr)
        m = lulc_metrics(ref, rec) if mod == "LULC" else cont_metrics(ref, rec)
        m.update(uniq=int(len(torch.unique(f))), tmin=int(f.min()), tmax=int(f.max()))
        rows.append((stem, m))
        tokens_out.setdefault(mod, {})[stem] = f[0].cpu().numpy().astype(int).tolist()
        figs.setdefault(mod, (ref, rec, stem))
    results[mod] = rows
    print(f"  {mod}")
    for stem, m in rows:
        core = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in m.items() if k not in ("tmin", "tmax"))
        print(f"    {stem:26s} {core}  range=[{m['tmin']},{m['tmax']}]/{C.CODEBOOK[mod]}")
print()

# figures
for mod, (ref, rec, stem) in figs.items():
    if mod == "LULC":
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
        ax[0].imshow(ref, cmap="tab10", vmin=0, vmax=9); ax[0].set_title("input classes")
        ax[1].imshow(rec, cmap="tab10", vmin=0, vmax=9); ax[1].set_title("decoded classes")
        ax[2].imshow(ref != rec, cmap="Reds"); ax[2].set_title("disagreement")
    else:
        b = {"S2L2A": 3, "S1RTC": 0, "S1GRD": 0}.get(mod, 0)
        r, q = ref[b], rec[b]
        lo, hi = np.percentile(r, [2, 98])
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
        ax[0].imshow(r, vmin=lo, vmax=hi, cmap="gray"); ax[0].set_title(f"input (band {b})")
        ax[1].imshow(q, vmin=lo, vmax=hi, cmap="gray"); ax[1].set_title("reconstruction")
        im = ax[2].imshow(q - r, cmap="RdBu_r"); ax[2].set_title("error")
        plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{mod} — {stem} — 256 crop, {TIMESTEPS} DDIM steps")
    fig.tight_layout(); fig.savefig(f"{FIG}/roundtrip_{mod}.png", dpi=110)
    plt.close(fig)
print(f"figures -> {FIG}/roundtrip_<MOD>.png")
print()

# ============================================================ 3.6 controls
print("=== 3.6 Control experiment: what does WRONG look like? ===")
CTRL = {}
for mod in ["DEM", "NDVI", "S1RTC", "S2L2A"]:
    tok = T.build(mod)
    tar, stem, _ = SAMPLES[0]
    _, arr = io.read_sample(mod, tar, stem)
    f0, ref0, rec0 = rt(mod, tok, arr)
    base = cont_metrics(ref0, rec0)
    row = {"correct": base["rmse"]}

    # (0) trivial baseline: predict the per-image per-band mean
    mean_pred = np.broadcast_to(ref0.mean(axis=(1, 2), keepdims=True), ref0.shape)
    row["mean-baseline"] = cont_metrics(ref0, mean_pred)["rmse"]

    # (a) reversed band order (only meaningful for C>1)
    if C.N_CHANNELS[mod] > 1:
        _, r, q = rt(mod, tok, arr, reverse_bands=True)
        row["reversed-bands"] = cont_metrics(r, q)["rmse"]

    # (b) no standardization
    _, r, q = rt(mod, tok, arr, standardize=False)
    row["no-standardization"] = cont_metrics(r, q)["rmse"]

    # (c) wrong modality's statistics
    other = "NDVI" if mod != "NDVI" else "DEM"
    xw = P.prepare(arr, mod, standardize=False)[0]
    mw = torch.tensor(C.V1_TOK_MEAN[other] * C.N_CHANNELS[mod],
                      device=DEV)[: C.N_CHANNELS[mod]].view(1, -1, 1, 1)
    sw = torch.tensor(C.V1_TOK_STD[other] * C.N_CHANNELS[mod],
                      device=DEV)[: C.N_CHANNELS[mod]].view(1, -1, 1, 1)
    gw = T.encode(tok, (xw - mw) / sw)
    qw = (decode(tok, gw) * sw + mw)[0].float().cpu().numpy()
    row[f"stats-of-{other}"] = cont_metrics(ref0, qw)["rmse"]

    # (d) 224 crop into a 256-built model
    f2, r2, q2 = rt(mod, tok, arr, crop=224)
    row["crop-224"] = cont_metrics(r2, q2)["rmse"]
    row["_crop224_grid"] = int(np.sqrt(f2.shape[1]))

    CTRL[mod] = row
    g = int(row.pop("_crop224_grid"))
    print(f"  {mod:6s} RMSE (physical units):")
    for k, v in row.items():
        ratio = v / row["correct"]
        flag = "  <-- CORRECT" if k == "correct" else (
            "  !! not separated" if ratio < 1.5 else "")
        print(f"      {k:22s} {v:12.4f}   x{ratio:6.2f}{flag}")
    print(f"      (224 crop produced a {g}x{g} token grid = {g*g} tokens)")
print()

# ============================================================ norm A/B
print("=== Normalization A/B: v1_pretraining_* vs terramesh_statistics.yaml ===")
print(f"  {'MOD':6s} {'tokens differing':>17s} {'of':>5s}   note")
for mod in ["DEM", "S2L2A", "S1GRD", "S1RTC", "NDVI"]:
    tok = T.build(mod)
    S = SAMPLES_SS if mod == "S1GRD" else SAMPLES
    diffs = []
    for tar, stem, _ in S:
        try:
            _, arr = io.read_sample(mod, tar, stem)
        except (KeyError, FileNotFoundError):
            continue
        a = T.encode(tok, P.prepare(arr, mod, stats="v1")[0])
        b = T.encode(tok, P.prepare(arr, mod, stats="yaml")[0])
        diffs.append(int((a != b).sum()))
    same_stats = C.V1_TOK_MEAN[mod] == C.YAML_MEAN[mod] and \
                 C.V1_TOK_STD[mod] == C.YAML_STD[mod]
    note = "NULL CONTROL (stats identical -> must be 0)" if same_stats else \
           f"stats differ"
    print(f"  {mod:6s} {str(diffs):>17s} {256:5d}   {note}")
print()

# ============================================================ 3.7 S1 alignment
print("=== 3.7 S1GRD vs S1RTC codebook alignment ===")
tg, tr = T.build("S1GRD"), T.build("S1RTC")
tar, stem, _ = SAMPLES_SS[0]
_, arr = io.read_sample("S1GRD", tar, stem)
xg, _ = P.prepare(arr, "S1GRD", stats="v1")
xr, _ = P.prepare(arr, "S1RTC", stats="v1")
a = P.flatten_tokens(T.encode(tg, xg))[0]
b = P.flatten_tokens(T.encode(tr, xr))[0]
c = P.flatten_tokens(T.encode(tr, xg))[0]   # same standardization, other tokenizer
print(f"  identical SAR tensor -> GRD tokenizer vs RTC tokenizer")
print(f"    own-standardization : {int((a == b).sum())}/256 token IDs equal")
print(f"    same-standardization: {int((a == c).sum())}/256 token IDs equal")
print(f"    GRD first 12: {a[:12].tolist()}")
print(f"    RTC first 12: {b[:12].tolist()}")
inter = len(set(a.tolist()) & set(b.tolist()))
print(f"    vocabulary overlap of the two 256-token sets: {inter}")
print(f"    expected if unrelated (256 draws from 15360): ~{256*256/15360:.1f}")

json.dump({"roundtrip": results, "controls": CTRL},
          open(f"{FIG}/metrics.json", "w"), indent=1, default=float)
json.dump(tokens_out, open(f"{FIG}/tokens.json", "w"))
print(f"\nwrote {FIG}/metrics.json, {FIG}/tokens.json")
