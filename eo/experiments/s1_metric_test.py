"""Which continuous metric separates the ceiling from a spatial shuffle? (2026-09-24)

Why this exists: on 16 identity scenes, S1RTC's W7 shuffle control collapsed
only 1.15x on channel-0 RMSE, below W7's 1.25x margin, and S1GRD 1.26x. The
question was whether a perceptual metric (LPIPS) could score S1 generations
while ignoring speckle. This script is the measurement that answered it, and
the evidence for decode_eo.ML_WINDOW.

For each modality it decodes, over 64 corpus-stratified held-out scenes
(generate_eo.py --identity --n-scenes 64 --out <CACHE>/identity64_<MOD>),
the ground-truth tokens (ceiling) and the same tokens spatially permuted
(shuffled), caches both, and scores every candidate metric at three |z|
thresholds: ratio shuffled/ceiling with a paired bootstrap 95% CI, the share
of scenes where the shuffle is worse (win), and d = mean gap / SD of the gap.

Candidates: rmse_z raw and after a 5 / 7 px box multilook; LPIPS (AlexNet,
VGG; per-band greyscale and, for S1, a (VV, VH, VV-VH) false colour); and an
unweighted perceptual distance in a SAR-trained backbone (SSL4EO-S12 MoCo
ResNet-50, torchgeo SENTINEL1_ALL_MOCO).

Outcome, recorded in docs/worklog.md 2026-09-24: 7 px multilooked rmse_z
beat every LPIPS variant and the SAR backbone on both S1 modalities, and left
DEM unchanged; VGG-LPIPS was the weakest metric tried.

Runs in `mor` (needs terratorch, plus `pip install --no-deps lpips`), from
the repo root, on the Titan X:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
        python eo/experiments/s1_metric_test.py S1RTC S1GRD DEM
"""
import sys, pathlib, json, numpy as np, torch
sys.path.insert(0, "eo/scripts"); sys.path.insert(0, "eo")
import decode_eo as D
from terramesh_tok import contract as C, preprocess as P, tokenizers as T
from scipy.ndimage import uniform_filter
import lpips
from torchgeo.models import ResNet50_Weights, resnet50

CACHE = pathlib.Path("/data/enric/generations/metric_test")
MODS = sys.argv[1:] or ["S1RTC", "S1GRD", "DEM"]
Z_THRESH = (1.0, 0.75, 0.5)
dev = T.DEVICE
D.assert_gpu(False)

def decode_cached(mod):
    f = CACHE / f"decoded_{mod}.npz"
    if f.exists():
        return dict(np.load(f))
    rows = np.load(CACHE / f"identity64_{mod}/slot_masked/rows_{mod}.npy")
    art = np.load(D.VAL / C.tok_dir_name(mod) / "tokens.npy", mmap_mode="r")
    truth = np.asarray(art[rows]).astype(np.int64).reshape(-1, C.GRID, C.GRID)
    rng = np.random.default_rng(0)
    shuf = np.stack([s.ravel()[rng.permutation(C.TOKENS_PER_SAMPLE)].reshape(C.GRID, C.GRID) for s in truth])
    tok = T.build(mod, device=dev)
    out = {n: D.decode_batched(tok, g, 4, 0, dev, 50).numpy() for n, g in (("ceiling", truth), ("shuffled", shuf))}
    del tok; torch.cuda.empty_cache()
    rasters = D.read_rasters(mod, rows)
    mean = np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None]; std = np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None]
    refs = []
    for a in rasters:
        r = (P.center_crop(a[0] if a.ndim == 4 else a).astype(np.float32) - mean) / std
        refs.append(np.where(np.isnan(r), 0.0, r))       # NaN -> mean, = contract NAN_POLICY
    out["ref"] = np.stack(refs); out["z"] = np.array([D.scene_z(mod, a) for a in rasters]); out["rows"] = rows
    np.savez(f, **out)
    return out

# ---- metrics: each takes (pred, ref) as (N, C, H, W) standardized arrays, returns (N,) ----
def rmse_box(w):
    def f(p, r):
        if w > 1:
            p = uniform_filter(p, size=(1, 1, w, w)); r = uniform_filter(r, size=(1, 1, w, w))
        return np.sqrt(((p - r) ** 2).mean(axis=(1, 2, 3)))
    return f

_lp = {}
def _lpips_net(net):
    if net not in _lp:
        _lp[net] = lpips.LPIPS(net=net, verbose=False).to(dev).eval()
    return _lp[net]

@torch.no_grad()
def _lp_batch(net, a, b):
    m = _lpips_net(net); out = []
    for i in range(0, len(a), 8):
        x = torch.from_numpy(a[i:i+8]).float().to(dev); y = torch.from_numpy(b[i:i+8]).float().to(dev)
        out.append(m(x, y).flatten().cpu().numpy())
    return np.concatenate(out)

def lpips_gray(net):
    """Each band replicated to 3 channels, scored separately, averaged over bands. Clipped to the decoder's [-1,1]."""
    def f(p, r):
        p, r = np.clip(p, -1, 1), np.clip(r, -1, 1)
        return np.mean([_lp_batch(net, np.repeat(p[:, c:c+1], 3, 1), np.repeat(r[:, c:c+1], 3, 1))
                        for c in range(p.shape[1])], axis=0)
    return f

def lpips_comp(net):
    """S1 false colour (VV, VH, (VV-VH)/2). One arbitrary mapping of several possible."""
    def f(p, r):
        def rgb(x):
            x = np.clip(x, -1, 1); return np.concatenate([x[:, :1], x[:, 1:2], (x[:, :1] - x[:, 1:2]) / 2], 1)
        return _lp_batch(net, rgb(p), rgb(r))
    return f

_s1 = {}
@torch.no_grad()
def s1net(mod):
    """LPIPS-style distance in a SAR-trained backbone (SSL4EO-S12 MoCo ResNet-50, VV/VH).
    Unit-normalise each layer's features over channels, squared difference, spatial mean,
    summed over layer1..layer4. No learned weights -- a plain 'perceptual loss'."""
    if "m" not in _s1:
        _s1["m"] = resnet50(weights=ResNet50_Weights.SENTINEL1_ALL_MOCO).to(dev).eval()
    m = _s1["m"]
    tmean = np.array(C.V1_TOK_MEAN[mod], np.float32)[:, None, None]; tstd = np.array(C.V1_TOK_STD[mod], np.float32)[:, None, None]
    nmean = np.array([-12.59, -20.26], np.float32)[:, None, None]; nstd = np.array([5.26, 5.91], np.float32)[:, None, None]
    def feats(x):
        x = m.conv1(x); x = m.bn1(x); x = m.act1(x); x = m.maxpool(x)
        fs = []
        for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
            x = layer(x); fs.append(x / (x.norm(dim=1, keepdim=True) + 1e-10))
        return fs
    def f(p, r):
        out = []
        for i in range(0, len(p), 8):
            a = ((p[i:i+8] * tstd + tmean) - nmean) / nstd; b = ((r[i:i+8] * tstd + tmean) - nmean) / nstd
            with torch.no_grad():
                fa, fb = feats(torch.from_numpy(a).float().to(dev)), feats(torch.from_numpy(b).float().to(dev))
            out.append(sum(((x - y) ** 2).sum(1).mean((1, 2)) for x, y in zip(fa, fb)).cpu().numpy())
        return np.concatenate(out)
    return f

def metrics_for(mod):
    ms = {"rmse_z": rmse_box(1), "rmse_z_box5": rmse_box(5), "rmse_z_box7": rmse_box(7),
          "lpips_alex_gray": lpips_gray("alex"), "lpips_vgg_gray": lpips_gray("vgg")}
    if mod.startswith("S1"):
        ms["lpips_alex_rgb"] = lpips_comp("alex"); ms["lpips_vgg_rgb"] = lpips_comp("vgg")
        ms["s1net"] = s1net(mod)
    return ms

def summarize(c, s, rng):
    ratio = s.mean() / c.mean()
    idx = rng.integers(0, len(c), (2000, len(c)))
    boot = s[idx].mean(1) / c[idx].mean(1)
    d = s - c
    return dict(n=len(c), ceiling=float(c.mean()), shuffled=float(s.mean()), ratio=float(ratio),
                ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                win=float((s > c).mean()), d=float(d.mean() / (d.std(ddof=1) + 1e-12)))

results = {}
for mod in MODS:
    dec = decode_cached(mod)
    print(f"\n== {mod}  (64 scenes; in-range at |z|<1.0: {(dec['z']<1).sum()}, <0.75: {(dec['z']<.75).sum()}, <0.5: {(dec['z']<.5).sum()})", flush=True)
    per = {}
    for name, fn in metrics_for(mod).items():
        per[name] = (fn(dec["ceiling"], dec["ref"]), fn(dec["shuffled"], dec["ref"]))
    results[mod] = {}
    print(f"  {'metric':16s} {'|z|<':>5s} {'n':>3s} {'ceiling':>9s} {'shuffled':>9s} {'ratio':>6s} {'95% CI':>13s} {'win':>5s} {'d':>5s}")
    for name, (c, s) in per.items():
        for zt in Z_THRESH:
            k = dec["z"] < zt
            if k.sum() < 3:
                continue
            r = summarize(c[k], s[k], np.random.default_rng(0))
            results[mod][f"{name}@{zt}"] = r
            print(f"  {name:16s} {zt:5.2f} {r['n']:3d} {r['ceiling']:9.4f} {r['shuffled']:9.4f} {r['ratio']:5.2f}x "
                  f"[{r['ci_lo']:.2f},{r['ci_hi']:.2f}] {r['win']:5.2f} {r['d']:5.2f}", flush=True)
(CACHE / f"metric_test_results_{'_'.join(MODS)}.json").write_text(json.dumps(results, indent=2))
print("\nALL DONE")
