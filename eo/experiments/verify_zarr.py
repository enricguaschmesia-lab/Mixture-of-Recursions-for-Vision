import io, tarfile, zipfile, json, time, numpy as np, fsspec, zarr, warnings
from numcodecs import Blosc
warnings.filterwarnings('ignore')
ROOT='/data/enric/data/TerraMesh/val'
codec=Blosc()
def first_member(mod, shard='majortom_shard_000009.tar'):
    tf=tarfile.open(f'{ROOT}/{mod}/{shard}')
    for m in tf:
        if m.name.endswith('.zarr.zip'): return m.name, tf.extractfile(m).read()
def via_zarr(raw):
    mp=fsspec.filesystem('zip', fo=io.BytesIO(raw), block_size=None).get_mapper('')
    g=zarr.open_consolidated(mp, mode='r', zarr_format=2)
    extra={k:np.asarray(g[k][...]).tolist() for k in ('center_lon','center_lat') if k in g}
    extra['has_cloud_mask']='cloud_mask' in g
    extra['bandnames']=[str(b) for b in g['band'][...]]
    return g['bands'][...], extra
def via_direct(raw):
    z=zipfile.ZipFile(io.BytesIO(raw)); meta=json.loads(z.read('bands/.zarray'))
    buf=codec.decode(z.read('bands/0.0.0.0'))
    return np.frombuffer(buf, dtype=np.dtype(meta['dtype'])).reshape(meta['shape'])
print(f"{'MOD':6s} {'shape':22s} {'dtype':8s} {'summary':34s} {'zarr==direct':12s}")
for mod in ['S2L2A','NDVI','DEM','S1RTC','S1GRD','LULC','S2L1C','S2RGB']:
    sh='majortom_shard_000009.tar' if mod!='S1GRD' else 'ssl4eos12_shard_000002.tar'
    name,raw=first_member(mod,sh); a,extra=via_zarr(raw); b=via_direct(raw)
    af=a.astype('f8')
    if mod in ('S2L2A','S2L1C','S2RGB'): summ=f'B[0]mean={np.nanmean(af[0,0]):.1f}'
    elif mod=='NDVI': summ=f'mean={np.nanmean(af):.3f}'
    elif mod=='DEM': summ=f'range={np.nanmin(af):.0f}-{np.nanmax(af):.0f} m'
    elif mod in ('S1RTC','S1GRD'): summ=f'vv={np.nanmean(af[0,0]):.1f} vh={np.nanmean(af[0,1]):.1f} dB'
    else: summ=f'classes={sorted(np.unique(a).tolist())}'
    summ+=f' nan={int(np.isnan(af).sum())}'
    print(f'{mod:6s} {str(a.shape):22s} {str(a.dtype):8s} {summ:34s} {str(np.array_equal(a,b)):12s} bands={extra["bandnames"]} cm={extra["has_cloud_mask"]}')
import terratorch
from terratorch.models.backbones.terramind.tokenizer.tokenizer_register import vqvae_available, tokenizers_available
print('terratorch OK; vqvae=',vqvae_available,'tokenizers=',tokenizers_available)
