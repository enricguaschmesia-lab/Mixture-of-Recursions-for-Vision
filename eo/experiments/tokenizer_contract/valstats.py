import sys, tarfile, zipfile, io, json, glob, os
sys.path.append('/data/enric/cache/tmp-numcodecs')
import numpy as np
from numcodecs import Blosc

ROOT='/data/enric/data/TerraMesh/val'
codec = Blosc()

def read_bands(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    meta = json.loads(z.read('bands/.zarray'))
    buf = codec.decode(z.read('bands/0.0.0.0'))
    a = np.frombuffer(buf, dtype=np.dtype(meta['dtype'])).reshape(meta['shape'])
    return a[0]  # drop time dim -> (C,H,W)

def scan(mod, shards, per_shard):
    acc = {}
    for sh in shards:
        p = f'{ROOT}/{mod}/{sh}'
        if not os.path.exists(p): continue
        corpus = sh.split('_')[0]
        d = acc.setdefault(corpus, dict(n=0, s=None, ss=None, px=0, imgstd=[]))
        tf = tarfile.open(p)
        cnt = 0
        for m in tf:
            if not m.name.endswith('.zarr.zip'): continue
            a = read_bands(tf.extractfile(m).read()).astype(np.float64)
            C = a.shape[0]
            v = a.reshape(C, -1)
            ok = ~np.isnan(v)
            if d['s'] is None:
                d['s'] = np.zeros(C); d['ss'] = np.zeros(C); d['px'] = np.zeros(C)
            d['s']  += np.where(ok, v, 0).sum(1)
            d['ss'] += np.where(ok, v, 0).__pow__(2).sum(1)
            d['px'] += ok.sum(1)
            d['imgstd'].append(np.nanstd(v, axis=1))
            d['n'] += 1; cnt += 1
            if cnt >= per_shard: break
        tf.close()
    return acc

def report(mod, acc):
    print(f'### {mod}')
    tot = dict(s=None, ss=None, px=None, n=0, imgstd=[])
    for corpus, d in acc.items():
        mu = d['s']/d['px']; sd = np.sqrt(d['ss']/d['px'] - mu**2)
        ims = np.mean(np.stack(d['imgstd']), axis=0)
        print(f'  [{corpus}] n={d["n"]}')
        print(f'    pooled mean : {np.round(mu,3).tolist()}')
        print(f'    pooled std  : {np.round(sd,3).tolist()}')
        print(f'    mean-of-img-std: {np.round(ims,3).tolist()}')
        if tot['s'] is None:
            tot['s']=d['s'].copy(); tot['ss']=d['ss'].copy(); tot['px']=d['px'].copy()
        else:
            tot['s']+=d['s']; tot['ss']+=d['ss']; tot['px']+=d['px']
        tot['n']+=d['n']; tot['imgstd'] += d['imgstd']
    if len(acc)>1:
        mu = tot['s']/tot['px']; sd = np.sqrt(tot['ss']/tot['px'] - mu**2)
        print(f'  [COMBINED] n={tot["n"]}')
        print(f'    pooled mean : {np.round(mu,3).tolist()}')
        print(f'    pooled std  : {np.round(sd,3).tolist()}')
        print(f'    mean-of-img-std: {np.round(np.mean(np.stack(tot["imgstd"]),axis=0),3).tolist()}')
    print()

MT = ['majortom_shard_%06d.tar'%i for i in range(1,82,2)]
SS = ['ssl4eos12_shard_%06d.tar'%i for i in range(1,10)]
for mod, shards in [('DEM', MT+SS), ('NDVI', MT+SS), ('S2L2A', MT+SS),
                    ('S1RTC', MT), ('S1GRD', SS)]:
    report(mod, scan(mod, shards, 25))
