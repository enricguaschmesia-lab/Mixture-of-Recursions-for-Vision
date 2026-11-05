import os, json
os.environ.setdefault('HF_HOME','/data/enric/hf')
from huggingface_hub import hf_hub_download, HfApi
api = HfApi()
REPOS = {
 'terramind_v1_tokenizer_s2l2a': ('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S2L2A','TerraMind_Tokenizer_S2L2A.pt'),
 'terramind_v1_tokenizer_s1rtc': ('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1RTC','TerraMind_Tokenizer_S1RTC.pt'),
 'terramind_v1_tokenizer_s1grd': ('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-S1GRD','TerraMind_Tokenizer_S1GRD.pt'),
 'terramind_v1_tokenizer_dem':   ('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-DEM','TerraMind_Tokenizer_DEM.pt'),
 'terramind_v1_tokenizer_lulc':  ('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-LULC','TerraMind_Tokenizer_LULC.pt'),
 'terramind_v1_tokenizer_ndvi':  ('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-NDVI','TerraMind_Tokenizer_NDVI.pt'),
 'terramind_v1_coords_tokenizer':('ibm-esa-geospatial/TerraMind-1.0-Tokenizer-Coords','config.json'),
}
out={}
for name,(repo,fn) in REPOS.items():
    sha = api.model_info(repo).sha
    p = hf_hub_download(repo_id=repo, filename=fn)
    sz = os.path.getsize(p)
    out[name]=dict(repo=repo, filename=fn, revision=sha, path=p, bytes=sz)
    print(f'{name:34s} {sz/1e9:6.3f} GB  rev={sha}', flush=True)
    if name=='terramind_v1_coords_tokenizer':
        for extra in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']:
            try: hf_hub_download(repo_id=repo, filename=extra)
            except Exception as e: print('   (skip',extra,')')
json.dump(out, open('/data/enric/weights/tokenizer_manifest.json','w'), indent=1)
print('MANIFEST WRITTEN')
