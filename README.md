# terramesh-tok

TerraMesh -> TerraMind FSQ token preprocessing contract.

Established and verified in Phase 1 Step 3 (see the project worklog and
notes/tokenizer_bringup.md). **This package is the single source of truth for
the contract** — Step 4 batch tokenization and Phase 4 routing analysis must
import it rather than re-derive any constant.

    terramesh_tok/contract.py     constants: crop, stats, codebooks, flatten order
    terramesh_tok/io.py           reading TerraMesh WebDataset shards
    terramesh_tok/preprocess.py   prepare(): raw array -> encoder-ready tensor
    terramesh_tok/tokenizers.py   build / encode / decode
    experiments/                  the Step 3 verification scripts, kept as the record

Environment: the `mor` conda env (Python 3.11, torch 2.5.1+cu121,
terratorch 1.2.11). Requires HF_HOME=/data/enric/hf.
