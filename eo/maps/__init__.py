"""Label maps as RGB-rendered images and their tokenizers (Paper A, A2: segmentation as map tokens).

The shared harness of A2 Step 1. Every ceiling, decoding rule and code statistic for a map tokenizer is
read through `eo.maps.harness.score`, never through an agent's own scorer.

    palette   the universal 32-colour palette and each task's fixed class -> colour assignment
    sets      the fixed map sets (training pools and read sets) and their manifest
    tiling    CloudSen12's 3 x 3 overlapping crops and the stitching of class scores
    decode    decoded RGB or palette logits -> class scores -> class map (nearest and region rules)
    scoring   GEO-Bench-2's multiclass Jaccard (torchmetrics' macro semantics), boundary accuracy,
              code statistics
    tokenizers  the map-tokenizer interface; Cosmos's discrete JITs (bf16) behind it
    harness   score(candidate, dataset, split) and the rule sweep on training maps

Runs in the `a2seg` conda env (a clone of `mor` with GeoBenchV2 0.9), never in `.venv`.
"""
