# `eo/experiments/`: one-off analyses, by topic

Each script produced one result or one set of figures and is kept as the record of how. They are
not a pipeline: nothing under `eo/` depends on them, except that `scripts/tokenize_clean.py` borrows
`clamp_test`. Reusable logic belongs in a package under `eo/`, and CLI entry points in `eo/scripts/`.

| Folder | What it holds | Env |
|---|---|---|
| `tokenizer_contract/` | The bring-up that established the preprocessing contract (`s3*.py`), the DiVAE clamp investigation (`clamp_test`, `clamp_generated`, `code_value_test`), S1 metrics, and small data checks. `clamp_test` is also a shared helper: it decodes tokens with and without the clamp | `mor` |
| `s2_offset/` | The +1000 DN BOA offset in S2L2A: the scans, its effect on tokens and routing, the checks behind `val_clean`, and the upstream-version check `s2_offset_train_check.py` | `mor`, some `.venv` |
| `generation/` | Generation against held-out truth: CE vs decoded metrics (`p1_ce_vs_decoded`), the any-to-any grid and its controls and chains (`s12_*`), and the head-to-head with TerraMind (`tm_compare_*`) | `.venv` to generate, `mor` to decode |
| `bigearthnet/` | BigEarthNet: tokens and round trips, spatial leakage, test analyses, forgetting, summaries and figures. `ben_t360_summary.py` is the current report | `.venv` and `mor` |
| `pastis/` | PASTIS-R inspection | `mor` |
| `pretrained_init/` | Pretrained-body inits (CLEVR, SmolLM): layer drop, comparisons, slides | `.venv` |
| `routing/` | Depth sweeps with random and learned routers: reports and numbers | `.venv` |
| `temporal/` | The temporal stage's checks: S1 alignment on train (`s1_dates`, `mor`; `s1_functional`) and what T360 does with dates it never saw together (`t360_context`, `t360_lattice`, `t360_best_layouts`) | `.venv`, `s1_dates` in `mor` |
| `continuous_io/` | The continuous input/output pilot (`cio_*`): data, training, decoding, analysis, slides | `.venv`, `mor` to decode |
| `figures/` | `midterm_figures.py` (the house plotting style, imported by other slide scripts) and `t360_slides.py` | `.venv` |

Run a script from the repo root, for example:

    .venv/bin/python eo/experiments/routing/depth_sweep_report.py

The `s3*.py` scripts and a few others import `terramesh_tok` directly, so they need `PYTHONPATH=eo`.
Most write under `/data/enric`: large outputs never go under `/home`.
