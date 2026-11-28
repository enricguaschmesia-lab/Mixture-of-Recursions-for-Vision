#!/usr/bin/env python
"""GEO-Bench-2's leaderboard on CloudSen12 (Enric's question, 2026-11-28): each model's test mIoU over its 5 seeds (mean, SD,
range), full fine-tuning and frozen, from the leaderboard Space's raw results CSVs (huggingface.co/spaces/aialliance/
GEO-Bench-2-Leaderboard, results/<id>/results_and_parameters.csv; the Space was last modified 2025-11-20). Copies and
SHA-256 under /data/enric/reports/a2_seg/leaderboard/gb2_space_2025-11-20/. `.venv`, CPU.

    python eo/experiments/segmentation/gb2_leaderboard_cloudsen12.py /data/enric/reports/a2_seg/leaderboard/gb2_space_2025-11-20/*/results_and_parameters.csv
"""
import sys
import pandas as pd
rows = []
for f in sys.argv[1:]:
    d = pd.read_csv(f)
    d["file"] = f.split("/")[-2][:8]
    rows.append(d)
d = pd.concat(rows)
c = d[(d.dataset == "cloudsen12") & (d.mlflow_run_status == "FINISHED")].copy()
print("metrics:", c.Metric.unique(), "decoders:", c.decoder.unique(), "partitions:", c["partition name"].unique() if "partition name" in c else "")
g = c.groupby(["frozen_or_full_ft", "backbone", "file"])["test metric"].agg(["count", "mean", "std", "min", "max"]).reset_index()
g["exp"] = c.groupby(["frozen_or_full_ft", "backbone", "file"])["experiment_name"].first().values
for mode in ("full_ft", "frozen"):
    print(f"\n== {mode}")
    for _, r in g[g.frozen_or_full_ft == mode].sort_values("mean", ascending=False).iterrows():
        print(f"{r['mean']*100:6.2f}  sd {r['std']*100:4.2f}  [{r['min']*100:5.2f}, {r['max']*100:5.2f}]  n={r['count']}  {r['backbone']}  ({r['file']}, {r['exp']})")
print("\nseeds:", sorted(c.Seed.unique()), "| hpo lr/bs of the top full-ft:", c[c.backbone == g[g.frozen_or_full_ft == 'full_ft'].sort_values('mean').iloc[-1].backbone][['lr', 'batch_size']].drop_duplicates().values.tolist())
