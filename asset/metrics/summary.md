| Checkpoint | Direction | Primary metric | 95% CI | FLOPs ratio | Failure rate | N |
|---|---|---|---|---|---|---|
| baseline_vanilla | tok_rgb@256_within | tf_perplexity=56.5627 [54.0668, 59.3642] | 1.000 | 0.000 | 699 |
| baseline_vanilla | rgb_caption | clip_score=34.1216 [33.9345, 34.3059] | 1.000 | 0.011 | 699 |
| baseline_vanilla | caption_rgb | pixel_mse=695.7513 [666.6207, 723.7785] | 1.000 | 0.000 | 699 |
| mor_5000_3r | tok_rgb@256_within | tf_perplexity=73.0771 [69.4323, 76.8125] | 0.537 | 0.000 | 699 |
| mor_5000_3r | rgb_caption | clip_score=33.9183 [33.7214, 34.1061] | 0.537 | 0.001 | 699 |
| mor_5000_3r | caption_rgb | pixel_mse=658.4759 [633.9351, 680.8108] | 0.537 | 0.000 | 699 |
| mor_5000_4r | tok_rgb@256_within | tf_perplexity=84.0775 [79.9471, 88.6479] | 0.454 | 0.000 | 699 |
| mor_5000_4r | rgb_caption | clip_score=34.0684 [33.8975, 34.2573] | 0.454 | 0.003 | 699 |
| mor_5000_4r | caption_rgb | pixel_mse=709.6903 [685.2807, 736.0279] | 0.454 | 0.006 | 699 |
| random_router_5000 | tok_rgb@256_within | tf_perplexity=77.1196 [72.7404, 81.3905] | 0.668 | 0.000 | 501 |
| random_router_5000 | rgb_caption | clip_score=34.0935 [33.8940, 34.2932] | 0.668 | 0.038 | 501 |
| random_router_5000 | caption_rgb | pixel_mse=816.6841 [779.5019, 852.8130] | 0.668 | 0.058 | 501 |
