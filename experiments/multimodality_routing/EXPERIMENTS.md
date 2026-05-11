# Multimodality MoR routing — experiments guide

This doc walks through each experiment in `scripts/eval_multimodality_routing.py`,
explaining **what we measure**, **why**, **how to read the figure**, and **what
the released checkpoint (`gbasi18/Vanilla-10000-Multimodality`) actually shows**.

## Setup recap

The checkpoint is a 29-layer SmolLM-135M with **token-choice MoR**, `Nr=3`
recursions, `middle_cycle` weight sharing. Architecturally:

```
layer 0  →  [MoR block: 9 shared sub-blocks × 3 recursions]  →  layer 28
```

For every token the router picks an exit depth `k ∈ {1, 2, 3}` (= it goes
through the middle stack 1, 2, or 3 times). A token assigned depth `k` uses
**k× the compute** of a token assigned depth `1`. The 4 trained modalities are:

- `caption` — GPT-2 text tokens
- `tok_rgb@256` — 256 Cosmos VQ tokens of the RGB image
- `tok_depth@256` — 256 tokens of the depth map
- `tok_normal@256` — 256 tokens of the surface-normals image

We probe the router on N test scenes from CLEVR.

---

## E1 — Per-modality depth distribution

**What it does.** Build a sequence containing all four modalities (caption + rgb
+ depth + normal, ~770 tokens). Run one forward pass with no KV cache and
record the depth choice for every token. Aggregate over `N` test samples.

**What we're testing.** Does the router treat every modality the same (uniform
1/3-1/3-1/3 thanks to the balancing loss), or does it allocate compute
differently per modality?

**How to read the figure (`E1_modality_distribution.png`).** Stacked bars, one
per modality. Bar height = fraction of tokens assigned each depth, with the
mean depth printed above.

**What we found.** Four very distinct compute lanes:

| modality          | mean | dominant |
|-------------------|------|----------|
| `tok_normal@256`  | 1.10 | 90% d=1  |
| `tok_rgb@256`     | 1.38 | 62% d=1, 38% d=2 |
| `caption`         | 1.96 | 94% d=2  |
| `tok_depth@256`   | 2.70 | 70% d=3  |

The router is essentially using depth as a **modality label**, not as a
fine-grained difficulty estimator. CLEVR normals are smooth, low-entropy
images, so the router shortcuts them. Depth-token entropy is high (Cosmos
codebook usage on CLEVR depth is more spread out), so it spends max budget.

---

## E2 — Spatial routing maps for images

**What it does.** For each image modality alone (no other-modality context),
reshape the 256 token depths into the 16×16 grid and average across `N`
samples.

**What we're testing.** Is the spatial pattern stable across samples (router
"learned the grid"), or is it sample-specific (follows object positions)?

**How to read the figure (`E2_spatial_maps.png`).** Two rows × three columns.
Top row: per-cell mean depth. Bottom row: per-cell std *across samples*.

- Solid color, low std → router decision is a **fixed function of position**.
- Speckled, high std → router decision depends on the input content.

**What we found.** `tok_rgb@256` shows a clean border-vs-center structure
(border patches more often need d=2). `tok_depth@256` is uniformly hot.
`tok_normal@256` is uniformly cold. All three have low per-cell std → the
router is roughly **position-conditioned**, not content-conditioned.

---

## E3 — Context effect on target routing

**What it does.** Fix a target modality (e.g. `tok_depth`) and feed it with
four different conditioning prefixes:

| context        | prompt                                          |
|---------------|-------------------------------------------------|
| `none`        | `<BO_tgt> body <EO_tgt>`                        |
| `caption`     | caption chunk + target chunk                    |
| `rgb`         | rgb chunk + target chunk                        |
| `caption+rgb` | both context chunks + target chunk              |

Measure the **mean exit depth on the target body only**.

**What we're testing.** If extra context makes prediction easier, an adaptive
router should ask for shallower recursion on the target. Does it?

**How to read the figure (`E3_context_effect.png`).** Four panels, one per
target modality. Each bar = mean depth with std bars across samples.

**What we found.** All bars are essentially flat — adding caption or rgb
shifts target mean depth by **<0.05**. The router decides on **token
identity** (which modality am I?), not on **predictability given context**.
This is a key negative result: the router on this checkpoint is **not** using
cross-modal context to adapt compute.

---

## E4 — Generation vs teacher-forced routing

**What it does.** For each image modality:
1. Run the model on the **ground-truth tokens** (teacher-forced) and record
   the routing on the body. → `tf` mean depth.
2. Have the model **generate** the body unconditionally (sampling, T=0.9),
   then route through that generated body. → `gen` mean depth.

**What we're testing.** Is the per-modality depth signature a property of the
inputs we feed (GT tokens), or does it persist on tokens the model rolled out
itself? If the router really learned a modality-conditioned policy, the two
should match.

**How to read the figure (`E4_gen_vs_tf.png`).** Side-by-side bars per
modality.

**What we found.** TF and gen mean depth agree within ~0.1 on all three image
modalities. → The per-modality fingerprint is a stable property of the
router, not an artifact of the conditioning tokens.

---

## E5 — Forced-depth ablation (the key causal probe)

**What it does.** Compare 4 modes on per-modality teacher-forced cross-entropy
(token-level prediction loss, lower = better):

1. **`router`** — the trained router, as-is.
2. **`d1`** — every token forced to depth 1 (¼ compute).
3. **`d2`** — every token forced to depth 2 (½ compute).
4. **`d3`** — every token forced to depth 3 (max compute).

**Important implementation detail.** A naive "force depth k" replaces the
router's softmax with a peaked one-hot — but that also changes the **gating
weight** the chosen expert is multiplied by (from ~0.4 to ~1.0), confounding
"depth k" with "weight 1.0". To isolate routing alone, we monkey-patch
`torch.topk` (only inside the MoR forward) so the argmax is overridden but the
**original softmax weight at the chosen expert is preserved**.

**What we're testing.** Does the router actually do useful work beyond just
picking *some* fixed depth? In a 3-recursion world, is `d=2` already
"good enough" everywhere, or are there modalities where router-style
adaptivity is necessary?

**How to read the figure (`E5_forced_depth_ce.png`).**
- Left: per-modality CE under each mode (log y).
- Right: ΔCE vs the trained router. Bars above zero = forcing that fixed
  depth is **worse** than the trained router.

**What we found** on the released checkpoint:

| mode    | caption | tok_rgb | **tok_depth** | tok_normal |
|---------|---------|---------|---------------|------------|
| router  | 1.31    | 4.17    | **2.15**      | 2.14       |
| d=1     | 2.12    | 4.75    | **9.15** 💥   | 2.31       |
| d=2     | 1.31    | 4.17    | **2.15**      | 2.15       |
| d=3     | 1.41    | 4.17    | **2.16**      | 2.14       |

Three messages:
1. **`d=2` is a free baseline** — it matches the router on every modality.
2. **`d=1` is catastrophic on `tok_depth`** (CE jumps to 9 nats — model is
   helpless without recursion). It also hurts caption.
3. **`d=3` slightly hurts caption** — extra compute is not strictly better.

In other words, on this checkpoint the router's job is to **avoid depth=1 on
hard modalities**. Adaptivity beyond that is not buying much yet, plausibly
because (a) only 10k training steps, (b) only `Nr=3` so the "medium" choice
is already near-optimal.

---

## How to use the script

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_multimodality_routing.py \
    --config infer/vanilla_10000_multimodality \
    --n-samples 16             # E1, E2 (cheap)
    --n-quality-samples 8      # E3, E5 (each forward is Nr× slower in E5)
    --n-gen-samples 3          # E4 (full image generation; slow)
```

Outputs land in `results/eval/<config-name>/` as `E1..E5.png` plus per-experiment
JSON files and a combined `summary.json`.

Swap `--config` to point at any token-choice MoR multimodality checkpoint that
has these four modalities active (e.g. a re-trained variant). E2 requires
square image bodies; E5 only requires that the model has a single MoR module
exposed (works for `middle_cycle`).

---

## TL;DR — what the experiments collectively tell us about MoR-for-multimodality

- The router collapses to **per-modality compute classes**, not per-token
  difficulty.
- That classification is robust: it survives changing the context (E3) and
  changing the input from GT to generated tokens (E4).
- The classification is **functionally useful**: it avoids a catastrophic d=1
  collapse on `tok_depth` (E5).
- Beyond that, with only 3 recursions, **a fixed d=2 baseline is competitive**
  with the trained router — suggesting most of MoR's benefit on this
  checkpoint is captured by giving hard modalities access to deeper recursion
  *at all*, not by token-level adaptivity.
