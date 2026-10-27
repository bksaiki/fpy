# mmasim serving: LLMs through the emitted MMA-Sim kernels

Code under `examples/mmasim/serve/`.  Every `nn.Linear` of a real LLM
computes through a Triton kernel emitted from an MMA-Sim design, under a
quantization scheme (BF16, FP8, MX, NVFP4), and the effect of each design's
arithmetic is measured.

**Status.**
- **Done:** the pipeline for BF16 and every quantized scheme; Qwen3-0.6B
  (RTN and three checkpoints) and Qwen3.5-0.8B results.
- **Primary evaluation: local per-layer metrics on cached activations.** A
  grid search over designs cannot afford the end-to-end evaluations, which
  take hours per design.
- **Done:** Stage 1 of the Roadmap.  **Parked:** Stage 2 (local error
  predicts the end-to-end effect, which is small at inference).  **Next:**
  Stage 3, its plan in the Roadmap.
- **Left:** the Roadmap, Essential gaps and Nice to have.

## Roadmap

**The point is feasibility.**  The FPy-to-Triton compiler turns
bit-accurate models of MMA hardware into kernels.  That makes it possible
to evaluate MMA designs, real or hypothetical, on real LLMs and real
workloads, even on a GPU with no such hardware: a TITAN V (sm_70, 2017),
with no FP8 or FP4 tensor cores.  The deliverable is that this evaluation
can be done at all, and at what cost.  An interesting trend in which design
characteristics matter would be a bonus, not the claim.

What is reported to show it, at every stage:
- **Fidelity:** every design's kernel is checked bit for bit against the
  FPy interpreter (`compile_triton.py -r`).
- **Throughput:** designs evaluated per GPU-hour on the TITAN V, per
  evaluation level; compile time per design; kernel speed against cuBLAS.
- **Validity:** the cheap evaluations rank designs as the expensive ones do,
  with uncertainty (Stage 2).

### Stage 1 -- Tractable end-to-end evaluations

**Done** (Metrics; Record):
- seeded subsets, paired statistics against R0 and R1, and teacher-forced
  divergence;
- measured sizes: 50 perplexity segments, KL against R1, separate the
  widest design pairs (4-15 min per design).  Decode ranks poorly and stays
  the generation view.  Zero-shot's continuous scores beat its accuracy but
  rank worse than perplexity, so it is the capability confirmation
  (`--items 500`, ~50 min per slow design);
- several designs per pass in local metrics, sharing their exact products;
  captured activations cached on disk; designs compiled in parallel;
- `serve/` split into `core/` plus scripts; chat under any scheme; leaner
  quantizers and scoring.

### Stage 2 -- Do local metrics predict end-to-end effects?

Local per-layer metrics on cached
activations are the primary evaluation because they are cheap (5-30 s a
design); end-to-end evaluations take minutes to hours.  The claim that the
cheap evaluation stands in for the expensive one needs evidence: that
designs ranked by local error rank the same by their end-to-end effect.

End to end, two measures, each a different question (Record, "Two
measures"):

- **the device's effect, quantization held fixed:** KL(R1 || design), how
  far the design's arithmetic moves the quantized model's output.  Its
  scale is the scheme's **K₀**, the cost of any inexact design (a
  perturbation of R1 flips downstream roundings to the scheme's grid),
  estimated by KL(R1 || R1 rounded toward zero); a design adds a dose to it
  linear in log2 of its backward error (Record, "The dose sweep");
- **its harm:** KL(R0 || design) - KL(R0 || R1), paired over segments,
  whether that movement makes the model worse than its exact product would.
  It is ~0 but for FP8 truncation, because the device's effect is
  quantization's error re-drawn, not added to (Record, "What the device's
  effect is").

**Status: parked (2026-10-01).**  Local error predicts the device's effect,
KL(R1 || design), when it is the normwise backward error: linear in its
log2 along both sweeps (~0.7-0.8% of K₀ a doubling), the split-k and width
mechanisms on one line, every resolved pair within a scheme agreeing.  But
the effect is small at inference: the best to the worst FP8 accumulator
(~2^12 in error) moves it 10% of K₀, and its harm is ~0 (the design
re-draws quantization's error, it does not add to it); the accumulator's
share of NLL is 0.02-0.06% against FP8 quantization's 0.45%.  No end-to-end
quantity here moves enough to be worth correlating further.  Unpark for
training (where DeepSeek-V3's concern lies), free generation, or a design
whose error lies far outside these.

**Reopened (2026-10-20): frozen runs** (Record, "Frozen runs").  K₀ is
re-rounding, not arithmetic: frozen to R1 (each layer multiplies R1's
quantized input, its own offset from it passed through the weight
unrounded), a design's KL is its arithmetic alone.  R1 rounded toward zero
falls from K₀ (1.3e-02) to 1.2e-11; Hopper is 8x10^6 Blackwell; the width
sweep is KL ∝ ε², 2.00 bits of KL a bit of normwise error.  Free runs stay
the deployment view.

The drivers and analysis this record names (`serve/stage2/`: `run.sh`, `after.sh`, `sweep.sh`, `splitk.sh`, `k0.sh`, `sweep.py`, `analyze.py`, `plot.py`) and their results were never committed (`4809280` does not hold them).  The width sweep's designs are the generator's Hopper E4M3 anchor with `F` stepped and `rho` toward zero at `min(F, 23)` bits: `F` 13 is Hopper's design, `F` 25 measures as Blackwell's.  Its changes to `serve/core` stay: the `<scheme>-exact-rtz` and `-exact-rand` runs (`swap.py`), references in host memory (`scoring.py`), and the local metrics (`metrics.py`).

#### Decisions

The Roadmap left these open; each is closed here with its reason, and says
what would reopen it.

##### Which designs, under which scheme, with which weights?

All 13 designs that take a scheme, each under the scheme it was built for:

| scheme | designs |
|---|---|
| `bf16` | nv.ampere.bf16.f32, nv.hopper.bf16.f32, amd.cdna2.bf16, amd.cdna2.bf16_1k, amd.cdna3.bf16 |
| `fp8-row` | nv.ada.e4m3.f32, nv.hopper.e4m3.f32, nv.blackwell.e4m3.f32 |
| `fp8-row:fnuz` | amd.cdna3.fp8 |
| `mxfp8` | nv.blackwell.mx.e4m3 |
| `mxfp4` | nv.blackwell.mx.e2m1, nv.blackwell.mxfp4 |
| `nvfp4` | nv.blackwell.nvfp4 |

- **`fp8-row`, not `fp8-block`, for the E4M3 designs:** one chain over all
  of `k` is where their accumulators differ most (Ada's truncation); a
  promotion every 128 shortens it, shrinking the spread Stage 2 needs.
  It is also Stage 1's scheme, so the two stages pair.
- **RTN weights from the BF16 master for every scheme,** not the
  checkpoints: one weight source across schemes, and a known master for the
  pre-quantization reference.  Checkpoints differ in more than arithmetic
  (their own quantizers and calibration).  Reopen for a
  checkpoint-specific claim.

##### Against which reference?

Local error against the quantized operands' exact product; end to end,
both references (revised after the first results):

- **R1** (`<scheme>-exact`) for the device's effect: it holds quantization
  fixed, so KL(R1 || design) is the design's arithmetic alone.
- **R0** (`fp32`) for harm, as a difference paired with R1:
  KL(R0 || design) - KL(R0 || R1) per segment.  Quantization's cost is in
  both terms and cancels; what is left is whether the design moves the
  model toward or away from R0.  KL(R0 || design) alone would rank schemes.

##### Pooled over schemes, or within each?

Revised twice; the Record has the history.

- **The device's effect: within each scheme only.**  Its scale is the
  scheme's: any perturbation of R1, even under an ulp, costs about the
  scheme's K₀.  Pooled, raw KL ranks the schemes' sensitivity (Spearman
  0.22).  Pooling over KL / K₀ was tried and withdrawn: it assumes the
  dose's slope is the same under every scheme, measured so far only under
  `fp8-row`.  The test is the pairs of designs within a model and scheme,
  each resolved or not by its paired difference over segments; each
  scheme's K₀ and the designs above it are reported as they are.
- **Harm: pooled** over schemes and models.  It is in the same units under
  every scheme (KL from R0 the design adds), so it pools; pairs within a
  scheme as well.

##### Which local metric, which end-to-end metric?

- **Local (x):** every metric `local.py` computes against the
  post-quantization reference: normwise relative error, componentwise
  backward error (the mean of its log2 per element, over the elements not
  exact, with the share exact), ULP bits (mean), correct-rounding rate, bias,
  magnitude bias.  Which predicts best is a result, not a choice; backward
  error is reported first (forward error is about it times the condition
  number, the problem's, not the design's).  `ulp` is bits of error, the
  mean of `log2(1 + |e| / ulp)`, not a count of ulps.
- **End-to-end (y):** per token, mean over segments: KL(R1 || design) for
  the device's effect (Stage 1's most sensitive, 2-39 segments for 80% power
  on the widest pairs), with top-1 disagreement and Δ NLL against R1 as
  secondary; KL(R0 || design) - KL(R0 || R1) for harm; `rho`, the share of
  quantization's error a design keeps.  Every design's point is in
  `analyze.py`'s `points.csv` (`plot.py` draws it).

##### Sizes

Stage 1's: local metrics on WikiText-2's first 2048 tokens; perplexity on
50 seeded segments of 2048 tokens.  Stage 1 measured those sufficient to
separate the widest pairs; smaller spreads may not separate, which the
intervals show.

##### Models

Qwen3-0.6B, then Qwen3.5-0.8B (another architecture: DeltaNet layers).
Stage 1 found Qwen3.5's end-to-end KL flat across designs (2.7-2.9e-5 on
segment 0) — worth knowing whether local error is flat too, or KL just
insensitive.  **Phi-4-mini** (`swap.MODELS`) is the second family the
Essential gaps ask for, but at 3.8B parameters its perplexity runs ~6x as
long: next, not in this run's 6-hour budget.

##### Analysis

- For each measure, the pairs of designs within a model and scheme: resolved
  where the paired per-segment difference is past 2 standard errors, and
  then whether the one with more local error has more of it.
- For harm, Spearman ρ between each local metric (log2) and it, pooled; a
  bootstrap over designs (10,000 resamples) for its interval.
- **Deferred:** the sensitivity-weighted predictor `ΔKL ≈ Σ_l a_l ε_l²`
  needs per-layer sensitivities `a_l` from noise injection, which is new
  code (a run that adds relative noise at one layer type).  It is worth
  building if the plain correlation leaves room: per layer type (7 in
  Qwen3), on 10 segments, is ~70 FP32 passes.
- **Deferred:** standard errors on local metrics clustered by sequence
  (needs per-token metrics, which `local.py` pools).

##### Controls

- `<scheme>-exact` against itself is zero on both axes by construction;
  not run.
- **An accumulation-order-only change:** `nv.ada.e4m3.f32` at `split_k =
  4` under `fp8-row`, on Qwen3-0.6B: the same design with shorter chains.
  It should move along the same line.
- **A dose sweep** (`serve/stage2/sweep.py`, `sweep.sh`): within one scheme
  the designs' local errors cluster (2^-19 to 2^-24, and Ada / Hopper at
  2^-9), too few and too far apart for a trend.  Hopper's E4M3 chain with
  its fused-sum width `F` stepped 13, 15, ..., 25 (rounded toward zero to
  `min(F, 23)` bits) runs from Hopper (`F` 13) to Blackwell (25): one
  hardware parameter, local error ~2.4 bits a step.  Both models, 50
  segments; the endpoints are Hopper's and Blackwell's own runs.

##### Zero-shot

Not in this run.  Stage 1 ranked it below perplexity; it confirms
capability on a few designs afterwards (the widest pairs, `--items 500`).

#### Budget

6 GPU-hours.  Stage 1 measured 4-17 s per perplexity segment per design
(BF16 slowest), so a model is ~1.5-2 h of perplexity; local metrics are
minutes.  The driver runs Qwen3-0.6B first, logging each script's time;
Qwen3.5-0.8B follows.

#### Record

##### Qwen3-0.6B (2026-09-29)

Local metrics 4 min for the six schemes; perplexity 85 min (`bf16` 35,
`fp8-row` 13, `fp8-row:fnuz` 8, `mxfp8` 6, `mxfp4` 15, `nvfp4` 9).

| scheme | design | normwise (log2) | KL vs R1 | top-1 disagreement |
|---|---|---|---|---|
| `bf16` | amd.cdna2.bf16_1k | -21.52 | 6.650e-05 | 0.45% |
| `bf16` | amd.cdna3.bf16 | -21.91 | 6.685e-05 | 0.46% |
| `bf16` | amd.cdna2.bf16 | -21.27 | 6.726e-05 | 0.45% |
| `bf16` | nv.hopper.bf16.f32 | -18.97 | 6.746e-05 | 0.48% |
| `bf16` | nv.ampere.bf16.f32 | -18.52 | 6.761e-05 | 0.47% |
| `mxfp4` | nv.blackwell.mx.e2m1 | -25.55 | 9.82e-03 | 1.17% |
| `mxfp4` | nv.blackwell.mxfp4 | -25.97 | 9.82e-03 | 1.17% |
| `fp8-row:fnuz` | amd.cdna3.fp8 | -23.54 | 1.245e-02 | 6.04% |
| `fp8-row` | nv.blackwell.e4m3.f32 | -21.09 | 1.261e-02 | 6.05% |
| `mxfp8` | nv.blackwell.mx.e4m3 | -21.04 | 1.314e-02 | 6.18% |
| `fp8-row` | nv.ada.e4m3.f32 | -8.80 | 1.373e-02 | 6.29% |
| `fp8-row` | nv.hopper.e4m3.f32 | -8.80 | 1.378e-02 | 6.28% |
| `nvfp4` | nv.blackwell.nvfp4 | -23.18 | 1.692e-01 | 20.56% |

- **Within a scheme, local error predicts the ranking.**  Of 14 pairs of
  designs sharing a scheme, the 4 that 50 segments resolve (the paired
  per-segment KL difference past 2 standard errors) all agree with every
  local metric; none disagrees.  Every unresolved pair's KL difference has
  the sign its local difference predicts, but for two near-ties
  (`cdna2.bf16_1k` / `cdna3.bf16`, 0.38 apart in log2; Ada / Hopper, equal).
  The resolution grows with the local gap: 12.3 in log2 (Ada or Hopper
  against Blackwell) is 12-14 standard errors, 2.3-3.4 (NV against AMD under
  `bf16`) is 0.6-2.8.
- **Pooled over schemes it does not** (Spearman 0.21 for normwise, 95% CI
  [-0.36, 0.76]).  KL against R1 has a scale per scheme (K₀), far above what
  the designs' errors separate: `bf16` 6.7e-5, `mxfp4` 9.8e-3, FP8 1.2-1.4e-2,
  `nvfp4` 0.17.  The MXFP4 designs err ~2^-25.5 locally, essentially exact,
  yet sit at 9.8e-3: under a quantizing scheme any difference from R1 flips
  some downstream activation's rounding to a neighbouring FP4/FP8 value, so
  the end-to-end divergence measures the scheme's sensitivity more than the
  design's error.  This is the pooling decision's reopen condition.
- **Next, K₀ per scheme:** R1 rounded toward zero instead of to nearest
  (under an ulp from R1), its KL against R1 an estimate of any inexact
  design's cost; a design's effect is its KL above it.

##### K₀, Qwen3.5-0.8B, the control (2026-09-29)

Qwen3.5's perplexity 3.3 h (`bf16` 48 min; the rest 12-22 min each, their
references in host memory after the first attempt ran out of GPU memory);
Qwen3-0.6B's K₀ runs 4 min a scheme, the control 7 min.

| scheme | K₀, Qwen3-0.6B | K₀, Qwen3.5-0.8B | KL / K₀, Qwen3.5 |
|---|---|---|---|
| `bf16` | 6.66e-05 | 4.85e-05 | 1.01-1.02 |
| `fp8-row` | 1.261e-02 | 9.18e-03 | Blackwell 1.02, Hopper 1.10, Ada 1.11 |
| `fp8-row:fnuz` | 1.254e-02 | 9.25e-03 | 1.00 |
| `mxfp8` | 1.314e-02 | 9.66e-03 | 1.01 |
| `mxfp4` | ~0 | ~0 | none |
| `nvfp4` | 1.708e-01 | 1.113e-01 | 1.00 |

- **K₀ is most of every design's KL.**  A 1-ulp perturbation of R1
  costs what the best designs cost; the worst (Ada, Hopper under FP8, 12-13
  bits more local error) add 9-11% to it.  The scheme sets KL's scale,
  the design a few percent on top.
- **(Withdrawn; see "Two measures".)**  Pooled over schemes and models on
  KL / K₀, Spearman was 0.80 for normwise ([0.51, 0.91]), 0.89 for
  backward error, log-log Pearson 0.96-0.97; but the four Ada / Hopper
  points carried most of it (without them, 18 designs: Spearman 0.65
  [0.20, 0.86], Pearson 0.69), and the ratio assumes a scaling the control
  contradicts.
- **Within a scheme, 12 pairs resolve over both models; every local metric
  but `backward_max` agrees on all 12**, none disagrees (`backward_max`: 8
  and 4).  Qwen3.5 resolves NV against AMD under `bf16` (z 2.0-4.3, 2.1-2.9
  bits apart) that Qwen3-0.6B did not.  The 16 unresolved are near-ties in
  local error (under 0.6 bits) or the two MXFP4 designs.
- **MXFP4 is exact, up to rare events.**  Its designs err 2^-25.5 (Qwen3)
  and 2^-36.8 (Qwen3.5) locally, exact on 100.0000% and 99.9997% of
  elements (so their backward error, a mean over the rest, is left out of
  `plot.py`): E2M1 products and power-of-2 scales are
  exact in FP32, so rounding toward zero almost never changes R1 (its
  K₀, ~1e-11, is refused as zero).  On Qwen3.5 KL is 1e-11, no top-1
  disagreement.  On Qwen3-0.6B, 47 of 50 segments match R1 and three carry
  the mean (KL 0.28, 0.17, 0.045), both designs alike: where they do
  differ, a flipped FP4 rounding downstream cascades.  Such a scheme needs
  a rate of divergent segments, not a mean.
- **The control agrees in sign and is resolved:** Ada at `split_k = 4`
  errs 2.3 bits less (-11.08 against -8.80), its KL drops 1.373e-02 to
  1.352e-02 (paired z 2.8).  Its KL above K₀ falls 19% for 4.9x less
  normwise error: well short of `ε²`, the scaling a sensitivity model
  `ΔKL ≈ Σ a_l ε_l²` assumes.

##### Two measures (2026-09-29)

KL against R1 answers what the design's arithmetic does to the quantized
model; it does not say whether that is worse.  Harm, KL(R0 || design) -
KL(R0 || R1) paired over segments, does; it needs no new runs (each design
was already paired with R0).

| model | scheme | design | harm vs R0 | z |
|---|---|---|---|---|
| Qwen3-0.6B | `fp8-row` | nv.hopper.e4m3.f32 | +4.3e-04 | +3.9 |
| Qwen3-0.6B | `fp8-row` | nv.ada.e4m3.f32 | +3.4e-04 | +3.1 |
| Qwen3-0.6B | `mxfp8` | nv.blackwell.mx.e4m3 | +2.7e-04 | +2.7 |
| Qwen3.5-0.8B | `fp8-row` | nv.ada.e4m3.f32 | +1.1e-04 | +1.5 |
| Qwen3.5-0.8B | `fp8-row` | nv.hopper.e4m3.f32 | +8.9e-05 | +1.3 |
| every other design | | | | \|z\| ≤ 1.9 |

Against quantization's own cost, KL(R0 || R1): `bf16` 4.5e-05 and 3.1e-05,
`fp8-row` 1.8e-02 and 1.2e-02, `nvfp4` 0.32 and 0.19 (Qwen3-0.6B, Qwen3.5).

- **Harm is below resolution for all but FP8 truncation.**  3 of 26
  designs resolve, all on Qwen3-0.6B; the largest, Hopper under `fp8-row`,
  is 2% of the scheme's quantization cost.  Under `bf16` every design's
  harm is within ±4e-07 (under 1% of 4.5e-05), none resolved, and bounded
  at 2.5% of that cost (2 standard errors).  Pooled over
  schemes and models, Spearman with normwise is 0.30 ([-0.14, 0.70]): no
  correlation, for want of a quantity that varies past its noise.
- **Pairs on harm:** 3 of 28 resolve (Ada and Hopper against Blackwell on
  Qwen3-0.6B, z 2.2 and 3.0; Ampere against Hopper under `bf16` on Qwen3.5,
  z 2.0), all agreeing with local error.
- **The device's effect: the scheme's K₀, then a dose** (the dose sweep,
  below, measures it: linear in log2 backward error, not a threshold).
  Within a scheme local error ranks it (12 of 12 resolved pairs); across
  schemes K₀, not the design, sets its size.  K₀ is the scheme's and the
  model's: Qwen3.5's are 0.65-0.73x
  Qwen3-0.6B's under every scheme, the order of schemes the same.
- **What Stage 2 answers** (with the dose sweep's result).  Local error
  predicts the device's effect within a scheme, on both models: its ranking
  (12 of 12 resolved pairs) and, under `fp8-row`, its size (linear in log2
  backward error).  It cannot be checked against harm, which, but for FP8
  truncation, 50 segments do not resolve: for inference quality,
  quantization dominates and designs that are not grossly worse do not
  measurably matter.  The device's effect remains real as divergence (6% of
  top-1 tokens change under FP8 against R1, 0.45% under `bf16`), which
  matters where outputs must agree across hardware.

##### The dose sweep (2026-09-29)

`sweep.sh`: local metrics 80 s a model; perplexity of the five middle
widths 23 min (Qwen3-0.6B), 28 min (Qwen3.5).  `sweep.f13` and `sweep.f25`
reproduce Hopper's and Blackwell's local metrics exactly, and R1 agrees
between the sweep's runs and the scheme's (the same segments), so the
endpoints are theirs.

| `F` | normwise, Qwen3-0.6B | ULP bits | KL / K₀ | harm z | normwise, Qwen3.5 | ULP bits | KL / K₀ | harm z |
|---|---|---|---|---|---|---|---|---|
| 13 (Hopper) | -8.80 | 12.47 | 1.093 | +3.9 | -10.10 | 12.40 | 1.103 | +1.3 |
| 15 | -10.90 | 9.85 | 1.065 | +0.9 | -12.68 | 9.78 | 1.087 | +0.6 |
| 17 | -13.27 | 7.20 | 1.045 | +0.6 | -15.37 | 7.12 | 1.058 | +1.6 |
| 19 | -15.72 | 4.63 | 1.014 | +0.8 | -18.13 | 4.52 | 1.044 | +0.2 |
| 21 | -18.07 | 2.34 | 0.997 | +0.9 | -20.91 | 2.17 | 1.028 | +0.7 |
| 23 | -20.82 | 0.88 | 0.988 | +0.1 | -23.57 | 0.75 | 1.011 | -0.2 |
| 25 (Blackwell) | -21.09 | 1.08 | 1.001 | +1.1 | -22.92 | 0.90 | 1.016 | +1.7 |

- **The device's effect falls with local error at every step**: each step
  from 13 to 21 (Qwen3-0.6B) or 23 (Qwen3.5) is resolved on its own
  (paired z 3.0-6.0); Spearman 0.89-0.96 (Qwen3-0.6B) and 1.00 (Qwen3.5)
  for every local metric.
- **It is linear in the backward error** (the mean of its log2 per element,
  as backward error is read throughout), the same slope on both models:
  KL / K₀ = 1.205 + 0.0075 x backward (R² 0.96, Qwen3-0.6B) and 1.205 +
  0.0066 x backward (R² 0.99, Qwen3.5).  Each doubling of backward error
  adds ~0.7% of K₀: the scheme's flips, for any design that is not exact,
  and a dose on top logarithmic in the error, not `ε²`.  Against ULP bits,
  also a mean of log2, R² 0.98 and 0.99.  (First measured with the log2 of
  the arithmetic mean, which large errors dominate: the fits were the same
  to 0.0002 in slope, every mean of log2 0.9-1.6 bits below it.)  K₀ is not a bound: a design whose
  error is under R1-rounded-toward-zero's sits just below it (0.988 at `F`
  23).
- **Harm resolves only at `F` 13, on Qwen3-0.6B** (z 3.9), and falls to
  noise one step later (the step's paired z 3.8); from `F` 15 to 25 it is
  flat and unresolved on both models, and nowhere resolved on Qwen3.5.  Of
  the local metrics |bias| grows most over that step (4.2 to 13.8 u,
  against ULP bits 9.9 to 12.5): harm may follow one-sided truncation error past
  a threshold, which one resolved step cannot settle.

##### What the device's effect is (2026-09-30)

Harm ~0 while the device's effect is as large as quantization's own cost
(`fp8-row`, Qwen3-0.6B: 1.3e-02 against 1.8e-02) needs explaining.  Taking
KL as half a squared distance (small perturbations), a design displaced
from R1 at right angles to R1's offset from R0 would be as far from R0 as
both costs added (3.1e-02), harm ~1.3e-02; measured, ~1e-04.  What fits:

- **Every design is as far from R0 as R1 is**: KL(R0 || design) =
  KL(R0 || R1) to 1-2%, every design and scheme, both models.
- **So the design re-draws part of quantization's error.**  That error has
  a fixed part (the weights' rounding, and roundings every run makes alike)
  and a chaotic part: activations near a grid boundary, which round either
  way.  Any difference in arithmetic flips some of those roundings and
  draws that part afresh: another realization, no larger.  The share kept
  is `rho`, the correlation of R1's and the design's offsets from R0 (the
  law of cosines on the three KLs; `analyze.py`'s `rho`).

| scheme | `rho`, Qwen3-0.6B | `rho`, Qwen3.5 | re-drawn, ~1 - `rho` |
|---|---|---|---|
| `bf16` | 0.25-0.26 | 0.21-0.22 | 75-80% |
| `fp8-row`, `fp8-row:fnuz` | 0.63-0.68 | 0.58-0.62 | 35-40% |
| `mxfp8` | 0.70 | 0.66 | 30-35% |
| `nvfp4` | 0.74 | 0.71 | 26-29% |
| `mxfp4` | 0.99 | 1.00 | ~0 (its designs are exact) |

- **`rho` is the scheme's**, the same across its designs (`bf16`
  0.254-0.263), and orders the schemes as the explanation predicts: least
  kept under `bf16`, whose weights are the BF16 master's exactly (all its
  error is activations'), most under the FP4 schemes, whose weights' error
  is fixed.  Its order is the same on both models.
- **It explains the device's effect's bands by scheme**: KL(R1 || design) ≈
  2 (1 - `rho`) KL(R0 || R1), both factors the scheme's.  (Plotted, with
  the device's effect against quantization's cost, a design's accuracy
  barely showed: Hopper's 12.9-13.3 bits more backward error than
  Blackwell's moved its point 0.04 decades.  `plot.py` draws instead the
  device's effect against backward error, in a band per scheme, and KL /
  K₀ within `fp8-row` and `bf16`.)
- **And harm**: a re-draw is as far from R0 as what it replaces, so harm
  is ~0; a design shows it only where it adds error of its own besides.
  Ada and Hopper keep a little less (0.63 against Blackwell's 0.66, Qwen3-0.6B).
- **Caveats.**  KL as half a squared distance is an approximation.  `bf16`'s
  `rho` is not 0: some activations round alike in every run, presumably the
  first layer's input, the same in all of them.

##### K₀'s second estimate (2026-09-30)

Is K₀ an artefact of rounding toward zero, one-sided and changing about half
the outputs?  `<scheme>-exact-rand` moves every element of R1 an ulp up or
down at random, without bias (`serve/stage2/k0.sh`, Qwen3-0.6B, 50 segments,
~4 min a scheme):

| scheme | K₀, toward zero | K₀, random sign | ratio | paired z | top-1, toward zero | top-1, random |
|---|---|---|---|---|---|---|
| `bf16` | 6.663e-05 | 6.614e-05 | 0.993 | -1.3 | 0.45% | 0.45% |
| `fp8-row` | 1.261e-02 | 1.253e-02 | 0.994 | -1.1 | 6.07% | 5.99% |
| `fp8-row:fnuz` | 1.254e-02 | 1.243e-02 | 0.991 | -1.6 | 5.87% | 5.96% |
| `mxfp8` | 1.314e-02 | 1.319e-02 | 1.004 | +0.6 | 6.26% | 6.17% |
| `mxfp4` | ~0 | 0.408 | | | 0.00% | 31.81% |
| `nvfp4` | 0.1708 | 0.1721 | 1.008 | +1.4 | 20.68% | 20.61% |

- **K₀ is the perturbation's, not its form's:** within 1% for every scheme
  but `mxfp4`, none resolved, though one estimate is biased and moves half
  the outputs and the other unbiased and moves all of them.  It measures
  that R1 was perturbed at all.
- **`mxfp4` has a K₀, and a large one** (0.41, 32% of top-1 tokens):
  rounding toward zero almost never changes its R1 (E2M1 products and
  power-of-2 scales, exact in FP32), so that estimate read ~0.  Its designs
  sit near 0 because they are exact, not because the scheme is insensitive;
  where they are not (three of Qwen3-0.6B's segments), they diverge as far.
  The analysis keeps rounding toward zero's estimate (the one both models
  have), so `mxfp4` stays without a K₀ there.

##### More from the data on disk, and the split-k sweep (2026-09-30)

- **First top-1 disagreement with R1** (`paired`'s `median`, the token
  position where teacher-forced top-1 first differs), a proxy for free
  generation: `bf16` 80-128 tokens, `fp8-row` 4.5-15, `nvfp4` 2-2.5, on
  both models.  Greedy decoding would leave R1's path within ~10 tokens
  under FP8, whatever the accumulator; within `fp8-row` the less accurate
  designs diverge sooner (Qwen3-0.6B: Hopper, Ada 5.5-7 against Blackwell
  13), their intervals overlapping.
- **The re-draw holds per segment:** across the 50 segments, KL(R1 ||
  design) tracks KL(R0 || R1) (Spearman 0.81-0.88 in five of six scheme and
  model pairs, 0.63 for `nvfp4` on Qwen3.5), their ratio tight per scheme
  (IQR ±3%: `bf16` 1.49 and 1.57, `fp8-row` 0.68 and 0.78, `nvfp4` 0.52
  and 0.59).
- **The split-k sweep** (`splitk.sh`): nv.ada.e4m3.f32 with `k` split into
  1-64 slices (64: one instruction per slice at `k` 1024), Qwen3-0.6B, 50
  segments, ~7 min a split.  Another mechanism for less error: shorter
  chains, the same accumulator.

| `split_k` | backward | KL / K₀ | width sweep's line | off, SE |
|---|---|---|---|---|
| 1 | -15.80 | 1.089 | 1.086 | +0.2 |
| 2 | -16.30 | 1.074 | 1.082 | -0.4 |
| 4 | -16.85 | 1.072 | 1.078 | -0.3 |
| 8 | -17.48 | 1.050 | 1.073 | -1.2 |
| 16 | -18.17 | 1.053 | 1.068 | -0.9 |
| 32 | -18.90 | 1.048 | 1.063 | -0.8 |
| 64 | -19.47 | 1.044 | 1.058 | -0.8 |

- **Backward error ranks the splits** (KL / K₀ falls 1.089 to 1.044 as they
  shorten, 8 and 16 a near-tie), **but splits and widths do not share one
  line.**  Paired over segments against a width design of about the same
  backward error, a split has less KL: split 8 against `F` 15 (1 bit apart,
  the split worse) -0.015 of K₀, z -2.4; split 16 against `F` 15 (0.3 bit)
  -0.011, z -2.0; split 64 matches `F` 17 (z -0.2) with 1.7 bits more
  backward error.  Along the splits KL falls ~1.2% of K₀ a bit, along the
  widths 0.75%.
- **Not bias:** at a given backward error the splits have more of it
  (magnitude bias, split 16 35u against `F` 15's 21u), yet less KL.
- So backward error ranks designs within a mechanism but is not quite
  sufficient across mechanisms: at equal backward error, shorter chains cost
  ~1-1.5% of K₀ less than a narrower accumulator.  Marginal (z ~ -2, several
  comparisons), one model.  Untested guess: the error's structure, a long
  truncated chain's errors correlated along an output row, split partials'
  less so.

##### Which local statistic the dose follows (2026-09-30)

Two local metrics added (`metrics.py`): **`flips`**, the share of outputs
whose code on the scheme's activation grid differs between the design's and
`fl(y)`, each quantized as a next layer's input would be (a first version
compared dequantized values, so a scale an ulp apart counted every element:
~100% under FP8; codes, not values); and **percentiles of log2 backward
error** (p50, p90, p99, from 1/8-bit histograms that pool exactly).  Every
local run redone (~15 min).

- **Flip rates:** Hopper and Ada 2.1% (Qwen3-0.6B, `fp8-row`), the width
  sweep 4.0e-03 (`F` 15) to 2.4e-06 (`F` 23), Blackwell 3.7e-06; `bf16`
  1.4e-04 to 8.2e-04; `nvfp4` 1.4e-07.  A few flipped codes per million
  outputs already cost K₀: any flip starts the cascade.  Along the sweep KL
  / K₀ follows log2 of the flip rate as it does backward error (Spearman
  the same, log-log Pearson 0.98, 0.99).
- **The split-k offset follows the weighting of large errors.**  The splits'
  mean distance from the width sweep's line, drawn in each metric (KL / K₀):

| local metric | weights | split offset |
|---|---|---|
| backward p50 | the typical element | -0.014 |
| backward (mean of log2) | every element alike | -0.014 |
| log2 flips | every element alike | -0.015 |
| backward p90 | | -0.010 |
| backward p99 | the tail | -0.008 |
| backward max | the largest | +0.007 |
| normwise | errors in the largest outputs | +0.002 to +0.003 (RMS 0.006) |

  Normwise relative error puts both mechanisms on one line, within noise;
  the per-element statistics leave the offset whether they count errors
  (flips) or size them (backward, its percentiles).  So the dose follows
  errors in the large outputs, which splitting shrinks less than the
  typical element's.
- **Normwise backward error does the same** (`backward_normwise`,
  `||ŷ - y||_F / |||x|ᵀ|w|||_F`, log2): split offset +0.003 (RMS 0.006),
  the width sweep's R² 0.93, 12 of 12 resolved pairs.  Within a scheme and
  model it is normwise error less a constant (3.54 bits under `fp8-row`,
  Qwen3-0.6B: `|||x|ᵀ|w||| / ||y||`, the problem's, not the design's), so
  the two rank and fit alike.  What matters is normwise against
  componentwise, not forward against backward: the dose follows the error's
  size against the whole output, which its largest elements set.
- **So `plot.py` and the fits now use normwise backward error.**  The width
  sweep: KL / K₀ = 1.181 + 0.0080 x backward_normwise (R² 0.93,
  Qwen3-0.6B), 1.190 + 0.0068 x (R² 0.99, Qwen3.5); every split within 0.5
  standard errors of it (+0.4 to -0.3).  `bf16`'s slopes 0.0036 and 0.0045.
  The earlier sections' fits are in componentwise backward error, as
  measured then.

##### Error binned by `k` (2026-09-30)

`local.py` pooled each design's layers by `k` in bins `[2^i, 2^(i+1))`
(since removed: it changed no conclusion); both models have two, 1024-2047 (q, k, v, gate, up) and 2048-4095 (o, down).

| bin | metric | width sweep R² | split offset (KL / K₀) |
|---|---|---|---|
| 1024-2047 | normwise backward | 0.968 | -0.009 |
| 2048-4095 | normwise backward | 0.940 | +0.007 |
| all | normwise backward | 0.943 | +0.002 |
| either, or all | componentwise backward | 0.97 | -0.014 to -0.016 |

(Qwen3-0.6B; Qwen3.5's width sweep alike, R² 0.98-0.99 in every bin.)

- **Along the width sweep the bins move together** (slopes 0.007-0.008,
  both models): a narrower accumulator scales every layer's error alike, so
  the sweep cannot say which `k` the dose follows.
- **No bin alone puts the splits on the line**, the short one below it and
  the long one above: splitting improves them unequally and the dose follows
  both, pooled normwise.
- **The componentwise offset is not `k`'s mix:** it is the same in every
  bin, so it is the weighting of elements, not the pooling of chains.
- Hopper's normwise backward error is 12.8 bits above Blackwell's at short
  `k`, 11.5-12.2 at long.  Attributing the dose to a `k` range needs designs
  that differ in one layer group at a time.

##### Frozen runs (2026-10-20)

`serve/frozen.py` (`scoring.frozen`, `swap.Frozen`): each segment runs R1
once, recording each linear layer's raw input `x_R1` and quantized input;
a frozen run's layer computes `matmul(Q(x_R1)) + (x - x_R1) @ W_q.T`.  Its
arithmetic's error enters every layer and propagates through the rest; no
activation rounds differently.  Free is the run as `perplexity.py` runs it.
Log-softmax in FP64: in FP32 its rounding alone is a KL near 1e-9.
Qwen3-0.6B, `fp8-row`, 10 segments:

| run | KL vs R1, free | KL vs R1, frozen | re-rounded |
|---|---|---|---|
| R1 rounded toward zero | 1.288e-02 | 1.21e-11 | 0.006% |
| R1 an ulp up or down | 1.299e-02 | 1.29e-11 | 0.007% |
| nv.blackwell.e4m3.f32 | 1.301e-02 | 2.51e-11 | 0.009% |
| nv.hopper.e4m3.f32 | 1.395e-02 | 1.98e-04 | 16% |
| nv.ada.e4m3.f32 | 1.413e-02 | 1.98e-04 | 16% |

(re-rounded: the share of activation elements the run's own input would
quantize differently from R1's.)  The width sweep, frozen:

| `F` | normwise | KL, frozen | re-rounded |
|---|---|---|---|
| 13 (Hopper) | -8.80 | 1.98e-04 | 16% |
| 15 | -10.90 | 9.16e-06 | 4.5% |
| 17 | -13.27 | 3.21e-07 | 1.2% |
| 19 | -15.72 | 1.13e-08 | 0.35% |
| 21 | -18.07 | 5.09e-10 | 0.08% |
| 23 | -20.82 | 3.94e-11 | 0.02% |
| 25 (Blackwell) | -21.09 | 2.51e-11 | 0.009% |

- **K₀ is re-rounding.**  Frozen, R1 rounded toward zero is 1.2e-11; free,
  1.3e-02.  Free, Blackwell cannot be told from R1 an ulp off at random;
  frozen, it is 2x it, resolved.
- **Frozen KL ∝ ε²:** 2.00 bits of KL a bit of normwise error over `F`
  13-21 (R² 0.9996; each step 1.9-2.1), flattening toward the floor past
  21.  The "not `ε²`" of the dose sweep is the re-rounding's, not the
  arithmetic's.
- **Not a bound on the free run.**  A quantizer absorbs most sub-cell
  offsets and moves a few a whole step; frozen passes every one through.
  Over where values fall in their cells the two agree on average (a
  crossing's chance is its offset's share of the cell), so frozen is the
  design's mean effect, free adds the re-rounding's variance, K₀.
- **Validity:** re-rounded under 1% from `F` 17 on; Hopper's 16% makes its
  frozen run the linearized effect of its arithmetic, not its own run.

#### Open items

##### Is backward error sufficient across mechanisms?

(Answered for Qwen3-0.6B: "Which local statistic the dose follows".
Normwise backward error removes the offset; the componentwise statistics
leave it.  Reopen on a second model.)
The split-k sweep puts shorter chains ~1-1.5% of K₀ below the width
sweep's line at equal backward error (z ~ -2, Qwen3-0.6B).  At stake:
whether local backward error alone predicts the device's effect, or its
structure matters too.  Replicating on Qwen3.5 (~40 min; its `k` 3584 stops
at 32 slices) would say whether the offset is real; not run.  If it is, a
per-row error correlation metric in `local.py` would test the guess.
**Provisional:** backward error ranks within a
mechanism; across mechanisms, treat differences under ~1.5% of K₀ as
unresolved.

##### Does the dose's slope hold under other schemes?

The dose sweep measured it under `fp8-row` only (~0.7% of K₀ per doubling
of backward error, both models).  At stake: whether `K₀ x (c + a x
log2(backward))` is a model of the device's effect or a fact about one
scheme.  `bf16`'s five designs, over 2^-27.4 to 2^-23.9 in backward error,
fit about half `fp8-row`'s slope (0.0034 and 0.0038 of K₀ a bit, Qwen3-0.6B
and Qwen3.5), too narrow a span to tell it from it; a sweep there (Hopper's BF16
chain, `F` stepped) would test it, ~1 h a model.  **Provisional:** a fact
about `fp8-row`; the sensitivity-weighted predictor (Analysis, deferred)
waits on it, since `ε²` is already contradicted.

##### How is a scheme without a K₀ evaluated?

(`mxfp4` has one, 0.41, by the random-sign estimate: "K₀'s second
estimate".)  `mxfp4`'s designs are R1 but for rare cascades, which a mean over 50
segments measures poorly (SE 6.5e-3 on 9.8e-3).  **Provisional:** report
the divergent segments' count per design; the two MXFP4 designs tie on it.
Reopen when an MXFP4 design with real accumulation error is built.

##### Is 50 segments enough for the narrow pairs?

Stage 1 sized it on the widest pairs.  A design pair whose KL difference is
within its intervals is a tie for ranking, not a disagreement.
**Provisional:** treat such pairs as ties.  Reopen if many pairs tie: then
more segments for those.

##### Is harm worth resolving?

Harm's standard errors are ~2-4e-07 under `bf16` and ~5e-05-1e-04 under
FP8.  Resolving effects of the size the table suggests (1e-07, 5e-05) takes
~10-100x the segments: 8-80 h a model under `bf16`.  At stake: a claim that
local error predicts harm, against the cheaper claim Stage 2 already has
(no design but FP8 truncation harms past the table's bounds).
**Provisional:** not resolved; harm is reported as a bound.  The dose
sweep filled the range between Blackwell's and Hopper's local error and
found harm flat there but at Hopper's end.  Reopen to separate |bias| from
backward error as harm's cause (e.g. the sweep's chains with the fused sum
and `rho` rounding to nearest, if that keeps the error's size and drops its
bias), or for free-running generation, where one
flipped token changes everything after it.

### Stage 3 -- Sweeping hardware design parameters

Implementation plan; one phase per commit.

Narrowed from the Roadmap's first sketch (successive halving over fidelity
levels, end to end on a subset): **local metrics only**, and **one parameter at a time**.
For a design `(P_0, ..., P_i, ..., P_n)` and a local metric `M_j`, show how
the design evaluates under `M_j` as `P_i` varies and every other parameter
is held at an anchor design's value.  Those graphs are the deliverable.

#### Context

- **The hardware models** are factories in `examples/mmasim/models/`, one per
  MMA-Sim algorithm, each taking its own parameters:

| factory | designs | parameters |
|---|---|---|
| `utils.make_fma_dpa` | FP64/FP32 FMA chains | rounding context |
| `amd.make_e_fdpa` | CDNA1 BF16/FP16 | `L` |
| `amd.make_ftz_addmul` | CDNA2 BF16/FP16 | input format, `P` |
| `amd.make_tr_fdpa` | CDNA3 TF32/BF16/FP16 | `L`, formats, `F`, `F2`, `rho`, `e_zero` |
| `amd.make_gtr_fdpa` | CDNA3 FP8 | as TR-FDPA, even/odd groups |
| `nv.make_t_fdpa(_chain)` | Volta-Blackwell TF32/BF16/FP16/FP8 | `L`, formats, `F`, `rho`, `e_zero`, `is_mma` |
| `nv.make_st_fdpa` | Blackwell MX | formats, scale format, `F`, `rho` |
| `nv.make_gst_fdpa` | Blackwell MXFP4/NVFP4 | `G`, scale format, `F`, `rho` |

- `compile.DESIGNS` names each instruction as one of these, with its
  argument types; `tests/test_nv.py` and `tests/test_amd.py` check the
  factories against MMA-Sim's reference bit for bit; `compile_triton.py -r`
  and `tests/test_triton.py` check their Triton kernels against the FPy
  interpreter.
- **The serving harness** measures local metrics per design on cached
  activations (`serve/local.py`, `serve/core/metrics.py`); `kernels.register`
  adds a design at runtime (Stage 2's `serve/stage2/sweep.py` registered
  synthetic Hopper chains that way).
- **What Stage 2 found** (above, parked): local error, as the
  normwise backward error, predicts the end-to-end effect of a design, and
  that effect is small at inference.  So Stage 3 measures locally only.

A sweep needs every parameter to be variable while the others hold, and a
remix (NV's fused sum with CDNA3's separate rounded join, say) needs the
factories' stages to compose.  Today each factory fixes its own stages.

#### The generator

Every factory is the same pipeline with different choices per stage.  One
parameterized builder, `examples/mmasim/generator/` (outside `models/`,
which keeps the paper's explicit models; it imports their helpers), takes a
`Design` (a frozen dataclass) and returns the dot-product-accumulate
function and its argument types, as `compile.DESIGNS`' builders do.

| stage | parameter | choices | existing designs |
|---|---|---|---|
| terms | `products` | `exact`; `rounded`, each by `rho` | exact; rounded (CDNA2) |
| | `ftz` | flush subnormal inputs to +0, results to signed 0 | on (CDNA2) |
| | `scales`, `G` | none; a scale pair per instruction; one per group of `G`, each group's sum exact | MX; MXFP4/NVFP4 (`G` 16) |
| groups | `groups` | `none`; `even-odd`, each half summed alone, then combined | even-odd (CDNA3 FP8) |
| block sum | `sum` | `fused`, aligned and truncated past `F` bits; `tree`, pairwise adds rounded by `rho`; `exact`; `fma`, a chain of adds each rounded by `rho` | fused (NV, CDNA3); tree (CDNA2); exact (CDNA1); fma (FP64/FP32) |
| | `F` | fraction bits kept | 13, 23-25, 35 |
| | `L` | elements per instruction | 1 (FMA) - 64 |
| joining `c` | `join` | `inside` (a fused sum's term, a tree's or chain's last add, in an exact sum); `separate`, the sum and `c` rounded down at `F2` and `F` bits and added | inside (NV); separate (CDNA3) |
| | `F2` | tied to `F`: `F2 = F + guard`, the join's adder `guard` bits wider than the sum (swept as `guard`, 2026-10-23) | 31 (CDNA3: `F` 24, `guard` 7) |
| output | `rho` | precision, rounding mode: the accumulator format the next instruction's `c` is in | RZ FP32, RZ E8M13 (NV FP8), RNE FP16, RNE FP32 (AMD) |
| per structure | `e_zero` (`None`: the model's default), special values | as the models derive them | not swept |

- **Hardware only.**  The kernel's loop over `k` (split-k, promotion
  between instruction chains, the order partials combine in) is software
  and out of scope; every design runs as one unsplit chain of its
  instruction.  The accumulator *format* is in scope: it is the
  instruction's output.
- **A catalog** (`generator/catalog.py`) maps each `compile.DESIGNS` entry
  to its `Design`.  At those points the generator agrees with the model bit
  for bit; that is what makes a remixed point a model of something.
- **`e_zero` and special-value semantics** follow the structure whose model
  has them (`_rule`): NVIDIA's for a fused sum joined inside; TR-FDPA's
  (products overflowing at 2^128) joined separate; GTR-FDPA's (NaN and
  infinity read at exponent -1) for even-odd groups joined separate;
  E-FDPA's for an exact sum; IEEE arithmetic's for a tree or chain.  Carried
  for that agreement, not swept.
- **Every combination of the stages is a design** (since 2026-10-01; earlier,
  21 were refused as "unformable").  The definitions the paper's models do
  not give: a chain of FMAs reads no `L` (it rounds after every element); a
  separate join takes any block sum `t`; even-odd halves are summed by the
  block sum and combined as it adds (a fused sum's rounded down at `F`, as
  GTR-FDPA's), joined inside in one fused sum with `c`; a tree over any `L`
  halves at the largest power of two below it; rounded products, scales and
  FTZ with any sum.  `invalid` refuses only sizes that disagree: `k` not
  whole instructions, `G` without scales or not dividing `L`, a scaled
  design longer than one instruction, even-odd groups of one term.  A tree
  or chain joined separate that overflows returns its special value: a
  fused or exact sum cannot.
- **How it is written:** three block bodies, one per call signature
  (unscaled, scaled per instruction, per group), each making the terms,
  their alignment exponents and special values, then a `finish` chosen in
  Python from small FPy pieces (the block sum over a list, the even-odd
  combine, the join).  The backend constrains the shape: a computed list
  crosses a call only to be read, a fixed-point sum loses its format across
  a call (GTR-FDPA's finish keeps its sums inline), a closure-constant
  branch must not give a variable two types, no loop-carried integer or
  `**`.  The generator compiles the catalog as the models do (60 of 62, the
  same 2 refused).

#### The sweep

- **One parameter at a time, around anchors.**  An anchor is a catalog
  design; a sweep of `P_i` holds the anchor's other parameters and steps
  `P_i` over a range.  Starting anchors: Hopper FP8 (`nv.hopper.e4m3.f32`),
  Blackwell FP8 (`nv.blackwell.e4m3.f32`), Hopper BF16, Ampere BF16, CDNA3
  BF16; more are identified along the way.
- **Ranges go past existing hardware** (`L` to 64 and beyond, `F` to 35
  under FP8, ...): what is plausible is not the question.  Every point is a
  design; one the compiler refuses is recorded with its reason.
- **Categorical parameters** (`block_sum`, `join`, `products`, `layout`,
  `rho`'s mode, `ftz`) sweep over their choices, plotted as categories.
- **Input formats are not swept:** they choose the scheme, so the workload.
  Each anchor runs under its scheme.
- **Workload:** Qwen3-0.6B, WikiText-2's first 2048 tokens, activations
  cached (`local.py --acts`); every design of one scheme in one `local.py`
  pass, so the exact FP64 product and `|x|ᵀ|w|` are computed once.
- **Metrics:** all of `metrics.METRICS`, normwise backward error first
  (Stage 2's predictor of end-to-end effect), then componentwise backward
  error, its max, bias, magnitude bias, flips.

#### Outputs

Under the scripts' `--results` root (`cli.RESULTS`, `results/` from
`examples/mmasim`):

- **A table**, `<results>/sweep/sweep.csv`: one row per evaluated design -
  its anchor, the parameter varied and its value, every parameter, every
  metric - and the points the compiler refused, with its reason.
- **Cached activations** under `<results>/cache/`.
- **The plots** from the table alone (`serve/plot_sweep.py`), replayable,
  in `<results>/plots/sweep/`: a grid of panels, one per `(P_i, M_j)`, a
  line per anchor, the anchor's own value marked.

#### Phases

##### Phase 1 - The generator and its catalog

**Done.**
- `generator/__init__.py`: `Design`, `unformable`, `build(design) -> (fn,
  arg_types)`; one FPy block per structure (`fused`, its scaled and
  group-scaled forms, `separate`, `even-odd`, `tree`, `exact`, `fma`),
  stages chosen by closure constants, chained `k / L` times.
  `generator/catalog.py`: `CATALOG`, all 62 designs.
- `tests/test_generator.py` (77 tests, ~20 s at `-n 8`): every catalog
  entry has its model's argument types and agrees with it bit for bit on
  400 draws, four mixes of `compile.py`'s samplers (over the range and near
  one; a quarter hard cases: zeros, extremes, specials; all hard cases;
  subnormal-centred); the refusals; a fused sum wide enough (`F` 600) equal
  to the exact sum rounded once; four remixes no model has build and run.
- **Diverged from the plan:** `compile.py`'s samplers, not
  `tests/common.py`'s (those need torch dtypes, which E2M1 and the scale
  formats lack); no `is_mma` (it only chose `e_zero`; the catalog gives
  `e_zero` where it differs); the tree generalized to any power of two `L`,
  its products `exact` or `rounded`; FTZ on every sum, not CDNA2's only.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -n 8 tests/test_generator.py

##### Phase 2 - Remixed points compile and register

**Done.**
- **The generator, for the backend.**  Two structures the interpreter took
  but Triton did not, restructured (bit for bit still: `test_generator.py`):
  - FTZ is chosen in Python (`_flushes`: a flush, or `_keep`), not branched
    on in FPy: a closure-constant branch is not folded before storage
    inference, so a reassigned list (`A = [flush(x) ...]`) had two element
    types even with FTZ off.  Products take flushed inputs; special values
    are decided on the products; exponents read the inputs (a product not 0
    has neither flushed).
  - The tree is nested halves of scalars (`_halves`): a loop-carried stride
    became a tile, which Triton's `range` refuses; `**` has no Triton
    signature; a list of computed values cannot cross a call.
- **Registering:** `kernels.register_design(design, name=None)` registers
  under `generator.name` (`gen.` and a digest of `generator.about`, the
  parameters readably: formats as `E<e>M<m>`, `rho` with its mode, each
  context's `repr` under `exact`), at `kernels.tile` (`(16, 64)` for a tree,
  `(16, 1)` from `L` = 64, else `(32, 1)`: the catalog's most common), its
  build kept so `precompile` compiles it in worker processes (the pool
  starts by `forkserver`, so a worker sees no runtime registration; the
  pickled build carries the design).  The kernel cache's key covers
  `generator/`'s sources.
- **Tests:**
  - `tests/test_triton.py`: ten remixes, every structure (NVIDIA's FP8
    chain joined as CDNA3 joins, and with FTZ; CDNA3's even-odd over E4M3 at
    13 bits; MX at 13 bits; NVFP4 with groups of 32; CDNA2's tree over 16;
    ...), agree with the interpreter at two tiles (a tile changes the
    launch, not the result); two fuse over a static `k`.
  - `test_a_backend_gap_is_refused` pins four refusals, each a compile-time
    error, never a wrong result:

| remix | refused because |
|---|---|
| an exact sum of FP8 products, E8M13 out | no storage holds the exact sum of the products and an FP32 accumulator (CDNA1, from the paper's builder too: 2 of the 62 designs) |
| a tree of exact BF16 products | held in FP64, adding them under FP32 rounds twice |

  - `serve/tests/test_kernels.py`: a registered generator design (Hopper's
    FP8 chain at `F` 17; CDNA2's BF16 tree over 16), compiled in workers,
    runs through `kernels.linear` bit for bit against the generator.
- **After Phase 2** (the policy: a design the compiler refuses gets a
  compiler fix): directed arithmetic.  The Triton target refused Add, Sub,
  Mul, Div, Fma and Sqrt in any mode but RNE; libdevice's `add_rz` and the
  rest are correctly rounded at FP32 and FP64, now in the op table
  (`target.py`; Sub as an add of the negation; the exact ops under directed
  contexts too; not FP16, which libdevice lacks).  CDNA2's tree and an FMA
  rounding toward zero agree with the interpreter.  Its tests
  (`test_directed_arithmetic_agrees_bit_for_bit`: six ops, three modes, FP32
  and FP64, edges and random) found **the interpreter wrong** on an exact
  zero sum rounding toward negative: IEEE 754 Sec. 6.3 makes `x + (-x)`,
  `(+0) + (-0)` and an FMA's exact zero `-0` there, `+0` in every other mode;
  FPy gave `+0` in all.  Fixed in `fpy2/ops.py` (`_zero_sum`, for add, sub,
  fma), with `tests/unit/test_ops.py::TestExactZeroSum`; the full unit suite
  passes on it.
- **Calibration** (Qwen3-0.6B, 2048 tokens, `local.py --register`, 8
  compile processes):

| pass | designs | cold | warm (kernels cached) | per design, kernels | shared metrics |
|---|---|---|---|---|---|
| `fp8-row`, Hopper's FP8 chain remixed | 11 | 112 s | 91 s | 4.6-12 s | 17 s |
| `bf16`, Hopper, Ampere, CDNA3, CDNA2 remixed | 10 | 159 s | 131 s | 3.5-20 s | 27 s |

  - Compiling is ~2-3 s a design at 8 processes; evaluating ~8 s (FP8) to
    ~13 s (BF16).  So ~300 designs are ~1 h, at 2048 tokens.
  - The generator's catalog points reproduce Stage 2's numbers exactly
    (normwise backward: Hopper FP8 -12.34, CDNA3 BF16 -25.59, CDNA2 BF16
    `_1k` -25.21), at the default tile 10% (CDNA3) to 1.7x (CDNA2, 2 s) the
    catalog design's time.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -n 8 tests/test_triton.py tests/test_generator.py
      cd examples/mmasim && ../../.venv/bin/python -m pytest -p no:xdist serve/tests/test_kernels.py

##### Phase 3 - The sweep runner and its table

**Done.**
- `serve/sweep.py`:
  - **The spec:** `ANCHORS` (Hopper and Blackwell FP8 under `fp8-row`;
    Hopper and Ampere BF16 and CDNA3 BF16 under `bf16`) and `PARAMS`, each
    a way to read it off a design and its values: `L` 2-64; `F` 8-35; `F2`
    16-35; `rho_bits`, `rho`'s fraction bits at FP32's exponents, 7-23;
    `rho_mode` RTZ, RNE, RTN, RTP; `sum`; `join`; `groups`; `ftz`.  Changing
    `L` keeps `k` whole instructions.
  - **`points(anchors, params)`:** one parameter at a time, the anchor's own
    value always among them (Ampere's `F` 24 is not in `F`'s list); a
    parameter the anchor's structure does not read is not swept (`F2`
    without a separate join, `F` without a fused sum).  The full sweep: 188
    points, 152 designs.
  - **A run:** every formable point registered (`register()`, also the
    `--register` of each `local.py` pass), compiled in parallel by
    `kernels.compile_designs`, which returns each refusal's error rather than
    raising, and keeps the compiled in the kernel cache; then one `local.py`
    pass per scheme over its compiled designs, as a subprocess; then the
    join.
  - **The table** `<results>/sweep/sweep.csv`: a row per point: anchor,
    scheme, parameter, value, `is_anchor`, `status` (`ok`, `unformable`,
    `refused`) and its reason, the design's name and every parameter
    (`generator.about`), and its metrics against the quantized operands'
    exact product (group `all`); `sweep.provenance.json` beside it.
  - `--list` prints the points and their status without running.
- `serve/tests/test_sweep.py` (191 tests, under 1 s): every point is its
  anchor but for its parameter (`rho` moves for `rho_bits` and `rho_mode`,
  `k` with `L`); each anchor's value appears once a parameter; an unread
  parameter is not swept; `register()` adds every formable design, and again
  without a clash.
- **Small runs** (Qwen3-0.6B, 512 tokens):
  - Hopper FP8, `F` (10 points, 47 s): normwise backward error -8.06 at
    `F` 8, -12.41 at 13 (Hopper), and flat at -13.05 from `F` 19: with `rho`
    held at Hopper's E8M13, a wider sum is truncated back by its output (an
    interaction, as the open item foresaw).
  - CDNA3 BF16, `sum`, `join`, `groups`: three unformable points recorded;
    an inside join -23.58 against CDNA3's separate -25.59; even-odd groups
    change nothing under BF16.
  - Ampere BF16, `sum`: the tree of exact products and the exact sum
    recorded as refused, with the compiler's messages (Triton backend,
    "What is left").

      cd examples/mmasim && ../../.venv/bin/python -m pytest -p no:xdist serve/tests/test_sweep.py

##### Phase 4 - The plots

**Done.**
- `serve/plot_sweep.py`: from `sweep.csv` alone, one figure per metric
  (`<results>/plots/sweep/<metric>.png`; `--metrics`, default normwise and
  componentwise backward error, its max, magnitude bias), a panel per
  parameter, a line per anchor.
  - Numeric parameters on their axis (`L`'s log2, as it doubles), labels
    thinned within a twentieth of the span (CDNA3's `F` 24 beside 25);
    categorical ones as categories, not joined (their values have no order).
  - An anchor's own value ringed; a point not measured an x on the panel's
    floor, the anchors' side by side so none hides another.
  - Colours: the dataviz reference's categorical slots 1-5, in that order,
    each anchor also its own marker, so identity is not colour alone (the
    palette validator needs Node, not installed here: the slots are the
    reference's validated order).
  - Each metric from `local.py`'s value: log2 ones as they are, `normwise`
    to log2, biases in `u`, `flips` as a log10 share.
- `serve/tests/test_sweep.py`: a figure draws from a hand-made table (a
  numeric and a categorical parameter, a refused point).

##### Phase 5 - The full sweep

**Done** (2026-10-01; Qwen3-0.6B, 2048 tokens, `results/sweep/`; rerun after
every combination of stages became a design, and twice 2026-10-02: after
the sub-FP32 roundings compiled, and after double rounding).
- **The run:** 188 points, 152 designs: 145 measured, 7 refused by the
  compiler, none unformable (before the reruns: 133 and 19, then 141 and
  11).  Compiling ~2 min at 8 processes (every design
  anew: the kernel cache's key covers the generator's source); the `bf16`
  pass ~30 min, `fp8-row` ~10; run detached (`nohup`), past the tool's
  30-minute background limit.
- **Refused** (`sweep.csv`'s `reason`; the Triton backend, "What is left"):
  - 5: an exact sum (every anchor's `sum` = `exact`); not planned, as it
    needs error-free transformations;
  - 2: CDNA3's tree and chain of FMAs.  The FP64 add rounds through FP64
    to odd now; what is left is bounding the separate join, which needs
    digit-bound inference to tie the block sum to `e` (Triton backend,
    "What a generated design still hits").  **Moved** to its own branch:
    index spaces by identity (`digit-bound.md`, "Dimensions"), then
    loop-carried values for the chain.
- **Since, on the digit-bound branch** (2026-10-19, every point compiled and
  launched once, no model run): 182 of 188 compile, 6 refused -- the 5 exact
  sums, and CDNA3's chain of FMAs, whose exit is bounded add by add
  (`digit-bound.md`, "Control").
  CDNA3's tree compiles (`test_triton.py`'s `cdna3-tree`).  The check found
  Hopper's and Ampere's BF16 at `F` 8-16 failing Triton's compile since #337:
  an exact round emitted its operand in the operand's storage, so the carried
  accumulator changed type; fixed in `_emit_round` (`hopper-bf16-f8`).
- **Since closed by double rounding** (`compile_triton` compiles with
  `DOUBLE_ROUND`): an Add or an Fma rounding to E8M13 toward zero
  (Hopper's tree and chain of FMAs), through FP32 toward zero, and
  Hopper's and Ampere's BF16 trees, through FP64 toward zero
  (`UnfoldKind.WIDE`).
- **Since closed:** 8 refusals of a rounding below FP32's precision other
  than toward zero (Hopper's E8M13 output to nearest, down or up; CDNA3's
  output at 7-19 bits to nearest).  `unfold_round`'s normal arm
  (`|x| >= 2 ** emin`) bounded `logb(x)` by `x`'s finest digit, not by
  `emin`, so the unfolded round looked finer than FP32; format inference now
  carries the arm's least magnitude to `logb` (`test_triton.py`'s
  `hopper-rne`, `cdna3-rne-7`).
- **What the graphs show** (`results/plots/sweep/`; normwise backward
  error, log2, unless said):
  - **`F` and `rho_bits` trade one for one until the other limits:** each
    bit of either removes about one bit of error, until the error reaches the
    other's precision.  Hopper's FP8 (`F` 13, `rho` E8M13) plateaus at -13
    either way; the BF16 designs plateau near -22 to -24.6 (`rho` FP32
    toward zero), CDNA3 at -25.6 (to nearest).
  - **`L`:** longer instructions err less, about half a bit a doubling, for
    every anchor but Hopper's FP8, flat at -12.4: its 13 bits, not its chain,
    set its error.  Its magnitude bias does halve a doubling (-210 u at `L`
    2, -88 at 64): fewer instructions truncate to E8M13.
  - **`rho_mode`:** to nearest is 2-3 bits better than toward zero and
    removes the bias; down and up are as toward zero in size, their bias of
    either sign.  Not for Hopper's FP8: -12.34 in every mode, its 13-bit sum
    truncated before `rho` rounds; to nearest cuts its bias by about 3.6x
    and leaves it negative.
  - **`rho_bits` under CDNA3's to nearest:** one bit of error a bit, -9.5
    at 7 to -25.6 at 23 (before, only 23 compiled).
  - **`ftz`:** flushing subnormals costs Blackwell's FP8 6 bits (-24.6 to
    -18.5); E4M3's subnormals carry information.  BF16's (FP32's range)
    change nothing.
  - **`join`:** CDNA3's separate join, rounded down, is 2 bits better than
    NVIDIA's inside one for the BF16 designs; for Hopper's FP8 its round-down
    at 13 bits makes its bias ten times larger (-1090 u).
  - **`F2`** (CDNA3): error falls until 28 bits, flat past.
  - **`groups`:** even-odd groups barely move the BF16 designs or
    Blackwell; Hopper's FP8 gains 2 bits (-12.3 to -14.3): each half
    truncates at its own largest exponent.
  - **`sum`:** a chain of FMAs is 2.5-4 bits worse than the fused sum (BF16
    -19.6, Blackwell's FP8 -20.8), a rounding per element; Hopper's FP8
    3.5 bits worse (-8.9 against -12.3), each FMA truncated to E8M13, its
    bias 130x the fused sum's.  A tree is within 0.6 bits of the fused sum
    (Blackwell's FP8 -24.4 against -24.6, the BF16 ones -22.5 and -21.6
    against -22.7 and -22.2); Hopper's FP8 tree is 0.7 bits *better*
    (-13.1), its pairwise adds keeping 13 bits each where the fused sum
    truncates its terms to 13 bits below the largest.

##### Phase 6 - The sweep end to end

**Done** (2026-10-21; Qwen3-0.6B and Qwen3.5-0.8B, 10 segments,
`results/sweep/sweep-e2e-<model>.csv`, plots in
`results/plots/sweep-e2e-<model>/`).
- `sweep.py --end-to-end` runs `frozen.py` per scheme over the same points:
  frozen KL vs R1, top-1 disagreement and re-rounded share, the anchors free
  too, R1 rounded toward zero as each scheme's floor (`floor_kl`, a dashed
  line).  `plot_sweep.py --table` draws it as the local sweep.
- **Cost:** 7 h (Qwen3-0.6B) and 8.5 h (Qwen3.5), not the 1.5 h 3.5 s a
  pass predicts; not timed by scheme (`bf16` runs `lm_head`, 151k rows,
  frozen; `fp8-row` leaves it to R0).
- **Refused:** the 10 of local's (5 exact sums, 4 NV even-odd groups,
  CDNA3's chain of FMAs).
- **Anchors**, frozen KL (Qwen3-0.6B; Qwen3.5): Hopper E4M3 1.98e-04
  (2.6e-05); Blackwell E4M3 2.5e-11 (1.2e-11); Ampere BF16 1.9e-10
  (1.1e-10); Hopper BF16 8.6e-11 (6.5e-11); CDNA3 BF16 8.2e-12 (4.6e-12).
  Floors: `fp8-row` 1.2e-11 (6.9e-12), `bf16` 2.0e-12 (1.0e-12).
- **What the graphs show** (both models alike, Qwen3.5 lower):
  - **`F`, `rho_bits`:** 9-12 decades, about two of KL per bit of local
    error, plateauing where local error does (Hopper's FP8 past `F` 16).
  - **`L`:** flat for Hopper's FP8; 1-2 decades over 2-64 for the others.
  - **`ftz`:** 5 decades for Blackwell's FP8 (-10.6 to -5.7, log10).
  - **`sum`:** a chain of FMAs 1.5-2 decades worse; a tree near the fused
    sum, but for Hopper's FP8 0.8 decades better.
  - **`rho_mode`:** down or up about a decade worse than toward zero; to
    nearest 0.3-1.7 decades better.  Hopper's FP8 the same in every mode.
  - **Re-rounded:** under 1% but for the Hopper-FP8-like points (11-16%)
    and the narrowest `F` and `rho_bits`.

##### Phase 7 - The design space

**The design space is the main figure's grid** (decided 2026-10-25): every
design the evaluation reports, and nothing it does not.  `sweep.GRID`: four
anchors -- NVIDIA's fused sum, `c` joined inside (Blackwell E4M3, Hopper
BF16), and AMD's separate join (CDNA3 BF16, CDNA3 E4M3 FNUZ under
`fp8-row:fnuz`) -- each with `F` 10-28 (its own added: CDNA3's 24; `F2 = F +
7` with it), `rho_bits` 13, 16, 23 and rounding toward zero or to nearest;
its other fields as built.  **N = 180** (a test holds it).
- `sweep.py --manifest` compiles the grid, launches each design once on a
  GPU (Triton compiles a kernel at its first launch), and writes
  `results/sweep/manifest.csv`: a row per design, its parameters, name and
  status -- the local error's list, so the two evaluations cover one set.
- Superseded: a wider space (2026-10-23, N = 576: two NVIDIA anchors x `F`
  x 6 accumulators x roundings x fused, tree or chain x `L` 8, 16, 32;
  `results/sweep/space.csv`, 516 run).  Its 60 refusals, all a tree's or
  chain's add (or FMA) rounded to nearest below FP32 ("`Add` rounds to odd,
  which is exact here only where its result is proven inside the format's
  range"), are a compiler gap still open; no grid design reaches it.
- Launching found two grid designs (BF16, `F` 10, a 13- or 16-bit
  accumulator to nearest) that compiled but failed in Triton: the emitter
  rounded an FP16-held value to the integers with `libdevice.nearbyint`,
  which has no FP16; it now rounds in FP32 (`_integral`; `test_triton.py`'s
  `hopper-bf16-f10-rne13`).

##### Phase 8 - The grid end to end

**Done** (2026-10-23; Qwen3.5-0.8B, WikiText-2, 10 segments).  `sweep.py
--end-to-end --grid`: `SPACE`'s anchors, `F` 10-25 x `rho_bits` 13, 16, 23
x toward zero or to nearest, 72 designs, frozen and free
(`sweep-grid-Qwen3.5-0.8B.csv`); `plot_grid.py` draws them 2 x 2, FP8 and
BF16 by as deployed and held to ref.
- **Held to ref:** every accumulator's line follows one path until `F`
  reaches its width, then flattens: the narrower width sets the error.  To
  nearest flattens ~2 decades below toward zero; the FP32 accumulator falls
  to the ±1 ulp line (BF16 to nearest at `F` 25 on it).
- **As deployed:** FP8, every design on K₀ (~1e-02).  BF16 has a knee: `F`
  10 and 13 above K₀, and a 13-bit accumulator above it at every `F` (-3.7
  against -4.3, log10); 16 and 23 bits on K₀ from `F` 16.

##### After the last phase

    .venv/bin/python -m pytest -n 8 tests/unit
    cd examples/mmasim && ../../.venv/bin/python -m pytest -n 8 tests serve/tests
    cd examples/mmasim && ../../.venv/bin/ruff check serve generator tests/test_generator.py

#### Open items

##### Does `compile.DESIGNS` come from the catalog?

One source would keep the factories and the generator from drifting; but
the factories are MMA-Sim's algorithms, line for line, and checked against
its reference.  **Provisional:** keep both, the catalog test holding them
equal.  Reopen if the two must change together often.

##### Which parameters' interactions need a second axis?

One at a time cannot show an interaction (`F` against `rho`'s precision,
say: a `rho` narrower than `F` discards what `F` kept).  **Provisional:**
one at a time, as asked; a pair whose one-at-a-time graphs disagree across
anchors is a candidate for a two-axis grid later.

##### What is the generator's interface for formats and scales?

Contexts as the factories take them (`EFloatContext`, `ExpContext`), or the
format names `quant` uses.  **Provisional:** contexts, as the factories
(Phase 1 built it so); Phase 2's registered names and `about` need a
readable form of each, derived from the context, not a second interface.

##### How many tokens per design?

2048 is Stage 2's size; the Roadmap suggested 512 for a cheap level.
**Provisional:** 2048: Phase 2's calibration puts ~300 designs at ~1 h.

##### Which anchors?

**Provisional:** the five above; more as the graphs suggest (a design at a
parameter's extreme, a vendor mix that stands out).

## Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* belong in this
  document, not in source comments.
- `pytest -n 8`, never `-n auto`; `serve/tests` serially (`-p no:xdist`):
  they load models on the GPU, and 8 workers can exhaust its 12 GB (147
  tests, ~70 s).

## Context

`examples/mmasim/compile_triton.py` compiles each MMA-Sim design into a Triton
matmul, `out[i][j] = design(A[i], BT[j], C[i][j])`.
- Every output bit matches the FPy interpreter.
- `m`, `n` and `k` are symbolic.
- 60 of 62 designs compile (`backend-triton.md`).

The designs used (`kernels.TILES`) are each validated against MMA-Sim:

| kind | designs | `k` per instruction | scales |
|---|---|---|---|
| BF16 -> FP32 | `nv.ampere.bf16.f32` (also Ada), `nv.hopper.bf16.f32` (also Blackwell), `amd.cdna2.bf16`, `amd.cdna2.bf16_1k`, `amd.cdna3.bf16` | 16, 32, 4, 4, 8; chained | none |
| OCP E4M3 -> FP32 | `nv.ada.e4m3.f32`, `nv.hopper.e4m3.f32`, `nv.blackwell.e4m3.f32` | 32; chained | none |
| FNUZ E4M3 -> FP32 | `amd.cdna3.fp8` | 16; chained | none |
| block-scaled | `nv.blackwell.mx.e4m3`, `nv.blackwell.mx.e2m1` | 32 | one E8M0 per operand row |
| block-scaled | `nv.blackwell.mxfp4`, `nv.blackwell.nvfp4` | 64 | four E8M0 / UE4M3 per operand row |

Storage in the kernels:
- BF16, FP8 and FP4 values are held in FP32 or FP16 storage, exactly.
- E8M0 scales are FP32; UE4M3 scales are FP16.

Speed: 170-1,500 GFLOP/s on the TITAN V, against ~27,000 for FP16
`torch.matmul`.

The FP16-accumulating designs (`*.f16`) compile but are not used: under
every recipe here, `448^2 > 65,504` overflows FP16 (Nice to have).

## Methodology

It follows low-precision and quantization evaluation.

### Matmuls

The matmuls share one model and one input stream.  They differ only in how
`nn.Linear` computes `x @ W.T`; everything else runs in FP32 with TF32 off.

| matmul | operands | product | answers |
|---|---|---|---|
| **R0 `fp32`**: the pre-quantization reference | FP32 activations, master weights | FP32 | the model as trained (Yuan et al.'s "LayerCast") |
| **R1 `<scheme>-exact`**: the post-quantization reference | quantized by the scheme | FP64, rounded once to FP32 | the cost of the scheme alone |
| **a design** | quantized by the scheme | the design's kernel, scales applied as the scheme says | the cost of the design |

The two references split a design's error:
- **design vs R0** is quantization plus design: the observable effect.
- **R1 vs R0** is quantization alone.
- **design vs R1** is the design alone.

R1 is an ideal MMA on the quantized operands: it rounds their exact product
once.  A design differs from it only in how it accumulates.  The design's
error is 2^10 or more times smaller than the quantization's, so it is
visible only against R1.

Settled choices:
- **Scope: every linear layer.** `lm_head` is included under `bf16`; under a
  quantizing scheme it is left out, as every checkpoint leaves it.
  Attention's `QK^T` / `PV` and the DeltaNet recurrence stay FP32 (Essential
  gaps).
- **The GEMM result stays FP32,** the designs' output format.
- **Accumulation order.** Within a slice of `k`, the design's instructions
  chain in order.
  - `split_k` slices, each from `C = 0`, are combined in FP32 left to right
    or as a tree.
  - Every result here is at `split_k = 1`.
- **Engine: `transformers`,** with each linear layer's `forward` replaced
  (`swap.patch`): deterministic and eager.

### Schemes, designs, models

- **A scheme** gives each operand's element format and its scales: format,
  block, static or dynamic.
- **A design** is an MMA.  It applies to a scheme when it takes the scheme's
  formats exactly (`kernels.applicable`), so the scheme picks its designs.
- **A model** stores its weights either in a master precision (BF16) or in
  a checkpoint's scheme.

| scheme | elements | scales | applied | designs |
|---|---|---|---|---|
| `bf16` | BF16 | none | -- | the five BF16 |
| `fp8-row` | E4M3 | FP32, x per token, w per output row | epilogue, `acc * s_x[i] * s_w[j]` | the E4M3 chains |
| `fp8-block` | E4M3 | FP32, x per 1 x 128, w per 128 x 128 | per 128 of `k`, `y + p * (s_x * s_w)` in FP32 (DeepGEMM) | the E4M3 chains |
| `mxfp8` | E4M3 | E8M0 per 32 | in the instruction | `mx.e4m3` |
| `mxfp4` | E2M1 | E8M0 per 32 | in the instruction | `mx.e2m1`, `mxfp4` (each scale to both 16-groups) |
| `nvfp4` | E2M1 | UE4M3 per 16, FP32 per tensor | block in the instruction, tensor `acc * (g_x * g_w)` | `nvfp4` |

- `:fnuz` gives the software-scaled FP8 schemes FNUZ elements, for CDNA3.
- A block-scaled design's `k` runs as a chain of launches, each taking the
  last one's result as `C`.

### Recipes

Weights from a master, and all activations, are quantized by round-to-nearest
quantization ("RTN").
- It is the PTQ baseline: scale, then round each value to the nearest level
  of its format (RNE), with no calibration.
- Activations are quantized from BF16, as a deployment holds them.
- The quantizers compute exactly what `torchao`'s do, in a few tensor
  ops (`quant.quantize`).  `test_quant.py` checks them bit for bit against
  torchao on hard finite inputs:
  - **FP8:** `Float8Tensor`'s `PerRow` / `PerBlock`, `s = amax / 448`.
  - **MX:** `MXTensor.to_mx` with `ScaleCalculationMode.RCEIL`.  This is
    NVIDIA's (cuBLAS's) rule: the power of two at or above `amax / max`, so
    no element saturates.
  - **NVFP4:** `nvfp4_quantize`, NVIDIA's two-level recipe.  Per tensor `g = amax / (448 * 6)`; per 16, `s =
    rne_e4m3((amax_block / 6) / g)`.  A dynamic `g` is per call, so on
    cached inputs it is per row, from the row's sequence's `amax`.
  - FP8's scale is bounded below, so an all-zero block quantizes to zeros,
    not NaN.
  - E2M1 is rounded to nearest-even in float ops: steps of 1/2, 1 and 2 by
    binade.
- `quant.Quantized.dequantize` is our own: it multiplies out in FP64, where
  `torchao`'s rounds to FP32.

Checkpoints (`checkpoints.py`, `compressed-tensors`):

| checkpoint | weights | activations |
|---|---|---|
| `RedHatAI/Qwen3-0.6B-FP8-dynamic` | `fp8-row`, BF16 scales | dynamic per token |
| `RedHatAI/Qwen3-0.6B-FP8-BLOCK` | `fp8-block`, BF16 scales; no `base_model` on its card, so `--baseline` | dynamic per 1 x 128 |
| `kaitchup/Qwen3-0.6B-NVFP4` | `nvfp4`; FP4 low nibble first, scale `s / g` | static per tensor (`input_global_scale`), dynamic per 16 |

Where a model's weights come from is recorded with every result
(`checkpoints.weights_for`):

| stored | under a scheme | source |
|---|---|---|
| a master | RTN | `rtn` |
| a checkpoint in the scheme | as stored | `checkpoint` |
| another scheme, lossless conversion | converted | `converted` |
| another scheme, lossy conversion | refused unless `--requantize` | `requantized` |

### Local metrics on cached activations (the primary evaluation)

**Capture** (`local.capture`):
- One prefill per workload sequence through the scheme's exact matmul.
- Each linear layer's input is recorded as BF16 on the host, at the sampled
  positions.  There is one tensor per distinct input: `q/k/v_proj` share
  one, as do `gate/up_proj`.

**Evaluate** (`local.evaluate`): each design runs only its kernel, on the
cached `(x, W)`.
- Every design sees identical inputs, and there is no propagation between
  layers.
- Means are within ~0.01 in log2 of each design's metrics on its own inputs,
  the model run through it (measured under `bf16`; `test_local.py` checks
  where the inputs agree).

**Two references** (parallel tables in every run; the JSON keys in
parentheses):
- **post-quantization (`quantized`):** the quantized operands' exact
  product, scales included, R1's product at that layer.  This measures the
  design's own effect.
- **pre-quantization (`unquantized`):** the captured BF16 activation times
  the master weight.  This measures quantization and design together.  It
  is left out for a checkpoint with no known master.
  - Its activation is BF16, not R0's FP32, so under `bf16` it coincides
    with the post-quantization reference (`quantization` is 0).
- `quantization` is the first reference's normwise error against the
  second.
- FP64 is exact for the BF16, FP8 and E8M0 products.  With FP32 scales it is
  exact to ~`2^-50` of `|x|ᵀ|w|`.

**Workloads** (`workloads.py`, `-w`):
- **`wikitext`:** the first 2048 tokens of the WikiText-2 test split, the
  calibration convention.
- **`mtbench`:** MT-Bench's 80 two-turn conversations as a user would have
  them.
  - The chat template is applied, with thinking off.
  - Each reply is R0's greedy decode, at most 512 tokens, generated once
    (18 min) and cached.
  - A conversation's sequence is its last turn as the model processes it;
    one prefill gives each layer the inputs incremental decoding would.
  - 2048 positions are sampled with a fixed seed.  Each is tagged by
    `category` and by `role` (`user`, `assistant`, `template`, the last
    including the empty think block); `--by` splits the metrics by either.
- Calibration data shifts quantization results (Williams & Aletras), hence
  a chat workload next to prose.

### End to end: frozen runs (the second evaluation)

Local metrics are the per-matmul evaluation (built separately).  This one
asks what local error cannot: how much of a design's error reaches the
application's output, and whether it matters there.  Decided 2026-10-21
from the frozen sweep (Stage 3, Phase 6; Record, "Frozen runs").

| | Local | End to end |
|---|---|---|
| question | how accurate is each matmul? | how much of that error reaches the model's output, and does it matter there? |
| metric | per-matmul error against the exact product | frozen KL(R1 \|\| design), in bits |
| adds | -- | the model's and workload's weighting; error structure local means underweight; where differences stop showing in deployment |

**Method** (`frozen.py`, `scoring.frozen`, `swap.Frozen`):
- WikiText-2, teacher-forced, 10 random 2048-token segments, paired across
  designs.
- Reference R1: the same quantized model, exact matmuls (FP64, rounded once
  to FP32).
- Each design is its compiled kernel, *frozen* to R1: a layer multiplies
  R1's quantized input `Q(x_R1)` by the design's arithmetic, and its own
  input's offset passes through the weight unrounded, `+ (x - x_R1) W_q`.
  Its arithmetic's error enters every layer and propagates; no activation
  rounds differently from R1's.  Without this, any inexact design's
  re-rounding costs K₀ and hides the arithmetic.
- KL in FP64 log-softmax (FP32's rounding alone is ~1e-9).
- Checks: R1 frozen is 0 exactly; KL ∝ ε² (slope 2.00, R² 0.9996, the
  width sweep), as KL ≈ ½ δᵀFδ predicts.
- Validity: the share of activation elements the design's own run would
  round differently (`rerounded`).  Under ~1% the frozen run is the
  design's; above (Hopper-like FP8 points, 11-16%), its linearized effect.

**Metric: frozen KL(R1 || design),** mean per token, on a log axis, read
against a calibration: the measured frozen KL of R1 with every matmul output
rounded to `t` fraction bits, `t` 8-22 (`-exact-p<t>`; Monte Carlo
arithmetic's virtual precision).  The calibration is the methods' evidence
and unit (one decade of KL is ~1.7 bits of output precision); the figure
leaves it out.
- **The curves** (every scheme, both models, 10 segments, 2026-10-21;
  `results/frozen/calib-*`): 2.0 bits of KL a bit of precision (KL ∝ ε²),
  every scheme but `mxfp4` (2.4-3.6: its outputs often have so few bits that
  rounding leaves them exact), until the floor.
- **The floor is FP32 inference's own rounding**, R1 rounded toward zero (a
  dashed line).  Freezing fixes the quantizer, not the model's FP32 norms,
  attention and softmax: any perturbation re-rounds them, the mechanism of
  K₀ a level down.  ~1e-11 under the FP8 and FP4 schemes (the curve flattens
  from `t` ~15), ~2e-12 under `bf16` (~19).  A design at the floor injects
  less error than FP32 inference's own rounding; it is reported so, not
  ranked.
- **Biased error compounds.**  Hopper's FP8 (truncation) is farther from R1
  than R1 rounded to 8 fraction bits everywhere -- over 15 bits lost of
  FP32's 23 -- where its local mean ULP error is 12.5 bits.  A chain of FMAs
  and Blackwell's `ftz` sit above their local error too ("How it relates to
  local").
- **K₀,** the free run's re-rounding (R1 rounded toward zero, free: `fp8-row`
  1.3e-02, `bf16` 6.7e-05), is reported beside the figure, not on it: above
  the calibration's top, it would take extrapolating.

**Harness: FP32** (decided 2026-10-21).  FP64 non-matmul operations were
tried (Qwen3-0.6B, `F` and `rho_bits`, 10 x 1024 tokens, `results/fp64`):
identical above the FP32 floor, the floor 40x lower under `fp8-row` (6x
under `bf16`), resolving Blackwell-class FP8 designs.  Dropped: a second
counterfactual, for differences below FP32 inference's own rounding, which
no deployment shows; and 2048 tokens do not fit 12 GB.  Its code is gone.
(A first reading of bits as `½ log2(KL / floor)` treated the floor as an
ulp and understated what designs lose by ~4.7 bits; withdrawn with the
fitted reading that followed it.)

**How it relates to local** (141 sweep points; local from `sweep.csv`,
Qwen3-0.6B; end to end as `½ log2 KL`, bits by the ε² law):

| model | scheme | R² vs mean ULP error | slope | residual SD |
|---|---|---|---|---|
| Qwen3-0.6B | `fp8-row` | 0.96 | 0.86 | 1.1 bits |
| Qwen3-0.6B | `bf16` | 0.98 | 1.02 | 0.8 bits |
| Qwen3.5-0.8B | `fp8-row` | 0.99 | 0.89 | 0.5 bits |
| Qwen3.5-0.8B | `bf16` | 0.99 | 1.04 | 0.6 bits |

- Correlated, as expected: end to end is mostly local error carried
  through the model.
- **The residuals are what it adds.**  Blackwell's `ftz` +4.5 bits beyond
  its local error (Qwen3.5 +2.2); a chain of FMAs +1.7; narrow `rho_bits`
  -1.5 (the model tolerates it).  The same Hopper FP8 design is ~1.5 bits
  milder on Qwen3.5.
- From 10 segments of one workload; a second (`mtbench`) before the
  residuals carry a claim.

**The figure** (`plot_sweep.py`, `kl.png`): x a design parameter, y log10
KL(ref ‖ design) on one scale across panels ("ref", R1, for the paper: the
quantized model, every matmul exact and rounded once to FP32), a line per anchor, each
scheme's FP32 floor dashed.  "Frozen" and the calibration are the
methods'.

**Cost:** profiled (2026-10-21), the design's emulated GEMM is 83%
(`fp8-row`) to 91% (`bf16`) of a frozen pass; copies were 4-8%, now
pinned and asynchronous (11% faster, bit-identical results).  Sweeps are
bound by the compiled kernels.

**A second workload: `mtbench`** (2026-10-21; Qwen3-0.6B, 10 conversations,
the `F` and `rho_bits` axes, 77 designs; `sweep-e2e-Qwen3-0.6B-mtbench.csv`,
plots beside WikiText-2's):
- **The same measurement up to a constant:** log2 KL on MT-Bench is 1.00x
  WikiText-2's, 0.7 bits lower under `fp8-row`, 0.4 under `bf16` (chat a
  little less sensitive); Spearman 0.98 and 0.999; the floors lower alike
  (7e-12, 1.3e-12).
- **Its calibration** is the same, 2.0 bits of KL a bit; K₀ 7.7e-03
  (`fp8-row`), 4.3e-05 (`bf16`).
- **The residuals over local error recur,** design for design: the BF16
  designs at `F` 22 +1.8 and +1.9 bits on both, at `rho_bits` 13 -1.4 and
  -1.5.  Not run here: `ftz` and the chain of FMAs (other axes).

### Metrics

`Y` is a layer's output matrix and `y` one element; `u = 2^-24`.  "log2"
columns report `log2` of the value, so `-24` is one unit roundoff.

| metric | definition | pooled over | source |
|---|---|---|---|
| normwise relative error | `‖Ŷ - Y‖_F / ‖Y‖_F`, log2 | every token stacked | Higham |
| componentwise backward error | `\|ŷ - y\| / (\|x\|ᵀ\|w\|)`, mean and max, log2 | elements | Oettli-Prager; Higham ch. 7 |
| ULP error | `log2(1 + \|ŷ - y\| / ulp(y))`, FP32 ulp, mean and max | elements | Herbie / FPBench |
| correct-rounding rate | fraction with `ŷ = fl(y)` (quantized reference only) | elements | |
| bias | mean of `(ŷ - y) / (\|x\|ᵀ\|w\|)` in u: drift about zero | elements | |
| magnitude bias | mean of `sign(y) (ŷ - y) / (\|x\|ᵀ\|w\|)` in u; negative leans toward zero | elements | |
| quantization | `‖Y_q - Y_0‖_F / ‖Y_0‖_F`, log2 | every token stacked | |

`-m` selects metrics, and only those are computed.

End-to-end metrics (`core/scoring.py`, `zeroshot.py`, `core/stats.py`):
- **Every matmul sees the same seeded subset of units:**
  - WikiText-2 segments of 2048 tokens (`--segments 50`);
  - MATH-500 prompts with R0's greedy reply (`--prompts 30`);
  - benchmark items (`--items 500` per task).
- **Each matmul is paired against a reference, per unit:**
  - R0 (pre-quantization), for every matmul;
  - R1 (`<scheme>-exact`, post-quantization), for every design.
- **Statistics:** means over units with standard errors
  `sd(d) / sqrt(n)` (the normal approximation); Holm's correction over the
  matmuls sharing a reference; a bootstrap over units for the median
  divergence index.
- `p_t` is the reference's next-token distribution, `q_t` the matmul's;
  teacher-forced on the unit's tokens.

| metric | per unit | source |
|---|---|---|
| Δ NLL | mean NLL of the unit's tokens, matmul minus reference (the log of the perplexity ratio) | GPTQ; HF guide |
| KL divergence | `mean_t KL(p_t ‖ q_t)` | llama.cpp |
| top-1 disagreement | `mean_t [argmax p_t ≠ argmax q_t]` | llama.cpp |
| divergence index | the first disagreement on R0's greedy reply; the fraction of prompts that diverge | Yuan et al. (on prefill) |
| Δ acc, flips | per item: correctness, and whether it changed | lm-eval; Dutta et al. |
| Δ log-likelihood, margin, choice KL | per item: the correct choice's log-likelihood, its margin over the best wrong one, KL over the choices | |

The per-matmul, token-level figures (PPL, llama.cpp's KL, top-1 and RMS Δp,
and harness accuracy) are kept for comparison with published ones; their
token-level standard errors understate the uncertainty.

## Interface

```
cd examples/mmasim
python serve/local.py                                   # bf16, Qwen3-0.6B, every design
python serve/local.py --scheme fp8-row --models Qwen/Qwen3-0.6B RedHatAI/Qwen3-0.6B-FP8-dynamic
python serve/local.py --scheme fp8-block --models RedHatAI/Qwen3-0.6B-FP8-BLOCK --baseline Qwen/Qwen3-0.6B
python serve/local.py --scheme mxfp4 --models kaitchup/Qwen3-0.6B-NVFP4 --requantize
python serve/local.py --scheme nvfp4 -d nv.blackwell.nvfp4 -w mtbench --by role -m normwise backward
python serve/{perplexity,zeroshot,decode}.py --model Qwen/Qwen3.5-0.8B   # bf16 matmuls
python serve/chat.py                                    # terminal chat, matmul switchable mid-conversation
python serve/chat.py --model kaitchup/Qwen3-0.6B-NVFP4 --scheme nvfp4 --matmul nv.blackwell.nvfp4
python serve/chat.py --list-models                      # also --list-schemes, --scheme nvfp4 --list-matmuls
```

Options:
- `--help` gives every option and its default; `--list-models`,
  `--list-schemes` and `--list-matmuls` (the scheme's) list what they take.
- `--scheme` defaults to `bf16`; `-d` defaults to every design applicable to
  the scheme.
- `--tokens` sets how many positions to sample.  `--layers` filters layers
  by regex.  `-o` writes the JSON; `local.py` also a table beside it
  (`.csv`, a row per model, reference, design and group).
- `--register FILE` (repeatable) runs the file's `register()`, which adds
  designs with `kernels.register(name, build, block, block_m, about=...)`,
  before anything else: they are then designs as any other (`-d`,
  `--matmuls`, the lists).  *about* describes the design for the results.
- `--results DIR` (default `results`): the outputs' root.  `cache/` holds
  what runs reuse: compiled kernels (`cache/kernels`, keyed by the design,
  its *about*, and a digest of every source a kernel depends on; a design
  registered without *about* is not kept), captured activations
  (`cache/acts`, keyed by their settings), the MT-Bench conversations.
  `plots/` holds plots; a run's outputs go in a directory of their own.
- **Every output** is written whole or not at all (`outputs.write_json`,
  `write_csv`: to a temporary file, then renamed), and carries its
  `provenance`: the command line, the commit and whether tracked files
  differed from it, the `--register` files' digests, every registered
  design's *about*.

## Layout

```
examples/mmasim/serve/
  local.py        capture, then local metrics per design (the primary evaluation)
  perplexity.py   WikiText-2 segments, scored against R0 and R1
  zeroshot.py     lm-evaluation-harness suite, paired Δ acc and flips
  decode.py       R0's greedy decode, then every matmul teacher-forced on it
  chat.py         terminal chat
  sweep.py        hardware design parameters swept around anchors (generator/), local metrics
  plot_sweep.py   the sweep's graphs, from its table
  core/           the library; no script imports another script
    kernels.py      compile each design (precompile in parallel); prepare, matmul, linear; register
    quant.py        schemes, torchao's RTN, Quantized with an exact FP64 dequantize
    swap.py         a model's nn.Linear forwards through a Run (mode, scheme)
    checkpoints.py  compressed-tensors checkpoints; weights_for, for_scheme
    metrics.py      the local metrics (Stats, local); paired, p_value, holm, bootstrap
    scoring.py      matmuls scored against references, teacher-forced (Totals, evaluate, against)
    workloads.py    WikiText-2, MT-Bench sessions, MATH-500 prompts; pick
    generate.py     greedy generation under the chat template
    cli.py          the scripts' shared options
  tests/          per module (workloads in test_local.py)
```

Outputs go under the scripts' `--results` root (`cli.RESULTS`, `results/`
from `examples/mmasim`, gitignored): `cache/` for artifacts reused across
runs (the MT-Bench cache, `cache/mtbench/<model>.json`), `plots/` for plots,
a run's outputs in a directory of its own.

## Results

2048 positions per workload, every linear layer pooled; log2 unless marked
u.  "RTN" is the master quantized here.  The checkpoints were run on
WikiText-2 only: on MT-Bench each would need its own conversations, ~18 min
of decoding apiece.

The quantizing schemes' tables predate two review fixes: the capture pass
now quantizes activations from BF16 (it took FP32), and NVFP4's dynamic
scale is per sequence on cached inputs (it pooled every row).  Re-run on
Qwen3-0.6B, the numbers move by at most 0.15 in log2, and the quantization
errors not at all:

| scheme | design | workload | normwise before | after |
|---|---|---|---|---|
| `nvfp4` | nv.blackwell.nvfp4 | WikiText | -23.03 | -23.18 |
| `nvfp4` | nv.blackwell.nvfp4 | MT-Bench | -23.25 | -23.21 |
| `fp8-row` | nv.blackwell.e4m3.f32 | WikiText | -21.04 | -21.09 |
| `fp8-row` | nv.ada.e4m3.f32 | WikiText | -8.80 | -8.80 |

### Qwen3-0.6B: the observable effect (against the pre-quantization reference)

The observable effect is each design's output against the unquantized
operands' exact product.  For every quantizing scheme and every design, it
equals `quantization` to within 0.02, even under Ada's and Hopper's FP8
accumulators.  So the table gives the quantization error alone:

| scheme | model | quantization (WikiText) | quantization (MT-Bench) |
|---|---|---|---|
| `fp8-row` | RTN | -5.24 | -5.25 |
| `fp8-row` | FP8-dynamic | -5.25 | - |
| `fp8-row:fnuz` | RTN | -5.24 | -5.26 |
| `fp8-block` | RTN | -5.25 | -5.28 |
| `fp8-block` | FP8-BLOCK | -5.25 | - |
| `fp8-block:fnuz` | RTN | -5.26 | -5.28 |
| `mxfp8` | RTN | -5.19 | -5.21 |
| `mxfp4` | RTN | -2.81 | -2.81 |
| `nvfp4` | RTN | -3.27 | -3.28 |
| `nvfp4` | NVFP4 | -3.26 | - |

Under `bf16` the cached inputs and the master are BF16 already, so the
quantization error is 0 and the observable effect is the design's own
(below).

### Qwen3-0.6B: the design's own effect (against the post-quantization reference)

Against the quantized operands' exact product:

| scheme | model | design | normwise | backward mean | correctly rounded | magnitude bias (u) | normwise (MT-Bench) |
|---|---|---|---|---|---|---|---|
| `bf16` | RTN | nv.ampere.bf16.f32 | -18.52 | -23.15 | 0.82% | -1.709 | -18.46 |
| `bf16` | RTN | nv.hopper.bf16.f32 | -18.97 | -23.41 | 0.85% | -1.429 | -18.95 |
| `bf16` | RTN | amd.cdna2.bf16 | -21.27 | -25.88 | 10.70% | -0.000 | -21.31 |
| `bf16` | RTN | amd.cdna2.bf16_1k | -21.52 | -26.08 | 11.76% | -0.000 | -21.56 |
| `bf16` | RTN | amd.cdna3.bf16 | -21.91 | -26.38 | 13.88% | +0.000 | -21.93 |
| `fp8-row` | RTN | nv.ada.e4m3.f32 | -8.80 | -14.61 | 0.01% | -125.913 | -8.41 |
| `fp8-row` | RTN | nv.hopper.e4m3.f32 | -8.80 | -14.63 | 0.01% | -104.968 | -8.41 |
| `fp8-row` | RTN | nv.blackwell.e4m3.f32 | -21.04 | -27.19 | 34.71% | -0.097 | -21.10 |
| `fp8-row` | FP8-dynamic | nv.ada.e4m3.f32 | -8.78 | -14.62 | 0.01% | -125.154 | - |
| `fp8-row` | FP8-dynamic | nv.hopper.e4m3.f32 | -8.79 | -14.64 | 0.01% | -104.254 | - |
| `fp8-row` | FP8-dynamic | nv.blackwell.e4m3.f32 | -21.08 | -27.21 | 35.14% | -0.095 | - |
| `fp8-row:fnuz` | RTN | amd.cdna3.fp8 | -23.56 | -28.23 | 47.76% | +0.000 | -23.54 |
| `fp8-block` | RTN | nv.ada.e4m3.f32 | -12.24 | -16.74 | 0.02% | -43.572 | -12.30 |
| `fp8-block` | RTN | nv.hopper.e4m3.f32 | -12.31 | -16.83 | 0.02% | -38.439 | -12.38 |
| `fp8-block` | RTN | nv.blackwell.e4m3.f32 | -23.63 | -27.90 | 34.35% | -0.014 | -23.64 |
| `fp8-block` | FP8-BLOCK | nv.ada.e4m3.f32 | -12.25 | -16.75 | 0.02% | -43.445 | - |
| `fp8-block` | FP8-BLOCK | nv.hopper.e4m3.f32 | -12.32 | -16.84 | 0.02% | -38.359 | - |
| `fp8-block` | FP8-BLOCK | nv.blackwell.e4m3.f32 | -23.62 | -27.91 | 34.56% | -0.013 | - |
| `fp8-block:fnuz` | RTN | amd.cdna3.fp8 | -23.65 | -27.96 | 35.25% | -0.001 | -23.67 |
| `mxfp8` | RTN | nv.blackwell.mx.e4m3 | -21.01 | -27.17 | 38.36% | -0.105 | -21.09 |
| `mxfp4` | RTN | nv.blackwell.mx.e2m1 | -25.39 | -50.24 | 100.00% | -0.000 | -inf |
| `mxfp4` | RTN | nv.blackwell.mxfp4 | -25.39 | -50.24 | 100.00% | -0.000 | -inf |
| `nvfp4` | RTN | nv.blackwell.nvfp4 | -23.03 | -29.18 | 75.56% | -0.000 | -23.25 |
| `nvfp4` | NVFP4 | nv.blackwell.nvfp4 | -23.23 | -29.18 | 76.84% | -0.005 | - |

### Qwen3.5-0.8B (RTN, WikiText-2)

Against the quantized operands' exact product.  As on Qwen3-0.6B, the
observable effect equals `quantization` to within 0.01 for every quantizing
design.

| scheme | design | normwise | backward mean | correctly rounded | magnitude bias (u) | quantization |
|---|---|---|---|---|---|---|
| `bf16` | nv.ampere.bf16.f32 | -19.19 | -23.08 | 0.73% | -1.830 | -inf |
| `bf16` | nv.hopper.bf16.f32 | -19.46 | -23.32 | 0.75% | -1.541 | -inf |
| `bf16` | amd.cdna2.bf16 | -21.59 | -25.18 | 11.39% | -0.003 | -inf |
| `bf16` | amd.cdna2.bf16_1k | -21.78 | -25.29 | 12.55% | -0.001 | -inf |
| `bf16` | amd.cdna3.bf16 | -22.10 | -26.27 | 14.84% | +0.000 | -inf |
| `fp8-row` | nv.ada.e4m3.f32 | -10.09 | -14.53 | 0.01% | -161.393 | -5.79 |
| `fp8-row` | nv.hopper.e4m3.f32 | -10.10 | -14.54 | 0.01% | -140.040 | -5.79 |
| `fp8-row` | nv.blackwell.e4m3.f32 | -22.92 | -27.38 | 43.53% | -0.082 | -5.79 |
| `fp8-row:fnuz` | amd.cdna3.fp8 | -23.87 | -28.21 | 53.62% | +0.000 | -5.79 |
| `fp8-block` | nv.ada.e4m3.f32 | -12.62 | -16.78 | 0.02% | -50.180 | -5.80 |
| `fp8-block` | nv.hopper.e4m3.f32 | -12.72 | -16.89 | 0.03% | -45.144 | -5.80 |
| `fp8-block` | nv.blackwell.e4m3.f32 | -23.82 | -27.91 | 37.50% | -0.003 | -5.80 |
| `fp8-block:fnuz` | amd.cdna3.fp8 | -23.83 | -27.92 | 37.65% | -0.001 | -5.81 |
| `mxfp8` | nv.blackwell.mx.e4m3 | -23.02 | -27.50 | 51.57% | -0.085 | -5.74 |
| `mxfp4` | nv.blackwell.mx.e2m1 | -36.63 | -46.24 | 100.00% | -0.000 | -3.47 |
| `mxfp4` | nv.blackwell.mxfp4 | -36.80 | -46.16 | 100.00% | -0.000 | -3.47 |
| `nvfp4` | nv.blackwell.nvfp4 | -24.71 | -28.93 | 72.66% | -0.001 | -3.91 |

### Findings

- **The observable effect is the quantization's.** Under every quantizing
  scheme, it swamps the design's own effect, even Ada's and Hopper's
  2^-8.8.  A design is observable only where nothing is quantized (`bf16`).
- **The designs separate cleanly on their own effect,** the same way on both
  models and both workloads.
  - **Ada's and Hopper's FP8 accumulators:** 13 bits, truncating.  They sit
    at 2^-8.8 to 2^-10.1, leaning strongly toward zero, as DeepSeek-V3
    reports.  Promotion every 128 of `k` (`fp8-block`) buys ~3 bits.
  - **Blackwell and CDNA3 FP8:** 2^-21 to 2^-24.
  - **The NV BF16 designs** lean toward zero (truncation) and drift upward
    about zero; the AMD designs are unbiased.
- **Blackwell's MXFP4 paths are exact** wherever the exact product fits
  FP32: E2M1 products under power-of-two scales.  Where it does not fit
  (Qwen3-0.6B's `layers.2.mlp.down_proj`; six of Qwen3.5's
  `linear_attn.out_proj`), each instruction's FP32 result rounds, but still
  ≥ 99.97% of outputs are correctly rounded.
- **CDNA2 flushes subnormal products** (FTZ-Mul).  On Qwen3.5 this puts its
  max backward error at 2^-8.9, from block 0's `in_proj_qkv`, whose weight
  row 423 holds values ~1e-37.
- **RTN stands in for the checkpoints:** they agree within ~0.05 on every
  metric and design.  Asking the FP8 checkpoint for `bf16` is refused, since
  196 layers would requantize lossily.
- **The workload matters little to the ranking, but it does matter.**
  - Designs rank the same on WikiText-2 and MT-Bench, in every role and
    category, with means within ~0.1.
  - Exceptions: `fp8-row`'s truncating accumulators are ~0.4 worse on
    MT-Bench; Ampere's BF16 drift is +0.35 u on user tokens, +0.82 u on its
    own replies and -0.51 u on template tokens.
- **MX scale rounding (RCEIL vs the OCP floor):**
  - MXFP8 quantization improves, 2^-4.97 to 2^-5.19.
  - MXFP4 gets slightly worse, 2^-2.87 to 2^-2.81: the coarser scale costs
    E2M1's few values more than saturating did.
  - The designs' own errors barely move.

### End-to-end, BF16, smoke scale (Qwen3-0.6B)

WikiText-2, first 8 segments:

| matmul | PPL | KL vs R0 | top-1 |
|---|---|---|---|
| fp32 | 17.8334 | 0 | 100% |
| bf16-exact | 17.8384 | 4.27e-5 | 99.59% |
| nv.ampere.bf16.f32 | 17.8370 | 4.28e-5 | 99.68% |
| nv.hopper.bf16.f32 | 17.8369 | 4.17e-5 | 99.57% |
| amd.cdna2.bf16 | 17.8400 | 4.11e-5 | 99.63% |
| amd.cdna2.bf16_1k | 17.8352 | 4.17e-5 | 99.68% |
| amd.cdna3.bf16 | 17.8365 | 4.24e-5 | 99.55% |

- **BF16 input rounding dominates** end to end: bf16-exact is as far from
  R0 as any design.  At `split_k = 4` (linear and tree), KL stays ~4e-5.
- **Zero-shot:** R0 in full is within one standard error of Zheng et al.'s
  FP16 Qwen3-0.6B on every task:

  | | PIQA | ARC-e | ARC-c | HellaSwag | WinoGrande | LAMBADA |
  |---|---|---|---|---|---|---|
  | R0 `acc` | 67.74 | 60.86 | 31.31 | 37.60 | 55.80 | 40.40 |
  | Zheng et al. | 67.3 | 60.8 | 31.7 | 37.6 | 56.2 | |

- **Decode:** 5 MATH-500 prompts, Yuan et al.'s setup.  bf16-exact diverged
  on 2, CDNA2 on 1, Hopper on 1.
- **Propagated error** (normwise against R0's output at each layer, in the
  retired `layers.py`) is input rounding's, ~2^-8.4, which is 2^10 to 2^13
  times the local error.  The exception is blocks 11-14, where the designs
  add to it.
- **Qwen3.5-0.8B:** PPL 12.46 on segment 0; KL 2.7-2.9e-5 for every matmul.

## Record

Condensed; what the code does not say.

- **Stage 1, end-to-end evaluations** (2026-09-27/28):
  - **Units needed for 80% power** (two-sided, α = 0.05) to separate the
    widest pairs, from 30 units each: `n = (2.8 sd / mean)²` of the
    per-unit difference between the two designs.

    | evaluation (unit) | statistic | Ampere vs CDNA3 (`bf16`) | Ada vs Blackwell (`fp8-row`) |
    |---|---|---|---|
    | perplexity (segment) | KL vs R1 | 39 | 2 |
    | | KL vs R0 | 5,391 | 46 |
    | | NLL | 58 | 433 |
    | decode, forced (prompt) | KL vs R1 | 20,139 | 44 |
    | | top-1 disagreement vs R1 | 78 | 1,005 |
    | | fraction diverged vs R1 | 236 | 114 |
    | zero-shot (item) | accuracy | 1,413 | no difference in 180 |
    | | Δ log-likelihood | 225 | 32,524 |
    | | Δ margin | 1,466 | 1,209 |
    | | choice KL vs R0 | 661 | 988 |

  - Against R0 the BF16 designs are indistinguishable, since input rounding
    swamps them.  Ada's truncating FP8 accumulator shows even against R0
    (+2.6% KL), being only ~2^3.6 below the quantization's error.
  - Decode ranks poorly.  R0's own greedy text is high-confidence, and
    under `fp8-row` every matmul diverges within ~45 tokens.
  - Zero-shot's signed Δ log-likelihood separates BF16 (Ampere's
    truncation biases it), but not FP8.
  - Seconds per segment: R0 0.3, R1 0.7, Ampere 10.5, CDNA3 17.3, Ada 4.4,
    Blackwell 4.7.  Zero-shot runs ~1 item/s for the slow BF16 designs.
- **Teacher-forced and free-running divergence differ prompt by prompt.**
  - The 5 seed-0 MATH-500 prompts:

    | matmul (vs R0) | free-running | forced |
    |---|---|---|
    | bf16-exact | 607, 225 | 607, 225 |
    | amd.cdna2.bf16 | 713 | 368, 607, 225 |
    | nv.hopper.bf16.f32 | 607 | 420, 607 |

  - The kernels are row-independent (bit-identical at `m` = 1, 7, 100).
    The cause is attention: prefill and decoding round it differently
    (~1e-7), which decides a design's near-ties.
  - R0 forced misses none of its own tokens, so that is no bound for the
    designs.
- **Local metrics, several designs per pass** (bit-identical to one at a
  time):
  - `bf16`, five designs: 16.0 s -> 10.1 kernels + 2.8 metrics at 512
    tokens, and 58.1 -> 39.5 + 10.8 at 2048.
  - The per-design comparison, not the FP64 GEMMs, is most of the metrics
    (~1.8 s per design at 2048 tokens).
  - Parallel compile: 13 designs in 9.2 s, against ~38 s.
  - `--acts`: MT-Bench at 512 tokens, 34 -> 13.5 s wall.
  - Blocked logits: a 2,300-token sequence peaks 0.88 GiB above the
    model, against 2.47 GiB.
- **`layers.py` retired.**  Its local metrics matched `local.py`'s within
  ~0.01 in log2 (a test keeps that check), and its `propagated` error is
  what end-to-end KL now measures.

- **Serving overhead.**
  - At `m = 1` the torch-side wrapper set decode speed: ~210 us CPU per
    call, against 19 us for `F.linear`.
  - Fixes: weights prepared once per layer; an input quantized once for the
    layers sharing it; `C` and the output one buffer; `block_m` capped at
    the power of two `>= m`; the exact matmul in row and column blocks.
  - A self-recursive `tree` closure was a reference cycle, holding
    +0.58 GB per `lm_head` call; it is now module-level `_tree`.
  - Decode after the fixes: 13-21 tokens/s per design.
  - Not done: bypassing the launcher's checks (~35 us), and CUDA graphs (a
    static cache changes attention's numerics).
- **End-to-end cost is why local metrics lead.**
  - Full WikiText-2 perplexity takes ~1.6 h per design, and the zero-shot
    suite hours.
  - Decode is launch-bound, at hours per design for 100 prompts.
  - Local metrics take 5-30 s per design on 2048 cached tokens.
- **Qwen3.5 memory.** Its vocabulary is 248,320 (2 GB of logits per
  segment), which forced:
  - R0's log-probabilities held on the host;
  - column blocks in the per-layer metrics;
  - `--batch-size 8` in the harness;
  - a config-built cache for DeltaNet state, and both EOS tokens.
- **Quantizers: torchao's arithmetic, not its code.**
  - Our first MX and FP8 quantizers matched torchao bit for bit.  On NVFP4
    they differed in the order of FP32 operations, so torchao's were taken.
  - An audit then found torchao's eager paths heavy: 31-58 CUDA kernels per
    MX or NVFP4 activation, with a pack to FP4 and our unpack back.
  - So `quant.quantize` now repeats torchao's operations in their order,
    with no packing: NVFP4 0.88 -> 0.29 ms, MXFP4 1.06 -> 0.32 ms, FP8
    0.20 -> 0.13 ms at `m = 1`.
  - They are bit-identical on every finite input tested.  They differ only
    where the input is infinite, or where an all-zero row under a per-row
    NVFP4 scale makes NaN scales in both.
  - MX scales are read as `float8_e8m0fnu` natively, since `exp2` gets
    2^-127 wrong on the GPU.
- **The audit's other easy fixes** (2026-09-28): identical results
  throughout.
  - Reference log-probabilities stay on the GPU, block by block: R0's and
    R1's scoring 2-2.5x cheaper per segment.
  - `generate.stream` keeps only the last position's logits.  For R0 this
    can change the first token after a prefill at a near-tie, since cuBLAS
    runs a different kernel for one row.
  - No host sync for a BF16 activation.
  - `metrics` computes the reference's terms once per block.
  - Decode speeds barely move (FP8 2.9 -> 3.0 tokens/s, NVFP4 1.3 -> 1.5,
    CDNA2 5.9 -> 6.5).  Deferred:
    - caching FP64 weights for exact matmuls (memory);
    - CUDA graphs for decode.
- **The scaled schemes in one kernel** (2026-09-29, `mmasim` branch).
  - A block-scaled design chained over `k`, or an E4M3 design under
    `fp8-block` promoted every 128, compiles as one kernel per static `k`
    (`compile_triton.fuse`, `--fuse K`; `kernels.fused`), bit-identical to
    the launch per instruction or block.  `cli.load` precompiles a run's
    fused kernels over the model's `k`s.
  - `kernels.matmul` fuses at most `FUSED_ROWS` rows: 32 for NVFP4 and
    MXFP4, which are 9-18% slower fused at prefill (at their best tile),
    and every `m` for the MX and `fp8-block` designs, 4-7% faster fused
    even at 2048 rows.
  - Decode, Qwen3-0.6B through `chat.py`, identical text before and after:

    | scheme (design) | before | after |
    |---|---|---|
    | `nvfp4` (checkpoint, `nv.blackwell.nvfp4`) | 2.0 tokens/s | 6.7 |
    | `mxfp4` (`nv.blackwell.mxfp4`) | 2.1 | 7.5 |
    | `mxfp8` (`nv.blackwell.mx.e4m3`) | 1.2 | 10.3 |
    | `fp8-block` (`nv.hopper.e4m3.f32`) | 2.7 | 12.9 |

    Perplexity is bit-identical and 13-27% faster per segment.
  - **What is left of decode is on the CPU:** ~47 ms a token of kernel
    launches, and small torch ops (activation quantization, the weight
    scales' `repeat_interleave`, recomputed every call).  Next: cache a
    weight's scales with its prepared elements, and fewer ops per
    quantization.
  - Backend fixes it needed (`fpy2/backend/triton/emitter.py`): a slice of
    a slice, a slice's offset as a column under a lane loop, and a slice
    along an outer dimension offset by its stride (a miscompile on `main`:
    `xs[1:3][1][5]` read `xs[1][6]`).
  - `serve` starts without `compressed_tensors` (`checkpoints.unpack_e2m1`),
    7.6 -> 2.2 s for `--help`; a chat start stays ~10 s, `transformers`
    importing itself (`sklearn`, `torchao`) and the model.
  - The CLI documents every option and default, lists models, schemes and
    a scheme's matmuls, and calls a run a "matmul" (`--matmuls`,
    `--matmul`) and a checkpoint's master `--baseline`.
- **`fp8-block` padding.** `PerBlock` takes only whole blocks, so short
  rows are padded with zeros, which leave `amax` unchanged.  A `k` that is
  not a multiple of 128 is refused; none occurs in these models.
- **`lm_head` under quantizing schemes** was quantized until Phase 5 of the
  quantized plan, which moved `fp8-row`'s quantization error from 2^-4.87
  to 2^-5.24.  All tables here leave it out.
- **Kernel details.**
  - The interpreter check compares NaN as NaN: CDNA2's kernel gives the
    payload `0x7fffffff`.
  - Each design runs at one fixed tile; autotuning would retune at every
    sequence length.

## Essential gaps

What a reviewer of the methodology would likely require.

### Attention's matmuls, and a long context

`QK^T` and `PV` run on tensor cores in deployment: ~15% of Qwen3-0.6B's
FLOPs at 2048 tokens, and the majority at long contexts.  Capturing only
the linear layers invites "you measured the easy GEMMs".
- **The work:** hook the attention function (batched, causal,
  grouped-query) and capture `Q`, `K`, `V` per head.  Then run `QK^T`, and
  `PV` on the softmax's output, through the designs.
- **The caveat:** FlashAttention rescales `PV`'s partials across key blocks
  (online softmax).  Either model that, or state the simplification: one
  FP32 softmax, then `PV` through the design.
- **A long-context workload** (e.g. 8-32k tokens) is what makes attention
  significant; the per-head capture stays small.

### Local metrics predict end-to-end effects

The claim that local metrics are "the evaluation" needs evidence on a few
designs.  Take the widest separations (Ada vs Blackwell under `fp8-row`,
Ampere vs CDNA3 under `bf16`) and show the end-to-end ordering matches: KL,
perplexity, flips.  This is Stage 2 of the Roadmap.  **Answered** (Stage 2,
parked): yes, as the normwise backward error, at inference on these models;
the effect it predicts is small.

### Uncertainty on local metrics

Pooled means carry no intervals.  Standard errors clustered by sequence
(WikiText-2 segments, MT-Bench conversations) would separate real design
differences of ~0.05-0.1 in log2 from sampling noise (`core/stats.py`
has the statistics).

### A larger model or a second family

The models here are 0.6-0.8B, both Qwen.  Add one larger model (Qwen3-8B,
or a Llama); capturing layer by layer keeps memory bounded.

### The accumulation order

Every result is at `split_k = 1`, one instruction chain over all of `k`.
Libraries split `k` (split-K, stream-K), which shortens each chain.
- For Ada and Hopper this bounds the truncation (`fp8-block` shows ~3
  bits).
- Either report local metrics at a library-realistic `split_k`, or state
  that `split_k = 1` is the worst case.

## Nice to have

- **If there is time: truncating FP8 accumulation at scale.**
  - The theory: Ada's and Hopper's 13-bit truncating FP8 accumulators looked
    harmless in small experiments, and the deviation shows only at scale.
    DeepSeek-V3 saw it at large K and promoted partials to FP32 every 128.
  - Here, on Qwen3-0.6B (K ≤ 3072), Ada adds ~2.6% KL on top of `fp8-row`'s
    quantization end to end.  Its local error already peaks at the largest
    K (`down_proj`).
  - Test it with local error against K (layers, or synthetic operands):
    Ada and Hopper should grow while Blackwell and CDNA3 stay flat.  Then a
    larger model (Qwen3-8B, K up to 12,288), captured layer by layer.
  - Training, where DeepSeek saw it, is out of scope.

- **FP8 with an FP16 accumulator.** The `*.f16` designs need a scheme whose
  scales keep `|x| |w|` summed within FP16's range, e.g. `amax / 16` per
  block, and `kernels.matmul` taking FP16 `C`.
- **More schemes:** E5M2, mixed pairs (E4M3 x, E5M2 w), FP16, TF32.
  Formats are scheme parameters already.
- **The DeltaNet recurrence's matmuls** through the designs (Qwen3.5).
- **A BF16-output variant:** round each GEMM result to BF16, as a deployed
  BF16 stack does.
- **MXFP4 checkpoints:** those found rotate first (FP-Quant, Hadamard) or
  mix MXFP4 and MXFP8 per layer (Intel).  A rotation is a fixed transform of
  both operands before quantizing.
- **The checkpoints on MT-Bench.**
- **vLLM serving:** a `@register_quantization_config` linear method calling
  `kernels.matmul`, `--enforce-eager`.  It needs compute capability 7.5+;
  the TITAN V is 7.0.
- **Dense linear algebra through the designs:** blocked LU, Cholesky and QR,
  with iterative refinement as in HPL-MxP.
  - Metrics: backward error `‖b - Ax‖ / (‖A‖ ‖x‖ + ‖b‖)`, refinement
    iterations to FP64, `‖I - QᵀQ‖`, growth factor.
  - Test matrices at a given condition number (`xLATMS`).
- **Kernel speed:** the compiler's concern (`backend-triton.md`).

## Open items

### How does a frozen run relate to the free one?

At stake: whether frozen KL says anything about deployment beyond the
noise-floor line.  Under the uniform quantization-noise model (Widrow &
Kollár) a quantizer passes a small offset with gain ~1 on average, so
frozen and free should agree on a linear metric (Δ NLL) in expectation;
KL, quadratic, differs by K₀.  FP8's 3 mantissa bits strain the model.
**Provisional:** the paper claims only what frozen defines (error at the
output, quantization decisions fixed to R1).  Reopen to test: frozen and
free Δ NLL on the width sweep, ~50 segments.

### Do the end-to-end residuals hold on another workload?

`ftz` (+4.5 bits), FMA chains (+1.7), narrow `rho_bits` (-1.5) are what
end to end adds over local.  **Answered in part** (2026-10-21): on the `F`
and `rho_bits` axes MT-Bench repeats WikiText-2's residuals design for
design (End to end, "A second workload").  `ftz` and FMA chains, on other
axes, are WikiText-2's alone; reopen with those axes on `mtbench`.

### Do forced and free-running divergence agree in rate?

They differ prompt by prompt at near-ties (Record).  If they agree in rate,
forced stands in for Yuan et al.'s numbers.  If not, the few designs a
paper quotes are run free (`generate.greedy(ref=...)`), and forced serves
the grid.  **Provisional:** forced, headlined by disagreement and KL.
**Settle** by comparing the fraction diverged on ~20 prompts for two
designs (~12 min of free running each).

### NVFP4's per-tensor activation scale: dynamic or calibrated?

Transformer Engine computes it per tensor at runtime; ModelOpt and
`compressed-tensors` calibrate it once and store it.  **Provisional:**
dynamic for RTN, the stored one for a checkpoint.

### The FP32 order of `fp8-block`'s promotion

Separate roundings, an FMA, or another scaling order changes the last bit.
**Provisional:** `y + p * (s_x * s_w)` in separate roundings.  Reopen if
matching a library bit for bit becomes a goal.

### A checkpoint's master

A checkpoint does not say what it was quantized from.  **Provisional:** its
card's `base_model`, else `--baseline`; without one, the unquantized table is
left out.

### Activations quantized from BF16 or FP32?

The model runs in FP32 outside the linear layers, so FP32 activations are
available.  **Provisional:** BF16, as deployed.

## After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n 8
cd examples/mmasim && ../../.venv/bin/python -m pytest tests serve/tests -q
.venv/bin/python -m mypy fpy2
.venv/bin/ruff check examples/mmasim/serve
```

## Sources

- Zheng et al., *An Empirical Study of Qwen3 Quantization*, 2025 --
  https://arxiv.org/abs/2505.02214
- Dutta et al., *Accuracy is Not All You Need*, NeurIPS 2024 --
  https://arxiv.org/abs/2407.09141
- Yuan et al., *Understanding and Mitigating Numerical Sources of
  Nondeterminism in LLM Inference*, NeurIPS 2025 --
  https://arxiv.org/abs/2506.09501
- Williams & Aletras, *On the Impact of Calibration Data in Post-training
  Quantization and Pruning*, ACL 2024 -- https://arxiv.org/abs/2311.09755
- DeepSeek-AI, *DeepSeek-V3 Technical Report* (FP8 accumulation, promotion)
  -- https://arxiv.org/abs/2412.19437
- Rouhani et al., *Microscaling Data Formats for Deep Learning* --
  https://arxiv.org/abs/2310.10537
- llama.cpp perplexity / KL divergence statistics --
  https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
- EleutherAI `lm-evaluation-harness` --
  https://github.com/EleutherAI/lm-evaluation-harness
- `torchao` -- https://github.com/pytorch/ao;
  `compressed-tensors` -- https://github.com/neuralmagic/compressed-tensors
- vLLM quantization methods --
  https://docs.vllm.ai/en/latest/features/quantization/index.html
- Qwen3-0.6B -- https://huggingface.co/Qwen/Qwen3-0.6B;
  Qwen3.5-0.8B -- https://huggingface.co/Qwen/Qwen3.5-0.8B
