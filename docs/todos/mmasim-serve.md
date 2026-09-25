# mmasim serve: an LLM through the emitted BF16 kernels

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.  Code lives under `examples/mmasim/serve/`.

## Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* -- what was tried, what
  a phase decided, why an ordering was chosen -- belong in this document, not in
  source comments.

## Context

`examples/mmasim/compile_triton.py` compiles each MMA-Sim design into a Triton
matmul, `out[i][j] = design(A[i], BT[j], C[i][j])`, bit-exact with the FPy
interpreter (tracker: 50/62 designs compile and agree).  `compile_matmul`
leaves `m`, `n` and `k` symbolic, so one kernel per design runs at any shape
whose `k` is a multiple of the design's own length.  The BF16-input designs,
each validated against MMA-Sim by `tests/test_nv.py` / `tests/test_amd.py`:

| design | also models | `k` multiple of |
|---|---|---|
| `nv.ampere.bf16.f32` | Ada | 16 |
| `nv.hopper.bf16.f32` | Blackwell, RTX Blackwell | 32 |
| `amd.cdna2.bf16` | | 4 |
| `amd.cdna2.bf16_1k` | | 4 |
| `amd.cdna3.bf16` | | 8 |

All are BF16 x BF16 -> FP32; `DESIGNS` lists a function once where several
architectures build the same one.  (`amd.cdna1.bf16` is refused by the Triton
backend.)

The kernels run at roughly 170-1,500 GFLOP/s on the TITAN V
(`bench/speed.py`), against ~27,000 for `torch.matmul` in FP16: slow, but
enough for evaluation at the 0.6B scale.

The goal is to run a real LLM with every linear layer's matmul computed by
these kernels, and to measure what each design's arithmetic does to the model
-- with methodology the ML community would recognize.

## What is standard (and what this plan follows)

The closest established practice is **low-precision / quantization
evaluation**, which asks the same question: the model's matmuls are computed
differently; how far does the model move?

- **Baseline: the model in full precision.**  BF16 checkpoint weights upcast
  to FP32 (exact), everything computed in FP32.  Quantization papers report
  every delta against the full-precision model, and the numerical
  nondeterminism study of Yuan et al. (NeurIPS 2025) uses FP32 as the
  reproducibility reference; its "LayerCast" -- 16-bit weights, all compute in
  FP32 -- is exactly this run.  TF32 must be off
  (`torch.backends.cuda.matmul.allow_tf32 = False`).
- **Emulating one op, the rest in high precision.**  Emulation libraries for
  new number formats (e.g. Microsoft's `microxcaling` for MX) replace the
  inputs/arithmetic of `Linear`/`MatMul` and leave everything else in higher
  precision.  This plan does the same, so the only difference between two runs
  is the linear layers' arithmetic.
- **Capability metrics** (what papers headline):
  - WikiText-2 perplexity, `wikitext-2-raw-v1` test split, 2048-token
    segments -- the GPTQ / QuaRot / SpinQuant convention.
  - Zero-shot accuracy with EleutherAI's `lm-evaluation-harness` on the usual
    suite: PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, LAMBADA.
- **Distance metrics** (what shows small differences; capability metrics
  barely move between FP32-accumulating designs):
  - Per-token KL divergence from the baseline's next-token distribution, top-1
    agreement ("same top p") and RMS probability difference -- llama.cpp's
    `perplexity --kl-divergence` statistics.
  - **Flips**: zero-shot answers that change from the baseline's, right or
    wrong -- Dutta et al., "Accuracy is Not All You Need" (NeurIPS 2024), who
    show aggregate accuracy hides them and KL correlates with them.
  - **Greedy-decode divergence index**: the first token position where a
    run's greedy output departs from the baseline's -- Yuan et al.'s
    `Div_Index`.
- **Uncertainty**: KL / top-1 as mean ± standard error over tokens; accuracy
  with the harness's standard error; flips as a count and percentage.

What is *not* standard, and why this plan does not do it: comparing against
`torch.matmul` in BF16 as the reference (on the TITAN V, which has no BF16
tensor cores, that is a software path, not any real GPU's behavior), and
reporting perplexity alone.

## Runs

Every run shares one model, one input stream and FP32 everywhere outside the
linear layers; they differ only in how `nn.Linear` computes `x @ W.T`.

| run | linear inputs | dot product | answers |
|---|---|---|---|
| **R0 fp32** (baseline) | FP32 activations, BF16 weights upcast | FP32 | the model as trained, full precision |
| **R1 bf16-exact** | activations rounded to BF16 | exact (FP64: BF16 products are exact in it) | the cost of BF16 *inputs* alone |
| **R2 design** (x5) | activations rounded to BF16 | the design's kernel, FP32 result | the cost of each design's *accumulation* |

R0 -> R1 is the input-rounding cost, the same for every design; R1 -> R2 is
the hardware-specific part MMA-Sim models, and the point of the study.  Every
metric is reported against R0; R1 -> R2 deltas are the ones that separate
designs.

Settled choices:

- **Models.** Qwen3-0.6B first: a plain dense transformer, BF16-only
  weights, 28 layers, hidden 1024, FFN 3072, GQA 16/8 heads of 128, vocab
  151,936, tied embeddings.  Every projection and `lm_head` is an `nn.Linear`
  in Hugging Face `transformers`, and every `k` (1024, 2048, 3072) is a
  multiple of 32, as are its quarters for `split_k = 4`.  Qwen3.5-0.8B second (Phase 6): a 3:1 hybrid of Gated
  DeltaNet and full attention with a vision tower, and some F32 tensors.
- **Scope of the swap: the linear layers, `lm_head` included.**  Attention's
  `QK^T` / `PV` and (for Qwen3.5) the DeltaNet recurrence stay torch FP32 in
  every run.  This is the established scope for low-precision linear layers
  (FP8 inference quantizes the linear layers and keeps attention in higher
  precision), covers the large majority of FLOPs at 2048 tokens, and is a
  module swap.  Attention is future work.
- **The GEMM result stays FP32** (the designs' output format), not rounded to
  BF16.  A BF16 store adds the same error to every design, larger than the
  differences between them, and the next layer already rounds its input to
  BF16 as a deployment does.  A BF16-output variant is future work.
- **Accumulation order is a knob, not a stance.**  How a GEMM orders its
  accumulation across `K` is an active research question, and libraries
  differ (in-order within a tile, split-K, stream-K), so the plan does not
  pick a "right" one.  Within a slice of `K` the design's chain of
  instructions accumulates in order, as MMA-Sim models an instruction
  sequence; across slices, `kernels.linear` takes `split_k` contiguous slices
  (each a multiple of the design's length), each through the kernel with
  `C = 0` to an FP32 partial, combined by FP32 adds `linear`ly (left to right)
  or as a `tree` (pairwise).  The default, `split_k = 1`, is plain in-order
  accumulation: a common, reasonable modeling choice, used for the headline
  results.  Every result states its setting.
- **Engine.** Hugging Face `transformers` with module replacement for
  evaluation: deterministic, eager, and what `lm-evaluation-harness` drives.
  vLLM is for serving (Phase 7) and does not run on this GPU.

## Metrics

Each metric, as the code computes it.  Notation: `p_t` is R0's next-token
distribution at position `t` and `q_t` the run's, both over the vocabulary;
`y_t` is the correct next token; logarithms are natural (nats); `N` is the
number of predicted tokens (2047 per segment) or items.

| metric | definition | pooled over | uncertainty | source |
|---|---|---|---|---|
| perplexity | `exp((1/N) Σ_t -log q_t(y_t))` | predicted tokens | standard error of the mean NLL, times PPL (delta method) | HF perplexity guide; GPTQ |
| KL divergence | `(1/N) Σ_t Σ_v p_t(v) (log p_t(v) - log q_t(v))`, i.e. KL(R0 ‖ run) | predicted tokens | standard error of the mean | llama.cpp |
| top-1 agreement | `(1/N) Σ_t [argmax p_t = argmax q_t]` | predicted tokens | binomial, `sqrt(a (1 - a) / N)` | llama.cpp ("same top p") |
| RMS Δp | `sqrt((1/N) Σ_t (q_t(y_t) - p_t(y_t))^2)` | predicted tokens | none | llama.cpp |
| `acc` | fraction of items whose highest-log-likelihood choice is correct; LAMBADA: the target word is the greedy continuation | items | sample standard deviation / `sqrt(N)` (harness) | lm-evaluation-harness |
| `acc_norm` | as `acc`, each choice's log-likelihood divided by its length in characters | items | as `acc` | lm-evaluation-harness |
| flips | items whose per-item `acc` (or `acc_norm`) differs from R0's, over the items both runs have | items | none (count and fraction) | Dutta et al. |
| divergence index | first generated position (0-based) where the run's greedy token differs from R0's; none if it matches R0 to R0's end | prompts: fraction diverged, mean and median index of those that do | none | Yuan et al. (`Div_Index`, `Div_Percent`) |
| normwise relative error | `‖Ŷ - Y‖_F / ‖Y‖_F` for a layer's output matrix (tokens x features), in log2 | every token and segment stacked into one matrix; a block row stacks its seven layers | none | Higham, *Accuracy and Stability of Numerical Algorithms* |
| componentwise backward error | `\|ŷ - y\| / (\|x\|ᵀ\|w\|)` per output element, mean and max, in log2 | elements, as above | none | Oettli-Prager; Higham, ch. 7 |
| ULP error | `log2(1 + \|ŷ - y\| / ulp(y))` per element, `ulp` in FP32 ("bits of error"), mean and max | elements | none | Herbie / FPBench |
| correct-rounding rate | fraction of elements with `ŷ = fl(y)`, `y` rounded to nearest FP32 | elements | none | |
| bias | mean of `(ŷ - y) / (\|x\|ᵀ\|w\|)`, in units of u = 2^-24: drift about zero | elements | none | |
| magnitude bias | mean of `sign(y) (ŷ - y) / (\|x\|ᵀ\|w\|)`, in units of u; negative leans toward zero | elements | none | |

The token-level standard errors treat tokens as independent, as llama.cpp
does; tokens in one segment are correlated, so they understate the
uncertainty.  In the per-layer metrics `Ŷ` is the run's output and `Y` the
exact product, in FP64, of its own BF16-rounded inputs `x` and weights `w`
(local), or R0's output at that layer (propagated, normwise only).

## Cost

Qwen3-0.6B is about 1.2 GFLOP per token (twice its ~0.6B matmul parameters,
`lm_head` included).  Measured in Phase 1, one 2048-token forward on the
TITAN V:

| run | s / window | tokens/s |
|---|---|---|
| R0 fp32 | 0.30 | 6,900 |
| R1 bf16-exact | 0.63 | 3,300 |
| `nv.ampere.bf16.f32` | 10.4 | 200 |
| `nv.hopper.bf16.f32` | 8.2 | 250 |
| `amd.cdna2.bf16` / `bf16_1k` | 1.8 | 1,100 |
| `amd.cdna3.bf16` | 17.2 | 120 |

So per design, from ~25 min (CDNA3: 40 min; CDNA2: 5 min) for WikiText-2's
~0.3M tokens to ~2-4 h for the five smaller zero-shot tasks' ~1.6M, plus the
HellaSwag subset.  Greedy decode runs at `m = 1`, the kernels' worst shape.

## Layout

```
examples/mmasim/serve/
  kernels.py      compile each BF16 design once; linear(x, w) -> FP32
  swap.py         replace a model's nn.Linear forwards with a run mode
  perplexity.py   paired WikiText-2 pass: PPL per run, KL / top-1 / RMS dp vs R0
  zeroshot.py     lm-evaluation-harness suite per run, flips vs R0
  decode.py       greedy decode, divergence index vs R0
  layers.py       per-linear-layer error on a calibration batch
  vllm_plugin.py  (Phase 7) the same kernels behind vLLM's linear-method hook
  tests/          one per module above
```

The paired pass runs R0 and the candidate on the same window and accumulates
the distance metrics on the fly: storing full-vocabulary logits for 0.3M
tokens would take ~180 GB.

## Phases

### Phase 1 -- Kernels and the swap

**Done.**  `serve/kernels.py` (`linear`, `compiled`, `BF16_DESIGNS`) and
`serve/swap.py` (`patch(model)` returns a `Run` whose `mode` / `split_k` /
`combine` select how every `nn.Linear` computes; one model, switched between
runs, so the paired pass needs one copy in memory).  Where it departed:

- The kernels hold BF16 values in FP32 storage (the Triton backend has no
  BF16 storage type), so `linear` rounds to BF16 and hands over the kernel's
  own dtype, read from `KernelSource.dtypes`.
- Each design runs at a fixed tile (`BF16_DESIGNS`), the fastest at a
  2048-token FFN shape: autotuning keys on the sizes and would retune at
  every new sequence length.
- The interpreter check compares NaN as NaN: CDNA2's kernel produces a
  different NaN payload (`0x7fffffff`) than the interpreter's canonical one,
  and a payload is not part of IEEE 754's result.
- `bf16-exact` computes its FP64 product in blocks of 256 rows: `lm_head` at
  2048 tokens is 2.5 GB in FP64 at once.

A sanity pass on one 2048-token window (README text, not WikiText): every
design's per-token KL from R0 is ~2e-5, the same order as R1's 2.1e-5, with
99.7-99.9% top-1 agreement -- input rounding dominates, as expected; the
comparison of record is Phase 2's.

- **What:** `kernels.py`: `compile_matmul` each BF16 design once, cached; a
  `linear(x, w, design)` that rounds `x` to BF16 (RNE), passes `W` `[n, k]` as
  `BT` directly (it is `nn.Linear`'s layout), `C = 0`, and returns the FP32
  output; refuses a `k` not a multiple of the design's length.  Its
  `split_k` / `combine` knob splits `K` into contiguous slices, one launch
  each, and sums the partials in FP32, `linear` or `tree`.  `swap.py`:
  `patch(model)` with `mode` in `fp32 | bf16-exact | <design>`,
  replacing each `nn.Linear.forward` (bias added in FP32 after the matmul,
  which Qwen3 has none of).  Measure tokens/s for R2 at a 2048-token window
  and replace the cost table's estimates.
- **Why first:** everything else drives a model through this.
- **Tests** (`tests/test_kernels.py`, `tests/test_swap.py`): `linear` agrees
  bit-for-bit with the FPy design on small random and hard-case matrices;
  `fp32` mode reproduces the unswapped FP32 model's logits exactly;
  `bf16-exact` agrees with an FP64 reference; a design mode's logits are close
  to `bf16-exact` (a sanity bound, not a claim); `split_k = S` equals the
  kernel run slice by slice with the partials summed in the stated order, and
  `linear` and `tree` differ where the partials' magnitudes make them.
  `cd examples/mmasim && ../../.venv/bin/python -m pytest serve/tests -q`

### Phase 2 -- Perplexity and distance on WikiText-2

**Done** at smoke scale; the full 146-segment pass (~1.6 h, each `split_k`
pass as long) was deliberately not run.  First 8 segments (16,376 tokens):

| run | PPL | KL vs R0 | top-1 |
|---|---|---|---|
| fp32 | 17.8334 | 0 | 100% |
| bf16-exact | 17.8384 | 4.27e-5 | 99.59% |
| nv.ampere.bf16.f32 | 17.8370 | 4.28e-5 | 99.68% |
| nv.hopper.bf16.f32 | 17.8369 | 4.17e-5 | 99.57% |
| amd.cdna2.bf16 | 17.8400 | 4.11e-5 | 99.63% |
| amd.cdna2.bf16_1k | 17.8352 | 4.17e-5 | 99.68% |
| amd.cdna3.bf16 | 17.8365 | 4.24e-5 | 99.55% |

RMS Δp is 0.16-0.18% for every run but R0.  `split_k = 4`, `linear` and
`tree`, run on one segment for CDNA2 and Hopper: KL stays ~4e-5, with no
conclusion drawn about order.  `split_k` first ran out of memory at
`lm_head`, whose 1.2 GB partials were all held at once.  `kernels.linear`
now sums the partials as it makes them and works in row blocks of 2^26
output elements, so memory does not grow with `split_k`.

`serve/perplexity.py`: WikiText-2 is now
`Salesforce/wikitext` on the Hub (same `wikitext-2-raw-v1` data); its test
split is 299,078 Qwen3 tokens, 146 whole 2048-token segments.  Perplexity is
the mean NLL over each segment's 2047 predicted tokens; Δp is the change in
the correct token's probability, as llama.cpp defines it.  Distributions are
compared 512 tokens at a time, one segment's full-vocabulary log-probabilities
being 1.2 GB.  Test: `tests/test_perplexity.py` (R0 against itself is zero
distance and its PPL is the model's cross-entropy).

- **What:** `perplexity.py`: `wikitext-2-raw-v1` test, joined and tokenized as
  the Hugging Face perplexity guide does, split into 2048-token segments
  (GPTQ convention); per run: perplexity, and against R0 per token: KL(R0 ||
  run), top-1 agreement, RMS Δp, each with standard error.  One table: R0, R1,
  each design.  Then a sensitivity check on accumulation order: each design at
  `split_k = 4`, `linear` and `tree`, KL and top-1 against R0 only, reported
  alongside the default without drawing conclusions about which order is
  right.
- **Tests:** a two-segment smoke run where R0 against itself gives KL = 0 and
  100% agreement; PPL of R0 matches the unswapped model's.

### Phase 3 -- Zero-shot suite and flips

**Done** at smoke scale; the designs' full suites (hours each) were
deliberately not run.  `serve/zeroshot.py` needs `lm-eval[hf]` (0.4.13 here,
for `accelerate`); `simple_evaluate(samples=...)` takes the HellaSwag subset
(a seeded random 2,000 of 10,042), logged under the true `doc_id`s.  Batch
size is fixed (16: at 32 the harness's `log_softmax` over the vocabulary
runs out of memory) so every run sees the same batches.  Flips are counted
for each per-item metric the task logs (`acc`, `acc_norm`), over the items
both runs have.  Each run's results are cached as `<out>/<run>.json` with its
settings, so the suite can run a design at a time.

R0, every task in full (~11 min), against Zheng et al.'s FP16 Qwen3-0.6B
(`acc`): within one standard error on each task.

| | PIQA | ARC-e | ARC-c | HellaSwag | WinoGrande | LAMBADA |
|---|---|---|---|---|---|---|
| R0 `acc` | 67.74 ±1.09 | 60.86 ±1.00 | 31.31 ±1.36 | 37.60 ±0.48 | 55.80 ±1.40 | 40.40 ±0.68 |
| R0 `acc_norm` | 67.79 | 55.98 | 34.13 | 47.30 | | |
| Zheng et al. | 67.3 | 60.8 | 31.7 | 37.6 | 56.2 | |

Also run: `--limit 10` for R0, R1 and `amd.cdna2.bf16` (~1 min, no flips at
that size); a 40-item HellaSwag subset under R0, 0 flips against the full
run's same items.  Test: `tests/test_zeroshot.py` (`flips`).

- **What:** `zeroshot.py`: `lm_eval.simple_evaluate(model=HFLM(pretrained=m),
  tasks=[piqa, arc_easy, arc_challenge, hellaswag, winogrande,
  lambada_openai], log_samples=True)` per run; accuracy (± stderr) and flips
  against R0 from the logged samples.  R0 and R1 run every task in full --
  R0's accuracy is checked against Qwen3-0.6B's published numbers, the sign
  the harness is set up right.  Each design runs the five smaller tasks in
  full and HellaSwag on a fixed subset (e.g. 2,000 items, the same in every
  run, R0 and R1 included for its flips), reported as such; HellaSwag is most
  of the cost.  If Phase 1 measures the kernels well above the estimate, run
  HellaSwag in full too.
- **Tests:** a `--limit 10` smoke run; flips of R0 against itself are 0.

### Phase 4 -- Greedy decode divergence

**Done** at smoke scale.  `serve/decode.py` follows Yuan et al.'s
non-reasoning setup: a seeded random 100 of MATH-500 (their benchmark best
suited to a 0.6B model; long step-by-step outputs), Qwen3's chat template
with thinking off (the model card warns against greedy decoding only when
thinking) and its math instruction, up to 2,048 new tokens.  Reported: the
fraction diverged (Yuan's `Div_Percent`) and the mean and median index over
those that do.  Each run's tokens are cached as `<out>/<run>.json`.

Decode is launch-bound at `m = 1`: 38 tokens/s for R0, 11-15 for every
design.  A run stops at its first departure from R0, but most prompts never
depart, so a design costs about R0's full length: ~1.6 h per design at 100
prompts (R0's mean output ~740 tokens), ~10 h for every run.

Smoke run, 5 prompts (~10 min): R0's mean length 736 tokens (one at the
limit); diverged: bf16-exact 2 (indices 416 mean), `amd.cdna2.bf16` 1 (713),
`nv.hopper.bf16.f32` 1 (607).  Test: `tests/test_decode.py` (R0 against
itself never diverges; a reference altered at position 3 stops decoding
there and reports 3).

- **What:** `decode.py`: a fixed prompt set, greedy decode N tokens per run,
  the divergence index against R0 (first differing position; "never" counted
  apart) and the fraction of prompts that diverge.
- **Tests:** R0 against itself never diverges.

### Phase 5 -- Per-layer error

**Done**, in full (4 segments, every run, every metric, ~4.5 min).
`serve/layers.py` hooks every linear layer and compares its FP32 output
elementwise with a reference.  The plan's "against R1 and R0" became two
kinds of metric (defined under Metrics):

- *local*, against the exact product `Y` of the same BF16-rounded inputs, in
  FP64 (the layer's own error): normwise relative error, componentwise
  backward error (mean and max), ULP error (mean and max), correct-rounding
  rate, bias (drift about zero) and magnitude bias (toward or away from
  zero);
- *propagated*, against R0's output at the same layer (what the model has
  gathered by then): normwise relative error.

`-m` selects metrics, and only their work is done: the backward error and
biases add a second FP64 product (`|x|ᵀ|w|`), and only `propagated` needs the
R0 pass.  Cosine similarity was dropped: at these magnitudes `1 - cos` is
about half the squared relative error, so it adds nothing.  `bf16-exact` now
fills a preallocated output instead of concatenating its row blocks, which
held `lm_head`'s 1.2 GB output twice and ran out of memory here.

Every layer pooled (errors as log2; u = 2^-24, so -24 is one unit roundoff):

| run | normwise | backward mean | backward max | ULP bits mean | correctly rounded | bias (u) | magnitude bias (u) | propagated |
|---|---|---|---|---|---|---|---|---|
| bf16-exact | -25.24 | -29.61 | -24.16 | 0.31 | 100.00% | 0.000 | 0.000 | -8.16 |
| nv.ampere.bf16.f32 | -18.53 | -23.16 | -15.44 | 4.31 | 0.83% | +0.439 | -1.702 | -8.10 |
| nv.hopper.bf16.f32 | -18.97 | -23.41 | -16.34 | 4.16 | 0.86% | +0.370 | -1.423 | -8.12 |
| amd.cdna2.bf16 | -21.28 | -25.89 | -18.48 | 2.04 | 10.79% | 0.000 | 0.000 | -8.13 |
| amd.cdna2.bf16_1k | -21.54 | -26.08 | -18.83 | 1.94 | 11.86% | 0.000 | 0.000 | -8.13 |
| amd.cdna3.bf16 | -21.91 | -26.39 | -19.49 | 1.78 | 14.00% | -0.001 | 0.000 | -8.15 |

- **bf16-exact** is correctly rounded everywhere (max 0.585 bits = half an
  ulp), as it must be: a check on the reference.
- **Local** error is flat with depth and orders the designs the same way on
  every metric: Ampere, Hopper, CDNA2, CDNA2 1k, CDNA3.  The NV designs'
  mean backward error is ~1.5-1.8 u, the AMD designs' ~0.2-0.3 u.  It peaks
  at `mlp.down_proj` (the largest `k`, 3072) in blocks 2 and 27, most for
  the NV designs.
- **Magnitude bias** separates the vendors: the NV designs' errors lean
  toward zero (Ampere's -1.70 u against a mean |error| of 1.79 u, so ~95%
  of it), consistent with truncation; the AMD designs' are unbiased.
- **Bias** about zero follows from it: the NV designs drift upward (Ampere
  +0.44 u), varying by block from -0.14 to +1.25 u: with errors toward zero,
  upward drift means more of the error falls on negative outputs than on
  positive ones.  The AMD designs do not drift.
- **ULP max** is 31-35 bits for every design: cancellation, where the exact
  `y` is near zero and its ulp tiny.  The backward error is the robust
  elementwise metric here; ULP error is kept as the conventional one.
- **Propagated** error is ~2^-9 at block 0 and ~2^-8.4 through most of the
  model, set by input rounding (`bf16-exact` alike): 2^10-2^13 times the
  local error.  Blocks 11-14 are the exception: there, the designs' errors
  add to it, largest in `k_proj`/`v_proj` (block 12: `bf16-exact` 9.2e-3,
  CDNA3 1.3e-2, Ampere 2.3e-2), and it settles back after block 15.

Test: `tests/test_layers.py` (`bf16-exact` is correctly rounded with its
input rounding as propagated error, a design errs in every block; only the
selected metrics are computed).

### Phase 6 -- Qwen3.5-0.8B

**Done** at smoke scale.  Every script takes `--model` (default
`swap.MODEL`, Qwen3-0.6B) and loads through `swap.load`.  `transformers`
5.17 loads `Qwen/Qwen3.5-0.8B` as `Qwen3_5ForCausalLM`, text only, with no
missing weights: 752M parameters, 24 layers (18 Gated DeltaNet, 6 full
attention), hidden 1024, FFN 3584, vocabulary 248,320, tied embeddings.
Without `causal_conv1d`, `flash-linear-attention` or `kernels` installed,
the short convolution and the delta rule take `transformers`' PyTorch
reference path, in FP32 (it warns, and it works on sm_70).  The 187 swapped
linear layers have `k` in {1024, 2048, 3584}, all multiples of 32 and at
`split_k = 4` too.

What the larger vocabulary (2 GB of logits per 2048-token segment) needed:

- `perplexity.py` holds R0's log-probabilities on the host.  A leak also
  surfaced: the comparison loop's last block, a view, kept each run's
  log-probabilities alive into the next run (1.2 GB on Qwen3 too); the
  comparison is now `Totals.compare`, so its locals die with it.
- `layers.py` works in blocks of output columns as well as rows:
  `lm_head`'s FP64 weights and their absolute values were 2 GB each.
- `zeroshot.py` needs `--batch-size 8` (16 runs out of memory in the
  harness's `log_softmax`).
- `decode.py` builds its cache from the model's config, which gives the
  DeltaNet layers their recurrent-state cache; and it stops on the
  tokenizer's EOS (`<|im_end|>`, the chat template's end of turn) as well as
  the model's (`<|endoftext|>`), Qwen3.5 having no `generation_config.json`.

Smoke runs, one each:

- Perplexity, 1 segment, every run: R0 12.4586; KL 2.7-2.9e-5 and top-1
  99.7-99.9% for the rest (Qwen3 at the same segment: 14.18, ~4e-5).  Per
  segment: Ampere 17 s, Hopper 15 s, CDNA2 4 s, CDNA3 28 s.
- Per-layer error, 1 segment, every run and metric (~1.8 min): the same
  ordering, magnitude bias (NV -1.5 to -1.8 u, AMD ~0) and upward NV drift
  (+0.7 to +0.85 u) as on Qwen3.  One new effect: both CDNA2 designs' max
  backward error is 2^-8.9, all in block 0's `in_proj_qkv`, whose weight
  row 423 holds values ~1e-37.  Their FP32 products are subnormal and
  CDNA2's FTZ-Mul flushes them (`models/amd.py`), so outputs of ~1e-36 are
  off by 2^-9 relative to `|x|ᵀ|w|`; CDNA3 does not flush.  Normwise error
  is unaffected.
- Zero-shot, `--limit 10`, R0 and `amd.cdna2.bf16` at batch 8 (~1 min): no
  flips.
- Decode, 2 prompts, R0 and `amd.cdna2.bf16` (~4 min): R0's outputs are
  1002 and 2048 (the limit) tokens and end at `<|im_end|>`; CDNA2 diverges
  on one, at 639.

No published zero-shot numbers for Qwen3.5-0.8B were found to check R0
against.

### Phase 7 -- Serving through vLLM

- **What:** `vllm_plugin.py`: a `@register_quantization_config` whose linear
  method's `apply` calls `kernels.linear` (weights `[out, in]`, as vLLM stores
  them); `--enforce-eager` (CUDA graphs and `torch.compile` would bypass the
  Python hook); an OpenAI-compatible endpoint serving Qwen3-0.6B through a
  design; tokens/s reported.  `lm_head` is not a linear method in vLLM, so it
  stays vLLM's.
- **Why last, and where:** mainline vLLM requires compute capability 7.5+,
  and the TITAN V is 7.0.  It runs on a newer GPU (or a Volta fork); nothing
  earlier depends on it.  It demonstrates serving; the numbers of record come
  from Phases 2-5.

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n auto
cd examples/mmasim && ../../.venv/bin/python -m pytest tests serve/tests -q
.venv/bin/python -m mypy fpy2
.venv/bin/ruff check fpy2 tests examples/mmasim/serve
```

and a results section here: one table per model (R0, R1, each design) with
PPL, KL, top-1, flips, accuracy and divergence index.

## Open items

None: output format, attention, the zero-shot subset, the designs and the
reduction order were settled in review (above and under Future work); the
Hopper/Blackwell BF16 designs landed in #323.

## Future work

### Attention's matmuls through the designs

Route `QK^T` and `PV` through the design kernels too: real inference runs
them on tensor cores (FlashAttention in BF16), they are roughly 15% of
Qwen3-0.6B's FLOPs at 2048 tokens and more at longer contexts, and emulation
libraries such as `microxcaling` cover `MatMul` / `BMM` as well as `Linear`.
The work is hooking the attention function rather than a module: the matmuls
are batched, causal and grouped-query (16 query heads over 8 KV heads), with
masking and softmax between them, so it is many small launches per layer.
The question it answers is whether the linear layers' differences between
designs compound once attention shares the design's arithmetic; worth doing if
Phase 2-5 show those differences at all.

### A BF16-output variant

Round each GEMM's FP32 result to BF16 before the next op, as a deployed BF16
stack does: a flag in `kernels.linear`, and variants of R1 and R2.  For
deployment realism if a reviewer asks; it dilutes the design-to-design signal
rather than sharpening it.

## Sources

- Zheng et al., *An Empirical Study of Qwen3 Quantization*, 2025 --
  https://arxiv.org/abs/2505.02214
- Dutta et al., *Accuracy is Not All You Need*, NeurIPS 2024 --
  https://arxiv.org/abs/2407.09141
- Yuan et al., *Understanding and Mitigating Numerical Sources of
  Nondeterminism in LLM Inference*, NeurIPS 2025 --
  https://arxiv.org/abs/2506.09501
- llama.cpp perplexity / KL divergence statistics --
  https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
- Hugging Face, *Perplexity of fixed-length models* --
  https://huggingface.co/docs/transformers/perplexity
- EleutherAI `lm-evaluation-harness` Python API --
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md
- Rouhani et al., *Microscaling Data Formats for Deep Learning* (emulation
  library `microxcaling`) -- https://arxiv.org/abs/2310.10537,
  https://github.com/microsoft/microxcaling
- vLLM out-of-tree quantization methods --
  https://docs.vllm.ai/en/latest/features/quantization/index.html
- Qwen3-0.6B config -- https://huggingface.co/Qwen/Qwen3-0.6B;
  Qwen3.5-0.8B -- https://huggingface.co/Qwen/Qwen3.5-0.8B
