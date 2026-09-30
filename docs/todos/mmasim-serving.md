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
- **Done:** Stage 1 of the Roadmap.  **Next:** Stage 2.
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

- **Matmuls:** all 13 designs that take a scheme (5 BF16, 4 FP8, 4
  block-scaled), on Qwen3-0.6B and Qwen3.5-0.8B.  Each gets local metrics
  and perplexity at Stage 1's sizes (50 segments).
  - Zero-shot (paired Δ acc with its interval, and flips) confirms
    capability on a few designs.
- **Pairs the design's own effect on both axes:**
  - local error against the quantized operands' product;
  - end-to-end effect against `<scheme>-exact`.

  Against R0, quantization swamps every design, so it would correlate
  schemes, not designs.
- **Analysis:**
  - Spearman rank correlation with a bootstrap interval; 13 designs x 2
    models is few points.
  - A sensitivity-weighted predictor, `ΔKL ≈ Σ_l a_l ε_l²`, with per-layer
    `a_l` from noise injection, once per model (HIGGS's linearity theorem).
    Also: which local metric predicts best (normwise, backward, bias).
- **Controls:** `<scheme>-exact` against itself is zero on both axes, and a
  `split_k` variant of one design differs only in accumulation order.
- **Needs:** end-to-end noise below the spread between designs, which
  Stage 1's Phase 6 checks.

### Stage 3 -- Exploring hypothetical designs

`K` MMA-Sim-like designs that mix and match characteristics, through the
same harness.
- **Axes the builders in `models/` already expose:**
  - instruction length `L`;
  - fraction bits kept in the fused sum `F` (and `F2`);
  - truncation or rounding of aligned terms;
  - output rounding `rho`;
  - `e_zero`;
  - subnormal flushing;
  - input and accumulator formats;
  - scale group size `G`;
  - accumulation order (`split_k`, `combine`).

  Combinations across vendors (NV's truncating T-FDPA with AMD's FTZ
  multiply, say) need one composable builder from `models/utils.py`.
- **Per design:** compile, check against the interpreter, pick a default
  tile, `kernels.register`.
- **Per-design cost bounds `K`:** `K ≈ budget / (compile + evaluation)`.
  So evaluate at several fidelity levels, by successive halving:
  1. every design, local metrics at 512 tokens, normwise only (~1-5 s);
  2. the top fraction, plus a diverse sample, at 2048 tokens, all metrics,
     both workloads;
  3. a handful, reduced end to end.

  Each level is valid only if it ranks designs as the next one up does,
  which Stage 2 measures between local and end to end.
- **Cheaper levels:**
  - The exact FP64 reference and `|x|ᵀ|w|` depend on the scheme, not the
    design.  Stage 1's Phase 7 shares them across the designs of one pass,
    a block at a time, leaving each design only its kernel.
  - A sample of layers may rank designs as all do, since local error is flat
    with depth.
- **Phase 1 of the plan is a calibration,** as Stage 1's Phase 6 is: the
  compile time of a synthetic design, which may dominate the cheap level;
  each level's cost; and each level's rank agreement with the next.  `K` and
  the halving fractions then follow from the budget.
- **Analysis:** a factorial or Latin-hypercube sample over the axes, and
  log2 error (and end-to-end KL where measured) regressed on the
  characteristics: which characteristics matter, not only a ranking.
- **Risks:**
  - FPy compile time (`backend-triton.md`);
  - slow kernels, which is why end to end runs on a subset;
  - combinations the compiler refuses, which the screen skips.

## Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* belong in this
  document, not in source comments.

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
  by regex.  `-o` writes the JSON.
- `kernels.register` adds a design, one grid point.

## Layout

```
examples/mmasim/serve/
  local.py        capture, then local metrics per design (the primary evaluation)
  perplexity.py   WikiText-2 segments, scored against R0 and R1
  zeroshot.py     lm-evaluation-harness suite, paired Δ acc and flips
  decode.py       R0's greedy decode, then every matmul teacher-forced on it
  chat.py         terminal chat
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
  results/        (gitignored) run outputs, the MT-Bench cache
```

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
perplexity, flips.  This is Stage 2 of the Roadmap.

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
