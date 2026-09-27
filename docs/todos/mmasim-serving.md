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
- **Left:** see Essential gaps and Nice to have.

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

### Runs

The runs share one model and one input stream.  They differ only in how
`nn.Linear` computes `x @ W.T`; everything else runs in FP32 with TF32 off.

| run | operands | product | answers |
|---|---|---|---|
| **R0 `fp32`** | FP32 activations, master weights | FP32 | the model as trained (Yuan et al.'s "LayerCast") |
| **`<scheme>-exact`** | quantized by the scheme | FP64, rounded once to FP32 | the cost of the scheme alone |
| **a design** | quantized by the scheme | the design's kernel, scales applied as the scheme says | the cost of the design |

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
- The quantizers are `torchao`'s:
  - **FP8:** `Float8Tensor`, `PerRow` / `PerBlock`, `s = amax / 448`.
  - **MX:** `MXTensor.to_mx`, `ScaleCalculationMode.RCEIL`.  This is
    NVIDIA's (cuBLAS's) rule: the power of two at or above `amax / max`, so
    no element saturates.
  - **NVFP4:** `NVFP4Tensor.to_nvfp4`, NVIDIA's two-level recipe.  Per
    tensor `g = amax / (448 * 6)`; per 16, `s = rne_e4m3((amax_block / 6) /
    g)`.
- `quant.Quantized.dequantize` is our own: it multiplies out in FP64, where
  `torchao`'s rounds to FP32.

Checkpoints (`checkpoints.py`, `compressed-tensors`):

| checkpoint | weights | activations |
|---|---|---|
| `RedHatAI/Qwen3-0.6B-FP8-dynamic` | `fp8-row`, BF16 scales | dynamic per token |
| `RedHatAI/Qwen3-0.6B-FP8-BLOCK` | `fp8-block`, BF16 scales; no `base_model` on its card, so `--master` | dynamic per 1 x 128 |
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
- One prefill per workload sequence through the scheme's exact run.
- Each linear layer's input is recorded as BF16 on the host, at the sampled
  positions.  There is one tensor per distinct input: `q/k/v_proj` share
  one, as do `gate/up_proj`.

**Evaluate** (`local.evaluate`): each design runs only its kernel, on the
cached `(x, W)`.
- Every design sees identical inputs, and there is no propagation between
  layers.
- Means are within ~0.01 in log2 of `layers.py`, where each design sees its
  own inputs.

**Two references** (parallel tables in every run):
- **quantized:** the quantized operands' exact product, scales included.
  This measures the design's own effect.
- **unquantized:** the BF16 activation times the master weight.  This
  measures quantization and design together.  It is left out for a
  checkpoint with no known master.
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
| propagated (`layers.py` only) | normwise, against R0's output at that layer | every token stacked | |

`-m` selects metrics, and only those are computed.

End-to-end metrics (secondary, smoke scale):
- `p_t` is R0's next-token distribution, `q_t` the run's.
- Paired token-level standard errors treat tokens as independent, as
  llama.cpp does, and so understate the uncertainty.

| metric | definition | source |
|---|---|---|
| perplexity | `exp(mean_t -log q_t(y_t))`, 2048-token segments | GPTQ; HF guide |
| KL divergence | `mean_t KL(p_t ‖ q_t)` | llama.cpp |
| top-1 agreement | `mean_t [argmax p_t = argmax q_t]` | llama.cpp |
| RMS Δp | `sqrt(mean_t (q_t(y_t) - p_t(y_t))^2)` | llama.cpp |
| `acc` / `acc_norm` | `lm-evaluation-harness`: PIQA, ARC-e/c, HellaSwag (seeded 2,000), WinoGrande, LAMBADA | EleutherAI |
| flips | items whose correctness differs from R0's | Dutta et al. |
| divergence index | first greedy token differing from R0's | Yuan et al. |

## Interface

```
cd examples/mmasim
python serve/local.py                                   # bf16, Qwen3-0.6B, every design
python serve/local.py --scheme fp8-row --models Qwen/Qwen3-0.6B RedHatAI/Qwen3-0.6B-FP8-dynamic
python serve/local.py --scheme fp8-block --models RedHatAI/Qwen3-0.6B-FP8-BLOCK --master Qwen/Qwen3-0.6B
python serve/local.py --scheme mxfp4 --models kaitchup/Qwen3-0.6B-NVFP4 --requantize
python serve/local.py --scheme nvfp4 -d nv.blackwell.nvfp4 -w mtbench --by role -m normwise backward
python serve/{perplexity,zeroshot,decode,layers}.py --model Qwen/Qwen3.5-0.8B   # bf16 runs
python serve/chat.py                                    # terminal chat, run switchable mid-conversation
```

Options:
- `--scheme` defaults to `bf16`; `-d` defaults to every design applicable to
  the scheme.
- `--tokens` sets how many positions to sample.  `--layers` filters layers
  by regex.  `-o` writes the JSON.
- `kernels.register` adds a design, one grid point.

## Layout

```
examples/mmasim/serve/
  kernels.py      compile each design once; prepare, matmul, chain, linear; register
  quant.py        schemes, torchao's RTN, Quantized with an exact FP64 dequantize
  checkpoints.py  compressed-tensors checkpoints; weights_for
  swap.py         a model's nn.Linear forwards through a Run (mode, scheme)
  local.py        capture, then local metrics per design (the primary evaluation)
  workloads.py    WikiText-2; MT-Bench sessions
  layers.py       per-layer metrics through the model, propagated included
  perplexity.py   WikiText-2 PPL, KL / top-1 / RMS dp vs R0
  zeroshot.py     lm-evaluation-harness suite, flips vs R0
  decode.py       greedy decode, divergence index vs R0
  chat.py         terminal chat
  tests/          per module (workloads in test_local.py)
  results/        (gitignored) run outputs, the MT-Bench cache
```

## Results

2048 positions per workload, every linear layer pooled; log2 unless marked
u.  "RTN" is the master quantized here.  The checkpoints were run on
WikiText-2 only: on MT-Bench each would need its own conversations, ~18 min
of decoding apiece.

### Qwen3-0.6B: the observable effect

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

### Qwen3-0.6B: the design's own effect

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

| run | PPL | KL vs R0 | top-1 |
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
- **Propagated error** (`layers.py`) is input rounding's, ~2^-8.4, which is
  2^10 to 2^13 times the local error.  The exception is blocks 11-14, where
  the designs add to it.
- **Qwen3.5-0.8B:** PPL 12.46 on segment 0; KL 2.7-2.9e-5 for every run.

## Record

Condensed; what the code does not say.

- **Serving overhead.**
  - At `m = 1` the torch-side wrapper set decode speed: ~210 us CPU per
    call, against 19 us for `F.linear`.
  - Fixes: weights prepared once per layer; an input quantized once for the
    layers sharing it; `C` and the output one buffer; `block_m` capped at
    the power of two `>= m`; the exact run in row and column blocks.
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
- **Quantizers are torchao's.** Our first MX and FP8 quantizers matched
  torchao bit for bit.  On NVFP4 they differed in the order of FP32
  operations, so torchao's were taken.
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
perplexity, flips.  This depends on tractable end-to-end evaluations (Nice
to have).

### Uncertainty on local metrics

Pooled means carry no intervals.  A bootstrap over sequences (WikiText-2
segments, MT-Bench conversations) would separate real design differences
of ~0.05-0.1 in log2 from sampling noise.

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

- **Tractable end-to-end evaluations** (next branch): paired statistics on
  sampled segments and items; a teacher-forced divergence index in place of
  free decoding; censoring at a token budget.
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
card's `base_model`, else `--master`; without one, the unquantized table is
left out.

### Activations quantized from BF16 or FP32?

The model runs in FP32 outside the linear layers, so FP32 activations are
available.  **Provisional:** BF16, as deployed.

## After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n auto
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
