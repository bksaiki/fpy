# mmasim serve: an LLM through the emitted BF16 kernels

Implementation plan, one phase per commit; code under `examples/mmasim/serve/`.
Phases 1-6 are done; local metrics on cached activations are the primary
evaluation; Phase 7 waits on a GPU vLLM supports.  Formats beyond BF16
continue in `docs/todos/mmasim-quantized.md`.

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
matmul, `out[i][j] = design(A[i], BT[j], C[i][j])`, bit-exact with the FPy
interpreter, with `m`, `n` and `k` symbolic.  The BF16 designs, each
validated against MMA-Sim (`tests/test_nv.py`, `tests/test_amd.py`):

| design | also models | `k` multiple of |
|---|---|---|
| `nv.ampere.bf16.f32` | Ada | 16 |
| `nv.hopper.bf16.f32` | Blackwell, RTX Blackwell | 32 |
| `amd.cdna2.bf16` | | 4 |
| `amd.cdna2.bf16_1k` | | 4 |
| `amd.cdna3.bf16` | | 8 |

The kernels run at 170-1,500 GFLOP/s on the TITAN V against ~27,000 for
`torch.matmul` in FP16.  The goal: run a real LLM with every linear layer's
matmul computed by these kernels, and measure what each design's arithmetic
does to the model, with methodology the ML community would recognize.

## Methodology

It follows low-precision / quantization evaluation, which asks the same
question: the model's matmuls are computed differently; how far does it move?

- **Baseline: the model in full precision** (R0): BF16 weights upcast to
  FP32, everything in FP32, TF32 off -- Yuan et al.'s reproducibility
  reference ("LayerCast").
- **One op emulated, the rest in high precision**, as emulation libraries
  (`microxcaling`) do: the runs differ only in the linear layers.
- **Capability metrics** (what papers headline): WikiText-2 perplexity on
  2048-token segments (GPTQ convention); zero-shot accuracy with
  `lm-evaluation-harness` on PIQA, ARC-e/c, HellaSwag, WinoGrande, LAMBADA.
- **Distance metrics** (what shows small differences): per-token KL, top-1
  agreement and RMS Δp against R0 (llama.cpp); flips (Dutta et al.); the
  greedy-decode divergence index (Yuan et al.).
- **Not done:** `torch.matmul` in BF16 as the reference (a software path on
  this GPU, no real GPU's behavior), or perplexity alone.

The runs share one model and one input stream and differ in how `nn.Linear`
computes `x @ W.T`:

| run | inputs | dot product | answers |
|---|---|---|---|
| **R0 `fp32`** | FP32 activations, BF16 weights | FP32 | the model as trained |
| **R1 `bf16-exact`** | both rounded to BF16 | FP64 (exact for BF16 products), rounded once | the cost of BF16 inputs alone |
| **R2 a design** (x5) | both rounded to BF16 | the design's kernel, FP32 | the cost of each design's accumulation |

R0 -> R1 is shared by every design; R1 -> R2 is what MMA-Sim models.
Settled choices:

- **Models:** Qwen3-0.6B (dense, `k` in {1024, 2048, 3072}); Qwen3.5-0.8B
  (Phase 6: Gated DeltaNet hybrid).
- **Scope: the linear layers, `lm_head` included.**  Attention's `QK^T` /
  `PV` and the DeltaNet recurrence stay FP32, as FP8 inference keeps them in
  higher precision.
- **The GEMM result stays FP32**, the designs' output format; the next layer
  rounds its input to BF16, as a deployment does.
- **Accumulation order is a knob, not a stance.**  Within a slice of `K` the
  design's instructions chain in order; `split_k` slices, each from `C = 0`,
  are combined in FP32 `linear`ly or as a `tree`.  The default is
  `split_k = 1`; every result states its setting.
- **Engine:** `transformers` with the linear layers' `forward` replaced:
  deterministic, eager, what the harness drives.

## Metrics

`p_t` is R0's next-token distribution at position `t` and `q_t` the run's;
`y_t` the correct next token; logarithms natural; `N` the predicted tokens or
items.

| metric | definition | pooled over | uncertainty | source |
|---|---|---|---|---|
| perplexity | `exp((1/N) Σ_t -log q_t(y_t))` | predicted tokens | standard error of the mean NLL, times PPL | HF perplexity guide; GPTQ |
| KL divergence | `(1/N) Σ_t Σ_v p_t(v) (log p_t(v) - log q_t(v))` | predicted tokens | standard error of the mean | llama.cpp |
| top-1 agreement | `(1/N) Σ_t [argmax p_t = argmax q_t]` | predicted tokens | binomial | llama.cpp |
| RMS Δp | `sqrt((1/N) Σ_t (q_t(y_t) - p_t(y_t))^2)` | predicted tokens | none | llama.cpp |
| `acc` | fraction of items whose highest-log-likelihood choice is correct; LAMBADA: the target word is the greedy continuation | items | harness | lm-evaluation-harness |
| `acc_norm` | as `acc`, log-likelihood divided by the choice's length in characters | items | harness | lm-evaluation-harness |
| flips | items whose per-item `acc` (or `acc_norm`) differs from R0's, over the items both runs have | items | count, fraction | Dutta et al. |
| divergence index | first generated position where the run's greedy token differs from R0's; none if it matches to R0's end | prompts: fraction diverged, mean and median index | none | Yuan et al. |
| normwise relative error | `‖Ŷ - Y‖_F / ‖Y‖_F`, a layer's output matrix, in log2 | every token stacked; a block row stacks its layers | none | Higham |
| componentwise backward error | `\|ŷ - y\| / (\|x\|ᵀ\|w\|)` per element, mean and max, in log2 | elements | none | Oettli-Prager; Higham ch. 7 |
| ULP error | `log2(1 + \|ŷ - y\| / ulp(y))` per element, FP32 ulp, mean and max | elements | none | Herbie / FPBench |
| correct-rounding rate | fraction of elements with `ŷ = fl(y)` | elements | none | |
| bias | mean of `(ŷ - y) / (\|x\|ᵀ\|w\|)`, in u = 2^-24: drift about zero | elements | none | |
| magnitude bias | mean of `sign(y) (ŷ - y) / (\|x\|ᵀ\|w\|)`, in u; negative leans toward zero | elements | none | |

Token-level standard errors treat tokens as independent, as llama.cpp does,
and so understate the uncertainty.  In the per-layer metrics `Y` is the FP64
product of the run's own BF16 inputs (local), or R0's output at that layer
(propagated, normwise only).

## Layout

```
examples/mmasim/serve/
  kernels.py      compile each BF16 design once; prepare / round_input / matmul,
                  linear(x, w) -> FP32; register
  swap.py         replace a model's nn.Linear forwards with a run mode
  perplexity.py   paired WikiText-2 pass: PPL per run, KL / top-1 / RMS dp vs R0
  zeroshot.py     lm-evaluation-harness suite per run, flips vs R0
  decode.py       greedy decode, divergence index vs R0
  chat.py         terminal chat through any run, switchable mid-conversation
  layers.py       per-linear-layer error through the model, on WikiText-2 segments
  local.py        local per-layer metrics on cached activations, per design
  workloads.py    token sequences to capture on: WikiText-2, MT-Bench sessions
  vllm_plugin.py  (Phase 7) the same kernels behind vLLM's linear-method hook
  tests/          one per module but chat.py (workloads in test_local.py)
  results/        (gitignored) run outputs and the MT-Bench cache
```

## Phases

### Phase 1 -- Kernels and the swap

**Done.**  `kernels.py` and `swap.py` (`patch(model)` returns a `Run` whose
`mode` / `split_k` / `combine` select how every `nn.Linear` computes).
Departures: the kernels hold BF16 values in FP32 storage (the backend has no
BF16 storage); each design runs at a fixed tile, as autotuning would retune
at every sequence length; the interpreter check compares NaN as NaN (CDNA2's
kernel gives payload `0x7fffffff`).  One 2048-token forward on the TITAN V:

| run | s | tokens/s |
|---|---|---|
| R0 fp32 | 0.30 | 6,900 |
| R1 bf16-exact | 0.63 | 3,300 |
| `nv.ampere.bf16.f32` | 10.4 | 200 |
| `nv.hopper.bf16.f32` | 8.2 | 250 |
| `amd.cdna2.bf16` / `_1k` | 1.8 | 1,100 |
| `amd.cdna3.bf16` | 17.2 | 120 |

### Phase 2 -- Perplexity and distance on WikiText-2

**Done** at smoke scale; the full 146-segment pass (~1.6 h) was deliberately
not run.  WikiText-2 is `Salesforce/wikitext` on the Hub (299,078 Qwen3
tokens); R0's log-probabilities are held on the host, compared 512 tokens at
a time.  First 8 segments:

| run | PPL | KL vs R0 | top-1 |
|---|---|---|---|
| fp32 | 17.8334 | 0 | 100% |
| bf16-exact | 17.8384 | 4.27e-5 | 99.59% |
| nv.ampere.bf16.f32 | 17.8370 | 4.28e-5 | 99.68% |
| nv.hopper.bf16.f32 | 17.8369 | 4.17e-5 | 99.57% |
| amd.cdna2.bf16 | 17.8400 | 4.11e-5 | 99.63% |
| amd.cdna2.bf16_1k | 17.8352 | 4.17e-5 | 99.68% |
| amd.cdna3.bf16 | 17.8365 | 4.24e-5 | 99.55% |

RMS Δp is 0.16-0.18% for every run but R0.  At `split_k = 4`, `linear` and
`tree`, KL stays ~4e-5 (one segment, CDNA2 and Hopper).  Input rounding
dominates: R1 is as far from R0 as any design.

### Phase 3 -- Zero-shot suite and flips

**Done** at smoke scale; the designs' full suites (hours each) were
deliberately not run.  `lm-eval[hf]`; HellaSwag on a seeded 2,000 of its
10,042 items by default (`samples=`, logged under the true `doc_id`s); a
fixed batch size, 16 (8 for Qwen3.5), so every run sees the same batches;
each run cached as `<out>/<run>.json`.  R0 in full (~11 min) is within one
standard error of Zheng et al.'s FP16 Qwen3-0.6B on every task:

| | PIQA | ARC-e | ARC-c | HellaSwag | WinoGrande | LAMBADA |
|---|---|---|---|---|---|---|
| R0 `acc` | 67.74 ±1.09 | 60.86 ±1.00 | 31.31 ±1.36 | 37.60 ±0.48 | 55.80 ±1.40 | 40.40 ±0.68 |
| R0 `acc_norm` | 67.79 | 55.98 | 34.13 | 47.30 | | |
| Zheng et al. | 67.3 | 60.8 | 31.7 | 37.6 | 56.2 | |

### Phase 4 -- Greedy decode divergence

**Done** at smoke scale.  Yuan et al.'s non-reasoning setup: a seeded 100 of
MATH-500, the chat template with thinking off and Qwen's math instruction, up
to 2,048 new tokens.  Decode is launch-bound at `m = 1` (38 tokens/s for R0,
13-22 for the designs), and most prompts never diverge, so a design costs
about R0's full output: hours per design at 100 prompts; the full run was
dropped.  5 prompts: R0's mean output 736 tokens; bf16-exact diverged on 2
(mean index 416), CDNA2 on 1 (713), Hopper on 1 (607).

### Phase 5 -- Per-layer error

**Done**, in full (4 segments, every run, ~4.5 min): `layers.py` hooks every
linear layer.  The plan's "against R1 and R0" became the local and
propagated metrics (Metrics); cosine similarity was dropped, `1 - cos` being
about half the squared relative error here.  Every layer pooled (log2; -24
is one unit roundoff):

| run | normwise | backward mean | backward max | ULP bits | correctly rounded | bias (u) | magnitude bias (u) | propagated |
|---|---|---|---|---|---|---|---|---|
| bf16-exact | -25.24 | -29.61 | -24.16 | 0.31 | 100.00% | 0.000 | 0.000 | -8.16 |
| nv.ampere.bf16.f32 | -18.53 | -23.16 | -15.44 | 4.31 | 0.83% | +0.439 | -1.702 | -8.10 |
| nv.hopper.bf16.f32 | -18.97 | -23.41 | -16.34 | 4.16 | 0.86% | +0.370 | -1.423 | -8.12 |
| amd.cdna2.bf16 | -21.28 | -25.89 | -18.48 | 2.04 | 10.79% | 0.000 | 0.000 | -8.13 |
| amd.cdna2.bf16_1k | -21.54 | -26.08 | -18.83 | 1.94 | 11.86% | 0.000 | 0.000 | -8.13 |
| amd.cdna3.bf16 | -21.91 | -26.39 | -19.49 | 1.78 | 14.00% | -0.001 | 0.000 | -8.15 |

- bf16-exact is correctly rounded everywhere, a check on the reference.
- Local error is flat with depth and ranks the designs the same on every
  metric; it peaks at `mlp.down_proj` (`k = 3072`) in blocks 2 and 27.
- The NV designs' errors lean toward zero (~95% of Ampere's), consistent
  with truncation, and so drift upward about zero; the AMD designs are
  unbiased.
- The ULP maximum (31-35 bits) is cancellation near `y = 0`: the backward
  error is the robust elementwise metric.
- Propagated error is input rounding's (~2^-8.4), 2^10-2^13 times the local
  error, except in blocks 11-14, where the designs add to it (`k/v_proj`,
  block 12: bf16-exact 9.2e-3, Ampere 2.3e-2).

### Phase 6 -- Qwen3.5-0.8B

**Done** at smoke scale; every script takes `--model`.  `transformers` 5.17
loads it text-only (`Qwen3_5ForCausalLM`, 24 layers: 18 Gated DeltaNet, 6
attention; vocabulary 248,320); without `causal_conv1d` or
`flash-linear-attention` the convolution and delta rule take the PyTorch
path in FP32, which runs on sm_70.  Every `k` (1024, 2048, 3584) suits every
design.  The 2 GB of logits per segment forced: R0's log-probabilities on
the host, column blocks in the per-layer metrics, `--batch-size 8` in the
harness; decoding needed the config-built cache (DeltaNet state) and both
EOS tokens (`<|im_end|>`, `<|endoftext|>`).

Perplexity on segment 0: R0 12.46, KL 2.7-2.9e-5 for the rest.  Per-layer
metrics (4 segments): the Qwen3 ordering and biases (normwise -19.20 Ampere,
-19.47 Hopper, -21.60 CDNA2, -21.80 CDNA2 1k, -22.11 CDNA3).  New: CDNA2's
max backward error is 2^-8.9, from block 0's `in_proj_qkv`, whose weight row
423 holds values ~1e-37; their products are FP32-subnormal and CDNA2's
FTZ-Mul flushes them (`models/amd.py`).  No published zero-shot numbers for
it were found.

### Serving overhead (after Phase 6)

**Done.**  At `m = 1` the torch-side wrapper, not the kernels, set decode
speed (~210 us CPU per call, 19 us for `F.linear`).  Now, bit-exactly:
weights prepared once per layer and passed as-is when already BF16-valued;
an input rounded once for the layers that share it; `C` and the output one
zeroed buffer (each program reads its `C` tile before writing it); the
`tree` combine no longer a self-recursive closure (a reference cycle that
held +0.58 GB per `lm_head` call); `block_m` capped at the power of two
`>= m`; `bf16-exact` in column blocks.  Decode, tokens/s: Ampere
14.6 -> 21.5, Hopper 14.7 -> 21.4, CDNA2 12.9 -> 19.2, CDNA3 11.2 -> 13.3.
Not done: bypassing the launcher's checks (~35 us, duplicating it), and CUDA
graphs (past `transformers`' own ~22 ms per token, but a static cache
changes attention's numerics, so every decode reference would change).

### Local metrics on cached activations: the primary evaluation

A grid search over designs cannot afford the end-to-end evaluations, so
unless they become orders of magnitude cheaper, local per-layer metrics are
*the* evaluation of a design, the end-to-end ones confirming a few.
`local.py` captures every linear layer's input once under `bf16-exact`, as
BF16 on the host (one tensor per distinct input, ~0.4 MB per token for
Qwen3-0.6B); each design then runs only its kernel on the cached inputs, so
every design sees identical inputs.  `kernels.register` adds a design (a
grid point); `--layers` filters by regex.  Qwen3-0.6B, first 2048 WikiText-2
tokens (77 s for the whole run):

| design | normwise | backward mean | backward max | ULP bits | correctly rounded | bias | magnitude bias | s (2048) | s (512) |
|---|---|---|---|---|---|---|---|---|---|
| nv.ampere.bf16.f32 | -18.52 | -23.15 | -15.56 | 4.33 | 0.82% | +0.467 | -1.709 | 12.4 | 3.3 |
| nv.hopper.bf16.f32 | -18.97 | -23.41 | -16.40 | 4.18 | 0.85% | +0.393 | -1.429 | 10.2 | 2.8 |
| amd.cdna2.bf16 | -21.27 | -25.88 | -18.38 | 2.06 | 10.70% | 0.000 | 0.000 | 3.7 | 1.1 |
| amd.cdna2.bf16_1k | -21.52 | -26.08 | -18.82 | 1.96 | 11.76% | 0.000 | 0.000 | 3.7 | 1.1 |
| amd.cdna3.bf16 | -21.91 | -26.38 | -19.47 | 1.79 | 13.88% | -0.001 | 0.000 | 19.1 | 5.0 |

Every mean is within ~0.01 in log2 of `layers.py`'s (each design's own
inputs), so shared inputs cost no fidelity; at 512 tokens the means move by
at most ~0.1, the order unchanged.

**Workloads** (`workloads.py`, `-w`).  WikiText-2 prose is the calibration
convention but narrow for a chat model, and calibration data shifts
quantization results (Williams & Aletras, 2024).  `mtbench` holds MT-Bench's
80 two-turn conversations as a user would: the chat template, thinking off,
each reply R0's greedy decode (at most 512 tokens), generated once (18 min)
and cached.  A conversation's sequence is its last turn as the model
processes it; one prefill gives each layer the inputs incremental decoding
would.  Positions are sampled over all conversations, each row tagged by
`category` and `role` (`user`, `assistant`, `template`), and `--by` splits
the metrics by either.

On Qwen3-0.6B (2048 tokens: 316 user, 1679 assistant, 53 template) the
designs rank as on WikiText-2 in every role and category, each mean within
~0.1 in log2.  The NV drift depends on the workload: Ampere's bias is +0.35 u
on user tokens, +0.82 u on its own replies, -0.51 u on template tokens.
Template tokens are ~0.25 bits worse for every design.  (Measured before the
empty think block was tagged `template`.)

### Phase 7 -- Serving through vLLM

- **What:** `vllm_plugin.py`: a `@register_quantization_config` whose linear
  method prepares each weight once (`kernels.prepare`) and whose `apply`
  calls `kernels.matmul`; `--enforce-eager` (CUDA graphs and `torch.compile`
  would bypass the hook); an OpenAI-compatible endpoint serving Qwen3-0.6B
  through a design, tokens/s reported.  `lm_head` stays vLLM's.
- **Why last, and where:** mainline vLLM needs compute capability 7.5+, the
  TITAN V is 7.0; nothing depends on it.

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n auto
cd examples/mmasim && ../../.venv/bin/python -m pytest tests serve/tests -q
.venv/bin/python -m mypy fpy2
.venv/bin/ruff check examples/mmasim/serve
```

and a results section here, per model: every design's local metrics on
WikiText-2 and MT-Bench, and the end-to-end metrics at the scale they were
run.

## Open items

None: output format, attention, the zero-shot subset, the designs and the
reduction order were settled in review.

## Future work

### Attention's matmuls through the designs

`QK^T` and `PV` run on tensor cores in real inference (~15% of Qwen3-0.6B's
FLOPs at 2048 tokens, more at longer contexts), and `microxcaling` covers
them.  It means hooking the attention function (batched, causal,
grouped-query, softmax between), many small launches per layer; worth it if
the linear layers' design differences compound.

### A BF16-output variant

Round each GEMM's FP32 result to BF16, as a deployed BF16 stack does: a flag
in `kernels.matmul` and `swap.Run`, and variants of R1 and R2.  Realism for a
reviewer who asks; it dilutes the design-to-design signal.

### Dense linear algebra through the designs

The kernels as the GEMM of blocked LU, Cholesky and QR, where most of the
flops are (HPL-MxP does LU in low precision and recovers FP64 by iterative
refinement).  Its metrics come from numerical linear algebra: a solve's
backward error `||b - Ax|| / (||A|| ||x|| + ||b||)`; the iterations
refinement takes to reach FP64, and whether it converges; QR's loss of
orthogonality `||I - QᵀQ||`; LU's growth factor.  Test matrices from the
usual generators (a given condition number, as LAPACK's `xLATMS`).

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
- llama.cpp perplexity / KL divergence statistics --
  https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md
- Hugging Face, *Perplexity of fixed-length models* --
  https://huggingface.co/docs/transformers/perplexity
- EleutherAI `lm-evaluation-harness` Python API --
  https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md
- Rouhani et al., *Microscaling Data Formats for Deep Learning* --
  https://arxiv.org/abs/2310.10537, https://github.com/microsoft/microxcaling
- vLLM out-of-tree quantization methods --
  https://docs.vllm.ai/en/latest/features/quantization/index.html
- Qwen3-0.6B -- https://huggingface.co/Qwen/Qwen3-0.6B;
  Qwen3.5-0.8B -- https://huggingface.co/Qwen/Qwen3.5-0.8B
