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

- **What:** `decode.py`: a fixed prompt set, greedy decode N tokens per run,
  the divergence index against R0 (first differing position; "never" counted
  apart) and the fraction of prompts that diverge.
- **Tests:** R0 against itself never diverges.

### Phase 5 -- Per-layer error

- **What:** `layers.py`: on a calibration batch (a few WikiText-2 segments),
  each linear layer's output relative error and cosine similarity against R1
  and R0, by depth: where accumulation error enters, and whether it grows.
- **Tests:** R1 against R1 is zero error.

### Phase 6 -- Qwen3.5-0.8B

- **What:** the same runs on the hybrid model, text only
  (`Qwen3_5ForCausalLM` or skipping the vision tower).  The DeltaNet recurrence
  and its short convolution stay torch FP32; confirm it runs on sm_70 through
  `transformers`' PyTorch fallback, and that every swapped `k` is a multiple
  of 16.
- **Why after:** a second, architecturally different model, once the
  pipeline is proven on the plain one.

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
