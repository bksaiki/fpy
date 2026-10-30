# mmasim serving: the grid evaluation

Code under `examples/mmasim/serve/`; how to run it: `serve/README.md`.
Every `nn.Linear` of an LLM computes through a Triton kernel emitted from an
MMA design, and the effect of each design's arithmetic on the model's
next-token predictions is measured: the paper's RQ1.

Earlier stages -- local per-layer metrics, the one-parameter-at-a-time
sweep, the precision calibration, perplexity, zero-shot, MATH-500 decoding,
MT-Bench, quantized checkpoints, `fp8-block`, split-K -- were removed after
`0d9673b7`.  Their record and code: `git show
0d9673b7:docs/todos/mmasim-serving.md`.

## Working policy

- **Pause after each phase for review.**  **Do not commit**: the repository
  owner does.
- **Run only the tests relevant to the phase**; the full suite once, at the
  end.
- **Comments stay succinct**; process notes belong here.
- `pytest -n 8`, never `-n auto`; `serve/tests` serially: they load models
  on the GPU (12 GB).

## The harness

**Matmuls.**  Runs share one model and one input stream and differ only in
how `nn.Linear` computes `x @ W.T`; everything else runs in FP32, TF32 off.

| matmul | operands | product |
|---|---|---|
| R0 `fp32` | FP32 activations, BF16 master weights | FP32 |
| R1 `<scheme>-exact` | quantized by the scheme | FP64, rounded once to FP32 |
| `<scheme>-exact-rand` | as R1 | R1 moved an ulp up or down at random: the floor |
| a design | quantized by the scheme | the design's kernel, scales applied as the scheme says |

R1 is an ideal MMA on the quantized operands, so a design differs from it
only in how it accumulates; its error is 2^10 or more times smaller than the
quantization's, visible only against R1.

**Schemes** (`core/quant.py`; a design applies when it takes the scheme's
formats exactly, `kernels.applicable`):

| scheme | elements | scales | applied | in the grid |
|---|---|---|---|---|
| `bf16` | BF16 | none | -- | yes |
| `fp8-row` (`:fnuz`) | E4M3 (FNUZ, CDNA3) | FP32, x per token, w per output row | epilogue | yes |
| `mxfp8` | E4M3 | E8M0 per 32 | in the instruction | yes |
| `mxfp4` | E2M1 | E8M0 per 32 | in the instruction | chat only |
| `nvfp4` | E2M1 | UE4M3 per 16, FP32 per tensor | instruction, then tensor | chat only |

Weights are the BF16 master's, quantized by round-to-nearest as `torchao`
does (bit for bit, `test_quant.py`).

**Settled:**
- **Every linear layer;** `lm_head` too under `bf16`, left at R0 under a
  quantizing scheme.  Attention's `QK^T` / `PV` and the DeltaNet recurrence
  stay FP32 (Gaps).
- **`k` in one chain** of the design's instructions, from `C = 0`: the
  worst case for a truncating design, as libraries' split-K shortens it.
- **Engine: `transformers`,** each linear layer's `forward` replaced
  (`swap.patch`): deterministic and eager.
- **Harness: FP32** (2026-10-21).  FP64 non-matmul operations were identical
  above the FP32 floor and lowered it 6-40x; a difference below FP32
  inference's own rounding shows in no deployment.
- **Non-matmul operations in `transformers`' reference PyTorch.**  Qwen3.5's
  linear attention falls back to it without `causal-conv1d` and
  `flash-linear-attention`; installing either switches those layers to
  fused kernels of other orders and precisions, and moves every result.
  Keep them out of this environment.

**Kernels** (`core/kernels.py`): each design compiles once with `m`, `n` and
`k` symbolic, a block-scaled one as a chain of its instructions
(`block-scaled-kernels.md`); compiled kernels and their compile times are
cached in `results/cache/kernels`.  Tiles are the fastest measured per kind
of design at the model's layer shapes (`kernels.tile`); a tile changes no
result.

## The two runs

`frozen.py` (`scoring.frozen`, `swap.Frozen`), WikiText-2, teacher-forced, 10
random 2048-token segments paired across designs:
- **Amplified (frozen):** each layer multiplies R1's quantized input
  `Q(x_R1)` by the design's arithmetic, and its own input's offset passes
  through the weight unrounded, `y = K(Q(x_R1), W) + (x - x_R1) W^T`.  The
  arithmetic's error enters every layer and propagates; no activation
  rounds differently from R1's.
- **Observed (free):** as deployed.  Re-quantizing the next layer's input
  costs K₀ (~1e-2 under FP8, ~5e-5 under BF16), above most designs.
- **Metric:** KL(R1 ‖ run) per token, from FP64 log-softmax (FP32's rounding
  alone is ~1e-9), with its standard error over segments.
- **Checks:** R1 frozen is 0 exactly; KL ∝ ε² (slope 2.00, R² 0.9996).
- **Validity:** `rerounded`, the share of activation elements the design's
  own run would round differently: under ~1% the frozen run is the design's;
  above, its linearized effect.
- **Floor:** `-exact-rand`, FP32 inference's own rounding.  A design at the
  floor injects less error than that; it is reported so, not ranked.

## The design space

`sweep.grid()`: each generation's BF16 and FP8 instruction (Blackwell/Hopper,
Ada, CDNA3, CDNA2; Blackwell's FP8 is MXFP8; CDNA2 has none), each over `F`
10-28 (its own added; the paper's `k`), and each again with `c` added the
other vendor's way (a separate join keeps `F2 = F + 7`); the accumulator and
the rest as built.  **N = 113** (`test_the_grid_is_113_designs`).  The
accumulator is fixed by the instruction encoding, so it is not swept.

## Results

**Done** (2026-10-29; Qwen3.5-0.8B, seed 0, a fresh run;
`results/sweep/sweep-grid-Qwen3.5-0.8B.csv`, `plot_grid.py`).  105 measured,
8 refused (CDNA3 FP8 with `c` added with the products: "`sum` is exact, and
no storage holds its elements and partial sums").
- **The dashed lines are CDNA3's post adder,** which rounds toward -inf:
  its error is a bias (mean ~85% of its RMS on a random BF16 matmul),
  amplified 1-1.8 decades worse than with the products at the same narrow
  `F`.  No MMA-Sim instruction has a post adder rounding toward zero, so the
  grid leaves it out.
- **Accumulator width caps `F`:** Hopper's and Ada's FP8 flatten past `F` 13
  at their E8M13 accumulators; the FP32 ones fall to the floor.
- **Cost:** 12.8 h on a TITAN V: 24 min compiling (one at a time, 9-20 s a
  design, MXFP8 ~75 s with its per-`k` kernels, 11-14 s as one kernel
  since), then per run 1.4-2.1 min for
  FP8, 2.3-4.4 min for BF16, 15 min for CDNA2 with `c` after the products.
  Recorded per row (`compile_seconds`, `seconds`, `free_seconds`).
- **Floors:** amplified 1.6e-12 (`bf16`), 7.4e-12 (FP8); observed (K₀)
  5.0e-05, 9.4e-03 to 9.9e-03.

## Gaps

- **Attention's matmuls and a long context.**  `QK^T` and `PV` run on tensor
  cores in deployment, the majority of FLOPs at long contexts.  Hook the
  attention function, run both through the designs (or state one FP32
  softmax, then `PV`), on an 8-32k-token workload.
- **A larger model or a second family.**  Both models are 0.6-0.8B Qwen.
- **The accumulation order.**  Every result is one chain over all of `k`;
  state it as the worst case, or measure a library-realistic split-K.

## Open items

### How does a frozen run relate to the free one?

Whether amplified KL says anything about deployment beyond the floor.  Under
the uniform quantization-noise model a quantizer passes a small offset with
gain ~1 on average, so frozen and free should agree on a linear metric
(Δ NLL) in expectation; KL, quadratic, differs by K₀.  **Provisional:** the
paper claims only what the amplified run defines.  Reopen to test: frozen
and free Δ NLL over `F`, ~50 segments.

## After the last phase

    .venv/bin/python -m pytest -n 8 tests/unit
    cd examples/mmasim && ../../.venv/bin/python -m pytest -n 8 tests && ../../.venv/bin/python -m pytest serve/tests
    .venv/bin/ruff check examples/mmasim/serve

## Sources

- Yuan et al., *Understanding and Mitigating Numerical Sources of
  Nondeterminism in LLM Inference*, NeurIPS 2025 --
  https://arxiv.org/abs/2506.09501
- Rouhani et al., *Microscaling Data Formats for Deep Learning* --
  https://arxiv.org/abs/2310.10537
- Xie et al., *Bit-Accurate Modeling of GPU Matrix Multiply-Accumulate
  Units* (MMA-Sim) -- https://arxiv.org/abs/2511.10909
- `torchao` -- https://github.com/pytorch/ao
- Qwen3-0.6B -- https://huggingface.co/Qwen/Qwen3-0.6B;
  Qwen3.5-0.8B -- https://huggingface.co/Qwen/Qwen3.5-0.8B
