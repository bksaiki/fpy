# mmasim quantized: local metrics beyond BF16

Implementation plan.  The design is settled; what follows is the phase
breakdown, one phase per commit.  Code lives under `examples/mmasim/serve/`,
next to the BF16 pipeline (`docs/todos/mmasim-serve.md`).

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

The serve pipeline evaluates MMA-Sim's BF16 designs.  Its primary
evaluation is local per-layer error on cached activations (`local.py`); the
end-to-end ones take hours per design (`mmasim-serve.md`).  Everything in it
assumes BF16 operands:

| piece | assumes |
|---|---|
| `kernels.BF16_DESIGNS`, `compiled` | the five BF16 x BF16 -> FP32 designs (`register` adds more of the same) |
| `kernels.round_input`, `prepare` | BF16 rounding, no scales |
| `kernels.matmul` | a design with arguments `(A, BT, C, out)`; `split_k` slices summed unscaled |
| `swap.Run` | modes `fp32`, `bf16-exact`, a design |
| `local.capture` | inputs recorded under `bf16-exact`, stored as BF16 |
| `layers.local` | the exact product of BF16-rounded `x` and `w` |

MMA-Sim also models the low-precision tensor-core instructions, and they
compile to Triton (`compile_triton.py`, 50/62; every FP32-accumulating FP8
and FP4 design compiles):

| kind | designs | instruction `k` | scales |
|---|---|---|---|
| unscaled chain, OCP E4M3 / E5M2 / E2M1 | `nv.ada.*.f32`, `nv.hopper.*.f32`, `nv.blackwell.*.f32` (incl. `e2m1`, mixed pairs), `nv.rtx_blackwell.*.f16.mma` | 32 (chains to any multiple) | none |
| unscaled chain, FNUZ E4M3 / E5M2 | `amd.cdna3.fp8`, `.bf8`, `.fp8.bf8` | 16 (chains) | none |
| block-scaled, one instruction | `nv.blackwell.mx.{e4m3,e5m2,e2m1,...}` | 32 | one E8M0 per operand row |
| block-scaled, one instruction | `nv.blackwell.nvfp4` / `.mxfp4` | 64 | four UE4M3 / E8M0 per operand row (16 each) |

The emitted kernels hold FP8/FP4 elements in FP16 storage, E8M0 scales in
FP32 and UE4M3 scales in FP16, all exactly.  The FP16-output designs at
`e_zero = -133` are refused by the compiler and out of scope.

Small quantized checkpoints of Qwen3-0.6B exist for the schemes wanted,
all in `compressed-tensors` format, all leaving `lm_head` in BF16:

| checkpoint | weights | activations |
|---|---|---|
| `RedHatAI/Qwen3-0.6B-FP8-dynamic` | E4M3, one BF16 scale per output row | E4M3, dynamic per token |
| `RedHatAI/Qwen3-0.6B-FP8-BLOCK` | E4M3, one BF16 scale per 128 x 128 block | E4M3, dynamic per 1 x 128 group |
| `kaitchup/Qwen3-0.6B-NVFP4` | E2M1 packed, UE4M3 per 16, FP32 per tensor | NVFP4, static per-tensor scale, dynamic per 16 |

(`Qwen/Qwen3-0.6B-FP8` is the same block scheme in Qwen's own format.  The
MXFP4 checkpoints found either apply Hadamard rotations first,
`ISTA-DASLab/...-FPQuant-RTN-MXFP4`, or mix MXFP4 and MXFP8 per layer,
`INCModel/Qwen3-0.6B-MXFP4-MXFP8`; see Open items.)

## Schemes, designs, models

Three separate things, each named once:

- **A scheme** says how each operand is represented: its element format, and
  its scales (format, block shape, static or dynamic).  BF16 is the
  scheme with no scales.
- **A design** is an MMA.  It consumes element formats, and, if it scales
  inside the instruction, a scale format and block.  A design is applicable
  to a scheme when it consumes the scheme's formats exactly; the scheme
  therefore picks its designs.
- **A model** stores its weights in some scheme: its master precision
  (BF16), or a quantized checkpoint's.

The schemes:

| scheme | elements (x, w) | scales | applied | designs |
|---|---|---|---|---|
| `bf16` | BF16 | none | -- | the five BF16 designs (today's pipeline) |
| `fp8-row` | E4M3 | x per token, w per output row | epilogue: `acc * s_x[i] * s_w[j]` | unscaled E4M3 chains |
| `fp8-block` | E4M3 | x per 1 x 128, w per 128 x 128 | per 128 of `k`: `y += partial * (s_x * s_w)` | unscaled E4M3 chains |
| `mxfp8` | E4M3 | E8M0 per 32 along `k`, both | in the instruction | `nv.blackwell.mx.e4m3` |
| `mxfp4` | E2M1 | E8M0 per 32 along `k`, both | in the instruction | `nv.blackwell.mx.e2m1`, `nv.blackwell.mxfp4` (each scale given to its two 16-groups) |
| `nvfp4` | E2M1 | UE4M3 per 16, both; FP32 per tensor | block in the instruction, tensor in the epilogue | `nv.blackwell.nvfp4` |

Element formats are parameters of a scheme, so `fp8-row:fnuz` (FNUZ E4M3)
is the same scheme for CDNA3's designs, and a mixed pair (E4M3 x, E5M2 w)
reaches the mixed designs; the six above are the named ones.  RTN scales are
FP32 unless the scheme says otherwise; a checkpoint's scales are taken in
the format it stores them (RedHatAI's FP8 scales are BF16).

### The recipes (round to nearest, "RTN")

Weights from a master, and all activations, are quantized by the standard
recipe for their scheme:

- **FP8 per row / per block**: `s = amax / 448` over the row or block,
  elements `rne_e4m3(v / s)`, saturating.
- **MX** (OCP MX v1.0, section 6.3): `X = 2^(floor(log2 amax) - emax)`,
  `emax` the element format's largest exponent (8 for E4M3, 2 for E2M1), as
  E8M0; elements `rne(v / X)`, clamped to the format's largest normal.
- **NVFP4** (NVIDIA's two-level recipe): per-tensor `g = amax / (448 * 6)`
  in FP32; per 16, `s = rne_e4m3(amax_block / (6 * g))`; elements
  `rne_e2m1(v / (s * g))`, saturating.

Activations are quantized from the BF16 values a deployment has in hand
(the captured inputs are already BF16).

### Where a model's weights come from

Every result records its weights' source.

| stored weights | what happens | why |
|---|---|---|
| in the scheme asked for | the checkpoint's elements and scales, as-is | the model as deployed |
| a master (BF16/FP32) | RTN by the scheme's recipe, by default (`weights: rtn`) | direct cast is the MX papers' methodology, and every checkpoint comes from a master |
| another scheme, converting exactly | allowed silently | nothing changes |
| another scheme, converting lossily | refused unless `--requantize` (`weights: requantized`) | it would measure compounded quantization, not the design |

A layer the checkpoint leaves unquantized (`ignore`, `lm_head` in every one
found) is left out of the scheme's metrics and listed in the results; an RTN
model follows the same convention.

### The two errors

With quantized operands, a layer's error splits in two, and both are
reported:

- **accumulation error**: the design's output against the exact product of
  the *quantized* operands, scales included.  This separates designs.
- **quantization error**: that exact product against the exact product of
  the unquantized operands (the BF16 activation and the master's weight).
  This separates schemes.

"Exact" is FP64, as today.  With E8M0 scales it is exact as before; with
FP32 scales a product of two dequantized operands can exceed 53 bits, so the
reference is exact to about `2^-50` relative to `|x|ᵀ|w|`, twenty-six bits
under any design's error.

### Capture

A scheme generalizes `bf16-exact`: `swap.Run` gains `<scheme>-exact`, the
quantized operands' exact product rounded once to FP32, and `local.capture`
records inputs under it, so every design under scheme S sees the inputs of
the S-quantized model computed exactly.  `bf16-exact` is `bf16`'s case,
unchanged.

### Interface

    python serve/local.py --scheme fp8-row --models Qwen/Qwen3-0.6B RedHatAI/Qwen3-0.6B-FP8-dynamic
    python serve/local.py --scheme mxfp4 -d nv.blackwell.mxfp4
    python serve/local.py --scheme nvfp4 --models kaitchup/Qwen3-0.6B-NVFP4 --requantize

`--scheme` defaults to `bf16`, which is today's behavior; `--designs`
defaults to every design applicable to the scheme; each model is evaluated
under the scheme per the table above.

## Phases

### Phase 1 -- Formats, schemes and recipes

- **What:** `serve/quant.py`: rounding to E4M3, E5M2, E2M1 (OCP and FNUZ)
  with RNE and saturation, E8M0 and UE4M3 scale rounding; a `Scheme`
  dataclass (per-operand element format, scale format, block shape, where
  applied) and the six named schemes; `quantize(t, scheme, operand)` -> a
  `Quantized` (elements in the kernel's storage, scales laid out per block),
  and `dequantize` for the exact reference.  Pure torch, no kernels.
- **Why first:** everything later consumes it, and it is testable alone.
- **Tests:** `tests/test_quant.py`: each rounding against FPy's own contexts
  for the format, over every BF16 value (65,536) and the ties and
  overflow cases; each recipe on hand-worked blocks from the OCP MX
  specification and NVIDIA's NVFP4 description; `dequantize(quantize(v))`
  within the format's half-ulp of `v / scale`.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_quant.py

### Phase 2 -- Unscaled FP8 designs and `fp8-row`, end to end in local metrics

- **What:** `kernels` generalizes from BF16: every compiled design with its
  consumed formats (read from its argument types) and storage; `round_input`
  and `prepare` become the scheme's quantizers; epilogue scaling
  (`acc * s_x[i] * s_w[j]`, FP32).  `swap.Run` takes a scheme and gains
  `<scheme>-exact`.  `layers.local` takes the quantized operands for its
  reference and gains the quantization error.  `local.py` gains `--scheme`
  and `--models`; RTN from the master only.
- **Why:** the smallest end-to-end step: the unscaled chains are the BF16
  pipeline with other element formats and a scaled epilogue.
- **Tests:** `test_kernels.py` extends the interpreter comparison to one
  design of each unscaled FP8/FP4 kind; the `bf16` scheme reproduces
  today's local metrics bit for bit (the regression net); `fp8-row`'s
  epilogue against an FP64 reference; `swap`'s `fp8-row-exact` against
  `layers.local`'s reference.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests

### Phase 3 -- `fp8-block`: scaled partials over `k`

- **What:** `kernels.matmul`'s slices, each from a zero accumulator, summed
  in FP32 with each partial scaled by its block's `s_x * s_w` (DeepGEMM's
  promotion every 128 of `k`).
- **Why separate:** it changes the combine that `split_k` shares, which
  deserves its own review.
- **Tests:** the scaled combine against an FPy reference of the same order
  on a small matmul; `split_k` unchanged.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_kernels.py

### Phase 4 -- Block-scaled instructions: `mxfp8`, `mxfp4`, `nvfp4`

- **What:** a `k` of many instructions as a chain of launches over one
  output buffer, each launch one instruction's `k` (32 or 64) with its
  scales and the previous result as `C`, as the hardware chains them (the
  aliased `C`/`out` launch allows it); NVFP4's per-tensor scale in the
  epilogue; MXFP4's scales given to both 16-groups under
  `nv.blackwell.mxfp4`.  `k / 32` launches per layer (32-96 for Qwen3), about
  a second per design on cached inputs.
- **Why separate:** the only designs whose scales live in the instruction.
- **Tests:** the chain against the interpreter's chain of the same design
  at `k` of two and three instructions; each scheme's exact reference.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_kernels.py serve/tests/test_quant.py

### Phase 5 -- Quantized checkpoints

- **What:** `serve/checkpoints.py`: a `compressed-tensors` reader for the
  three checkpoints above (FP8 per channel and per block; NVFP4 packed,
  with its global-scale convention), yielding each linear layer's elements
  and scales in its scheme and the model's other weights; the weight-source
  rules (as-is, RTN, exact conversion, `--requantize`); the master for the
  quantization error.
- **Why last:** RTN covers every scheme before it; checkpoints add loading
  and the rules, not arithmetic.
- **Tests:** each reader against the master (the dequantized weights near
  the BF16 ones, per scheme); the rules on small hand-made tensors, including
  a lossy conversion refused without `--requantize`.

      cd examples/mmasim && ../../.venv/bin/python -m pytest -q serve/tests/test_checkpoints.py

### After the last phase

```
.venv/bin/python -m pytest tests/unit -q -n auto
cd examples/mmasim && ../../.venv/bin/python -m pytest tests serve/tests -q
.venv/bin/python -m mypy fpy2
.venv/bin/ruff check examples/mmasim/serve
```

and a results section here: per scheme, the local metrics of every
applicable design on Qwen3-0.6B (RTN) and on each checkpoint, both errors,
on WikiText-2 and MT-Bench.

## Open items

### Which MX scale rounding: the specification's floor, or rounding up?

OCP MX v1.0 takes `floor(log2 amax)`, which can push a block's largest
element past the format's range, where it saturates; NVIDIA's MXFP8 recipe
in Transformer Engine rounds the scale up instead, never saturating, at the
cost of a coarser scale.  Provisional: the specification's floor, as the
reference definition.  Reopen if saturation shows in the quantization error.

### NVFP4's per-tensor scale for activations: dynamic or calibrated?

Transformer Engine computes it per tensor at runtime; ModelOpt and
`compressed-tensors` calibrate it once and store it (`input_global_scale`).
Provisional: dynamic for RTN, the stored one for a checkpoint.

### The FP32 order of `fp8-block`'s promotion

DeepGEMM scales and adds each partial on CUDA cores; whether as
`acc + partial * (s_x * s_w)` in separate roundings, with an FMA, or scaling
in another order changes the last bit.  Provisional: separate roundings in
that order.  Reopen if matching a library bit for bit becomes a goal.

### The quantization error's master for a checkpoint

A checkpoint does not say what it was quantized from.  Provisional: its
model card's `base_model`, else `--master`; without one, the quantization
error is not reported.

### Activations quantized from BF16 or from FP32?

Deployments quantize the BF16 activations they hold, and the capture already
stores BF16; the model here runs FP32 outside the linear layers, so FP32
activations are available too.  Provisional: BF16, as deployed.

### MXFP4 checkpoints

The ones found rotate with Hadamard transforms first (FP-Quant) or mix
MXFP4 and MXFP8 per layer (Intel's).  Provisional: MX schemes are RTN only.
Reopen for a rotation-free MXFP4 checkpoint, or to support the rotation
(it is a fixed transform of both operands before quantizing).
