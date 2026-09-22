# Roadmap: an FPy-to-Triton backend

## Goal

Compile an FPy function to a `@triton.jit` kernel that runs on torch tensors,
so an FPy numerical model is callable from PyTorch with no build step.

The deliverable is the integration, not the code generation. A CUDA backend
would be cheaper to implement — a CUDA thread runs the scalar program, so SIMT
is free — but it ships as an `nvcc` dependency, a build system and an extension
ABI. Triton emits **Python source text**: the backend `exec`s a string, Triton
JITs it at first call, and the result takes `torch.Tensor` arguments directly.
That is less machinery than the C++ backend already carries, and it is the
reason to prefer the harder codegen target.

## The contract: semantics-preserving, floating-point included

[backend-cpp.md](backend-cpp.md) states the criterion this project inherits:
*if the compiler succeeds, the emitted code must behave as the FPy interpreter
does wherever FPy's semantics are defined.* A refusal is always acceptable; a
different answer is not.

**Holding that on a GPU is the unusual part.** Compilation to GPUs normally
treats floating-point semantics as approximately preserved, and the
approximation is not considered a defect: fast-math is on, fp32 matmul inputs
are silently substituted with TF32, reductions are reassociated into trees
because that is how a parallel reduction works, multiply-add is contracted,
transcendentals are a few ULP off, denormals may flush. Each is a deliberate
trade of numerical agreement for throughput, and for the workloads GPUs are
usually compiled for it is the right trade.

It is the wrong trade here. The object being compiled is a *model of an
arithmetic*, and a model that disagrees with itself under compilation has
stopped being one. So this backend takes the opposite default wherever the two
conflict:

| Where the GPU default trades accuracy | What this backend does |
|---|---|
| `default_dot_input_precision = "tf32"` | pin `input_precision='ieee'` |
| contract multiply-add freely | contract only where the analysis proves the product exact |
| implicit fp16 arithmetic on fp16 operands | cast per `StorageInfer` |
| tree-reduce a fold | keep FPy's left fold unless reassociation is discharged |
| approximate transcendentals | do not emit them at all |
| `/` and `tl.sqrt` | `div_rn` and `sqrt_rn` |

The point is not that the fast paths are wrong. It is that choosing between
them is a *semantic* decision, and this compiler holds enough information to
make it: "is this optimization observable?" has an answer in `FormatInfer` and
`ValueClassInfer`. A target that cannot ask has to assume the worst or ignore
the problem. Traditional GPU compilation ignores it. This one asks.

**And where it cannot ask, it refuses.** Rejecting what it does not like is
the mechanism FPy gets the most mileage from, and it is what keeps the contract
affordable — every open question below has a refusal as its fallback answer.

## Scope: float storage is FP16 and wider

The float storage ladder is `fp16`, `fp32`, `fp64`, plus the integer rungs.
**FP8 and narrower are not storage types, and neither is `bf16`.** They remain
usable as *rounding targets* — held in a wider rung, with the rounding lowered
by `unfold_round` — which is the storage-contains-a-format rule from
[backend-cpp.md](backend-cpp.md) applied deliberately rather than by omission.

Dropping `bf16` makes the float ladder a **chain**: prec 11 ⊂ 24 ⊂ 53. bf16
(es 8, prec 8) and fp16 (es 5, prec 11) are mutually incomparable, and ordering
them was the one genuinely open decision in the target description. It is
closed by scope. `TF32` goes with it — never a storage type, only a `tl.dot`
input precision, so `input_precision` is a value to pin rather than a choice.

The ladder is still a strict superset of the C++ backend's, which has no `fp16`.

**What makes "rounding target" a real answer rather than a deferral**: per
[native-lowering-roadmap.md](native-lowering-roadmap.md), `unfold_special →
unfold_overflow → float_to_fixed → rescale_fixed` rewrites a float rounding
into arithmetic that is bit-exact against the interpreter across all eight FPy
rounding modes and fourteen target formats, needing no support library at all —
`CPP_HELPERS` is empty. That property was bought for C++ and is worth more on a
GPU, which has no support library to link even in principle. It is also what
carries every non-RNE mode: Triton exposes no per-instruction rounding
modifier, so a non-RNE context reaches codegen through that sequence or not at
all.

## The running example

```python
@fp.fpy(ctx=fp.REAL)
def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        p = x * y                 # REAL: exact
        with fp.FP32:
            acc = acc + p         # rounded to FP32
    return acc
```

with `arg_types=[ListType(RealType(FP16), K)] * 2`. FP16 arguments, exact
products, FP32 accumulation — small enough to reason about completely, and it
exercises every mechanism here, including the ones that *fail*.

`x * y` is exact in fp32 and the pipeline knows it: FP16 carries prec 11, so the
product needs 22 bits against fp32's 24. `FormatInfer` bounds it, `StorageInfer`
picks the smallest containing rung. Checkable today — the C++ backend emits
`float p = (x * y);` here, and for FP32 arguments, where the product needs prec
48, `double p = (static_cast<double>(x) * static_cast<double>(y));`. Those
widening casts are the whole mechanism; the Triton emitter spells them
`.to(tl.float32)`.

Two things the example shows that the prose cannot. The C++ signature is
`std::array<float, 8>` — with no fp16 rung the parameters widen at the boundary,
so that kernel cannot consume an fp16 buffer. And the emitted loop is
`for (int8_t _i = 0; ...)` with `x` and `y` read by subscript: `ZipElim` and
`CompToLoop` deleted the lockstep fact before any emitter saw it.

## Examples

Every count in this document comes from the same corpus, so the numbers are
comparable across items. It is 94 `Function`s:

| Source | What it contributes |
|---|---|
| `fpy2.libraries.core` | scalar primitives — `logb`, `max_e`, `tree_sum` |
| `fpy2.libraries.vector` | 1-D maps and reductions — `max_element`, dot products |
| `fpy2.libraries.matrix` | 2-D nested loops writing `out[i][j]` — `add`, `is_diagonal` |
| `fpy2.libraries.eft` | error-free transforms, where exactness is the point |
| `fpy2.libraries.metrics` | comparisons and error measures |
| `examples/mmasim` (`utils`, `nv`, `amd`) | the hardware models — multi-return, `zip`, special-value handling |

`exploration/triton/programs.py` holds the running example on its own: a
batched dot product in three storage variants, which is what the hand-written
kernels are compared against.

The mmasim functions are the demanding half. They are the ones with early
returns inside loops, they iterate list *arguments* whose length nothing fixes
until specialization, and they carry the user-written asserts — so they set the
refusal boundary for most items here, while the library functions establish
what the common shape is.

## What is implemented

Not TODOs. Kept because each states a constraint the rest still has to honor.

### Numerics, measured on hardware — `exploration/triton/`

Hand-written kernels for the running example, bit-compared against the
interpreter on an sm_70 card at n=2000.

**`enable_fp_fusion` is derivable, not a global pin.** Contracting `acc + x*y`
into an `fma` rounds once over an exact product, so it is the same operation
wherever the product is already exact: the FP16-in program is unchanged by
fusion (0/2000 either way), an all-FP32 program differs under it (590/2000) and
matches without it. Safe exactly where `scalar_fits_in(product_format,
product_storage)` holds, which the pipeline already computes — derive it per
kernel rather than pin it off and pay `--fmad=false` everywhere.

**The fp16 cast trap is total.** `x.to(tl.float32) * y.to(tl.float32)` matches
the interpreter (0/2000 differ); `(x * y).to(tl.float32)` differs on
**2000/2000**, because Triton types `fp16 op fp16` as fp16. The casts come from
`StorageInfer`; nothing in Triton asks for them.

**Open, untestable below sm_100:** whether `enable_fp_fusion=False` reaches the
packed `mul.rn.f32x2` / `add.rn.f32x2` emitted on Blackwell. It would be a
silent bit-exactness hole rather than a refusal.

### Normal form — `fpy2/backend/triton/normalize.py`

`normalize` / `normalize_module`, built on `FuncInline`, `SingleExit` (#313)
and `SimplifyIf` (#303, #314). Reaches the form on **86 of 94** corpus
functions; the other 8 are loop-shaped and are item 2's input.

**The order is `SingleExit` → `FuncInline` → `SimplifyIf`, and leaves-first
across the call graph.** `FuncInline` refuses a callee with more than one
return, so a function must be single-exit before anyone can inline it;
`Module.map` supplies the order and rebinds each caller's `Call.fn` to the
transformed callee. `SimplifyIf` runs last because sinking a `return` *creates*
the `if`s it consumes. A per-function `normalize` cannot fix a callee, which is
why the module-level entry point exists.

**Refusals that still bind:**

- A `return` inside a loop — FPy has no `break`. `Specialize` → `unroll_for` →
  `single_exit` gets all seven multi-return mmasim functions through, but see
  *How a loop lowers* before reaching for it as a general answer.
- A `return` under a `with` that only *sometimes* returns: moving the
  continuation inside would change its rounding context.
- A call in a branch, until inlining removes it — a callee's body is not
  scanned, so an `assert` or an overflowing rounding inside one would reach the
  hoist unseen.
- `SingleExit` copies the continuation where both arms fall through, so its
  cost is exponential in nesting depth (3 here, refused past 8). FPy has no
  undefined value, so every formulation avoiding the copy needs a typed dummy
  and collapses back into it.

**`SimplifyIf` runs in the default mode, not `strict`.** A guarded subscript is
the common shape and lowers well to a mask. `strict` would additionally refuse
out-of-range subscripts, unresolved contexts, and an operation whose context
cannot hold an infinity or NaN it might produce — and it over-refuses, since
refusals are judged on the arm as written, before it is known to inline.

**On evaluation order.** An `if` arm that reduces to expressions is placed
*inside* the `IfExpr`, which the interpreter evaluates lazily. `tl.where`
evaluates both arms, but the GPU does not trap where the interpreter would —
`logb(0)` is `-inf` in hardware and `tl.where` discards it — so the two agree
on the value. The consequence is that correctness for these shapes rests on
inlining rather than on refusal, so arm coverage is what to watch.

### Tileability — `fpy2/backend/triton/vectorize.py`

`why_not_tileable(stmt, func)` returns the reason a loop body cannot be
evaluated as a tile, or `None`. **37 of 59 corpus loops are tileable.**

**Map versus fold is the wrong axis** — the measurement says so. A
list-building loop still carries its output list, so by a carried-variable
test every real loop is a fold. What decides is the *combine*, per carried
variable:

| | |
|---|---|
| `max`, `min`, `and`, `or` | select an operand — any grouping agrees |
| the same literal written repeatedly | idempotent, so the order stops mattering |
| `+`, `-`, `*` | round — regrouping agrees only where every step is exact |
| anything else | refused |

The third row is the one worth having. `rounds_exactly` decides it from
inferred formats, so a tile reduction is admitted only when it is
*bit-identical* — where LLVM's vectorizer takes a `reassoc` flag from the user
and calls an `FAdd` reduction ordered otherwise. Same taxonomy as
`RecurrenceDescriptor`; the difference is deciding it by proof.

**The idempotent row is not an optimization.** `if p(x): ok = False` is an
`and`-fold that does not look like one, and it is 8 corpus loops plus three of
the four guarded-`for` refusals blocking item 1.

**A list write must be at the loop's own index** — every subscript the loop
variable or invariant across the loop. `out[i % 2]`, `out[k]`, `out[i + 1]`
are refused rather than sent to a dependence test. That costs 6 of 30
list-writing loops and buys soundness outright; the alternative is
reimplementing Banerjee or Omega, and if that is ever wanted the concepts
should come from ISL rather than be grown here.

### Tiling — `tile_loops`, `SplitLoopStrategy.MASK`

`tile_loops(func, width)` splits every loop `why_not_tileable` accepts, using
`MASK`, and leaves a refused one sequential.  Composed with the normal form it
holds on the corpus: 20 functions reach `normalize` → tileable → split, and
**19 of 19** checkable ones agree with the original on sampled inputs.

**The innermost tileable loop carries the tile**, and the ones enclosing it
are left as loops.  Both idioms take that shape: the fused softmax makes the
row the program instance and the columns the tile, and the matmul takes its
block indices from the program id and loops over tiles of `K`.  So `out[i][j]`
comes out as an untouched `for i` around a tiled `j`, not a nest of tiles.

Of the 29 tileable corpus loops, 20 are innermost and 9 enclose another; those
9 are the ones this leaves for the emitter to map to a program id or a
sequential loop.



`split` had two remainder policies and neither fit: `PEEL` emits a *second*
copy of the body as a residual loop, and `STRICT` demands the length divide the
factor. `MASK` chunks the whole length so the last chunk over-runs, and guards
the body with `j < n`:

```python
for i in range(0, n, f):       # runtime trip count
    for j in range(i, i + f):  # constant width
        if j < n:              # the mask
```

That is idiom 2 from *How a loop lowers*, which is the requirement: the trip
count is a runtime value, so one compiled kernel serves every length.

**It synthesizes no remainder, and that is load-bearing rather than tidy.**
The first draft padded the length up to a multiple of the factor, which needs
one — and this op table has neither `Fmod` nor `Mod`, only `Div`, with integer
`Div` omitted because FPy truncates where Triton's `//` floors. So the padded
form could not have been lowered here at all. It is also unnecessary:
`range(0, n, f)` already yields `ceil(n / f)` chunks. What remains uses only
`len`, `+` and `<`.

`PEEL` and `STRICT` still do synthesize a remainder, and take a `use_fmod` flag
so a caller picks the spelling its backend can lower. Neither spelling helps
here, which is why `MASK` having none is the point.

### The compiler — `fp.TritonCompiler`

`TritonCompiler().compile(func, ctx=..., arg_types=...)` returns a
`KernelSource`: the `@triton.jit` text, its parameters, and the derived
`enable_fp_fusion`.  `compile_module` does every public entry.

**What it is for is the order.**  Three of the steps' orderings are not
obvious and were each found by hitting them:

- `Specialize` first — a kernel argument is a bare pointer, so the only length
  available for offset arithmetic is the one specialization proved.
- `ConstFold` before emitting — `tl.static_range` needs its trip count as a
  constant, and a `range(K)` naming a *foreign* constant arrives as a free
  variable, since `Specialize` monomorphizes contexts and types, not closure
  values.
- Tiling *after* the normal form — a masked body is a guarded element write,
  which `SimplifyIf` refuses to hoist, so normalizing after tiling would
  reject this pipeline's own output.

**And what the ABI asks of the program.**  A kernel writes through pointers
its launcher owns and returns nothing, and its tile width is a compile-time
parameter — so a compiled function takes its output as an argument and its
tile width as one too.  Both are the principle *Not recommended* states for
the batch dimension: say it in the program rather than invent a convention.
A missing tile-width parameter is a refusal, not a guess.

### Launching — `fpy2/backend/triton/launcher.py`

`launch(src, args, block=...)` runs an emitted kernel on torch tensors, and
**the emitted kernel agrees with the interpreter bit-for-bit** on an sm_70
card — checked at `n` of 1, 6, 8 and 9 against a `BLOCK` of 4 and 8, so a
full tile, several tiles, and three different partial tiles all ran.

**CI checks the emitted source, not the execution.**  GitHub's GPU runners
are paid and not free for public repositories, so the differential cannot run
in CI.  What can is an expect test over the *text*: the whole kernel is
pinned, so an emitter change shows as a diff on any machine.  That covers the
compiler end to end without hardware, and leaves execution as a manual check
here.

Pinning the whole text rather than fragments is deliberate -- the trap this
backend exists to avoid is a cast on the wrong side of a multiply, which is a
change of one token's position and not of any substring worth grepping for.

`FPY_REQUIRE_GPU=1` turns the skip into an error, so a machine that *has* a
card enforces the differential instead of quietly passing.

**Triton has no CPU target.**  Mainline builds `amd` and `nvidia` only, and a
CPU tensor fails with *"Pointer argument cannot be accessed from Triton"*.  So
`triton` importing is not enough to run anything: `unavailable()` checks torch,
triton *and* a device, and the tests skip on its answer.

Two constraints the hardware taught, neither guessable from the source:

- **A `@triton.jit` function must live in a file.**  It reads its own source
  back with `inspect.getsourcelines`, so `exec` into a namespace fails with
  *"@jit functions should be defined in a Python file"* — the same constraint
  `@fp.fpy` has, for the same reason.  The launcher writes the kernel out and
  imports it.
- **A `tl.constexpr` does not survive a copy.**  `SplitLoop` binds the factor
  to a temporary, and `tl.arange`'s arguments must be `constexpr`, so a tile's
  width has to reach `arange` as the parameter itself.  The emitter resolves
  it back through the copies that bound it.

`enable_fp_fusion` is read off the `KernelSource` rather than taken from the
caller: whether contracting a multiply-add is observable is a property of the
program, and letting a launcher override it would make the answer depend on
who ran the kernel.

### Target description — `fpy2/backend/triton/`

`types.py`, `storage.py`, `target.py`; 37 tests, no emitter. `StorageInfer` runs
against the domain directly, so the running example's storage is checkable
without generating a line of Triton.

Ladder: `u8, s8, u16, s16, f16, u32, s32, f32, u64, s64, f64`. `F16` after `S16`
is the one real decision — it must follow the 8-bit integers, which nest in it,
and against the 16-bit ones it is incomparable, so placing it later means
"integers in [0, 2000]" takes `u16`.

`is_native_ctx` is a predicate on the **context**, not on `(op, context)`. The
two readings the cpp backend can conflate come apart here — `Add` at FP16
dispatches and `Div` at FP16 does not — and the *cast* reading must survive,
since `x.to(tl.float16)` is FP16's round-to-nearest-even.

Every omission in the op table is a refusal: all transcendentals (none
correctly rounded — which makes the differential check's exclusion list
*empty*), `Div` at FP16 (computed in fp32, a double rounding), integer `Div`
(FPy truncates, `//` floors), `/` and `tl.sqrt` (fast variants), every rounding
mode but RNE. No list storage — a proven-length list unrolls into registers,
an unproven-length one is refused.

## The pipeline inlines everything

`triton.jit` takes `noinline` as an *opt-in*, so Triton inlines by default and
an emitted device call buys nothing at runtime; FPy forbids recursion, so
inlining always terminates.  Inlining is therefore the policy, and it leaves
`SimplifyIf` no call to refuse and the emitter no call ABI to define.

It was gated on single-exit normalization, since `inline` declines a callee
with more than one return — 7 of 20 mmasim functions, against 0 of 74 in
`fpy2/libraries`, because the reference algorithms cascade early exits for NaN,
infinity and zero.  That gate is lifted (#313).

One trap: `inline` with `where=None` *silently skips* a call it refuses, so a
pipeline does not fail there.  It fails later, at `SimplifyIf`, naming the
callee — which reads as a `SimplifyIf` problem when it is an inlining one.

## The shape of the compiler

**Tiling is a program transformation, not an emitter feature.** The top-level
loop is written by the user; a pass splits it into an outer loop and an inner
one whose extent is the tile width, and the inner body vectorizes. Almost
nothing that decides is a target fact:

| Decision | Where |
|---|---|
| rewrite `if` statements to `if` expressions | transform — `SimplifyIf`, **done** |
| reject what will not rewrite | transform |
| split a loop into outer × inner of width B | transform — `split`, exists |
| unroll | transform — `unroll_for`, exists |
| may this fold be reassociated? | `ValueClassInfer` + exactness |
| is B a legal tile width | target |
| spell it `tl.where` / `tl.sum` / `tl.load` | emitter |

Everything above the line is a rewrite from FPy source to FPy source, so a
tiled program is still an FPy program and **the interpreter is its oracle**.
The whole tiling layer can be developed and tested with no GPU, which is
[backend-independence.md](backend-independence.md)'s criterion (2) applied to
the one part of this project that would otherwise need hardware to test at all.
It also keeps the emitter thin: it receives a program already split, already
vectorized, already if-expression-only, and spells it.

**Uniformity needs no analysis.** The split point *is* the boundary — outside
the inner loop is scalar, inside is tile — and Triton broadcasts a scalar
against a tile, so a loop-invariant value inside the body needs no special
handling. What is needed is "derived from the inner index", a forward
propagation, not a dataflow analysis. An earlier draft of this roadmap budgeted
a divergence analysis here; the explicit-loop formulation removes it.

**`BLOCK` is the one number FPy should not pick.** Emit it as a `tl.constexpr`
and let `triton.autotune` choose. Consistent with
[scheduling-language.md](scheduling-language.md)'s rejection of cost
estimation rather than a reversal of it: FPy decides the shape of the schedule,
and borrows a tuner for the number it has no basis to choose. `split` already
accepts a non-literal factor, so a specialized free variable works today.

## How a loop lowers

Checked against the Triton tutorials and `exploration/triton/kernels.py` before
committing to a plan, because the obvious plan was wrong.  Triton kernels write
loops three ways, and only one of them is a loop over scalars.

**1. No loop — the dimension becomes a tile.**  The per-row work in the fused
softmax tutorial:

```python
col  = tl.arange(0, BLOCK_SIZE)          # BLOCK_SIZE = next_pow2(n_cols)
mask = col < n_cols
row  = tl.load(ptr + col, mask=mask, other=-float('inf'))
m    = tl.max(row, axis=0)
den  = tl.sum(tl.exp(row - m), axis=0)
```

The iteration is implicit in the tile, and the reduction is one `tl.sum`.  Fast,
and it **reassociates** — so it is available to a map, and to a fold only where
the reassociation is discharged.

**2. A runtime loop over tiles — the workhorse.**  The matmul tutorial's `K`
loop:

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += BLOCK_SIZE_K * stride_ak
```

The trip count is a *runtime* value and the body is tile-shaped, so one compiled
kernel serves every `K`.  This is what item 2's `split` produces.

**3. `tl.static_range` — unrolled at compile time.**  What `dot_exact` uses:

```python
for k in tl.static_range(K):             # K must be tl.constexpr
    x = tl.load(xs_ptr + row * K + k, mask=mask, other=0.0)
    acc = acc + x.to(tl.float32) * y.to(tl.float32)
```

**Unrolling is a fallback, not the general lowering.**  `tl.static_range`
requires the count as `constexpr`, which means one compiled kernel per input
length and the body replicated that many times — at a realistic `K` that is a
compile-time bomb and it wrecks the instruction cache.  `dot_exact` pays it
deliberately and only because `K` is small: batch-lifting one dot product per
lane is what keeps FPy's left fold exactly, where a `tl.sum` over the tile would
reassociate.  That is a semantics choice, not a performance one.

**So the normal form must not pre-unroll.**  This is why item 1 keeps
`ListComp`, `Sum`, `Zip` and `Enumerate`: unrolling deletes the iteration that
item 2 needs in order to choose between idiom 1 and idiom 2.  A `for` under an
`if` wants a **mask**, which is idiom 1 and 2's native shape, not an unroll.

**What was measured while establishing this**, so it is not re-derived:

- `trip_count` in `fpy2/analysis/array_size.py` already answers the count, but
  only models `Range1`.  Of 59 corpus loops, 51 are `Range1`, and the other 8
  (`Var` 3, `ListSlice` 3, `Zip` 2) return `None`.  Nothing proves a count until
  `Specialize` runs with concrete `arg_types`; after it, `matrix.is_diagonal`'s
  two loops report `[3, 3]`.
- `ForUnroll.sites` must be queried with the `times` the rewrite will get.
  Asking with the default on a length-3 loop returns *zero* sites, because
  `STRICT` refuses a `k` that does not divide the length — the count has to be
  computed first and fed to both calls.
- Unrolling multiplies `SingleExit`'s continuation copying.  One user-written
  `assert` in `utils.dpa_special_values` became **27** after unroll plus
  single-exit; the duplication is exponential in nesting depth and unrolling is
  what creates the depth.  A disable-asserts flag would remove both that and the
  `SimplifyIf` refusal it then hits.

## This pipeline runs backwards for this target

`CppCompiler.specialize()` normalizes *toward statements*, because the C++
emitter wants statements. This backend wants expressions. Three passes are
actively destructive here, and all three are already in the pipeline:

| Pass | Destroys | Wanted instead |
|---|---|---|
| `Hoistable` | ternaries → `IfStmt` | the exact inverse of `SimplifyIf` |
| `CompToLoop` | comprehensions → indexed loops | the comprehension *is* the map |
| `ZipElim` / `UnfoldZip` | `zip` → subscripts | `zip` *is* the lockstep fact |

This is not an incidental mismatch, it is the structural one: most of the
tiling work is **declining to run passes that erase the idioms**, not
recovering the idioms afterwards. The corollary is that this backend needs its
own normal form rather than a tweak to `_to_statement_form`.

## The shared-pipeline prerequisite

`StorageInfer` refuses a definition whose `FormatInfer` bound came back
`REAL_FORMAT` — *"cannot store an unconstrained real value in any storage
format"*. No storage ladder contains `REAL`, so a wider ladder fixes none of
those and a new emitter fixes none of them. It is the dominant refusal on
programs that round at a format computed at runtime. Shared-pipeline precision
work, tracked separately, and a gate on how much this backend accepts rather
than on any item below.

## The work

### 1. A Triton normal form — built, one piece left

`fpy2/backend/triton/normalize.py`; see *What is implemented*. The stated
obstacle turned out not to be one: `Hoistable` and `CompToLoop` were thought to
be mutually dependent in a way that dropping both would leave unanswered, but
`SimplifyIf` removes the statement/expression distinction that created the
problem, and **not one** of the 8 refusals is a missing statement slot.

| | |
|---|---|
| 86 | normal form |
| 4 | a `for` would run unconditionally |
| 4 | a `return` inside a loop |

Both refusal groups are loop-shaped and neither is a `SimplifyIf` question.
Unrolling clears them on paper, but *How a loop lowers* says why that is the
wrong default — pre-unrolling deletes the iteration item 2 needs, and a guarded
loop wants a mask. **Settle item 2 first and let it say what the normal form
should leave standing.**

### 2. Split and vectorize

The decision is built (see *Tileability*), the rewrite is built, and the
policy is settled: the innermost tileable loop carries the tile.  What remains
is the emitter's half -- turning a tiled loop into `tl.arange` plus a masked
load, and an enclosing one into a program id or a loop over tiles.

`split` is semantics-preserving — `for i in range(n)` into outer x inner of
width B evaluates the body in exactly the same order — and now has the
remainder policy this target needs, `MASK` (see *Masked tails*). What remains
is to consume `why_not_tileable`: emit idiom 1 or 2 from *How a loop lowers*
for a tileable body, and keep a refused one sequential per lane.

**Split runs after the normal form, not before.** A masked body is a guarded
`IndexedAssign`, and `SimplifyIf` refuses to hoist a list write — correctly,
since hoisting would make the out-of-range store unconditional. So a masked
loop does not reduce to an `if` expression, and normalizing *after* splitting
would reject this pipeline's own output. Running it second also avoids
splitting loops inside callees that are about to be inlined away. No exemption
to item 1's "a remaining `IfStmt` is an error" is needed, provided the order
holds.

The running example stays sequential: accumulating in FP32 is the point, so
the adds round and `rounds_exactly` is false. That still parallelizes, across
the batch — one lane per dot product, which is what the hand-written kernels
do, and it is bit-exact where a tile reduction is not.

Tails are a `mask`, not a generated tail loop — which `MASK` now provides.
That is simpler than the `specialize`-based tail generation [scheduling-language.md](scheduling-language.md)
§7 points at, and it is one of the few places this target is *easier* than a CPU
one.

### 3. The emitter

`fpy2/backend/triton/emitter.py`. Its input is a program already normalized,
split and vectorized, so its job is spelling plus `tl.load` / `tl.store` at the
loop boundary and `tl.where` for an `IfExpr`.

**The input shape is settled, and it matches the hand-written kernel.**  Run
the batched dot product -- the batch as a *user-written* loop, per *Not
recommended* -- through `normalize` then `tile_loops`, and the structure that
comes out corresponds one-to-one with `kernels.dot_exact`:

| pipeline output | `dot_exact` |
|---|---|
| `for i in range(0, n, B)` | `tl.program_id(0) * BLOCK` |
| `for j in range(i, i + B)` | `tl.arange(0, BLOCK)` |
| `if j < n` | `mask = row < n_rows` |
| `r = t[j]` | `row` |
| `for k in range(K)` — refused, so sequential | `tl.static_range(K)` |
| `acc + xss[r][k] * yss[r][k]` under FP32 | `acc + x.to(tl.float32) * y.to(tl.float32)` |
| `out[r] = acc` | `tl.store(out_ptr + row, acc, mask=mask)` |

The refusal is load-bearing in that table: `why_not_tileable` declines the `K`
loop because the accumulation rounds, which is exactly why `dot_exact` keeps it
sequential per lane.  The parallelism is the batch, and the batch is a loop the
user wrote.

So the emitter is a structural mapping with both sides written down, not a
port of the cpp emitter's 4291 lines -- most of which is `std::vector`,
`std::shared_ptr`, unboxing and aliasing that this input cannot contain.

Ports nearly verbatim from the cpp emitter: `_IndentedWriter`, visitor
dispatch, `_emit_at`'s merge reconciliation, `_dispatch`, and the cast
discipline — `_maybe_cast` rejecting lossy implicit conversions,
`_explicit_cast` for user casts. That discipline matters *more* here: the fp16
trap is a silent narrowing of exactly the kind it exists to catch.

Nothing to port for `fesetround` or for `std::shared_ptr` / `std::vector`.

`enable_fp_fusion` is emitted per kernel from `scalar_fits_in`, not pinned.

### 4. Launcher and differential harness

A generated Python launcher taking `torch.Tensor` arguments, and the GPU
counterpart of `tests/infra/backend/cpp.py`. Budget it honestly: that file is
2,469 lines and `tests/unit/backend/cpp/` is 11,505 more, and between them they
are why the C++ backend is trusted. This one needs a GPU in CI and a torch
dependency, and it gets an **empty** `_NON_CR_OPS` — so unlike the C++ harness,
every function it compiles it can also check bit-for-bit.

Items 1–4 are a working, testable, torch-callable backend.

### 5. `tl.dot`

**Blocked by the same finding that retired reductions.** A tensor-core matmul
reassociates its `K` accumulation, so it is available exactly where a tile
reduction is — which the measurement below says is nowhere that wants it. The
running example is a dot product whose accumulation must stay a left fold, so
`tl.dot` cannot serve even the program this backend was designed around.

Revisit only if a program appears whose accumulation is provably exact *and*
large enough to want a tensor core. Nothing before that can express an
operand.

## Effort

Estimates, and [backend-independence.md](backend-independence.md) is on record
that estimates framed as line counts misled every prediction made under them.
The ordering is the useful content.

| Item | Sketch |
|---|---|
| 1. Triton normal form | built, less the 8 loop-shaped refusals |
| 2. Split and vectorize | built |
| 3. Emitter | built; coverage is one program wide |
| 4. Launcher + harness | built, differential passing; CI checks the text, not the run |
| 5. `tl.dot` | blocked; see the item |

**The gap is coverage, not capability.**  The pipeline compiles the running
example end to end and agrees with the interpreter bit-for-bit, but it has
only ever been driven on that program and a one-line map.  Running the corpus
through the emitter -- unspecialized, so several failures are the measurement's
fault rather than the emitter's -- surfaces three real holes:

- **`Max` and `Min` are absent from the op table**, and they are exactly the
  combines `why_not_tileable` calls *exact*: 17 of 29 tileable corpus loops
  reduce with one.  So the analysis names them the prime tile-reduction target
  and the emitter cannot spell them.  `tl.maximum` and `tl.minimum` exist and
  select an operand rather than rounding, so this reads as an oversight rather
  than one of the table's deliberate refusals.
- **A comprehension has no spelling**, and item 1 deliberately *keeps*
  `ListComp` for the vectorizer.  Two halves of this design disagree: 24
  corpus functions hit it.
- **Destructuring assignment has no spelling** -- 17 corpus functions.

Items 1–2 are all interpreter-testable, so they parallelize with each other and
need no hardware.

## Not recommended

- **A divergence/uniformity analysis.** The explicit-loop formulation makes the
  split point the boundary; an analysis would rediscover what the syntax states.
- **Implicit whole-program lifting.** Treating every scalar as a tile and
  inventing an ABI for which parameters are lifted. The loop says what is
  iterated, and a user-written loop is inspectable where an ABI convention is not.
- **Predicating everything.** The cheap version of if-conversion is unsound
  exactly where a branch guards an undefined operation, which is the case that
  matters. Refuse instead.
- **FP8 and narrower as storage.** Revisit only if a value must cross the kernel
  boundary in its own width — an ABI question, not an arithmetic one.
- **Inline PTX for non-RNE arithmetic.** `tl.inline_asm_elementwise` could reach
  `add.rz.f32`, at the cost of opting out of every Triton optimization around
  it. `DOUBLE_ROUND` covers the same ground within the existing machinery.
- **A global fast-math escape hatch.** There is no single Triton knob, and the
  per-op spellings in the target description are more precise than one would be.
- **Tile reductions (`tl.sum` and friends).** Retired from the work, and
  measured rather than argued. `tl.sum` reassociates, so it is sound only
  where every accumulation step is exact; `rounds_exactly` decides that, and
  the control confirms it sees through a loop — a loop-carried integer add
  answers `True`, the same add at FP32 answers `False`. On this corpus **0 of
  7** arithmetic folds qualify.

  The two conditions oppose each other: `tl.sum` buys throughput only on a
  *single large* reduction, and the larger a floating-point accumulation is
  the more certainly it rounds. What does qualify is integer accumulation
  under an unbounded context, which is exactly where nobody needs the
  throughput. Batch-lifting meanwhile reaches full parallelism *bit-exactly*
  for batched work, which is the shape the running example has.

  The residue, if a program ever needs it: `tl.sum` over an integer
  accumulation, gated on `rounds_exactly`. A note, not a work item.

- **Autograd.** These are numerical models, not layers.
