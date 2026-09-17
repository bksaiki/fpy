# Roadmap: an FPy-to-Triton backend

## Goal

Compile an FPy function to a `@triton.jit` kernel that runs on torch tensors,
so an FPy numerical model is callable from PyTorch with no build step.

The deliverable is the integration, not the code generation. A CUDA backend
would be *cheaper to implement* — a CUDA thread runs the scalar program, so
SIMT is free and §5–§7 below disappear — but it ships as an `nvcc` dependency,
a build system and an extension ABI. Triton emits **Python source text**: the
backend `exec`s a string, Triton JITs it at first call, and the result takes
`torch.Tensor` arguments directly. That is strictly less machinery than the C++
backend already carries, and it is the whole reason to prefer the harder
codegen target.

## Scope: float storage is FP16 and wider

The float storage ladder is `fp16`, `fp32`, `fp64`, plus the integer rungs.
**FP8 and narrower are not storage types, and neither is `bf16`.** They remain perfectly
usable as *rounding targets* — an `S1E4M3` or `MX_E5M2` value is held in a
wider rung and its rounding is lowered by `unfold_round` — which is the
storage-contains-a-format rule from [backend-cpp.md](backend-cpp.md) applied
deliberately rather than by omission.

Three things follow, and they are why the restriction is worth stating up front
rather than discovering later:

- The fnuz-versus-IEEE fp8 question disappears. `S1E4M3` is `float8e4b8` and
  not `float8e4nv`, a distinction that is a correctness trap when fp8 is a
  native storage type and a non-question when it is not.
- `supported_fp8_dtypes` and its per-architecture variation stop mattering, so
  the target description does not have to be arch-conditional.
- The sub-fp16 formats reach the same lowering path as every other non-native
  format, so there is one mechanism rather than two.

**Dropping `bf16` makes the float ladder a chain**: prec 11 ⊂ prec 24 ⊂ prec 53,
each containing the last. That was not true with `bf16` in it — bf16 (es 8,
prec 8) and fp16 (es 5, prec 11) are mutually incomparable, and §2 billed
choosing between them as the hardest open decision in the target description.
It is now closed by scope rather than by analysis. `TF32` goes with it: it was
never a storage type, only a `tl.dot` input precision, so `input_precision` is
a value to pin rather than a decision to make.

Both remain reachable as *rounding targets* on the same footing as fp8, so a
program that rounds to `BF16` or `TF32` still compiles. What is given up is
holding one in its own width across a kernel boundary.

Even so this ladder is a strict superset of the C++ backend's, which has no
`fp16`.

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

compiled with `arg_types=[ListType(RealType(FP16), K)] * 2`. FP16 arguments,
exact products, FP32 accumulation. It is small enough to reason about
completely and it exercises every mechanism this backend depends on.

`x * y` is exact in fp32 and the pipeline knows it: FP16 carries prec 11, so
the product needs 22 bits against fp32's 24, and the exponent range is
comfortable. `FormatInfer` bounds the product, `StorageInfer` picks the
smallest rung containing that bound, and the result is checkable today —
the C++ backend emits `float p = (x * y);` for FP16 arguments and, for FP32
arguments, where the product needs prec 48:

```c++
double p = (static_cast<double>(x) * static_cast<double>(y));
```

Those explicit widening casts are the whole mechanism, and they are what the
Triton emitter will spell as `.to(tl.float32)`.

**What C++ cannot do with it, and Triton can.** The C++ signature is
`std::array<float, 8>`: with no fp16 rung the parameters widen at the boundary,
so the kernel cannot consume an fp16 buffer and the caller pays double the
memory. The emitted arithmetic is still *correct* — the values remain
FP16-bounded and their product is exact either way — but the ABI is wrong for
the use case. An fp16 rung fixes exactly that, which is the concrete argument
for §2's ladder.

**And the `zip` is already gone.** The emitted loop is
`for (int8_t _i = 0; _i < xs.size(); ++_i)` with `x` and `y` read by
subscript — `ZipElim` and `CompToLoop` between them have deleted the lockstep
fact before any emitter saw it. See *Tensorization*.

## What the path rests on

**Triton has no rounding-mode control.** PTX carries rounding modifiers per
instruction; Triton exposes none of them outside `tl.inline_asm_elementwise`
and `fp_downcast_rounding` on casts. So the op table dispatches natively on RNE
and nothing else. Every non-RNE rounding is lowered by
`UnfoldMode.ROUNDINGS`; arithmetic *under* a non-RNE context is refused, or
handled by `DOUBLE_ROUND` where the double-rounding rules permit it.

That is a real restriction and it should be sized honestly rather than
explained away. It costs nothing for a program whose arithmetic is exact — the
mode is unobservable when no rounding occurs, and storage only has to hold the
exact result — and it costs a refusal for a program that computes under, say,
an RTZ context. `DOUBLE_ROUND` is the escape hatch and it is a stronger claim
than `ROUNDINGS`, so it stays opt-in.

**The unfolded roundings need no runtime.** Per
[native-lowering-roadmap.md](native-lowering-roadmap.md), `unfold_special →
unfold_overflow → float_to_fixed → rescale_fixed` is bit-exact against the
interpreter across all eight FPy rounding modes and fourteen target formats,
and `CPP_HELPERS` is empty. That property was bought for C++ and is worth more
on a GPU, where there is no support library to link.

**Transcendentals are out of scope**, which closes Triton's one gap with no
knob: there is no `exp_rn`, `tl.exp` goes through libdevice `__nv_expf`, and
`tl.exp2` may lower to `ex2.approx.f32`. The C++ backend has the same exposure
and handles it by exclusion — `_NON_CR_OPS` in `tests/infra/backend/cpp.py`
lists 27 operators that disqualify a function from the bit-exact differential
check. Emitting none of them makes that set empty, so **every** compiled
function becomes eligible for bit-exact comparison. This backend gets a
stronger validation gate than the C++ one, not a weaker one.

## The fp16 trap

The scope decision above puts the ladder's weight on `fp16`, and Triton's type
rules there are not uniform. Measured against
`python/triton/language/semantic.py`:

| Expression | Triton's computation type |
|---|---|
| `fp16 op fp16` (non-div) | fp16 |
| `fp16 / fp16` | **fp32** — no hardware div below 32 bits |
| `fp16` with anything wider | fp16 |

(The `bf16` rows — division, and min/max, both promoting to fp32 — are out of
scope with `bf16` itself, but they are the same shape and would return with
it.)

The running example is a live instance, not a hypothetical: `x * y` on two
fp16 operands computes **in fp16** under these rules, rounding the product that
the program requires to be exact. What saves it is that `StorageInfer` gives
the product fp32 storage and the emitter therefore casts both operands in —
the same discipline that already produces `static_cast<double>(x)` in C++. The
cast is mandatory and nothing in Triton will ask for it.

This is precisely the storage-versus-rounding trap that
[backend-cpp.md](backend-cpp.md) records three miscompiles for: `a / b` under
an `FP16` context computed in fp32 and narrowed is a *double rounding*, and it
is not FP16's division. Nothing announces the promotion.

**The architecture already answers it.** The op table is keyed by
`(op type, operand types, output context)`, so the target description simply
does not contain a `Div` signature at FP16.
Ordinary dispatch then falls back to the fp32 signature, and `_maybe_cast`
refuses the lossy narrowing with the message telling the user to write
`fp.round(...)` — which is the correct outcome, since writing the rounding is
exactly what makes the double rounding explicit and intended.

One consequence for §2 survives the scope cut: C++'s `is_native_ctx` is a
predicate on a *context*, because `<cmath>` is uniform over a context's ops.
Triton's is honestly a predicate on `(op, context)`, since `Div` at FP16 is not
native while `Add` at FP16 is. `unfold_round` consumes `is_native_ctx`, so
either it is widened or the Triton target supplies the conservative
context-level answer and accepts some unnecessary lowering. Decide in §2, not
later.

## The shared-pipeline prerequisite

`StorageInfer` refuses a definition whose `FormatInfer` bound came back
`REAL_FORMAT` — *"cannot store an unconstrained real value in any storage
format"*. **No storage ladder contains `REAL`**, so a wider ladder fixes none
of those and a new emitter fixes none of them. It is the dominant refusal on
programs that round at a format computed at runtime.

This is shared-pipeline precision work, tracked separately, and it is **not a
gate on §1–§4**. It is a gate on how much this backend accepts.

## Tensorization: compiler, or transformations?

The open question, and the answer decides how big this backend is. Two axes get
conflated; they are independent.

**Lifting** — one FPy function per lane, N instances across a tile. The tile is
over the batch and the function's own values stay scalar. This is §5–§7.

**Tensorization** — the function's *own* arrays become tiles: a length-K list
is a `(K,)` tile rather than K unrolled registers, and a fold over it is a tile
reduction. Combined with lifting this is a `(BATCH, K)` tile reducing along
axis 1, which Triton expresses directly.

### It is a scheduling problem, and FPy already has a scheduling language

The instinct is to put tensorization in the backend, because the output is
`tl.sum` and `tl.where`. That is the wrong cut. Almost nothing tensorization
*decides* is a target fact:

| Decision | Kind | Where it belongs |
|---|---|---|
| Is this loop a map / reduce / scan? | program | keep the idiom — do not run `CompToLoop` |
| May this reduction be reassociated? | program | `ValueClassInfer` |
| Split a loop into outer × inner of width B | program | `strategies.split` — **exists** |
| Unroll the inner loop | program | `strategies.unroll_for` — **exists** |
| If-convert `IfStmt` → merge via `IfExpr` | program | a transform; FPy has ternary |
| Which values vary across lanes | program | §5, parameterized by which params vary |
| Is B a legal tile width? | **target** | target description |
| Spell it `tl.sum` / `tl.where` / `tl.load` | **target** | emitter |

Everything above the line is a rewrite from FPy source to FPy source.
`strategies.split` is already Halide's split, taking a factor and a cursor;
`unroll_for` already exists; cursors already forward across passes; the failure
contract is already `TransformDeclined` versus `TransformReferenceError`. This
is [scheduling-language.md](scheduling-language.md) §7 — Exo 2's
`optimize_level_1`: one entry point taking the function, a location and a
**target descriptor**, built by composing public operators.

A tile is representable in FPy, which is what makes this work at all: a
fixed-length list whose length `ArraySizeInfer` proves. A tiled program is
`for i in range(0, n, B): block = xs[i:i+B]; ...` with elementwise work as
comprehensions over `block` — ordinary FPy, which the **interpreter can run**.

### That is the decisive argument

A backend-independent tiling pass is validated by running the interpreter on
the tiled program and checking it agrees with the untiled one. No GPU, no
Triton, no torch. All of tensorization can be developed and tested with zero
GPU access, and it is
[backend-independence.md](backend-independence.md)'s criterion (2) —
"decisions became testable without a C++ string" — applied to the one part of
this project that would otherwise need hardware to test at all.

It also means the emitter stays thin. It receives a program already tiled and
already if-converted, and its job is spelling.

### What Triton decides, and what it will never ask about

Triton is smart *below* the tile abstraction and deliberately absent *above*
it. Given a tile and an index expression it does layout assignment, memory
coalescing and load vectorization (via its divisibility/contiguity analysis),
software pipelining of loops (`num_stages`), shared-memory allocation and sync
insertion. That is the work that is miserable in CUDA, and it is the reason to
target Triton at all.

It does not choose the block size, the tiled axis, the loop order, or whether a
loop is a map or a reduction. Those are inputs. Write scalar code and you get
scalar code.

Take `def f(xs, ys): ... for x, y in zip(xs, ys): ... return ...`. Every
choice is above the line:

- **Which axis is the tile?** If `f` runs on one pair of vectors, the loop is
  the only parallelism and the tile is the zip axis. If `f` is batch-lifted
  over N pairs, the tile can be the batch and the loop stays sequential per
  lane. Different kernels, same FPy source.
- **Map or reduction?** A body that builds a list is a map; a body that
  accumulates into a value carried past the loop is a reduction, and lowers to
  per-lane partials plus a cross-lane combine.
- **What is `BLOCK`?** See below.

And `zip` is the *signal*, which this pipeline currently discards twice:
`ZipElim` rewrites it to an indexed comprehension in `specialize()`, then
`CompToLoop` rewrites that to an `IndexedAssign` loop. `zip(xs, ys)` says
"these two arrays are indexed in lockstep over a common axis" — the exact fact
a tile lowering needs, deleted by two passes before the emitter sees it. That
is the §8 thesis in miniature.

**The one thing that cannot be delegated — and the running example fails it.**
A fold tensorizes to per-lane partials plus a tree reduction, and that
reassociates the addition. Triton will do it without comment. But FPy's `sum`
is a *left fold seeded with the first element unrounded*, performing n−1
additions, with the empty list an exact `+0` — that is the interpreter's
`_eval_sum`, and it is language semantics, not a convention.

Reassociating is sound when the additions are exact. In `dot` they are
**deliberately not**: accumulating in FP32 is the entire point, and an FP32
sum of exact fp32 products rounds at every step. `ValueClassInfer` cannot
discharge the precondition here because the precondition is false. Tensorizing
this fold produces a different number — defensibly, even preferably, for a
performance workload, but not the number FPy specifies.

**Which reorders §7 against §8.** Batch-lifting N independent `dot` calls —
tile over the batch, each lane running its own sequential fold — is *both*
fully parallel *and* bit-exact. Tensorizing the fold inside one `dot` is
neither. So for the workload this backend exists to serve, checking a numerical
model against hardware, §7 delivers the exactness FPy's objective names and §8
does not. §8 buys throughput on workloads that accept reassociation, and it
should be sequenced and justified as such rather than as the natural
continuation of §7.

The map half is unaffected: `x * y` over the zip axis is exact and tensorizes
freely. A program can therefore want a tile for its products and a sequential
fold for its accumulator, which is a scheduling decision and exactly the kind
the operators in [scheduling-language.md](scheduling-language.md) express.

**`BLOCK` is the one number FPy should not pick.** Emit it as a `tl.constexpr`
and let `triton.autotune` choose. This is consistent with
[scheduling-language.md](scheduling-language.md)'s *Not recommended* entry on
cost estimation — FPy's objective is exactness first and code shape second,
which is not a scalar — rather than a reversal of it. FPy decides the shape of
the schedule; Triton's autotuner supplies the number FPy has no basis to
choose.

### What this does not make free

- **`Hoistable` and `CompToLoop` are entangled.** Each supplies what the other
  lacks: `CompToLoop` declines a comprehension in a ternary arm or a `while`
  condition for want of a slot, and `Hoistable` makes the slot. Running
  `Hoistable` without `CompToLoop` may not reach a fixpoint. Decide this first;
  it gates the rest.
- **Tails.** `split` carries STRICT-divisibility `ValueError`s that
  [scheduling-language.md](scheduling-language.md) §1 flags as declined-shaped
  but uncatchable. Exo 2 generates tail cases from `specialize` plus
  simplification; FPy has `Specialize`, so the route exists but is unwalked.
- **Tiles are values; FPy lists are references.** A tiled program must stay in
  the subset `Alias` / `unbox` prove value-like. Already analyzed, not yet
  stated as a precondition anywhere.

A loop with a genuine loop-carried dependence stays a `for` over tiles, one
tile per iteration. That is the blocked form a hand-written kernel would use,
and it needs nothing new.

## The items, in the order they pay off

### 1. Flag audit and a hand-written witness

**Largely done — `exploration/triton/`**, on an sm_70 card at n=2000, every
prediction holding. What it settled:

**`enable_fp_fusion` is derivable, not a global pin.** This section used to say
to pin it off. Contracting `acc + x * y` into an `fma` rounds once over an
exact product, where the unfused form rounds the product and then the sum —
the same operation wherever the product is *already* exact. Predicted from
FPy's semantics alone (each program beside its `fp.fma` twin, no GPU), then
confirmed against the flag: the FP16-in/FP32-accumulate program is unchanged
by fusion (0/2000 either way), and an all-FP32 program differs under it
(590/2000) and matches without it (0/2000).

So fusion is safe exactly where `scalar_fits_in(product_format,
product_storage)` holds — a predicate the pipeline already computes. The target
description should *derive* the flag per kernel rather than pin it off and pay
the reported ~30% cost of `--fmad=false` everywhere. First concrete case of
this backend knowing something Triton never asks about.

**The fp16 cast trap is total.** `x.to(tl.float32) * y.to(tl.float32)` matches
the interpreter on 0/2000 differ; the naive `(x * y).to(tl.float32)` differs on
**2000/2000**. The casts come from `StorageInfer`, not from Triton.

**A hand-written batch-lifted kernel is bit-exact.** One lane per dot product,
fold sequential within a lane — full parallelism with FPy's left-fold order
intact, which is §7's argument over §8.

Still to pin: `input_precision='ieee'` is now a value to set rather than a
decision, `bf16` having left the scope with TF32. The one open question is
whether `enable_fp_fusion=False` reaches the packed `mul.rn.f32x2` /
`add.rn.f32x2` emitted on Blackwell — narrower than it was, since finding 1
means the flag is only needed where the product rounds, but still a silent
bit-exactness hole rather than a refusal, and untestable below sm_100.

The point of §1 was that it could invalidate §2–§9 for a week's cost. It did
not; it revised one of its own instructions and confirmed the rest.

### 2. Target description

`backend/triton/{types,storage,target}.py`. Self-contained and testable with no
emitter — `StorageInfer` runs against a domain directly, as
`tests/unit/backend/cpp/test_storage_ladder.py` does.

The real content is the ladder's *order*. `StorageDomain.sigma` is a sequence,
not a set, and its docstring already warns that containment over formats is not
a join-semilattice and that the order decides which programs compile. Adding
`fp16` and `bf16` adds two mutually incomparable rungs — fp16 is (es 5,
prec 11), bf16 is (es 8, prec 8) — neither of which contains `u16`, so their
placement against the integer rungs is a real decision with no obviously right
answer. C++ never had to rule on it.

Also settle the `is_native_ctx` arity question from *The fp16/bf16 trap*.

### 3. Scalar emitter

`backend/triton/emitter.py`, emitting a `@triton.jit` device function over
scalars. Reuses `CppCompiler`'s pipeline unchanged — every analysis, plus
`unfold_round` once §2 supplies `is_native_ctx` and `make_op_table`, which
`backend/cpp/unfold_round.py` already documents as the backend's contribution.

Ports nearly verbatim from the C++ emitter: `_IndentedWriter`, visitor
dispatch, `_emit_at`'s merge reconciliation, `_dispatch`, and the cast
discipline — `_maybe_cast` rejecting lossy implicit conversions,
`_explicit_cast` for user casts. Triton needs that discipline *more* than C++
does; the promotion table above is exactly the kind of silent narrowing it
exists to catch.

Deleted outright: the `fesetround` boundary (`_fenv_scope`,
`_validate_context_rm`, `_current_rm`) and everything spelling a
`std::shared_ptr` or `std::vector`.

### 4. Torch launcher and the differential harness

A generated Python launcher taking `torch.Tensor` arguments, and the GPU
counterpart of `tests/infra/backend/cpp.py`. Budget it honestly: that file is
2,469 lines and `tests/unit/backend/cpp/` is 11,505 more, and between them they
are why the C++ backend is trusted. This one needs a GPU in CI and a torch
dependency, and it gets an empty `_NON_CR_OPS` — so unlike the C++ harness,
every function it compiles it can also check bit-for-bit.

§1–§4 is a working, testable, torch-callable backend. Everything below makes it
fast.

### 5. Uniformity analysis

Which values are tile-invariant and which vary across lanes. A forward dataflow
over `DefineUse`, seeded by which parameters are lifted — shaped like the
existing passes in `fpy2/analysis/`, and backend-independent: any SIMD target
needs it, which is the right frame per
[backend-independence.md](backend-independence.md).

### 6. If-conversion

An `IfStmt` under a varying condition becomes a merge through `IfExpr`, which
the emitter spells `tl.where`. Targeting FPy's own ternary rather than a Triton
primitive is what keeps the pass backend-independent and interpreter-testable.
The AST carries explicit merge structure — `VariableAlloc`'s `is_intro` phis
and the `hoists_before` classification are where the pass hangs;
`transform/simplify_if.py` and `transform/if_bundling.py` are the neighbours.

The subtlety, and the one that will cost time: a branch that *guards* an
undefined operation cannot be freely predicated. `fpy2/analysis/value_class.py`
is what says when the guard was load-bearing, and it is already consulted for
exactly this reason in `_undefined_guard`.

Note the interaction with the unfolded roundings: they are branch-heavy by
construction — the FP16 example in
[native-lowering-roadmap.md](native-lowering-roadmap.md) is some thirty lines
with five data-dependent branches — and predicated over a tile, every lane
executes every path. That is the standing cost of choosing Triton over CUDA,
and it is a throughput cost, never a correctness one.

### 7. Lifting and the kernel ABI

`tl.program_id` / `tl.arange` / masked `tl.load` prologue, masked `tl.store`
epilogue, and a convention for what a lifted parameter is. FPy has no pointer
type, so this is an ABI decision rather than a language change.
`Specialize(size_key=True)` already models compile-time constant
specialization, which is `tl.constexpr` — that much is free.

### 8. Tensorization, as scheduling operators

Per the section above, this is not backend work. It is: a Triton-specific
`_to_statement_form` that keeps `ListComp`, `Sum`, `Zip` and `Enumerate`; tile
lowerings for each in the emitter; and — the substantial part — the
[scheduling-language.md](scheduling-language.md) §7 recipe, composing `split`,
`unroll_for` and if-conversion against a target descriptor. Resolve the
`Hoistable` / `CompToLoop` entanglement before starting.

Sequence it *after* §7 and treat it as opt-in per reduction, not as a default:
per *The one thing that cannot be delegated*, tensorizing an inexact fold
changes the answer, and §7 already reaches full parallelism bit-exactly for
batched work. Develop it against the interpreter, not the GPU — it is the one
item here with no hardware dependency, so it parallelizes with everything
else.

Until §8, static-length lists unroll to registers — `ArraySizeInfer` +
`Specialize(size_key=True)` + `ForUnroll` already do this. Dynamic-length lists
are refused in §1–§7; a refusal is always acceptable under the criterion in
[backend-cpp.md](backend-cpp.md).

### 9. `tl.dot`

Only once §8 exists and a program appears that should *use* a tensor core.
Nothing before §8 can express an operand to it.

## Effort

Measured where it says measured; everything else is an estimate, and
[backend-independence.md](backend-independence.md) is on record that estimates
framed as line counts misled every prediction made under them. Treat the
ordering as the useful content and the numbers as a sketch.

| § | Work | Sketch |
|---|---|---|
| 1 | Flag audit, hand-written witness | ~1 week |
| 2 | Target description | 2 weeks |
| 3 | Scalar emitter | 4–8 weeks |
| 4 | Launcher + GPU differential harness | 2–3 weeks |
| 5–7 | Uniformity, if-conversion, lifting | 6–10 weeks |
| 8 | Tensorization (scheduling layer) | unscoped; opt-in per reduction; gated on the `Hoistable` decision; no GPU needed |

## Not recommended

- **FP8 and narrower as storage types.** See *Scope*. Revisit only if a program
  needs an fp8 value to cross the kernel boundary in its own width, which is an
  ABI question and not an arithmetic one.
- **Inline PTX for non-RNE arithmetic.** `tl.inline_asm_elementwise` could
  reach `add.rz.f32`, at the cost of opting out of every Triton optimization
  around it. `DOUBLE_ROUND` covers the same ground within the existing
  machinery; reach for asm only if it demonstrably does not.
- **A global fast-math escape hatch.** There is no single Triton knob, and the
  per-op spellings in §2 are more precise than one would be anyway.
- **Autograd.** These are numerical models, not layers.
- **Waiting on `REAL_FORMAT` precision before starting.** §1–§4 are testable
  against the unit corpus, and §2's ladder question is answerable with no
  whole-program compile at all.
