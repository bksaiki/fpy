# Path-sensitive value classes

A four-atom lattice — is this value a NaN, an infinity, a zero, or a finite
non-zero? — refined at every branch. Much weaker than the numeric reasoning in
[symbolic-exponent-inference.md](symbolic-exponent-inference.md), and it closes a
different and larger set of holes.

## Why

The lowered `FP16` rounding emits four guards, all from one fact nothing can
prove:

```c++
assert(std::isfinite(_t)  && "fpy: rounding is undefined for this value");
double _t7 = (std::isfinite(_tmp2) ? std::ldexp(x, static_cast<int>(_tmp2))
                                   : std::pow(2.0, _tmp2) * x);
assert(std::isfinite(_t7) && "fpy: rounding is undefined for this value");
t = (std::isfinite(exp) ? std::ldexp(_t8, static_cast<int>(exp))
                        : std::pow(2.0, exp) * _t8);
```

Two exist because the target context refuses NaN and the infinities, so the
backend must assert they do not arrive.  Two exist because `std::ldexp` takes an
`int` exponent and converting a non-finite value to one is undefined.

All four are unreachable.  The program says so, three branches earlier:

```python
if fp.isnan(x):   ...
elif fp.isinf(x): ...
elif x == 0:      ...
else:                     # every guard lives here
    e = fp.logb(x)
```

Format inference cannot see it.  `x` is `FP32`, so its bound admits a NaN and
both infinities, and `logb` faithfully passes that on.

## The domain

Per value, a subset of `{NaN, Inf, Zero, Finite}` — sixteen elements, where
`Finite` means *finite and non-zero*.  The join is union, the meet is
intersection, and the height is 4, so no widening is needed.

Forward transfer functions are the special-value tables the interpreter already
implements.  `logb` is the one that matters:

| argument | `logb` |
|---|---|
| `Zero` | `Inf` (negative, but the atom does not track sign) |
| `Inf` | `Inf` |
| `NaN` | `NaN` |
| `Finite` | `{Zero, Finite}` — `logb(1.5)` is `0` |

Verified against the interpreter.  The rest are the familiar rules: NaN
propagates through arithmetic; `Zero * Inf` is `NaN`; `Finite ± Finite` is
`{Zero, Finite}`; and `min`/`max` union their operands.

Rounding is the one to get right.  Under a **bounded** context it maps `Finite`
to `{Zero, Finite, Inf}`: a small enough value underflows to zero and a large
enough one overflows — `FP16.round(1e30)` is `+inf`, and `FP16.round(1e-30)` is
`+0`, both verified.  Only an **unbounded** context — or one whose overflow has
already been moved into program text by
[`unfold_overflow`](../source/strategies.rst) — drops the `Inf`.  Reading the
bounded case as `{Zero, Finite}` would prove exactly the guards that exist to
catch it.

## Refinement

On a branch, intersect the tested value's class with what the test implies:

| test | true arm | false arm |
|---|---|---|
| `fp.isnan(v)` | `{NaN}` | `{Inf, Zero, Finite}` |
| `fp.isinf(v)` | `{Inf}` | `{NaN, Zero, Finite}` |
| `fp.isfinite(v)` | `{Zero, Finite}` | `{NaN, Inf}` |
| `v == 0` | `{Zero}` | `{NaN, Inf, Finite}` |

**The `v == 0` row is the trap.**  A false result does *not* mean non-zero: a
NaN compares false to everything, so `NaN` stays in the false arm.  Verified —
`nan == 0` and `inf == 0` are both false.  An implementation that reads
`not (x == 0)` as "non-zero and finite" is unsound.

The chain works because refinements *intersect* down an `elif` ladder.  Reaching
the final `else`:

```
x : {NaN, Inf, Zero, Finite}
  ∩ {Inf, Zero, Finite}      not isnan
  ∩ {NaN, Zero, Finite}      not isinf
  ∩ {NaN, Inf, Finite}       not (x == 0)
  = {Finite}
```

and then `logb(x) : {Zero, Finite}`, `e - 10 : {Zero, Finite}` — neither `NaN`
nor `Inf`, which is exactly what the four guards ask.

## What it closes

- **The four guards above.** Two runtime branches and two assertions gone from
  every lowered rounding.
- **The libm mapping's specials condition** — the one unmet correctness
  side-condition in that lowering.  The interpreter raises where `std::trunc`
  returns a NaN; today only a branch nothing reads reconciles them, so the
  mapping rests on an unproven claim.  See gap 3 of
  [native-lowering-roadmap.md](native-lowering-roadmap.md).
- **`float_to_fixed`'s deliberate `enable_nan` omission.** It leaves the flag off
  because "a NaN reaches a rounding only as its operand, and the branches above
  take that case" — a comment asserting precisely what this analysis would prove.

## What it does not close

Anything about *magnitude*.  The `FP64` source still has no storage, because
that needs `|x| < 2^16` from a comparison and the relational
`2^-exp · x ∈ [2^10, 2^11)`.  Classes and magnitudes are independent, and the
numeric half is recorded separately.

The split is clean enough to state as a rule: **classes fix the specials
problems, magnitudes fix the width problems.**  Four of the five open items in
the roadmap are the former.

## Why a separate lattice, not three more flags

`AbstractFormat` already carries `has_nan`, `has_pos_inf`, `has_neg_inf` — three
of the four atoms.  What it structurally cannot say is **not zero**: its own
documentation notes that `pos_bound >= 0 >= neg_bound` holds by convention and
nothing excludes zero, so every format represents a `+0.0`.

That bit is load-bearing.  Knowing only `x : {Zero, Finite}` leaves `logb(x)`
possibly an infinity and every guard stays.  The chain needs `x : {Finite}`.

So this wants to be its own analysis rather than a fourth flag on the format
lattice — which also keeps it independent of the format lattice's arithmetic,
where a no-zero bit would have to be threaded through `__add__`, `__mul__`, the
join, and storage selection for no benefit.

## Plan

Commit-sized phases; the full suites run only at the end.

- [x] **1 — the analysis alone, no consumers.**  `fpy2/analysis/value_class.py`,
  exported from `fpy2.analysis`.  Transfer functions swept against the
  interpreter, refinement checked per arm, and the suite mutation-tested (break
  the `logb` table, drop `0 * inf → NaN`, disable refinement: 15 failures each).
- [x] **2 — `_ldexp_exponent`.**  Threaded into `SpecAnalyses` and the emitter;
  the predicate answers `True` where the class rules out a NaN and an infinity
  even though the format admits them.  Both selects and both `pow` calls gone
  from the lowered `FP16` output, with `test_lowered_roundtrip.py` still
  bit-exact across all fourteen formats — that suite is the proof, not the token
  count, and its inputs include a NaN, both infinities and both zeros.
- [x] **3 — the specials guards.**  The remaining emitter sites below, plus the
  `Cast` round-trip's NaN-aware equality disjunct.  The lowered `FP16` rounding
  is down to **2 asserts, 0 `isfinite`, 0 `pow`** — the two survivors are bound
  checks.  Each site is tested as a pair of programs differing only in a branch;
  an unsound version that drops every guard is caught by 20 tests, two of them
  compile-and-run differentials.
- [x] **4 — the transforms.**  `float_to_fixed` drops its `isnan` / `isinf` /
  `== 0` branches per class, and `unfold_overflow` drops the finiteness test in
  front of the early check and a specials branch.  Both compute the analysis
  themselves when the caller does not supply one.

  It needs **concrete argument types**: an unmonomorphized parameter is a type
  variable and carries no class, so every branch stays.  Two cases where it pays
  — an integer-typed operand, and a program that has already tested — and the
  standard `FP32`-source pipeline is not one of them, as predicted.  Dropping a
  branch is less forgiving than dropping an assertion: nothing checks it at
  runtime, so the equivalence tests compare the rewritten program against the
  reference in the interpreter, bit-exactly.
- [ ] **5 — full suites, and the stale listings.**  The emitted-program excerpt
  in this file and in [native-lowering-roadmap.md](native-lowering-roadmap.md)
  both predate phases 2–3; refresh them once, at the end.

### Found along the way, unrelated and pre-existing

Both reproduce at the commit before this work, so neither is a regression, and
neither is a value-class problem — recorded here only because that is where they
turned up.

- **`unfold_overflow` emits an unconstructible context.** A source with a
  `nan_value` / `inf_value` substitute has it written into the rewritten
  program as a *numeric literal*, and `MPFixedContext` requires a `Float` —
  so `MPFixedContext(-4, inf_value=7)` raises and the rewritten program cannot
  be evaluated at all.
- **An integer-typed source cannot be compiled.** `with fp.FP16: round(x)` over
  a `SINT32` argument lowers cleanly but crashes format inference on the way to
  C++: `AbstractFormat.format()` builds an `MPBFixedFormat` whose `pos_maxval`
  of `1024` is unrepresentable at the chosen `nmin`.

### Where the results are consumed

Measured on the lowered `FP16` rounding, which emits **4 asserts, 4
`isfinite`, 2 `std::pow`** today:

| site | drops |
|---|---|
| `_ldexp_exponent` (`emitter.py`) | 2 `isfinite ? ldexp : pow` selects, 2 `pow` calls |
| `_undefined_guard` | 2 `assert(isfinite(...))` |
| `_guard_float_to_integer` | the assert before a float→int cast |
| `_bound_test`'s `!isfinite(operand) ||` exemption | a disjunct |
| `_assert_fixed_exact`'s specials assert | one assert per `fp.cast` |
| `_emit_ieee_min_max`'s NaN select | the select, and both operand bindings |

Phases 2–3 should leave **2 asserts, 0 `isfinite`, 0 `pow`**; the two survivors
are `fabs(_tmp1) <= 1024` bound checks, which are magnitude facts classes cannot
touch.

The **transforms are weaker consumers than they look.**  `float_to_fixed`
*creates* the `isnan`/`isinf`/`== 0` ladder because `logb` is undefined on all
three — it cannot be removed, and it is what establishes the refinement.
`unfold_overflow`'s `check_finite` and its NaN/Inf branches could be skipped for
an excluded class, but on the standard pipeline the operand is a parameter at
top, so it pays nothing.  `round_elim` and `format_infer` are not consumers at
all: both ask magnitude questions.

An expression key is an identity, so **any rewrite invalidates the result** — a
transform must query the AST it was handed, before rewriting it.

## Decisions the implementation settled

- **Rounding is not modelled per context.**  Any operation that rounds yields a
  value its context represents, so the result class under a concrete non-`REAL`
  context is just what the context can hold — probed by rounding a NaN and both
  infinities and seeing what comes back (`representable_classes`).  Under `REAL`
  the exact table stands.  Two cases instead of an overflow/underflow/substitute
  model per context, and it is precise for an integer context for free.
- **That default is unsound for an operation that does not round.**  `min`
  returns an operand untouched, so it can carry a NaN out of a context with no
  NaN; likewise `fst`/`snd`, and a `Call` whose result the callee produced.
  Those four are handled explicitly.
- **Executions that error are not described.**  An operation handed a value its
  context refuses has no result, so it contributes no class.  That is what makes
  a guard removable; a consumer drops one only where no class reaching the
  operation is refused, so the abort survives wherever FPy has no answer.
- **Sign stays out.**  Splitting `±Inf` and `±0` would let `signbit` refine and
  would let `unfold_overflow` decide its sign-choice branches statically, but the
  chain above needs none of it and sixteen elements becomes sixty-four.
- **Loops need no widening.**  Each phi starts at its pre-loop class and only
  joins, so a height-4 lattice settles; the bound is four rounds per phi, and
  exceeding it drops the phis to top rather than spinning.

Still untaught: magnitudes (`x > 1` says nothing), `assert` statements as
refinements, a bool variable holding a test's result, structural classes for a
list or tuple element, and the class of a numeric free variable.
