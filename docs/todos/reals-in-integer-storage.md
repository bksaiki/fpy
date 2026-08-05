# A real stored in an integer

The C++ backend narrows a `RealType` value to an integer type when its bound says
every value is a small integer — `acc = 0.0` becomes `uint8_t`, and a list of
integral constants becomes `std::vector<uint8_t>`. An integer holds neither a
**signed zero** nor a **NaN**, so this used to be the root cause of several wrong
answers.

It no longer is. The narrowing is now gated on the bound genuinely excluding those
values, and the gate is in the analysis rather than the backend.

## The wrong answers this caused, re-measured

```python
def f(y: fp.Real) -> fp.Real:
    return 0.0 * y
```

| `y` | interpreter | compiled |
|---|---|---|
| `inf` | `nan` | `nan` |
| `nan` | `nan` | `nan` |
| `-0.0` | `-0.0` | `-0.0` |

All three agree, and the emitted signature is `double f(double y)`, not
`uint8_t f(double y)`. Two mechanisms did it, one per format domain:

**`SetFormat` states which zero it holds.** Its element type is
`SetValue = Fraction | NegZero`. A `Fraction` has no signed zero, so `Fraction(0)`
*is* `+0.0` and the sentinel carries `-0.0`. Arithmetic over the domain
(`format_infer._SET_BINOPS`) follows IEEE 754 §6.3, checked against the
interpreter: a sum is `-0.0` only when both addends are; `a - b` is `a + (-b)`; a
zero product takes the XOR of the operand signs. So `neg({+0.0})` is exactly
`{-0.0}` and `{+0.0} * {-1}` is exactly `{-0.0}`.

**`AbstractFormat` carries signed-zero membership.** `has_neg_zero` joins
`has_pos_inf` / `has_neg_inf` / `has_nan` as a fourth membership implication in
containment. Since the C++ ladder is built by abstracting each type's own
`Format`, the integer rungs report `has_neg_zero=False` and reject any bound that
carries one — which is what keeps such a value off an integer storage. The
operators derive it: `add` conjoins, `sub` takes the left operand, `mul`
disjoins, `neg` carries over, `abs` clears.

`Format` gained `enable_neg_zero` (on `MPFixedFormat` / `MPBFixedFormat`) so the
flag survives `AbstractFormat.format()` and back. Without it every
integer-valued bound materialized on the way to storage selection acquired a
signed zero and lost its integer storage — `int8 + int8` became `float`.

## Retired: the carve-out on `MPFixedFormat`

One divergence outlived the change above:

```python
def f(x: fp.Real) -> fp.Real:      # x : INTEGER
    with fp.INTEGER:
        return -x
```

For `x == 0` the interpreter returned `-0.0` and compiled code returned `0`.
`AbstractFormat.from_format` took every `MPFixedFormat` as having no signed zero
whatever it said, because believing FPy's `INTEGER` cost integer storage
everywhere: measured, it retyped every `range` counter as `float` and diverged
the loop fixpoint in `test_while{5,6,7}_rounded` to `REAL_FORMAT`, which no C++
type can hold.

Fixed by the first of the two candidates this document listed — the flag now
reaches the *contexts*, and `INTEGER` declines the signed zero at the source:

```python
INTEGER = MPFixedContext(-1, RM.RTZ, enable_neg_zero=False)
```

An `int` has a single zero, so this is what `INTEGER` always meant. The
`MPFixedContext` underneath still *has* a signed zero, and a caller wanting one
can ask; `INTEGER` declines. That collapsed `_INTEGER_FORMAT` — which existed
only to say "a count has one zero" — back to `INTEGER.format()`, and let
`from_format` believe every probe. A format that does claim a signed zero is now
refused every integer rung of the ladder, correctly.

Two things fell out. `FixedContext` no longer needs its `_round_at` override, since
`enable_neg_zero=False` says the same thing declaratively (*"two's complement has
a single encoding of zero"*). And `MPFixedFormat.to_fractional_ordinal` had a
latent bug: its fast path was gated on `representable_in`, but the interpolation
below it needs the stricter condition that the value lie *strictly between* two
grid points. A `-0.0` under `enable_neg_zero=False` is on the grid yet
unrepresentable, so it fell through to a zero-width interval and divided by zero.

A *literal* zero was always fine: its bound is a `SetFormat`, which states the
sign and so reaches a float storage.

**The `S64` fallback** in `choose_storage_scalar` — taken for an unbounded
`MPFixedFormat` after the ladder search fails — used to skip the membership
checks, so a format carrying a NaN could land in `int64_t`. It now re-checks all
four flags.

## Retired: "the fix has to be uniform"

This document used to argue:

> **A `RealType` value is stored in a float. Never in an integer.**

on the grounds that half-measures produce *disagreements* — some zeros becoming
`float` while others stayed `uint8_t`, then meeting:

```
test_ife2:      cannot implicitly cast `float` to `uint8_t`: conversion is lossy
test_list_set1: storing a `float` into a slot of `uint8_t` would narrow it
```

That argument applied to gating the narrowing on a *syntactic* condition (does the
set contain a zero). Measured then: refusing an integer storage for any
zero-containing set cost 9 corpus functions; restricting it to exactly `{0}` cost
7. Gating on the *value* instead — is this zero negative, can this bound hold a
NaN — produces no disagreement, because the two zeros are now different values and
every consumer asks the same question about them. The corpus is unchanged across
the whole change (54/54 compiling, byte-identical emitted C++ except one
improvement) and the differential harness is clean at 111/116.

So the blanket rule is not needed. Keep it in mind only if the per-value gate
turns out to have a hole this document has not found.

## Whether to keep the narrowing at all

Still an open question, on size rather than correctness. Measured earlier: 21 of
the corpus's 112 list element types are value-narrowed reals (all in `test_list*`,
`test_range*`, `test_list_comp*`), plus scalars like `uint8_t acc` inside FP64
functions. Dropping it would trade smaller objects for emitted code that says
`double` wherever FPy says real.

*Integer*-typed FPy values keep integer storage regardless; this is only about
reals. `range(...)` needs an integer list and must stay one — that is what an
early measured attempt broke.
