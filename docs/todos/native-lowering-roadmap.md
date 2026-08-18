# Roadmap: float rounding to native C++ integer code

## Goal

Rewrite a program whose roundings are floating-point into one whose roundings
are integer, and compile that to C++ that needs no support library — no MPFR,
no soft-float, no `fpy::` runtime.

## Where we are

**Reached, for an `FP32` source.** Four operators, each one idea, plus an
optional fifth:

```
monomorphize      pin the argument format
unfold_special    NaN / infinity / zero -> branches on the operand   (optional)
unfold_overflow   IEEEContext      -> MPSFloatContext + bound checks
float_to_fixed    unbounded float  -> MPFixedContext at a position from logb
rescale_fixed     MPFixedContext   -> position zero, integer values
```

The result compiles to plain C++ and is bit-exact against the interpreter across
fourteen target formats — all eight FPy rounding modes, the MX family, a
saturating `IEEEContext`, and an `EFloatNanKind.NEG_ZERO` format. It
needs no support library, and now unconditionally: the backend emits no support
code at all, so `CPP_HELPERS` is empty. Pinned by
`tests/unit/backend/cpp/test_lowered_roundtrip.py`.

```c++
double f(float x) {
    double y{};
    if ((static_cast<double>(x) >= static_cast<double>(65536))) {
        y = std::numeric_limits<float>::infinity();
    } else if ((static_cast<double>(x) <= static_cast<double>(-65536))) {
        y = (-std::numeric_limits<float>::infinity());
    } else {
        double t{};
        if (std::isnan(x)) {
            t = std::numeric_limits<float>::quiet_NaN();
        } else if (std::isinf(x)) {
            t = (std::signbit(x) ? (-std::numeric_limits<float>::infinity()) : std::numeric_limits<float>::infinity());
        } else if ((x == static_cast<float>(0))) {
            t = (std::signbit(x) ? -0.0 : static_cast<float>(0));
        } else {
            float e = std::logb(x);
            if ((e < static_cast<float>(-14))) {
                double _t = (static_cast<double>(16777216) * static_cast<double>(x));
                float _tmp1 = std::nearbyint(_t);
                assert((std::fabs(_tmp1) <= 1024) && "fpy: overflow occurred so rounding is undefined");
                float _t6 = _tmp1;
                t = (5.960464477539063e-08 * _t6);
            } else {
                float exp = (e - static_cast<float>(10));
                auto&& _tmp2 = (-exp);
                double _t7 = std::ldexp(static_cast<double>(x), static_cast<int>(_tmp2));
                float _tmp3 = std::nearbyint(_t7);
                assert((std::fabs(_tmp3) <= 2048) && "fpy: overflow occurred so rounding is undefined");
                float _t8 = _tmp3;
                t = std::ldexp(static_cast<double>(_t8), static_cast<int>(exp));
            }
        }
        if ((t > static_cast<double>(65504))) {
            y = std::numeric_limits<float>::infinity();
        } else if ((t < static_cast<double>(-65504))) {
            y = (-std::numeric_limits<float>::infinity());
        } else {
            y = t;
        }
    }
    return y;
}
```

## What the path rests on

Three ideas, each recorded where it was learned:

- **Integer rounding needs no integer type.** `std::trunc` and friends are
  `double -> double`, so the rounding stays in a float type. That keeps the
  signed zero, needs no integer wide enough for the value, and covers all eight
  FPy modes. Five are one libm call — `RTZ`/`RTN`/`RTP`/`RNA`/`RNE` as
  `trunc`/`floor`/`ceil`/`round`/`nearbyint`; `RAZ`, `RTO` and `RTE` are
  composed from those.
- **A context's unrepresentable values compile to assertions.** The bound
  becomes `assert(std::fabs(r) <= B)` — or a pair of comparisons where the two
  bounds are asymmetric — and an operand the format has no result for becomes
  `assert(std::isfinite(v))`. Derived from the context's own flags, so a format
  that admits NaN gets no guard — and from the *value*, per
  `fpy2/analysis/value_class.py`, so neither does an operand a branch has already
  ruled out.
- **A scale must be `ldexp`, not `pow`.** `std::pow(2, n)` is not required to be
  exact — C11 F.10 requires correct rounding of no math function and IEEE 754
  only *recommends* it for `exp2`. On this platform it happens to be exact for
  all 2098 integral `n` in double's range, so nothing observably broke; the
  change is right on the standard, not on the measurement.

## The gaps that remain

### 1. An `FP64` source has no storage

The blocker for the format that matters most. Storage selection fails on the
scale-in, because `exp` and the value it scales are *correlated* and the domain
represents them independently. An `FP32` source works only because the same
products stay inside `F64`'s `2^1024`.

**Two path-sensitive facts are enough** — measured, by pinning the argument's
declared context to what each branch guarantees and running the real pipeline:

| refinement available | result |
|---|---|
| none | fails on `y` |
| `\|x\| < 65536` only | fails on `_t7` |
| `\|x\| >= 2**-14` only | fails on `y` |
| **both** | **compiles** |

They fix different fields. `\|x\| < 65536`, from the else arm of `early_check`,
is a forward refinement against a constant and fixes the **bound**. `\|x\| >=
2**-14`, from the else arm of `e < -14`, needs the backward `logb` transfer
(`logb(x) ∈ [lo, hi] ⟹ \|x\| ∈ [2^lo, 2^(hi+1))`) and fixes the **digit
position** — which is what survives once the lower clamp has handled the bound,
and which a bound-only refinement leaves at `2^-1090`.

So this is path sensitivity over magnitudes, not a new abstract domain; the
symbolic design in
[symbolic-exponent-inference.md](symbolic-exponent-inference.md) would derive a
*tighter* answer and retire the asserted bound, but is not what unblocks the
format. That doc also records the annotation approach, which was tried and does
not work.

### 2. Backend cleanups

- `(2 ** n) * x` fails for an `n` typed `SINT64` or `INTEGER`: `cannot implicitly
  cast int64_t to double: conversion is lossy`. `SINT8`/`SINT16`/`SINT32` work,
  since those convert exactly. The message does not suggest the fix.
- Cosmetic: redundant `static_cast<double>` on integer literals, and a doubled
  `static_cast<double>(static_cast<double>(2))`.

### 3. A recipe

`monomorphize → unfold_overflow → float_to_fixed → rescale_fixed` is the
sequence, verified bit-for-bit against the interpreter across fourteen formats in
`tests/unit/backend/cpp/test_lowered_roundtrip.py`. `simplify` composes with it
but is not in that check. It deserves one entry point rather than a comment in a
sandbox.

`unfold_special` composes in front of `unfold_overflow` and is worth including:
it states the specials once at the outside, so `float_to_fixed` emits no ladder of
its own (value classes read the branches) and `logb` hoists to a single call. Same
instruction count, one ladder instead of two nested inside the rounding. Not in
the roundtrip check either.

Unrelated to lowering, but found along this path — each reproduces well before
the change that turned it up, so none is a regression:

- `fp.round_at(x, n)` raises `IndexError: tuple index out of range` from
  `fpy2/analysis/type_infer.py:440` — a crash in type inference where a
  diagnostic belongs.
- **`unfold_overflow` emits an unconstructible context.** A source with a
  `nan_value` / `inf_value` substitute has it written into the rewritten program
  as a *numeric literal*, and `MPFixedContext` requires a `Float` — so
  `MPFixedContext(-4, inf_value=7)` raises and the rewritten program cannot be
  evaluated at all.
- **An integer-typed source cannot be compiled.** `with fp.FP16: round(x)` over a
  `SINT32` argument lowers cleanly but crashes format inference on the way to
  C++: `AbstractFormat.format()` builds an `MPBFixedFormat` whose `pos_maxval` of
  `1024` is unrepresentable at the chosen `nmin`.

## Order of work

The mode table, the `Round`/`Cast` context checks and the value-class analysis
are done and their sections retired; what they settled is recorded under *What
the path rests on*. What is left, cheapest first:

1. **An `FP64` source** — two path-sensitive facts, as gap 1 measures. The
   traversal is the work: `FormatInfer` is flow-insensitive by construction,
   though `fpy2/analysis/value_class.py` has the refinement pattern already, and
   in SSA a definition inside a refined arm has one path to it, so the result
   still keys per definition as storage selection expects.
2. **Backend cleanups and a recipe** — gaps 2 and 3, both small.

## Open questions

- **Where should the integer width come from?** Answered for the composed path:
  `early_check` bounds the operand by `infval`, so the rescaled integer fits
  `pmax + 1` bits — 12 for `FP16`, checked. What remains is that the transform
  *asserts* this rather than the analysis deriving it.
- **Is the scaled form the right final shape?** It costs two scalings per
  rounding. The alternative — reinterpreting the float's bits and shifting the
  significand — is what a hand-written soft-float does, and would remove
  `ldexp`, `logb`, and the float round-trip entirely. A larger rewrite, and
  probably the step after this roadmap.
