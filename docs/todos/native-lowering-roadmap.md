# Roadmap: float rounding to native C++ integer code

## Goal

Rewrite a program whose roundings are floating-point into one whose roundings
are integer, and compile that to C++ that needs no support library — no MPFR,
no soft-float, no `fpy::` runtime.

## Where we are

**Reached, for an `FP32` source.** Three operators, each one idea:

```
monomorphize      pin the argument format
unfold_overflow   IEEEContext      -> MPSFloatContext + bound checks
float_to_fixed    unbounded float  -> MPFixedContext at a position from logb
rescale_fixed     MPFixedContext   -> position zero, integer values
```

The result compiles to plain C++ and is bit-exact against the interpreter across
eleven target formats — every IEEE rounding mode with a libm counterpart, the MX
family, a saturating `IEEEContext`, and an `EFloatNanKind.NEG_ZERO` format. It
needs no support library: zero `fpy::` references, and it still compiles with the
helper namespace stripped. Pinned by
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
                assert(std::isfinite(_t) && "fpy: rounding is undefined for this value");
                float _tmp1 = std::nearbyint(_t);
                assert((std::fabs(_tmp1) <= 1024) && "fpy: overflow occurred so rounding is undefined");
                float _t7 = _tmp1;
                t = (5.960464477539063e-08 * _t7);
            } else {
                float exp = (e - static_cast<float>(10));
                auto&& _tmp2 = (-exp);
                auto&& _tmp3 = static_cast<double>(x);
                double _t8 = (std::isfinite(_tmp2) ? std::ldexp(_tmp3, static_cast<int>(_tmp2)) : std::pow(2.0, _tmp2) * _tmp3);
                assert(std::isfinite(_t8) && "fpy: rounding is undefined for this value");
                float _tmp4 = std::nearbyint(_t8);
                assert((std::fabs(_tmp4) <= 2048) && "fpy: overflow occurred so rounding is undefined");
                float _t9 = _tmp4;
                auto&& _tmp5 = static_cast<double>(_t9);
                t = (std::isfinite(exp) ? std::ldexp(_tmp5, static_cast<int>(exp)) : std::pow(2.0, exp) * _tmp5);
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
  signed zero, needs no integer wide enough for the value, and covers more than
  just `RTZ`. Five of FPy's eight modes have a libm function — `RTZ`/`RTN`/`RTP`/
  `RNA`/`RNE` as `trunc`/`floor`/`ceil`/`round`/`nearbyint`; `RAZ`, `RTO` and
  `RTE` have none and are declined (see gap 2).
- **A context's unrepresentable values compile to assertions.** The bound
  becomes `assert(std::fabs(r) <= B)` — or a pair of comparisons where the two
  bounds are asymmetric — and an operand the format has no result for becomes
  `assert(std::isfinite(v))`. Derived from the context's own flags, so a format
  that admits NaN gets no guard.
- **A scale must be `ldexp`, not `pow`.** `std::pow(2, n)` is not required to be
  exact — C11 F.10 requires correct rounding of no math function and IEEE 754
  only *recommends* it for `exp2`. On this platform it happens to be exact for
  all 2098 integral `n` in double's range, so nothing observably broke; the
  change is right on the standard, not on the measurement.

## The gaps that remain

### 1. An `FP64` source has no storage

The blocker for the format that matters most. Storage selection fails on the
scale-in and scale-out, because `exp` and the value it scales are *correlated*
and the domain represents them independently:

| | inferred | true |
|---|---|---|
| `(2 ** -exp) * x` | `2^2107` | `[2^10, 2^11)` |
| `(2 ** exp) * _t8` | `2^1034` | `65504` |

`F64` holds `2^1024`, so both exceed it. An `FP32` source works only because
the same products land at `2^286` and `2^138`.

Not fixable by annotation — see
[symbolic-exponent-inference.md](symbolic-exponent-inference.md), which records
both the designs that would work and the one that was tried and does not.

### 2. `RAZ` has no libm function

`copysign(std::ceil(std::fabs(x)), x)` is the two-operation spelling. The only
rounding mode the lowering still declines, and the last entry missing from the
mode table.

### 3. `Round` / `Cast` checked storage, not the context

**Closed.** `fp.cast(v)` tested *storage* exactness only
(`assert(arg == tmp)`), so under a context bounded at 1024 whose representable
values are the integers, `cast(2048.0)` and `cast(0.5)` both raised in the
interpreter and passed silently in C++. `_assert_fixed_exact` now emits the
specials, representability and bound checks for a fixed-point context, and
refuses a non-zero position rather than assuming it; the same-storage shortcut no
longer skips them. Verified against the interpreter per value, 88/88, in
`tests/unit/backend/cpp/test_cast_exactness.py`.

The mirror of it on `Round` is closed too, and was worse than expected: a
fixed-point context reaching a bare `static_cast` dropped its bound *and* its
overflow rule. A context bounded at 100 returned 120 where `ASSERT` raises,
`SATURATE` says 100 and `WRAP` says -81 — on **both** storage paths, not only the
integer one. `_emit_integral_round` now either lowers a fixed-point context
faithfully or refuses it, and a non-`ASSERT` rule is refused rather than dropped.
Verified value-for-value against the interpreter in
`tests/unit/backend/cpp/test_round_fixed_bound.py`.

The predicate for "the cast *is* the rounding" is `target.is_native_ctx`, the
same one gap 4 introduced, for the same reason: a format carries no overflow
rule, so a `-128..127` context under `ASSERT` is format-equal to `int8_t` and
still needs the assertion.

One item in this area stays open — `static_cast` to an integer type is undefined
for a non-finite operand, on the native integer path that emits no assertions at
all. Recorded in [backend-cpp.md](backend-cpp.md).

The libm mapping has a second, subtler hole in the same area: the interpreter
*raises* where `std::trunc` returns a NaN, and today only an unread branch
reconciles them.  [value-class-analysis.md](value-class-analysis.md) closes that
one, and four emitted guards with it.

### 4. Backend cleanups

- `(2 ** n) * x` fails for an `n` typed `SINT64` or `INTEGER`: `cannot implicitly
  cast int64_t to double: conversion is lossy`. `SINT8`/`SINT16`/`SINT32` work,
  since those convert exactly. The message does not suggest the fix.
- `fpy::min` / `fpy::max` are support-library calls on the float path where the
  integer path uses `std::`. The lowered program happens to use neither, so the
  no-library claim holds today, but a program that does would break it.
- Cosmetic: redundant `static_cast<double>` on integer literals, and a doubled
  `static_cast<double>(static_cast<double>(2))`.

### 5. A recipe

`monomorphize → unfold_overflow → float_to_fixed → rescale_fixed` is the
sequence, verified bit-for-bit against the interpreter across eleven formats in
`tests/unit/backend/cpp/test_lowered_roundtrip.py`. `simplify` composes with it
but is not in that check. It deserves one entry point rather than a comment in a
sandbox.

Unrelated to lowering, but found here: `fp.round_at(x, n)` raises `IndexError:
tuple index out of range` from `fpy2/analysis/type_infer.py:440` — a crash in
type inference where a diagnostic belongs.

## Order of work

Gap 3 is done. What is left, cheapest first:

1. **The non-finite integer conversion** — one assertion, undefined behavior
   today; the only thing holding it is measuring how much emitted output changes.
   Recorded in [backend-cpp.md](backend-cpp.md).
2. **[Value classes](value-class-analysis.md)** — a four-atom lattice, refined
   at branches.  Removes two runtime branches and two assertions from every
   lowered rounding and discharges the libm mapping's last side-condition.
3. **`RAZ`** — small, completes the mode table, and needs a branch rather than a
   function.
4. **An `FP64` source** — the largest, and gated on the *numeric* half of
   inference rather than on the backend.
5. **Backend cleanups and a recipe.**

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
