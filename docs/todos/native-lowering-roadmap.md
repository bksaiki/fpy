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
  `trunc`/`floor`/`ceil`/`round`/`nearbyint`; the other three are composed from
  those (see gap 2).
- **A context's unrepresentable values compile to assertions.** The bound
  becomes `assert(std::fabs(r) <= B)` — or a pair of comparisons where the two
  bounds are asymmetric — and an operand the format has no result for becomes
  `assert(std::isfinite(v))`. Derived from the context's own flags, so a format
  that admits NaN gets no guard, and from the *value*, so neither does an operand
  a branch has already ruled out (gap 6).
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

### 2. The mode table is complete

**Closed**, and it was three modes short rather than one. `RAZ` is
`copysign(ceil(fabs(x)), x)` — the same spelling
[mpfx](https://github.com/bksaiki/mpfx)'s `round_to_integral` uses.

`RTO` and `RTE` ask for the parity of the *result*, which no libm function
reports, so each is built from what is already there:

- `RTO` — `o = floor(x * 0.5) * 2` is the even integer at or below `x`, and `o +
  1` is the odd neighbour, which serves both `(o, o+1)` and `(o+1, o+2)`.
- `RTE` — halve, round to nearest-even, double. `fabs` then separates the one
  case that must not move: an odd integer, already exact and a full step from
  that even neighbour.

`RTE` is mpfx's spelling, with `std::nearbyint` standing in for C23 `roundeven`.
That is the same substitution `RNE` already makes here — no compiler is required
to have the builtin, and adding it as an `fpy::` helper would break
`test_needs_no_support_library`. It carries the same cost: `nearbyint` follows the
dynamic rounding mode, so `RTE` inherits `RNE`'s `FE_TONEAREST` precondition and
is refused inside a scope that set another mode. `RTO` has no such dependency.

Every step of all eight is exact. Verified bit-for-bit against the interpreter
in `test_round_fixed_bound.py`, and the three new modes are in
`test_lowered_roundtrip.py`'s target list, which now covers fourteen formats.

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

The predicate for "the cast *is* the rounding" is `target.is_native_ctx`,
introduced here: a format carries no overflow rule, so a `-128..127` context
under `ASSERT` is format-equal to `int8_t` and still needs the assertion.

The libm mapping had a second, subtler hole in the same area: the interpreter
*raises* where `std::trunc` returns a NaN, and only an unread branch reconciled
them. Closed by gap 6.

### 6. Value classes read the branches

**Closed.** A four-atom lattice — `{NaN, Inf, Zero, Finite}` — refined at every
branch that tests a value's kind, in `fpy2/analysis/value_class.py`. It answers
the question the guards above were asking and the *format* could not: a format
says whether some value in it is a NaN, not whether this one is, and it
structurally cannot say **not zero**.

The lowered `FP16` rounding went from **4 assertions, 4 `isfinite`, 2 `std::pow`**
to **2, 0, 0** — the survivors are the two bound checks, which are magnitude
facts. `std::ldexp` no longer needs its `isfinite ? … : pow` fallback, and the
libm mapping's specials side-condition is discharged rather than assumed.

The transforms read it too: `float_to_fixed` drops an `isnan`/`isinf`/`== 0`
branch and `unfold_overflow` its finiteness test where the operand cannot be that
kind of value. That needs concrete argument types and pays nothing on the
standard `FP32`-source pipeline, where the operand is a parameter at top.

### 4. Backend cleanups

- `(2 ** n) * x` fails for an `n` typed `SINT64` or `INTEGER`: `cannot implicitly
  cast int64_t to double: conversion is lossy`. `SINT8`/`SINT16`/`SINT32` work,
  since those convert exactly. The message does not suggest the fix.
- Cosmetic: redundant `static_cast<double>` on integer literals, and a doubled
  `static_cast<double>(static_cast<double>(2))`.

### 5. A recipe

`monomorphize → unfold_overflow → float_to_fixed → rescale_fixed` is the
sequence, verified bit-for-bit against the interpreter across fourteen formats in
`tests/unit/backend/cpp/test_lowered_roundtrip.py`. `simplify` composes with it
but is not in that check. It deserves one entry point rather than a comment in a
sandbox.

Unrelated to lowering, but found here: `fp.round_at(x, n)` raises `IndexError:
tuple index out of range` from `fpy2/analysis/type_infer.py:440` — a crash in
type inference where a diagnostic belongs.

## Order of work

Gaps 2, 3 and 6 are done, and with them the non-finite integer conversion: a
float-to-integer cast asserts `std::isfinite` first where it can arrive, on the
native integer path as well. What is left, cheapest first:

1. **Backend cleanups and a recipe** — gaps 4 and 5, both small, and the recipe
   is what makes the path usable by someone who did not write it.
2. **An `FP64` source** — the largest, and gated on the *numeric* half of
   inference rather than on the backend.

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
