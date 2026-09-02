# Roadmap: float rounding to native C++ integer code

## Goal

Rewrite a program whose roundings are floating-point into one whose roundings
are integer, and compile that to C++ that needs no support library — no MPFR,
no soft-float, no `fpy::` runtime.

## Where we are

**Reached, for `FP32` *and* `FP64` sources.** Four operators, each one idea, plus
an optional fifth:

```
monomorphize      pin the argument format
unfold_special    NaN / infinity / zero -> branches on the operand   (optional)
unfold_overflow   IEEEContext      -> MPSFloatContext + bound checks
float_to_fixed    unbounded float  -> MPFixedContext at a position from logb
rescale_fixed     MPFixedContext   -> position zero, integer values
```

The result compiles to plain C++ and is bit-exact against the interpreter for
both sources across fourteen target formats — all eight FPy rounding modes, the MX family, a
saturating `IEEEContext`, and an `EFloatNanKind.NEG_ZERO` format. It
needs no support library, and now unconditionally: the backend emits no support
code at all, so `CPP_HELPERS` is empty. Pinned by
`tests/unit/backend/cpp/test_lowered_roundtrip.py`.

```c++
float f(float x) {
    float y{};
    if ((static_cast<double>(x) >= static_cast<double>(65536))) {
        y = std::numeric_limits<float>::infinity();
    } else if ((static_cast<double>(x) <= static_cast<double>(-65536))) {
        y = (-std::numeric_limits<float>::infinity());
    } else {
        float t{};
        if (std::isnan(x)) {
            t = std::numeric_limits<float>::quiet_NaN();
        } else if (std::isinf(x)) {
            t = (std::signbit(x) ? (-std::numeric_limits<float>::infinity()) : std::numeric_limits<float>::infinity());
        } else if ((x == static_cast<float>(0))) {
            t = (std::signbit(x) ? -0.0 : static_cast<float>(0));
        } else {
            float e = std::logb(x);
            if ((e < static_cast<float>(-14))) {
                float _t = static_cast<float>((static_cast<double>(16777216) * static_cast<double>(x)));
                float _tmp1 = std::nearbyint(_t);
                assert((std::fabs(_tmp1) <= 1024) && "fpy: overflow occurred so rounding is undefined");
                float _t6 = _tmp1;
                t = (5.960464477539063e-08 * _t6);
            } else {
                auto&& _tmp2 = (e - static_cast<float>(10));
                auto&& _tmp3 = static_cast<float>(static_cast<float>(-24));
                float exp = ((_tmp2 < _tmp3 || (_tmp2 == _tmp3 && std::signbit(_tmp2))) ? _tmp3 : _tmp2);
                auto&& _tmp4 = (-exp);
                float _t7 = std::ldexp(x, static_cast<int>(_tmp4));
                float _tmp5 = std::nearbyint(_t7);
                assert((std::fabs(_tmp5) <= 2048) && "fpy: overflow occurred so rounding is undefined");
                float _t8 = _tmp5;
                t = std::ldexp(_t8, static_cast<int>(exp));
            }
        }
        if ((t > static_cast<float>(65504))) {
            y = std::numeric_limits<float>::infinity();
        } else if ((static_cast<double>(t) < static_cast<double>(-65504))) {
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
- **A width comes from a branch, not only from a format.** The scale-in
  `(2 ** -exp) * x` is in `[2^10, 2^11)` for any source, because `exp` comes from
  `logb(x)` — but the two are *correlated* and the domain describes them
  separately, so the product was inferred at `2^2108` for an `FP64` source. Two
  facts the program already states close it: `|x| < infval` from `early_check`
  fixes the **bound**, and `logb(x) >= emin` from the subnormal branch fixes the
  **digit position**, the half a bound alone never reaches. Both are ordinary
  path sensitivity over magnitudes (`FormatInfer._refined`), not a new domain.
- **A scale must be `ldexp`, not `pow`.** `std::pow(2, n)` is not required to be
  exact — C11 F.10 requires correct rounding of no math function and IEEE 754
  only *recommends* it for `exp2`. On this platform it happens to be exact for
  all 2098 integral `n` in double's range, so nothing observably broke; the
  change is right on the standard, not on the measurement.

## The gaps that remain

Neither blocks anything, so the open questions below are the interesting work.

### 1. Cosmetics

Redundant `static_cast<double>` on integer literals, and a doubled
`static_cast<double>(static_cast<double>(2))`.

### 2. A recipe

`monomorphize → unfold_special → unfold_overflow → float_to_fixed →
rescale_fixed → simplify` is the sequence, verified composed and bit-for-bit
against the interpreter from both an `FP32` and an `FP64` source across fourteen
targets — `_lower` in `tests/unit/backend/cpp/test_lowered_roundtrip.py`. Each of
`unfold_special` and `simplify` changes the result there, so the coverage is real
and not nominal.

`unfold_special` belongs in front of `unfold_overflow`: it states the specials
once at the outside, so `float_to_fixed` emits no ladder of its own (value classes
read the branches) and `logb` is computed once. `unfold_neg_zero` is *not* in the
sequence — nothing reaches it, since the zero branch has already said what each
zero rounds to.

**The cpp backend has an entry point**, `CppCompiler(unfold=...)`: it finds its
own sites and runs the sequence, with a `split_round` step in front for
arithmetic the op table cannot spell. See *Recovering from an unsupported
rounding* in [backend-cpp.md](backend-cpp.md).

What is still missing is a **parameterized** recipe for a caller lowering by
hand — a function, a location and a target descriptor. Composition used to have
no way to carry a location, and that is fixed: `where` takes a cursor the
strategies forward across each step, so one location aims the whole sequence
(`_lower_at` in the same file lowers one rounding of a two-rounding program and
leaves the other alone). Only the entry point itself is left. See
[scheduling-language.md](scheduling-language.md) items 3 and 7.

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
