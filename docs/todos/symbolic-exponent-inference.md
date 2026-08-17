# Relating a value to its own exponent in format inference

Follow-up to gap 1 of [native-lowering-roadmap.md](native-lowering-roadmap.md).
Not blocking: `logb`/`pow` inference (619ed2b) took the lowered program from
unbounded to bounded, and the asserted bound (7e9e8a4) gets the widths right.
This records what it would take for the analysis to *derive* what the transform
currently asserts.

## The problem, measured

`rescale_fixed` emits, per rounding:

```python
exp = e - 10                      # e = logb(x)
with fp.REAL:
    _t7 = (2 ** -exp) * x         # scale in
_t8 = fp.round(_t7)               # round at position zero
```

`(2 ** -exp) * x` is in `[2^10, 2^11)` — always, for any input format —
because `exp` is derived from `logb(x)`. Instrumenting 200k FP32 inputs:

| | actual | inferred (before the assert) |
|---|---|---|
| normal branch `(2**-exp)*x` | `[1024, 2048]`, bound `2^11` | bound `2^286` |
| subnormal `2**24 * x` | `[5.1e-35, 1024]`, bound `2^10` | bound `2^151` |

Off by `2^275`. Nothing is wrong with either operand's format — each is tight
for what it knows. `logb(x)` really does range over `[-149, 127]` for FP32, so
`2 ** -exp` really does reach `2^159`, and `x` really does reach `2^127`. The
product of two tight non-relational facts is `2^286`.

What's lost is that the two are *correlated*: `2 ** -exp` is inversely
proportional to `x` by construction. That is a relation between two variables,
and the domain represents them independently.

The round *result* follows from the argument: its integer width comes from the
bound, so `2^286` needs a 287-bit integer where the truth is 12 bits — `pmax + 1`
for FP16.

## The invariant

`logb` is the operation that *names* a value's exponent, and once named the
value factors against it:

```
e = logb(x)   ⟹   x = ±m · 2^e   with m ∈ [1, 2)
```

Verified on 100k FP32 values, subnormals included: `m` observed in
`[1.0, 1.999992847442627)`. It holds for subnormals too, since `logb` is
`floor(log2 |x|)` regardless of where the value sits in the format.

That single fact is enough:

```
e   = logb(x)          x = ±m·2^e,  m ∈ [1,2)
exp = e - 10           s = e - 10
2 ** -exp              (1, 10 - e)
(2**-exp) * x          (m·1, e + (10 - e))  =  (m, 10)   →  [2^10, 2^11)
```

Exponents cancel because they stay symbolic; mantissas multiply because they
are concrete.

## What will not work

**An `[lsb, msb]` domain.** It is `AbstractFormat` in different coordinates —
`lsb = exp`, `msb + 1 ≈ log2(bound)`, `prec = msb - lsb + 1`. Re-parameterizing
adds no relational power, and the descriptions are already tight. Only
*symbolic* `lsb`/`msb` would help, which is the design below.

**Path sensitivity alone.** Worth having for other reasons — refining `x` from
`|x| < 65536`, and clearing `has_nan`/`has_inf` from the `isnan`/`isinf` guards,
would tighten every program these passes touch. It also fixes the *subnormal*
branch, whose looseness really is a branch condition (`e < -14` ⟹ `|x| < 2^-14`),
though that needs a backward transfer function for `logb`:

```
logb(x) ∈ [lo, hi]   ⟹   |x| ∈ [2^lo, 2^(hi+1))
```

the exact inverse of the forward rule already implemented. But the normal branch
has no condition to exploit, so path sensitivity leaves `2^286` untouched.

**Annotating the result with a bounded `Cast`.** Tried and abandoned; recorded so
it is not re-attempted. The idea was that since `float_to_fixed` knows `maxval`
and `expmin`, the scale-out's envelope is the *static* context
`MPBFixedContext(expmin - 1, maxval)`, and casting into it would give the result
a bounded format:

```python
with fp.REAL:
    _s = (2 ** exp) * _t8
with fp.MPBFixedContext(-25, 65504, overflow=fp.OverflowMode.ASSERT):
    t = fp.cast(_s)
```

Three findings, in increasing order of how fatal they are:

1. Putting the arithmetic *inside* the bounded context instead —
   `t = fp.cast((2 ** exp) * _t8)` — does not compile at all. `fp.cast(expr)`
   evaluates `expr` under the enclosing context, so `Pow` and `Mul` round under
   a fixed-point context, which the backend has no emission for.
2. Keeping the arithmetic under `REAL`, as above, compiles the *cast* but not the
   program: storage selection still fails, now on `_s`. **An annotation can only
   tighten a value that already has storage** — the unbounded intermediate is the
   failure, and narrowing the result afterwards is too late.
3. The cast would not even be checked. In C++ a `Cast` compiles to a
   `static_cast` plus `assert(arg == tmp)`, which tests *storage* exactness only.
   At position zero, where the emitter permits it, `cast(2048.0)` under a context
   bounded at 1024 and `cast(0.5)` under an integer context both raise in the
   interpreter and pass silently in C++; when argument and target share a storage
   type, no cast and no assert are emitted at all.

So the bound has to exist where the value is *computed*, which is what makes this
an inference problem rather than an annotation problem. Finding (3) is a separate
gap worth fixing on its own: the bound and representability checks live in
`_emit_integral_round` and should be hoisted into the general round/cast path.

## Design A — symbolic exponents

Represent a value as `m · 2^s`, with `m` a concrete interval and `s` a symbolic
linear expression over program variables.

| operation | rule |
|---|---|
| `logb(x)` | introduce `x = ±m·2^e`, `m ∈ [1,2)` — the novel one |
| `+`/`-` on exponents | symbolic linear, normalized so `(e-10) + (10-e) → 0` |
| `Pow(2, k)`, `Exp2(k)` | `(1, k)` |
| `Mul` | multiply mantissas, add exponents |
| `Round` at position zero | concretize `s` through the existing domain |

Conveniently, that is the whole set these passes emit — there is no addition of
scaled values and no loop around a rounding.

Where it degrades:

- **Adding two values** needs their exponents aligned; symbolically the
  difference is unknown, so it falls back.
- **Join** across branches with different `s` must concretize.
- **Loops** grow symbolic expressions, so widening is required.
- **Concretization** bounds `s` with the existing non-relational domain — for
  our case `s` has cancelled to the constant `10`, giving `2^11` exactly.

This is the standard loss-of-correlation problem; the textbook example is
`x - x` under interval arithmetic and the standard answer is symbolic or affine
forms (Astrée's symbolic domain, Fluctuat's affine arithmetic). Ours is the
multiplicative version.

Size: a new module of a few hundred lines as a reduced product beside
`AbstractFormat`, plus threading through a 2600-line `FormatInfer`.

## Design B — one relation, not a domain

Cheaper and probably the right size. Tag a format bound with *this value's
exponent is the variable `e`*, then teach only `Mul` and `Pow(2, ·)` to cancel
against the tag. A field on the bound and two cases in transfer functions that
already exist.

It covers the whole family these passes emit, because they always compute
`logb` and then scale by a function of it. It degrades gracefully: an
unrecognized scale falls back to today's behavior rather than being wrong.

## Relation to what shipped

`float_to_fixed` now states the operand's reach as a bound with
`OverflowMode.ASSERT` (7e9e8a4), and `rescale_fixed` folds the syntactic
cancellation `2 ** (k + c)` shifted by `2 ** -k` into the constant `2 ** c`.
That reaches widths 11 and 12 on the FP16 lowering, checked over 6120
composition cases with no assertion failures.

The difference is where the fact comes from. Today the transform asserts it and
the runtime checks it, which only helps output the transforms generated. Either
design above *derives* it, so it also applies to a hand-written program that
scales by its own exponent — and it would let the asserted bound go away, along
with the `MPBFixedContext` it reintroduced into the lowered output.  That bound
is a *claim* (`OverflowMode.ASSERT`) rather than an edge rule, which is the
distinction `test_no_overflow_behavior_survives` now pins.

`min`/`max` now tighten by ordering rather than only joining, so a clamp against
a constant bounds its result: `min(logb(x), 5)` over `FP64` reads `[-1074, 5]`
instead of `[-1074, 1023]`. That bounds the digit position, but not the values
scaled by it — `float_to_fixed` emits only an *upper* clamp, and `2 ** -exp`
needs the lower end, which comes from the subnormal branch rather than a `max`.
Adding a lower clamp is semantically free in the normal branch (`e >= emin`
already holds there) and measurably helps, though not enough on its own:

| scale-in bound | today | with a lower clamp |
|---|---|---|
| `FP32` source | `2^286` | `2^151` — fits `F64` |
| `FP64` source | `2^2107` | `2^1047` — still over `2^1024` |

An `FP64` source additionally needs `x` itself bounded, which `early_check`
states in program text and only path sensitivity reads. That is why the whole
matrix passes for an `FP32` source and fails for `FP64`.

### The class half was separable, and is done

Refining special-value membership at branches turned out to be worth separating:
it is a four-atom lattice rather than a numeric domain, it needs none of the
machinery below, and it closed a correctness gap as well as tightening bounds.
That is `fpy2/analysis/value_class.py`; see gap 6 of
[native-lowering-roadmap.md](native-lowering-roadmap.md).  What follows concerns
only the *magnitude* half, which it cannot help with.

### A second payoff for the `isnan`/`isinf` refinement

**Done**, by the class half rather than by anything here: the `ldexp` lowering
asked whether its exponent could be non-finite and answered from the exponent's
*format*, which for an exponent out of `logb` admits both specials and so cost a
runtime branch.  Reading the upstream tests instead drops it.  See gap 6 of
[native-lowering-roadmap.md](native-lowering-roadmap.md).

## Open questions

- Does the tag in design B survive `min`/`max` on the exponent? The clamp is
  gone on the composed path, but `float_to_fixed` still emits one when run
  alone.
- Should the mantissa interval be signed, or should sign be tracked separately?
  `logb` discards sign, so `x = ±m·2^e` needs the sign from elsewhere.
- Is `logb` the only introducer worth having? `frexp`-style splitting and
  `round_at` would name an exponent too.
