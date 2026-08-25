# Relating a value to its own exponent in format inference

**Not blocking, and no longer the route to anything.** The `FP64` source it was
written for is done — by ordinary path sensitivity over magnitudes plus a lower
clamp on the position, neither of which needs the machinery below (see *What will
not work*, and *What the path rests on* in
[native-lowering-roadmap.md](native-lowering-roadmap.md)).

What is left for a symbolic design is a *tighter* answer than refinement reaches
— `2^11` where it gets `2^40` — and retiring the bound the transform currently
asserts in favour of one the analysis derives. That also extends to a
hand-written program that scales by its own exponent, which no branch of ours
conditions. The simplest such program — one `logb`, rounded directly — has a
cheaper route than either design here, being one transfer rule rather than a
domain ([Design C](#design-c--one-rule-not-a-relation)); anything richer does
not. [Which design each shape needs](#which-design-each-shape-needs) sorts
them, and the shape this repo's own examples use needs an ingredient none of
the three has.

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

**~~Path sensitivity alone.~~** This was wrong, and the lower clamp is why. The
claim was that the normal branch has no condition to exploit, so refinement
leaves `2^286` untouched — true when the position was unclamped, since the
looseness lived in `2 ** -exp`. The clamp caps that side syntactically, and what
remains is `x`, which *both* branches condition. **Shipped** in `FormatInfer`
(`_refined` / `_implied` / `_implied_logb`), and an `FP64` source now compiles and
is bit-exact against the interpreter across fourteen targets.

Two facts, fixing different fields:

- `|x| < 65536`, from the else arm of `early_check` — a forward refinement
  against a constant. Fixes the **bound**.
- `|x| >= 2^-14`, from the else arm of `e < -14` — needs a backward transfer for
  `logb`, the exact inverse of the forward rule already implemented:

  ```
  logb(x) ∈ [lo, hi]   ⟹   |x| ∈ [2^lo, 2^(hi+1))
  ```

  Fixes the **digit position**. Without it a bound-only refinement leaves `_t7`
  at `exp=-1090`, sixteen binades past what `F64` holds, because the analysis
  still believes a subnormal `x` can be scaled by a small factor.

Measured after both: `_t7` at `exp=-82, bound=2^40` for an `FP64` source, where
the truth is `[2^10, 2^11)`. Loose, and inside `F64` — which is what storage
selection needed. Closing the remaining slack is what the designs below are for.

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
an inference problem rather than an annotation problem. Finding (3) is **fixed**:
`_assert_fixed_exact` now emits the specials, representability and bound checks on
the general `Cast` path, so a fixed-point context no longer passes a
storage-exactness test off as a context one.

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

## Design C — one rule, not a relation

The cheapest of the three, and the only one that reaches a rounding whose
position is *still* symbolic — which is the shape a hand-written program takes:

```python
e = fp.logb(x)
with fp.MPFixedContext(e - 10):
    y = fp.round(x)
```

`y` infers as `RealFormat()` today. `_scope_format` resolves the scope through
`_resolve_active_ctx`, which returns a `Context` or nothing; `MPFixedContext(e -
10)` is a `Call`, so the `REAL_FORMAT` fallback fires and the rounding reads as
exact. Nothing is *degraded* — the scope's shape is right there in the AST — it is
discarded.

Measured on an `FP32` source, sweeping every binade including the subnormal
extremes:

| | |
|---|---|
| truth | `𝒜(11, -149, ~2^128)` |
| inferred | `RealFormat()` |

The width is `k + 1`, saturating at the operand's own precision — a quantum finer
than `x`'s grid returns `x` unchanged, which is also why the digit position
bottoms out at `x`'s own `exp` and *not* at `exp - k`:

| `k` in `logb(x) - k` | 0 | 1 | 5 | 10 | 23 | 40 |
|---|---|---|---|---|---|---|
| observed `prec` | 1 | 2 | 6 | 11 | 24 | 24 |
| `min(k + 1, 24)` | 1 | 2 | 6 | 11 | 24 | 24 |

So the whole content is one closed-form transfer rule:

```
round_{𝒜(∞, logb(x) - k + 1)}(x)   →   𝒜(min(k + 1, prec_x), exp_x, 2^(e_max + 1))
```

with `e_max` the top of `logb(x)`'s inferred range. The exponents still cancel —
`(e+1) - (e-9) = 10` — but inside the rule, so nothing symbolic escapes into the
lattice: no linear arithmetic over program variables, no join or widening story
for symbolic exponents, no threading through a 2600-line `FormatInfer`.

It needs the first two of the three facts A and B need, and not the third:

1. **A scope whose shape is known and whose position is an expression**, so
   `_scope_format` stops falling back to `REAL_FORMAT`. The structure —
   fixed-point, unbounded, one position parameter — is syntactic.
2. **The `logb` relation at a definition rather than at a branch.**
   `_implied_logb` already states `logb(v) >= lo ⟹ |v| >= 2^lo`, but only as a
   refinement conditioned on a comparison; here there is no branch to hang it on.

**What it does not cover** is anything where the rounded value is not the value
whose `logb` was taken. That includes the output of `rescale_fixed`, which is what
the rest of this page is about — there the rounding has already been rewritten to
`(2 ** -exp) * x` at position zero, so the correlation has to survive a `Mul` and
needs Design B's tag. It also excludes hand-written programs more interesting than
the one above: see [Which design each shape
needs](#which-design-each-shape-needs), where a two-factor version already falls
to Design B. So Design C is the *one-`logb`, direct-rounding* rule, not "the
hand-written case" in general.

Degrades by falling back: a position not recognizably `logb` of the value being
rounded reads as it does today.

## Which design each shape needs

Three shapes, measured. Each needs strictly more than the last, and the third
needs something none of the designs above has.

### One `logb`, rounded directly — Design C

`with fp.MPFixedContext(logb(x) - k): y = fp.round(x)`.
`prec = min(k + 1, prec_x)`, `exp = exp_x`. See above.

### `n` `logb`s cancelling through arithmetic — Design B

```python
ex = fp.logb(x)
ey = fp.logb(y)
p = x * y                                   # exact, under REAL
with fp.MPFixedContext(ex + ey - 10):
    y = fp.round(p)
```

`prec = k + 2`, not `k + 1`: `mx · my ∈ [1, 4)` spans two binades where a single
mantissa spans one. Measured, adversarially — both mantissas near 2, so the
product's is near 4:

| `k` | 0 | 1 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| observed `prec` | 2 | 3 | 7 | 12 | 22 |
| `k + 2` | 2 | 3 | 7 | 12 | 22 |

Random sampling reports `k + 1` at larger `k`, because the extra bit needs
`mx·my > 4 - 2^-k`; it has to be searched for. Generally
`prec = min(k + n, prec_of_value)` for `n` tagged factors, with Design C the
`n = 1` case. This is Design B's territory exactly — multiply mantissas, add
exponents — and it is *hand-written*, which is why Design C cannot claim that
whole class.

### A `max`-aligned fused sum — Design B **plus an order relation**

What `examples/mmasim/nv.py` and `amd.py` actually do:

```python
es = [exponent(a, emin_a) + exponent(b, emin_b) for a, b in zip(A, B)]
e_max = max(max(es), e_c)
s = fused_sum(join(prods, [c]), e_max - F - 1, fp.RM.RTZ)
```

with `exponent(x, emin) = max(fp.logb(x), emin)` and `fused_sum` rounding each
summand at `2^(e_max - F)` before adding exactly. Bound
`F + 2 + ⌈log₂(L+1)⌉`, sound with a few bits of slack because the terms'
exponents are usually spread:

| L, F | 4, 13 | 8, 13 | 16, 13 | 4, 35 | 16, 35 | 64, 35 |
|---|---|---|---|---|---|---|
| observed | 17 | 17 | 17 | 38 | 39 | 40 |
| bound | 18 | 19 | 20 | 40 | 42 | 44 |

Five requirements stack, and **the fourth is in none of the designs above**:

1. the tag survives `max(logb(x), emin)` against a constant — listed under *Open
   questions* as hypothetical; here it is on every term, unavoidably
2. the tag survives a *sum of two tags* (Design B)
3. `e_max = max(max(es), e_c)` — a `max` over a *list* of symbolic exponents
4. **only the ordering `e_i ≤ e_max` is needed, never `e_max`'s value.** Each
   term is `≤ 2^(e_i+2) ≤ 2^(e_max+2)`, so contributes `≤ 2^(F+2)` quanta. That
   is an *inequality between symbolic exponents*; both designs above are built
   entirely on cancellation, which is a different mechanism — and a cheaper one
   to satisfy, since the value never has to be evaluated.
5. it crosses a *call edge* into `fused_sum` and a *comprehension* inside it

Today `s` reads as `RealFormat()`. Until recently `FormatInfer.analyze` did not
get that far: it raised on these models, since a comprehension argument crossing a
call edge into a callee that fills an `fp.empty` hit a join with no case. Fixed by
`VarFormat` (`e433da8`).

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
instead of `[-1074, 1023]`.

**The lower clamp is now emitted.** `float_to_fixed` used to state only the
*upper* one, leaving `2 ** -exp` free to reach as far as the smallest subnormal
of the source. `max(logb(x) - P + 1, expmin)` is redundant at run time — the
subnormal branch takes every `logb(x) < emin`, and `emin - P + 1 == expmin` for
every format with subnormals, verified across nine of them — and it fires on none
of 428,977 `FP32` values. It is stated because inference reads a `max` and not a
branch condition:

| scale-in bound | before | after |
|---|---|---|
| `FP32` source | `2^287` | `2^152` |
| `FP64` source | `~2^2108` | `~2^1048` |

It costs two lines and a redundant `signbit` comparison per rounding, since the
position is stored as a float and so takes the IEEE `maximum` spelling.

An `FP64` source additionally needed `x` itself constrained, from both
directions — see *Path sensitivity alone* above, which the clamp turned from a
dead end into the route.

### The class half was separable, and is done

Refining special-value membership at branches turned out to be worth separating:
it is a four-atom lattice rather than a numeric domain, it needs none of the
machinery below, and it closed a correctness gap as well as tightening bounds.
That is `fpy2/analysis/value_class.py`.  What follows concerns only the
*magnitude* half, which it cannot help with.

### A second payoff for the `isnan`/`isinf` refinement

**Done**, by the class half rather than by anything here: the `ldexp` lowering
asked whether its exponent could be non-finite and answered from the exponent's
*format*, which for an exponent out of `logb` admits both specials and so cost a
runtime branch.  Reading the upstream tests instead drops it, in
`fpy2/analysis/value_class.py`.

## Open questions

- Does the tag in design B survive `min`/`max` on the exponent? Now load-bearing
  twice over: the *lower* clamp is emitted on every path, so a `max` always sits
  between `logb` and the scale — and `exponent(x, emin) = max(logb(x), emin)` puts
  one on every term of the fused sum, which is what `examples/mmasim` is built
  from. See [Which design each shape needs](#which-design-each-shape-needs).
- Should the mantissa interval be signed, or should sign be tracked separately?
  `logb` discards sign, so `x = ±m·2^e` needs the sign from elsewhere.
- Is `logb` the only introducer worth having? `frexp`-style splitting and
  `round_at` would name an exponent too.
- For design C: should the rule also fire for `RoundAt` and `Cast` at a symbolic
  position, and for a *bounded* symbolic context (`MPBFixedContext`), where the
  bound is concrete while the position is not? The `min`/`max` question above
  applies to it unchanged, since the clamp sits between `logb` and the use on
  every path.
