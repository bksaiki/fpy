# Digit-bound inference: fixed-point precision for `fused_sum`

**In progress.** `examples/mmasim/utils.py:28` aligns every summand at a
run-time position and sums them exactly. The rounded summands have precision
`F + 2` and the sum `F + 2 + ceil(log2 L)`; `FormatInfer` alone reports
`RealFormat()` for both, so no storage can be selected and the models do not
lower. This page is about deriving those two numbers.

[What landed](#what-landed) has the analysis; [What is left](#what-is-left) is
the remaining distance to a compiled program. Measurements are at `12fdef0c`
with this branch applied.

## The shape

```python
@fp.fpy(ctx=fp.REAL)
def fused_sum(xs, n, rm):
    with fp.MPFixedContext(n, rm):
        ts = [fp.round(x) for x in xs]
    return sum(ts)
```

Five call sites (`nv.py:161`, `nv.py:237`, `nv.py:302`, `amd.py:209`,
`amd.py:277`), all of the form `fused_sum(list, e - F - 1, fp.RM.RTZ)` where `e`
is a `max` over the exponents of the very elements being summed.

The same shape appears hand-written without the call:

```python
e = max([fp.logb(x) for x in xs])
with fp.MPFixedContext(e - 12):
    ys = [fp.round(x) for x in xs]
return sum(ys)
```

`ys` is 13 bits wide — 12 for the top binade, plus the carry to `2^(e+1)` that
`MPFixedContext`'s default `RM.RNE` allows — and the sum of 32 of them is 17.
Both read `RealFormat()` today.

## The target, measured

Random inputs plus an adversarial generator — every product in one binade with
mantissas near the top, so the terms cannot be spread:

| config | term prec | `F+2` | sum prec | `F+2+ceil(log2(L+1))` |
|---|---|---|---|---|
| Volta FP16, L=4, F=23 | 25 | 25 | 27 | 28 |
| Turing FP16, L=8, F=24 | 26 | 26 | 29 | 30 |
| Ada FP8, L=16, F=13 | 15 | 15 | 18 | 20 |
| Hopper FP8, L=32, F=13 | 15 | 15 | 18 | 21 |
| Hopper FP16, L=16, F=25 | 27 | 27 | 30 | 32 |

The per-term bound is exactly tight. The sum bound carries up to three bits of
slack: saturating it needs every term maximal, same-signed, and in the top
binade simultaneously.

## What the callee needs: one number

`fused_sum` is parametric in `n` and `rm`, so its precondition has to be stated
in its own terms. With the quantum written `2^(n+1)` (`MPFixedContext.nmin` is
the first *un*representable digit):

```
forall x in xs.  |x| < 2^(n + 1 + W)
  ==> prec(ts[i])    <= W                     RTZ/RTO;  W + 1 otherwise
  ==> prec(sum(ts))  <= W + ceil(log2 len(xs))
```

`W` is how many binades the list's magnitude bound sits above the alignment
quantum. That single difference is the whole deliverable.

At the NVIDIA call sites `W = F + 2`. `es[i]` is the *exponent-field sum*
`exponent(a, emin_a) + exponent(b, emin_b)`, and `exponent(x, emin) =
max(logb(x), emin) >= logb(x)`, so `es[i] >= logb(a) + logb(b) >= logb(a*b) - 1`
— a product reaches `2^(e_max+2)` while the accumulator, bounded by its own
`e_c <= e_max`, only reaches `2^(e_max+1)`.

## Everything non-relational is already tight

`make_t_fdpa(FP16, FP16, FP32, F=24)` monomorphized at `L = 8`:

```
(a * b)            MPBFloatFormat(pmax=22, emin=-27, maxval ~ 2^32)
exponent(a,emin_a) MPBFixedFormat(nmin=-1, maxval=15, ...)
max(max(es), e_c)  MPBFloatFormat(pmax=7, ...)            e_max in [-132, 127]
((e_max - F) - 1)  MPBFixedFormat(...)                            in [-157, 102]
join(prods, [c])   ListFormat(elt=IEEEFormat(es=8, nbits=32))
fused_sum(...)     RealFormat()
```

`F`, `e_zero` and `emin_a` arrive as `SetFormat({24})`, `{-132}`, `{-14}` — the
factory's closure constants are already visible to the analysis. Every line
above is tight for what it knows; composing them gives `2^128 / 2^-156` = 284
bits against a truth of 26. The entire gap is one fact:

```
forall i.  logb(prods[i]) <= es[i] + 1   and   es[i] <= e_max
```

## The derivation

```
A.elt, B.elt              atoms ea, eb        |a| < 2^(ea+1)
exponent(a, emin_a)       >= logb(a)          only the >= side of max, vs a constant
prods.elt = a*b           |p| < 2^(ea+eb+2)   exponents add, mantissa in [1,4)
es.elt = ea+eb | e_zero                       the e_zero arm has p == 0
e_max = max(max(es), e_c) ==> es.elt <= e_max,  e_c <= e_max
---------------------------- re-anchor, atoms eliminated ----------------------
prods.elt  |p| < 2^(e_max+2)
c          |c| < 2^(e_max+1)
join(...)  |.| < 2^(e_max+2)     an ordinary join in one index set; survives join()'s
                                 fp.empty + store loops on existing ListFormat
fused_sum(_, e_max-F-1, RTZ)     W = (e_max+2) - (e_max-F) = F+2
```

Eliminating the per-element atoms at the `max` is what keeps this small: exactly
one live anchor flows downstream, so nothing else in the lattice needs a
symbolic component.

The `if p == 0 else e_zero` arm needs no disjunction. `|p| < 2^(es[i]+2)` holds
in both arms — trivially when `p` is zero — so the merged anchor is the same
term.

## The abstract domain

The analysis is **digit-bound inference**: it bounds the number of binades between a
value and the position it will be rounded at. That count is the fixed-point
precision, and `oo` — no bound — is the degenerate answer that `RealFormat()`
already gives.

The domain is **relational**, so almost nothing lives in the per-expression
cell: each expression gets a *handle*, and every fact about it lives in one
shared store. Concretely, per function instantiation:

```
for each real-valued SSA definition v:
    fmt(v) : FormatBound      today's non-relational cell, unchanged
    msb(v) : Int              an upper bound on v's leading digit,
                              i.e. on logb(v)                    -- how big
    lsb(v) : Int              a lower bound on where v's last
                              digit sits                         -- how fine

for each exponent-valued (integer) SSA definition e:
    e      : Int              the value itself

one store : a conjunction of linear integer constraints over all of the above
```

`msb` and `lsb` are *variables*, not numbers — that is the whole difference from an
interval domain, and the reason `AbstractFormat` cannot express any of this.
A list contributes one pair of variables for its **element summary**, not one
per element; see [Element-wise summaries](#element-wise-summaries).

`Logb` is what ties the two sorts together: `e = logb(x)` makes `e` and `msb(x)`
the same variable. Everything else is arithmetic on integers.

Concretization: a tuple of runtime values is in the store's γ iff **every `v`
is finite** and some assignment satisfies every constraint with `msb(v) >=
logb(v)` and `lsb(v) <=` the position of `v`'s least significant digit, for all
`v`. An unsatisfiable store is bottom (unreachable code); the empty store is
top, and means `RealFormat()` everywhere.

**The finiteness is a side condition, not a consequence.** It is tempting to
argue it away — `logb(inf)` is `+inf`, so no finite `msb` satisfies `msb(v) >=
logb(v)` and the state falls out of γ by itself. That works for an infinity
and *not* for a NaN: `logb(NaN)` is NaN, `exact_logb` models it that way, and
`msb(v) >= NaN` is undefined rather than unsatisfiable. The same hole is in the
grid half, where "the position of `v`'s least significant digit" has no
reading for either. So γ has to exclude them by construction.

It is also not new. `_logb_range`, which seeds every bound, reads a format's
*finite* maxval even where the format admits infinities, and the rules have
always stated bounds a non-finite run violates: FP16 `x * x` is bounded at
`2**15` where the interpreter gives `+inf` at `x = 300`, and FP32 `sqrt(x)` at
`2**127` where `x = -1` gives NaN. What the store answers is a bound on the
finite executions, and a program that must stay in range on a non-finite input
needs the guard the models already write — `sum_special_values` before the
fused sum.

Two consequences worth having in view:

- **The side condition does not reach the consumer.** `_tighter` carries the
  incoming format's `has_pos_inf`/`has_nan` onto a bound it narrowed from a
  binade, so the emitted format says "may be `inf`" and "below `2**(mag+1)`"
  at once, and storage selection reads it with no notion of the caveat. This
  predates the analysis, but `_visit_return`'s non-finite-path rule is the
  first to discard a *finite* magnitude on the strength of it. Surfacing the
  caveat — `_tighter` clearing the special-value flags when it narrows — is
  the honest fix and is not done.
- **`_visit_branches` still joins a non-finite-guarded arm** where
  `_visit_return` drops one, so `if isnan(x): return BIG` and `y = SMALL; if
  isnan(x): y = BIG` give different bounds for the same program. The join is
  the *sound* half; the asymmetry is a precision difference and a surprise,
  not an unsoundness.

One query, an optimization over the store:

```
prec of v  =  max(msb(v) - lsb(v) + 1) + carry
```

`prec_at` is the same query against the grid a rounding context implies: a
context names the first *un*representable digit, so `lsb = n + 1` and the query
reads `max(msb(v) - n) + carry`.  Note `prec` counts digits where a span would be
one less.

`lsb` is only needed for an unrounded value, and only for values the program has not
already rounded — see
[Asking for a precision needs a third field](#asking-for-a-precision-needs-a-third-field).

## The constraint store

The constraint language is:

- **definitional equalities**, one per exponent-valued SSA definition, each an
  affine combination of earlier ones (`es = ea + eb`, `n = e_max - F - 1`);
- **one-sided inequalities** from `max` and `min` (`es.elt <= e_max`);
- **anchors** attaching a value's magnitude bound to a term (`logb(p) <= es + 1`).

The only query is an upper bound on a single term. Everything stays in
*exponent* space — that discipline is what keeps `2^x` out of the encoding —
so the store is linear integer arithmetic over a handful of variables.

### What is deliberately not tracked

**No mantissa.** The query is `msb(v) - n`: how many *binades* separate the
value's top from the quantum. Where a value sits inside its binade is worth at
most one bit, through the carry, which the `carry` term already handles. It
would recover part of the sum's one-to-three bit slack, but the terms that
saturate that bound are exactly the ones whose mantissas are near maximal, so
the interval would sit at `[1, 2)` anyway. And the mantissa magnitude already
exists non-relationally in `AbstractFormat` — only the relational exponent part
is missing.

A *magnitude* interval would also cost linearity, since mantissas compose
multiplicatively where exponents compose additively. That is not an argument
against mantissa information in general, only against that representation of
it: counts of leading and trailing zero/one bits are **positional**, so they
compose additively and stay in the same store — which is exactly what SELTZO
does, see [Prior work: SELTZO](#prior-work-seltzo). The reason to omit them here
is that a fixed-point precision depends on the binade count and not on the bit
shape, not that they could not be tracked.

**No lower bound on a value's magnitude.** Two different things here, and only
one is omitted. Lower bounds on *exponent variables* (`e_c >= -132`,
`e_max >= es`) are tracked and load-bearing. What is dropped is the lower half
of the `logb` introduction, `2^e <= |x|`, propagated through operations:

1. Nothing queries it. A fixed-point value's least significant digit is pinned
   at `n` by the context; bits below it are gone by construction, not by
   inference, so "this value is not tiny" has no consumer.
2. `Add` cannot supply one. Cancellation drives `logb(a+b)` unboundedly low, so
   the field would be bottom at every `Add`, `Sub`, `Sum` and `Fma` — every
   interesting node in a corpus built out of sums.
3. The asymmetry is what makes coverage broad. The 27 ops in the cancelling
   tier are there because upper bounds compose through `max` and `+`; demanding
   both directions drops `Add` `Sub` `Sum` `Fma` `Hypot` `Fdim` `Min` `Max`
   out of it.

The price is `Div` and general `Pow` staying partial, recorded under
[Where the corpus really does under-constrain the design](#where-the-corpus-really-does-under-constrain-the-design).

### Asking for a precision needs a third field

Precision is `msb(v) - lsb(v) + 1`, where `lsb(v)` is the position of `v`'s least
significant digit — its *grid*. That is a different quantity from the magnitude
lower bound above, and the distinction decides the formula: subtracting the
smallest exponent `v` attains over all executions gives the **dynamic range**
(276 for `FP32`), not the precision (24).

For every value this analysis currently asks about, `lsb` is *given* rather than
inferred: a rounded value's grid is the context's position, `lsb = n + 1`, so the
query `msb(v) - n` already is the precision. The field is only needed for an
**unrounded** expression under `REAL`, whose grid is inherited from its
operands. It composes in the sound (downward) direction, and stays linear:

| op | rule |
|---|---|
| `Mul` | `lsb(ab) >= lsb(a) + lsb(b)` |
| `Add` `Sub` `Sum` | `G >= min_i lsb(x_i)` |
| `Fma` | `>= min(lsb(a) + lsb(b), lsb(c))` |
| `Neg` `Abs` `Copysign` | `>= lsb(x)` |
| `Round` `Cast` at `n` | `>= n + 1` — exact, operand-independent |
| `Logb` and the integer-valued ops | `>= 0` |
| unsupported | unconstrained, so `P = oo` |

Reason 2 above does **not** apply to it: cancellation destroys a magnitude lower
bound but leaves the grid alone, since `a + b` is still a multiple of
`2^min(lsb(a), lsb(b))`. So this field is much cheaper than the one it resembles.

The tie between `msb` and `lsb` has to be per value, not an interval. For an `FP16`
product, with `lsb(x) >= msb(x) - 10` stated against *that value's* `logb`:

```
relational  msb - lsb + 1 : 22        exact
independent msb - lsb + 1 : 81        msb <= 32 and lsb >= -48 bounded separately
AbstractFormat pmax   : 22        what FormatInfer already reports
```

Breaking the correlation is worse than not asking — the maximum of `msb` and the
minimum of `lsb` are never attained together.

`mmasim` gains nothing from this: every value whose precision it needs has been
rounded, so the grid comes from the context. It is what a general
"what is the precision of this expression" query would require.

Two things this deliberately is not:

- **A symbolic-exponent domain threaded through the lattice.** Anchors exist
  only between a `logb` and the rounding that consumes it; the re-anchoring step
  above discharges them at the `max`.
- **An `[lsb, msb]` re-parameterization of `AbstractFormat`.** That is the same
  domain in different coordinates and adds no relational power. The formats
  above are already tight.

### Emit, then query

Constraints are emitted while walking the AST; precisions are queried at rounding
sites. That split is the architecture regardless of what answers the query, and
it is what keeps the solver out of `FormatInfer`'s widening loop — solver
answers are not a lattice. Known loop counts unroll (`ArraySizeInfer` supplies
them here); unknown counts would need invariant inference, which a solver alone
does not provide.

### Feeding back into format inference

The two domains tighten each other, so this is a genuine reduced product rather
than a pipeline. The store reads concrete bounds out of `AbstractFormat`
(`la <= 15` comes from `exact_logb` on `FP16`), and a derived precision materializes
back as a strictly tighter format. `ts[i]` in `t_fdpa` goes from `RealFormat()`
to

```
A(26, -157, 2^133)      prec = F + 2; position and bound at their concrete extremes
```

which is expressible because `AbstractFormat.prec` is independent of the
exp/bound span — the same shape as `FP32`'s own `A(24, -149, 2^128)`.

**Materializing is lossy, so it is for consumers only — never to re-derive.**
Feeding that element format back through `_sum_bound`'s pairwise simulation:

```
sum via AbstractFormat  : prec = 294
sum via the store       : prec = 30      F + 2 + ceil(log2 9)
```

The shared-grid correlation is exactly what the materialized format drops. A
second `FormatInfer` pass reading materialized formats therefore gets strictly
less than the store does, which is the argument against an alternate-the-passes
architecture: the store has to answer *both* queries directly.

What to do instead is emit and query in the **same** walk. SSA dependency order
means one pass gets the forward direction completely — by the time a rounding
site is queried, every operand's format and every constraint it needs is already
emitted. Loops re-emit per iteration inside `FormatInfer`'s existing fixpoint.

An outer fixpoint only pays for *backward*-flowing facts: something learned late
tightening something computed early, such as an `assert` after the use or a
callee's return refining a caller's argument. It terminates — the precisions are
integers, decreasing, and bounded below by the truth — but it should be capped
the way `loop_iter_limit` caps the existing one, and it is not worth building
until a program needs it.

### Use z3's `Optimize`

The whole store for `make_t_fdpa(FP16, FP16, FP32, F=24)`:

```python
store = [
    la <= 15, lb <= 15, lc <= 127,        # exact_logb on FP16 / FP32
    la <= ea, lb <= eb, lc <= e_c,        # exponent(x,emin) = max(logb x, emin) >= logb x
    lp <= la + lb + 1,                    # logb(a*b) <= logb a + logb b + 1
    es == ea + eb,                        # the comprehension body
    e_max >= es, e_max >= e_c,            # max(max(es), e_c)
    n == e_max - F - 1,                   # the call-site argument
]
o.maximize(lv - n)                        # |v| < 2^(lv+1), quantum 2^(n+1)
```

```
product summand : 26 (F+2 = 26)
accumulator     : 25 (F+1 = 25)
without max     : oo
```

Four reasons to prefer this over a hand-rolled tracker:

- `maximize` *is* the query. A difference-bound matrix decides `T <= anchor + k`
  and would need a search loop to find the least `k`.
- `oo` is exactly `RealFormat()`, so the fallback is total and needs no special
  case for an unconstrained term.
- `ite` is free. `es[i] = e_zero if p == 0 else ...` and `gtr_fdpa`'s
  `cr = 0 if e_c < e - F - 1 else ...` happen to collapse — both arms share an
  anchor — but that is a per-site hand proof. A hand-rolled domain needs
  disjunctive completion to stop depending on it.
- Branch conditions are assertions rather than a separate refinement mechanism.

The costs are real and belong in the decision:

- **Dependency.** `fpy2` has three runtime deps and no solver, and storage
  selection would come to need one. Making it optional is worse than either
  choice: the emitted C++ type would depend on whether z3 is installed.
- **Timeouts must degrade to `oo`**, never to an error — at this size the
  question is hygiene rather than performance.
- **It buys the easy half.** Deciding which facts are *sound* to emit is the
  hard part (see [Element-wise summaries](#element-wise-summaries)); a solver will
  prove whatever follows from unsound premises.

The queries that actually arise are difference-shaped, so a DBM remains a
drop-in replacement behind the same interface if the dependency turns out
unacceptable.

## Operator coverage

The design above is derived from one corpus, so the risk is a rule set that fits
`mmasim` and nothing else. The check is the operator table, not the examples.
FPy has 89 op classes: 14 constants, 10 predicates, 11 structural (`Len`,
`Range*`, `Zip`, ...), and **54 real-valued**.

Every query has one shape — an upper bound on `logb` — and `logb` upper bounds
compose through `max` and `+`. Writing `msb(v)` for that bound:

| rule | ops |
|---|---|
| `msb(x)` | `Neg` `Abs` `Copysign` |
| `max(msb(a), msb(b)) + 1` | `Add` `Sub` `Hypot` `Fdim` |
| `msb(a) + msb(b) + 1` | `Mul` |
| `max(msb(a)+msb(b), msb(c)) + 1` | `Fma` |
| `max(msb(a), msb(b))` | `Min` `Max` `AMin` `AMax` |
| `max_i msb(x_i) + ceil(log2 n)`, `n` from `ArraySizeInfer` | `Sum` |
| `<= msb(b)` — tightening | `Mod` `Fmod` `Remainder` |
| `max(msb(x), 0)` | `Floor` `Ceil` `Trunc` `NearbyInt` `RoundInt` |
| `msb(x) + carry`, carry = 0 for `RTZ`/`RTO` and 1 otherwise | `Round` `Cast` |
| `= k` exactly — the second introducer | `Exp2`, `Pow(2, k)` |
| introduces the atom, `2^e <= |x| < 2^(e+1)` | `Logb` |
| `k·msb(y) <= msb(x)` | `Sqrt` (k=2), `Cbrt` (k=3) |
| `<= 0` or `<= 1` from a bounded range | `Sin` `Cos` `Tanh` `Erf` `Erfc` `Atan` `Asin` `Acos` |

### Having a rule is not the same as doing the job

The analysis exists to make a precision **independent of the exponent's range** —
`F + 2`, not "bounded by 161 because `FP32`'s exponent bottoms out at -149".
That happens only when the rule's term is `msb(operands) + const`, so a position
derived from the same `logb` cancels. Sorting the 54 by that criterion, rather
than by whether a rule exists:

| tier | count | ops |
|---|---|---|
| **cancels** — precision is a constant | 27 | `Abs` `Neg` `Copysign` `Add` `Sub` `Mul` `Fma` `Min` `Max` `AMin` `AMax` `Sum` `Round` `Cast` `RoundAt` `Logb` `Exp2` `Floor` `Ceil` `Trunc` `RoundInt` `NearbyInt` `Mod` `Fmod` `Remainder` `Hypot` `Fdim` |
| **bounded, never cancels** | 10 | `Sqrt` `Cbrt` `Sin` `Cos` `Tanh` `Erf` `Erfc` `Atan` `Asin` `Acos` |
| **partial** — needs a symbolic lower bound | 2 | `Div` `Pow` (general; `Pow(2,k)` cancels) |
| **no linear rule** | 15 | `Exp` `Expm1` `Log` `Log1p` `Log2` `Log10` `Tan` `Sinh` `Cosh` `Asinh` `Acosh` `Atanh` `Atan2` `Lgamma` `Tgamma` |

Only `RTZ` and `RTO` avoid the carry, and not for the same reason. `RTZ` never
increases a magnitude at all; the other directed modes each carry on one sign —
`RTP` rounds away from zero for positive values, `RTN` for negative ones — and
this domain bounds a magnitude without tracking sign, so neither is safe.
Measured across all eight modes.

`RTO` needs care, and is safe only under two conditions. Counting in quanta,
carrying out means landing on `2**prec`, which is even once `prec >= 1`, and
round-to-odd never produces an even significand — but at `prec == 0` the
carried count is `1`, which is odd, so `RTO` takes it. And the argument is
about a **fixed-point** destination: for a floating-point one the carried value
is a power of two whose significand is `1`, and a one-bit format makes every
representable value odd. The rule as implemented is gated on
`MPFixedContext` and on the precision, and reusing it elsewhere means redoing the
argument.

`Sqrt` is the clearest case of the distinction. `2·msb(sqrt x) <= msb(x)` is sound
and tight — verified over 200k `FP32` values, minimum slack 0 — but it *halves*
the exponent instead of shifting it, so under `MPFixedContext(e - 12)` the precision
is `floor(e/2) - (e - 12)`, which grows without bound as `e` falls. It pays off
only when the position is derived through the same halving
(`logb(x)//2 - k`), a shape nothing writes. `Cbrt` is identical with 3.

The bounded-range ops sit in the same tier for the opposite reason: `msb(sin x)
<= 0` is a constant, so the precision tracks the *position's* range rather than
cancelling. Finite where today's answer is `RealFormat()`, and still far too
wide for a narrow datapath.

So the honest count is **27 of 54 ops in the tier that matters**, not 37.

### Reachability is a second filter

A relational bound only beats the non-relational one where the scope is
symbolic — under a concrete context `AbstractFormat` already has a tight bound.
Two scopes qualify, and they differ:

- Under `REAL`, only these evaluate at all: `fabs` `neg` `ceil` `floor` `trunc`
  `roundint` `logb` `add` `sub` `mul` `div` `pow` `copysign` `fmax` `fmin`
  `fma`. `sqrt`, `exp`, `hypot`, `fmod` and the rest raise
  `NotImplementedError` — they are irrational, so FPy has nothing exact to
  return.
- Under `MPFixedContext(<symbolic>)` — the scope this analysis is about —
  **every** op evaluates, `sqrt` and `exp` included.

So the tier-2 and tier-3 rules are reachable, just not useful, and the first
list is a good guide to which rules earn their implementation first.

### Why `Add` needs no alignment

Addition is the operator a relational value domain handles worst: adding two
symbolically scaled quantities requires their scales to agree, and in general
they do not. In exponent space the problem disappears —
`|a+b| <= 2·max(|a|,|b|)` gives `msb(a+b) <= max(msb(a), msb(b)) + 1`
unconditionally, with no alignment and no fallback. What alignment would buy is
the *lower* bound, which cancellation destroys and which this analysis never
queries. That asymmetry is why the coverage above is broad rather than
corpus-shaped.

### Where the corpus really does under-constrain the design

- **Every rule above is upper-only**, by the argument in
  [What is deliberately not tracked](#what-is-deliberately-not-tracked). The
  omission is principled, but `mmasim` never asks for the other direction, so
  the design has never had to confront a program that does.
- **One introducer.** `Logb` alone. `frexp`-style splitting and a `RoundAt`
  position name exponents too.
- **One index set.** The element-wise machinery is built around
  comprehension-over-`zip`. An indexed `for` writing into `fp.empty` is a
  different shape; `gst_fdpa` (`nv.py:290`) and `join` use it, so it is
  exercised, but only there.
- **No division and no transcendentals in the corpus**, so
  [Degradation](#degradation) is a claim about those 17 ops rather than an
  observation.

## Degradation

The store is a conjunction of sound facts, so *omitting* a constraint only
enlarges the concretization. Emitting fewer facts is always safe, and the
fallback is the limit of the normal path rather than a special case.

A fresh unconstrained variable is not the right fallback, though — it discards
what `AbstractFormat` already knows. Emitting only the non-relational bound is
free and strictly better:

```
rule known       (lp <= la + lb + 1)   26     the real answer
op unsupported,  concrete bound only  189     FP16^2 < 2^32, e_max >= -132
op unsupported,  nothing at all        oo     == RealFormat(), today's answer
```

An unsupported operator under a *concrete* context still yields a finite precision
downstream. `oo` bites only under a symbolic context, where `AbstractFormat`
had nothing to offer either. The relational store can only improve on it; its
failure mode is "no improvement".

Five paths land here, and all five must:

1. an operator with no rule — a variable bounded only by `AbstractFormat`;
2. a nonlinear term (`e1 * e2` in a position) — *not* emitted, or the store
   leaves LIA and decidability with it;
3. a foreign value or an opaque `Call`;
4. `_analyze_callee` declining a non-`Function` callee;
5. the solver answering `unknown`.

The last is a determinism requirement rather than hygiene: a timeout that
changes the emitted C++ type is worse than a type that is merely wide. Keeping
the store in LIA is what makes `unknown` a bug rather than a routine path.

## Element-wise summaries

`es` and `prods` are different lists built by comprehensions over the same
`zip`. The atoms `ea`, `eb` are the per-element summary variables of `A.elt` and
`B.elt`, so facts about the two lists compose: both are universally quantified
over the same index space, and an atom is a function of the element rather than
of the loop.

Relating *different* elements is what would be unsound, so a summary has to be
identified by its iteration space. `amd.py:277` makes that load-bearing:

```python
e_even = max([es[i] for i in range(0, L, 2)])
t_even = fused_sum([prods[i] for i in range(0, L, 2)], e_even - F - 1, fp.RM.RTZ)
```

`e_even` bounds only the even `es`, so the whole-list summary is useless here.
The sub-list comprehensions have to carry the atoms through, and pairing them is
sound only because the two strides match. Checking that — same iterable, same
stride — is the sharpest requirement the models impose.

**Landed**, as *index-set instantiation*. A range with a start or a step is a
part rather than a cover, so `xs[i]` under one gets variables of its own, and
`DigitBoundStore.instance` replays onto them every constraint whose variables are
*all* per-element. Such a constraint holds at every index, so it holds at an
index drawn from any subset; one naming anything else may be an aggregate over
the whole list (`logb(sum xs) <= msb(xs) + k`), true of the list and false of a
part, and is left alone. Per-element is tracked positively — minted inside a
loop body or a comprehension, minted by a callee called from one, or the
summary of a list — and `_at_index_set` declines rather than sharing when it
cannot vouch for a variable.

`_range_key` is the "same iterable, same stride" check: a literal by its value
and a variable by its definition, so the two loops over `range(0, L, 2)` reach
one renaming and `es` and `prods` keep the pairing they had. A gather also
*covers* its destination — every element it writes is an element of the source
at an index in the range, and one it never writes is uninitialized.

Measured on a list of 8 fp32, `max` over the evens and a round of the evens at
`ev - 12`: **288 digits to 12**, and the same program reading the *odds*
stays at 288, which is the whole content of keying on the range.

## Prior work: SELTZO

The closest existing design is **SELTZO** — *Sign, Exponent, Leading/Trailing
Zeros/Ones* — from David K. Zhang and Alex Aiken, "Automatic Verification of
Floating-Point Accumulation Networks", CAV 2025
([arXiv:2505.18791](https://arxiv.org/abs/2505.18791); tool:
[FPANVerifier](https://github.com/dzhang314/FPANVerifier)). It abstracts a
float to six fields: sign, exponent, and the counts of leading and trailing
zeros and ones in the mantissa.

Four things match, closely enough to be worth taking as validation of the
encoding discipline rather than coincidence:

- the exponent is a **symbolic unbounded integer**, not an interval;
- the abstraction is **relational**, and its content is inequalities between the
  exponents of *different* values (`e_x - e_y >= p` there, `es[i] <= e_max`
  here);
- both reduce to **QF_LIA and hand it to z3**;
- both are **precision-independent** in the same sense — the format parameter
  appears symbolically (`p` there, `F` here) rather than by enumeration.

Four differences, each explained by the different goal — SELTZO verifies a fixed
network, this infers a precision to drive code generation:

| | SELTZO | here |
|---|---|---|
| mantissa | leading/trailing zero/one counts | nothing |
| query | unsat of a counterexample condition | `maximize` (OMT) |
| operators | `TwoSum`, via 70+ QF_FP lemmas | 54, with much weaker rules |
| unknown input | not modelled | must degrade, see [Degradation](#degradation) |

Three things worth taking:

1. **`ntz` is a better `lsb`.** SELTZO's trailing-zero count *is* the grid
   position, expressed relative to the exponent rather than absolutely:
   `lsb(v) = e_v - p + 1 + ntz_v`. Relative is the better representation because
   it stays precision-independent, so
   [the third field](#asking-for-a-precision-needs-a-third-field) should be
   spelled that way.
2. **Consistency conditions separate from transfer rules.** SELTZO emits
   well-formedness constraints per abstract value, independently of the
   operation that produced it. That is a cleaner split than folding format
   facts into each rule, and it is how the store is organized.
3. **Unbounded exponents handle subnormals by shifting** rather than by a
   special case. `exponent(x, emin) = max(logb(x), emin)` is exactly a
   subnormal clamp and appears on every term of the fused sums, so this is
   directly load-bearing.

What SELTZO does not offer is the shallow-and-wide half: a rule for every
operator, a graceful fallback, and a reduced product with an existing
non-relational format domain. Those are compilation concerns rather than
verification ones.

## What landed

Measured at `d8e836b5` with `fpy2/` and `examples/` carrying this branch.

- **Contexts keep their shape.** A `with` whose expression does not reduce
  becomes a `PartialContext` — constructor known, arguments filled where they
  evaluated — instead of a bare symbolic variable. This also fixed a soundness
  bug: `_resolve_active_ctx` substituted the *caller's* context for any
  unresolved scope, so an inner `with fp.MPFixedContext(n):` read as the
  caller's and the rounding looked like the identity.
- **Call sites pin what they know.** `FunctionFormat` carries argument values
  and store terms in both directions, so a callee's rounding cancels against a
  position its caller built.
- **The store.** `fpy2/analysis/digit_bound/store.py` — affine `Term`s, linear
  constraints including a disjunctive `le_max`/`ge_min`, and one query, backed
  by z3's `Optimize`.
- **The pass.** `fpy2/analysis/digit_bound/infer.py` — `DigitBoundInfer.analyze(func,
  view)` returning the constraint system and what names it. A reduced product
  with `FormatInfer`, which runs it between two passes of its own.
- **Rules** for the operators in the cancelling tier, element-wise summaries on
  both the comprehension and the desugared loop form, and the `lsb` field that
  lets an aligned sum keep its summands' grid.
- **Partial writes.** A loop that fills only part of a list still bounds it:
  every element a read can reach was written by one of the writes, because an
  element no write reaches is uninitialized and reading it is undefined. That
  removes the tiling proof `join` would otherwise need, and the bound is
  stated once per loop — per iteration it would name the loop-carried value on
  both sides of a `<=max` and say nothing.
- **One level of guard sensitivity.** `e_zero if p == 0 else exponent(a) +
  exponent(b)` is reached with `e_zero` only where `p` is zero, whose `logb` is
  below every bound. `ge_min(m, [iff, logb(subject) - 1])` states exactly
  that — either `iff` is the answer, or nothing the subject's magnitude bounds
  has to hold — and it is what ties `e_max` back to the products. It cost the
  `logb` lower seed, which now sits on the `logb` *expression*, where the
  argument may be taken as non-zero: pinning every `msb` above a format's least
  non-zero value made a zero's own reading contradictory, and an unsatisfiable
  store reads as `-inf`.
- **A guard no path names.** `all([p == 0 for p in group]) or scale == 0`
  zeroes `sum(group) * scale`, one annihilation step from either disjunct.
  The store satisfies a `ge_min` *once*, so a disjunct naming a value only one
  path zeroes lets it drive the merge down while the value the bound is about
  stays high; what is usable is the intersection over paths, closed under what
  annihilates.  It is stated *beside* the plain arm merge rather than instead
  of it -- the vacuity disjunction alone leaves the merge free below, and an
  absolute position is as necessary as a relative one.
- **Two returns.** A function returning on two paths used to keep a term only
  where both named the *same* one, which is what a callee's rounding needs to
  cancel against its caller's position -- but it threw away everything else.
  The paths join now, and a path returning nothing but infinities and NaNs is
  skipped outright: `logb` bounds the finite values and that path has none.
  A call's result also picks up the bound its *site* knows, which is where a
  branch the caller refined away gets stated.  `amd.py`'s `overflow_inf` is
  the shape, and it was the whole of CDNA3's `fused_sum`: `msb(xs) - n` went
  from 84 to 26 against a target of 25.
- **A binary `Add` consults the store.** It ran for `Sum` and `Round`/`Cast`
  only, and the reason `Sum` gives holds just as well one add at a time:
  adding over *formats* drops the shared grid the summands were rounded onto.
  Two values aligned at a run-time position span whatever that position
  reaches, where their difference is the handful of digits the alignment
  left -- 249 bits against the store's 39, for CDNA3's `t + cr`.
- **Tuple and `zip` elements.** A `zip`'s fields are the operands' elements,
  whether the walk meets it as a `Zip` or as a list a loop materialized —
  comprehension elimination leaves one form, `_to_statement_form` the other.
- **Upstream, on `main`.** #295 gave `join` a known length; #296 exempted
  `Empty` from the operators needing a resolvable context and had an unbounded
  fixed-point context take its storage from the value's inferred format; #297
  fixed the last emitter path still asking the context. Together they removed
  every compile blocker this doc listed.

Against the interpreter, adversarially:

| | inferred | actual |
|---|---|---|
| `max([logb(x) for x in xs])`, round at `e-12` | 13 | 13 |
| the same over `x * y`, aligned at `max(logb x + logb y)` | 14 | 14 |
| `sum` of 32 such terms | 18 | 18 |
| a rescaled rounding at `logb(x) - k`, `k = 12` | 12 | 12 |

Equal on the comprehension and desugared forms alike.

### A program compiles

The pipeline that gets furthest:

```python
st.simplify(st.rescale_fixed(st.comp_to_loop(f)))
```

`comp_to_loop` first is what makes `rescale_fixed` apply: a rounding inside a
comprehension has no statement-level position for the scale-in and scale-out
statements the rewrite emits, and lowering the comprehension gives it one. The
other order declines silently, leaving the emitter to refuse the untouched
`MPFixedContext(e - 12)` as symbolic.

The context must also name `RM.RTZ`: integer storage rounds by the cast, and
C++ integer conversion truncates. It then compiles to the code the models
want:

```cpp
assert((std::fabs(std::trunc(_t)) <= 4096) && "fpy: overflow occurred so rounding is undefined");
int16_t _tmp7 = static_cast<int16_t>(_t);
```

`4096` is `2 ** 12`, which the store derives. #296 made an unbounded
fixed-point context take its storage and its bound from the rounded value's
inferred format rather than from itself, and #297 fixed the last emitter path
still asking the context. Before that, `float_to_fixed` stated such a bound as
a claim with `OverflowMode.ASSERT`, which only helped the output those
transforms generated.

**This analysis is what makes it compile at all**, not merely what makes it
narrow. Without the branch the rescaled rounding's format spans the operand's
whole reach:

| | inferred format | storage |
|---|---|---|
| `1f493b6a` alone | `nmin=-37`, `maxval ~ 2**73` | none — storage selection fails |
| with this branch | `pmax=12, emin=11` | `int16_t`, bound `4096` |

## What is left

`fused_sum` reaches `F + 2`; see [The mmasim corpus](#the-mmasim-corpus) for
where each design stops.

Unexercised, and so unproven:

- **Coverage.** A partial write claims nothing about elements no write
  reaches, which is sound only because reading one is undefined. A language
  that defined `empty`'s contents would need the tiling proof after all.
- **The corpus is the only real net.** `_vacuous` fires 127 times compiling
  the 16 designs and twice across the whole 5028-test unit suite, and
  `compile_all.py` is in no test target and not in the `Makefile` -- it is a
  tracker, as it says. The path that takes the corpus from 3/16 to 14/16 is
  therefore covered almost entirely by a script nobody runs automatically,
  which is how the unsoundness above survived three audits.

## The mmasim corpus

`examples/mmasim/compile_all.py` compiles all 16 designs and reports where each
one stops.  A roadmap tracker, not a test.  **14/16**, all fourteen clean under
`g++ -std=c++20 -fsyntax-only`.

`amd.cdna3.bf8` was the last to go, and took three things rather than the one
the sub-range summary looked like:

- the summary itself -- [Element-wise summaries](#element-wise-summaries);
- **a seeded parameter's summary was not marked per-element.**  `analyze`'s
  seed loop shares a caller's terms into a parameter and so bypasses the
  `_fresh_interval` that marks a list's summary, leaving the callee unable to
  replay `msb(prods) <= msb(A) + msb(B) + 1` at an index set.  The same call
  answered 26 analyzed standalone and 86 analyzed as a callee, and the
  callee's is the answer storage selection reads.  `_mark_elt` serves both
  places now;
- **a two-return callee dropped its value.**  `_merge_returns` keeps an exact
  value only where both returns agree -- "a join of two values is not one" --
  and `exponent0` is `exponent` with an early `return -1` for a non-finite
  argument, so `exponent0(x) >= logb(x)` never reached the caller.
  Bracketing the way the `logb` channel does would not recover it: `m >=
  min(-1, max(logb x, emin))` is `-1` exactly when the value is large.  What
  does is a **guard-sensitive `_visit_return`** -- a return only a non-finite
  value reaches states nothing, the same reading `has_finite` already gives a
  return whose *value* has none.  `_only_non_finite` is the test and a depth
  counter over `_visit_if1`/`_visit_if` carries it.  It reads a condition
  through the temporary lowering hoists it into, as `_zero_paths` does, since
  a rule that switched off under lowering would move a compiled program's
  storage type.  Seed **91 -> 26**, and 25 of `exponent0`'s returns dropped
  across the corpus.

  The alternative is a **reachability** argument, which the block supplies
  locally: `gtr_fdpa_block` guards with `any([not fp.isfinite(p) for p in
  prods])` before the `fused_sum`, so every path reaching the position built
  from `es` has finite `a` and `b`.  That reason is stronger -- it drops the
  `-1` only where it is really dead, rather than at every call site -- but
  the guard sits *after* the `es` loop that states the constraint, so using
  it means re-stating a fact downstream of a guard: path sensitivity, and
  much larger than this rule.  Unmeasured; see
  [Reachability is a second filter](#reachability-is-a-second-filter).

The two that do not:

- **`amd.cdna1.{bf16,f16}`** -- `make_e_fdpa` adds its products with no
  truncation at all, so `e_fdpa_block`'s `s` wants **524** significand bits
  against a widest storage of 53.  The bound is right and unstorable: the model
  asks for an exact accumulator.  That wants a program transform -- the result
  is rounded to FP32, so a guard/round/sticky form computes the same answer --
  rather than a sharper analysis.

Three things the corpus taught that were *not* analysis gaps, kept because each
cost a day to find:

- **A sentinel becomes a scale exponent.**  `E_ZERO_TR = -999` only had to sit
  below the smallest subnormal product; the slack was free until `RescaleFixed`
  turned the alignment position into `2 ** -k` and asked for a `2 ** 1023`.
  Both AMD factories take `e_zero` as a keyword now, defaulting to
  `expmin_a + expmin_b - 1`.
- **A slot check asked about types where only values had to fit.**
  `_try_widen` used `scalar_fits_in`; `_value_fits` exists to say that is the
  wrong question.  A rescaled rounding hands the scale-out an `int64_t` holding
  29 significand bits, which no float slot accepts by type and every one
  accepts by value.
- **A pinned parameter stayed in the signature.**  Substituting a value into
  every use leaves the parameter dead, and a dead parameter has nothing to
  infer a type from.  Dead parameters of private specs are dropped once the
  specs settle -- which is also how the seed-misalignment bug above got in.

Not yet reached: `inline` declines a function with two returns, which every
model has.  Inlining is not needed now that specialization carries formats, but
it is the other route.

## Audit findings

Three audits, after the corpus was brought to 13/16 by following one blocker at
a time.  The worry that prompted them -- that the result is a pile of special
cases rather than a design -- is borne out in one specific way: **the same
concept is repeatedly implemented in one of the two places it occurs**, because
FPy lowers programs and only the lowered form was ever exercised.

Each item says who confirmed it.  "verified here" means reproduced
independently of the audit that raised it; "reported" means the audit's
evidence only.

### Unsound

- [x] **Seed misalignment across a dropped parameter.**  `_drop_dead_args`
      rebuilds a spec with fewer parameters *after* `_expand` captured its
      seed, and `analyze` binds terms positionally with `zip`, which truncates
      in silence.  **19 misaligned seeds on the corpus today** (verified here);
      they survive only because every dropped parameter happens to be the last,
      so the truncation is harmless.  Drop any earlier one and every following
      parameter inherits the previous argument's terms.
      *Fixed:* `_drop_dead_args` takes `seeds` and reindexes alongside the
      parameters it drops, and `analyze` raises on an arity mismatch rather
      than letting `zip` truncate.  19 -> 0 on the corpus.
- [x] **One spec, two seeds.**  `_bounds_fingerprint` hashes only `by_def`, so
      two call sites whose *relational* context differs share a spec, and the
      seed is whichever `_expand` reached first.  Verified here: a callee with
      no assignments, called once with its rounding position tied to the value
      rounded and once untied, yields a single spec with one name.
      *Fixed:* `_bounds_fingerprint` covers `by_expr` as well -- the two maps
      together are what the backend reads, so specs agreeing on both emit the
      same code.  `by_expr` contributes its bounds without their keys:
      rendering every expression to key them cost 40% of compile time and
      what separates two callers is the bounds either way.
- [x] **A zero-trip loop takes the body's element.**  `_visit_for` gives a
      *list* phi the body's terms unconditionally.  That is vacuous when the
      pre-loop value is an `Empty`, which is the case `_carried` was written
      for -- but wrong when it is a real list.  The scalar branch beside it
      already joins with `pre`; the list branch skips it.  (reported)
      *Fixed:* shares only where zero trips really is vacuous -- an `empty`
      has no element, and a *covering* write runs as many times as the list is
      long -- and joins with `pre` otherwise.
- [x] **Floor/Ceil/RoundInt/NearbyInt are off by one.**  The rule is
      `logb(round_to_int(a)) <= max(logb a, 0)`, but only `Trunc` rounds toward
      zero: `ceil(1.75) = 2` has `logb 1` where the rule claims `0`.  True by
      arithmetic.  *Fixed:* `Trunc` keeps the old rule, the rest carry.
- [x] **`_round_will_carry`'s RTO test proves the wrong direction.**  It asks
      whether the value *always* truncates to zero, where the no-carry branch
      needs *never*.  When neither is provable it takes no-carry and states
      `le(t, src)`, false whenever the value falls below one quantum.  (reported)
      *Fixed:* `RTO` carries with the rest.  Taking the exception needs proof
      the value *never* truncates, and the store answers the greatest
      precision, not the least -- so the honest options were "carry" or a
      mid-walk query that makes the rule depend on emission order.  Dropping
      the query also removes that order dependence.
- [x] **Every magnitude rule ignores the ambient rounding context.**  `Add`,
      `Abs`, `Mul` and the rest state the bound for the *exact* result; in FPy
      the expression denotes the result rounded at the enclosing context, which
      can carry out of that binade.  `Round`/`Cast` are the only cases that
      model a carry.  Systemic rather than per-rule, and it bites hardest in
      the target setting -- arithmetic inside a coarse `MPFixedContext`.
      (reported; the largest item here)
      *Fixed:* `_ambient` applies the active context once, centrally -- the
      rules bound a fresh term and the expression reaches one binade further
      -- and puts the quantum under the grid, since a rounded value has no
      digit below it.  The grid floor pays for the carry: the corpus is
      unchanged at 13/16.
- [x] **A refined bound escapes the path it holds on.**  `_visit_call`
      emits `le(sub.ret.logb, hi)` from the call site's own range, but for a
      callee that returns an argument verbatim `sub.ret.logb` *is* the caller's
      variable for that argument, and `hi` may be branch-refined.  (reported;
      the attribution was wrong, and diagnosing it found a *second*, larger
      leak of the same shape.)
      *Diagnosed:* `_visit_assign` bound a definition to its expression with
      an **equality**, so the definition's own seed -- which branch
      refinement narrows to the path the assignment sits on -- was stated
      about the expression, whose term every use shares.  `y = x` under
      `if x < 1 and x > -1` pinned `x` to `|x| <= 1` *everywhere*: reported
      `max logb(x) = 0` against FP64's 1023.  That masked the `_visit_call`
      one, which is real and separate.
      *Fixed:* the assignment states a one-directional bound, and the call
      site states its bound only on a term the callee minted -- not on one it
      was handed.
- [x] **Stochastic rounding is not modelled.**  `_round_will_carry` decides on
      `rm` alone; stochastic `RTZ` rounds *away* from zero.
      `MPFixedContext.is_stochastic()` exists and neither analysis calls it.
      (reported; no corpus design uses one)
      *Fixed:* `_rounding` reports no mode for a stochastic context, and an
      unresolved mode is already assumed to carry.
- [x] **`_tighter` drops the special values.**  `alt` is built with the
      infinities, NaN and negative zero cleared, and containment accepts it.
      All 1610 takes on the corpus drop at least `has_neg_zero`, which is what
      keeps a value off the integer rungs of the C++ ladder.  `has_finite`
      exists but is consulted only at `_visit_return`.  (reported; no exploit
      built)  *Fixed:* `alt` keeps whichever specials the incoming format
      admits instead of clearing them.
- [x] **`_forced_zero` reads a single-element test as a whole-list fact**, and
      **`_vacuous` sweeps expressions from other iterations and branches**.
      (reported, reasoning only)
      *Fixed (the first):* a `Sum` is zero only where the atom is zero on
      *that* path **and** came from an `all(...)` over the whole list --
      `xs[i] == 0` says nothing about the other summands.  Getting the
      per-path half wrong cost a design until caught, which is the argument
      for both halves.
      *Fixed (the second), and not as reported:* the **sweep is inert**.
      `ge_min` is a disjunction, so a disjunct accepted from another branch or
      iteration only ever weakens it and can never falsify the constraint --
      a dominance filter would have closed nothing.  What breaks is the
      *base case*, the guard's own subject, against `value_of(Logb)`: that
      rule floors a `logb`'s term at its argument's minimum exponent
      **path-insensitively**, so reading `logb(x)` in the sibling arm floors
      `msb(x)` on the very path where `x` is zero and has no exponent.
      `_vacuous` then offers `msb(x) - 1` as a disjunct and the merge is pushed up
      to that floor.  Two rules, each right alone, contradicting each other.
      Verified end to end: a kernel guarding `x == 0` with a sentinel
      exponent and rounding an *unrelated* value emitted `float` where
      `double` was needed and returned `65536` for `65535.999999940395`.
      The corpus escaped by numeric accident -- `nv.py:157` is the same
      shape and survives only because its `e_zero` sits above that disjunct's
      floor.  `_usable` now states such a disjunct only while the store admits it at
      or below the `then` arm, and `_check_vacuous` re-asks once the walk is
      over, the mid-walk read being the one thing that could go stale
      (never does on the corpus; deferring the emission instead costs a
      design).  13/16 -> 14/16 unchanged; regression test
      `test_the_sentinel_stands_when_another_value_is_rounded`.
      **The root is not localisable in this domain**, which is why the
      check sits at the merge rather than at the floor.  `logb(x)` *is*
      `msb(x)`: the read denotes no new quantity, so "x is non-zero on the
      path through this read" has no term of its own, and a store with no
      notion of path states it about every path or not at all.  Minting a
      per-read term does not help -- tie it below and the floor leaks
      through anyway, tie it above and the position is overstated, which
      understates precision.  Nor can the floor simply go: measured, the
      corpus still reaches 14/16 without it, but 30 unit tests do not, and
      they are the core capability (`TestSelfAnchoredRounding`,
      `TestAnchorsAcrossACall`, the guarded-exponent anchor).  A real fix is
      a path-sensitive store, which is the same redesign
      [symbolic-exponent-inference.md](symbolic-exponent-inference.md)
      contemplates for the relational half.  Until then the floor stays
      stated too widely and the one place its overreach is detectable
      refuses to build on it.

### Latent

- [x] **`_logbs`/`_grids` drop operands where `_join` refuses to.**  A
      `le_max` over a strict subset of operands is unsound; `_join` guards
      all-or-nothing and the emitters do not.  Instrumented over 2850 tests:
      every drop was all-or-nothing and non-real, so no live bug -- but the
      guard is missing and the `Floor` family's `+ [0]` already turns an
      all-drop into a bound rather than silence.
      *Fixed:* both are all-or-nothing now, as `_join` is.

### Capability holes -- the same concept in one of two places

**Out of scope for now** -- these lose precision but are sound, so they are
left for a later pass.

- [x] **`Range1` in a comprehension.**  `_visit_for` registers `for i in
      range(len(xs))` as covering; `_visit_list_comp` had no such case.
      Verified here: the identical computation gives **prec 9** as a loop or as
      `[f(x) for x in xs]`, and **285** as `[f(xs[i]) for i in range(len(xs))]`
      -- which is the form `ZipElim` itself emits.
      *Fixed* along with index-set instantiation, which needs the same two
      readings of a range in both places to hold of the comprehension form.
- [ ] **`zip` in a loop.**  The exact dual: `_visit_list_comp` handles
      `(TupleBinding, Zip)`, `_visit_for` does not.  `for a, b in zip(A, B)`
      appears in the corpus source.
      *Fix for both:* one shared clause-binding routine over all three
      iterable shapes.
- [ ] **`ListSlice` is unhandled everywhere.**  A slice's elements are a subset
      of its source's, so inheriting the summary is unconditionally sound.  The
      models are full of `A[i:i+L]`; they survive only on the format-domain
      seed.
- [ ] **`And` in `_zero_paths`.**  `Or` is handled, `And` is not, so a strictly
      stronger hypothesis does strictly worse.  Also no `Not` / `!=`
      normalisation.
- [ ] **`Fma`, and the `lsb` rule for the integer-valued ops** -- both specified
      in the rule tables above and never implemented.  The `msb` half of the
      integer-valued rule *is* implemented and never executes.
- [ ] **No `_visit_while`.**  `_visit_for` carries ~35 lines of phi merging and
      `_partial`/`_fields`/`_elt_expr` propagation; `WhileStmt` gets none.
      Sound, and mmasim has no `while`.

### Comments and dead code

- [x] **The index-set comment is false.**  It says `xs[i]` "always *inherits*
      the summary's bounds whatever `i` is"; the code gates inheritance on
      `_covers` too.  Verified here: covering **9**, non-covering **285**.  The
      comment describes a design that was measured, found to buy nothing on the
      corpus, and reverted -- then written up as if it shipped.
- [x] Two more stale comments: the zero-arm comment credits `_merge_returns`
      with a drop that lives in `_visit_return`, and `_merge_returns`' own
      docstring describes a fast path that never runs.
- [x] **Dead arms.**  Re-measured after the fixes, over the corpus *and*
      `tests/unit/{analysis,transform,backend}`: **17 unhit executable lines
      of 471**, and the audit's list had gone stale -- `Copysign`,
      `Mod|Fmod|Remainder` and the `Floor` arm are all reached once the
      backend tests are included, and `Exp2` in `_exp2_arg` is reached by the
      new rounding tests.  What is still unhit is defensive: `bounds()`'
      open-value guards, the seed-arity raise, the public entry's type check.
      Those are soundness guards and stay -- untested is not unreachable.

      Removed: `_merge_returns`' same-term path, a redundant phi branch in
      `_universal_zeros`, and `format_infer`'s own `_round_will_carry` -- 36
      lines with **no call site at all**, carrying the RTO argument this
      audit has since disproved.  Also `store.prec_at` lost its only caller
      when RTO was fixed; it is still exercised by `test_digit_bound.py` and left
      as store API.

      Worth knowing: the **symbolic-rounding-position path is dead on the
      corpus** -- `RescaleFixed` turns it into a `Pow(2, k)` scale -- so the
      branch this analysis is named for is kept alive by unit tests alone.
- [x] **`Or`/`AllOf` in `_zero_paths` serve one source line** (`gst_fdpa`'s
      guard, 8 firings corpus-wide).  Not wrong; should say so.

### Performance -- after the above

**74s to 20s on the corpus**, a 73% cut, and **5,672 z3 invocations to
3,138**.  Half the remaining time went to a change that touches no solver at
all -- see *A spec key is a value*, below.  The starting point is itself above the 81.7s measured before the
audit: the non-finite-path rule leaves more bounds finite, so `_tighter` has
more to ask about -- 14,996 asks against 8,229.

Every win came from *which question is asked, and when*.  Nothing that made
the solver itself faster survived measurement.

- [x] **A spec key is a value, not a string.**  36s -> 20s, and no solver
      call changed.  `_SpecKey` held SHA-1s of rendered formats, so every
      callee at every fixpoint round paid to `repr` each of its bounds and
      sort them.  The fields are now the values themselves -- `_Pin` for what
      an argument type pins, a `frozenset` of `_DefBound`/`_ExprBound` for
      what the caller derived inside -- so equality is structural, exact, and
      order-independent by construction.  A string survives only in
      `_mangle_private`, which labels the emitted symbol and is the one place
      a key becomes text.
- [x] **Run the relational half only where it is read.**  40s -> 36s.
      `FormatInfer.analyze` takes `digit_bounds`, **off by default**: a caller that
      reads a relational bound asks for it.  Only two do -- `Specialize`,
      whose spec key carries those bounds so two callers with different
      context do not share a spec, and the emitter, where storage selection
      is the one consumer.  `RoundElim` and the rounding rewrites in
      `transform/utils.py` do not: eliminability is a question about a
      rounding's *operand*, which the non-relational pass settles alone.
      Verified byte-for-byte -- with the hash seed pinned, the emitted C++ is
      identical either way.
- [x] **Ask a decision question before an optimisation.**  The biggest one,
      74s -> 59s.  But not in the form proposed here: the condition suggested
      was `alt.prec <= cur.prec`, and measured over the corpus that settles
      only **4.8%** of the asks.  Containment fails on the *quantum* or the
      *bound* far more often -- 32.5%, against 10% that are used at all.
      `DigitBoundAnalysis.escapes` asks those two before `bounds` costs three
      optimisations.  Verified exact rather than argued: every call was run
      both ways over the corpus, **5,641 skips, zero disagreements**.
- [x] **Ask the disjunct that settles it first.**  46s -> 40s, and 5,672 z3
      invocations -> 3,925, from reordering two lines.  `escapes` stops at
      the first question that reaches -- and the magnitude settles **1,747**
      of the 1,778 skips against the grid's **31**, so asking the grid first
      bought a wasted solver call almost every time.
- [x] **Give a component a plain solver beside its `Optimize`.**  57s -> 54s.
      A `check` does not need `Optimize`'s machinery and is markedly faster
      without it -- 6.5ms to 3.8ms per decision.
- [x] **Merge components that are asked about together.**  59s -> 57s.  Over
      half the queries named two components -- an `msb` and the `lsb` beside it
      are related by nothing but the question -- and an objective spanning
      several matched no cache, so its solver was rebuilt every time.
- [x] **Bisection, which now wins.**  54s -> 46s, **all 1,305 answers
      identical** to `Optimize`'s.  This *reverses* the earlier reading here.
      Not the engines but the encoding: refutation reuses the plain solver
      the decision questions already built, where `Optimize` is a second
      encoding of the same component.
- [x] ~~Dedupe `le_max`'s disjuncts~~ -- **zero** of the corpus's 7,306
      `<=max` constraints have a repeated arm now.

Measured and **rejected**, each with the number that killed it:

- `smt.arith.solver=2`: 53s -> 49s on the `Optimize` path, **nothing** over
  bisection (43.5s against 43.3s).
- **Bounding the integers** so branch-and-bound is finite: **no gain** (44.5s
  -> 44.1s at a box of `2**40`, *slower* at `2**20`).
- **All three of `bounds`' objectives in one box-mode `Optimize`**: **61s
  against 39s**.
- **`mag - exp + 1` instead of the third query**: **3/16**.  The three maxima
  really are independent, and the precision is the one that is load-bearing.
- **A precision disjunct in `escapes`**: it *would* be exact -- the escape
  clause for a value inside the format's subnormal region needs `mag - exp +
  1 <= prec`, and `bounds` already caps its answer there, so the clause
  cannot fire on `alt`.  But it is a wash: **3,161 calls against 3,138**,
  trading 45 optimisations for 68 checks.  Not worth the code.
- **Interval propagation as a pre-filter**: sound where it fires and settles
  **488** of 2,620 threshold questions -- but only negative ones, and an
  over-approximation cannot prove a value *reaches* a threshold, which is
  1,778 of them.
- **Memoising the analyses**: 645 calls over 176 functions, only **75 exact
  repeats**.  The query memo already earns its keep -- clearing it on every
  one of 79,145 constraints costs nothing, because **no answer is ever
  recomputed after a clear**: the walk states its constraints and asks
  afterwards.

Two things learned that are not performance matters:

- **The emitted code was not reproducible across runs**, which made a
  byte-comparison of two builds useless until the seed was pinned.  A spec's
  mangled name is a digest of its bounds, and `SetFormat`'s dataclass `repr`
  rendered a `frozenset` -- whose order follows element hashes, and a
  `Special`'s hash is salted per process.  `SetFormat.__repr__` sorts now.
  Bodies were always identical; only the names moved.
- **Specialization runs to a fixpoint, not a fixed number of rounds** --
  two or three on the corpus, capped at eight.  The second round always
  renames without changing how many specs there are (`+N -N`, every design),
  which looks like churn and is not: stopping after the first gives
  **3/16**.  Round one specializes, round two re-derives each callee's
  bounds now that it has a spec of its own, round three confirms.

  What a round is compared *by* is now the `_Instance`s it produced rather
  than the names it gave them.  A name would nearly do, being a digest of
  the instantiation -- but a public entry keeps its own name, so however
  much its instantiation sharpened the comparison could not see it, and the
  loop could stop a round early.  Same two-or-three rounds on the corpus,
  and the same output; one fewer thing decided by a string.

Where the remaining 36s sits: 3,138 solver calls, 1,068 full optimisations
from `bounds` and 2,070 capped decisions, together about half the time.  The
other half is the two analyses walking the program.

`Solver` keeps exactly two methods.  The decision question is *not* a third:
`maximize` takes a `cutoff`, which says the caller will not distinguish
anything at or above it, so the backend may stop at one refutation and answer
`inf`.  Every answer stays an upper bound, so a backend that ignores the
cutoff is still correct -- which keeps the interface honest rather than a z3
feature leaking through a `getattr`.

Still open: **87% of the binade runs that remain change nothing** -- 447 of
511 produce formats identical to the first pass alone.  Neither cheap
predictor tried separates them: "the first pass left a `RealFormat`" misses
61 of the 64 that matter, and "the function computes a context at run time"
misses 31.  A predictor would be worth more than anything left on the solver
side.

## Open questions

- Integer storage requires `RTZ`, so a rescaled rounding under the default
  `RM.RNE` is refused. The restriction is right for arithmetic under the
  context, but the only operation under a rescaled one is the rounding, which
  `_emit_integral_value` already spells in every mode — `nearbyint` then cast.
- `RoundAt` is deliberately **not** handled: it names its position as an
  operand rather than through a context, so neither the scope-shaped rule nor
  the operand-independent grid rule applies as written. `Cast` at a symbolic
  position, and `MPBFixedContext` where the bound is concrete while the
  position is not, are still open.
- `Logb` is the only introducer. `frexp`-style splitting and a `RoundAt`
  position name an exponent too.
- The sum's slack comes from bounding each term independently. Is there a cheap
  way to exploit that the terms' exponents are spread, or should it be accepted?
- Symbolic *lower* bounds on a magnitude are unimplemented, by the argument in
  [What is deliberately not tracked](#what-is-deliberately-not-tracked). The
  omission is principled, but `mmasim` never asks for the other direction, so
  the design has not had to confront a program that does.
- Where should the derived precision be *stated* — inferred at every use, or
  materialized once into the AST so later passes need not re-derive it?
