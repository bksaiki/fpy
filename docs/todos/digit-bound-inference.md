# Digit-bound inference

Relational bounds on the precision of a value rounded at a run-time position
(`fpy2/analysis/digit_bound/`), read by format inference as a reduced product.

- [Part I: semantics and soundness](#part-i-semantics-and-soundness) -- the
  contract: the domain, its concretization, the transfer rules, the soundness
  theorem, and the encoding the solver sees.
- [Part II: design and history](#part-ii-design-and-history) -- how the
  analysis was built for `fused_sum` and the mmasim corpus, with the
  measurements that shaped it.
- [Part III: status and plan](#part-iii-status-and-plan) -- where the
  implementation departs from Part I, and what changes.
- [Part IV: implementation plan](#part-iv-implementation-plan) -- the phases
  that bring the implementation to Part I, one commit each.

**Status.**  `fused_sum` reaches `F + 2`, and the mmasim corpus compiles 60/62
to C++.  Known unsound since v0.2.4: an upper bound on an aggregate stated on a
list's summary ([Where the implementation departs](#where-the-implementation-departs)).

## Part I: semantics and soundness

### 1. Problem

A rounding under a context whose position is computed at run time,
`with MPFixedContext(n): y = round(x)` with `n = max(es) - F - 1`, has a format
no non-relational domain can bound: `n` ranges over hundreds of values and so
does `x`'s exponent, but they move *together*.  The analysis bounds the
**precision** of such a value, the number of digits between its leading digit
and its last, by tracking linear relations between integer exponents.

### 2. Concrete semantics

For a finite real `v`:

- `E(v) = floor(log2 |v|)`, its leading digit (`logb`); `E(0) = -inf`;
- `G(v)`, its last digit: the largest `g` with `v` a multiple of `2^g`;
  `G(0) = +inf`;
- its precision `P(v) = E(v) - G(v) + 1` for `v != 0`.

A program runs over **dynamic instances**.  A definition `d` nested in binders
`b1, ..., bk` (loops and comprehension clauses) takes one value per iteration
vector `i = (i1, ..., ik)`; write `d[i]`.  A list value is a finite map from
indices to elements.  An execution `rho` assigns every dynamic instance its
value.

Only finite values are described.  NaN and the infinities fall outside the
concretization by construction (see [The abstract domain](#the-abstract-domain)
in Part II); a program needing them in range guards them first.

### 3. The abstract domain

Two layers.  The first is the standard, always-safe summarization of a list by
one element; the second is a pointwise relation between lists, which needs the
lists' indices to correspond.  The analysis needs both: `fused_sum`'s `F + 2`
rests on a pointwise fact, `msb(prods[i]) <= es[i] + 1`.

#### Variables

The analysis works over integer variables:

| entity | term | denotes |
|---|---|---|
| a real definition or expression `v` | `msb(v)` | `E(v)`, exactly |
| | `lsb(v)` | a lower bound on `G(v)` |
| an integer definition or expression `n` | `val(n)` | `n`, exactly |
| a list `xs` | the terms of its element | an element's |

`msb` is exact, not an upper bound, because `logb` reads it as a value:
`val(logb(x)) = msb(x)`.  The rules therefore state *valid relations between
exact exponents*, and a query reports a maximum over them.

#### Layer 1: summaries

A list's element terms are a **summary**: one set of variables standing for
*any* element, as summarized dimensions do in a numeric domain (Gopan, DiMaio,
Dor, Reps and Sagiv, "Numeric domains with summarized dimensions", TACAS 2004).
A constraint naming a summary holds for every element; one naming two summaries
holds for every *pair* of elements, one from each.  This reading is safe by
construction, and states nothing that depends on which element is which.

A variable minted inside a loop or comprehension is a summary of that binder's
iterations in the same way: it stands for any iteration.

#### Layer 2: pointwise relations

An **index space** is an equivalence class of binders (a loop, a comprehension
clause) and lists, under **correspondence**: a binder that iterates a list, or
reads or writes it at its own index in order (`_covers`), corresponds with it,
and so does a list it builds.  `es = [logb(x) for x in xs]` puts `xs`, `es`
and the comprehension in one space.

A constraint naming summaries of one index space holds **pointwise**: for every
index, the elements at that index.  Correspondence is the whole premise, so it
is established only where the program states it, never inferred from a length
or a depth; without it, a constraint over two summaries has its layer-1
reading.  This is array-content analysis with a shared index (Halbwachs and
Peron, "Discovering properties about arrays in simple programs", PLDI 2008;
also Cousot, Cousot and Logozzo, "A parametric segmentation functor", POPL
2011).

Each variable has a **signature**, the index spaces it is a summary over;
constraints are read universally over their variables' signatures, variables
sharing a space at the same index.

#### Abstract states

An abstract state is a pair `(Phi, T)`:

- `T` maps each entity to its terms;
- `Phi` is a finite set of constraints, each `t1 <= t2`, `t1 <= max(ts)`, or
  `t1 >= min(ts)` over affine terms, optionally **guarded** by literals: boolean
  variables, themselves with signatures, standing for facts such as "`x` is not
  zero".

#### Concretization

An execution `rho` is in `gamma(Phi, T)` iff there is an assignment
`sigma`, one integer per variable per point of its signature, one boolean per
literal per point of its, such that

1. `sigma` is the **exact** assignment: `sigma(msb(v))[i] = E(rho(v[i]))` and
   `sigma(val(n))[i] = rho(n[i])` wherever those are finite, `sigma(lsb(v))[i]
   <= G(rho(v[i]))`, and every literal is the truth of its fact;
2. every constraint of `Phi` holds under `sigma` at every point, where it is
   guarded only at points its literals hold.

A zero's `E` is `-inf`, which no integer is; a constraint that would bound it
below is guarded by its "not `-inf`" literal (`not_neg_inf`), and `sigma` takes
an arbitrarily low integer there.

### 4. Transfer rules

Each rule adds constraints to `Phi`.  A rule is **valid** if every execution in
the concretization before it is in the concretization after: the constraints it
adds hold of the exact assignment, at every point.

#### Pointwise facts

Relate one dynamic instance and its operands, at the same point: the operator
table of [Operator coverage](#operator-coverage).

| construct | constraint |
|---|---|
| `a * b` | `msb <= msb(a) + msb(b) + 1` (`+ 0` with a power of two); `lsb >= lsb(a) + lsb(b)` |
| `a + b`, `a - b` | `msb <= max(msb(a), msb(b)) + 1`; `lsb >= min(lsb(a), lsb(b))` |
| `round(x)` at position `n`, mode `rm` | `msb <= max(msb(x) + c, n + 1)`; `lsb >= n + 1`, `c` = 0 for `RTZ`, else 1 |
| `round(x)` to a float format, mode `rm` | `msb <= msb(x) + c` |
| `logb(x)` | `val = msb(x)` |
| `2 ** n` | `msb <= val(n)`, `lsb >= val(n)` |
| `max(a, b)` of integers | `val >= val(a)`, `val >= val(b)` |
| seeds from format inference | `msb(v) <= hi`, `lsb(v) >= lo` |

Each is a theorem about `E` and `G`.  Operands inside one binder share its
index space, so the fact is pointwise there, which is what makes `prods` and
`es` pair.

#### Aggregates

A reduction over a list relates one value to all of its elements, in one of two
directions:

- **Envelope**, the aggregate bounding every element: `m = max(xs)` gives
  `val(m) >= val(xs)`, true of every element, so a layer-1 fact.
- **Upper bound**, every element bounding the aggregate:
  `E(sum xs) <= max_i E(xs[i]) + ceil(log2 n)`.  This is a fact about the
  *worst* element, and no reading of the summary states it:
  `msb(sum xs) <= msb(xs) + k` claims it of every element, which is false.  The
  valid form names one element, existentially: the summary is **expanded**
  (Gopan et al.'s `expand`) into a fresh copy, the witness, and the bound is
  stated on it.  Each aggregate gets its own witness.  `sum` and the `msb` of
  `max` and `min` over a list take this shape.

#### Element reads

`xs[e]` reads the summary where `e` is the index of a binder in `xs`'s space,
and an expanded copy anywhere else: a constant `k`, an affine `a + s * k`, a
part `range(a, b, s)`.  Two reads at one point share a copy, which keeps them
paired.

#### Control

- **Branches** guard the facts their arms state by the arm's literal; a
  failed test that names a value's zero is a literal on that value.
- **A loop of static count that carries a real value** is unrolled: each
  iteration is its own instance, and the carried value relates consecutive
  iterations pointwise.
- **Any other loop** is one generic iteration, a summary over its iterations.
  A value it carries relates iteration `j + 1` to `j`, which no constraint over
  one iteration states; it takes only its seeds.
- **Calls** instantiate the callee at the call site: callee variables get the
  call site's signature plus the callee's own binders, and arguments share
  the caller's terms.

### 5. Queries and soundness

The query at a rounding `y` is

    prec(y) = max { sigma(msb(y)) - sigma(lsb(y)) + 1 : sigma satisfies Enc(Phi) }

where `Enc` is the quantifier-free encoding of section 6.  It materializes back
into format inference as a precision, met with the format there.

**Theorem (soundness).** Let `rho` be any execution of the program on inputs in
their declared formats, and `y[i]` any dynamic instance of a rounding with a
finite non-zero value.  If format inference is sound (its bounds contain every
value), then `P(rho(y[i])) <= prec(y)`.

*Proof sketch.* By induction over the walk, `rho` is in the concretization of
every state, by the validity of each rule.  Section 6's lemma gives an
assignment of `Enc(Phi)` agreeing with the exact assignment at `i`, so the
maximum is at least `E - G + 1` there.

### 6. The encoding

The solver takes quantifier-free linear arithmetic, so a summary is one set of
variables, its **representative**, and a copy (an element read, a witness) is
another set: the variables whose signature contains the space renamed, every
constraint over them replayed.

**Lemma (copies).** For any execution in the concretization, any choice of
points for the representative and each copy, the exact assignment at those
points satisfies `Enc(Phi)`.

*Proof.* Every constraint holds universally, so at every point; a copy is the
constraints at its point; a witness's point is its aggregate's worst element.
Guarded constraints are replayed only under literals that hold at every point or
none.  No rule relates two copies of one space.

The lemma is what the signatures are for: a copy must rename exactly the
variables that vary over the copied space.  Renaming fewer shares a value
across points, which is unsound; renaming more loses a pairing or an envelope,
which is safe.

## Part II: design and history

Written as the analysis was built; a number is from then unless it says
otherwise.

### The shape

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

### The target, measured

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

### What the callee needs: one number

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

### Everything non-relational is already tight

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

### The derivation

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

### The abstract domain

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

Concretization: [Part I](#concretization) states it, with `msb` *exact* --
`logb` reads it as a value -- rather than the upper bound first written here.
Every `v` must be finite; an unsatisfiable store is bottom (unreachable code),
and the empty store is top, meaning `RealFormat()` everywhere.

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

### The constraint store

The constraint language is:

- **definitional equalities**, one per exponent-valued SSA definition, each an
  affine combination of earlier ones (`es = ea + eb`, `n = e_max - F - 1`);
- **one-sided inequalities** from `max` and `min` (`es.elt <= e_max`);
- **anchors** attaching a value's magnitude bound to a term (`logb(p) <= es + 1`).

The only query is an upper bound on a single term. Everything stays in
*exponent* space — that discipline is what keeps `2^x` out of the encoding —
so the store is linear integer arithmetic over a handful of variables.

#### What is deliberately not tracked

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

#### Asking for a precision needs a third field

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

#### Emit, then query

Constraints are emitted while walking the AST; precisions are queried at rounding
sites. That split is the architecture regardless of what answers the query, and
it is what keeps the solver out of `FormatInfer`'s widening loop — solver
answers are not a lattice. Known loop counts unroll (`ArraySizeInfer` supplies
them here); unknown counts would need invariant inference, which a solver alone
does not provide.

#### Feeding back into format inference

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

#### What the store holds

The backend now answers by bisection (`Z3Solver.bisect`); `Optimize` is kept
only for the differential test.

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

### Operator coverage

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

#### Having a rule is not the same as doing the job

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

#### Reachability is a second filter

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

#### Why `Add` needs no alignment

Addition is the operator a relational value domain handles worst: adding two
symbolically scaled quantities requires their scales to agree, and in general
they do not. In exponent space the problem disappears —
`|a+b| <= 2·max(|a|,|b|)` gives `msb(a+b) <= max(msb(a), msb(b)) + 1`
unconditionally, with no alignment and no fallback. What alignment would buy is
the *lower* bound, which cancellation destroys and which this analysis never
queries. That asymmetry is why the coverage above is broad rather than
corpus-shaped.

#### Where the corpus really does under-constrain the design

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

### Degradation

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

### Element-wise summaries

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

### Prior work: SELTZO

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

### What landed

`fpy2/analysis/digit_bound/` — an affine constraint store over integer
exponents (`store.py`, `solver.py`) and a walk that states the constraints
(`infer.py`), read by `FormatInfer` under `use_digit_bounds` as a reduced
product: the non-relational pass supplies the seeds, the relational one the
precisions. `Specialize` keys a spec on the bounds its caller derived, so a
callee analyzed for two callers does not share one answer.

Contexts keep their shape across a call (`PartialContext`), and a call site
pins what it knows in both directions, which is what lets a callee's rounding
cancel against a position its caller built.

### The mmasim corpus

`examples/mmasim/compile.py` compiles all 16 designs and reports where each
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

### Audit findings

Three audits, after the corpus was brought to 13/16 by following one blocker
at a time. The worry that prompted them -- that the result is a pile of
special cases rather than a design -- is borne out in one specific way: **the
same concept is repeatedly implemented in one of the two places it occurs**,
because FPy lowers programs and only the lowered form was ever exercised.
Everything the audits found is fixed except the holes below, which lose
precision but are sound.

#### Capability holes -- the same concept in one of two places

**Out of scope for now** -- these lose precision but are sound, so they are
left for a later pass.

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

### The floor under `logb` holds only where the argument is non-zero

`logb(0)` is `-inf`, which the integer store stands in for with an arbitrarily
low value, so a lower bound on a term that may be `-inf` holds only under the
literal saying it is not (`DigitBoundStore.not_neg_inf`).  `value_of(Logb)`'s
floor at the argument's least exponent is stated under it, and so is a seed
`_fresh_value` reads off a format that admits `-inf`.  Two things assume it:

- a `with` building a fixed-point context at a position affine in `logb(x)`,
  which is undefined at `-inf`, for everything the `with` dominates;
- the `else` arm of a failed `x == 0`: `_merge_arms` bounds the merge below by
  the lesser arm floor, each under its arm's facts -- a constant, which carries
  nothing onto the zero path.

Stated on every path, the floor contradicted the vacuous disjunct on the zero
arm, which cost every FP16-accumulator mmasim design at `e_zero = -133`, and it
was unsound wherever `logb(0)` is read in a defined way, as in
`max(logb(x), -1000)`.  The corpus now compiles 60/62.

Neither establishes a literal on a variable of a list's summary, which stands
for every element: a `with` at `logb(xs[i]) - k`, or a failed `xs[i] == 0`,
says nothing of the other elements.  And a failed test is read back only where
each zero it names makes the test true: not through a phi, which may be an
`and`.

#### Still open: the `+inf` mirror

`logb(inf)` is `+inf`, and the value channel reads `logb(x)` as `msb(x)`, whose
upper seed bounds only *finite* values.  So an upper bound read through a
`min` fails at `x = inf`:

```python
@fp.fpy(ctx=fp.REAL)
def hi(x, y):                       # x: FP16, y: FP32
    e = min(fp.logb(x), 1000)       # 1000 at x = inf
    with fp.MPFixedContext(e - 10, fp.RM.RAZ):
        return fp.round(y)
```

The round is bounded below `2 ** 129`, and `hi(inf, 1.0)` is about `2e298`.
Nothing in mmasim reaches it: every model sends a non-finite input down a
special-value path first.  The `-inf` fix does not dualise as is, since
`msb(x)`'s upper seed is also the magnitude bound on finite values.  The fix
separates `logb(x)`'s value from `msb(x)` where `x` may be infinite, tied by
the finiteness literal `_lit(d)` that already exists.

### Still open from the performance work

The corpus went 74s -> 20s and 5,672 z3 invocations -> 3,138, almost entirely
by asking the solver less often rather than by making it faster.

- **87% of the runs that remain change nothing**: 447 of 511 produce formats
  identical to the first pass alone. Neither cheap predictor tried separates
  them -- "the first pass left a `RealFormat`" misses 61 of the 64 that
  matter, and "the function computes a context at run time" misses 31. A
  predictor would be worth more than anything left on the solver side.
- The **symbolic-rounding-position path is dead on the corpus** --
  `RescaleFixed` turns it into a `Pow(2, k)` scale -- so the branch this
  analysis is named for is kept alive by unit tests alone.

### Open questions

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
  **Accepted.**  Summing per-term bounds rather than `(L+1)·max` gains nothing
  at the NVIDIA sites.  `L` is a power of two there, so `L·2^W + 2^(W-1)`
  needs `W + ⌈log2(L+1)⌉` bits either way.  Volta's 28 bits is attained:
  four maximal products plus `c` at `e_c = e_max`.  The measured 27 in the table
  above is a generator that never put `c` in the top binade.  What is left
  is about one bit in the GST designs, from mantissas counted per binade
  (`576 < 2^10`, `1.75² < 4`), which needs magnitude tracking within a binade.
- Symbolic *lower* bounds on a magnitude are unimplemented, by the argument in
  [What is deliberately not tracked](#what-is-deliberately-not-tracked). The
  omission is principled, but `mmasim` never asks for the other direction, so
  the design has not had to confront a program that does.
- Where should the derived precision be *stated* — inferred at every use, or
  materialized once into the AST so later passes need not re-derive it?

## Part III: status and plan

### Where the implementation departs

| | this page | `infer.py` today | consequence |
|---|---|---|---|
| what a variable varies over | its signature, by correspondence | `_elt_vars` and loop depth (`_var_level`, `_anchor_level`); correspondence by length (`_covers`) | pairing where no correspondence is stated, and copies renaming too much or too little: the outer-loop case in [The index-space mechanism](#the-index-space-mechanism) |
| an upper bound on an aggregate | stated on an expanded witness | stated on the summary (`Sum`, `AMax`/`AMin`'s `msb`) | reads the summary as the worst element, which no element is in general |
| a loop of static count carrying a real value | unrolled | walked once; the carried value is unbounded | the FMA chain |
| copies | keyed by point, renaming by signature | index sets keyed by range, renaming every per-element variable; constant and inline indices added this branch | as the first row |
| envelopes | a rule shape | a store flag (`envelope`) added this branch | agrees |

The second row is a **soundness bug**, present since v0.2.4.  Summarizing is
safe; reading a summary as the worst element is not.  It holds only while one
assignment can make the summary the worst element for every aggregate in a
solve, and two sums over paired lists whose maxima sit at different indices
break that:

```python
@fp.fpy(ctx=fp.REAL)
def f(xs):                                  # xs: list[FP32], length 2
    ys = [2 ** (-fp.logb(x)) for x in xs]   # msb(ys) = -msb(xs), paired
    s1 = sum(xs)                            # msb(s1) <= msb(xs) + 1
    s2 = sum(ys)                            # msb(s2) <= msb(ys) + 1
    p = s1 * s2                             # so msb(p) <= 3, whatever the element
    e = max([fp.logb(x) for x in xs])
    with fp.MPFixedContext(e - 10, fp.RM.RTZ):
        r = fp.round(p)
    return p + p + r
```

At `xs = [2^60, 2^-60]`, `s1` and `s2` are both about `2^60` and `p` about
`2^120`.  The store bounds `msb(p)` at 3, and format inference materializes it:
`p + p` is inferred below `2^5` and the rounding below `2^4`, neither of which
holds the value (`representable_in` is false).  The upper-only discipline does
not save it, since the product adds two terms whose worst elements differ.  A
single aggregate beside pointwise queries on its own list should not be broken
this way, since the maximum is free to put the representative at the worst
element -- an argument, not a test.  The fix is the expanded witness of
Part I, section 4.

### The index-space mechanism

How layer 2's correspondence and signatures replace the depth proxy in
`infer.py`; Part I's [Layer 2](#layer-2-pointwise-relations) is the semantics.

#### Today: the depth proxy

A list contributes one *element summary* to the store: variables that stand
for an arbitrary element, universally quantified over its index.  Three
places need to know whether a variable varies with some index:

| consumer | question | today |
|---|---|---|
| `_instance` (a part of a list: `xs[0]`, `xs[k + 1]`, a gather) | which variables does the part rename? | every variable in `_elt_vars` |
| `_elt_lsb` | is a list's grid uniform, or does it move with the list's index? | `_term_level(lsb) > _elt_depth` |
| `_grid_ge` | what a grid bound inherits from the terms bounding it | the deepest `_term_level` |

`_elt_vars` is one global set, every variable minted inside any loop or
comprehension, or a summary.  `_term_level` is a loop *depth*: `_var_level`
(the depth a variable is minted at) and `_anchor_level` (the depth of what
bounds it; a list summary sits one deeper than its list).

Depth stands in for identity, and two different index spaces at one depth
are conflated:

```python
for j in range(2):                        # depth 1: j
    es = [fp.logb(x) for x in xs]         # es, xs summaries: depth 1, xs's index
    t = xs[0]                             # a part of xs
    e = max(es)                           # depth 1: varies with j, not xs's index
    with fp.MPFixedContext(e - 12, fp.RM.RTZ):
        out[j] = fp.round(t)
```

Instantiating `{0}` of `xs` renames everything in `_elt_vars`, `e`
included, so the envelope `es_i <= e` reaches the part as `es@0 <= e@0` and
says nothing about `e`.  Renaming by depth instead ("at least as deep as the
list's summary") fails the other way: `e` and `xs`'s summary are both at
depth 1.  The precision is 288 where 12 is right.  The mmasim generator's tree
before CDNA3's separate join is this shape.

#### What an index space is

Covering is by length: `_covers` lets any `range(n)` loop read every list of
length `n` through the list's own summary variables, and two lists are paired
exactly when one loop body reads both.  So the identity is not a loop but the
class covering builds:

- every list definition has an **element space**;
- every binder (a `for`, a comprehension clause) has an **index space**;
- a binder that covers a list, or iterates it directly, *unifies* its index
  space with the list's element space (union-find);
- a list a binder builds -- a comprehension, an `fp.empty` filled by a covering
  loop -- has its element space unified with that binder's.

`es = [logb(x) for x in xs]` thus puts `es`, `xs` and the comprehension in one
class, which is the pairing the summaries already rely on.  `j`'s loop covers
nothing here, so it is a class of its own.

#### What a variable varies over

Each variable carries a set of classes, `spaces(v)`:

- **minted** inside binders `B1 .. Bn`: their classes;
- a list's **summary**: its element space, plus wherever the list itself
  sits;
- a **grid** bounded below by terms (`_grid_ge`, the `_anchor_level` role):
  the union of theirs, except that an lsb's own bound does not add its
  list's space -- the exception `_moves_at` already makes.  Other
  constraints, a `max`'s among them, propagate nothing, as today;
- a callee's, called from inside binders: the callee's plus the call site's
  -- `_raise_anchor(i, _elt_depth)` today.

An aggregate over a list -- `max`, `min`, `sum` -- is minted *outside* the
binder that built the list, so it does not vary over the list's space: `e`
above is `{j}`.  No rule removes a space; it is never added.

#### The consumers

- `_instance` for a part of a list in class `S` renames exactly the variables
  with `S` in their spaces.  `e` keeps its name, and the envelope reaches the
  part as `es@0 <= e`.
- `_elt_lsb` keeps a summary's grid when the list's own class is not among the
  grid bound's spaces: uniform over the list's index.
- `_grid_ge` unions spaces instead of taking a maximum depth.

`_elt_vars` becomes "has a non-empty set", and the depth fields go.

#### What does not change

- Covering stays by length, and what it claims stays the same; this only
  gives the claim a name.
- The store, its constraints, and every operator rule.
- Parts keep their keys, so `es` and `prods` over one range still share a
  renaming.

### Measurements, and the design they settle

Measured on the mmasim corpus (60 catalog designs and 16 remixes compiled to
Triton, 62 designs to C++), without changing the analysis:

- **No relation over a list of symbolic length is needed.**  With those
  relations disabled (`_bind_elt` and `_instance` declining a list whose
  length is unknown), all 76 kernels and all 62 C++ sources are byte for byte
  the same.
- **Expanding every static list and loop costs too much.**  Constraints grow
  10x (C++, median; 12.7x worst) and 4.1x (Triton), queries 12x and 5x, and
  the solver's corpus time an estimated 51 s to 300 s (C++) and 73 s to 220 s
  (Triton).  Rejected.
- **Unrolling only static loops that carry a real value is cheap.**  The C++
  solver time goes from 50 s to about 79 s (one design 3.4x), Triton's is
  unchanged: nearly every such loop is the symbolic one over instructions.

So the design is:

1. layer-1 summaries, read as any element, with aggregates bounded only through
   envelopes and expanded witnesses;
2. layer-2 pointwise relations, only between lists whose correspondence the
   program states, with signatures by correspondence rather than depth;
3. static loops carrying a real value unrolled, every other loop one generic
   iteration.

Summaries are what keep the solver's time where it is; signatures are what keep
them sound.  Parts, constant and affine reads, and witnesses are all one
operation, an expanded copy at a point.

### What is left

`fused_sum` reaches `F + 2`; see [The mmasim corpus](#the-mmasim-corpus) for
where each design stops.

Index-set instantiation covers a constant index (`range(k, k + 1)`) and an
inline `a + s * k`, and replays an *envelope* -- `max(xs) >= x_i`, which holds
at every index -- along with the per-element constraints, at the read and
again once the walk ends, so a `max` stated after the read still reaches it.
A rounding to a float format keeps or carries its operand's binade, as one to
a fixed-point format does.

Unexercised, and so unproven:

- **Coverage.** A partial write claims nothing about elements no write
  reaches, which is sound only because reading one is undefined. A language
  that defined `empty`'s contents would need the tiling proof after all.
- **The corpus is the only real net.** `_vacuous` fires 127 times compiling
  the 16 designs and twice across the whole 5028-test unit suite, and
  `compile.py` is in no test target and not in the `Makefile` -- it is a
  tracker, as it says. The path that takes the corpus from 3/16 to 14/16 is
  therefore covered almost entirely by a script nobody runs automatically,
  which is how the unsoundness above survived three audits.

## Part IV: implementation plan

Implementation plan.  The design is settled (Part I, and Part III's
[Measurements, and the design they settle](#measurements-and-the-design-they-settle));
what follows is the phase breakdown, one phase per commit.

### Working policy

- **Pause after each phase for review.** Do not begin the next phase until the
  current one has been looked at.
- **Do not commit.** The author of the change leaves the working tree dirty;
  commits are made by the repository owner.
- **Run only the tests relevant to the phase.** The full unit suite runs once,
  at the end, after the last phase.
- **Comments stay succinct**, and notes about *process* -- what was tried, what
  a phase decided, why an ordering was chosen -- belong in this document, not
  in source comments.
- **Every phase that touches a rule runs the corpus**: `compile.py` at 60/62,
  all agreeing.

### Context

`fpy2/analysis/digit_bound/infer.py` departs from Part I in four places
([Where the implementation departs](#where-the-implementation-departs)): index
correspondence read off loop depth, aggregates bounded on the summary (unsound
since v0.2.4), static loops walked once, and copies renaming every per-element
variable.  Three strict xfails in `tests/unit/analysis/test_digit_bound.py` pin
the first two.  The phases fix them in the order the dependencies force.

**Why index spaces come before the soundness fix.**  A witness is a copy, and a
copy made with today's renaming renames every per-element variable -- including
an instruction's `e_max = max(es)`, which sits inside the loop over
instructions in every corpus design.  The witness would lose `es_w <= e_max`,
and `fused_sum` would fall from `F + 2` to unbounded across the corpus.  So
signatures land first, then witnesses on them.

### Phases

#### Phase 1 - Regression net

**Done** (with the doc merge).  `TestAnAggregateHasItsOwnWorstElement` (the
paired sums, in the store and in the formats) and
`TestAnElementReadInsideAnOuterLoop`, strict xfails that fail on their
assertions: 3 against 120, a format excluding the value, 288 against 12.

    .venv/bin/pytest -q -n 8 tests/unit/analysis/test_digit_bound.py

#### Phase 2 - Index spaces, recorded but not read

**Done.**  Where it diverged from the plan:

- **The table lives on the store** (`DigitBoundStore.spaces`), not handed to
  instances beside it: variables live there, so every walk sharing them --
  a callee, a `DigitBoundParams` continuation -- shares their signatures with
  no threading.  `IndexSpaces` (`spaces.py`) wraps `fpy2.utils.Unionfind`.
- **Keys are per instance**, `('binder', id(walk), id(node), clause)` and
  `('elements', id(walk), id(list))`, settling the open item: it costs
  nothing.
- **A definition mints under its own binders.**  Terms are minted on first
  read (`_def`), so a parameter first read inside `for j` was recorded as
  varying with `j`.  `_def_binders` takes the binders around the definition's
  site instead (none for a parameter; a loop's own for its target and phis;
  a comprehension's clauses for its targets).  The depth proxy has the same
  flaw, `_var_level` set at the read; Phase 3 retires it.
- **Copies record the binders at their read** (`_IndexSet.binders`), since
  the store mints a replayed copy's variables itself.
- Correspondence is recorded at `_bind_elt`, covering reads (`_visit_expr`,
  `_visit_assign`, `_unpack_elt`), covering and gathering writes (with the
  innermost binder, and the index's), a comprehension's result and its
  clauses, list assignments, branch and loop phis, a list parameter's seed
  (`_corresponds`, against the caller's spaces but those around the call),
  and a call's list result.

Unchanged by construction, and checked: all 138 corpus outputs (62 C++
sources, 60 catalog kernels, 16 remixes) are byte for byte those of the
commit before.  `TestIndexSpaces` reads the table: a comprehension pairs its
list with its source; an outer loop's `max` varies with the loop alone; a
callee's comprehension pairs with the caller's list and varies with the loop
around the call.

    .venv/bin/pytest -q -n 8 tests/unit/analysis/test_digit_bound.py tests/unit/analysis/test_format_infer.py

#### Phase 3 - Copies rename by signature

**Done.**

- `_instance` and `_replay` rename exactly the variables varying over the
  copied list's space, through a membership test (`_Varying`) asked only of
  the variables a replay meets; a variable not varying with the index is the
  part's too, so `_instance` no longer declines a list with one.
- `_elt_lsb` keeps a list's grid where it varies over nothing outside the
  binders around the walk; `_grid_ge` unions what its bounds vary over
  (`_varies`, keeping the lsb exception).
- Deleted: `_elt_depth`, `_elt_depth_of`, `_elt_vars`, `_var_level`,
  `_anchor_level`, `_raise_anchor`, `_moves_at`, `_term_level`, and
  `_mark_elt`'s `lsb` parameter.
- **Not lifted yet:** the store still skips a replayed constraint naming a
  variable outside the copied space unless it is an envelope.  Shared, such a
  variable would be valid to replay -- but not while the worst-element
  aggregate rules stand, so Phase 4 lifts it with them.

Measured: `TestAnElementReadInsideAnOuterLoop` passes (marker removed), every
other test unchanged; the corpus at 60/62, all agreeing on 256 draws; all 138
corpus outputs byte for byte the commit before's; the Triton tests pass; and
CDNA3's tree before a separate join, and Ampere's, compile and agree on 512
outputs.  Times unchanged (C++ corpus 44 s).

    .venv/bin/pytest -q -n 8 tests/unit/analysis/test_digit_bound.py tests/unit/analysis/test_format_infer.py
    cd examples/mmasim && ../../.venv/bin/python compile.py -j 8 -r 256
    cd examples/mmasim && ../../.venv/bin/pytest -q -n 8 tests/test_triton.py

#### Phase 4 - Aggregate upper bounds on witnesses

- `_emit_logb`'s `Sum` case, and its `AMax`/`AMin` case where the operand is a
  list, state their bound on an expanded copy of the list's element terms,
  keyed by the aggregate expression, instead of on the summary.
- An audit, recorded here: every rule reading a list's summary, each marked an
  envelope (true of every element) or an upper bound (needs a witness).

Separate because it is the soundness fix, and needs Phase 3's renaming.  It
also lifts the store's skip of a replayed constraint naming a variable outside
the copied space, once no worst-element rule is left to make that unsound.

Tests: both `TestAnAggregateHasItsOwnWorstElement` tests pass (markers
removed); `TestFusedSum` and the corpus keep `F + 2`; the format-inference
precision tests unchanged.

    .venv/bin/pytest -q -n 8 tests/unit/analysis/test_digit_bound.py tests/unit/analysis/test_format_infer.py
    cd examples/mmasim && ../../.venv/bin/python compile.py -j 8 -r 256

#### Phase 5 - Unroll static loops that carry a real value

- `_visit_for` walks the body once per iteration where the count is static
  (`_len_of(iterable)`) and a loop phi carries a `RealType` value, under a cap;
  each iteration's definitions get terms of their own, and the carried value
  enters iteration `k` as iteration `k - 1`'s.  Past the cap, or with a
  symbolic count, the loop stays one generic iteration.

Separate because it is the only phase that changes how much the solver is
asked, so its cost is measured on its own.

Tests: a chain of seven rounded adds bounded relative to its operands' `max`;
CDNA3's chain of FMAs before a separate join compiling; the C++ corpus's
digit-bound solver time against the estimate (50 s to about 79 s).

    .venv/bin/pytest -q -n 8 tests/unit/analysis/test_digit_bound.py
    cd examples/mmasim && ../../.venv/bin/python compile.py -j 8 -r 256
    cd examples/mmasim && ../../.venv/bin/pytest -q -n 8 tests/test_triton.py

#### Phase 6 - Corpus, sweep and docs

- `examples/mmasim/tests/test_triton.py`: CDNA3's tree and chain of FMAs as
  agreeing remixes.
- Rerun the sweep; CDNA3's two `sum` points measured, leaving the exact sums
  as the only refusals.
- Part III rewritten to the implementation as it stands; `backend-triton.md`
  and `mmasim-serving.md`'s records updated.

Separate because it is the mmasim side, which rebases with the stuck PR.

    cd examples/mmasim && ../../.venv/bin/pytest -q -n 8 tests/test_triton.py tests/test_generator.py
    cd examples/mmasim && ../../.venv/bin/python serve/sweep.py

### After the last phase

    .venv/bin/pytest -q -n 8 tests/unit
    .venv/bin/ruff check fpy2 && .venv/bin/mypy fpy2
    cd examples/mmasim && ../../.venv/bin/python compile.py -j 8 -r 256
    cd examples/mmasim && ../../.venv/bin/pytest -q -n 8 tests/test_triton.py tests/test_generator.py
    cd examples/mmasim && ../../.venv/bin/pytest -q -p no:xdist serve/tests

### Open items

#### Is unification by length sound for lists that are never read together?

Covering is by length, so two unrelated lists of one length land in one class
as soon as one loop covers both, and a class only grows: a variable can be
renamed for a part of a list it never met.  Renaming more is the safe
direction.  **Provisional:** accept it; reopen if a design loses a bound to it.

#### Should a binder key be the AST node, or the node in one callee instance?

**Settled in Phase 2:** by instance, which costs nothing.

#### Does a list of symbolic length keep its relational facts?

The ablation found no output depending on one, so dropping them would simplify
Phase 3; keeping them is what the semantics allows.  **Provisional:** keep them
(no change), and reopen if they cost precision or time.

#### Does a `while` loop get a binder?

It is not per-element today.  **Provisional:** no, as today.

#### Does the `lsb` exception carry over exactly?

`_moves_at` counts only what bounds a list's own grid, not where it was minted.
**Provisional:** carried over as is in Phase 3; Phase 2's tests pin it first.

#### How are an iteration's terms kept apart when a loop is unrolled?

Terms are per `Definition`, and a body's definitions are the same objects every
iteration.  Phase 5 needs a per-iteration scope for them -- a saved and
restored `by_def` per iteration, or terms keyed by `(Definition, iteration)`.
**Open:** settle it at the start of Phase 5.

#### What is the unroll cap?

Large enough for the corpus's chains (seven iterations), small enough to bound
the solver.  **Provisional:** 64, matching the widest instruction; reopen on
Phase 5's measurement.
