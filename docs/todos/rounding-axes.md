# Roadmap: the rounding axes

## Goal

Make the number of roundings in a program a schedulable axis, in both
directions — insert a rounding that changes nothing, and expand one rounding
into two that compose to the same answer. The first axis is now closed in both
directions; the second is not built.

Rules and terminology follow the double-rounding paper, *When Double Rounding
is Correct*: format containment is §5.1, the correct-double-rounding table is
§5.2 Figure 8, format inference is §6.1, and the canonicalize/finitize pair is
§6.2.

## Where we are

The paper's analysis half is **already built**, as
`fpy2/analysis/format_infer`:

| Paper | FPy |
|---|---|
| Abstract format 𝒜(p, exp, b) (§4.2) | `AbstractFormat` (`format_infer/format.py`) |
| ⊕ / ⊗ (§6.1) | `AbstractFormat.__add__` / `__mul__`, plus `exact_binop`, `exact_unop`, `exact_select`, `exact_logb`, `exact_exp2` |
| 𝒜-Contains-Prec and 𝒜-Contains-Sub (Fig. 7) | `AbstractFormat._is_contained_in` — condition 3's fallback (`pos_bound <= 2^(exp + other.prec)`) *is* 𝒜-Contains-Sub |
| Format inference (§6.1) | `FormatInfer` / `FormatAnalysis` (`format_infer/analysis.py`), including loop fixpoints, call edges, and branch refinement |
| Canonicalization (§6.2, left to right) | `RoundElim` / `elim_round` — see [round-elim.md](round-elim.md) |
| Finitization (§6.2, right to left) | `RoundInsert` / `insert_round` — **done**, see gap 1 below |

So two of the paper's four rewrites are implemented, and the analysis the other
two need is in place and in use.

Two places where FPy is *ahead* of the paper, and which every new operator has
to respect:

- **Special values.** The paper sets them aside (§4.1). `AbstractFormat` carries
  `has_pos_inf` / `has_neg_inf` / `has_nan` / `has_neg_zero`, and
  `specials_contained_in` is a separate condition of containment for exactly
  this reason: the finite values fitting does not make a rounding an identity,
  and a `-0.0` reaching a format without one is a wrong answer rather than an
  imprecise one. The comment on `specials_contained_in` records that this was
  once a real bug.
- **Value sets.** `SetFormat` bounds an expression by a finite set of values,
  strictly stronger than any 𝒜, and it is what makes many roundings provably
  redundant in practice.

## The two axes

Borrowing the vocabulary of [rounding-operator-basis.md](rounding-operator-basis.md),
where `unfold_overflow` / `fold_overflow` are the bound axis in both directions:

**The rounding axis** — one rounding against none.

```
elim_round     rounded op  ->  with fp.REAL:      exists
insert_round   with fp.REAL:  ->  rounded op      exists
```

This axis is closed.  What remains of §6.2 is `insert_round`'s *widening mode*:
the same predicate aimed at operands rather than results, which is what §8's
operand promotion needs.

**The double-rounding axis** — one rounding against two.

```
split_round    rnd_F1,rm1  ->  rnd_F1,rm1 . rnd_F2,rm2    missing
merge_round    rnd_F1,rm1 . rnd_F2,rm2  ->  rnd_F1,rm1    missing
```

Widening — §8's operand promotion, which makes a heterogeneous operation
uniform-precision so a library like SoftFloat can implement it — is **not** a
third axis. Promoting an operand to a larger format that contains it is
inserting a rounding that does nothing; it is `insert_round` aimed at operands
instead of results. Keeping it as a mode rather than an operator is what stops
this from repeating `float_to_fixed`'s seam, where one operator quietly owns
part of another's axis.

### Which axis a redundant second rounding belongs to

"The second of two roundings is redundant" splits into two cases, and only one
of them is §5's. The dividing line is whether the *final* format is narrower
than the intermediate.

**Widening, then a no-op.** The second rounding is an identity, so this is pure
containment (§5.1) and `elim_round` already does it:

```python
@fp.fpy(ctx=fp.REAL)
def f(x: fp.Real) -> fp.Real:
    with fp.FP16:
        y = fp.round(x)
    with fp.FP32:
        return fp.round(y)      # -> return y
```

`FP16 = 𝒜(11, -24, 65504)` is contained in `FP32 = 𝒜(24, -149, 3.4e38)`, so
`round_is_identity` fires and `RoundElim`'s `Round`-node case collapses the
round to its argument. No double-rounding reasoning is involved, and no new
operator is needed.

**Widening, then narrowing.** The second rounding is *not* an identity — it
changes the value — and the question becomes whether the intermediate perturbed
the final answer. This is the only case Figure 8 exists for, and `elim_round`
correctly leaves it alone:

```python
    with fp.FP64:
        y = fp.round(x)
    with fp.FP32:
        return fp.round(y)      # untouched: FP64 is not contained in FP32
```

So the double-rounding axis earns its keep only below the containment
boundary. Above it, the work is already done.

## The gaps that remain

### 1. `insert_round`'s widening mode

**Done: the result half.** `insert_round(func, ctx, where=None)` gives an exact
operation a format, verified by `round_is_identity` — the same helper `RoundElim`
decides with, so nothing in the number tower changed.

Its candidates are **operations**, not blocks, and the rewrite mirrors
`RoundElim._hoist`. A context applies to every operation in its block, so an
operation is given a format of its own by being lifted into a block alone: each
operand that is not already a `Var` is bound under the original scope first, and
because contexts nest the new block goes *inside* the one it found rather than
splitting it.

```python
with fp.REAL:                     with fp.REAL:
    t = (x * x) + (y * y)   ->        with fp.FP64:
                                          _t = (x * x)
                                      t = _t + (y * y)
```

Per-operation granularity is what makes this usable, and it is sound for a
reason worth stating plainly: **the inserted rounding is verified to be an
identity, so it changes no value.** An operation reading the result sees what it
would have seen, so operations can be given formats one at a time and in any
order — including one whose result a later exact operation consumes. There is no
dependence hazard, and no all-or-nothing per block.

It declines a stochastic target, an unbounded operand, and an operation too wide
for the target. An operation that already has a format is not a candidate at
all, so the listing no longer carries sites that always refuse.

**Remaining: operand sites.** Same predicate aimed at operands: wrap each in a
cast to the target where the operand's inferred format is contained in it,
making the operation uniform-precision. This is the rewrite

```
mul_R(E5M2, E5M2)  ->  mul_FP32(rnd_FP32(a), rnd_FP32(b))
```

that §8 needs before SoftFloat or hardware can run the multiply. Whether it is
a `mode=` on `insert_round` or a sibling operator is open: the verification is
identical, the site kind is not. This is the next commit on the page.

**The search variant.** `finitize(func, where, available=[...])` picks the
smallest format from a list of environment-supported formats that contains the
inferred format. That is §6.3's search, and it is what a user actually reaches
for; `insert_round` with an explicit context is its primitive. Inferring the
context instead is not worth it: reading the enclosing scope only reproduces
`elim_round`'s own choice, and deriving a format from the inferred bound (via
`AbstractFormat.format()`) yields formats no hardware has. It belongs with the
recipes, not the operators — gap 2 of
[native-lowering-roadmap.md](native-lowering-roadmap.md) and item 7 of
[scheduling-language.md](scheduling-language.md) are the same shape of
question.

### 2. Rounding-mode plumbing

Blocks gaps 3 and 4, blocks nothing else, and is mechanical.

- **No rounding mode on the base `Context`.** `Context` exposes `format()`,
  `round_params()`, `is_stochastic()`, `with_params()`, `round()`, and
  `round_at()`. Concrete contexts carry `.rm: RoundingMode`, but nothing is
  lifted. Figure 8 dispatches on the *pair* of modes, so this needs either an
  abstract `rm` accessor alongside `format()`, or a probe that returns `None`
  for a context with no single mode. `RoundingMode` in `fpy2/number/round.py`
  already includes `RTO`, so no new modes are needed.
- **No `next(b)` on `AbstractFormat`.** Every Figure 8 premise bumps the bound
  to the next representable value. `RealFloat.next_up` exists
  (`fpy2/number/number/reals.py`), so this is a wrapper — but note that
  RND-RTO-RNE takes `next` at precision `p1+1, exp1-1` rather than at the
  target's own precision, so the wrapper has to take the precision as a
  parameter.

`with_prec_offset` and `with_exp_offset` already exist on `AbstractFormat`, and
they are exactly the `p+k, exp-k` knobs the table needs.

### 3. `split_round`

Expand one correctly-rounded operation into two roundings. This is the operator
§5 exists to justify, and §5.3's modular-library recipe as a single schedule
step: compute in high precision under RTO, re-round to the target under
whatever mode the target wants.

Figure 8 admits six of the sixteen mode pairings. Following the paper's naming,
`rm2` is the intermediate and `rm1` the final; `F1 = 𝒜(p1, exp1, b1)` is the
target and `F2 = 𝒜(p2, exp2, b2)` the intermediate:

| rule | final `rm1` | intermediate `rm2` | premise |
|---|---|---|---|
| RND-RTZ-RTZ | RTZ | RTZ | `𝒜(p1, exp1, next(b1))` ⊆ `F2` |
| RND-RAZ-RAZ | RAZ | RAZ | `𝒜(p1, exp1, b1)` ⊆ `F2` |
| RND-RTO-RTO | RTO | RTO | `𝒜(p1, exp1, next(b1))` ⊆ `F2` **and** `p2 >= 2` |
| RND-RTO-RTZ | RTZ | RTO | `𝒜(p1+1, exp1-1, next(b1))` ⊆ `F2` |
| RND-RTO-RAZ | RAZ | RTO | `𝒜(p1+1, exp1-1, next(b1))` ⊆ `F2` |
| RND-RTO-RNE | RNE | RTO | `𝒜(p1+2, exp1-2, next_{p1+1,exp1-1}(b1))` ⊆ `F2` |

Two details that look like typos and are not: RAZ-RAZ needs no `next(b)` bump
while RTZ-RTZ does, and the `p2 >= 2` side condition on RTO-RTO is there to
avoid a parity mismatch when both formats have one bit of precision. Every rule
is conditional on the target rounding not overflowing, and the paper chooses
`b2` from `{b1, next(b1)}` precisely so that no-overflow transfers from the
single rounding to the composition.

Modes off the table: **RTP** and **RTN** reduce to RAZ and RTZ *by the sign of
the operand* (§3.2), which FPy's `value_class` analysis could discharge and
which should otherwise decline; **RNA** appears nowhere in Figure 8 and always
declines.

The interesting call is `via=None`, which *derives* the tightest sound RTO
intermediate rather than asking the user for one:

```python
k = 2 if rm1 is RoundingMode.RNE else 1
target.with_prec_offset(k).with_exp_offset(-k)   # plus the next(b1) bump
```

With an explicit `via`, the operator instead checks the user's chosen
intermediate against the table. Fixed-point intermediates should be admissible
— §8's MPFX accumulator is `𝒜(inf, -32, 2^96)`, and the premises are
containment checks on 𝒜 that do not care which family the format comes from —
but that overlaps `float_to_fixed` and `rescale_fixed`, so it needs thought
rather than an assumption.

### 4. `merge_round`

Collapse a nested rounding into a single one when the Figure 8 premise holds.
This complements `elim_round`, which only removes roundings that are
*identities*; `merge_round` removes roundings that change the value but not the
final answer.

**Its inputs come from `split_round`, not from hand-written programs.** Every
`fp.FP*` context carries `rm=RoundingMode.RNE`, so the obvious hand-written
narrowing — an FP64 intermediate re-rounded to FP32 — is RNE-RNE, the last row
of the paper's Table 2 and unsound no matter how wide the intermediate. Figure
8 has no RNE-RNE rule, so `merge_round` declines it. The programs it *can*
merge have an RTO intermediate, or two agreeing modes drawn from RTZ / RAZ /
RTO, and under FPy's default contexts nobody writes those by hand. That makes
this operator worth building mainly to close the axis and to check
`split_round`'s output, which is why it sits last in the order below despite
sharing gap 3's predicate.

**It shares gap 3's predicate.** Write `double_round_ok(f1, rm1, f2, rm2)`
once: `split_round` uses it to validate a candidate intermediate,
`merge_round` to validate one already in the program. This is the one
structural decision on the page that is expensive to get wrong.

The work here is site detection rather than verification, and that is genuinely
unsurveyed: `Round` over `Round`, `Cast` over arithmetic, `RoundAt` over a
rounded operand, and nested `with` blocks are all candidate spellings of a
nested rounding in FPy, and which of them appear in real programs decides how
much of this operator is worth building.

### Expression positions neither operator can reach

Both operators need a statement slot for the block they hoist into, so an
expression that is evaluated conditionally or repeatedly is closed to them.
Three positions, in the order they were closed:

- **A comprehension element** — *solved.* `comp_to_loop`
  (`fpy2/transform/comp_to_loop.py`) lowers the comprehension into an `fp.empty`
  allocation plus a `for` loop, after which the element is an ordinary statement.
  Measured on `[x * x for x in xs]` with FP32 arguments: `elim_round` does
  nothing and `insert_round` has 0 sites; lowered first, `elim_round` hoists the
  product and `insert_round` has 1
  (`tests/unit/strategies/test_comp_to_loop.py::TestUnblocksTheRoundingAxis`).
  The lowering is precision-neutral — element format and length both survive it,
  since `Empty` starts at the bottom of the type and each `IndexedAssign` joins
  the stored value back in (`test_comp_to_loop.py::TestPrecision`).
- **An `IfExpr` branch** — open, and the same shape of fix: bind the ternary to a
  temp assigned in both arms of an `if` statement, which preserves semantics
  exactly since one arm evaluates either way. No such transform exists
  (`simplify_if` is unrelated). `RoundElim._visit_if_expr` and
  `RoundInsert._visit_if_expr` are the two refusals it would open, and
  `CompToLoop`'s a third.
- **A dependent clause list** in a comprehension — `[b for a in xs for b in a]`,
  whose length is a sum rather than a product, so `fp.empty` has nowhere to get
  it. Left alone rather than refused. Totality here needs either a syntactic ban
  or an append-shaped list builder in FPy, which would let `CompToLoop` drop the
  size expression for every clause shape.
- **A `while` condition** — open and hardest, since the condition is re-evaluated
  every iteration; it needs loop rotation rather than a hoist. Suppressed in all
  three passes; measured, hoisting out of it turns a terminating loop into an
  out-of-bounds slice.

## Order of work

1. ~~**`insert_round`**~~ — **done.** `RoundInsert` in `fpy2/transform/`,
   `insert_round` in `fpy2/strategies/`, registered in `sites.py`'s `_SITES`,
   tested in `tests/unit/transform/test_round_insert.py` and
   `tests/unit/strategies/test_insert_round.py`.
2. **Widening mode** — same predicate, operand sites. With step 1 this
   reproduces §8's SoftFloat mapping of the MX dot product, which is the
   natural end-to-end test for both. **Next.**
3. **Plumbing** (gap 2) — touches the number tower rather than the transform
   layer, so worth isolating from the operators that consume it.
4. **`double_round_ok` and `split_round`** — the table as a predicate, plus the
   `via=None` derivation.
5. **`merge_round`** — reuses step 4's predicate; survey the site spellings
   first. Deliberately after `split_round` rather than beside it: step 4 is what
   produces the RTO intermediates this operator can actually merge.
6. **The §6.3 recipe** — canonicalize (`elim_round` to fixpoint, then
   `merge_round`), then finitize (widening, `insert_round`, `split_round`
   against an environment's format list). A documented composition and a worked
   example, not a new operator; the MX dot product of §2 and §8 is the example.

Steps 1 and 2 are worth having even if the rest never lands: they close §6.2 in
both directions, which is the whole of the specification-to-implementation step.

## Open questions

- **Resolved: alternating `insert_round` and `elim_round` diverges, and the
  step-6 recipe needs explicit fuel.** Neither guard bounds the composition, and
  it does not cycle either — each round trip wraps the operation in one more
  block, because *both* operators hoist into a nested block rather than
  replacing the scope they found. Pinned by
  `test_alternating_the_two_does_not_converge`.

  **Fixed, in `DeadCodeEliminate`.** It now drops a `with` block that installs
  the context already in force, and one that no operation under it reads at all
  — the tower is all of the latter but its innermost block, which then matches
  the function's own context. Measured on the four-round trip: 8 blocks to 0,
  same values. So `simplify` does now converge the composition, and a recipe
  that alternates freely wants it in between. The operators themselves are
  unchanged: each still hoists into a fresh block, which is what
  `test_alternating_the_two_does_not_converge` pins.

  Where `elim_round` *declines* to hoist — an unbounded scope, refused by the
  strictly-tighter guard — `insert_round` has no site at all, so the pair is a
  no-op rather than an inverse. Pinned by
  `test_an_unbounded_scope_leaves_nothing_to_insert`.
- **Stochastic contexts.** `insert_round` declines `is_stochastic()` targets
  outright. Still open whether `round_is_identity` is already right for them on
  an exactly-representable value, which would let the decline be dropped.
- **`elim_round` takes no `where`.** It applies whole-program and does not
  forward cursors, which leaves it the odd one out once the rest of the axis is
  aimed — and it cannot join a per-rounding recipe of the kind `_lower_at` in
  `tests/unit/backend/cpp/test_lowered_roundtrip.py` builds. Retrofitting is
  not cheap: `RoundElim` uses its own `_Ctx` accumulator rather than
  `SiteRewriter` / `BlockRewriter`, rewrites at expression level with a
  greedy-outermost policy, and hoists — so it cannot claim
  `exprs_preserved=True`, and forwarding, not site listing, is the work. See
  item 3 of [scheduling-language.md](scheduling-language.md).
- **Negative tests for the ten unsound pairings.** Table 2 of the paper gives
  counterexamples. RNE-RNE is the one to pin down, for the reason in gap 4: it
  is both the pairing a hand-written FPy program falls into by default and the
  one a well-meaning future patch is most likely to "fix". A test that asserts
  the FP64-to-FP32 decline is the guard.
