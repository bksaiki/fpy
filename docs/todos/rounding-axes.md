# Roadmap: the rounding axes

## Goal

Make the number of roundings in a program a schedulable axis, in both
directions — insert a rounding that changes nothing, and expand one rounding
into two that compose to the same answer. The first axis is closed in both
directions; the second has its forward direction and its predicate, and wants
only `merge_round`.

Rules and terminology follow the double-rounding paper, *When Double Rounding
is Correct*: format containment is §5.1, the correct-double-rounding table is
§5.2 Figure 8, format inference is §6.1, and the canonicalize/finitize pair is
§6.2. **The proofs are the source of truth** where this page and they disagree:
[Mpfx/DoubleRounding.lean](https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean).

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
| Correct double rounding (§5.2, Fig. 8) | `double_round_ok` (`format_infer/double_round.py`) — **done** |
| Splitting one rounding into two (§5) | `SplitRound` / `split_round` — **done**, see gap 3 below |

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

This axis is closed, and §6.2 with it.

**The double-rounding axis** — one rounding against two.

```
split_round    rnd_F1,rm1  ->  rnd_F1,rm1 . rnd_F2,rm2    exists
merge_round    rnd_F1,rm1 . rnd_F2,rm2  ->  rnd_F1,rm1    missing
```

The forward direction is built; only `merge_round` remains, and it reuses
`split_round`'s predicate unchanged.

**Widening is not on either axis, and not in FPy.** §8's operand promotion
makes a heterogeneous operation uniform-precision so a library like SoftFloat
can implement it. Promoting an operand to a format that contains it is
inserting a rounding that does nothing — and *that is exactly why it says
nothing here*: the cast is a verified identity, so `elim_round` removes it
again, and no analysis, no backend and no interpreter result distinguishes the
promoted program from the original. Measured: `elim_round` strips the casts
straight back out. Uniform precision is a property of the *target* environment,
so widening belongs to native lowering, not to a rewrite on FPy programs. See
gap 2 of [native-lowering-roadmap.md](native-lowering-roadmap.md).

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

## The gaps, and what closed them

### 1. `insert_round` — **done**

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

**Nothing remains.** An operand-sites mode was planned — wrapping each operand
in a cast to the target, giving `mul_R(E5M2, E5M2) -> mul_FP32(rnd_FP32(a),
rnd_FP32(b))` for §8 — and dropped: see the widening note above. The rewrite is
a no-op in FPy's semantics.

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

### 2. Rounding-mode plumbing — **done**

- **`Context.rounding_mode() -> RoundingMode | None`**, abstract, implemented
  across every concrete context; `RealContext` answers `None`. Note it has no
  `rm` attribute at all, so a base implementation reading `self.rm` would have
  raised.
- **`AbstractFormat.next_bound()`**, Figure 8's `next(b)`, taking **no**
  precision argument: the proofs call `boundAfterNext` on a format and read the
  grid off the receiver, so a caller extends first and asks second. A `prec=`
  parameter would also have been inert — `next_up` normalizes to *at most* `p`
  bits, so raising `p` without lowering `n` cannot refine the grid, and the
  RTO-to-nearest premise would have silently used the target's own grid.
- Found on the way: `with_prec_offset`, `with_exp_offset` and
  `with_bounds_scale` all dropped `has_neg_zero` when rebuilding, and the
  constructor defaults it to `False`. None of the three had a caller, so nothing
  had noticed. Fixed and pinned.

What it was blocking, for the record:

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

### 3. `split_round` — **done**

Expand one correctly-rounded operation into two roundings. This is the operator
§5 exists to justify, and §5.3's modular-library recipe as a single schedule
step: compute in high precision under RTO, re-round to the target under
whatever mode the target wants.

`SplitRound` in `fpy2/transform/split_round.py`, `split_round(func, ctx, where)`
in `fpy2/strategies/`. `ctx` is the **intermediate**; the target is read from
the program. Sites are the operations that already round — the complement of
`insert_round`'s. `derive_intermediate` computes a suitable intermediate, and
fixed-point ones work, so the concern below about `float_to_fixed` overlap did
not materialize. Explicit `Round` / `Cast` nodes are deliberately not sites:
splitting a rounding is `merge_round`'s inverse.

Two things worth knowing. The rewrite emits an explicit `round` in the
*enclosing* block, because an assignment rounds nothing in FPy — that is what
applies `rm1`, and it is asserted structurally rather than assumed. And it is
**not idempotent**: the operation lands under an RTO intermediate and RTO over
RTO is itself admissible, so a second application splits again. One pass
terminates; a schedule wanting a fixpoint has to bound it.

**The source of truth is the Lean development**, not this table:
[Mpfx/DoubleRounding.lean](https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean).
An earlier revision of this page transcribed Figure 8 four ways that the proofs
do not support; the rows below follow the theorems, each named.

Following the paper's naming, `rm2` is the intermediate and `rm1` the final;
`F1 = 𝒜(p1, exp1, b1)` is the target and `F2` the intermediate. `extend k` is
precision `+k` with exponent `-k`, and `next_F(b)` is the successor of `b` *in
`F`'s own grid* — which grid is the part worth reading twice, since the two
RTO-to-nearest premises differ only in that.

| final `rm1` | intermediate `rm2` | premise | theorem |
|---|---|---|---|
| RTZ | RTZ | `F1` ⊆ `F2` | `rndRTZ_RTZ` |
| RAZ | RAZ | `F1` ⊆ `F2` | `rndRAZ_RAZ` |
| RTP | RTP | `F1` ⊆ `F2` | `rndRTP_RTP` |
| RTN | RTN | `F1` ⊆ `F2` | `rndRTN_RTN` |
| RTO | RTO | `F1` ⊆ `F2` **and** `p2 >= 2` | `rndRTO_RTO` |
| RTZ | RTO | `𝒜(p1+1, exp1-1, next_{F1}(b1))` ⊆ `F2` | `rndRTO_RTZ` |
| RAZ | RTO | `𝒜(p1+1, exp1-1, next_{F1}(b1))` ⊆ `F2` | `rndRTO_RAZ` |
| RNE **or** RNA | RTO | `𝒜(p1+2, exp1-2, next_{extend(F1,1)}(b1))` ⊆ `F2` | `rndRTO_RN` |

Eight rules over nine mode pairs. Where this page used to differ:

- **No `next(b)` bump on the same-mode rules.** `rndRTZ_RTZ` and `rndRTO_RTO`
  take plain containment, exactly as RAZ-RAZ does. The RTZ/RAZ asymmetry this
  page called "a detail that looks like a typo and is not" is not in the proofs.
- **RTP-RTP and RTN-RTN are admitted** on plain containment. The sign branching
  is *internal* to the proofs — `rndRTP_RTP` splits into RAZ for `x > 0` and RTZ
  for `x <= 0` — so the pair is sound unconditionally and needs no `value_class`
  refinement.
- **RNA is admitted as a final mode.** `rndRTO_RN` is parametric in
  `tb : TieBreak`, covering `.toEven` and `.awayZero`; RNA declines only as an
  *intermediate*.
- **`p2 >= 2` is explicit only for RTO-RTO.** For the RTO-to-directed and
  RTO-to-nearest rules the proofs *derive* it from the bound-aware containment,
  so an implementation should not re-check it there.

Still unsound, and the ones to pin negatively: **RNE-RNE** — no rule exists, it
is the last row of the paper's Table 2, and it is the pairing every `fp.FP*`
context falls into by default — plus **RNA or RTE as an intermediate**, and
**stochastic** on either side.

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

### 4. Bounded intermediates for `split_round`

`derive_intermediate` returns an **unbounded** intermediate, and `split_round`
declines a bounded one. That is sound but leaves capability on the table.

The premises are containment checks on `A`, which says what is representable and
nothing about what happens beyond it — so a bounded intermediate has an overflow
of its own that the premise cannot see. Measured on an FP32 target, products
straddling its maxval:

| target | intermediate bound | overflow mode | wrong |
|---|---|---|---|
| RNE / RNA / RAZ | `next(b1)` | either | 0 |
| RTZ | `next(b1)` | saturating | 0 |
| RTZ | `next(b1)` | overflowing | **175** |
| RTO | `b1` (the premise *passes*) | saturating | **176** |
| RTO | `b1` | overflowing | **1** |
| any | FP64-wide | either | 0 |

Two things to read off this. A bounded intermediate is usually fine, and fails
only where its bound sits close to the target's — the failure is the *boundary*,
not the width. And no single overflow mode is right: a target that clamps (RTZ)
needs the intermediate not to overflow, one that overflows (RTO) needs it to.
That is why `derive_intermediate` sidesteps the question rather than answering
it: an unbounded intermediate cannot overflow, so the only rounding that can is
the target's, exactly as before the split.

**The fix is a proof, not a mode.** `RoundingScopes` already carries
`format_info`, so the operation's inferred bound is in hand: where that bound
lies inside the intermediate's finite range, the intermediate provably cannot
overflow and the boundary never arises. That admits every row above whose
mismatch count is zero — including the FP64-wide case, which is what a caller
reaching for a familiar format would pass — and declines the rest with a reason.
`RoundInsert._verify` reads the same field, so the machinery is there.

The residue after that proof is the genuinely open part: a bounded intermediate
whose range the operation *can* exceed. Settling it needs the overflow boundary
per mode pair, which Figure 8 does not cover — its theorems are `RoundsFinite`.

### 5. `merge_round`

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
sharing `split_round`'s predicate.

**Its predicate is built.** `double_round_ok(f1, rm1, f2, rm2)` is in
`format_infer/double_round.py` and `merge_round` reuses it unchanged --
`split_round` validates a candidate intermediate with it, `merge_round` an
intermediate already in the program. What remains here is site detection, not
verification.

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
2. ~~**Widening mode**~~ — **dropped**, not deferred: a widening cast is a
   verified identity, so it says nothing in FPy. See the widening note under
   [The two axes](#the-two-axes).
3. ~~**Plumbing** (gap 2)~~ — **done.** `Context.rounding_mode()` and
   `AbstractFormat.next_bound()`, tested in `tests/unit/number/test_context.py`
   and `tests/unit/analysis/test_format.py`.
4. ~~**`double_round_ok` and `split_round`**~~ — **done.** The predicate and
   `derive_intermediate` in `format_infer/double_round.py`, tested against the
   Lean theorems in `tests/unit/analysis/test_double_round.py`; the operator in
   `fpy2/transform/split_round.py` and `fpy2/strategies/round_split.py`.
5. **Bounded intermediates for `split_round`** (gap 4) — a no-overflow proof
   from `format_info`, which admits the useful cases the current blanket refusal
   turns away.
6. **`merge_round`** — reuses step 4's predicate unchanged; survey the site
   spellings first. The only operator left on the page.
7. **The §6.3 recipe** — canonicalize (`elim_round` to fixpoint, then
   `merge_round`), then finitize (`insert_round`, `split_round` against an
   environment's format list). A documented composition and a worked example,
   not a new operator; the MX dot product of §2 and §8 is the example.

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
- **`split_round` is not idempotent, by construction.** The operation lands
  under an RTO intermediate, and RTO over RTO is admissible, so each pass splits
  again -- one more block every time, values unchanged. That is the same shape as
  the loop rewrites rather than a defect, but it means a recipe that reaches a
  fixpoint has to bound this operator explicitly, as it must bound `unroll`.
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
  counterexamples. RNE-RNE is the one to pin down, for the reason in gap 5: it
  is both the pairing a hand-written FPy program falls into by default and the
  one a well-meaning future patch is most likely to "fix". A test that asserts
  the FP64-to-FP32 decline is the guard.
