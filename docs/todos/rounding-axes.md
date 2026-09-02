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

The paper's analysis half is **already built**, as `fpy2/analysis/format_infer`:

| Paper | FPy |
|---|---|
| Abstract format 𝒜(p, exp, b) (§4.2) | `AbstractFormat` (`format_infer/format.py`) |
| ⊕ / ⊗ (§6.1) | `AbstractFormat.__add__` / `__mul__`, plus `exact_binop`, `exact_unop`, `exact_select`, `exact_logb`, `exact_exp2` |
| 𝒜-Contains-Prec and 𝒜-Contains-Sub (Fig. 7) | `AbstractFormat._is_contained_in` — condition 3's fallback (`pos_bound <= 2^(exp + other.prec)`) *is* 𝒜-Contains-Sub |
| Format inference (§6.1) | `FormatInfer` / `FormatAnalysis` (`format_infer/analysis.py`), including loop fixpoints, call edges, and branch refinement |
| Canonicalization (§6.2, left to right) | `RoundElim` / `elim_round` — see [round-elim.md](round-elim.md) |
| Finitization (§6.2, right to left) | `RoundInsert` / `insert_round` |
| Correct double rounding (§5.2, Fig. 8) | `double_round_ok` (`format_infer/double_round.py`) |
| Splitting one rounding into two (§5) | `SplitRound` / `split_round` |
| *Beyond the paper:* per-operation double rounding (Roux 2014) | `double_round_op_ok` |

Three of the paper's four rewrites are implemented; `merge_round` is the one
left.

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

### What closed, and where it lives

Five gaps, all done; the source of truth for each is the code and the Lean
development, not this page.

| gap | where |
|---|---|
| `insert_round` — give an exact operation a format | `transform/round_insert.py`, `strategies/round_insert.py` |
| Rounding-mode plumbing — `Context.rounding_mode()`, `AbstractFormat.next_bound()` | `number/context/`, `analysis/format_infer/` |
| `split_round` — expand one rounding into two | `transform/split_round.py`, `strategies/round_split.py` |
| Figure 8's eight rules over nine mode pairs | `format_infer/double_round.py`; [Mpfx/DoubleRounding.lean](https://github.com/bksaiki/mpfx-lean/blob/main/Mpfx/DoubleRounding.lean) |
| Bounded intermediates, and Roux 2014's per-operation rules | `SplitRound._within`, `double_round_op_ok` |

Together the last two mean all five basic operations split **FP32 → FP64** and
**FP16 → FP32** under plain RNE, which Figure 8 alone admits at no width. The
three conditions that are load-bearing rather than cautious — nearest-only,
every operand representable in the *target*, and no mixed exponent family for
`/` and `sqrt` — are recorded with the measured consequence of dropping each in
the `fpy2/transform/split_round.py` module docstring.

Two follow-ons left behind:

- **`insert_round` still has the arithmetic-only candidate set.** `split_round`
  widened to any real-valued operation, since the rules quantify over an
  arbitrary real and what produced it does not matter; the same argument applies
  here. `min` / `max` stay out — they *select* an argument and hand it back with
  its own format, so there is no rounding to give or split.
- **A tight bounded intermediate that saturates is declined though it is
  exact** for an RTZ target. Admitting it needs the overflow boundary per mode
  pair, which Figure 8 does not cover: its theorems are `RoundsFinite`.

### 5. `merge_round`

Collapse a nested rounding into a single one when the Figure 8 premise holds.
This complements `elim_round`, which only removes roundings that are
*identities*; `merge_round` removes roundings that change the value but not the
final answer.

**It now has hand-written inputs, which it did not when this was written.** The
note here used to say its only inputs came from `split_round`, because the
obvious hand-written narrowing — an FP64 intermediate re-rounded to FP32 — is
RNE-RNE, which Figure 8 refuses at every width. The operation-specific rules
above overturn exactly that case: an FP32 sum, product, quotient or root
computed in FP64 and re-rounded is sound, so the shape a person actually writes
is now the shape this operator collapses. That moves it from
close-the-axis bookkeeping to the more useful of the two directions.

**Its predicates are built** — all three of them, and it needs all three, since
merging is splitting read right-to-left. `double_round_ok`,
`double_round_op_ok`, and the exact-intermediate check in
`SplitRound._exact_result` are in `format_infer/double_round.py` and
`fpy2/transform/split_round.py`; `split_round` validates a candidate
intermediate against them, `merge_round` an intermediate already in the program.
The operand premise carries over too: the operation rules need every operand
representable in the *final* format, which for a merge is the outer one. What
remains here is site detection, not verification.

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

Items 1-6 are done — see *What closed, and where it lives*. Widening mode was
**dropped** rather than deferred: a widening cast is a verified identity, so it
says nothing in FPy (see the widening note under [The two
axes](#the-two-axes)). What is left:

1. **`merge_round`** — reuses *all three* of the predicates above, since merging
   is splitting read right-to-left: the shape `split_round` emits is exactly what
   it should collapse, so the Roux rules let the two undo each other. Survey how
   a double rounding is spelled first (nested `with`, explicit `Round`, `Cast`,
   an assignment across a scope boundary) — the predicate is the easy half. The
   only operator left on the page.
2. **The §6.3 recipe** — canonicalize (`elim_round` to fixpoint, then
   `merge_round`), then finitize (`insert_round`, `split_round` against an
   environment's format list). A documented composition and a worked example,
   not a new operator; the MX dot product of §2 and §8 is the example.

## Open questions

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
  counterexamples. RNE-RNE is the one to pin down: it is the pairing a
  hand-written FPy program falls into by default and the one a well-meaning
  future patch is most likely to "fix". The guard is
  `TestRefused.test_rne_over_rne` in `tests/unit/analysis/test_double_round.py`,
  asserting `double_round_ok` refuses it at any width.

  Note the example this bullet used to give — the FP64-to-FP32 decline — is no
  longer one: that pairing is *admitted* now, by the operation-specific rules,
  which is sound and is the point of them. What stays refused is RNE-RNE as a
  rule for arbitrary reals. The two must not be conflated when writing the
  negative tests.
