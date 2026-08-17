# Are the rounding operators a basis?

A scheduling operator should be one idea, and the set of them should be
*orthogonal*: no operator doing part of another's job, and nothing reachable
only by accident. The rounding operators are not there yet, and this records
the one place where they are not, along with what it would take to fix it.

## The non-orthogonal seams

### The bound axis

`float_to_fixed` does two things when its source is bounded:

1. expresses a float rounding as a fixed-point one at a position from `logb`
2. carries the source's *bound and overflow rule* into the target

(2) is `unfold_overflow`'s axis. That is why both passes contain
overflow-reasoning: `_Policy`, `_overflow_policy`, three of four `_ctx_call`
branches, the `inf_value=nan` trick for `E4M3`, and the trailing `Copysign` that
restores the sign a substituted NaN drops — about 120 lines, a quarter of
`float_to_fixed`, all of it there only because a *context* has to produce the
overflow value.

### The specials axis

`float_to_fixed` also emits the `isnan` / `isinf` / `== 0` ladder itself, because
`logb` is undefined on all three. That is now `unfold_special`'s axis: it applies
to any statically-known context, float ones included, and states the same three
branches from the format's own answers.

The two are not redundant — `float_to_fixed` must still work when run alone — but
they overlap, and running `unfold_special` first is strictly better output:
value classes then make `float_to_fixed` skip its ladder, so the specials are
handled once at the outside and `logb` hoists to a single call. The decomposition
would be for `float_to_fixed` to *require* the operand already be finite and
non-zero, leaving the ladder to `unfold_special` alone. Not scheduled, and it
costs `float_to_fixed` its standalone use.

## Deleting it is not expressibility-neutral

The obvious move is to delete that machinery and require `unfold_overflow`
first. Measured, that **shrinks the set of reachable programs**: a bounded
fixed-point rounding at a computed position is reachable only through
`float_to_fixed`'s bounded path, and no other operator emits an
`MPBFixedContext` from a float source.

What is *not* lost is reachable semantics — the composed path computes the same
function. So the loss is a program *shape*, not a behaviour.

## The decomposition that does span

Delete `_Policy` **and** add `fold_overflow`, the inverse of `unfold_overflow`:
fold `t = round_U(x); if t > B: ...` back into a bounded context.

- `float_to_fixed` becomes purely the position axis
- `unfold_overflow` / `fold_overflow` are the bound axis, in both directions
- the lost shapes come back as `float_to_fixed → fold_overflow`

And the set *grows*: today a bounded fixed-point context is reachable only from
a float source, because only `float_to_fixed` builds one. With an inverse it is
also reachable from a hand-written unbounded fixed-point program.

The honest cost is that the classification code moves rather than vanishes —
`fold_overflow` has to decide which context flags reproduce a given pair of
overflow values, which is `_Policy`'s four-way classification driven from
program text instead of from a source context. `float_to_fixed` sheds its 120
lines; the bound axis gains most of them back. What is bought is that every
operator does one thing and the bound axis becomes invertible.

**Not scheduled.** The capability is in use and nothing is broken; this is a
design debt note, not a bug. Revisit if a second consumer of the bound axis
appears, or if `float_to_fixed`'s bounded path starts costing maintenance.

## Smaller questions in the same area

- **Resolved: the edge rules are standalone operators, not knobs.**
  `unfold_special` states NaN, the infinities, and the operand's zero as
  branches — for a float source as well as a fixed-point one, since stating a
  special needs only a known context while *shedding* its rule from the format
  needs a format that states it as a parameter; `unfold_neg_zero` states the sign of a zero *result*; and
  `rescale_fixed`'s `fold_specials` is gone — a format with a finite
  `nan_value`/`inf_value` is declined until `unfold_special` has taken the
  rule out of the context. The asymmetry that kept them separate: NaN and
  the infinities are *operand*-driven (test `x` before the rounding), while
  the sign of zero is *result*-driven (any tiny negative rounds to zero, so
  the rounding has to happen before the sign can be restored). The full
  fixed-point lowering is now a composition of one-idea operators:
  `unfold_special → unfold_neg_zero → unfold_overflow → rescale_fixed`.

- **`where` counts candidate blocks, and these operators change how many there
  are.** Unfolding overflow turns one rounding into several statements, so a
  later pass selecting "the *n*th candidate" sees a different structure than it
  would on the original program. Composing `where` across operators is therefore
  order-dependent in a way nothing currently documents.
