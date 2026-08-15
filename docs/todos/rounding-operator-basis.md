# Are the rounding operators a basis?

A scheduling operator should be one idea, and the set of them should be
*orthogonal*: no operator doing part of another's job, and nothing reachable
only by accident. The rounding operators are not there yet, and this records
the one place where they are not, along with what it would take to fix it.

## The non-orthogonal seam

`float_to_fixed` does two things when its source is bounded:

1. expresses a float rounding as a fixed-point one at a position from `logb`
2. carries the source's *bound and overflow rule* into the target

(2) is `unfold_overflow`'s axis. That is why both passes contain
overflow-reasoning: `_Policy`, `_overflow_policy`, three of four `_ctx_call`
branches, the `inf_value=nan` trick for `E4M3`, and the trailing `Copysign` that
restores the sign a substituted NaN drops — about 120 lines, a quarter of
`float_to_fixed`, all of it there only because a *context* has to produce the
overflow value.

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

- **Should `rescale_fixed`'s `fold_specials` merge with `unfold_overflow`?**
  Both take a value class out of the context and into a branch — overflow in one
  case, NaN and the infinities in the other. They are separate knobs today, and
  a third is proposed in [unfold-neg-zero.md](unfold-neg-zero.md) for the sign
  of zero. If all three land, one operator parameterized by *which* edge rule to
  externalize may read better than three.

- **`where` counts candidate blocks, and these operators change how many there
  are.** Unfolding overflow turns one rounding into several statements, so a
  later pass selecting "the *n*th candidate" sees a different structure than it
  would on the original program. Composing `where` across operators is therefore
  order-dependent in a way nothing currently documents.
