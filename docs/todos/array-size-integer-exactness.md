# Array-size analysis: integer-valued arithmetic as an exactness witness

## Context

`array_size`'s slice- and range-size logic pins a size from a *symbolic*
bound when the offset cancels — e.g. `x[i : i + 32]` -> 32, or
`range(i, i + 32)` -> 32. The cancellation `(i + 32) - i == 32` is only
sound when the `+` does **not** round, so `_affine` only descends through
`+` / `-` nodes for which `_is_exact(e)` holds:

```python
def _is_exact(self, e: Expr) -> bool:
    scope = self._ctx_use.use_to_scope.get(e)
    return scope is not None and scope.ctx == REAL
```

i.e. the node must sit inside a `with fp.REAL:` scope.

That test is too weak. The motivating example:

```python
@fp.fpy
def foo(xs):
    acc = 0
    for i in range(0, len(xs), 32):
        slc = xs[i:i+32]          # <- no size derived
        with fp.REAL:
            slc_acc = sum(slc)
        with fp.FP32:
            acc += slc_acc
    return acc
```

`slc` gets **no** size. The slice bound `i + 32` is evaluated under the
function's (symbolic) default context, not `REAL`, so `_is_exact` returns
`False` and `_affine` refuses to peel the `+ 32` offset.

But the arithmetic *is* exact, for a reason `_is_exact` can't see: `i` is
a `range` loop index, hence integer-valued, and FPy computes integer
arithmetic exactly. `FormatInfer` already reaches this conclusion with no
`REAL` annotation:

```
i        : MPFixedFormat(...)   # range elements are integers
32       : SetFormat({32})
(i + 32) : RealFormat()         # exact
```

The only current workaround is to wrap the slice in an explicit
`with fp.REAL:` — a non-obvious requirement for ordinary index math that
nobody writes in practice.

## Goals

1. Derive a size for slice / range bounds whose offset cancels over
   **integer** index arithmetic, without requiring an explicit `REAL`
   scope (`x[i : i + k]`, `range(i, i + k)`, `range(0, len(xs), s)`,
   etc.).
2. Stay sound: only treat arithmetic as exact when it provably is.
3. Don't regress: bounds that are genuinely unknown-to-be-exact (e.g.
   arithmetic on a `Real` parameter) must still yield `None`.
4. No new analysis-dependency cycle.

## Non-goals

- Reasoning about size *expressions* / non-cancelling offsets
  (`len(xs) + 1`, `2 * i`). Out of scope, same as the symbolic-size
  proposal (`array-size-symbolic.md`).
- Proving integer arithmetic can't overflow a narrow format. We match
  `FormatInfer`'s existing assumption that integer index arithmetic is
  exact; we do not independently bound magnitudes.
- Tracking integer-ness for any purpose beyond the slice/range exactness
  witness.

## Root cause

`_is_exact` answers *"is this node inside a `with fp.REAL` scope?"* when
the question affine cancellation actually needs is *"does this arithmetic
round?"*. There are two independent sufficient conditions for "no
rounding":

1. the node is under the exact (`REAL`) context — what we check today;
2. the operands are integer-valued — integers are exact, and FPy
   computes integer arithmetic exactly (this is why `FormatInfer` gives
   `i + 32` a `RealFormat`).

Index math hits (2), essentially never (1).

## Design

Add an **integer-valued witness** to the exactness test, *in addition to*
the existing `REAL` check (strictly more permissive, never less):

```python
# in _affine, the descent guard becomes
case Add() if self._is_exact(e) or self._is_int_op(e):
    ...
case Sub() if self._is_exact(e) or self._is_int_op(e):
    ...
```

where `_is_int_op(Add/Sub)` holds iff both operands are integer-valued,
and `_is_int_valued(e)` is a small structural predicate:

| Expression form | integer-valued? | how we know |
|---|---|---|
| `Integer` literal / `_const_int(e) is not None` | yes | a known integer constant |
| `Len`, `Dim`, `Size` | yes | integer-producing builtins |
| `Var` bound to a `range` loop target | yes | `range` yields integers |
| `enumerate` index projection | yes | the `int` slot of the tuple |
| `Add` / `Sub` / `Mul` / `Neg` / `Mod` of integer-valued operands | yes | integer arithmetic is closed |
| anything else | no | — |

These rules mirror the integer-producing cases `FormatInfer` already
encodes (`Len`/`Dim`/`Size`/`Range`/`Enumerate` -> `INTEGER` regardless
of context).

### The only stateful piece: `range` loop targets

`_is_int_valued(Var i)` needs to know `i` is a `range` index. Record
`range` loop targets as they're bound:

- In `_visit_for`, when `stmt.iterable` is a `Range1` / `Range2` /
  `Range3`, add the loop target's `Definition`(s) to a set
  `self._int_defs`.
- `_is_int_valued(Var)` resolves the use to its def (`def_use`) and
  returns `True` if it's in `_int_defs`.
- For a `Var` bound to an assignment `j = <expr>`, optionally recurse on
  the RHS with memoization + an in-progress guard so loop phis terminate
  (treat a still-in-progress def as `False`). In practice the base of a
  slice/range bound is a bare loop index or a `len`, so the recursion is
  shallow; the assignment-chain case can be deferred if it isn't needed.

No call into `FormatInfer` — that would be a dependency cycle
(`FormatInfer` already imports `ArraySizeInfer` for `_known_iter_count`).
We replicate the small rule set instead.

## Why it's sound

- **Consistent with the compiler.** Treating integer index arithmetic as
  exact is precisely the assumption `FormatInfer` — the correctness-
  critical analysis the cpp backend trusts — already makes. This change
  introduces no *new* unsoundness relative to the rest of the compiler.
- **The existing negative tests are the guard.** The current
  `test_list_slice_symbolic_offset_not_under_real_is_unknown` and
  `test_range2_symbolic_offset_not_under_real_is_unknown` use a `Real`
  *parameter* `i` (`def f(x, i: fp.Real)`), which is **not**
  integer-valued — so those slices stay `None`, unchanged. The motivating
  case differs precisely because `i` is a `range` index → integer →
  exact. That distinction is the correct one: a `range` index is provably
  an integer; an arbitrary `Real` argument is not.

Post-fix behavior:

- `xs[i : i + 32]` / `range(i, i + 32)` with `i` a `range` index, no
  `REAL` → **size known** (fixes the example);
- `x[i : i + 16]` with `i` a `Real` parameter, no `REAL` → still `None`;
- everything under `REAL` → unchanged.

## Implementation plan

1. Add `self._int_defs: set[Definition]`, populated in `_visit_for` for
   `Range*` iterables (and, if cheap, the `enumerate` index target).
2. Add `_is_int_valued(e)` (the table above) and `_is_int_op(e)` (both
   operands integer-valued).
3. Widen the `_affine` `Add` / `Sub` descent guard to
   `self._is_exact(e) or self._is_int_op(e)`.
4. Tests:
   - integer-index slice / range *without* `REAL` → known
     (`xs[i:i+32]`, `range(i, i+32)`, `range(i, i+32, 2)`);
   - `len`-based bounds (already partially covered);
   - `Real`-parameter base without `REAL` → still `None` (regression
     guard, keep the existing ones);
   - end-to-end: the `foo`/`MX_E4M3` reproducer derives size 32 for
     `slc` with the explicit `with fp.REAL:` removed.

## Risks / open questions

1. **`Var` assignment-chain recursion + phis.** A `Var` bound to
   `j = i + 1` needs RHS recursion; loop-carried defs need a cycle guard.
   Start with the direct cases (loop target, `len`/`size`/`dim`, literal,
   one-level arithmetic) and add chained resolution only if real programs
   need it.
2. **Duplication with `FormatInfer`.** The integer-ness rules now live in
   two places. Acceptable short-term; see below.
3. **`Mul` exactness.** Integer `*` is mathematically exact but more prone
   to magnitude blow-up than `+`/`-`. Since `_affine` only peels additive
   *constant* offsets, `Mul` only matters for deciding a *base* is
   integer-valued (not for the offset), so it's safe to include — but if
   in doubt, restrict the first cut to `+`/`-`/`Neg`.

## Out of scope / possible follow-up

- **Shared integer-valued analysis.** The cleaner long-term shape is a
  small standalone "is this expression integer-valued?" analysis that
  **both** `FormatInfer` and `ArraySize` consume, eliminating the
  duplicated rule set. Larger change (touches `FormatInfer`); do the
  self-contained version first and DRY it up only if the duplication
  starts to bite.
- Non-cancelling / size-expression bounds — see `array-size-symbolic.md`.
