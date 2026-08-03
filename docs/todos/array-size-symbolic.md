# Array-size analysis: symbolic sizes

**Implemented.** `array_size` used to track sizes as a flat `int | None`, which is
precise for compile-time constants and throws away everything about runtime sizes
that are *statically constrained to be equal* — `ys = xs` makes `len(ys) ==
len(xs)`, and the old lattice recorded `None`. It now tracks those equalities.

What remains is **consumer integration**: nothing reads them yet. This document
is the as-built record plus that remaining work; the original design proposal has
been dropped, since the implementation diverged from it in ways worth stating
directly rather than reconstructing from a diff.

## As built

- **A size variable is a `NamedId`** minted by a `Gensym`, with a shared
  `Unionfind[NamedId | int]` — following the type-inference precedent rather than
  a bespoke `SymbolicSize` class. So `ArraySize: TypeAlias = int | NamedId | None`.
- **A concrete `int` is the class representative.** Pinning is `union(int, var)`
  with the `int` as leader; no separate pin/resolve map.
- **The result is fully resolved and the union-find is not exposed.** `_resolve`
  (cf. type inference's `_resolve_type`) replaces every size by its
  representative before returning, so a size is known iff it is an `int`, and
  `concrete_size` / `is_size_eq` take no union-find argument.
- **Propagation is mostly free.** `ys = xs`, `[f(x) for x in xs]`,
  `enumerate(xs)`, `xs[:]`, and `IndexedAssign` already thread `.size`, so the
  variable rides along. Arguments and free variables mint one fresh variable for
  their outer dimension (`_arg_bound`).
- **`zip` is strict, not `min`.** Any concrete input pins the symbolic ones to it;
  all-symbolic inputs merge and the result keeps the representative.
- **Unconditional `assert` seeds equalities.** `assert len(xs) == len(ys)` merges
  the two variables and `assert len(xs) == N` pins one — only when not nested in
  an `if` or loop, so the equality holds on every execution.
- **Calls adopt only concrete callee sizes.** `_refine_sizes` ignores a callee's
  size *variables*: they belong to that run and must not leak across the call.

Two deliberate departures from the original plan:

- **No optimistic loop-phi merge.** Merging the pre-loop and post-body size
  variables assumes the body preserves length, which `ys = ys[1:]` violates. The
  loop fixpoint uses the ordinary join instead: a size-preserving body
  re-propagates the *same* variable so the join keeps it, and a size-changing body
  yields a different size so the join goes to `None`. Sound with no optimism.
- **Convergence needs the union-find in the check.** A merge does not change the
  `NamedId` stored in `by_def`, so bound-stability alone can end the fixpoint one
  iteration early — a `zip`-merge inside a loop body would be missed. The visitor
  keeps a monotone `_uf_changes` counter and the fixpoint requires both stored
  bounds *and* that counter to be stable.

## Remaining work: nothing consumes the equalities

`format_infer` ignores non-`int` sizes, which is a safe no-op.

- **Bounds-check elimination** — the flagship payoff. Discharge `ys[i]` when
  `is_size_eq` proves `len(ys) == len(xs)` and `i < len(xs)`. Three equality
  sources already feed it (arguments, strict `zip`, asserts). This is also what
  the C++ backend's bounds-check TODO would want, so the two are worth doing
  together — see `backend-cpp.md`.
- **`format_infer` `Sum`** — fall back to a runtime-bounded loop when the size is
  a tracked variable rather than a static `int`.

## Caveats for consumers

- **Size variables are per-`analyze` call.** Two runs mint unrelated variables, so
  comparing symbols across functions fails. That is correct — one function's
  arguments are unrelated to another's — but a consumer holding results from two
  runs must not compare them.
- **`Empty(m, n)` mints one variable per dimension**, keyed including position, so
  reusing one `Expr` as two dimensions is fine. Worth a focused test if anything
  starts depending on it.

## Out of scope

- Arbitrary integer expressions (`len(xs) + 1`, `2 * len(xs)`). A much larger
  design — linear arithmetic, Presburger. This stops at equivalence.
- Interprocedural symbolic sizes.
