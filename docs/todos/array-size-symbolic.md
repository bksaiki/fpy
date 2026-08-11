# Array-size analysis: symbolic sizes

**Implemented.** `array_size` used to track sizes as a flat `int | None`, which is
precise for compile-time constants and throws away everything about runtime sizes
that are *statically constrained to be equal* — `ys = xs` makes `len(ys) ==
len(xs)`, and the old lattice recorded `None`. It now tracks those equalities.

Consumer integration has begun: the C++ backend reads *concrete* sizes — a
proven-length unboxed list compiles to `std::array<T, K>` (`backend/cpp/unbox.py`
`_region_sizes`), and `Specialize` keys specs on argument lengths so sizes cross
call edges. The *equalities* (symbolic `NamedId` sizes) still have no consumer.

## The API a consumer needs

- `ArraySize: TypeAlias = int | NamedId | None`. A size variable is a `NamedId`
  minted by a `Gensym`, sharing a `Unionfind[NamedId | int]`; a concrete `int` is
  the class representative.
- Results are fully resolved and the union-find is not exposed, so a size is
  known iff it is an `int`, and `concrete_size` / `is_size_eq` take no
  union-find argument.
- Equalities come from three places: arguments and free variables (one fresh
  variable per outer dimension), strict `zip` (any concrete input pins the
  others), and unconditional `assert len(xs) == len(ys)` — only when not nested
  in an `if` or loop, so the equality holds on every execution.
- Calls adopt only *concrete* callee sizes; a callee's variables belong to that
  run and must not leak across the call.

Two departures not to re-propose:

- **No optimistic loop-phi merge.** Merging the pre-loop and post-body variables
  assumes the body preserves length, which `ys = ys[1:]` violates. The ordinary
  join is sound with no optimism: a size-preserving body re-propagates the *same*
  variable, a size-changing one yields a different size and the join goes `None`.
- **Convergence needs the union-find in the check.** A merge does not change the
  `NamedId` stored in `by_def`, so bound-stability alone can end the fixpoint one
  iteration early. The visitor keeps a monotone `_uf_changes` counter and
  requires both to be stable.

## Remaining work: nothing consumes the equalities

`format_infer` ignores non-`int` sizes, which is a safe no-op, and the C++
backend's array selection deliberately drops them too (a symbolic size is a
per-run gensym — it can neither name a `std::array<T, N>` without templates
nor enter a specialization fingerprint deterministically).

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
