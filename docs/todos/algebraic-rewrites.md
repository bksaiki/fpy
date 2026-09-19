# Roadmap: algebraic rewrites

## Context

The motivating schedule is a fused sum that quantizes to a shared exponent:

```python
@fp.fpy(ctx=fp.REAL)
def fused_sum(xs: list[fp.Real]) -> fp.Real:
    if all([fp.isfinite(x) for x in xs]):
        e = max([fp.logb(x) for x in xs])
        with fp.MPFixedContext(e - P, rm=fp.RM.RTZ, enable_neg_zero=False):
            ts = [fp.round(x) for x in xs]
        return sum(ts)
    else:
        with fp.FP32:
            return sum(xs)
```

`comp_to_loop; rescale_fixed; simplify` leaves the then-branch as:

```python
ts = fp.empty(len(xs))
for t12 in range(len(xs)):
    x = xs[t12]
    _k = ((e - 12) + 1)                 # loop-invariant, recomputed each iteration
    _t = ((2 ** -_k) * x)
    with fp.MPFixedContext(-1, rm=fp.RM.RTZ, enable_neg_zero=False):
        _t15 = fp.round(_t)
    ts[t12] = ((2 ** _k) * _t15)        # unscale, per element
return sum(ts)
```

`2 ** _k` is loop-invariant, so it can be pushed past the `sum`.

## Target form

What the whole schedule should produce once everything below has landed —
`fuse; to_anf; comp_to_loop; rescale_fixed; hoist_invariant; hoist_scale;
simplify`:

```python
@fp.fpy(ctx=fp.REAL)
def fused_sum(xs):
    acc = True
    for x in xs:
        b = fp.isfinite(x)
        acc = acc and b
    if acc:
        t8 = fp.empty(len(xs))
        for t9 in range(len(xs)):
            x = xs[t9]
            t8[t9] = fp.logb(x)
        e = max(t8)
        _k = ((e - 12) + 1)                 # hoisted: invariant
        _r = (2 ** -_k)                     # hoisted: invariant, needs `to_anf`
        _s = (2 ** _k)                      # hoisted out of the reduction
        ts = fp.empty(len(xs))
        for t11 in range(len(xs)):
            x = xs[t11]
            _t = (_r * x)
            with fp.MPFixedContext(-1, rm=fp.RM.RTZ, enable_neg_zero=False):
                _t14 = fp.round(_t)
            ts[t11] = _t14                  # integers: position zero
        return (_s * sum(ts))
    else:
        with fp.FP32:
            return sum(xs)
```

Three hoists, not one.  `_k` and `_r` are ordinary loop-invariant code motion —
`_r` only after `to_anf` gives the subexpression a name of its own, since
`hoist_invariant` moves statements rather than subexpressions.  `_s` is the
reduction hoist, and it is the one that needs the side conditions below.

The payoff is not fewer multiplies.  The loop body is reduced to one multiply
and one round, and the `ts` are *integers* — `MPFixedContext(-1)` is position
zero — so `sum(ts)` is an integer accumulation scaled once at the end, which is
what the backend wants.

**Checked, not assumed.**  Written out by hand and interpreted against the
current `fuse; comp_to_loop; rescale_fixed; simplify` output over ten inputs
spanning subnormals, overflow, cancellation, `inf` and `NaN`: 8 of 10 are
bit-identical, and the other two are exact zeros differing only in the stored
exponent of the zero (`c=0`, same sign, `==` holds).  Nothing in the target form
changes a value the current schedule produces.

Nothing in FPy discovers this.  `fpy2.rewrite` is a syntactic `l -> r` rewriter
that checks nothing; `ConstFold` is partial evaluation, so a symbolic `2 ** _k`
is inert.  This roadmap is the passes that would close the gap, in the order
they pay off.

## Why none of this can be a `@pattern` rule

`Rewrite` is context-blind.  With `scale_out : sum([c * e for x in xs]) ->
c * sum([e for x in xs])` applied to a reduction under `fp.FP32`, it fires
silently and changes results:

| `xs` | `c` | before | after |
|---|---|---|---|
| `[1e20, 1.0, -1e20]` | `3` | `0` | `6.01226e12` |
| `[1.0, 1.0, 1.0]` | `1e30` | `3.000000196258126e30` | `2.999999894026671e30` |

A pattern binds expressions, not scopes, so the gate cannot be written as one.
Every rewrite below has a side condition and therefore has to be a transform
that can **decline**.

## The REAL gate is necessary but not sufficient

`RewriteUtils.is_exact` (`fpy2/transform/utils.py:323`) is the gate — with the
caveat that `scope_ctx` returns `None` for a symbolic scope, which must decline
rather than fall through to "assume exact".  It is a **per-site** question:
`fused_sum` holds one reduction under `REAL` and another under `FP32`.

But scale-out is *still* wrong under `REAL`, on special values:

| case | before | after |
|---|---|---|
| `xs = []`, `c < 0` | `+0.0` | `-0.0` |
| `xs = [+0.0, -0.0]`, `c < 0` | `+0.0` | `-0.0` |
| `xs = [1.0, -1.0, 1.0]`, `c = inf` | `NaN` | `+inf` |

A negative `c` moves the sign onto a zero the original never signed, and an
infinite `c` turns a cancellation (`inf + -inf = NaN`) into a surviving `inf`.
So the factor needs a **value-class** side condition as well: finite, and either
non-negative or over a reduction whose result excludes zero.

## Passes

Four, plus one analysis.  Only #1 and #2 are needed for the motivating example.

### 0. Loop-invariance query *(analysis, prerequisite for #1)*

"Does this expression evaluate to the same value on every iteration."
`define_use` and `reaching_defs` carry the dependency information; nothing
packages this question yet (the only mention of loop invariance in the tree is
a comment in `format_infer/analysis.py:2702`).

### 1. `hoist_invariant` — loop-invariant code motion

Moves `_k = ((e - 12) + 1)` above the loop.  Worth having on its own: the
recomputation is a visible wart in every `rescale_fixed` output.

**Not REAL-gated.**  It relocates an expression rather than re-associating one,
so it is sound under any context — but only to a destination with the *same*
active scope.  Hoisting a body statement out of a loop nested in a `with` would
silently re-evaluate it under a different context, so the check is `ctx_use`
scope equality, not `is REAL`.

**Special values / effects.**  A zero-trip loop makes the hoisted expression run
when it otherwise would not.  Per the value-class soundness assumption, an
operation whose context refuses its result *raises*, so this can turn a clean
run into an abort.  Condition: prove the loop runs at least once, or prove the
hoisted expression cannot raise.

**Why it goes first:** it makes #2's side condition syntactic.

### 2. `hoist_scale` — pull an invariant factor out of a `sum`

The example's rewrite.  Matches a loop whose only contribution to `ts` is
`ts[i] = c * e`, consumed by `sum(ts)`.

**Conditions.**  Each one has a counterexample above or in the tables:

1. `is_exact` at the reduction site (declining on symbolic scope).
2. Every free variable of `c` is defined *outside* the loop — a `define_use`
   dominance check, no invariance lattice needed, given #1 has run.
3. `c` excludes `NAN` and `INF` — `ValueClassAnalysis.is_finite`.
4. `c` is non-negative, **or** the reduction result excludes `ZERO`.
5. `ts` has no other use (`live_vars`), and the loop writes every element.

**What the analyses deliver today.**  Measured on the motivating example, not
assumed — and conditions 3 and 4 do **not** currently discharge:

| schedule | `e` | `2 ** _k` |
|---|---|---|
| `comp_to_loop; rescale_fixed; simplify` | `TOP` | `NAN\|POS_INF\|ZERO\|FINITE` |
| `fuse; comp_to_loop; rescale_fixed; simplify` | `NEG_INF\|ZERO\|FINITE` | `POS_INF\|ZERO\|FINITE` |

Two things follow.

*`fuse` must run first.*  `value_class` refines through a fold accumulator
(`_implied_fold`), not through `all(t6)` over a materialized list, so without
`reduce_fusion` the `all([fp.isfinite(x) …])` guard buys nothing and everything
is `TOP`.  With it, `x` refines to `ZERO|FINITE` in the then-branch.  Worth
confirming whether this is a `value_class` gap worth closing directly — the
guard is perfectly readable in source form.

*The residual `POS_INF` is the all-zero input.*  `fp.isfinite(0.0)` is true, so
the guard admits `xs = [0.0, …]`, giving `e = fp.logb(0) = -inf` and
`2 ** _k = +inf`.  That execution *raises* in the source program —
`MPFixedContext(nmin=-inf)` is rejected — so under the value-class soundness
assumption it contributes no class and the factor really is finite.  But no
analysis proves this: the constraint lives in the context constructor's domain,
and **`rescale_fixed` has already erased it**, rewriting the context to the
literal `MPFixedContext(-1)`.  After rescaling nothing forces `_k` finite.

So condition 3 needs one of: a `Pow`-with-literal-base-2 rule, `format_infer`
bounds, or teaching `value_class` that a context constructor constrains its
arguments.  Condition 4 needs a sign source regardless — `ValueClass`
deliberately does not track it (`ZERO` is "either signed zero", `FINITE` is
"either sign"), so the `POS_INF|ZERO|FINITE` above rules out `-inf` but not a
negative finite `c`.  `exact_exp2` is already exported and may be the right
hook for both.  **Resolve this before implementing.**

### 3. `hoist_scale` for monotone selections — `max([c * x …]) -> c * max([x …])`

Rounding is monotone and `max` selects rather than accumulates, so this one may
hold **outside** `REAL` — the only such rewrite here.  It needs its own proof
around NaN propagation, signed zero in ties, and `c <= 0` flipping the
selection.  Not assumed; not started.

### 4. REAL-gated identity table — `2**a * 2**b -> 2**(a+b)`, `2**k * (2**-k * x) -> x`

Nice to have, **not** needed by the motivating example: after #2 no
`2**k * 2**-k` adjacency survives.  REAL-only, same as #2 — under a rounding
context each operation rounds separately and the exponent range differs.  Note
`2**k * (2**-k * x) -> x` is wrong even under `REAL` at `k = inf`
(`inf * (0 * inf) = NaN`, not `x`), so this is an identity *table with side
conditions*, not a pattern file.

Natural home is a gated rule set inside `ConstFold` rather than a new pass,
since it is pure expression-level folding.

## Sequencing

**One PR per new transform under `fpy2/transform`**, end to end: the pass, its
`fpy2.strategies` wrapper, the `docs/source/strategies.rst` entry, and tests.
Analysis work ships inside the PR whose rewrite needs it, never on its own.

That makes **two new transforms**, and they are the whole critical path:

### PR 1 — `HoistInvariant` *(critical path)*

`fpy2/transform/hoist_invariant.py`, wrapper `fpy2/strategies/invariant_hoist.py`
exporting `hoist_invariant`.  #0 lands here: the loop-invariance query has no
other consumer.

*Works afterwards:* `_k = ((e - 12) + 1)` is computed once above the loop rather
than once per iteration, in every `rescale_fixed` output — not just this one.  A
standalone scheduling primitive, useful whether or not PR 2 ever lands.

### PR 2 — `HoistScale` *(critical path)*

`fpy2/transform/hoist_scale.py`, wrapper `fpy2/strategies/scale_hoist.py`
exporting `hoist_scale`.  Carries **everything needed to make it fire on
`fused_sum`**: the `value_class` refinement through `all` / `any` over a
materialized list, and the finiteness and sign source for `2 ** k`.

*Works afterwards:* `fused_sum` schedules to an integer accumulation —
`ts[t12] = _t15` in the loop, `return ((2 ** _k) * sum(ts))` outside — with the
`FP32` branch of the same function correctly declined.

**This is the big one, and it carries the unknown.**  Settle the finiteness and
sign route first (`exact_exp2` hook vs. `format_infer` bounds vs.
context-constructor argument domains) — a spike, not a PR.  Shipping
`hoist_scale` against a `value_class` that cannot discharge its side conditions
would mean a pass that declines on its own motivating example.

### Follow-ups — not new transforms

Neither #3 nor #4 adds a file under `fpy2/transform`, so neither is a PR of the
kind above:

- **#3**, the `max` / `min` monotone-selection variant, *widens* `HoistScale` —
  same producer/consumer matching, different reduction operator and a different
  proof.  Either fold it into PR 2 or follow up against the same file.
- **#4**, the REAL-gated identity table, is a change to the existing
  `ConstFold`.

### Order

PR 1, then PR 2.  The follow-ups are unblocked from the start.
