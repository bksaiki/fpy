# Round-elimination transform: follow-ups

## Context

`fpy2/transform/round_elim.py` rewrites expressions whose implicit
rounding to the active context is provably an identity:

- **Arithmetic ops** (`Add` / `Sub` / `Mul` / `Abs` / `Neg`) whose
  unrounded `F` is contained in `C.F` get hoisted into a
  `with fp.REAL:` preamble.  Each operand is bound to a fresh
  `_tN` under the *current* (original) context before being passed
  into the REAL block.  Operands that are already `Var` references
  skip the bind (a copy `_t = v` has no rounds to preserve).
- **Explicit `Round` / `RoundExact` / `Cast` nodes** whose argument
  bound is already contained in the target context collapse to the
  argument — a pure node-level rewrite that fires anywhere, even
  inside positions where statement-level hoists are suppressed.

The decision uses the public helpers `exact_binop`, `exact_unop`,
and `round_is_identity` from `fpy2.analysis.format_infer`.

Hoisting is suppressed inside `ListComp` element / iterable
positions and inside `IfExpr` branches — those positions either
reference loop-scoped variables or are conditionally evaluated,
neither of which is sound for unconditional statement-level
preambles.

The unconditional-bind scheme intentionally produces redundant
copies (`_t = literal`, per-op REAL blocks for nested
eliminations).  Idempotence falls out: a second RoundElim pass
sees only `Var`-argumented ops under REAL, none of which trigger
another hoist.  The slack in the intermediate AST is the trade
for keeping the local rewrite small — downstream cleanup is
expected to do the rest.

## Open items

### Pair RoundElim with a const-prop + copy-prop + DCE cleanup invocation

The per-op unconditional-bind scheme leaves redundant `_t = literal`
and `_t = v` chains.  The full cleanup chain is `ConstPropagate`
then `CopyPropagate` then `DeadCodeEliminate`: const-prop inlines
literal-bound temps so they become dead, copy-prop collapses
Var→Var aliases, DCE removes the unused assigns.  None of these
are invoked from inside `RoundElim` today.  Either:

- Make `RoundElim.apply` optionally chain them (e.g.,
  `RoundElim.apply(func, cleanup=True)`), so callers get a tight
  AST in one step.
- Or document the recommended chain in the module docstring (done)
  and expect pipeline callers to wire it themselves.

The latter is simpler; the former is friendlier.  The cpp
backend's `optimize=True` pipeline currently does neither —
adding the chain after `RoundElim` in `_run_pipeline` is the
natural integration point.

### Extend to `Sum`

`Sum(xs)` is a rounded op — pairwise reductions accumulate rounds
at each step.  `_sum_bound` in `format_infer/analysis.py` already
simulates the unrounded reduction (`af_acc = af_elt + … n-1 times`)
and checks containment.  RoundElim can reuse that by exposing the
unrounded simulation as a public helper (`exact_sum(elt_fmt, n)`),
then handling `Sum` in `_unrounded_format` and `_visit_expr`.
Hoist shape is identical to the binary-op case.

### Extend to `Call` to known FPy `Function`s

A call's return value is implicitly rounded to the caller's active
context.  When the callee is a known `Function` and format
inference recorded its sub-analysis
(`format_info.by_call[call_node]`), the return-format is
available.  If the return-format fits in the caller's `C.F`, the
round on the call is the identity — hoist the call into REAL.
Care needed: the callee's body may have its own rounding
internals; the *return* round is what we're eliminating, not the
callee's interior rounds.

### Relax hoist suppression inside `ListComp` iterables

The first iterable of a list comp is evaluated at the enclosing
block's scope (it doesn't reference any loop target).  It's safe
to hoist.  Later iterables in multi-stage comps can reference
earlier targets, so they must stay suppressed.  Implementation:
in `_visit_list_comp`, pass the statement-level `_Ctx` to the
first iterable and `None` to the rest.

Lower priority — multi-iterable comps are rare in FPy programs.

### Re-run `FormatInfer` on the output for downstream consumers

`RoundElim.apply` runs `FormatInfer` to *decide* the rewrites but
doesn't refresh it on the output.  Consumers calling
`FormatInfer.analyze(round_elim_output)` get the tighter formats
the rewrite enables (hoisted ops report unrounded `F`).  A wrapper
that returns the rewritten `FuncDef` *and* its post-rewrite
`FormatAnalysis` would save the duplicated analyze call.  Trivial
to add when a downstream consumer asks for it.

### Lift fully-eliminable function bodies to a `fp.REAL` decoration

When *every* rounded op in a function body is eliminated, the
function effectively executes under REAL.  Replacing the
function's declared `ctx` with `fp.REAL` (and dropping the
now-redundant per-op hoists) would simplify the AST further.
Pattern detection is straightforward — after `RoundElim`, walk
the body and check for any remaining rounded ops outside
`with fp.REAL:` blocks.  Mostly cosmetic; the runtime semantics
are already equivalent.

### Phase B: per-value `SetFormat` rounding in `_bound_if_fits`

Not RoundElim-specific but related: format inference's SetFormat
fast path is fits-or-bail.  A precise per-value image
(`{C.round(v) for v in F.values}` via `Context.round`) would let
RoundElim eliminate more rounds (constants like `fp.round(1/3)`
become singleton `SetFormat({rounded_value})` rather than
widening to the scope's format).  Tried once and rolled back —
the per-value work in tight loops caused the cpp infra's
`whetsone1` example to hang.  Re-attempt with either a size cap
on the SetFormat or an explicit fall-back when iteration count
crosses a threshold.

Tracked as a TODO comment in `_bound_if_fits` itself.
