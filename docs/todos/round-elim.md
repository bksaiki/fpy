# Round-elimination transform: follow-ups

## Context

`fpy2/transform/round_elim.py` rewrites expressions whose implicit
rounding to the active context is provably an identity:

- **Arithmetic ops** (`Add` / `Sub` / `Mul` / `Abs` / `Neg`) whose
  unrounded `F` is contained in `C.F` get hoisted into a
  `with fp.REAL:` preamble, binding the result to a fresh `_tN`
  that replaces the original expression site.
- **Explicit `Round` / `RoundExact` / `Cast` nodes** whose argument
  bound is already contained in the target context collapse to the
  argument.

The decision uses the public helpers `exact_binop`, `exact_unop`,
and `round_is_identity` from `fpy2.analysis.format_infer`.  Greedy
at the outermost fitting subtree.  Hoisting is suppressed inside
`ListComp` element / iterable positions and inside `IfExpr`
branches — those positions either reference loop-scoped variables
or are conditionally evaluated, neither of which is sound for
unconditional statement-level preambles.

These follow-ups extend coverage or simplify the implementation.

## Open items

### Extend to `Sum`

`Sum(xs)` is a rounded op — pairwise reductions accumulate rounds
at each step.  `_sum_bound` in `format_infer/analysis.py` already
simulates the unrounded reduction (`af_acc = af_elt + … n-1 times`)
and checks containment.  RoundElim can reuse that by exposing the
unrounded simulation as a public helper (`exact_sum(elt_fmt, n)`),
then handling `Sum` in `_unrounded_format` and `_visit_expr`.  Hoist
shape is identical to the binary-op case.

### Extend to `Call` to known FPy `Function`s

A call's return value is implicitly rounded to the caller's active
context.  When the callee is a known `Function` and format inference
recorded its sub-analysis (`format_info.by_call[call_node]`), the
return-format is available.  If the return-format fits in the
caller's `C.F`, the round on the call is the identity — hoist the
call into REAL.  Care needed: the callee's body may have its own
rounding internals; the *return* round is what we're eliminating,
not the callee's interior rounds.

### Drop `_safe_to_hoist` once the collapse-first invariant is
proven

The current implementation runs the `Round` / `Cast` collapse and
the arithmetic-op hoist in the same pass, decided per node in
`_visit_expr`.  Because collapse happens at each Round/Cast node
*before* any enclosing arithmetic op is hoisted, a hoisted subtree
should never contain an eliminable round node — only non-eliminable
ones remain to be checked.  `_safe_to_hoist` is the belt-and-
suspenders that scans for them.

If a property test or formal argument confirms the invariant always
holds, drop the scan — saves an O(subtree) walk per hoist decision.

### Relax hoist suppression inside `ListComp` iterables

The first iterable of a list comp is evaluated at the enclosing
block's scope (it doesn't reference any loop target).  It's safe to
hoist.  Later iterables in multi-stage comps can reference earlier
targets, so they must stay suppressed.  Implementation: in
`_visit_list_comp`, pass the statement-level `_Ctx` to the first
iterable and `None` to the rest.

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
function effectively executes under REAL.  Replacing the function's
declared `ctx` with `fp.REAL` (and dropping the now-redundant per-op
hoists) would simplify the AST further.  Pattern detection is
straightforward — after `RoundElim`, walk the body and check for any
remaining rounded ops outside `with fp.REAL:` blocks.  Mostly
cosmetic; the runtime semantics are already equivalent.

### Phase B: per-value `SetFormat` rounding in `_bound_if_fits`

Not RoundElim-specific but related: format inference's SetFormat
fast path is fits-or-bail.  A precise per-value image
(`{C.round(v) for v in F.values}` via `Context.round`) would let
RoundElim eliminate more rounds (constants like `fp.round(1/3)`
become singleton `SetFormat({rounded_value})` rather than widening
to the scope's format).  Tried once and rolled back — the per-value
work in tight loops caused the cpp infra's `whetsone1` example to
hang.  Re-attempt with either a size cap on the SetFormat or an
explicit fall-back when iteration count crosses a threshold.

Tracked as a TODO comment in `_bound_if_fits` itself.
