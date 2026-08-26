# Plan: transforming a program into ANF

The implementation plan for
[backend-independence.md](backend-independence.md) §1 — a pass binding every
non-atomic subexpression to a name in a statement slot, so the C++ emitter never
has to invent a place for one.

Read §1 first for why this is backend-independent and where a temporary may go.
This document is only the sequence.

## The predicate the plan turns on

ANF must be total, which means lowering the class-3 positions rather than
refusing them — a `while` condition, a ternary arm, an `and` / `or` tail.
Lowering them *unconditionally* would pessimize every pure one into a statement
chain, so the lowerings are gated:

> `needs_slot(e)` — does `e` contain a node whose lowering may **allocate, bind a
> name, or assert**?

True for `Round` / `Cast`, the reductions (`Sum`, `AMin` / `AMax`,
`AnyOf` / `AllOf`, `Min` / `Max`), the allocating forms (`ListComp`, `ListExpr`,
`Empty`, `ListSlice`, `Range1..3`, `Zip`, `Enumerate`) and `Call`. False for
`Var`, literals, arithmetic, `Compare`, `Not`, `And` / `Or`, the FP predicates,
the nullary constants and `Fst` / `Snd` — recursively, so an `IfExpr` is
slot-free only when both arms are.

This is `emitter._PURE_COND_OPS` lifted to the language and complemented. It is
deliberately an *over*-approximation of "some backend may need a statement here",
so no backend is forced into an impossible position, and phase 10's tripwire is
what verifies it was conservative enough. `while x > 0:` and `y = a if c else b`
keep their natural shape; only the impure cases lower.

## Phases

Each is a commit. Pause after each for review; the executor does not commit.
Only the named tests run per phase — the full suites come at the end.

### 1. ANF core — classes 1 and 2, scalars only

`fpy2/transform/anf.py`, `ANF.apply(func)`, a `DefaultTransformVisitor` over
statements that names non-atomic *proper* subexpressions into the preceding slot.
`Gensym(reserved=def_use.names())` for names, `SyntaxCheck.check(...,
ignore_unknown=True)` after, exported from `fpy2/transform/__init__.py`.

Atomic: `Var`, a literal, an access path over a `Var` (`iter_elim.is_access_path`
— the emitter folds an `Fst` / `Snd` chain into one `std::get`, and naming each
level would break the fold). Class-3 positions are left entirely alone until
phase 2. Never hoists across a `ContextStmt`: crossing a `with` changes the
active context and so the rounding.

Tests: new `tests/unit/transform/test_anf.py`, mirroring
`test_comp_to_loop.py`'s three parts — structural shape, semantic equivalence
through the interpreter, and the positions it declines.

```
pytest tests/unit/transform/test_anf.py -q
```

### 2. `needs_slot`, and `while`-condition rotation

The predicate, then the rotation where it holds of the condition:

```python
c = <cond>
while c:
    <body>
    c = <cond>
```

`c` is an ordinary loop-carried variable the phi machinery already handles, and
the condition's temporaries land in a slot that runs exactly as often as FPy
evaluates it. Put the predicate in `anf.py` unless another pass wants it, in
which case `transform/utils.py`.

Tests: equivalence, plus `while max([y, 0.0]) > 0.0` — one of the three
miscompile witnesses in [backend-cpp.md](backend-cpp.md) — now rotating.

### 3. `IfExpr` → `IfStmt` plus a phi

For every ternary but `x1 if c else x2` over atoms, which is already in normal
form. **Not** gated on `needs_slot` — and the reason is not the backend.
`_visit_if_expr` spells a ternary inline and needs no place for one, so lowering
deletes no emitter code; `needs_slot` would be the right gate if C++ were the
only consumer.

What an unflattened arm costs is *reach*. `transform/utils.py`'s `SiteRewriter`
suppresses its preamble inside an `IfExpr` branch, and `round_elim.py` says so
outright — hoisting needs a statement slot, so it is disabled inside a ternary
arm. `RoundElim` and `RoundInsert` therefore cannot rewrite there at all, which
is `comp_to_loop.py`'s own stated reason for existing, applied to ternaries.
Flattening makes those positions schedulable.

The merge phi is `is_intro`, so the emitter hoists the declaration before the
`if`, which already works. Where the ternary is a statement's whole right-hand
side the branches assign that name directly, and an arm that is itself a lowered
ternary branches on the same name — so a chain becomes one `if`/`elif` ladder
rather than a chain of copies. Nothing runs after this pass to remove a copy, so
that matters.

Tests: equivalence, plus the `0.0 if x > 1e30 else fp.cast(x)` witness.

### 4. `And` / `Or` → a short-circuit `If1Stmt` chain

The same atom rule, for the same schedulability reason — `_emit_bool_chain`
joins with `&&` / `||` and never needs a place, so this buys the backend only
witness 3. FPy's `and` / `or` compile to Python's `BoolOp`, so they
short-circuit and the lowering must preserve that order.

The guards are **flat**, not nested: once an `or`'s accumulator is true every
later `if not t` fails, so no further operand runs, and dually for `and`. A
nested form would indent one level per operand for nothing.

Tests: equivalence, plus the `x > 1e30 or fp.cast(x) > 0.0` witness, and a
program whose tail would assert if evaluated — which pins the order rather than
merely the value.

### 5. Type-directed atomicity

Replace the syntactic exclusion list with a type test. `_AGGREGATE` was a
hand-written 16-entry blacklist standing in for a question `TypeInfer` answers:
it excluded `g(g(x))` even when `g` returns a `Real`, and `xss[i][0]` even though
its result is one. `TypeAnalysis.by_expr` gives the real answer, queried on the
original node before it is rebuilt.

The rule: name an expression iff its type is `RealType` or `BoolType`. A
whitelist, so `ListType`, `TupleType`, `ContextType` and an unresolved `VarType`
all stay inline — naming one wrongly is the case with a consequence.

That flattens down to the aggregate boundary and no further, which is the point:
a chain is named at its outermost *scalar*, so the spine stays inline and **no
name ever holds a list**.

```python
t  = x * 2.0
t1 = g(t)
t2 = g(t1)          # nested calls fully unfold
t3 = i + 1
t4 = xss[t3][0]     # named at the scalar; `xss[t3]` is a list, so it stays
```

The `_shares_storage` → boxing → `STRICT`-refusal exposure only appears when a
name holds a list region, so this phase never touches it. The aggregate scope in
*Assumptions* narrows accordingly: from "calls, projections, containers, ranges,
slices, zip/enumerate" to just "expressions whose type is aggregate".

Also fixes a copy this exposed: `_visit_expr` tested the *original* node's kind,
so a lowered ternary or chain — which comes back as the name it accumulated into
— was bound a second time (`t4 = t`). It tests the rebuilt result now.

One thing to watch at phase 7, and it is the plan's existing warning rather than
a new one: a named projection gets its storage from *its own* format bound, which
`backend-cpp.md` records can be narrower than the container's declared element
type. Value-preserving, but it would emit an implicit narrowing where the house
rule wants an explicit `static_cast`. The differential harness is what finds it.

### 6. Totality

**Done.** The invariant is reported rather than enforced: `anf.refusals(func)`
lists every sealed position holding an expression `needs_slot` admits, one entry
per position with the reason. An empty list is the strong form — no expression
needing a place sits where there is none. Whether a given refusal *matters* is
the backend's question and the answer differs by position, so the pass reports
and does not refuse.

Measured over the 230-function corpus (`tests/unit/transform/test_anf_profile.py`,
modelled on `test_unbox_profile.py`):

| | |
|---|---|
| functions ANF applies to | 230 / 230, no errors |
| refusals in a ternary arm, a bool tail, or a `while` condition | **0** |
| a comprehension's iterable | 18 |
| a comprehension's element | 7 |

The zero is the load-bearing number. Those three are exactly the positions the
cpp emitter cannot slot — the three miscompiles in
[backend-cpp.md](backend-cpp.md) — and phases 2–4 empty them across the whole
corpus.

The residue is entirely comprehensions, and neither half is over-conservatism:

- **An iterable is a list by definition**, so hoisting it would create the
  aggregate name phase 5 is built to avoid. Not fixable under scalars-only at
  any price.
- **An element runs once per iteration**, and only `CompToLoop` can give it a
  slot.

Neither is a defect. The cpp emitter gives an element the loop body it generates
and an iterable the `for` header, so `needs_slot` is simply conservative there —
checked: a dependent nested comprehension (`[[1.0 for _ in range(i)] for i in
range(n)]`) compiles correctly today, with the inner bound inline in the header.

**Generated programs too**, since the corpus is shallow — its residue is nothing
nested deeper than two levels, so a lowering meeting another lowering is only
reachable by generating it. `test_anf_property.py` draws from `ANF_PROFILE` (in
`tests/unit/generators/profiles.py`) and checks four properties per draw: the
pass applies, no dangerous refusal survives, it is idempotent, and the
interpreter agrees before and after *including on which exception it raises* —
which is what witnesses 2 and 3 actually are.

The profile is deliberately skewed. On `DEFAULT_GRAMMAR` a 120-draw sample hit
**zero** bool chains and 2 ternaries, so the first clean fuzz run proved almost
nothing; cutting the compound productions to three per type takes that to ~42
bool chains and ~10 ternaries. One gap remains and is not closable from here:
the generator's loop template is `c = 0; while c < N`, whose condition is pure by
construction, so **no draw can reach rotation**. That path rests on the corpus
profile and the unit tests.

### 7. A scheduling operator

`fpy2/strategies/anf.py`, exporting `to_anf(func)` — the whole-program operator,
following `close` and `simplify`: a `Function` in, a `Function` out, no `where`.
ANF has no *sites* to aim at; rewriting one nest and not another leaves the
program in a state no consumer wants, so this operator takes the whole function
by construction.

The one design question is whether it should also accept a `Module`. `simplify`
and `close` take a `Function`, and `Module.map` already lifts a per-function
rewrite, so `Function` is the consistent choice unless a caller turns up that
needs the module form.

Tests: `tests/unit/strategies/`, matching whatever the sibling operators assert
— that it is the transform under a `Function` and that the docstring example
holds.

### 8. Wire into the C++ pipeline

In `CppCompiler.specialize()`, after `RoundElim` — ANF destroys the shapes
`ReduceFusion`, `ZipElim` and `EnumerateElim` match on, so it runs last. It
belongs in `specialize()` and not in `_emit`, or `signature()` and
`compile_module()` analyze different ASTs, which `_analyze_all`'s docstring
records as having been a real ABI bug once.

The risky phase: new defs mean new storage classes, so expected-output strings
churn and some storage choices may genuinely shift.

```
pytest tests/unit/backend/cpp -q -n 8
```

### 9. Delete what ANF made dead

The `_bind_operand` call sites, and the `_emit_at` build-at-`want` machinery that
no longer meets a nested constructor.

**Not zero call sites** — that was the original metric and it is wrong under
scalars-only. The sites that survive are exactly the aggregate ones:
`_adapt_arg`, `_emit_tuple_accessor`, `_visit_list_slice`, `_list_range`,
`_convert_storage`, `_rebuild_list`. A nested call (`g(g(t))`) and a projection
chain (`xss[t][0]`) are still nested after this pass, by design.

Most of them go *quiet* rather than away: `_bind_operand` returns its argument
untouched when it is already an identifier, so `sum(xs)` and
`_emit_ieee_min_max`'s twice-named operands stop minting a temporary while the
call remains. So the metric is **how many calls actually mint a temporary across
the corpus**, measured before and after. Deleting a site needs its operand
proven an atom, which for the aggregate forms waits on the follow-on.

The `_fresh_temp` sites are *not* in scope and will not move — they name C++
scaffolding no FPy expression corresponds to (loop indices, output buffers,
accumulators, the saved `fenv` mode). Those retire with §4 and §7 of the roadmap,
when the lowerings that invented them move upstream.

```
pytest tests/unit/backend/cpp -q -n 8
```

### 10. The emitter tripwire

Gate `while` conditions, `IfExpr` arms and boolean tails on `_is_pure_cond`,
raising `CppEmitError`. With phases 2–4 landed this rejects nothing under the
default pipeline; it protects `optimize=False`, and it catches a future emitter
change that emits a statement from a new expression path. That closes the three
defects in [backend-cpp.md](backend-cpp.md)'s open issues by construction rather
than by pipeline history — and it is the check on `needs_slot` being
conservative enough.

Tests: the three witnesses as compile-and-run cases — exit zero, correct value.

### Then the full suites

```
pytest tests/unit -q -n 8
python -m tests.infra
python -m tests.infra.backend.cpp
```

## Assumptions and scope

**Scalars only**, and after phase 5 that means exactly "expressions whose type is
aggregate". Naming one creates a def `same_object_defs` unions with nothing, so
its class can be narrower than the value's and a shared list then hits
`_refuse_unsharing`; the extra apparent place also pushes `_shares_storage`
toward boxing, which under `UnboxMode.STRICT` is a compile failure rather than a
slowdown. A follow-on with a measurement of its own.

**ANF runs unconditionally**, not under `optimize`. Standard practice — do as
much in the middle end as possible, so the backend has less to handle — and the
alternative was worse: a gated pass leaves both emitter paths alive, so none of
phase 9's deletions could happen, and the class-3 miscompiles would stay live
under `optimize=False`.

The cost is that `optimize=False` no longer means "compiles the surface AST
verbatim": it now means the surface AST in statement form, with the
optimizing transforms still skipped. Phase 8 updates `CppCompiler`'s docstring
to say so.
