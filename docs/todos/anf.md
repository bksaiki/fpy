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

**Not** the atom rule, unlike a ternary — the asymmetry is measured. Lowering a
ternary buys reach nothing else provides (phase 7 shows 0 → 2 `REAL` blocks).
Lowering a *pure* chain buys nothing anyone uses and costs something real:
`not isnan(a) and not isnan(b)` is what `ValueClassInfer._implied` reads to drop
a runtime guard, and it reads the `And`. Once the conjuncts are separate
statements joined by a phi the conjunction is gone and the guard comes back —
three `test_class_guards` cases, measured both ways.

So a chain lowers only where an operand *after the first* needs a place, which is
the case the lowering exists for: an operand whose emission is not pure must not
run when the chain short-circuits past it. FPy's `and` / `or` compile to Python's
`BoolOp`, so they short-circuit and the lowering must preserve that.

The guards are **flat**, not nested: once an `or`'s accumulator is true every
later `if not t` fails, so no further operand runs, and dually for `and`. A
nested form would indent one level per operand for nothing.

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

**Done.** `fpy2/strategies/anf.py` exports `to_anf(func)` — a `Function` in, a
`Function` out, no `where`, following `close` and `simplify`. It takes a
`Function`, not a `Module`: `simplify` and `close` do the same, and `Module.map`
already lifts a per-function rewrite.

No `where`, and the docstring says why: every other strategy aims at one site and
leaves the rest alone, which is what makes a schedule a sequence of decisions.
Normal form is not a decision of that kind — a program flattened in one nest and
not another is in no state a consumer wants.

It is deliberately **not** in `sites.py`'s `_REFUSALS` table. That table answers
"why was each candidate not a site", which is a site-based notion; ANF has no
sites, and its refusals are residue rather than declined candidates. They are
reached through `ANF.refusals` directly, and the shapes differ too — `Expr`
rather than `Cursor`, and no `within`.

Tests (`tests/unit/strategies/test_to_anf.py`) cover the wrapper — the docstring
example verbatim, identity with the transform, runtime and parent preserved,
`TypeError` on a `FuncDef`, `TypeError` on a positional `where` — and then the
thing that justifies the operator existing at all:

```python
# before: elim_round cannot enter either arm
return (1.0 + 2.0) if cond else (3.0 + 4.0)

# after to_anf, elim_round hoists both
if cond:
    with fp.REAL: _t4 = (_t + _t3)
    t = _t4
else:
    with fp.REAL: _t7 = (_t5 + _t6)
    t = _t7
```

Zero `with fp.REAL:` blocks before, two after. That is the schedulability claim
from phases 3–4, measured rather than asserted.

### 8. Wire into the C++ pipeline

**Done**, and it needed a companion change that turned out to be the larger half.

The wiring is one line at the end of `CppCompiler.specialize()`, unconditional.
Last for two reasons, and the second is the stronger: naming an expression
**materializes** it, so a pass that would have *deleted* one can only reach
inside the name afterwards. `RoundElim` collapses `fp.round(0.0)` to a literal;
run ANF first and it collapses only the initializer, leaving `t0 = 0` declared as
`uint8_t`. Measured — `ANF` before `Specialize` is a strict superset of failures
(16 to 15). The rule: **ANF goes after everything that removes or folds, and
nothing that removes or folds runs after it.**

`optimize=False` no longer means "the surface AST verbatim"; `CppCompiler`'s
docstring says so.

#### The companion: `DefineUseAnalysis.defining_expr`

Wiring produced 15 failures, and 8 shared one cause — a syntactic
pattern-matcher now sees a `Var`:

| matcher | lost |
|---|---|
| `ValueClassInfer._implied` | `not isnan(v)` → no refinement |
| `ArraySizeInfer._const_int` | `fp.empty(len(xs))` → no `std::array` |
| `ArraySizeInfer._len_size` | `assert len(xs) == 4` → parameter not pinned |
| `ArraySizeInfer._affine` | `xs[i:i+32]` → slice length lost |
| emitter `_emit_scale_by_pow2` | `2 ** n * x` → no `ldexp` |

**None of these was an ANF regression.** Compiled against the *unwired*
pipeline, `n = len(xs); fp.empty(n)` already gave `std::vector` and
`t = fp.isnan(v); if not t:` already emitted the full guard. ANF makes an
existing fragility universal, because every program is now in the "via temp"
form.

So the fix is one helper — follow a `Var` through its reaching def to the
expression that computes it, stopping at a phi, a parameter, a loop target or an
`xs[i] = e`. Sound because a *definition*, not a name, identifies a value: the
`Var` nodes in the returned expression sit at the assignment, so
`t = isnan(v); v = 3.0; ... t` still speaks about the first `v`. Each of the five
matchers gained one call.

One wart: the emitter case leaves the named power computed and unused, because
the peephole reads through the name but the binding is still a statement.
Harmless (`-Werror` is only `=return-type`) and removed by moving the peephole
upstream — §6 of the roadmap.

### 9. Delete what ANF made dead

**Measured: nothing is.** Not the outcome the phase assumed, and the measurement
is the deliverable.

Across the 201 corpus functions that compile, `_bind_operand` still mints a
temporary at five sites:

| site | calls | mints |
|---|---|---|
| `_emit_empty` | 25 | 0 |
| `_emit_enumerate` | 3 | 0 |
| `_visit_list_slice` | 11 | 1 |
| `_emit_ieee_min_max` | 28 | 3 |
| `_emit_sum` | 13 | 3 |
| `_emit_zip` | 12 | 4 |
| `_list_range` | 21 | 5 |

Every one is an **aggregate** operand — a list, a slice, a `zip` — or a *cast*
result: `_emit_ieee_min_max` binds `static_cast<double>(a)`, which is not an
identifier however atomic `a` is. Neither is something this pass addresses.

The two zero-mint sites are **unexercised, not dead**: `fp.empty` and
`enumerate` accept a compound operand, no corpus program gives them one, and
deleting a site on that evidence would be wrong.

`_emit_at`'s build-at-`want` machinery is likewise alive, and not marginally —
62 `ListExpr`, 43 `TupleExpr`, 23 `ListComp`, 13 `Empty`. Its premise was that
ANF stops nested constructors reaching it; under scalars-only it does not,
because a constructor is an aggregate.

So the deletions this phase was for are **blocked on the aggregate follow-on**,
and on nothing else. What lands instead is
`tests/unit/backend/cpp/test_bind_profile.py`, pinning the counts and the corpus
size in the style of `test_unbox_profile.py`: no correctness test can see a mint,
so when aggregate naming arrives the drop should be a decision rather than a
surprise. It also fails if a *new* site starts minting, which is the tripwire for
something scalar reaching the emitter nested.

### 10. The emitter tripwire

**Done**, but not on `_is_pure_cond` as planned. That predicate is a whitelist
built for `else if` cosmetics — it excludes arithmetic, so it would refuse
`while (x * x) > 0.0`, which compiles correctly today. A tripwire that rejects
working programs is worse than none.

Measured exactly instead: `_emit_inline(emit, what, at)` records the writer's
line count, runs the emission, and raises if it grew. `_is_pure_cond`'s docstring
rejects that approach because *rewinding* is impossible — but a tripwire does not
rewind, it aborts, so the objection does not apply. Exact means no false
refusals: over the 201 compiling corpus functions, 44 of these positions are
emitted and **none** writes a statement.

Wired into the three positions and nowhere else: a `while` condition, both
ternary arms (scalar path and the aggregate `_emit_at` path), and every
short-circuited operand after the first. An `if` / `if1` condition is *not*
gated — it runs once, just before the branch, so its statements belong in the
enclosing block. That distinction moved into `_emit_guarded_block`, which now
takes the condition already emitted, since where its statements may go is the
caller's question.

It raises `CppInternalError`: ANF lowers all three before codegen, so arriving
here is a violated invariant rather than a program the backend cannot express.

`tests/unit/backend/cpp/test_statement_form.py` has both halves. The three
witnesses now **compile and run** — `f(3.0) == 0` for the loop that used to hang,
and no abort for the two assertions that used to fire on an untaken path. The
runs are the point; no string comparison catches a hang. And `TestTheNet`
monkeypatches `ANF.apply` to the identity, which is exactly the state a future
change to the pass could leave a program in, and checks all three are refused
rather than miscompiled.

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
