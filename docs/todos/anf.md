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
so no backend is forced into an impossible position, and phase 9's tripwire is
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

### 5. Totality

Phases 2–4 create the slots, so a class-3 subexpression can be named like any
other. What remains is to state the invariant and check it.

**It is not "an ANF program has no `IfExpr`".** A ternary with slot-free arms is
left alone on purpose — no backend needs a place for it, and an `IfStmt` is
bulkier for nothing. The invariant is uniform across all three sealed positions
instead:

> No expression needing a slot sits in a position that has none, and every
> position a lowering reaches for free is flattened.

The first half is what phase 9 checks from the emitter side: every `IfExpr`,
`And` / `Or` and unrotated `while` condition is slot-free by `needs_slot`. The
second half is why a surviving ternary or chain is over atoms and nothing looser
— an unflattened one is a position no pass with a preamble can enter. A `while`
condition is the exception, and deliberately: rotation duplicates it.

Assert it over the `examples/` corpus. Expect a residue: `CompToLoop` is partial
— it refuses a dependent-clause list — so a comprehension element can still hold
something needing a slot. Either ANF refuses such a program or the invariant is
stated modulo comprehensions; measure how many corpus functions are affected
before choosing.

### 6. A scheduling operator

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

### 7. Wire into the C++ pipeline

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

### 8. Delete what ANF made dead

The `_bind_operand` call sites, and the `_emit_at` build-at-`want` machinery that
no longer meets a nested constructor. Success metric: `_bind_operand` reaches
zero call sites.

The `_fresh_temp` sites are *not* in scope and will not move — they name C++
scaffolding no FPy expression corresponds to (loop indices, output buffers,
accumulators, the saved `fenv` mode). Those retire with §4 and §7 of the roadmap,
when the lowerings that invented them move upstream.

```
pytest tests/unit/backend/cpp -q -n 8
```

### 9. The emitter tripwire

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

**Scalars only.** Naming an aggregate creates a def `same_object_defs` unions
with nothing, so its class can be narrower than the value's and a shared list
then hits `_refuse_unsharing`; the extra apparent place also pushes
`_shares_storage` toward boxing, which under `UnboxMode.STRICT` is a compile
failure rather than a slowdown. Aggregates are a follow-on with a measurement of
their own.

**ANF runs unconditionally**, not under `optimize`. Standard practice — do as
much in the middle end as possible, so the backend has less to handle — and the
alternative was worse: a gated pass leaves both emitter paths alive, so none of
phase 8's deletions could happen, and the class-3 miscompiles would stay live
under `optimize=False`.

The cost is that `optimize=False` no longer means "compiles the surface AST
verbatim": it now means the surface AST in statement form, with the
optimizing transforms still skipped. Phase 7 updates `CppCompiler`'s docstring
to say so.
