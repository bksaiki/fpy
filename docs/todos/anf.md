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
so no backend is forced into an impossible position, and phase 8's tripwire is
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

Where `needs_slot` holds of either arm. The merge phi is `is_intro`, so the
emitter hoists the declaration before the `if`, which already works.

Tests: equivalence, plus the `0.0 if x > 1e30 else fp.cast(x)` witness.

### 4. `And` / `Or` → a short-circuit `IfStmt` chain

Where `needs_slot` holds of a non-first operand. FPy's `and` / `or` compile to
Python's `BoolOp`, so they short-circuit and the lowering must preserve that
order.

Tests: equivalence, plus the `x > 1e30 or fp.cast(x) > 0.0` witness, and a
program whose tail would assert if evaluated — which pins the order rather than
merely the value.

### 5. Totality

Phases 2–4 create the slots, and ANF runs after them, so a class-3
subexpression now gets named like any other. Add the check: no non-atomic
subexpression survives outside an atomic position. Assert it over the `examples/`
corpus.

### 6. Wire into the C++ pipeline

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

### 7. Delete what ANF made dead

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

### 8. The emitter tripwire

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

**`optimize=False` is unresolved** — §1's second blocking question. It bites only
phase 6's wiring. If ANF ends up gated on the flag, phase 8 stops being
belt-and-braces and becomes the only fix on that path.
