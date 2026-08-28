# Hoistable form: making statement insertion always legal

## Goal

FPy has statements and expressions, so a pass that needs a temporary for an
expression has to hoist it into a statement above. That is not always sound:
hoisting out of `while e: s` changes how often `e` runs, and hoisting out of a
ternary arm or a short-circuited `and`/`or` operand runs it on a path FPy never
takes. Every pass that emits a statement — `RoundElim`, `RoundInsert`,
`FuncInline`, `CompToLoop` — refuses those positions, and each re-derives the
same refusal.

`fpy2/transform/hoistable.py` establishes one invariant, once:

> Every expression node is evaluated exactly once, unconditionally, whenever its
> enclosing statement is reached.

Under it the slot immediately before the enclosing statement is *always* a legal
place for a hoisted temporary, so a pass mints one on demand and no pass reasons
about conditional evaluation again.

## Why not ANF

`fpy2/transform/anf.py` already establishes the invariant, but as a side effect
of a second, stronger job:

| job | what it does | who wants it |
|---|---|---|
| lowering | restructures the positions FPy evaluates conditionally or repeatedly, so each gets a statement slot | every transform pass |
| atomization | binds every nameable non-atomic subexpression to a fresh name | the cpp emitter, which needs a *place* for every operand |

Atomization is what makes ANF unusable as a general normalization for the
scheduling language: it rewrites the whole program to buy a property a pass needs
at one site. Over `examples/` — 885 functions, 10937 expression nodes — ANF names
up to 5096 subexpressions, where only **71 positions** need a lowering:

| position | count |
|---|---|
| `while` condition | 38 |
| `and` / `or` | 19 |
| comprehension | 11 |
| ternary | 3 |

Hoistable form is the lowering half alone. `ANF` stays as it is and remains the
backend's normalization; this pass does not edit it.

## The non-strict positions

The invariant is reachable because the set is small and closed. It is the same
list as `anf.py`'s `_SEALED_REASON`:

| non-strict position | lowering |
|---|---|
| `IfExpr` arms | → `IfStmt` assigning one name |
| `and` / `or` tail | → flat guarded `If1Stmt` chain |
| `while` condition | rotate: evaluate before the loop and at the end of the body |
| comprehension element / iterables | → alloc + `for` (`CompToLoop`, a separate pass) |

Everything else is a statement operand or an operand of a strict operator, and
`sub_exprs` in `transform/path.py` lists those in evaluation order.

## The ordering hazard

Lowering alone is **not** semantics-preserving. Hoisting a lowered construct out
of an operand moves it above the operands to its left, which are then evaluated
later than they were:

```python
return g(a) + (h(b) if c else 0.0)     # raises g's assertion

if c: t = h(b)                          # naive lowering
else: t = 0.0
return g(a) + t                         # raises h's assertion -- wrong
```

ANF does not have this bug only because atomization names `g(a)` too, in
left-to-right order. So the weak pass keeps *part* of the naming:

> **Prefix rule.** At any node, let `last` be the position of the last child (in
> `sub_exprs` order) that a lowering fires inside. Every earlier child that is
> not already an atom is bound to a name.

The rule does not apply across the children of a *lowered* `IfExpr` or
`and`/`or`: the condition lands in the `IfStmt` condition and each arm in its own
block, so order is preserved structurally. It also stops at a sealed position,
where no lowering fires at all. Lowerings are rare, so the naming it induces is
rare — that is what keeps the pass weak.

## Decisions

- A `while` loop is rotated **whenever its condition is not an atom**, not on
  `needs_slot` as ANF does. That is what makes the guarantee total, at the cost
  of duplicating essentially every loop condition. Each copy is the *original*
  expression, so each is small.
- Same criterion for the other gates: a ternary lowers when an arm is not an atom
  (ANF's rule already), and an `and`/`or` lowers when any operand after the first
  is not an atom — stronger than ANF's `needs_slot` gate.
- `CompToLoop` is the caller's job, and must run **first**: it creates the loop
  body that is the element's slot. `Hoistable.refusals` reports what it left.
- No `TypeInfer`. Naming is driven by the prefix rule rather than by type, so the
  pass needs only `DefineUse`, `Reachability` and `SyntaxCheck`.

## Consequences to keep in mind

1. **`and`/`or` guards are destroyed.** `ValueClassInfer._implied` matches an
   `And` to drop a runtime check; a lowered chain is statements joined by a phi.
   `anf.py`'s `_lowers_chain` avoids this by leaving a pure chain alone — a total
   guarantee cannot. Hoistable form is for the transform path.
2. **A rotated condition exists in two places.** A pass rewriting inside a
   `while` condition must rewrite both copies, and `sites()` reports two. ANF has
   this already.
3. **Aggregates get named.** The prefix rule names a left sibling whatever its
   type, so unlike ANF this pass can introduce a name holding a list — a second
   *place*, which the cpp backend's sharing analysis counts. Another reason it is
   not the backend's normalization.

## Checklist

- [x] 1. This document.
- [x] 2. The analyses — `lowers`, `lowers_inside`, `force_names` — as pure
      functions, with tests on the sets they compute.
- [x] 3. The transform: `_HoistableInstance`, `Hoistable.apply`,
      `Hoistable.refusals`, exported from `fpy2/transform/__init__.py`.
- [x] 4. Property tests over `ANF_PROFILE` draws (the interpreter must agree
      *including on which exception it raises* — that is what catches an ordering
      regression) and a corpus profile pinning how little the pass does.
- [x] 5. The scheduling operator `fpy2.strategies.to_hoistable`.

## What it cost, measured

Over the 230-function corpus, both passes establishing the same invariant:

| | statements added | residue |
|---|---|---|
| `Hoistable` | +66 | 36, every one a comprehension |
| `ANF` | +324 | — |

`CompToLoop` then `Hoistable` leaves 5, all comprehensions that pass declines.
`test_hoistable_profile.py` pins each of these; the growth figure is the only
test that can see the pass becoming less weak.

## Where this leaves the rest of the codebase

The pass currently sits beside `ANF` with no relationship to it, which is why the
ternary, chain and rotation rewrites exist twice. The fix is not to share them:
it is to make `ANF` *require* the guarantee rather than establish it.

> `ANF` does not lower. It throws when it cannot guarantee its own invariant, and
> `Hoistable` is what a caller runs to make sure it can.

That deletes the duplication rather than managing it, and it puts the dependency
the right way round — the strong normalization requires the weak one.

### What ANF's precondition is

Exactly the positions ANF itself would have to emit a statement into and cannot:
a sealed position holding something `needs_slot`. This is already what
`ANF.refusals` reports, so the guard is a filter on it, not a new analysis:

| in the input | ANF |
|---|---|
| `while i < n:` | accepts — nothing in the condition needs naming |
| `while fp.round(x) < n:` | throws |
| `y if c else fp.round(x)` | throws |
| `a and fp.round(b) > 0` | throws |
| `[f(x) for x in xs]` | accepts, and reports it, as today |

A comprehension is *not* a precondition failure: the cpp emitter gives the
element the loop body it generates and the iterable the `for` header, so ANF
declining to normalize one is a shape nothing gets wrong.

`fpy2.strategies.to_anf` inherits this — it raises rather than composing, so each
operator stays one rewrite and a schedule spells the order it wants.

### Measured before committing to it

Over the 230-function corpus:

| | |
|---|---|
| `ANF(f)` and `ANF(Hoistable(f))` byte-identical | 209 / 230 |
| dangerous positions in the raw corpus, which ANF would now throw on | 27 |
| ... after `Hoistable` | **0** — the precondition is satisfiable by construction |
| ANF's own residue, both ways | unchanged: 18 iterable, 8 element |
| statements: `ANF` vs `Hoistable` then `ANF` | +324 vs +361 |

The 21 functions that differ, and the +37 statements, are the loop conditions
`Hoistable` rotates and ANF's `needs_slot` gate did not. That cost is now paid in
generated C++ rather than in an intermediate form — the price of a total
guarantee, and the reason the two gates were allowed to differ in the first place.

### Phases

- [x] 6. This section.
- [x] 7. Move `_ATOMIC`, `_SEALED_REASON` and `_reads` from `anf.py` into
      `hoistable.py` and flip the import direction. `needs_slot` stays in
      `anf.py`, which is now its only user. A pure move: no behavior change, so
      every existing test must pass untouched.
- [x] 8. `ANF` drops the lowerings and gains the precondition. Deletes
      `_lowers`, `_lowers_chain`, `_branch_on`, `_bind`, `_arm`,
      `_short_circuit`, `_rotate` and the lowering branches of
      `_visit_if_expr` / `_visit_naryop` / `_visit_while` / `_visit_assign`
      (~250 lines, and `Reachability` stops being an ANF dependency). The
      ~22 tests in `test_anf.py`'s `TestRotation`, `TestTernaryLowering` and
      `TestBoolChainLowering` move to `test_hoistable.py`, minus what its 46
      tests already cover — audit rather than assume, several are not
      duplicated (`test_nested_loops_each_rotate`,
      `test_the_accumulator_never_clobbers_an_operand`,
      `test_the_guards_are_flat`, `test_short_circuit_is_preserved`).
      `test_anf_property.py` and `test_anf_profile.py` run `Hoistable` first;
      the profile's `EXPECTED_RESIDUE` is unchanged, and `EXPECTED_DANGEROUS`
      becomes the precondition assertion.
      The pipeline change came with it rather than after: a precondition and its
      only caller cannot land in separate commits. `CppCompiler.specialize()`
      runs `Hoistable` **unconditionally, just before the `if optimize:` block
      holding `RoundElim`** — see *Where it goes in the cpp pipeline* below.
- [ ] 9. **`ValueClassInfer._implied` reads a lowered guard.** Routing cpp
      through `Hoistable` lowers `not isnan(a) and not isnan(b)` into guarded
      statements, so `_implied` stops matching the `And` and `fp.fmin` emits a
      `quiet_NaN` check it used to prove away. The fact is *moved*, not lost:
      inside `if t:` where `t`'s reaching definitions are `not isnan(a)` and
      `not isnan(b)`, both hold. `_implied` already reads through a name via
      `DefineUseAnalysis.defining_expr`; what it cannot do is join two reaching
      definitions, and `ReachingDefs` in `fpy2/analysis/` is the missing piece.
      Three tests in `test_class_guards.py` carry a **strict** `xfail` naming
      this phase — remove the markers with the fix, and the suite will tell you
      if you forget.
- [ ] 10. The remaining paperwork:
      a line in `RoundElim`'s docstring and a test case saying it preserves
      hoistable form, which the pipeline slot relies on;
      `test_statement_form.py`'s `anf_disabled` fixture must patch out
      `Hoistable`, not `ANF`, or the net it tests is no longer down; the
      `ANF lowers all three` comment at `emitter.py:473`; `to_anf`'s docstring
      and `test_to_anf.py`'s `TestWhatItUnblocks`, which is now
      `to_hoistable`'s story and already covered in `test_to_hoistable.py`; and
      §1 of `backend-independence.md`, which describes ANF as the pass that
      creates the slots.

### Where it goes in the cpp pipeline

```
FreeVarElim
if optimize:  EnumerateElim -> ZipElim -> ReduceFusion
Specialize
Hoistable                       <- here, unconditional
if optimize:  RoundElim
ANF
```

**Before `RoundElim`, not after.** After it, the pass only satisfies ANF's
precondition and the lowering is pure overhead. Before it, `RoundElim` stops
refusing a ternary arm and a short-circuited operand — the reach the pass exists
to provide. On an eliminable rounding inside a ternary arm it is 2 eliminations
against 0.

The hazard that argues the other way does not materialize. Naming a left sibling
was expected to hide a rounding from `RoundElim`, but naming moves the expression
*into a statement*, which is if anything easier to reach: `fp.round(0.0)` as a
left sibling is folded either way.

**Unconditional**, since ANF requires it and ANF runs whatever `optimize` says.
With `optimize=False` the order degenerates to `Hoistable` then `ANF` adjacent.

**Not before `Specialize`.** `EnumerateElim`, `ZipElim` and `ReduceFusion` match
shapes in iterable position and need no statement slot, so the pass could only
break their matches.

**What the slot costs.** Every pass between `Hoistable` and `ANF` must preserve
hoistable form, and `RoundElim` is the only one. Over the corpus it breaks
neither the full guarantee nor ANF's precondition — but that is an observed fact,
not a stated contract, which is why phase 10 writes it down. A regression is loud
rather than silent: ANF's precondition turns it into a `TransformError` at
compile time instead of an un-normalized program reaching the emitter.

### What the stronger gates cost the backend

`Hoistable` rotates whenever a condition is not an atom, and lowers a chain
whenever a tail operand is not one. `ANF` gated both on `needs_slot`. Routing cpp
through `Hoistable` therefore changes emitted code in two ways, and they are not
the same kind of thing.

**Rotation is cosmetic.** Every loop is emitted as
`bool c = cond; while (c) { body; c = cond; }`. Measured on the countdown loop
from `test_emit_while.py`:

| | plain | rotated |
|---|---|---|
| `-O0` | 18 instructions | 26 |
| `-O2` | 9 instructions | 9 — *identical machine code* |

Loop rotation is a transformation the C++ compiler performs itself, so the cost
is readability of the emitted source and `-O0` only. The three tests that pinned
the old shape now pin the new one.

**The guard loss is real** — an `isnan` branch the C++ compiler cannot prove
away — and is phase 9.

A narrower gate for the cpp path was considered and rejected. It would have to be
`needs_slot`, which is not a property of the program but a *whitelist predicting
what an emitter will want a statement for* — its own docstring says
"plausibly", and §1 of `backend-independence.md` is explicit that a
normalization takes no target fact. A `Hoistable` mode gated on it would have no
statable invariant: `while x*y > 0` comes back un-rotated, so the output is not
in hoistable form, `refusals` reports a position the pass itself declined to fix,
and "run `Hoistable` first" stops being sound advice. Not lowering chains at all
was also considered and is impossible: the corpus has 13 chain positions holding
something `needs_slot`, ANF must name inside them, and the emitter has a recorded
miscompile there.
