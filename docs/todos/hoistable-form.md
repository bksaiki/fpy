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
- [ ] 2. The two analyses — `lowers`, `_lowers_inside`, `force_names` — as pure
      functions, with tests on the sets they compute.
- [ ] 3. The transform: `_HoistableInstance`, `Hoistable.apply`,
      `Hoistable.refusals`, exported from `fpy2/transform/__init__.py`.
- [ ] 4. Property tests over `ANF_PROFILE` draws (the interpreter must agree
      *including on which exception it raises* — that is what catches an ordering
      regression) and a corpus profile pinning how few names the pass introduces.
- [ ] 5. The scheduling operator `fpy2.strategies.to_hoistable`.

### Follow-up

**Unify the lowerings with `anf.py`.** The two passes now hold separate copies of
the ternary, chain and rotation rewrites; only `needs_slot` and `_ATOMIC` are
shared by import. A lowering bug has to be fixed twice. Once both passes are
settled, `ANF` should be `Hoistable` plus atomization — one implementation of the
lowerings, with the naming predicate and the rotation gate as the only
differences.
