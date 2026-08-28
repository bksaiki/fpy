"""
Hoistable form: every expression sits where a statement may be inserted above it.

FPy has statements and expressions, so a pass needing a temporary for an
expression has to hoist it into a statement above -- and that is not always
sound.  This pass establishes the invariant that makes it always sound:

    Every expression node is evaluated exactly once, unconditionally, whenever
    its enclosing statement is reached.

Under it the slot immediately before the enclosing statement runs exactly as
often, and under exactly the same condition, as any expression in that
statement, so it is always a legal place for a hoisted temporary.  A pass mints
one on demand and never reasons about conditional evaluation again.

**The non-strict positions.**  Four, and the list is closed -- every other
expression position is a statement operand or an operand of a strict operator:

- an ``IfExpr`` arm, which becomes an ``IfStmt`` assigning one name;
- an ``and``/``or`` tail, which becomes a flat chain of guarded statements;
- a ``while`` condition, whose loop is *rotated* -- the condition evaluated once
  before the loop and once at the end of the body, FPy's own order;
- a comprehension's element and iterables, which
  :class:`~fpy2.transform.CompToLoop` turns into an allocation and a loop.
  That pass is a caller's job and must run *first*: it creates the loop body
  that is the element's slot.

**Weaker than ANF, deliberately.**  :class:`~fpy2.transform.ANF` establishes the
same invariant, but as a side effect of also binding every nameable
subexpression to a name -- which the cpp emitter needs and a rewrite does not.
Over ``examples/`` ANF names up to 5096 subexpressions where only 71 positions
need a lowering.  This pass does the lowering half and leaves ``anf.py`` alone.

**The ordering hazard.**  Lowering alone is *not* semantics-preserving.  Hoisting
a lowered construct out of an operand moves it above the operands to its left,
which are then evaluated later than they were:

.. code-block:: python

    return g(a) + (h(b) if c else 0.0)   # raises g's assertion

    if c: t = h(b)                       # naive lowering
    else: t = 0.0
    return g(a) + t                      # raises h's assertion -- wrong

ANF avoids this only because atomization names ``g(a)`` too, left to right.  So
this pass keeps *part* of the naming, and only that part: see
:func:`force_names`.

The design notes are in ``docs/todos/hoistable-form.md``.
"""

from ..ast.fpyast import (
    And,
    Expr,
    IfExpr,
    ListComp,
    Or,
    Stmt,
)
from .anf import _ATOMIC
from .path import sub_exprs


def lowers(e: Expr) -> bool:
    """Whether this pass emits a statement *at* `e`.

    A ternary lowers whenever an arm is not an atom, and a chain whenever an
    operand after the first is not one: those are exactly the operands with
    nowhere to put a statement.  Both criteria are stricter than
    :func:`~fpy2.transform.anf.needs_slot`, which asks whether a *particular*
    lowering wants a slot rather than whether one could ever be wanted.
    """
    match e:
        case IfExpr():
            return not (
                isinstance(e.ift, _ATOMIC) and isinstance(e.iff, _ATOMIC)
            )
        case And() | Or():
            return any(not isinstance(a, _ATOMIC) for a in e.args[1:])
        case _:
            return False


def lowers_inside(e: Expr) -> bool:
    """Whether this pass emits a statement anywhere in `e`, `e` itself included.

    A comprehension is the only *sealed* position needing an exception.  The
    other two seal an unlowered ternary's arms and an unlowered chain's tail --
    but unlowered means those operands are atoms, and an atom has no children,
    so the recursion finds nothing there on its own.
    """
    if lowers(e):
        return True
    if isinstance(e, ListComp):
        return False
    return any(lowers_inside(sub) for _field, _i, sub in sub_exprs(e))


def force_names(node: 'Stmt | Expr') -> set[Expr]:
    """The expressions in `node` to bind to a name, so a lowering to their right
    does not overtake them.

    The *prefix rule*: at any node, let ``last`` be the position of the last
    child -- in :func:`~fpy2.transform.path.sub_exprs` order, which is
    evaluation order -- that a lowering fires inside.  Every earlier child that
    is not already an atom is named, since a lowering hoists above the whole
    statement and would otherwise run before them.

    .. code-block:: python

        f(g(y), a if c else b)   # -> {g(y)}: the ternary hoists above it
        f(a if c else b, g(y))   # -> {}: nothing runs before the ternary
        xs[i + 1] = a if c else b  # -> {i + 1}: an index runs before the value

    A lowered ``IfExpr`` or chain is exempt: its condition lands in the
    ``IfStmt`` condition and each arm in a block of its own, so their order is
    preserved structurally -- and naming an arm is the very bug the rule exists
    to prevent.  A comprehension is sealed.

    Identity, not structure: ``Expr`` defines no ``__eq__``, so the set holds
    the nodes themselves and two structurally-equal operands stay distinct.

    Blocks are not entered.  A statement of a nested block gets its own call,
    since its own block is the slot its temporaries belong in.
    """
    out: set[Expr] = set()
    _collect(node, out)
    return out


def _collect(node: 'Stmt | Expr', out: set[Expr]) -> None:
    """Accumulate :func:`force_names` for `node` and everything under it."""
    if isinstance(node, ListComp):
        return
    kids = [sub for _field, _i, sub in sub_exprs(node)]
    # A ternary or chain is exempt: lowered, its condition lands in the
    # `IfStmt` condition and each arm in a block of its own; unlowered, its
    # arms are atoms and the rule would find nothing anyway.
    if not isinstance(node, (IfExpr, And, Or)):
        lowering = [i for i, kid in enumerate(kids) if lowers_inside(kid)]
        if lowering:
            out.update(
                kid for kid in kids[:max(lowering)]
                if not isinstance(kid, _ATOMIC)
            )
    for kid in kids:
        _collect(kid, out)
