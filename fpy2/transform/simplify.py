"""
Simplification: constant folding, copy propagation, dead-code elimination and
context unnesting, run to a fixpoint.

Each pass exposes what the others need to make progress on: folding an operand
to a literal makes a copy redundant, propagating a copy makes its definition
dead, deleting a definition can leave a block with nothing in it to fold, and
deleting a statement can leave a nested ``with`` at the edge of its parent,
where it can be unnested.  Running them once in any order leaves work behind,
so they run until none of them reports a change.  `UnnestContext` runs last, so
that a block the other three expose is taken in the same round.

Termination: measure the program by **(number of statements, number of nested
``ContextStmt`` ancestor pairs)**, ordered lexicographically.  Folding, copy
propagation and dead-code elimination only ever remove a definition or replace
an expression with something smaller, so they can only decrease the first
component, and removing statements never creates a new ancestor relationship.
`UnnestContext` restructures rather than shrinks: it holds the statement count
fixed and strictly decreases the second.  So no cycle can grow the program.

This is the cleanup a lowering-heavy pipeline needs.  A rewrite that lands a
loop where an expression was leaves debris behind it -- a length read into a name
nothing goes on to use, a copy of an accumulator -- and the pass that emitted it
is not the one that can tell.
"""

from ..ast.fpyast import FuncDef
from .const_fold import ConstFold
from .copy_propagate import CopyPropagate
from .dead_code import DeadCodeEliminate
from .unnest_context import UnnestContext


class Simplify:
    """Constant folding, copy propagation, dead-code elimination and context
    unnesting, to a fixpoint."""

    @staticmethod
    def apply(
        func: FuncDef, *,
        enable_const_fold: bool = True,
        enable_const_fold_context: bool = False,
        enable_const_fold_op: bool = True,
        enable_copy_prop: bool = True,
        enable_dead_code_elim: bool = True,
        enable_unnest_context: bool = True,
    ) -> FuncDef:
        """Simplify *func* until none of the enabled passes reports a change.

        Context folding is off by default: it replaces a ``with``-block's
        expression with the resolved :class:`~fpy2.Context` object, whose
        ``repr`` is not FPy source for anything but the named formats, so the
        result no longer re-parses.
        """
        func, _ = Simplify.apply_with_status(
            func,
            enable_const_fold=enable_const_fold,
            enable_const_fold_context=enable_const_fold_context,
            enable_const_fold_op=enable_const_fold_op,
            enable_copy_prop=enable_copy_prop,
            enable_dead_code_elim=enable_dead_code_elim,
            enable_unnest_context=enable_unnest_context,
        )
        return func

    @staticmethod
    def apply_with_status(
        func: FuncDef, *,
        enable_const_fold: bool = True,
        enable_const_fold_context: bool = False,
        enable_const_fold_op: bool = True,
        enable_copy_prop: bool = True,
        enable_dead_code_elim: bool = True,
        enable_unnest_context: bool = True,
    ) -> tuple[FuncDef, bool]:
        """Same as :meth:`apply` but also returns a ``changed`` flag —
        ``True`` iff at least one pass rewrote something."""
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\' for {func}, got {type(func)}')

        ever = False
        while True:
            changed = False
            if enable_const_fold:
                func, c = ConstFold.apply_with_status(
                    func,
                    enable_context=enable_const_fold_context,
                    enable_op=enable_const_fold_op,
                )
                changed |= c
            if enable_copy_prop:
                func, c = CopyPropagate.apply_with_status(func)
                changed |= c
            if enable_dead_code_elim:
                func, c = DeadCodeEliminate.apply_with_status(func)
                changed |= c
            if enable_unnest_context:
                func, c = UnnestContext.apply_with_status(func)
                changed |= c
            if not changed:
                return func, ever
            ever = True
