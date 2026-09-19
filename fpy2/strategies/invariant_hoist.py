"""
Scheduling language: loop-invariant code motion
"""

from ..function import Function
from ..transform import Cursor, HoistInvariant


def hoist_invariant(
    func: Function,
    where: int | Cursor | None = None,
) -> Function:
    """:class:`fpy2.transform.HoistInvariant` over *func*: a binding in a loop
    body whose value cannot change from one iteration to the next is computed
    once, above the loop.

    Relocation, not re-association, so it holds under any rounding context --
    unlike the algebraic rewrites, which need an exact one.  Only a *direct*
    child of the body moves, which is what keeps the destination in the scope
    the binding was already written in: neither loop form opens a context
    scope, so a statement one level down is already in the scope it lands in.
    A binding under a ``with`` inside the body therefore stays.

    A binding moves when its expression is pure, every name it reads was bound
    before the loop, its own name is bound just once in the body, and that name
    is not read from outside the body.  The last of those is what a loop that
    runs zero times needs: a reader after the loop would otherwise see the
    hoisted value in place of whatever reached the loop.  Hoisting *does* make
    the expression evaluate where it previously would not have, which FPy
    permits -- the value the function returns is unchanged.

    One pass.  A chain comes out together, each hoisted binding counting as
    invariant for the ones after it, but a binding freed by hoisting out of an
    *inner* loop needs another; :func:`fpy2.strategies.simplify` provides that
    fixpoint.

    Cursors forward across this pass.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which loop to hoist out of: an index counting the loops this rewrite
        acts on, in visit order, outermost-first, or a cursor or region, which
        takes every one at or beneath it. If `None`, hoist out of them all. A
        loop with nothing to hoist is not one of them and takes no index;
        naming it with a cursor says why each of its bindings stayed.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a loop with nothing to hoist; the message
        says why each binding stayed.
    TransformReferenceError
        If an explicit `where` names no such loop, or a cursor of a program
        this one was not derived from.

    Examples
    --------
    ::

        @fp.fpy(ctx=fp.REAL)
        def scaled_sum(xs: list[fp.Real]) -> fp.Real:
            n = len(xs)
            acc = 0.0
            for x in xs:
                c = n + 1
                acc = acc + c * x
            return acc

    ``hoist_invariant(scaled_sum)`` yields::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def scaled_sum(xs):
            n = len(xs)
            acc = 0
            c = (n + 1)
            for x in xs:
                acc = (acc + (c * x))
            return acc
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(
        HoistInvariant.apply_with_edits(func.ast, func.rebase(where))
    )
