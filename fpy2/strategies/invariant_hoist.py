"""
Scheduling language: loop-invariant code motion
"""

from ..function import Function
from ..transform import Cursor, HoistInvariant


def hoist_invariant(
    func: Function,
    where: int | Cursor | None = None,
) -> Function:
    """:class:`fpy2.transform.HoistInvariant` over *func*: work in a loop body
    whose result cannot change from one iteration to the next is done once,
    above the loop.

    Relocation, not re-association, so it holds under any rounding context --
    unlike the algebraic rewrites, which need an exact one.  Only a *direct*
    child of the body is considered, which is what keeps the destination in the
    scope the work was already written in: neither loop form opens a context
    scope, and nor does any expression, so a statement one level down and every
    subexpression of it are already in the scope they land in.  A binding under
    a ``with`` inside the body therefore stays.

    A whole binding moves when its expression is pure, every name it reads was
    bound before the loop, its own name is bound just once in the body, and no
    *other* definition of that name is ever read.  The last of those covers two
    readers that would otherwise see the hoisted value: one after a loop that
    runs zero times, which should see whatever reached the loop, and one
    earlier in the body, which reads through the loop's phi and so should see
    the previous iteration's value.  Hoisting *does* make the expression
    evaluate where it previously would not have, which FPy permits -- the value
    the function returns is unchanged.

    Where the binding is pinned, its invariant *subexpressions* still move,
    each bound to a fresh name above the loop.  That is what reaches an operand
    which was never a statement -- ``(2 ** -k) * x`` keeps the multiply in the
    body and takes the power out -- and it applies to a pinned binding's whole
    right-hand side too, since what pins the name says nothing about the work.
    A bare name is left alone, as is an expression that reads nothing, which is
    :func:`fpy2.strategies.simplify`'s to fold rather than this pass's to move.
    So is a position that is not evaluated every time its statement is reached
    -- an ``and``/``or`` operand after the first, a ternary arm, a
    comprehension element -- since moving one of those runs it where the
    original may never have, and the guard in front of it is often the point.

    A loop body that writes into a list in place pins every read of a list:
    reaching definitions model ``zs[i] = v`` as a definition of ``zs`` alone,
    so a read of an aliased ``ys`` would otherwise look invariant.

    Idempotent: a second application finds only names and non-invariant
    expressions where the first left them.

    One pass.  A chain comes out together, each hoisted binding counting as
    invariant for the ones after it, but a binding freed by hoisting out of an
    *inner* loop needs another -- apply the strategy again where a schedule
    wants that.  :func:`fpy2.strategies.simplify` does *not* run this pass:
    moving a computation is not a simplification, and nothing downstream
    depends on it having happened.

    Cursors forward across this pass, except an expression cursor inside a
    statement a subexpression was lifted out of: that statement's expressions
    were rewritten, so the cursor no longer names what it named.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which loop to hoist out of: an index counting the loops this rewrite
        acts on, in visit order, outermost-first, or a cursor or region, which
        takes every one at or beneath it. If `None`, hoist out of them all. A
        loop with nothing to hoist is not one of them and takes no index;
        naming it with a cursor says why each of its statements stayed.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a loop with nothing to hoist; the message
        says why each statement stayed.
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
