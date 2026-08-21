"""
Scheduling language: rounding insertion
"""

from ..function import Function
from ..number import Context
from ..transform import Cursor, RoundInsert


def insert_round(
    func: Function, ctx: Context, where: int | Cursor | None = None
) -> Function:
    """
    Give an exact operation in `func` a format, where the rounding does nothing.

    A correctly-rounded operation is a real computation followed by a rounding,
    so wherever the real result already lies in the target format the two are
    the same function.  :func:`fpy2.strategies.elim_round` reads that identity
    one way, dropping a rounding no one can observe; this reads it the other,
    giving an operation under ``fp.REAL`` a format it can be lowered onto.

    The point is not to add work.  An operation under ``fp.REAL`` names no
    format, so no environment's arithmetic implements it; the same operation
    under ``fp.FP64`` is a hardware multiply.  Inserting the rounding is what
    makes a real-valued specification implementable.

    Only a ``with fp.REAL:`` block holding a single assignment is rewritten.
    A block that already rounds has no rounding to insert, and a block of
    several assignments is declined: rounding the first would change the
    operand of the next, which needs an analysis this pass does not do.  Run
    :func:`fpy2.strategies.elim_round` first, which emits one block per
    operation.

    `ctx` is required and not inferred.  Reading the enclosing scope would
    only reproduce ``elim_round``'s own choice, and deriving a format from the
    operation's inferred bound yields formats no hardware has — the point of
    the rewrite is to land on an operation the environment provides.

    Parameters
    ----------
    func : Function
        The function to transform.
    ctx : Context
        The format to round to. The rewrite is refused wherever rounding to it
        would change the result.
    where : int | Cursor | None
        Which block to round: an index counting candidate blocks (the
        structurally-matching exact blocks, whether or not they verify)
        in visit order, outermost-first, or a cursor or region, which takes every
        candidate at or beneath it. If `None`, round every candidate that
        verifies and skip the rest.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a candidate this rewrite refuses, or a
        region whose every candidate it refuses; the message says why.
    TransformReferenceError
        If an explicit `where` names no candidate block, or a cursor of a
        program this one was not derived from.

    Examples
    --------
    With ``prod3`` monomorphized to FP32 arguments under an FP64 context,
    ``elim_round`` proves the inner multiply exact and unrounds it::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def prod3(x, y, z):
            with fp.REAL:
                _t = (x * y)
            return (_t * z)

    ``insert_round(hoisted, fp.FP64)`` gives it back a format, since the
    48-digit product of two FP32 values fits FP64's 53 exactly::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def prod3(x, y, z):
            with fp.FP64:
                _t = (x * y)
            return (_t * z)

    ``insert_round(hoisted, fp.FP32)`` is declined: 48 digits do not fit in 24,
    so the rounding would change the result.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(RoundInsert.apply_with_edits(
        func.ast, ctx, where=func.rebase(where)
    ))
