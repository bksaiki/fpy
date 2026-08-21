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

    The sites are individual operations.  A context applies to every operation
    in its block, so an operation is given a format of its own by being lifted
    into a block alone, each operand that is not already a variable bound under
    the original scope first.  One operation therefore gains a rounding and the
    rest of its statement stays exact — and since the inserted rounding is
    verified an identity, it changes no value, so operations can be given
    formats one at a time and in any order.

    An operation whose scope already rounds is not a site, and neither is one
    with nowhere to put the block: a loop condition, a comprehension, a
    conditional branch.

    Run :func:`fpy2.strategies.simplify` afterwards to fold away the
    temporaries the rewrite introduces.

    Parameters
    ----------
    func : Function
        The function to transform.
    ctx : Context
        The format to round to. The rewrite is refused wherever rounding to it
        would change the result.
    where : int | Cursor | None
        Which operation to round: an index counting candidate operations (the
        roundable operations whose scope rounds exactly, whether or not they
        verify) in visit order, outermost-first, or a cursor, which names one
        exactly, or a statement cursor or region, which takes every candidate
        at or beneath it. If `None`, round every candidate that verifies and
        skip the rest.

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
        If an explicit `where` names no candidate operation, or a cursor of a
        program this one was not derived from.

    Examples
    --------
    A sum of squares computed exactly, with FP32 arguments::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def sum_sq(x, y):
            with fp.REAL:
                t = ((x * x) + (y * y))
            return t

    ``insert_round(sum_sq, fp.FP64)`` gives each multiply a format, since the
    48-digit product of two FP32 values fits FP64's 53 exactly::

        @fp.fpy(
            ctx=fp.FP64,
        )
        def sum_sq(x, y):
            with fp.REAL:
                with fp.FP64:
                    _t = (x * x)
                with fp.FP64:
                    _t4 = (y * y)
                t = (_t + _t4)
            return t

    The add is left exact, and `where=0` naming it is declined: the exact sum
    of two 48-digit products needs far more than 53 digits, so rounding it
    would change the result.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(RoundInsert.apply_with_edits(
        func.ast, ctx, where=func.rebase(where)
    ))
