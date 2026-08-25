"""
Scheduling language: rounding splitting
"""

from ..function import Function
from ..number import Context
from ..transform import Cursor, SplitRound


def split_round(
    func: Function, ctx: Context, where: int | Cursor | None = None
) -> Function:
    """
    Expand one rounding in `func` into two that compose to the same answer.

    A correctly-rounded operation is a real computation followed by a rounding.
    Where the pair of formats and modes satisfies the correct-double-rounding
    table, that single rounding equals a rounding to an intermediate followed by
    a rounding to the target, so either may replace the other.
    :func:`fpy2.strategies.elim_round` and :func:`fpy2.strategies.insert_round`
    trade one rounding against none; this trades one against two.

    The point is to implement an operation the environment does not have.  A
    library with no FP32 multiply can compute the product wide under round-to-odd
    and re-round to FP32, and get exactly what the FP32 multiply would have
    given.  That is the recipe this operator is one step of.

    `ctx` is the **intermediate**, not the target: the target is whatever format
    the operation already rounds to, read from the program.
    :func:`fpy2.analysis.format_infer.derive_intermediate` computes a suitable
    intermediate for a target, so a caller need not work out the extra precision
    by hand.

    The sites are individual rounded operations — the complement of
    :func:`fpy2.strategies.insert_round`'s, which takes the exact ones.  An
    operation is split by being lifted into a block of the intermediate, with an
    explicit rounding back to the target in the enclosing block, since an
    assignment rounds nothing in FPy.

    Most pairs are unsound and refused.  In particular **round-to-nearest over
    round-to-nearest is not sound**, whatever the intermediate's width — and
    every ``fp.FP*`` context is round-to-nearest, so a hand-written program
    almost always wants a round-to-odd intermediate.  Also refused: a scope that
    rounds exactly or is symbolic, a stochastic context, a bounded intermediate,
    and an operation with nowhere to put the block.

    Not idempotent: round-to-odd over round-to-odd is itself admissible, so
    applying it again splits again.  Run :func:`fpy2.strategies.simplify`
    afterwards to fold away the temporaries.

    Parameters
    ----------
    func : Function
        The function to transform.
    ctx : Context
        The intermediate to round through, carrying both its format and its
        rounding mode. The rewrite is refused wherever the composition would not
        equal the single rounding it replaces.
    where : int | Cursor | None
        Which operation to split: an index counting the operations this rewrite
        would split through `ctx`, in visit order, outermost-first, or a cursor,
        which names one exactly, or a statement cursor or region, which takes
        every one at or beneath it. If `None`, split them all. An operation this
        rewrite refuses is not one of them and takes no index; naming it with a
        cursor says why it was refused.
        :func:`fpy2.strategies.sites` lists them, and needs the same `ctx`.

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
    An FP32 product::

        @fp.fpy(
            ctx=fp.FP32,
        )
        def scale(x, y):
            return (x * y)

    With ``via = derive_intermediate(fp.FP32)`` — 26 digits under round-to-odd —
    ``split_round(scale, via)`` computes the product there and re-rounds::

        @fp.fpy(
            ctx=fp.FP32,
        )
        def scale(x, y):
            with MPBFloatContext(pmax=26, ..., rm=RoundingMode.RTO):
                _t = (x * y)
            return fp.round(_t)

    Passing ``fp.FP64`` instead is declined: both it and the FP32 target round to
    nearest, and no width of intermediate makes that composition an identity.
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(SplitRound.apply_with_edits(
        func.ast, ctx, where=func.rebase(where)
    ))
