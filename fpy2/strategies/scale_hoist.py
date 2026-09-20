"""
Scheduling language: reduction scale-hoisting
"""

from ..function import Function
from ..transform import Cursor, HoistScale


def hoist_scale(
    func: Function,
    where: int | Cursor | None = None,
) -> Function:
    """:class:`fpy2.transform.HoistScale` over *func*: where every element of a
    reduction is scaled by the same factor, the scaling becomes one multiply
    after the reduction instead of one per element.  `sum`, `max` and `min`
    are all reductions for this purpose, under different conditions.

    Two shapes, the same rewrite.  A comprehension is the one a program is
    written in::

        sum([c * e for x in xs])  ->  c * sum([e for x in xs])

    and a loop filling a list is what it becomes after
    :func:`fpy2.strategies.comp_to_loop`, which
    :func:`fpy2.strategies.rescale_fixed` requires.

    Algebra, not relocation, so unlike :func:`fpy2.strategies.hoist_invariant`
    it needs an **exact** scope: under a rounding one the partial sums round
    and the two orders disagree -- `sum([3 * x for x in [1e20, 1.0, -1e20]])`
    is `0` under `fp.FP32` and `6.01226e12` the other way round.  It is a
    per-site question, so one reduction of a function may be rewritten while
    another is refused.

    A **selection** -- `max` or `min` -- needs that too, and one thing more.
    It may look as though it should not, since it picks an element rather than
    accumulating and so does not round.  But the *multiply* rounds, and
    `c * x` makes a NaN out of `0 * inf`, which a selection propagates from any
    element while `c * max(xs)` only ever computes the selected one.  So the
    factor must be **non-zero** as well as finite.

    Whichever the reduction, the factor must be:

    - **pure**, since it goes from once per element to once;
    - **invariant** -- it may not read what the comprehension or the loop binds
      per element;
    - **non-negative**.  A negative factor reorders a selection outright, and
      signs a zero a sum never signed: `sum([])` is `+0.0` and `c * sum([])`
      is `-0.0`.

    Sign is established syntactically: the factor must be a power with a
    positive literal base, which is what `rescale_fixed` emits.  A correct
    factor of another shape is declined rather than accepted -- incomplete,
    not unsound.

    Over a list a loop filled there are two further conditions, neither of
    which a comprehension raises: the loop must write **every** element, or an
    unwritten one would be scaled too, and the list must be read by nothing but
    the reduction.  Coverage is decided by the size union-find, so it holds for
    a symbolic length as much as a concrete one.

    Cursors forward across this pass, except an expression cursor inside a
    statement it rewrote -- the reduction's own statement, and the one binding
    the product.

    Parameters
    ----------
    func : Function
        The function to transform.
    where : int | Cursor | None
        Which reduction to rewrite: an index counting the reductions this
        rewrite acts on, in visit order, or a cursor or region, which takes
        every one at or beneath it. If `None`, rewrite them all. A reduction
        this rewrite refuses is not one of them and takes no index; naming it
        with a cursor says which condition failed.

    Returns
    -------
    Function
        The transformed function.

    Raises
    ------
    TransformDeclined
        If an explicit `where` names a reduction this rewrite refuses; the
        message says why.
    TransformReferenceError
        If an explicit `where` names no such reduction, or a cursor of a
        program this one was not derived from.

    Examples
    --------
    ::

        @fp.fpy(ctx=fp.REAL)
        def scaled_sum(xs: list[fp.Real], k: fp.Real) -> fp.Real:
            if fp.isfinite(k):
                return sum([(2 ** k) * x for x in xs])
            else:
                return 0.0

    ``hoist_scale(scaled_sum)`` yields::

        @fp.fpy(
            ctx=fp.REAL,
        )
        def scaled_sum(xs, k):
            if fp.isfinite(k):
                return ((2 ** k) * sum([x for x in xs]))
            else:
                return 0
    """
    if not isinstance(func, Function):
        raise TypeError(f"Expected a \'Function\', got {func}")

    return func.with_edits(
        HoistScale.apply_with_edits(func.ast, func.rebase(where))
    )
