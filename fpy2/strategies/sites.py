"""
Scheduling language: listing the sites a strategy can be aimed at
"""

from collections.abc import Callable

from ..function import Function
from ..transform import (
    Cursor,
    FloatToFixed,
    ForUnroll,
    FuncInline,
    RescaleFixed,
    RoundInsert,
    SplitLoop,
    UnfoldNegZero,
    UnfoldOverflow,
    UnfoldSpecial,
    WhileUnroll,
)
from .fixed_rescale import rescale_fixed
from .float_lower import float_to_fixed
from .func_inline import inline
from .loop_split import split
from .loop_unroll import unroll_for, unroll_while
from .neg_zero_unfold import unfold_neg_zero
from .overflow_unfold import unfold_overflow
from .round_insert import insert_round
from .special_unfold import unfold_special

_SITES: dict[Callable, Callable] = {
    unfold_special: UnfoldSpecial.sites,
    unfold_neg_zero: UnfoldNegZero.sites,
    unfold_overflow: UnfoldOverflow.sites,
    float_to_fixed: FloatToFixed.sites,
    rescale_fixed: RescaleFixed.sites,
    insert_round: RoundInsert.sites,
    split: SplitLoop.sites,
    unroll_for: ForUnroll.sites,
    unroll_while: WhileUnroll.sites,
    inline: FuncInline.sites,
}
"""Which strategies can be aimed, and what lists their sites.

A strategy absent from this table takes no `where`: it applies to the whole
program or to nothing.
"""


def sites(
    strategy: Callable,
    func: Function,
    within: Cursor | None = None,
    **kwargs,
) -> list[Cursor]:
    """
    The sites `strategy` could rewrite in `func`, in the order a `where` index
    counts them.

    The kind of cursor is the kind of site the strategy is aimed at: a
    :class:`fpy2.strategies.StmtCursor` for the rounding and loop rewrites, an
    :class:`fpy2.strategies.ExprCursor` for :func:`fpy2.strategies.inline`,
    whose sites are calls.

    For the rounding rewrites a listing is *what `where=None` would rewrite*: a
    candidate the strategy refuses is not a site, so it neither appears here nor
    consumes an index.  Naming one with a cursor still says why it was refused.
    The loop and call rewrites refuse nothing, so the distinction does not
    arise for them.

    Some strategies do not forward cursors (they rewrite at sites they do not
    report); aim what you need before one of those, or re-list afterwards.

    Parameters
    ----------
    strategy : Callable
        The strategy whose sites to list.
    func : Function
        The function to scan.
    within : Cursor | None
        Keep only the sites at or beneath this cursor or region; one from an
        earlier program is forwarded first. An expression cursor bounds a search
        here, where as a `where` it names one site exactly. If `None`, list the
        whole program.
    kwargs
        Forwarded to the strategy's own listing; :func:`fpy2.strategies.inline`
        takes `funcs`, matching its filter.

    Returns
    -------
    list[Cursor]
        The sites, in visit order, outermost-first.

    Raises
    ------
    ValueError
        If `strategy` takes no `where`, so has no sites to list.

    Examples
    --------
    ::

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP16:
                p = fp.round(x)
            with fp.FP16:
                q = fp.round(y)
            return p + q

    ``sites(unfold_special, f)`` names both rounding blocks, and the first is
    what ``where=0`` means::

        [body[0] at `f.py:8:8`, body[1] at `f.py:10:8`]
    """
    if not isinstance(func, Function):
        raise TypeError(f'Expected a \'Function\', got {func}')
    lister = _SITES.get(strategy)
    if lister is None:
        name = getattr(strategy, '__name__', strategy)
        raise ValueError(f'`{name}` takes no `where`, so it has no sites')
    return lister(func.ast, func.rebase(within), **kwargs)
