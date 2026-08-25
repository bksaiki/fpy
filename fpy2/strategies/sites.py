"""
Scheduling language: listing the sites a strategy can be aimed at
"""

from collections.abc import Callable

from ..function import Function
from ..transform import (
    CompToLoop,
    Cursor,
    FloatToFixed,
    ForUnroll,
    FuncInline,
    RescaleFixed,
    RoundInsert,
    SplitLoop,
    SplitRound,
    UnfoldNegZero,
    UnfoldOverflow,
    UnfoldSpecial,
    WhileUnroll,
)
from .comp_lower import comp_to_loop
from .fixed_rescale import rescale_fixed
from .float_lower import float_to_fixed
from .func_inline import inline
from .loop_split import split
from .loop_unroll import unroll_for, unroll_while
from .neg_zero_unfold import unfold_neg_zero
from .overflow_unfold import unfold_overflow
from .round_insert import insert_round
from .round_split import split_round
from .special_unfold import unfold_special

_REFUSALS: dict[Callable, Callable] = {
    unfold_special: UnfoldSpecial.refusals,
    unfold_neg_zero: UnfoldNegZero.refusals,
    unfold_overflow: UnfoldOverflow.refusals,
    float_to_fixed: FloatToFixed.refusals,
    rescale_fixed: RescaleFixed.refusals,
    insert_round: RoundInsert.refusals,
    split_round: SplitRound.refusals,
    comp_to_loop: CompToLoop.refusals,
    inline: FuncInline.refusals,
    split: SplitLoop.refusals,
    unroll_for: ForUnroll.refusals,
}
"""Which strategies can explain a refusal, and what explains it.

A strategy absent from this table reports none, which is right only where it
refuses nothing: `unroll_while` is the one such strategy.
"""


_SITES: dict[Callable, Callable] = {
    unfold_special: UnfoldSpecial.sites,
    unfold_neg_zero: UnfoldNegZero.sites,
    unfold_overflow: UnfoldOverflow.sites,
    float_to_fixed: FloatToFixed.sites,
    rescale_fixed: RescaleFixed.sites,
    insert_round: RoundInsert.sites,
    split_round: SplitRound.sites,
    comp_to_loop: CompToLoop.sites,
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
    /,
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

    A listing depends on whatever else decides the answer, so pass the same
    arguments the rewrite will get: `insert_round` needs `ctx`, and `split` and
    `unroll_for` need the `strategy` (and its `factor` / `times`), since only
    ``STRICT`` refuses a loop.

    Some strategies do not forward cursors (they rewrite at sites they do not
    report); aim what you need before one of those, or re-list afterwards.

    Parameters
    ----------
    strategy : Callable
        The strategy whose sites to list. Positional: a strategy of its own may
        take a `strategy` argument, which `kwargs` then carries.
    func : Function
        The function to scan.
    within : Cursor | None
        Keep only the sites at or beneath this cursor or region; one from an
        earlier program is forwarded first. An expression cursor bounds a search
        here, whereas a `where` names one site exactly. If `None`, list the
        whole program.
    kwargs
        Forwarded to the strategy's own listing, matching that strategy's own
        parameters: :func:`fpy2.strategies.inline` takes `funcs` and
        :func:`fpy2.strategies.insert_round` takes `ctx`, which it needs because
        whether an operation is a site depends on the format being inserted.

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


def refusals(
    strategy: Callable,
    func: Function,
    /,
    within: Cursor | None = None,
    **kwargs,
) -> list[tuple[Cursor, str]]:
    """
    Why each program point `strategy` could have acted on in `func` is not a
    site, paired with the reason, in visit order.

    :func:`fpy2.strategies.sites` reports what `strategy` *would* rewrite; this
    reports what it passed over and why.  A refused point takes no index, so a
    listing cannot show it and only a cursor can name it -- which is what makes
    this the way to find one.  Together the two account for every point the
    strategy considered.

    Always empty for `unroll_while`, which refuses nothing.

    Parameters
    ----------
    strategy : Callable
        The strategy whose refusals to explain.
    func : Function
        The function to scan.
    within : Cursor | None
        Keep only the points at or beneath this cursor or region; one from an
        earlier program is forwarded first. If `None`, scan the whole program.
    kwargs
        Forwarded as they are for :func:`fpy2.strategies.sites`, since the same
        parameters decide the answer: :func:`fpy2.strategies.insert_round` takes
        `ctx`.

    Returns
    -------
    list[tuple[Cursor, str]]
        Each refused point and its reason, in visit order, outermost-first.

    Raises
    ------
    ValueError
        If `strategy` takes no `where`, so has no sites to refuse.

    Examples
    --------
    Why :func:`fpy2.strategies.rescale_fixed` leaves a float program alone::

        [(body[0] at `f.py:8:8`,
          'the context is neither a statically-known fixed-point format nor a '
          'constructor call to shift symbolically'), ...]
    """
    if not isinstance(func, Function):
        raise TypeError(f'Expected a \'Function\', got {func}')
    if strategy not in _SITES:
        name = getattr(strategy, '__name__', strategy)
        raise ValueError(f'`{name}` takes no `where`, so it has no sites to refuse')
    lister = _REFUSALS.get(strategy)
    if lister is None:
        return []
    return lister(func.ast, func.rebase(within), **kwargs)
