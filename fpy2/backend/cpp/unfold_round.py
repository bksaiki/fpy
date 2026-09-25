"""
cpp backend: this backend's answers to :mod:`fpy2.backend.unfold_round`.

*enable_fenv* false shrinks what counts as native, so a mode the emitter may no
longer set is a site here rather than a refusal there.
"""

from functools import cache

from ...ast.fpyast import FuncDef
from ...number import RM, IEEEContext, MPBFixedContext, MPFixedContext, OverflowMode
from ...number.context.context import Context
from ...transform import Cursor
from .. import unfold_round as _u
from ..unfold_round import UnfoldKind, UnfoldMode, UnfoldSite, UnfoldTarget
from .target import is_native_ctx, make_op_table

__all__ = ['UnfoldKind', 'UnfoldMode', 'UnfoldSite', 'sites', 'unfold', 'unfold_arith']


def _fixed_is_lowerable(ctx: Context) -> bool:
    """Whether `_emit_integral_round` lowers *ctx* as it stands.

    Its digits at position zero (``nmin == -1`` is the last unrepresentable
    one), no random bits, and either unbounded or asserting its bound.
    """
    if not isinstance(ctx, MPFixedContext | MPBFixedContext):
        return False
    if ctx.nmin != -1 or ctx.num_randbits != 0:
        return False
    return (
        not isinstance(ctx, MPBFixedContext)
        or ctx.overflow is OverflowMode.ASSERT
    )


@cache
def _target(enable_fenv: bool) -> UnfoldTarget:
    def native(ctx: Context) -> bool:
        return is_native_ctx(ctx, enable_fenv=enable_fenv)

    table = make_op_table(enable_fenv=enable_fenv)
    return UnfoldTarget(
        native=native,
        rounds=lambda c: native(c) or _fixed_is_lowerable(c),
        ops=frozenset({*table.unary, *table.binary, *table.ternary}),
        # round-to-nearest only: the per-operation rules take it and the
        # exactness rule takes any mode, and it needs no `fesetround` boundary
        intermediates=tuple(
            c for es, nbits in ((8, 32), (11, 64))
            if native(c := IEEEContext(es, nbits, RM.RNE))
        ),
    )


def sites(
    func: FuncDef, within: Cursor | None = None, *, enable_fenv: bool = True,
) -> list[UnfoldSite]:
    """See :func:`fpy2.backend.unfold_round.sites`."""
    return _u.sites(func, _target(enable_fenv), within)


def unfold_arith(func: FuncDef, *, enable_fenv: bool = True) -> FuncDef:
    """See :func:`fpy2.backend.unfold_round.unfold_arith`."""
    return _u.unfold_arith(func, _target(enable_fenv))


def unfold(
    func: FuncDef, mode: UnfoldMode, *, enable_fenv: bool = True,
) -> FuncDef:
    """See :func:`fpy2.backend.unfold_round.unfold`."""
    return _u.unfold(func, mode, _target(enable_fenv))
