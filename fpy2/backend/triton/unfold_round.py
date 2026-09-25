"""
triton backend: this backend's answers to :mod:`fpy2.backend.unfold_round`,
plus dropping the fixed-point bounds a kernel could never check.
"""

from fractions import Fraction

from ...analysis import ContextUse, DefineUse, FormatAnalysis, FormatInfer
from ...analysis.format_infer import to_abstract
from ...ast import DefaultTransformVisitor
from ...ast.fpyast import Cast, ContextStmt, Expr, ForeignVal, FuncDef, Round
from ...number import MPBFixedContext, MPFixedContext, OverflowMode
from ...transform import Cursor
from .. import unfold_round as _u
from ..unfold_round import UnfoldKind, UnfoldMode, UnfoldSite, UnfoldTarget
from .emitter import _integral_round
from .target import _DIV_SHAPES, _fp_ctxs, castable, is_native_ctx, make_op_table

__all__ = ['UnfoldKind', 'UnfoldMode', 'UnfoldSite', 'sites', 'unfold', 'unfold_arith']


def _within(e: Expr, bound: Fraction, nmin: int, info: FormatAnalysis) -> bool:
    """Is every finite value of *e* at most *bound* in magnitude, once rounded
    to digit position ``nmin + 1``?"""
    af = to_abstract(info.by_expr[e])
    if af is not None and max(abs(af.pos_bound), abs(af.neg_bound)) <= bound:
        return True
    # the store says ``|e| < 2 ** (msb + 1)``, and rounding stays below the
    # next power of two when that power is on the grid
    db = info.digit_bound
    terms = db.by_expr.get(e) if db is not None else None
    if db is None or terms is None or terms.msb is None:
        return False
    msb = db.store.maximum(terms.msb)
    return isinstance(msb, int) and nmin < msb + 1 and 2 ** (msb + 1) <= bound


class _Unbound(DefaultTransformVisitor):
    def __init__(self, drop: dict[ContextStmt, MPFixedContext]):
        self.drop = drop

    def _visit_context(self, stmt: ContextStmt, ctx):
        s, ctx = super()._visit_context(stmt, ctx)
        if stmt in self.drop:
            s = ContextStmt(s.target, ForeignVal(self.drop[stmt], stmt.loc), s.body, s.loc)
        return s, ctx


def _drop_proven_bounds(func: FuncDef) -> FuncDef:
    """*func* with each asserted fixed-point bound its roundings provably meet
    removed.

    A kernel cannot raise, so the emitter refuses a bounded context; where the
    assertion cannot fire, the unbounded context is the same one.  Infinities
    and NaN round the same way under both.
    """
    ctx_use = ContextUse.analyze(func, def_use=DefineUse.analyze(func))
    todo = [
        s for s in ctx_use.scopes
        if isinstance(s.site, ContextStmt)
        and isinstance(s.ctx, MPBFixedContext)
        and s.ctx.overflow is OverflowMode.ASSERT
    ]
    if not todo:
        return func
    info = FormatInfer.analyze(func, use_digit_bounds=True)
    drop: dict[ContextStmt, MPFixedContext] = {}
    for scope in todo:
        c = scope.ctx
        assert isinstance(c, MPBFixedContext) and isinstance(scope.site, ContextStmt)
        bound = min(c.pos_maxval.as_rational(), -c.neg_maxval.as_rational())
        if all(
            isinstance(u, Round | Cast) and _within(u.arg, bound, c.nmin, info)
            for u in ctx_use.uses[scope]
        ):
            drop[scope.site] = MPFixedContext(
                c.nmin, c.rm, c.num_randbits, rng=c.rng,
                enable_nan=c.enable_nan, enable_inf=c.enable_inf,
                enable_neg_zero=c.enable_neg_zero,
                nan_value=c.nan_value, inf_value=c.inf_value,
            )
    return _Unbound(drop)._visit_function(func, None) if drop else func


_TABLE = make_op_table()

_TARGET = UnfoldTarget(
    native=is_native_ctx,
    rounds=lambda c: is_native_ctx(c) or castable(c) or _integral_round(c) is not None,
    ops=frozenset({*_TABLE.unary, *_TABLE.binary, *_TABLE.ternary}),
    # unlike FP16, every operation in the table has a signature at both
    intermediates=tuple(_fp_ctxs(_DIV_SHAPES)),
    post=_drop_proven_bounds,
)


def sites(func: FuncDef, within: Cursor | None = None) -> list[UnfoldSite]:
    """See :func:`fpy2.backend.unfold_round.sites`."""
    return _u.sites(func, _TARGET, within)


def unfold_arith(func: FuncDef) -> FuncDef:
    """See :func:`fpy2.backend.unfold_round.unfold_arith`."""
    return _u.unfold_arith(func, _TARGET)


def unfold(func: FuncDef, mode: UnfoldMode) -> FuncDef:
    """See :func:`fpy2.backend.unfold_round.unfold`."""
    return _u.unfold(func, mode, _TARGET)
