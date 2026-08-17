"""Unit tests for :func:`fpy2.strategies.unfold_special`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_unfold_special.py``; these tests pin the
wrapper's behavior and its composition with the other scheduling operators.
"""

import pytest

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import Compare, ContextStmt, Copysign, IsInf, IsNan
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import (
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from fpy2.strategies import (
    unfold_special,
    unfold_neg_zero,
    unfold_overflow,
    rescale_fixed,
    simplify,
)

_SAT = fp.OverflowMode.SATURATE
_ZERO = fp.Float(c=0)
_MAX255 = fp.Float(x=RealFloat(exp=0, c=255))


def _block_ctxs(ast) -> list:
    """The value of every statically-known block context in *ast*."""
    eval_info = PartialEval.apply(ast)
    found = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            value = eval_info.by_expr.get(stmt.ctx)
            if value is not None:
                found.append(value)
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return found


def _nodes(ast, node_type) -> list:
    found: list = []

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, node_type):
                found.append(e)
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return found


@fp.fpy(ctx=fp.REAL)
def _quantized_sum(A):
    acc = 0
    for a in A:
        with fp.MPFixedContext(-8, enable_nan=True):
            aq = fp.round(a)
        with fp.FP64:
            acc += aq
    return acc


_SAMPLE = [0.1, 0.25, -3.5, 1e-6, -1e-6, 7.0, 0.0, -2.5]


def _same(a, b) -> bool:
    if a.isnan or b.isnan:
        return a.isnan and b.isnan
    if a.isinf or b.isinf:
        return a.isinf and b.isinf and a.s == b.s
    return a.as_rational() == b.as_rational() and a.s == b.s


class TestUnfoldSpecial:

    def test_returns_a_function(self):
        out = unfold_special(_quantized_sum)
        assert isinstance(out, Function)
        assert out is not _quantized_sum

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            unfold_special(_quantized_sum.ast)  # type: ignore[arg-type]

    def test_removes_the_rules_from_the_context(self):
        assert any(
            isinstance(c, MPFixedContext) and c.enable_nan
            for c in _block_ctxs(_quantized_sum.ast)
        )
        out = unfold_special(_quantized_sum)
        ctxs = _block_ctxs(out.ast)
        assert not any(
            isinstance(c, MPFixedContext) and c.enable_nan for c in ctxs
        )
        # the FP64 accumulation is untouched: its body is arithmetic, not a round
        assert fp.FP64 in ctxs

    def test_preserves_results(self):
        out = unfold_special(_quantized_sum)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))
        nan_sum = [0.1, fp.Float(isnan=True), 0.2]
        assert _same(out(nan_sum), _quantized_sum(nan_sum))

    def test_idempotent(self):
        """The rewritten program rounds under a format with no stated
        special, which has nothing left to take out."""
        once = unfold_special(_quantized_sum)
        twice = unfold_special(once)
        assert twice.ast.is_equiv(once.ast)

    def test_where_selects_one_block(self):
        """The wrapper passes `where` through to the transform."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8, enable_nan=True):
                aq = fp.round(a)
            with fp.MPFixedContext(-4, enable_nan=True):
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        def kept(fn):
            return [
                c.nmin for c in _block_ctxs(fn.ast)
                if isinstance(c, MPFixedContext) and c.enable_nan
            ]

        assert kept(unfold_special(f, 0)) == [-4]
        assert kept(unfold_special(f, 1)) == [-8]
        assert _same(unfold_special(f, 0)(0.1, 0.2), f(0.1, 0.2))

    def test_composes_with_simplify(self):
        out = simplify(
            unfold_special(_quantized_sum), enable_const_fold_context=False,
        )
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))


def _quantizer(ctx) -> fp.Function:
    @fp.fpy(ctx=fp.REAL)
    def q(x):
        with ctx:
            y = fp.round(x)
        return y
    return q


def _samples(ctx) -> list:
    xs = [
        fp.Float(isnan=True), fp.Float(isnan=True, s=True),
        fp.Float(isinf=True), fp.Float(isinf=True, s=True),
        fp.Float(c=0), fp.Float(c=0, s=True),
    ]
    grid = [
        RealFloat(exp=ctx.nmin - 3, c=1),   # rounds to zero
        RealFloat(exp=ctx.nmin, c=1),       # the tie at half a step
        RealFloat(exp=ctx.nmin + 1, c=1),   # the grid's finest step
        RealFloat(exp=0, c=1),
    ]
    if isinstance(ctx, MPBFixedContext):
        B = ctx.pos_maxval
        grid += [B, RealFloat(exp=B.exp + 5, c=B.c)]
    for g in grid:
        xs.append(fp.Float(x=g, ctx=fp.REAL))
        xs.append(fp.Float(x=RealFloat(s=True, exp=g.exp, c=g.c), ctx=fp.REAL))
    return xs


def _assert_agrees(want_fn, got_fn, x) -> None:
    try:
        want = want_fn(x)
    except ValueError:
        with pytest.raises(ValueError):
            got_fn(x)
        return
    assert _same(got_fn(x), want), x


_PIPELINE_CTXS = [
    MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT,
                    nan_value=_ZERO, inf_value=_MAX255),
    MPBFixedContext(-2, RealFloat(exp=0, c=100),
                    neg_maxval=RealFloat(s=True, exp=0, c=50),
                    overflow=_SAT, nan_value=_ZERO),
    fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT, nan_value=_ZERO),
    fp.SMFixedContext(-8, 16, fp.RoundingMode.RTZ, _SAT,
                      nan_value=_ZERO, inf_value=_ZERO),
]
_PIPELINE_IDS = ['mpb_values', 'mpb_asym', 'sm_nan', 'sm_both']


class TestComposition:
    """``unfold_special`` composes with the other rounding operators: each
    peels one family of rules off the same format, leaving a rounding that
    states only its grid."""

    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_rescale_after_unfold(self, ctx):
        """The composed route is what ``rescale_fixed``'s ``fold_specials``
        knob used to do: the specials come out first, then the scale."""
        q = _quantizer(ctx)
        out = rescale_fixed(unfold_special(q))
        for x in _samples(ctx):
            _assert_agrees(q, out, x)

    def test_unblocks_the_rescale(self):
        """A finite substituted constant does not commute with scaling, so
        the plain rescale declines the format — until the rule is out of the
        context."""
        ctx = MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT,
                              nan_value=_ZERO)
        q = _quantizer(ctx)
        assert rescale_fixed(q).ast.is_equiv(q.ast)
        unfolded = unfold_special(q)
        assert not rescale_fixed(unfolded).ast.is_equiv(unfolded.ast)

    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_pipeline_preserves_results(self, ctx):
        """The full fixed-point lowering: specials, then the sign of zero,
        then the bound, then the scale."""
        q = _quantizer(ctx)
        out = rescale_fixed(unfold_overflow(unfold_neg_zero(unfold_special(q))))
        for x in _samples(ctx):
            _assert_agrees(q, out, x)

    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_no_edge_rule_survives(self, ctx):
        """No surviving rounding states a special value, a signed zero, a
        bound, or an overflow rule — each was moved into program text.  What
        is left rounds integers, and its operand is finite and non-zero."""
        q = _quantizer(ctx)
        out = rescale_fixed(unfold_overflow(unfold_neg_zero(unfold_special(q))))

        rounding = [
            c for c in _block_ctxs(out.ast)
            if isinstance(c, MPFixedContext | MPBFixedContext)
        ]
        assert rounding
        for c in rounding:
            assert not isinstance(c, MPBFixedContext)
            assert not c.enable_nan and not c.enable_inf
            assert c.nan_value is None and c.inf_value is None
            assert c.enable_neg_zero is False
            assert c.nmin == -1  # position zero: the values are integers

    def test_branch_structure_for_analysis(self):
        """The value-class shape: the rounding sits behind branches on every
        class ``logb`` is undefined for, so an analysis can call its operand
        finite and non-zero."""
        ctx = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT,
                                nan_value=_ZERO, inf_value=_ZERO)
        out = unfold_special(_quantizer(ctx))
        assert len(_nodes(out.ast, IsNan)) == 1
        assert len(_nodes(out.ast, IsInf)) == 1
        assert len(_nodes(out.ast, Compare)) == 1  # `x == 0`

    def test_pipeline_composes_with_simplify(self):
        ctx = MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT,
                              nan_value=_ZERO, inf_value=_MAX255)
        q = _quantizer(ctx)
        out = simplify(
            rescale_fixed(unfold_overflow(unfold_neg_zero(unfold_special(q)))),
            enable_const_fold_context=False,
        )
        for x in _samples(ctx):
            _assert_agrees(q, out, x)

    def test_one_zero_fixup_across_operators(self):
        """`unfold_special`'s zero branch tests the operand;
        `unfold_neg_zero`'s fixup tests the result.  Composing the two emits
        each once."""
        ctx = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT,
                                nan_value=_ZERO)
        out = unfold_neg_zero(unfold_special(_quantizer(ctx)))
        assert len(_nodes(out.ast, Copysign)) == 1
        # `x == 0` from the special unfold, `t == 0` from the sign unfold
        assert len(_nodes(out.ast, Compare)) == 2
