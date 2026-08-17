"""Unit tests for :func:`fpy2.strategies.unfold_neg_zero`.

The transform itself is tested exhaustively in
``tests/unit/transform/test_unfold_neg_zero.py``; these tests pin the
wrapper's behavior and its composition with the other scheduling operators.
"""

import pytest

import fpy2 as fp

from fpy2.analysis import PartialEval
from fpy2.ast import Compare, ContextStmt, Copysign
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import (
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from fpy2.strategies import (
    unfold_neg_zero,
    unfold_overflow,
    rescale_fixed,
    simplify,
)

_SAT = fp.OverflowMode.SATURATE


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
        with fp.MPFixedContext(-8):
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


class TestUnfoldNegZero:

    def test_returns_a_function(self):
        out = unfold_neg_zero(_quantized_sum)
        assert isinstance(out, Function)
        assert out is not _quantized_sum

    def test_rejects_non_function(self):
        with pytest.raises(TypeError):
            unfold_neg_zero(_quantized_sum.ast)  # type: ignore[arg-type]

    def test_removes_the_flag_from_the_context(self):
        assert any(
            isinstance(c, MPFixedContext) and c.enable_neg_zero
            for c in _block_ctxs(_quantized_sum.ast)
        )
        out = unfold_neg_zero(_quantized_sum)
        ctxs = _block_ctxs(out.ast)
        assert not any(
            isinstance(c, MPFixedContext) and c.enable_neg_zero for c in ctxs
        )
        # the FP64 accumulation is untouched: its body is arithmetic, not a round
        assert fp.FP64 in ctxs

    def test_preserves_results(self):
        out = unfold_neg_zero(_quantized_sum)
        assert _same(out(_SAMPLE), _quantized_sum(_SAMPLE))

    def test_preserves_the_sign_of_zero(self):
        @fp.fpy(ctx=fp.REAL)
        def q(x):
            with fp.MPFixedContext(-8):
                y = fp.round(x)
            return y

        out = unfold_neg_zero(q)
        want = q(-1e-9)
        got = out(-1e-9)
        assert want.is_zero() and want.s
        assert _same(got, want)

    def test_idempotent(self):
        """The rewritten program rounds under a format with no signed zero,
        which has nothing left to take out."""
        once = unfold_neg_zero(_quantized_sum)
        twice = unfold_neg_zero(once)
        assert twice.ast.is_equiv(once.ast)

    def test_where_selects_one_block(self):
        """The wrapper passes `where` through to the transform."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8):
                aq = fp.round(a)
            with fp.MPFixedContext(-4):
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        def kept(fn):
            return [
                c.nmin for c in _block_ctxs(fn.ast)
                if isinstance(c, MPFixedContext) and c.enable_neg_zero
            ]

        assert kept(unfold_neg_zero(f, 0)) == [-4]
        assert kept(unfold_neg_zero(f, 1)) == [-8]
        assert _same(unfold_neg_zero(f, 0)(0.1, 0.2), f(0.1, 0.2))

    def test_composes_with_simplify(self):
        out = simplify(
            unfold_neg_zero(_quantized_sum), enable_const_fold_context=False,
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
    xs = [fp.Float(c=0), fp.Float(c=0, s=True)]
    grid = [
        RealFloat(exp=ctx.nmin - 3, c=1),   # far below the grid: rounds to zero
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


_PIPELINE_CTXS = [
    fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT),
    fp.SMFixedContext(-8, 16, fp.RoundingMode.RTZ, _SAT),
    fp.SMFixedContext(-4, 8, fp.RoundingMode.RTP, _SAT),
    MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT),
    MPBFixedContext(-2, RealFloat(exp=0, c=100),
                    neg_maxval=RealFloat(s=True, exp=0, c=50), overflow=_SAT),
]
_PIPELINE_IDS = ['sm_16', 'sm_rtz', 'sm_rtp', 'mpb_sat', 'mpb_asym']


class TestComposition:
    """``unfold_neg_zero`` then ``unfold_overflow`` peels the sign rule and
    the bound off the same format, leaving a rounding that states a grid and
    nothing else."""

    def test_one_zero_branch(self):
        """The rewritten context has no signed zero, so `unfold_overflow`'s
        own zero fixup (`drop_neg_zero`) stays quiet: one `copysign`, one
        ``== 0`` test, two bound comparisons — not two zero branches."""
        q = _quantizer(fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT))
        out = unfold_overflow(unfold_neg_zero(q))

        assert len(_nodes(out.ast, Copysign)) == 1
        assert len(_nodes(out.ast, Compare)) == 3

    def test_either_order(self):
        """After `unfold_overflow`, the counterpart still carries the flag,
        so the two operators compose in both orders."""
        ctx = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT)
        q = _quantizer(ctx)
        a = unfold_overflow(unfold_neg_zero(q))
        b = unfold_neg_zero(unfold_overflow(q))

        for out in (a, b):
            assert len(_nodes(out.ast, Copysign)) == 1
            target = next(
                c for c in _block_ctxs(out.ast) if isinstance(c, MPFixedContext)
            )
            assert target.enable_neg_zero is False
        for x in _samples(ctx):
            assert _same(a(x), q(x)), x
            assert _same(b(x), q(x)), x

    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_pipeline_preserves_results(self, ctx):
        """The full fixed-point lowering: sign rule, then bound, then scale.
        What is left rounds at position zero with every edge rule in program
        text."""
        q = _quantizer(ctx)
        out = rescale_fixed(unfold_overflow(unfold_neg_zero(q)))
        for x in _samples(ctx):
            assert _same(out(x), q(x)), (ctx, x)

    @pytest.mark.parametrize('ctx', _PIPELINE_CTXS, ids=_PIPELINE_IDS)
    def test_no_edge_rule_survives(self, ctx):
        """No surviving rounding keeps a signed zero, a bound, or an overflow
        rule — each was moved into program text."""
        q = _quantizer(ctx)
        out = rescale_fixed(unfold_overflow(unfold_neg_zero(q)))

        rounding = [
            c for c in _block_ctxs(out.ast)
            if isinstance(c, MPFixedContext | MPBFixedContext)
        ]
        assert rounding
        for c in rounding:
            assert c.enable_neg_zero is False
            assert not isinstance(c, MPBFixedContext)
            assert c.nmin == -1  # position zero: the values are integers

    def test_pipeline_composes_with_simplify(self):
        ctx = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT)
        q = _quantizer(ctx)
        out = simplify(
            rescale_fixed(unfold_overflow(unfold_neg_zero(q))),
            enable_const_fold_context=False,
        )
        for x in _samples(ctx):
            assert _same(out(x), q(x)), x
