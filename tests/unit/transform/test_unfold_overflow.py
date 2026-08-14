"""
Unit tests for the :class:`fpy2.transform.UnfoldOverflow` transform.

A bounded rounding is an unbounded rounding plus a comparison, so the tests
assert:

1. **Structural shape**: the bounded context is gone, replaced by an unbounded
   one, and the bound appears as a comparison against a constant.
2. **Negative checks**: contexts and bodies the rewrite must not touch compare
   via ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the interpreter, bit-exactly, across formats,
   rounding modes, and every value class — specials, the bound, the rounding
   boundary just below the overflow threshold, and beyond it.
"""

import fpy2 as fp
import pytest

from fpy2.analysis import PartialEval
from fpy2.ast.fpyast import (
    Call,
    Compare,
    ContextStmt,
    FuncDef,
    IsFinite,
    IsInf,
    IsNan,
    Round,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import (
    REAL,
    EFloatContext,
    EFloatNanKind,
    MPBFixedContext,
    MPBFloatContext,
    MPFixedContext,
    MPSFloatContext,
    RealFloat,
)
from fpy2.transform import UnfoldOverflow


# ----------------------------------------------------------------------
# Helpers


def _blocks(ast: FuncDef) -> list[ContextStmt]:
    """Every ``ContextStmt`` in *ast*, outermost first."""
    found: list[ContextStmt] = []

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            found.append(stmt)
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return found


def _block_ctxs(ast: FuncDef) -> list:
    """The value of every statically-known block context in *ast*."""
    eval_info = PartialEval.apply(ast)
    return [
        v for s in _blocks(ast)
        if (v := eval_info.by_expr.get(s.ctx)) is not None
    ]


def _nodes(ast: FuncDef, node_type) -> list:
    found: list = []

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, node_type):
                found.append(e)
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return found


def _eval(ast: FuncDef, fn: fp.Function, *args):
    return fn.with_ast(ast)(*args)


def _same(a, b) -> bool:
    """Bit-exact comparison that also matches NaN to NaN."""
    if a.isnan or b.isnan:
        return a.isnan and b.isnan
    if a.isinf or b.isinf:
        return a.isinf and b.isinf and a.s == b.s
    return a.as_rational() == b.as_rational() and a.s == b.s


def _quantizer(ctx) -> fp.Function:
    @fp.fpy(ctx=fp.REAL)
    def q(x):
        with ctx:
            y = fp.round(x)
        return y
    return q


def _samples(ctx) -> list:
    """Values covering every class the rewrite has to get right."""
    B = ctx.maxval().as_real()
    xs = [
        fp.Float(isnan=True), fp.Float(isnan=True, s=True),
        fp.Float(isinf=True), fp.Float(isinf=True, s=True),
        fp.Float(c=0), fp.Float(c=0, s=True),
    ]
    grid = [
        B,                                          # the bound itself
        ctx.infval().as_real(),                     # certain overflow
        RealFloat(exp=B.exp - 1, c=2 * B.c - 1),    # just below the bound
        RealFloat(exp=B.exp - 1, c=2 * B.c + 1),    # between bound and infval
        RealFloat(exp=B.exp, c=B.c + 1),            # the tie above the bound
        RealFloat(exp=B.exp + 1, c=B.c),            # well past
        RealFloat(exp=B.exp + 40, c=B.c),
        RealFloat(exp=ctx.expmin, c=1),             # smallest subnormal
        RealFloat(exp=ctx.expmin - 1, c=1),         # below it: rounds to zero
        RealFloat(exp=ctx.emin, c=1),               # smallest normal
        RealFloat(exp=-2, c=1), RealFloat(exp=0, c=1), RealFloat(exp=1, c=3),
    ]
    for g in grid:
        xs.append(fp.Float(x=g, ctx=REAL))
        xs.append(fp.Float(x=RealFloat(s=True, exp=g.exp, c=g.c), ctx=REAL))
    return xs


def _fixed_samples(ctx) -> list:
    """As `_samples`, for a format whose grid is uniform and which commonly
    has no value for NaN or an infinity."""
    B, N = ctx.maxval().as_real(), ctx.maxval(s=True).as_real()
    grid = [
        B, N, ctx.infval().as_real(), ctx.infval(s=True).as_real(),
        RealFloat(exp=B.exp - 1, c=2 * B.c - 1),    # just below the bound
        RealFloat(exp=B.exp - 1, c=2 * B.c + 1),    # just above it
        RealFloat(exp=B.exp + 1, c=B.c),            # well past
        RealFloat(exp=B.exp + 30, c=B.c),
        RealFloat(exp=ctx.nmin, c=1),               # below the grid
        RealFloat(exp=ctx.nmin, c=3),
        RealFloat(exp=ctx.nmin + 1, c=1),           # the grid's finest step
        RealFloat(exp=0, c=1),
    ]
    xs = [fp.Float(c=0), fp.Float(c=0, s=True)]
    for g in grid:
        xs.append(fp.Float(x=g, ctx=REAL))
        xs.append(fp.Float(x=RealFloat(s=True, exp=g.exp, c=g.c), ctx=REAL))
    return xs


# ----------------------------------------------------------------------
# The rewrite


class TestShape:

    def test_context_loses_its_bound(self):
        f = _quantizer(fp.FP16)
        out = UnfoldOverflow.apply(f.ast)

        ctxs = _block_ctxs(out)
        assert REAL in ctxs
        assert not any(isinstance(c, fp.IEEEContext) for c in ctxs)
        target = next(c for c in ctxs if isinstance(c, MPSFloatContext))
        assert (target.pmax, target.emin) == (fp.FP16.pmax, fp.FP16.emin)
        # the rounding itself is untouched; only its format changed
        assert len(_nodes(out, Round)) == 1

    @pytest.mark.parametrize('ctx', [
        fp.FP16, fp.MX_E4M3, fp.MX_E3M2,
        fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
    ], ids=['fp16', 'e4m3', 'e3m2', 'saturating'])
    def test_target_cannot_overflow(self, ctx):
        """The point of the rewrite: what is left rounds under a format with
        no bound at all, so it states no overflow behavior for a backend to
        reproduce."""
        out = UnfoldOverflow.apply(_quantizer(ctx).ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPSFloatContext))
        assert not hasattr(target, 'maxval')
        assert not hasattr(target, 'overflow')
        # its grid runs as far as the value does
        big = RealFloat(exp=ctx.maxval().as_real().exp + 300, c=1)
        assert target.round(big).as_real() == big

    def test_target_is_written_as_a_constructor(self):
        """The emitted context is a call the printed program can be loaded
        back from, rather than an opaque value."""
        out = UnfoldOverflow.apply(_quantizer(fp.FP16).ast)

        call = next(
            s.ctx for s in _blocks(out)
            if isinstance(s.ctx, Call) and s.ctx.fn is MPSFloatContext
        )
        assert [a.val for a in call.args] == [fp.FP16.pmax, fp.FP16.emin]  # type: ignore[attr-defined]
        assert 'fp.MPSFloatContext(11, -14)' in out.format()

    def test_bound_becomes_a_comparison(self):
        """One test per side, so a format need not mirror its bound."""
        f = _quantizer(fp.FP16)
        out = UnfoldOverflow.apply(f.ast)

        bounds = {int(c.args[1].val) for c in _nodes(out, Compare)}  # type: ignore[attr-defined]
        assert bounds == {65504, -65504}

    def test_no_early_check_by_default(self):
        f = _quantizer(fp.FP16)
        out = UnfoldOverflow.apply(f.ast)
        # only the two post-checks, against `maxval` rather than `infval`
        assert len(_nodes(out, Compare)) == 2

    def test_early_check_tests_the_operand(self):
        f = _quantizer(fp.FP16)
        out = UnfoldOverflow.apply(f.ast, early_check=True)

        bounds = {int(c.args[1].val) for c in _nodes(out, Compare)}  # type: ignore[attr-defined]
        # the guard uses `infval`, one grid point above the bound
        assert bounds == {65504, -65504, 65536, -65536}

    def test_early_check_is_not_complete(self):
        """FP16's tie at 65520 lies below `infval` yet rounds up and
        overflows, so the check after the rounding has to stay."""
        f = _quantizer(fp.FP16)
        assert float(fp.FP16.infval()) == 65536.0
        for out in (
            UnfoldOverflow.apply(f.ast),
            UnfoldOverflow.apply(f.ast, early_check=True),
        ):
            assert _eval(out, f, 65520.0).isinf
            assert _same(_eval(out, f, 65510.0), f(65510.0))  # rounds back to the bound

    @pytest.mark.parametrize('ctx', [
        fp.FP16, fp.MX_E4M3,
        MPBFixedContext(-4, RealFloat(exp=0, c=255),
                        overflow=fp.OverflowMode.SATURATE, enable_nan=True),
    ], ids=['fp16', 'e4m3', 'fixed_with_nan'])
    def test_nan_sign(self, ctx):
        """A float format canonicalizes NaN to positive while a fixed-point
        one keeps the sign it was given.  Each counterpart does the same as its
        source, so the comparison that decides whether NaN needs a branch can
        hold the sign against it."""
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast)

        neg_nan = fp.Float(isnan=True, s=True)
        assert _eval(out, f, neg_nan).s == f(neg_nan).s

    def test_no_special_branches_for_ieee(self):
        """An IEEE format's NaN and infinities survive the rewrite untouched,
        so nothing has to be said about them."""
        out = UnfoldOverflow.apply(_quantizer(fp.FP16).ast)
        assert not _nodes(out, IsNan)
        assert not _nodes(out, IsInf)

    def test_special_branch_only_where_needed(self):
        """`E3M2` substitutes its bound for NaN, which the comparison cannot
        produce; its infinity needs no branch, since overflow lands there
        anyway."""
        out = UnfoldOverflow.apply(_quantizer(fp.MX_E3M2).ast)
        assert len(_nodes(out, IsNan)) == 1
        assert not _nodes(out, IsInf)

    def test_saturating_format_keeps_its_infinity(self):
        """Overflow saturates to the bound, but an infinite *operand* stays
        infinite, so the two cannot share a branch."""
        ctx = fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE)
        out = UnfoldOverflow.apply(_quantizer(ctx).ast)
        assert len(_nodes(out, IsInf)) == 1
        assert not _nodes(out, IsNan)

    def test_format_without_negative_zero(self):
        """`NEG_ZERO` spends that encoding on NaN, which the unbounded
        counterpart cannot express, so the sign is dropped explicitly."""
        ctx = EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0)
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast)
        assert not _eval(out, f, -1e-9).s
        assert _same(_eval(out, f, -1e-9), f(-1e-9))

    def test_returned_round(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                return fp.round(x)

        out = UnfoldOverflow.apply(f.ast)
        assert not any(isinstance(c, fp.IEEEContext) for c in _block_ctxs(out))
        for x in _samples(fp.FP16):
            assert _same(_eval(out, f, x), f(x)), x

    def test_several_rounds_in_one_block(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                aq = fp.round(a)
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        out = UnfoldOverflow.apply(f.ast)
        assert not any(c is fp.FP16 for c in _block_ctxs(out))
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))
        assert _same(_eval(out, f, 1e5, 1.0), f(1e5, 1.0))

    def test_inside_a_loop(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A):
            acc = 0
            for a in A:
                with fp.FP16:
                    aq = fp.round(a)
                with fp.FP64:
                    acc += aq
            return acc

        out = UnfoldOverflow.apply(f.ast)
        A = [0.1, 0.25, -3.5, 1e-6, 7.0, 70000.0]
        assert _same(_eval(out, f, A), f(A))


# ----------------------------------------------------------------------
# Fixed-point sources


_SAT = fp.OverflowMode.SATURATE


class TestFixedPoint:
    """A bounded fixed-point format unfolds the same way; its counterpart
    is ``MPFixedContext``, which states a digit position and nothing else."""

    def test_position_is_kept(self):
        src = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT)
        out = UnfoldOverflow.apply(_quantizer(src).ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPFixedContext))
        assert target.nmin == src.nmin
        assert not hasattr(target, 'maxval')
        assert not hasattr(target, 'overflow')

    def test_asymmetric_bound(self):
        """Two's complement reaches one further below zero than above it.  The
        two comparisons are already separate, so nothing has to mirror."""
        src = fp.FixedContext(True, -16, 32, fp.RoundingMode.RNE, _SAT)
        f = _quantizer(src)
        out = UnfoldOverflow.apply(f.ast)

        assert src.maxval().as_real() != -src.maxval(s=True).as_real()
        bounds = {c.args[1] for c in _nodes(out, Compare)}
        assert len(bounds) == 2
        for x in _fixed_samples(src):
            assert _same(_eval(out, f, x), f(x)), x

    @pytest.mark.parametrize('src, neg_zero', [
        (fp.FixedContext(True, -8, 16, fp.RoundingMode.RNE, _SAT), False),
        (fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT), True),
    ], ids=['twos_complement', 'sign_magnitude'])
    def test_negative_zero_follows_the_source(self, src, neg_zero):
        """``MPFixedContext`` can state whether it keeps a negative zero, so
        unlike the float case no fixup after the rounding is needed."""
        out = UnfoldOverflow.apply(_quantizer(src).ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPFixedContext))
        assert target.enable_neg_zero is neg_zero
        assert not _nodes(out, IsNan)  # nor any zero fixup branch
        assert len(_nodes(out, Compare)) == 2

    def test_rejection_is_preserved(self):
        """A fixed-point format has no value for NaN or an infinity, and the
        rewrite has no way to state a refusal — so the counterpart must refuse
        them too, rather than the checks answering for it."""
        src = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT)
        f = _quantizer(src)
        out = UnfoldOverflow.apply(f.ast)

        for x in (fp.Float(isnan=True), fp.Float(isinf=True), fp.Float(isinf=True, s=True)):
            with pytest.raises(ValueError):
                f(x)
            with pytest.raises(ValueError):
                _eval(out, f, x)

    def test_early_check_excludes_infinities(self):
        """An infinity is past every bound, so the guard would claim it as an
        overflow; testing finiteness first lets it reach the rounding, which
        refuses it as the source did."""
        src = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT)
        f = _quantizer(src)
        out = UnfoldOverflow.apply(f.ast, early_check=True)

        assert len(_nodes(out, IsFinite)) == 2
        with pytest.raises(ValueError):
            _eval(out, f, fp.Float(isinf=True))

    def test_float_early_check_needs_no_finiteness_test(self):
        """A float counterpart represents the infinities, so its guard can
        claim one without changing what the format makes of it."""
        out = UnfoldOverflow.apply(_quantizer(fp.FP16).ast, early_check=True)
        assert not _nodes(out, IsFinite)

    @pytest.mark.parametrize('early_check', [False, True], ids=['plain', 'early_check'])
    @pytest.mark.parametrize('src', [
        fp.FixedContext(True, -16, 32, fp.RoundingMode.RNE, _SAT),
        fp.FixedContext(True, -4, 8, fp.RoundingMode.RNE, _SAT),
        fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT),
        fp.SMFixedContext(-8, 16, fp.RoundingMode.RTZ, _SAT),
        MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT),
        MPBFixedContext(-4, RealFloat(exp=0, c=255),
                        overflow=fp.OverflowMode.OVERFLOW, enable_inf=True),
        MPBFixedContext(-2, RealFloat(exp=0, c=100),
                        neg_maxval=RealFloat(s=True, exp=0, c=50), overflow=_SAT),
    ], ids=['fixed_32', 'fixed_8', 'sm_16', 'sm_rtz', 'mpb_sat', 'mpb_inf', 'mpb_asym'])
    def test_equivalence(self, src, early_check):
        f = _quantizer(src)
        out = UnfoldOverflow.apply(f.ast, early_check=early_check)
        assert not out.is_equiv(f.ast)
        for x in _fixed_samples(src):
            assert _same(_eval(out, f, x), f(x)), (src, x)


class TestFixedPointUnchanged:

    def test_wrapping_overflow(self):
        """Wrapping gives a different answer at every magnitude, so no
        constant states it."""
        f = _quantizer(fp.FixedContext(True, -16, 32))  # WRAP is the default
        assert fp.FixedContext(True, -16, 32).overflow is fp.OverflowMode.WRAP
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_unsigned_format(self):
        """An unsigned format states no bound below zero."""
        f = _quantizer(fp.FixedContext(False, -16, 32, fp.RoundingMode.RNE, _SAT))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_overflow_to_an_absent_infinity(self):
        """The format overflows to infinity but cannot represent one, so the
        rounding raises and there is no value to write out."""
        f = _quantizer(fp.FixedContext(
            True, -4, 8, fp.RoundingMode.RNE, fp.OverflowMode.OVERFLOW,
        ))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_already_unbounded(self):
        f = _quantizer(MPFixedContext(-8))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Selecting a single site


class TestWhere:
    """``where`` picks one candidate block by index, in visit order."""

    @staticmethod
    def _two() -> fp.Function:
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                aq = fp.round(a)
            with fp.FP32:
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s
        return f

    @pytest.mark.parametrize('where, left', [
        (0, [fp.FP32]),
        (1, [fp.FP16]),
        (None, []),
    ])
    def test_selects_one_block(self, where, left):
        f = self._two()
        out = UnfoldOverflow.apply(f.ast, where=where)
        # the FP64 block is arithmetic, so it is never a candidate
        remaining = [c for c in _block_ctxs(out) if c in (fp.FP16, fp.FP32)]
        assert remaining == left
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

    def test_index_past_the_last_site(self):
        f = self._two()
        assert UnfoldOverflow.apply(f.ast, where=9).is_equiv(f.ast)

    def test_rejects_a_non_integer(self):
        f = self._two()
        with pytest.raises(TypeError):
            UnfoldOverflow.apply(f.ast, where='first')  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Blocks the rewrite must leave alone


class TestUnchanged:

    def test_unbounded_context(self):
        """There is no bound to take out."""
        f = _quantizer(MPSFloatContext(11, -14))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_fixed_point_context(self):
        f = _quantizer(fp.FixedContext(True, -16, 32))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_overflow_that_raises(self):
        """A format that refuses to round an overflow at all states no value
        for the rewrite to write out."""
        f = _quantizer(fp.IEEEContext(
            5, 16, fp.RoundingMode.RNE, fp.OverflowMode.ASSERT,
        ))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_stochastic_context(self):
        """Stochastic rounding would have to draw its bits under the same
        format, which the unbounded counterpart is not."""
        f = _quantizer(fp.IEEEContext(5, 16, fp.RoundingMode.RNE, num_randbits=2))
        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_cast_body(self):
        """``cast`` asserts exactness, which the rewrite would not preserve."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                y = fp.cast(x)
            return y

        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_arithmetic_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                p = a * b
            return p

        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_round_of_an_expression(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                y = fp.round(a + b)
            return y

        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)

    def test_bound_context(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16 as c:
                y = fp.round(x)
            return y

        assert UnfoldOverflow.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:
    """The rewrite must be bit-exact: it is the same rounding, with the bound
    stated rather than built in."""

    @pytest.mark.parametrize('early_check', [False, True], ids=['plain', 'early_check'])
    @pytest.mark.parametrize('ctx', [
        fp.FP16, fp.FP32, fp.FP64, fp.IEEEContext(4, 8), fp.IEEEContext(8, 32),
    ], ids=['fp16', 'fp32', 'fp64', 'ieee_4_8', 'ieee_8_32'])
    def test_formats(self, ctx, early_check):
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast, early_check=early_check)
        assert not out.is_equiv(f.ast)
        for x in _samples(ctx):
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    @pytest.mark.parametrize('early_check', [False, True], ids=['plain', 'early_check'])
    @pytest.mark.parametrize('ctx', [
        fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
        EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0),
        EFloatContext(4, 8, False, EFloatNanKind.NONE, 2),
    ], ids=['ieee_saturating', 'neg_zero_nan', 'shifted_exponent'])
    def test_edge_rules(self, ctx, early_check):
        """A format states its edge rule in its own terms; the rewrite asks
        rather than assumes.  A shifted exponent encoding is already accounted
        for by the precision and exponent range read off the format."""
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast, early_check=early_check)
        assert not out.is_equiv(f.ast)
        for x in _samples(ctx) + [1e-9, -1e-9, 0.1, -0.1]:
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    @pytest.mark.parametrize('early_check', [False, True], ids=['plain', 'early_check'])
    @pytest.mark.parametrize('ctx', [
        fp.MX_E5M2, fp.MX_E4M3, fp.MX_E3M2, fp.MX_E2M3, fp.MX_E2M1,
    ], ids=['e5m2', 'e4m3', 'e3m2', 'e2m3', 'e2m1'])
    def test_mx_formats(self, ctx, early_check):
        """The MX formats differ in what overflow becomes: an infinity for
        `E5M2`, a NaN for `E4M3`, the bound for the rest."""
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast, early_check=early_check)
        assert not out.is_equiv(f.ast)
        for x in _samples(ctx) + [0.1, -0.1, 12.5, -12.5]:
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    def test_multiprecision_bounded_format(self):
        ctx = MPBFloatContext(11, -14, fp.FP16.maxval().as_real())
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast)
        assert not out.is_equiv(f.ast)
        for x in _samples(ctx):
            assert _same(_eval(out, f, x), f(x)), x

    @pytest.mark.parametrize('rm', [
        fp.RoundingMode.RNE, fp.RoundingMode.RNA, fp.RoundingMode.RTZ,
        fp.RoundingMode.RTP, fp.RoundingMode.RTN, fp.RoundingMode.RAZ,
    ])
    def test_rounding_modes(self, rm):
        """The overflow *value* follows the mode as much as the bound does:
        under `RTZ` a value past the bound saturates rather than becoming an
        infinity."""
        ctx = fp.IEEEContext(5, 16, rm)
        f = _quantizer(ctx)
        out = UnfoldOverflow.apply(f.ast, early_check=True)
        xs = _samples(ctx) + [0.1, -0.1, 65519.0, 65519.996, -65519.996, 1.0009765625]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), (rm, x)

    def test_overflow_boundary(self):
        """Rounding at the top of the range decides between the bound and an
        infinity; the comparison must make the same call."""
        f = _quantizer(fp.FP16)
        out = UnfoldOverflow.apply(f.ast, early_check=True)
        for x in (65504.0, 65515.0, 65519.0, 65519.996, 65520.0, 65536.0, 1e5, 1e10):
            for sx in (x, -x):
                assert _same(_eval(out, f, sx), f(sx)), sx
