"""
Unit tests for the :class:`fpy2.transform.UnfoldNegZero` transform.

A rounding that keeps the sign of zero is a rounding that does not, plus a
sign restoration from the operand, so the tests assert:

1. **Structural shape**: the source context loses its ``enable_neg_zero``
   flag, and the sign comes back as a ``copysign`` behind a ``== 0`` test.
2. **Negative checks**: contexts and bodies the rewrite must not touch —
   including formats the fixup cannot reproduce, like wrapping overflow —
   compare via ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the interpreter, bit-exactly (the sign of
   zero included), across formats, rounding modes, and every value class.
"""

import fpy2 as fp
import pytest

from fpy2.analysis import PartialEval
from fpy2.ast.fpyast import (
    Call,
    Compare,
    ContextStmt,
    Copysign,
    Expr,
    ForeignVal,
    FuncDef,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import (
    REAL,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from fpy2.transform import TransformDeclined, TransformReferenceError, UnfoldNegZero


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
        return a.isnan and b.isnan and a.s == b.s
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
    """Both zeros, values rounding to zero from either side, and enough of
    the grid to notice a rounding gone wrong elsewhere."""
    grid = [
        RealFloat(exp=ctx.nmin - 3, c=1),   # far below the grid: rounds to zero
        RealFloat(exp=ctx.nmin, c=1),       # the tie at half a step
        RealFloat(exp=ctx.nmin, c=3),
        RealFloat(exp=ctx.nmin + 1, c=1),   # the grid's finest step
        RealFloat(exp=ctx.nmin + 1, c=7),
        RealFloat(exp=0, c=1),
    ]
    if isinstance(ctx, MPBFixedContext):
        B = ctx.pos_maxval
        grid += [
            B,
            RealFloat(exp=B.exp + 1, c=B.c),    # past the bound
            RealFloat(exp=B.exp + 30, c=B.c),   # far past
        ]
    xs = [fp.Float(c=0), fp.Float(c=0, s=True)]
    for g in grid:
        xs.append(fp.Float(x=g, ctx=REAL))
        xs.append(fp.Float(x=RealFloat(s=True, exp=g.exp, c=g.c), ctx=REAL))
    return xs


_SAT = fp.OverflowMode.SATURATE


# ----------------------------------------------------------------------
# The rewrite


class TestShape:

    def test_context_loses_its_signed_zero(self):
        f = _quantizer(MPFixedContext(-8))
        out = UnfoldNegZero.apply(f.ast)

        ctxs = _block_ctxs(out)
        assert REAL in ctxs
        target = next(c for c in ctxs if isinstance(c, MPFixedContext))
        assert target.nmin == -8
        assert target.enable_neg_zero is False

    def test_sign_restored_by_copysign(self):
        out = UnfoldNegZero.apply(_quantizer(MPFixedContext(-8)).ast)
        assert len(_nodes(out, Copysign)) == 1
        # a single `t == 0` test; the fixup is dead for every other value
        compares = _nodes(out, Compare)
        assert len(compares) == 1
        assert int(compares[0].args[1].val) == 0  # type: ignore[attr-defined]

    def test_written_call_keeps_its_form(self):
        """A constructor call keeps its written form with only the flag
        stated, so the rewritten program reads like the original."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8):
                y = fp.round(x)
            return y

        out = UnfoldNegZero.apply(f.ast)
        call = next(
            s.ctx for s in _blocks(out)
            if isinstance(s.ctx, Call) and s.ctx.fn is MPFixedContext
        )
        assert dict(call.kwargs).keys() == {'enable_neg_zero'}
        assert 'fp.MPFixedContext(-8, enable_neg_zero=False)' in out.format()

    def test_sm_fixed_rebuilt_as_its_base(self):
        """`SMFixedContext` has its signed zero by construction, so the flag
        comes off in the `MPBFixedContext` it derives from."""
        src = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT)
        f = _quantizer(src)
        out = UnfoldNegZero.apply(f.ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert type(target) is MPBFixedContext
        assert target.enable_neg_zero is False
        assert target.pos_maxval == src.pos_maxval
        assert target.neg_maxval == src.neg_maxval
        assert target.nmin == src.nmin
        # no constructor states the rebuilt class, so it is the value itself
        assert any(
            isinstance(s.ctx, ForeignVal) and s.ctx.val == target
            for s in _blocks(out)
        )

    def test_rejection_is_preserved(self):
        """The format has no value for NaN or an infinity, and the rewrite
        has no way to state a refusal — so the rebuilt context must refuse
        them too."""
        src = MPFixedContext(-8)
        f = _quantizer(src)
        out = UnfoldNegZero.apply(f.ast)

        for x in (fp.Float(isnan=True), fp.Float(isinf=True), fp.Float(isinf=True, s=True)):
            with pytest.raises(ValueError):
                f(x)
            with pytest.raises(ValueError):
                _eval(out, f, x)

    def test_returned_round(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8):
                return fp.round(x)

        out = UnfoldNegZero.apply(f.ast)
        assert not any(
            isinstance(c, MPFixedContext) and c.enable_neg_zero for c in _block_ctxs(out)
        )
        src = MPFixedContext(-8)
        for x in _samples(src):
            assert _same(_eval(out, f, x), f(x)), x

    def test_several_rounds_in_one_block(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8):
                aq = fp.round(a)
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        out = UnfoldNegZero.apply(f.ast)
        assert len(_nodes(out, Copysign)) == 2
        # each emitted block occupies distinct AST nodes, the shared written
        # context included
        exprs = _nodes(out, Expr)
        assert len(exprs) == len(set(map(id, exprs)))
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))
        assert _same(_eval(out, f, -1e-9, 1e-9), f(-1e-9, 1e-9))

    def test_idempotent(self):
        """What is left keeps no signed zero, so there is nothing to unfold
        twice."""
        f = _quantizer(MPFixedContext(-8))
        once = UnfoldNegZero.apply(f.ast)
        assert UnfoldNegZero.apply(once).is_equiv(once)


# ----------------------------------------------------------------------
# Blocks the rewrite must leave alone


class TestUnchanged:

    def test_twos_complement(self):
        """`FixedContext` already has a single zero."""
        f = _quantizer(fp.FixedContext(True, -8, 16, fp.RoundingMode.RNE, _SAT))
        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_flag_already_off(self):
        f = _quantizer(MPFixedContext(-8, enable_neg_zero=False))
        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_float_context(self):
        """A float format has no flag to take out."""
        f = _quantizer(fp.FP16)
        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    @pytest.mark.parametrize('src', [
        MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=fp.OverflowMode.WRAP),
        fp.SMFixedContext(-8, 16),  # WRAP is the default
    ], ids=['mpb_wrap', 'sm_wrap'])
    def test_wrapping_overflow(self, src):
        """Wrapping goes around the full signed range by ordinal, so a
        negative operand can land on ``+0`` — which no sign restoration from
        the operand reproduces."""
        f = _quantizer(src)
        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_special_value_that_is_a_zero(self):
        """``nan_value=+0`` hands a NaN operand's sign to the fixup, which
        the source would not keep."""
        src = MPFixedContext(-8, nan_value=fp.Float(c=0))
        assert not src.round(fp.Float(isnan=True, s=True)).s
        f = _quantizer(src)
        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_stochastic_context(self):
        """Stochastic rounding would have to draw its bits under the same
        format, which the rebuilt context is not."""
        f = _quantizer(MPFixedContext(-8, fp.RoundingMode.RNE, 2))
        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_cast_body(self):
        """``cast`` asserts exactness, and an exact result never rounds to
        zero from anything but zero."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8):
                y = fp.cast(x)
            return y

        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_arithmetic_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8):
                p = a * b
            return p

        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_round_of_an_expression(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8):
                y = fp.round(a + b)
            return y

        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)

    def test_bound_context(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8) as c:
                y = fp.round(x)
            return y

        assert UnfoldNegZero.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Selecting a single site


class TestWhere:
    """``where`` picks one candidate block by index, in visit order."""

    @staticmethod
    def _two() -> fp.Function:
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8):
                aq = fp.round(a)
            with fp.MPFixedContext(-4):
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s
        return f

    @pytest.mark.parametrize('where, kept', [
        (0, [-4]),
        (1, [-8]),
        (None, []),
    ])
    def test_selects_one_block(self, where, kept):
        f = self._two()
        out = UnfoldNegZero.apply(f.ast, where=where)
        # the FP64 block is arithmetic, so it is never a candidate
        remaining = [
            c.nmin for c in _block_ctxs(out)
            if isinstance(c, MPFixedContext) and c.enable_neg_zero
        ]
        assert remaining == kept
        assert _same(_eval(out, f, -1e-9, -1e-9), f(-1e-9, -1e-9))

    def test_index_past_the_last_site(self):
        f = self._two()
        with pytest.raises(TransformReferenceError):
            UnfoldNegZero.apply(f.ast, where=9)

    def test_naming_a_declined_block_raises(self):
        """A float format is structurally a candidate; naming it says why it
        cannot be rewritten."""
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                y = fp.round(x)
            return y

        with pytest.raises(TransformDeclined, match='fixed-point'):
            UnfoldNegZero.apply(f.ast, where=0)

    def test_rejects_a_non_integer(self):
        f = self._two()
        with pytest.raises(TypeError):
            UnfoldNegZero.apply(f.ast, where='first')  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:
    """The rewrite must be bit-exact: it is the same rounding, with the sign
    of zero stated rather than built in."""

    @pytest.mark.parametrize('rm', [
        fp.RoundingMode.RNE, fp.RoundingMode.RNA, fp.RoundingMode.RTZ,
        fp.RoundingMode.RTP, fp.RoundingMode.RTN, fp.RoundingMode.RAZ,
    ])
    def test_rounding_modes(self, rm):
        """A directed mode decides which side of zero a tiny value lands on,
        and the sign of that zero is the whole point."""
        src = MPFixedContext(-8, rm)
        f = _quantizer(src)
        out = UnfoldNegZero.apply(f.ast)
        assert not out.is_equiv(f.ast)
        for x in _samples(src):
            assert _same(_eval(out, f, x), f(x)), (rm, x)

    @pytest.mark.parametrize('src', [
        MPFixedContext(-8),
        MPFixedContext(0),
        MPFixedContext(4, fp.RoundingMode.RTZ),
        MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT),
        MPBFixedContext(-2, RealFloat(exp=0, c=100),
                        neg_maxval=RealFloat(s=True, exp=0, c=50), overflow=_SAT),
        MPBFixedContext(-4, RealFloat(exp=0, c=255),
                        overflow=fp.OverflowMode.OVERFLOW, enable_inf=True),
        fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT),
        fp.SMFixedContext(-8, 16, fp.RoundingMode.RTZ, _SAT),
    ], ids=[
        'mp_8', 'mp_0', 'mp_coarse', 'mpb_sat', 'mpb_asym', 'mpb_inf',
        'sm_16', 'sm_rtz',
    ])
    def test_formats(self, src):
        f = _quantizer(src)
        out = UnfoldNegZero.apply(f.ast)
        assert not out.is_equiv(f.ast)
        for x in _samples(src):
            assert _same(_eval(out, f, x), f(x)), (src, x)

    def test_specials_pass_through(self):
        """A representable NaN or infinity survives the rebuilt rounding
        untouched, so nothing has to be said about it."""
        src = MPFixedContext(-8, enable_nan=True, enable_inf=True)
        f = _quantizer(src)
        out = UnfoldNegZero.apply(f.ast)
        assert not out.is_equiv(f.ast)
        xs = _samples(src) + [
            fp.Float(isnan=True), fp.Float(isnan=True, s=True),
            fp.Float(isinf=True), fp.Float(isinf=True, s=True),
        ]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), x

    def test_inside_a_loop(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A):
            acc = 0
            for a in A:
                with fp.MPFixedContext(-8):
                    aq = fp.round(a)
                with fp.FP64:
                    acc += aq
            return acc

        out = UnfoldNegZero.apply(f.ast)
        A = [0.1, 0.25, -3.5, 1e-6, -1e-6, 7.0]
        assert _same(_eval(out, f, A), f(A))
