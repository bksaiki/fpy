"""
Unit tests for the :class:`fpy2.transform.FloatToFixed` transform.

Float rounding is fixed-point rounding once the digit position is known, so
the tests assert:

1. **Structural shape**: the float context is gone, replaced by fixed-point
   blocks whose context is built at the computed position.
2. **Negative checks**: contexts and bodies the rewrite must not touch compare
   via ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the interpreter, bit-exactly, across formats,
   rounding modes, and every value class — specials, subnormals, the boundary
   where rounding overflows, and beyond it.
"""

import fpy2 as fp
import fpy2.strategies as st
import pytest

from fpy2.analysis import PartialEval
from fpy2.ast.fpyast import (
    Call,
    ContextStmt,
    FuncDef,
    Integer,
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
    MPFloatContext,
    MPSFloatContext,
)
from fpy2.transform import FloatToFixed, TransformDeclined, TransformReferenceError
from fpy2.types import RealType


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
    """The value of every statically-known block context in *ast*.

    A context built per value (the normal branch's) has no static value, so
    it does not appear here; :func:`_ctx_calls` covers those."""
    eval_info = PartialEval.apply(ast)
    return [
        v for s in _blocks(ast)
        if (v := eval_info.by_expr.get(s.ctx)) is not None
    ]


def _ctx_calls(ast: FuncDef, ctx_type: type) -> list[Call]:
    """Every block context in *ast* written as a call to *ctx_type*."""
    return [
        s.ctx for s in _blocks(ast)
        if isinstance(s.ctx, Call) and s.ctx.fn is ctx_type
    ]


def _has_node(ast: FuncDef, node_type) -> bool:
    found = [False]

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, node_type):
                found[0] = True
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return found[0]


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


def _samples(ctx) -> list[float]:
    """Values covering every class the lowering has to get right."""
    B = float(ctx.maxval())
    return [
        0.0, -0.0, float('inf'), float('-inf'), float('nan'),
        1.0, -1.0, B, -B, B * 1.001, B * 2, -B * 2,          # bound and past it
        2.0 ** ctx.emin, -2.0 ** ctx.emin,                    # smallest normal
        2.0 ** ctx.expmin, 3 * 2.0 ** (ctx.expmin - 1),       # subnormal range
        2.0 ** (ctx.expmin - 1), -2.0 ** (ctx.expmin - 1),    # below it: rounds to zero
    ]


# ----------------------------------------------------------------------
# The rewrite


class TestLowering:

    def test_shape(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                y = fp.round(x)
            return y

        out = FloatToFixed.apply(f.ast)
        # the float context is gone; what is left rounds under fixed-point
        # contexts, with everything else exact
        ctxs = _block_ctxs(out)
        assert REAL in ctxs
        assert not any(isinstance(c, fp.IEEEContext) for c in ctxs)
        # one rounding per finite branch: subnormal and normal
        assert len(_ctx_calls(out, MPBFixedContext)) == 2
        assert _has_node(out, Round)

    def test_specials_are_folded(self):
        """NaN, the infinities, and the zeros are constants, so the branches
        assign what the format makes of them rather than rounding."""
        f = _quantizer(fp.FP16)
        out = FloatToFixed.apply(f.ast)

        # nothing rounds outside the two finite branches
        assert len(_ctx_calls(out, MPBFixedContext)) == 2
        for x in (float('nan'), float('inf'), float('-inf'), 0.0, -0.0):
            assert _same(_eval(out, f, x), f(x)), x

    def test_nan_is_canonicalized(self):
        """Every float format gives back a positive NaN, while a fixed-point
        round would keep the sign it was given."""
        f = _quantizer(fp.FP16)
        out = FloatToFixed.apply(f.ast)

        neg_nan = fp.Float(isnan=True, s=True)
        assert not _eval(out, f, neg_nan).s
        assert _same(_eval(out, f, neg_nan), f(neg_nan))

    def test_subnormal_branch_is_static(self):
        """Below `emin` the format is fixed-point already, so that branch's
        context is a constant — nothing there depends on the exponent."""
        f = _quantizer(fp.FP16)
        out = FloatToFixed.apply(f.ast)

        calls = _ctx_calls(out, MPBFixedContext)
        static = [c for c in calls if isinstance(c.args[0], Integer)]
        assert len(static) == 1
        assert static[0].args[0].val == fp.FP16.expmin - 1

    def test_subnormal_branch_format(self):
        """The subnormal branch rounds at the format's finest position, against
        the format's own bound and rounding mode."""
        src = fp.IEEEContext(5, 16, fp.RoundingMode.RTZ)

        f = _quantizer(src)
        out = FloatToFixed.apply(f.ast)
        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))

        assert target.nmin == src.expmin - 1
        assert target.pos_maxval == src.maxval().as_real()
        assert target.rm is src.rm

    def test_computed_context_carries_the_bound(self):
        """The normal branch builds its context per value; only the position
        varies, and the bound stays the format's own."""
        f = _quantizer(fp.FP16)
        out = FloatToFixed.apply(f.ast)

        calls = _ctx_calls(out, MPBFixedContext)
        computed = [c for c in calls if not isinstance(c.args[0], Integer)]
        assert len(computed) == 1
        assert computed[0].args[1].val == 65504       # maxval, written exactly

    def test_returned_round(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                return fp.round(x)

        out = FloatToFixed.apply(f.ast)
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

        out = FloatToFixed.apply(f.ast)
        assert not any(c is fp.FP16 for c in _block_ctxs(out))
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

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

        out = FloatToFixed.apply(f.ast)
        A = [0.1, 0.25, -3.5, 1e-6, 7.0, 70000.0]
        assert _same(_eval(out, f, A), f(A))


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
        out = FloatToFixed.apply(f.ast, where=where)
        # the FP64 block is arithmetic, so it is never a candidate
        remaining = [c for c in _block_ctxs(out) if c in (fp.FP16, fp.FP32)]
        assert remaining == left
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

    def test_index_past_the_last_site(self):
        f = self._two()
        with pytest.raises(TransformReferenceError):
            FloatToFixed.apply(f.ast, where=9)

    def test_naming_a_declined_block_raises(self):
        """A fixed-point format is structurally a candidate; naming it says
        why it cannot be lowered."""
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8):
                y = fp.round(x)
            return y

        with pytest.raises(TransformDeclined, match='float format'):
            FloatToFixed.apply(f.ast, where=0)

    def test_rejects_a_non_integer(self):
        f = self._two()
        with pytest.raises(TypeError):
            FloatToFixed.apply(f.ast, where='first')  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Blocks the rewrite must leave alone


class TestUnchanged:

    def test_asymmetric_bound(self):
        """An emitted context states one bound and mirrors it, so a format
        that sets the two apart cannot be expressed."""
        f = _quantizer(MPBFloatContext(
            4, -3, fp.RealFloat(exp=0, c=15),
            neg_maxval=fp.RealFloat(s=True, exp=0, c=7),
        ))
        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_bound_the_format_cannot_represent(self):
        """A bound needing more digits than the format has is not representable
        at the position the clamp rounds at."""
        f = _quantizer(MPBFloatContext(3, -3, fp.RealFloat(exp=0, c=13)))
        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_overflow_that_raises(self):
        """A format that refuses to round an overflow at all is declined,
        rather than the refusal escaping the pass."""
        f = _quantizer(fp.IEEEContext(
            5, 16, fp.RoundingMode.RNE, fp.OverflowMode.ASSERT,
        ))
        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_shifted_exponent_encoding(self):
        """A non-zero `eoffset` is not accounted for by the parameters read
        off the format."""
        f = _quantizer(EFloatContext(4, 8, False, EFloatNanKind.NONE, 2))
        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_fixed_point_context(self):
        f = _quantizer(fp.FixedContext(True, -16, 32))
        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_stochastic_context(self):
        """Stochastic rounding would have to draw its bits at the same position."""
        f = _quantizer(fp.IEEEContext(5, 16, fp.RoundingMode.RNE, num_randbits=2))
        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_cast_body(self):
        """``cast`` asserts exactness, which the lowering would not preserve."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                y = fp.cast(x)
            return y

        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_arithmetic_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                p = a * b
            return p

        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_round_of_an_expression(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP16:
                y = fp.round(a + b)
            return y

        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)

    def test_bound_context(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16 as c:
                y = fp.round(x)
            return y

        assert FloatToFixed.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:
    """The lowering must be bit-exact: it is the same rounding, expressed
    against a different format."""

    @pytest.mark.parametrize('ctx', [
        fp.FP16, fp.FP32, fp.FP64, fp.IEEEContext(4, 8), fp.IEEEContext(8, 32),
    ], ids=['fp16', 'fp32', 'fp64', 'ieee_4_8', 'ieee_8_32'])
    def test_formats(self, ctx):
        f = _quantizer(ctx)
        out = FloatToFixed.apply(f.ast)
        for x in _samples(ctx):
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    @pytest.mark.parametrize('ctx', [
        fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
        EFloatContext(4, 8, False, EFloatNanKind.NEG_ZERO, 0),
    ], ids=['ieee_saturating', 'neg_zero_nan'])
    def test_edge_rules(self, ctx):
        """A format states its edge rule in its own terms; the lowering asks
        rather than assumes.  `NEG_ZERO` also spends the negative-zero
        encoding on NaN, so a value rounding to zero comes back positive."""
        f = _quantizer(ctx)
        out = FloatToFixed.apply(f.ast)
        assert not out.is_equiv(f.ast)
        for x in _samples(ctx) + [1e-9, -1e-9, 0.1, -0.1]:
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    @pytest.mark.parametrize('ctx', [
        MPBFloatContext(11, -14, fp.FP16.maxval().as_real()),
        MPSFloatContext(11, -14),
        MPFloatContext(11),
    ], ids=['mpb_float', 'mps_float', 'mp_float'])
    def test_multiprecision_formats(self, ctx):
        """`MPSFloatContext` has no bound, so its target needs none either;
        `MPFloatContext` has no subnormals, so it needs no branch for them."""
        f = _quantizer(ctx)
        out = FloatToFixed.apply(f.ast)
        assert not out.is_equiv(f.ast)

        xs = [0.0, -0.0, float('inf'), float('-inf'), float('nan'), 1.0, -1.0,
              0.1, -0.1, 1e9, -1e9, 2.0 ** -30, 12.5]
        if hasattr(ctx, 'emin'):
            xs += [2.0 ** ctx.emin, 2.0 ** (ctx.expmin - 1)]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    @pytest.mark.parametrize('ctx', [
        fp.MX_E5M2, fp.MX_E4M3, fp.MX_E3M2, fp.MX_E2M3, fp.MX_E2M1,
    ], ids=['e5m2', 'e4m3', 'e3m2', 'e2m3', 'e2m1'])
    def test_mx_formats(self, ctx):
        """The MX formats differ in what overflow becomes: an infinity for
        `E5M2`, a NaN for `E4M3`, the bound for the rest."""
        f = _quantizer(ctx)
        out = FloatToFixed.apply(f.ast)
        assert not out.is_equiv(f.ast)

        xs = _samples(ctx) + [
            fp.Float(isnan=True, s=True), 0.1, -0.1, 12.5, -12.5,
        ]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), (ctx, x)

    @pytest.mark.parametrize('rm', [
        fp.RoundingMode.RNE, fp.RoundingMode.RNA, fp.RoundingMode.RTZ,
        fp.RoundingMode.RTP, fp.RoundingMode.RTN, fp.RoundingMode.RAZ,
    ])
    def test_rounding_modes(self, rm):
        ctx = fp.IEEEContext(5, 16, rm)
        f = _quantizer(ctx)
        out = FloatToFixed.apply(f.ast)
        # values that land between representable ones, where the mode decides
        xs = _samples(ctx) + [0.1, -0.1, 65519.0, 65519.996, -65519.996, 1.0009765625]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), (rm, x)

    def test_overflow_boundary(self):
        """Rounding at the top of the range decides between maxval and
        infinity; the fixed target must make the same call."""
        f = _quantizer(fp.FP16)
        out = FloatToFixed.apply(f.ast)
        for x in (65504.0, 65515.0, 65519.0, 65519.996, 65520.0, 65536.0, 1e5, 1e10):
            for sx in (x, -x):
                assert _same(_eval(out, f, sx), f(sx)), sx


# ----------------------------------------------------------------------
# Branches the operand cannot take


def _guarded_quantizer(ctx) -> fp.Function:
    """A quantizer whose caller has already ruled out the specials."""
    @fp.fpy(ctx=fp.REAL)
    def q(x: fp.Real) -> fp.Real:
        if fp.isfinite(x):
            with ctx:
                y = fp.round(x)
        else:
            y = 0
        return y
    return q


class TestBranchesTheOperandCannotTake:
    """``logb`` is undefined on a NaN, an infinity and a zero, so each gets a
    branch -- but only where the operand can be one.

    The information is :class:`fpy2.analysis.ValueClassInfer`'s, which needs
    *concrete* argument types: an unmonomorphized parameter is a type variable
    and carries no class, so every branch stays.  The zero branch is not a
    special case of the other two -- ``logb(0)`` is undefined however the
    operand got there -- so it goes only when the operand cannot be zero.
    """

    def test_an_integer_operand_needs_neither_specials_branch(self):
        mono = st.monomorphize(_quantizer(fp.FP16), args=[RealType(fp.SINT32)])
        out = FloatToFixed.apply(mono.ast)
        assert not _has_node(out, IsNan)
        assert not _has_node(out, IsInf)
        assert 'x == 0' in out.format()

    def test_a_float_operand_keeps_them(self):
        """The same program, one argument type apart."""
        mono = st.monomorphize(_quantizer(fp.FP16), args=[RealType(fp.FP32)])
        out = FloatToFixed.apply(mono.ast)
        assert _has_node(out, IsNan)
        assert _has_node(out, IsInf)

    def test_a_guarded_program_needs_neither(self):
        mono = st.monomorphize(
            _guarded_quantizer(fp.FP16), args=[RealType(fp.FP32)])
        out = FloatToFixed.apply(mono.ast)
        # the program's own test survives; the lowering adds none of its own
        assert _has_node(out, IsFinite)
        assert not _has_node(out, IsNan)
        assert not _has_node(out, IsInf)

    def test_an_unmonomorphized_program_keeps_them(self):
        out = FloatToFixed.apply(_quantizer(fp.FP16).ast)
        assert _has_node(out, IsNan)
        assert _has_node(out, IsInf)

    def test_the_guarded_lowering_is_still_bit_exact(self):
        """The branches are dropped as unreachable, so the value cannot change
        -- checked over every class, the specials included: those take the
        program's own ``else`` arm."""
        f = _guarded_quantizer(fp.FP16)
        mono = st.monomorphize(f, args=[RealType(fp.FP32)])
        out = FloatToFixed.apply(mono.ast)
        for x in _samples(fp.FP16):
            assert _same(_eval(out, mono, x), mono(x)), x

    def test_an_integer_operand_lowering_is_still_bit_exact(self):
        f = _quantizer(fp.FP16)
        mono = st.monomorphize(f, args=[RealType(fp.SINT32)])
        out = FloatToFixed.apply(mono.ast)
        for x in (0, 1, -1, 3, -3, 65504, 65505, 65536, -65536, 2 ** 30):
            assert _same(_eval(out, mono, x), mono(x)), x
