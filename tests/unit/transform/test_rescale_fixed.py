"""
Unit tests for the :class:`fpy2.transform.RescaleFixed` transform.

The rewrite mints fresh ``_tN`` names via ``Gensym``, so comparing
against a hand-written golden AST is brittle.  These tests assert:

1. **Structural shape**: the block's context lands at scale zero, and
   the round is wrapped by two ``with fp.REAL:`` scalings.
2. **Negative checks**: blocks the rewrite must not touch compare via
   ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the interpreter, since scaling by a
   power of two commutes with fixed-point rounding exactly — including
   at the wrap/saturate boundary and for every rounding mode.
"""

import fpy2 as fp
import pytest

from fpy2.ast.fpyast import Call, ContextStmt, ForeignVal, FuncDef, Integer
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import REAL, OverflowMode, RealFloat, RoundingMode
from fpy2.transform import RescaleFixed
from fpy2.transform.rescale_fixed import _scale_of


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


def _fixed_scales(ast: FuncDef, ctx_type: type = fp.FixedContext) -> list[int]:
    """The scale of every fixed-point block in *ast*, however it is written."""
    scales: list[int] = []
    for stmt in _blocks(ast):
        e = stmt.ctx
        if isinstance(e, Call) and e.fn is ctx_type:
            # the position argument sits after `signed` for `FixedContext`
            index = 1 if ctx_type is fp.FixedContext else 0
            pos = e.args[index].val if len(e.args) > index else None
            if pos is None:
                pos = next(v.val for n, v in e.kwargs if n in ('scale', 'nmin'))
            scales.append(pos if ctx_type in (fp.FixedContext, fp.SMFixedContext) else pos + 1)
        elif isinstance(e, ForeignVal) and isinstance(e.val, ctx_type):
            scales.append(_scale_of(e.val))
    return scales


def _real_blocks(ast: FuncDef) -> int:
    """The number of ``with fp.REAL:`` blocks in *ast*."""
    return sum(
        1 for s in _blocks(ast)
        if isinstance(s.ctx, ForeignVal) and s.ctx.val is REAL
    )


def _eval(ast: FuncDef, fn: fp.Function, *args):
    """Run *ast* through the interpreter using *fn*'s env."""
    return fn.with_ast(ast)(*args)


def _same(a, b) -> bool:
    """Bit-exact comparison that also matches NaN to NaN."""
    if a.isnan or b.isnan:
        return a.isnan and b.isnan
    if a.isinf or b.isinf:
        return a.isinf and b.isinf and a.s == b.s
    return a.as_rational() == b.as_rational() and a.s == b.s


_SUBSTITUTING = fp.FixedContext(
    True, -16, 32, RoundingMode.RNE, OverflowMode.WRAP,
    nan_value=fp.FixedContext(True, -16, 32).maxval(s=True),
    inf_value=fp.FixedContext(True, -16, 32).maxval(s=True),
)
"""a format that substitutes a finite value for NaN and infinity"""


def _quantizer(ctx: fp.FixedContext) -> fp.Function:
    @fp.fpy(ctx=fp.REAL)
    def q(x):
        with ctx:
            xq = fp.round(x)
        return xq
    return q


# ----------------------------------------------------------------------
# The rewrite


class TestRescale:
    """A quantization block moves to scale zero, with exact scalings
    around the round."""

    def test_shape(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _real_blocks(out) == 2
        assert _same(_eval(out, f, 0.1), f(0.1))

    def test_keeps_the_written_call_form(self):
        """The context stays a ``FixedContext(...)`` call, so the
        rewritten program reads like the original."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -8, 16):
                aq = fp.round(a)
            return aq

        block = _blocks(RescaleFixed.apply(f.ast))[0]
        assert isinstance(block.ctx, Call) and block.ctx.fn is fp.FixedContext

    def test_positive_scale(self):
        """A scale above zero shifts down instead of up."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, 4, 16):
                aq = fp.round(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        for x in (0.0, 1.0, 100.0, -37.5, 1e4):
            assert _same(_eval(out, f, x), f(x)), x

    def test_several_rounds_in_one_block(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _real_blocks(out) == 4
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

    def test_chained_rounds(self):
        """A round of an earlier output still sees the original
        magnitude, since each round scales in and back out."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
                bq = fp.round(aq)
            return bq

        out = RescaleFixed.apply(f.ast)
        assert _same(_eval(out, f, 0.1), f(0.1))

    def test_cast(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -4, 16):
                aq = fp.cast(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _same(_eval(out, f, 0.25), f(0.25))

    def test_returned_round(self):
        """A block whose body returns the round rewrites too, scaling out
        into a temporary that the return then names."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                return fp.round(a)

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _real_blocks(out) == 2
        assert _same(_eval(out, f, 0.1), f(0.1))

    def test_keyword_context_arguments(self):
        """The position argument is replaced whether it was written
        positionally or by keyword."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(signed=True, scale=-16, nbits=32):
                aq = fp.round(a)
            return aq

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        assert _same(_eval(out, f, 0.1), f(0.1))

    def test_inside_a_loop(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A):
            acc = 0
            for a in A:
                with fp.FixedContext(True, -16, 32):
                    aq = fp.round(a)
                with fp.FP64:
                    acc += aq
            return acc

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        A = [0.1, 0.25, -3.5, 1e-6, 7.0]
        assert _same(_eval(out, f, A), f(A))


# ----------------------------------------------------------------------
# The other fixed-point contexts


class TestContextVariants:
    """Every fixed-point context shifts: `FixedContext` and
    `SMFixedContext` by ``scale``, `MPFixedContext` and `MPBFixedContext`
    by ``nmin``, which is one position below the scale."""

    @pytest.mark.parametrize('scale', [-16, -1, 3])
    def test_sm_fixed(self, scale):
        f = _quantizer(fp.SMFixedContext(scale, 16))
        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out, fp.SMFixedContext) == [0]
        for x in (0.1, -0.1, 1.0, 300.0, 0.0):
            assert _same(_eval(out, f, x), f(x)), x

    @pytest.mark.parametrize('scale', [-16, -1, 3])
    def test_mp_fixed(self, scale):
        """Unbounded magnitude: only the digit position shifts."""
        f = _quantizer(fp.MPFixedContext(nmin=scale - 1))
        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out, fp.MPFixedContext) == [0]
        for x in (0.1, -0.1, 1.0, 1e9, 0.0):
            assert _same(_eval(out, f, x), f(x)), x

    @pytest.mark.parametrize('scale', [-16, -1, 3])
    def test_mpb_fixed(self, scale):
        """The bound shifts with the format, so the integer range is
        unchanged."""
        maxval = RealFloat(exp=scale, c=(1 << 15) - 1)
        f = _quantizer(fp.MPBFixedContext(scale - 1, maxval))
        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out, fp.MPBFixedContext) == [0]
        for x in (0.1, -0.1, 1.0, 1e9, -1e9, 0.0):
            assert _same(_eval(out, f, x), f(x)), x

    def test_mpb_fixed_keeps_its_range(self):
        """The rescaled context represents exactly the shifted values."""
        maxval = RealFloat(exp=-4, c=(1 << 15) - 1)
        src = fp.MPBFixedContext(-5, maxval)
        f = _quantizer(src)
        out = RescaleFixed.apply(f.ast)

        block = _blocks(out)[0]
        assert isinstance(block.ctx, ForeignVal)
        dst = block.ctx.val
        assert dst.nmin == -1
        assert dst.pos_maxval.as_rational() == maxval.as_rational() * 2 ** 4


# ----------------------------------------------------------------------
# Folding the special values


class TestFoldSpecials:
    """``fold_specials`` takes NaN and the infinities out of the rounding,
    where the format defines what they become."""

    _SUBST = _SUBSTITUTING

    def test_substituting_format_is_rescalable(self):
        """A finite substitute would have to shift with the format, so the
        pass refuses it — unless the specials are folded out first."""
        f = _quantizer(self._SUBST)
        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

        out = RescaleFixed.apply(f.ast, fold_specials=True)
        assert not out.is_equiv(f.ast)
        assert _fixed_scales(out) == [0]

    @pytest.mark.parametrize('x', [
        0.1, -3.25, 1000.0, 0.0, -0.0, float('nan'), float('inf'), float('-inf'),
    ])
    def test_preserves_results(self, x):
        f = _quantizer(self._SUBST)
        out = RescaleFixed.apply(f.ast, fold_specials=True)
        assert _same(_eval(out, f, x), f(x)), x

    def test_undefined_specials_are_left_to_the_rounding(self):
        """A plain fixed format defines none of them, so nothing is folded
        and they keep raising exactly as before."""
        f = _quantizer(fp.FixedContext(True, -16, 32))
        plain = RescaleFixed.apply(f.ast)
        folded = RescaleFixed.apply(f.ast, fold_specials=True)
        assert folded.is_equiv(plain)

        for x in (float('nan'), float('inf')):
            with pytest.raises(ValueError):
                f(x)
            with pytest.raises(ValueError):
                _eval(folded, f, x)

    def test_off_by_default(self):
        f = _quantizer(self._SUBST)
        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_folding_drops_nan_from_the_format(self):
        """A NaN reaches a rounding only as its operand, and that case is
        folded away, so the rescaled format has no need of NaN."""
        src = fp.MPBFixedContext(
            -17, fp.FixedContext(True, -16, 32).maxval().as_real(),
            RoundingMode.RNE, OverflowMode.SATURATE,
            enable_nan=True, enable_inf=True,
        )
        f = _quantizer(src)
        out = RescaleFixed.apply(f.ast, fold_specials=True)

        dst = next(
            s.ctx.val for s in _blocks(out)
            if isinstance(s.ctx, ForeignVal)
            and isinstance(s.ctx.val, fp.MPBFixedContext)
        )
        assert not dst.enable_nan
        # an overflow can still produce one, so infinity stays
        assert dst.enable_inf
        for x in (float('nan'), float('inf'), float('-inf'), 0.5, -1e9):
            assert _same(_eval(out, f, x), f(x)), x

    def test_overflow_keeps_its_substitute(self):
        """Folding takes NaN out of the rounding, but an overflow still
        *produces* an infinity, so the substitute for one has to stay."""
        src = fp.MPBFixedContext(
            -4, RealFloat(exp=0, c=100), RoundingMode.RNE, OverflowMode.OVERFLOW,
            enable_nan=True, inf_value=fp.Float(isnan=True),
        )
        f = _quantizer(src)
        out = RescaleFixed.apply(f.ast, fold_specials=True)
        assert _same(_eval(out, f, 1000.0), f(1000.0))

    def test_zeros_are_not_folded(self):
        """A zero survives the shift on its own, so it needs no branch."""
        f = _quantizer(self._SUBST)
        out = RescaleFixed.apply(f.ast, fold_specials=True)
        for x in (0.0, -0.0):
            assert _same(_eval(out, f, x), f(x)), x


# ----------------------------------------------------------------------
# Selecting a single site


class TestWhere:
    """``where`` picks one candidate block by index, in visit order."""

    @staticmethod
    def _three() -> fp.Function:
        @fp.fpy(ctx=fp.REAL)
        def f(a, b, c):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
            with fp.FixedContext(True, -8, 32):
                bq = fp.round(b)
            with fp.FixedContext(True, -4, 32):
                cq = fp.round(c)
            with fp.FP64:
                s = aq + bq + cq
            return s
        return f

    @pytest.mark.parametrize('where, expect', [
        (0, [0, -8, -4]),
        (1, [-16, 0, -4]),
        (2, [-16, -8, 0]),
        (None, [0, 0, 0]),
    ])
    def test_selects_one_block(self, where, expect):
        f = self._three()
        out = RescaleFixed.apply(f.ast, where=where)
        assert _fixed_scales(out) == expect
        assert _same(_eval(out, f, 0.1, 0.2, 0.3), f(0.1, 0.2, 0.3))

    def test_index_past_the_last_site(self):
        f = self._three()
        assert RescaleFixed.apply(f.ast, where=9).is_equiv(f.ast)

    def test_rejects_a_non_integer(self):
        f = self._three()
        with pytest.raises(TypeError):
            RescaleFixed.apply(f.ast, where='first')  # type: ignore[arg-type]

    def test_counts_only_candidates(self):
        """A block the rewrite would skip does not consume an index."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FP64:                       # not fixed-point: not a candidate
                p = a * b
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a)
            return aq

        out = RescaleFixed.apply(f.ast, where=0)
        assert _fixed_scales(out) == [0]


# ----------------------------------------------------------------------
# Contexts whose position is only known at run time


class TestSymbolicPosition:
    """A context built per value shifts the same way; the factors become
    ``2 ** scale`` expressions rather than constants, and a stated bound
    shifts with the position."""

    def test_mpb_fixed(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.MPBFixedContext(k - 1, 65504, overflow=OverflowMode.OVERFLOW,
                                    enable_inf=True):
                y = fp.round(x)
            return y

        out = RescaleFixed.apply(f.ast)
        assert not out.is_equiv(f.ast)
        # the position is now zero, so the values are integers; the bound came along
        call = _blocks(out)[0].ctx
        assert isinstance(call, Call) and call.fn is fp.MPBFixedContext
        assert call.args[0].val == -1
        assert not isinstance(call.args[1], Integer)

        for x in (0.1, -3.25, 1000.0, 0.0, -0.0, 1e-9, 70000.0):
            for k in (-16, -4, 0, 3):
                assert _same(_eval(out, f, x, k), f(x, k)), (x, k)

    def test_fixed(self):
        """A format whose bound comes from `nbits` needs no bound rewriting."""

        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.FixedContext(True, k, 32):
                y = fp.round(x)
            return y

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]
        for x in (0.1, -3.25, 1000.0, 0.0):
            for k in (-16, -1, 0, 3):
                assert _same(_eval(out, f, x, k), f(x, k)), (x, k)

    def test_mp_fixed(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.MPFixedContext(k - 1):
                y = fp.round(x)
            return y

        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out, fp.MPFixedContext) == [0]
        for x in (0.1, -3.25, 1e9):
            for k in (-16, -1, 0, 3):
                assert _same(_eval(out, f, x, k), f(x, k)), (x, k)

    def test_idempotent(self):
        """The rewritten context sits at position zero, so a second run has
        nothing to shift."""

        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.MPBFixedContext(k - 1, 65504, overflow=OverflowMode.OVERFLOW,
                                    enable_inf=True):
                y = fp.round(x)
            return y

        once = RescaleFixed.apply(f.ast)
        assert RescaleFixed.apply(once).is_equiv(once)

    def test_scale_in_a_variable_needs_no_binding(self):
        """``nmin = k - 1`` means the scale is ``k`` itself: the two cancel,
        leaving only the scale-in and scale-out blocks."""

        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.MPFixedContext(k - 1):
                y = fp.round(x)
            return y

        assert _real_blocks(RescaleFixed.apply(f.ast)) == 2

    def test_computed_scale_binds_once(self):
        """Any other position expression is bound once, so it is evaluated
        once no matter how many times the factors appear."""

        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.MPFixedContext(k - 2):
                y = fp.round(x)
            return y

        out = RescaleFixed.apply(f.ast)
        assert _real_blocks(out) == 3          # the binding, then in and out
        for x in (0.1, -3.25, 1e9):
            for k in (-8, 0, 5):
                assert _same(_eval(out, f, x, k), f(x, k)), (x, k)

    def test_positional_rounding_mode_survives(self):
        """A bound written positionally must not displace the argument after
        it, which is the rounding mode."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.MPBFixedContext(-16, 1024, RoundingMode.RTZ, OverflowMode.SATURATE):
                y = fp.round(a)
            return y

        out = RescaleFixed.apply(f.ast)
        for x in (0.5, -0.5, 3.7, 1e9):
            assert _same(_eval(out, f, x), f(x)), x

    def test_no_shared_ast_nodes(self):
        """A node belongs in one place in the tree: expressions compare by
        identity, and analyses key on them."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with _SUBSTITUTING:
                p = fp.round(a)
                r = fp.round(b)
            with fp.FP64:
                s = p + r
            return s

        seen: set[int] = set()
        shared: list[str] = []

        class _C(DefaultVisitor):
            def _visit_expr(self, e, ctx):
                if id(e) in seen:
                    shared.append(type(e).__name__)
                seen.add(id(e))
                super()._visit_expr(e, ctx)

        _C()._visit_function(RescaleFixed.apply(f.ast, fold_specials=True), None)
        assert not shared

    def test_unrecognized_call_is_left_alone(self):
        """Only the fixed-point constructors are rewritten."""

        @fp.fpy(ctx=fp.REAL)
        def f(x, k):
            with fp.IEEEContext(5, 16):
                y = fp.round(x)
            return y

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Blocks the rewrite must leave alone


class TestUnchanged:

    def test_already_integral(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, 0, 32):
                aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_already_integral_mp_fixed(self):
        """``nmin = -1`` is the integer position, so there is nothing to shift."""
        f = _quantizer(fp.MPFixedContext(nmin=-1))
        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_not_fixed_point(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FP64:
                aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_arithmetic_in_body(self):
        """Scaling does not commute with a product of scaled values."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FixedContext(True, -16, 32):
                p = a * b
            return p

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_round_of_an_expression(self):
        """The argument must be a variable: an inner expression would
        round under the block's context before the shift applies."""

        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.FixedContext(True, -16, 32):
                aq = fp.round(a + b)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_bound_context(self):
        """``with C as c:`` exposes the context to the body as a value,
        which the rescaled context would change."""

        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32) as c:
                aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)

    def test_nested_block_in_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a):
            with fp.FixedContext(True, -16, 32):
                with fp.FP64:
                    aq = fp.round(a)
            return aq

        assert RescaleFixed.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:
    """Rounding commutes with a power-of-two shift exactly, so the
    rewrite must be bit-exact for every format parameter."""

    @pytest.mark.parametrize('signed', [True, False])
    @pytest.mark.parametrize('scale', [-16, -4, -1, 1, 3])
    @pytest.mark.parametrize('nbits', [8, 32])
    def test_formats(self, signed, scale, nbits):
        ctx = fp.FixedContext(signed, scale, nbits)
        f = _quantizer(ctx)
        out = RescaleFixed.apply(f.ast)
        assert _fixed_scales(out) == [0]

        hi = 2.0 ** (nbits + scale)
        xs = [0.0, 2.0 ** scale, 1.0, hi / 3, hi, 2 * hi, 1e-9]
        if signed:
            xs += [-x for x in xs]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), (signed, scale, nbits, x)

    @pytest.mark.parametrize('rm', list(RoundingMode))
    @pytest.mark.parametrize('overflow', [OverflowMode.WRAP, OverflowMode.SATURATE])
    def test_rounding_and_overflow_modes(self, rm, overflow):
        ctx = fp.FixedContext(True, -8, 16, rm, overflow)
        f = _quantizer(ctx)
        out = RescaleFixed.apply(f.ast)
        for x in (0.1, -0.1, 2.0 ** -9, 127.9961, 128.0, 1e5, -1e5, 0.0, -0.0):
            assert _same(_eval(out, f, x), f(x)), (rm, overflow, x)
