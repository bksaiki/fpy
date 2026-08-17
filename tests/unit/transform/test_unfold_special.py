"""
Unit tests for the :class:`fpy2.transform.UnfoldSpecial` transform.

A format's special-value rules become branches on the operand, so the tests
assert:

1. **Structural shape**: a branch per shed rule plus the zero branch, and a
   surviving context with the shed rules removed — its operand provably
   finite and non-zero.
2. **Negative checks**: contexts and bodies the rewrite must not touch —
   refusals, float formats, and sides the format itself needs (overflow that
   produces an infinity) — compare via ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the interpreter, bit-exactly, refusals
   included, across formats and every value class.
"""

import fpy2 as fp
import pytest

from fpy2.analysis import PartialEval
from fpy2.ast.fpyast import (
    Compare,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    IsInf,
    IsNan,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import (
    REAL,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from fpy2.transform import UnfoldSpecial


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
    """Bit-exact comparison that also matches NaN to NaN, sign included."""
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
    """Every value class: both NaNs, both infinities, both zeros, values
    rounding to zero, the grid, and — for a bounded format — the bound and
    past it."""
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
        grid += [B, RealFloat(exp=B.exp + 1, c=B.c), RealFloat(exp=B.exp + 30, c=B.c)]
    for g in grid:
        xs.append(fp.Float(x=g, ctx=REAL))
        xs.append(fp.Float(x=RealFloat(s=True, exp=g.exp, c=g.c), ctx=REAL))
    return xs


def _assert_agrees(f: fp.Function, out: FuncDef, x) -> None:
    """The rewrite matches the source on `x`, a refusal included."""
    try:
        want = f(x)
    except ValueError:
        with pytest.raises(ValueError):
            _eval(out, f, x)
        return
    assert _same(_eval(out, f, x), want), x


_SAT = fp.OverflowMode.SATURATE
_ZERO = fp.Float(c=0)
_MAX255 = fp.Float(x=RealFloat(exp=0, c=255))


# ----------------------------------------------------------------------
# The rewrite


class TestShape:

    def test_branches_per_rule(self):
        """One branch per shed rule, plus the zero branch."""
        src = MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT,
                              nan_value=_ZERO, inf_value=_MAX255)
        out = UnfoldSpecial.apply(_quantizer(src).ast)

        assert len(_nodes(out, IsNan)) == 1
        assert len(_nodes(out, IsInf)) == 1
        compares = _nodes(out, Compare)
        assert len(compares) == 1
        assert int(compares[0].args[1].val) == 0  # type: ignore[attr-defined]

    def test_context_loses_its_rules(self):
        src = MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT,
                              nan_value=_ZERO, inf_value=_MAX255)
        out = UnfoldSpecial.apply(_quantizer(src).ast)

        ctxs = _block_ctxs(out)
        assert REAL in ctxs
        target = next(c for c in ctxs if isinstance(c, MPBFixedContext))
        assert target.nan_value is None and target.inf_value is None
        assert not target.enable_nan and not target.enable_inf

    def test_nan_only(self):
        """A rule that is not there gets no branch."""
        src = MPFixedContext(-8, nan_value=_ZERO)
        out = UnfoldSpecial.apply(_quantizer(src).ast)
        assert len(_nodes(out, IsNan)) == 1
        assert not _nodes(out, IsInf)
        assert len(_nodes(out, Compare)) == 1  # the zero branch stays

    def test_written_call_keeps_its_form(self):
        """Shedding a rule is dropping its keyword: the defaults are exactly
        'no rule'."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8, enable_nan=True, enable_inf=True):
                y = fp.round(x)
            return y

        out = UnfoldSpecial.apply(f.ast)
        assert 'fp.MPFixedContext(-8)' in out.format()
        assert 'enable_nan' not in out.format()

    def test_representable_specials_keep_their_sign(self):
        """A fixed-point format keeps the sign of the NaN it was given, so
        the branch chooses by the operand's sign."""
        src = MPFixedContext(-8, enable_nan=True, enable_inf=True)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)
        for x in (fp.Float(isnan=True, s=True), fp.Float(isinf=True, s=True)):
            assert _eval(out, f, x).s == f(x).s

    def test_rebuilt_as_its_base(self):
        """`FixedContext` fixes its flags by construction, so a shed rule
        rebuilds it as `MPBFixedContext` — a single zero included."""
        src = fp.FixedContext(True, -8, 16, fp.RoundingMode.RNE, _SAT,
                              nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert type(target) is MPBFixedContext
        assert target.nan_value is None
        assert target.enable_neg_zero is False
        assert any(
            isinstance(s.ctx, ForeignVal) and s.ctx.val == target
            for s in _blocks(out)
        )

    def test_refusal_is_preserved(self):
        """Only the NaN rule is stated; an infinity is still refused, by the
        rebuilt context exactly as by the source."""
        src = MPFixedContext(-8, nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        assert not f(fp.Float(isnan=True)).isnan  # substituted
        for x in (fp.Float(isinf=True), fp.Float(isinf=True, s=True)):
            with pytest.raises(ValueError):
                f(x)
            with pytest.raises(ValueError):
                _eval(out, f, x)

    def test_returned_round(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8, enable_nan=True, enable_inf=True):
                return fp.round(x)

        out = UnfoldSpecial.apply(f.ast)
        src = MPFixedContext(-8, enable_nan=True, enable_inf=True)
        assert src not in _block_ctxs(out)
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_several_rounds_in_one_block(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8, enable_nan=True):
                aq = fp.round(a)
                bq = fp.round(b)
            with fp.FP64:
                s = aq + bq
            return s

        out = UnfoldSpecial.apply(f.ast)
        assert len(_nodes(out, IsNan)) == 2
        # each emitted block occupies distinct AST nodes, the shared written
        # context included
        exprs = _nodes(out, Expr)
        assert len(exprs) == len(set(map(id, exprs)))
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))
        assert _same(_eval(out, f, fp.Float(isnan=True), 1.0),
                     f(fp.Float(isnan=True), 1.0))

    def test_idempotent(self):
        """What is left states no special value, so there is nothing to
        unfold twice."""
        f = _quantizer(MPFixedContext(-8, enable_nan=True, enable_inf=True))
        once = UnfoldSpecial.apply(f.ast)
        assert UnfoldSpecial.apply(once).is_equiv(once)


# ----------------------------------------------------------------------
# Sides the format keeps


class TestPartialShed:

    def test_overflow_keeps_its_infinity(self):
        """Overflow of a *finite* operand produces the infinity, which the
        branches never see — so that side stays in the format and only the
        NaN rule is stated."""
        src = MPBFixedContext(-4, RealFloat(exp=0, c=255),
                              overflow=fp.OverflowMode.OVERFLOW,
                              enable_inf=True, nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        assert len(_nodes(out, IsNan)) == 1
        assert not _nodes(out, IsInf)
        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert target.enable_inf is True
        assert target.nan_value is None
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_only_an_unsheddable_side(self):
        """With nothing else to state, the block is left alone."""
        f = _quantizer(MPBFixedContext(
            -4, RealFloat(exp=0, c=255),
            overflow=fp.OverflowMode.OVERFLOW, enable_inf=True,
        ))
        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Blocks the rewrite must leave alone


class TestUnchanged:

    @pytest.mark.parametrize('src', [
        fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT),
        MPFixedContext(-8),
        fp.FixedContext(True, -8, 16, fp.RoundingMode.RNE, _SAT),
    ], ids=['sm', 'mp', 'twos_complement'])
    def test_pure_refusal(self, src):
        """A refusal cannot be stated as a branch, so a format with no
        special value of its own has nothing to unfold."""
        f = _quantizer(src)
        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_float_context(self):
        """A float format cannot shed its specials."""
        f = _quantizer(fp.FP16)
        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_stochastic_context(self):
        f = _quantizer(MPFixedContext(-8, fp.RoundingMode.RNE, 2, enable_nan=True))
        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_cast_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8, enable_nan=True):
                y = fp.cast(x)
            return y

        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_arithmetic_body(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8, enable_nan=True):
                p = a * b
            return p

        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_round_of_an_expression(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8, enable_nan=True):
                y = fp.round(a + b)
            return y

        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_bound_context(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8, enable_nan=True) as c:
                y = fp.round(x)
            return y

        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)


# ----------------------------------------------------------------------
# Selecting a single site


class TestWhere:
    """``where`` picks one candidate block by index, in visit order."""

    @staticmethod
    def _two() -> fp.Function:
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.MPFixedContext(-8, enable_nan=True):
                aq = fp.round(a)
            with fp.MPFixedContext(-4, enable_nan=True):
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
        out = UnfoldSpecial.apply(f.ast, where=where)
        # the FP64 block is arithmetic, so it is never a candidate
        remaining = [
            c.nmin for c in _block_ctxs(out)
            if isinstance(c, MPFixedContext) and c.enable_nan
        ]
        assert remaining == kept
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

    def test_index_past_the_last_site(self):
        f = self._two()
        assert UnfoldSpecial.apply(f.ast, where=9).is_equiv(f.ast)

    def test_rejects_a_non_integer(self):
        f = self._two()
        with pytest.raises(TypeError):
            UnfoldSpecial.apply(f.ast, where='first')  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Semantic equivalence


class TestEquivalence:
    """The rewrite must be bit-exact, a refusal counting as an outcome: it is
    the same rounding, with the rules stated rather than built in."""

    @pytest.mark.parametrize('src', [
        MPFixedContext(-8, enable_nan=True),
        MPFixedContext(-8, enable_inf=True),
        MPFixedContext(-8, enable_nan=True, enable_inf=True),
        MPFixedContext(-8, nan_value=_ZERO),
        MPFixedContext(-8, fp.RoundingMode.RTZ, nan_value=_ZERO),
        MPBFixedContext(-4, RealFloat(exp=0, c=255), overflow=_SAT,
                        nan_value=_ZERO, inf_value=_MAX255),
        MPBFixedContext(-2, RealFloat(exp=0, c=100),
                        neg_maxval=RealFloat(s=True, exp=0, c=50),
                        overflow=_SAT, nan_value=_ZERO),
        MPBFixedContext(-4, RealFloat(exp=0, c=255),
                        overflow=fp.OverflowMode.WRAP, nan_value=_ZERO),
        fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT,
                          nan_value=_ZERO),
        fp.FixedContext(True, -8, 16, fp.RoundingMode.RNE, _SAT,
                        nan_value=_ZERO, inf_value=_ZERO),
    ], ids=[
        'mp_nan', 'mp_inf', 'mp_both', 'mp_nan_value', 'mp_rtz',
        'mpb_values', 'mpb_asym', 'mpb_wrap', 'sm_nan_value', 'fixed_values',
    ])
    def test_formats(self, src):
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)
        assert not out.is_equiv(f.ast)
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_inside_a_loop(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A):
            acc = 0
            for a in A:
                with fp.MPFixedContext(-8, enable_nan=True):
                    aq = fp.round(a)
                with fp.FP64:
                    acc += aq
            return acc

        out = UnfoldSpecial.apply(f.ast)
        A = [0.1, 0.25, -3.5, 1e-6, -1e-6, 7.0]
        assert _same(_eval(out, f, A), f(A))
