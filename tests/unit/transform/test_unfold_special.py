"""
Unit tests for the :class:`fpy2.transform.UnfoldSpecial` transform.

A format's special-value rules become branches on the operand, so the tests
assert:

1. **Structural shape**: a branch per shed rule plus the zero branch, and a
   surviving context with the shed rules removed — its operand provably
   finite and non-zero.
2. **Negative checks**: contexts and bodies the rewrite must not touch —
   refusals, encoded float formats, and sides the format itself needs (overflow
   that produces an infinity) — compare via ``is_equiv`` against the original
   AST.
3. **Semantic equivalence** via the interpreter, bit-exactly, refusals
   included, across formats and every value class.
"""

import fpy2 as fp
import pytest

from fpy2.analysis import PartialEval, ValueClass
from fpy2.ast.fpyast import (
    Compare,
    ContextStmt,
    Expr,
    ForeignVal,
    FuncDef,
    IsInf,
    IsNan,
    Signbit,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import (
    REAL,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from fpy2.transform import TransformDeclined, TransformReferenceError, UnfoldSpecial
from fpy2.transform.unfold_special import _describe


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
    grid = [RealFloat(exp=0, c=1)]
    # `MPFloatContext` has no least digit: it is unbounded in both directions
    nmin = getattr(ctx, 'nmin', None)
    if nmin is not None:
        grid += [
            RealFloat(exp=nmin - 3, c=1),   # rounds to zero
            RealFloat(exp=nmin, c=1),       # the tie at half a step
            RealFloat(exp=nmin + 1, c=1),   # the grid's finest step
        ]
    if isinstance(ctx, MPBFixedContext | fp.MPBFloatContext):
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

    def test_subclass_keeps_its_class(self):
        """`FixedContext` states no NaN of its own, so a substituted value
        comes off in-class and the format keeps its identity — a single zero
        included."""
        src = fp.FixedContext(True, -8, 16, fp.RoundingMode.RNE, _SAT,
                              nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert type(target) is fp.FixedContext
        assert target.nan_value is None
        assert target.enable_neg_zero is False
        assert any(
            isinstance(s.ctx, ForeignVal) and s.ctx.val == target
            for s in _blocks(out)
        )

    def test_sm_fixed_keeps_its_class(self):
        """`SMFixedContext` is an `EncodableContext`; shedding its
        substituted value must not demote it to the plain base class."""
        src = fp.SMFixedContext(-8, 16, fp.RoundingMode.RNE, _SAT,
                                nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert type(target) is fp.SMFixedContext
        assert target.nan_value is None

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

    def test_overflow_keeps_its_infinity_in_the_format(self):
        """Overflow of a *finite* operand produces the infinity, which the
        branches never see — so that side cannot leave the format.  It is still
        *stated*, which is what leaves the surviving rounding a finite operand;
        only the NaN rule comes out of the format itself."""
        src = MPBFixedContext(-4, RealFloat(exp=0, c=255),
                              overflow=fp.OverflowMode.OVERFLOW,
                              enable_inf=True, nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        assert len(_nodes(out, IsNan)) == 1
        assert len(_nodes(out, IsInf)) == 1     # stated
        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert target.enable_inf is True        # but not shed
        assert target.nan_value is None         # this one is
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_a_shed_side_is_stated_even_where_the_class_rules_it_out(self):
        """The branch is the only thing that can supply the value once the rule
        has left the format, so it is emitted whatever the operand's class.

        Omitting it here would make the rewrite rest on the class analysis being
        right -- a NaN reaching the shed format has no answer at all.
        """
        src = MPFixedContext(-8, enable_nan=True, enable_inf=True)

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            if fp.isfinite(x):          # so the operand can be neither special
                with src:
                    y = fp.round(x)
            else:
                y = 0
            return y

        out = UnfoldSpecial.apply(f.ast)
        assert len(_nodes(out, IsNan)) == 1
        assert len(_nodes(out, IsInf)) == 1
        shed = next(c for c in _block_ctxs(out) if isinstance(c, MPFixedContext))
        assert not shed.enable_nan and not shed.enable_inf

    def test_a_side_that_cannot_be_shed_is_still_stated(self):
        """Nothing leaves the format, so the block keeps its context verbatim --
        and the branch is still worth emitting for the operand it leaves."""
        src = MPBFixedContext(
            -4, RealFloat(exp=0, c=255),
            overflow=fp.OverflowMode.OVERFLOW, enable_inf=True,
        )
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        assert len(_nodes(out, IsInf)) == 1
        assert not _nodes(out, IsNan)           # refused, so unstateable
        target = next(c for c in _block_ctxs(out) if isinstance(c, MPBFixedContext))
        assert target == src                    # untouched
        for x in _samples(src):
            _assert_agrees(f, out, x)


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
        with pytest.raises(TransformReferenceError):
            UnfoldSpecial.apply(f.ast, where=9)

    def test_rejects_a_non_integer(self):
        f = self._two()
        with pytest.raises(TypeError):
            UnfoldSpecial.apply(f.ast, where='first')  # type: ignore[arg-type]

    @staticmethod
    def _real_then_fp16() -> fp.Function:
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            with fp.REAL:
                aq = fp.round(a)
            with fp.FP16:
                bq = fp.round(b)
            return aq + bq
        return f

    def test_a_declined_block_counts_toward_where(self):
        """Candidacy is structural: the ``REAL`` block is index 0 whether or
        not it verifies, so index 1 names the ``FP16`` block."""
        f = self._real_then_fp16()
        out = UnfoldSpecial.apply(f.ast, where=1)
        assert not out.is_equiv(f.ast)
        assert _same(_eval(out, f, 0.1, 0.2), f(0.1, 0.2))

    def test_naming_a_real_block_is_declined(self):
        f = self._real_then_fp16()
        with pytest.raises(TransformDeclined, match='REAL'):
            UnfoldSpecial.apply(f.ast, where=0)

    def test_naming_a_runtime_context_is_declined(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x, n):
            with fp.MPBFixedContext(n, 255, enable_nan=True):
                y = fp.round(x)
            return y

        with pytest.raises(TransformDeclined, match='statically known'):
            UnfoldSpecial.apply(f.ast, where=0)

    def test_naming_an_unfolded_block_is_declined(self):
        """The second pass has nothing left to state, and saying so beats a
        silent no-op."""
        once = UnfoldSpecial.apply(_quantizer(fp.FP16).ast)
        with pytest.raises(TransformDeclined, match='nothing to state'):
            UnfoldSpecial.apply(once, where=0)

    def test_an_annotated_assign_is_not_a_candidate(self):
        """The rewrites cannot carry an annotation, so the block is left
        alone rather than rewritten without it."""
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.FP16:
                y: fp.Real = fp.round(x)
            return y

        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)
        with pytest.raises(TransformReferenceError):
            UnfoldSpecial.apply(f.ast, where=0)


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

    def test_cast_body(self):
        """A cast substitutes a special exactly as a round does — the
        substitution happens before the exactness check — so it sheds the
        same rules, and the rewrite asserts exactness as the source did."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(-8, nan_value=_ZERO):
                y = fp.cast(x)
            return y

        out = UnfoldSpecial.apply(f.ast)
        assert not out.is_equiv(f.ast)
        xs = [
            fp.Float(isnan=True), fp.Float(isnan=True, s=True),
            fp.Float(c=0), fp.Float(c=0, s=True), 0.5, -3.0,
        ]
        for x in xs:
            assert _same(_eval(out, f, x), f(x)), x
        inexact = fp.Float(x=RealFloat(exp=-12, c=1), ctx=REAL)
        with pytest.raises(ValueError):
            f(inexact)
        with pytest.raises(ValueError):
            _eval(out, f, inexact)

    def test_stochastic_context(self):
        """A special never reaches the random draw, so a stochastic rounding
        sheds its rules too; the surviving context keeps its random bits."""
        src = MPFixedContext(-8, fp.RoundingMode.RNE, 2, nan_value=_ZERO)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)
        assert not out.is_equiv(f.ast)

        target = next(c for c in _block_ctxs(out) if isinstance(c, MPFixedContext))
        assert target.num_randbits == 2
        assert target.nan_value is None
        # the special paths draw no bits, so they stay deterministic
        for x in (fp.Float(isnan=True), fp.Float(isnan=True, s=True),
                  fp.Float(c=0), fp.Float(c=0, s=True)):
            assert _same(_eval(out, f, x), f(x)), x

    def test_non_dyadic_operand(self):
        """The emitted branches test the operand under ``fp.REAL``, where it
        can be an exact non-dyadic rational; the guards must accept whatever
        the rounding accepted."""

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            y = x / 3
            with fp.MPFixedContext(-8, enable_nan=True):
                z = fp.round(y)
            return z

        out = UnfoldSpecial.apply(f.ast)
        assert not out.is_equiv(f.ast)
        for x in (1.0, -1.0, 0.0, -0.0, fp.Float(isnan=True)):
            assert _same(_eval(out, f, x), f(x)), x

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


# ----------------------------------------------------------------------
# Hoisting without shedding


class TestFloatFormats:
    """An *encoded* float states no special-value rule as a parameter — it
    spells NaN as a ``nan_kind``, and an encoding always has a NaN by
    construction — so it can shed nothing.  It can still have its specials
    *stated*, which is the half that leaves the surviving rounding a finite
    non-zero operand.  (The MP float family does shed; see
    ``TestMPFloatFormats``.)

    Getting the sign from the format rather than assuming it is the point: a
    hand-written ``return fp.inf()`` is wrong for ``-inf``, and
    ``return 0`` for ``-0.0``.
    """

    @pytest.mark.parametrize('src', [
        fp.FP16, fp.FP32, fp.MX_E4M3, fp.MX_E5M2,
        fp.IEEEContext(5, 16, fp.RoundingMode.RNE, _SAT),
    ], ids=['fp16', 'fp32', 'e4m3', 'e5m2', 'saturating'])
    def test_the_specials_are_stated_and_the_format_kept(self, src):
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        assert len(_nodes(out, IsNan)) == 1
        assert _block_ctxs(out).count(src) == 1, 'the format itself is unchanged'
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_the_sign_comes_from_the_format(self):
        """`FP16` keeps both signs, so each branch has to choose."""
        out = UnfoldSpecial.apply(_quantizer(fp.FP16).ast)
        assert len(_nodes(out, Signbit)) == 2      # one per signed pair
        f = _quantizer(fp.FP16)
        for x in (fp.Float(isinf=True, s=True), fp.Float(c=0, s=True)):
            _assert_agrees(f, out, x)

    def test_a_real_context_states_nothing(self):
        """``REAL`` rounds exactly, so its specials pass through untouched and
        the branches would be pure noise."""
        f = _quantizer(fp.REAL)
        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)

    def test_it_is_idempotent(self):
        """The second pass sees an operand the branches already made finite and
        non-zero, so it states nothing."""
        once = UnfoldSpecial.apply(_quantizer(fp.FP16).ast)
        assert UnfoldSpecial.apply(once).is_equiv(once)

    def test_a_context_built_per_value_is_declined(self):
        """The branch values are the context's own answers, so a context that is
        not known until run time cannot be unfolded."""
        @fp.fpy(ctx=fp.REAL)
        def f(x, n):
            with fp.MPBFixedContext(n, 255, enable_nan=True):
                y = fp.round(x)
            return y

        assert UnfoldSpecial.apply(f.ast).is_equiv(f.ast)


###########################################################
# Tests: the MP float family sheds its stated rules


_MAXVAL = RealFloat(exp=5, c=0x7ff)
"""an `FP16`-shaped bound, for the bounded float context"""


class TestMPFloatFormats:
    """`MPFloatContext`, `MPSFloatContext` and `MPBFloatContext` state NaN and
    infinity as flags, so they shed like the fixed-point contexts do.

    Their flags default to *enabled*, the opposite of the fixed-point ones, so a
    shed rule has to be written into the surviving constructor call rather than
    dropped from it -- a context that merely lost the keyword would still have
    the rule.
    """

    @pytest.mark.parametrize('src', [
        fp.MPFloatContext(11),
        fp.MPSFloatContext(11, -14),
    ], ids=['mp', 'mps'])
    def test_an_unbounded_float_sheds_both(self, src):
        """Nothing finite rounds to a special without a bound to overflow, so
        both rules come off."""
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        assert len(_nodes(out, IsNan)) == 1
        assert len(_nodes(out, IsInf)) == 1
        surviving, = [c for c in _block_ctxs(out) if c != REAL]
        assert not surviving.enable_nan
        assert not surviving.enable_inf
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_a_bounded_float_keeps_the_infinity_its_overflow_makes(self):
        """An overflow under `OVERFLOW` *produces* an infinity, which the
        branches never see, so that rule stays while its branch is still
        emitted.  The NaN has no such source and comes off."""
        src = fp.MPBFloatContext(11, -14, _MAXVAL)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        surviving, = [c for c in _block_ctxs(out) if c != REAL]
        assert not surviving.enable_nan
        assert surviving.enable_inf, 'the overflow still has to reach infinity'
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_a_saturating_bound_sheds_the_infinity_too(self):
        """Saturation never produces an infinity, so nothing finite needs the
        rule and it comes off -- the same format, shed differently, decided by
        the probes rather than by the class."""
        src = fp.MPBFloatContext(11, -14, _MAXVAL, fp.RoundingMode.RNE, _SAT)
        f = _quantizer(src)
        out = UnfoldSpecial.apply(f.ast)

        surviving, = [c for c in _block_ctxs(out) if c != REAL]
        assert not surviving.enable_nan
        assert not surviving.enable_inf
        for x in _samples(src):
            _assert_agrees(f, out, x)

    def test_the_shed_rule_is_written_into_the_call(self):
        """The constructor defaults the flags on, so shedding one is saying so
        outright -- dropping the keyword would leave the rule in place."""
        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPSFloatContext(11, -14):
                y = fp.round(x)
            return y

        out = UnfoldSpecial.apply(f.ast)
        text = out.format()
        assert 'enable_nan=False' in text
        assert 'enable_inf=False' in text

    @pytest.mark.parametrize('src', [
        fp.MPFloatContext(11),
        fp.MPSFloatContext(11, -14),
        fp.MPBFloatContext(11, -14, _MAXVAL),
    ], ids=['mp', 'mps', 'mpb'])
    def test_it_is_idempotent(self, src):
        once = UnfoldSpecial.apply(_quantizer(src).ast)
        assert UnfoldSpecial.apply(once).is_equiv(once)


###########################################################
# Tests: which rules a finite operand still needs


class TestShedable:
    """`_shedable` decides the shed analytically: a rule stays iff a *finite*
    operand can still reach it, which only an overflow does.  These are the
    cases the shape tests above do not reach.

    The rewrites themselves are checked end-to-end by the classes above, which
    round every sample under both the original and the rewritten program -- an
    over-claiming predicate fails there, not here.
    """

    def _shed(self, ctx) -> tuple[bool, bool]:
        src = _describe(ctx)
        return bool(ValueClass.NAN & src.shed), bool(ValueClass.INF & src.shed)

    def _bounded(self, **kwargs):
        return fp.MPBFloatContext(11, -14, _MAXVAL, **kwargs)

    def test_a_saturating_direction_keeps_nothing_back(self):
        """`RTZ` rounds an overflow down to the bound, so no finite operand ever
        reaches the infinity and the rule comes off."""
        assert self._shed(self._bounded(rm=fp.RoundingMode.RTZ)) == (True, True)

    def test_a_one_sided_direction_still_keeps_the_rule(self):
        """`RTP` overflows to `+inf` but saturates below, so the rule is needed
        on one side -- which is enough to keep it."""
        assert self._shed(self._bounded(rm=fp.RoundingMode.RTP)) == (True, False)

    @pytest.mark.parametrize('mode', [fp.OverflowMode.SATURATE, fp.OverflowMode.ASSERT])
    def test_an_overflow_that_makes_no_special_keeps_nothing_back(self, mode):
        """Saturation stays finite and an assertion raises, so neither reaches a
        special and both rules come off."""
        assert self._shed(self._bounded(overflow=mode)) == (True, True)

    def test_a_substituted_overflow_keeps_the_rule(self):
        """With the infinity off, a finite overflow lands on `inf_value` --
        shedding would clear it, so the rule stays."""
        ctx = self._bounded(enable_inf=False, inf_value=fp.Float(x=_MAXVAL))
        assert self._shed(ctx) == (True, False)

    def test_a_refused_overflow_has_no_rule_to_keep(self):
        """With the infinity off and no substitute the overflow is refused, and
        shedding leaves it refused -- but there is also no value for a branch to
        take over, so the infinity is neither stated nor shed."""
        ctx = self._bounded(enable_inf=False)
        assert self._shed(ctx) == (True, False)

    def test_an_unbounded_float_has_no_overflow_at_all(self):
        for ctx in (fp.MPFloatContext(11), fp.MPSFloatContext(11, -14)):
            assert self._shed(ctx) == (True, True)
