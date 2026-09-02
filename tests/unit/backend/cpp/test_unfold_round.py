"""
Detection for the rounding-unfold ladder — `fpy2.backend.cpp.unfold_round`.

`sites` answers on the specialized AST what the emitter answers during
emission: which roundings its op table cannot spell.  It reports *that*
question only — a program with no sites can still be refused for an unrelated
reason, storage selection being the usual one.
"""

import pytest

import fpy2 as fp
import fpy2.strategies as st
from fpy2 import Module
from fpy2.backend.cpp import CppCompileError, CppCompiler
from fpy2.backend.cpp.unfold_round import (
    UnfoldKind,
    UnfoldMode,
    sites,
    unfold,
    unfold_arith,
)
from fpy2.types import RealType


def _kinds(func):
    return [s.kind for s in sites(func.ast)]


def _refuses(func, src=None) -> bool:
    """Whether the backend refuses *func*.

    Pass *src* wherever the argument carries no context of its own — storage
    selection refuses an unconstrained real first, and then the refusal under
    test never fires.
    """
    try:
        if src is None:
            m = Module()
            m.add(func)
            CppCompiler().compile_module(m)
        else:
            CppCompiler().compile(func, ctx=src, arg_types=[RealType(src)])
    except CppCompileError:
        return True
    return False


class TestArith:
    """An operation the op table has no signature for under its context."""

    def test_a_native_context_is_not_a_site(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return x + x

        assert _kinds(f) == []

    def test_a_non_native_context_is(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                return x + x

        assert _kinds(f) == [UnfoldKind.ARITH]
        assert _refuses(f, fp.FP16)

    def test_real_is_not_a_site(self):
        """`REAL` is the one non-native context the table reaches, by widening
        to an op that gives the exact result and rounds to itself."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.REAL:
                return x + x

        assert _kinds(f) == []

    def test_min_does_not_dispatch(self):
        """The table's keys are the predicate, and `Min` is not one of them: it
        selects an operand and hands it back with its own format."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                return min(x, x)

        assert _kinds(f) == []
        assert not _refuses(f, fp.FP16)


class TestRounding:
    """A `Round` or `Cast` whose target context has no C++ analogue."""

    def test_a_non_native_float(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                return fp.round(x)

        assert _kinds(f) == [UnfoldKind.FLOAT_ROUND]
        assert _refuses(f, fp.FP32)

    def test_an_unbounded_float_is_still_one(self):
        """What `unfold_overflow` leaves: the bound is program text now, but
        the format is no more native than it was."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = fp.round(x)
            return y

        mono = st.monomorphize(f, args=[RealType(fp.FP32)])
        assert _kinds(st.unfold_overflow(mono, early_check=True)) == [
            UnfoldKind.FLOAT_ROUND,
        ]

    def test_a_fixed_context_away_from_position_zero(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.MPFixedContext(-4, fp.RoundingMode.RTZ):
                return fp.round(x)

        assert _kinds(f) == [UnfoldKind.FIXED_ROUND]

    def test_a_fixed_context_at_position_zero_is_not_a_site(self):
        """`_emit_integral_round` lowers it as it stands, so the ladder has
        nothing to do — even though storage selection still refuses this
        program, which is a different gap."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.MPFixedContext(-1, fp.RoundingMode.RTZ):
                return fp.round(x)

        assert _kinds(f) == []


class TestTheLadder:
    """Against the sequence of `test_lowered_roundtrip`, which is what the
    detection has to agree with."""

    @pytest.mark.parametrize('target', [
        fp.FP16,
        fp.IEEEContext(5, 16, fp.RoundingMode.RNE, fp.OverflowMode.SATURATE),
        fp.MX_E2M1,
    ], ids=['fp16', 'saturating', 'e2m1'])
    def test_each_stage_names_its_own_recovery(self, target):
        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with target:
                y = fp.round(x)
            return y

        mono = st.monomorphize(q, args=[RealType(fp.FP32)])
        assert _kinds(mono) == [UnfoldKind.FLOAT_ROUND]
        assert _refuses(mono)

        # `float_to_fixed` trades the float row for the fixed one: the digits
        # land away from position zero, which is `rescale_fixed`'s job.
        fixed = st.float_to_fixed(st.unfold_overflow(
            st.unfold_special(mono), early_check=True,
        ))
        assert _kinds(fixed) == [UnfoldKind.FIXED_ROUND]

        low = st.simplify(st.rescale_fixed(fixed))
        assert _kinds(low) == []
        assert not _refuses(low)


class TestWithin:
    def test_within_keeps_one_of_two(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP16:
                a = fp.round(x)
                b = fp.round(a)
            return b

        all_sites = sites(f.ast)
        assert len(all_sites) == 2
        first = all_sites[0]
        assert [s.cursor for s in sites(f.ast, first.cursor.stmt())] == [
            first.cursor,
        ]


class TestUnfoldArith:
    """The arithmetic row alone: compute at a native intermediate and re-round
    to the target.  `unfold` goes on to lower the rounding that leaves, which
    is what the ladder tests cover."""

    def _mono(self, func, ctx, arity=2):
        return st.monomorphize(func, args=[RealType(ctx)] * arity)

    def test_an_add_lands_under_a_native_context(self):
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        mono = self._mono(f, fp.FP16)
        out = mono.with_ast(unfold_arith(mono.ast))
        # the arithmetic is gone from the sites; what is left is the rounding
        # back to the target, which is the next row's work
        assert _kinds(out) == [UnfoldKind.FLOAT_ROUND]
        assert 'with fp.FP32:' in out.format()

    def test_the_narrowest_admissible_intermediate_wins(self):
        """`FP32` satisfies the add rule for a nearest `FP16` target, so `FP64`
        is never reached: the intermediate's width is the arithmetic's
        storage."""
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        mono = self._mono(f, fp.FP16)
        assert 'fp.FP64' not in mono.with_ast(unfold_arith(mono.ast)).format()

    def test_a_directed_target_goes_through_exactness(self):
        """The intermediate is always nearest, so a directed target reaches no
        mode rule -- but the exactness rule takes any mode, and the exact sum of
        two `FP16` values wants more than `FP32`'s 24 bits.  So it splits, at
        `FP64`.
        """
        target = fp.IEEEContext(5, 16, fp.RoundingMode.RTZ)

        @fp.fpy(ctx=target)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        mono = self._mono(f, target)
        assert 'with fp.FP64:' in mono.with_ast(unfold_arith(mono.ast)).format()

    def test_a_directed_target_without_an_exact_rule_refuses(self):
        """`div` has no exact result to hold and no rule under a directed
        target, so it keeps its refusal."""
        target = fp.IEEEContext(5, 16, fp.RoundingMode.RTZ)

        @fp.fpy(ctx=target)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x / y

        mono = self._mono(f, target)
        assert unfold_arith(mono.ast) is mono.ast

    def test_every_site_is_taken(self):
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return (x + y) * (x - y) + x / y

        mono = self._mono(f, fp.FP16)
        assert _kinds(mono) == [UnfoldKind.ARITH] * 5
        assert _kinds(mono.with_ast(unfold_arith(mono.ast))) == [
            UnfoldKind.FLOAT_ROUND,
        ] * 5

    def test_native_arithmetic_is_untouched(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        mono = self._mono(f, fp.FP64)
        assert unfold_arith(mono.ast) is mono.ast

    def test_a_transcendental_keeps_its_refusal(self):
        """The rules cover ``+ - * /`` and ``sqrt``, plus anything whose exact
        result the intermediate holds.  `exp` under a nearest target is none of
        those, so its refusal stands while the add beside it is taken."""
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return fp.exp(x) + y

        mono = self._mono(f, fp.FP16)
        assert _kinds(mono) == [UnfoldKind.ARITH] * 2
        assert _kinds(mono.with_ast(unfold_arith(mono.ast))) == [
            UnfoldKind.ARITH, UnfoldKind.FLOAT_ROUND,
        ]

    def test_an_unpinned_operand_refuses_every_candidate(self):
        """The per-operation rules hold for operands the *target* represents,
        and an argument with no context of its own is finer than any format.
        `Specialize` pins them in the compiler; a direct caller monomorphizes.
        """
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        assert unfold_arith(f.ast) is f.ast


class TestTheFlag:
    """`CppCompiler(unfold=...)`, which is the whole point."""

    def _compile(self, func, ctx, arity, **kw):
        return CppCompiler(**kw).compile(
            func, ctx=ctx, arg_types=[RealType(ctx)] * arity,
        )

    def test_a_rounding_compiles(self):
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real) -> fp.Real:
            return fp.round(x)

        with pytest.raises(CppCompileError, match='no C.. analogue'):
            self._compile(f, fp.FP16, 1)
        assert 'std::logb' in self._compile(
            f, fp.FP16, 1, unfold=UnfoldMode.DOUBLE_ROUND,
        )

    def test_arithmetic_compiles(self):
        """Both rows in one program: the add becomes native, and the rounding
        it gains becomes integer code."""
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(CppCompileError, match='no matching signature'):
            self._compile(f, fp.FP16, 2)
        out = self._compile(f, fp.FP16, 2, unfold=UnfoldMode.DOUBLE_ROUND)
        assert 'float f(float x, float y)' in out
        assert 'std::logb' in out

    def test_roundings_alone_leaves_the_arithmetic(self):
        """The middle level is a smaller claim: lowering a rounding is a
        rewrite of one operation, while rewriting arithmetic rounds twice."""
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        with pytest.raises(CppCompileError, match='no matching signature'):
            self._compile(f, fp.FP16, 2, unfold=UnfoldMode.ROUNDINGS)

    def test_roundings_alone_still_lowers_a_rounding(self):
        @fp.fpy(ctx=fp.FP16)
        def f(x: fp.Real) -> fp.Real:
            return fp.round(x)

        assert 'std::logb' in self._compile(
            f, fp.FP16, 1, unfold=UnfoldMode.ROUNDINGS,
        )

    def test_a_bool_is_refused(self):
        with pytest.raises(TypeError, match='must be an UnfoldMode'):
            CppCompiler(unfold=True)   # type: ignore[arg-type]

    def test_a_native_program_is_unchanged(self):
        """The flag is opt-in and costs nothing where nothing is unsupported —
        which is what keeps it off the corpus."""
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y + x

        assert self._compile(f, fp.FP64, 2) == self._compile(
            f, fp.FP64, 2, unfold=UnfoldMode.DOUBLE_ROUND,
        )
