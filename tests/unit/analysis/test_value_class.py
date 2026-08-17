"""
Value-class analysis: can a value be a NaN, an infinity, a zero, or finite?

Two things need checking and they need it differently.  The **transfer
functions** are claims about arithmetic, so they are swept against the
interpreter: every combination of operand classes, every sample value in each,
and the observed result must fall inside what the table predicted.  The
**refinement** is a claim about control flow, so it is checked by reading the
class the analysis gives a marked expression in each arm.
"""

import pytest

import fpy2 as fp
import fpy2.strategies as st
from fpy2.analysis import ValueClass, ValueClassInfer, class_of, representable_classes
from fpy2.analysis.value_class import _LOGB, _POW_POS_BASE, _exact_add, _exact_mul, _map
from fpy2.ast.fpyast import Expr
from fpy2.ast.visitor import DefaultVisitor
from fpy2.types import ListType, RealType

NAN, INF, ZERO, FINITE = (
    ValueClass.NAN, ValueClass.INF, ValueClass.ZERO, ValueClass.FINITE)
TOP = ValueClass.TOP

_NAN, _INF = float('nan'), float('inf')

_ATOMS = (NAN, INF, ZERO, FINITE)

_SAMPLES: dict[ValueClass, list[float]] = {
    NAN: [_NAN],
    INF: [_INF, -_INF],
    ZERO: [0.0, -0.0],
    FINITE: [1.0, -1.0, 2.5, -0.5, 3.0, 1e300, -1e-300],
}


def _analyze(func):
    return ValueClassInfer.analyze(func.ast)


def _find_all(ast, text: str) -> list[Expr]:
    """Every expression in *ast* that prints as *text*.

    The ``fp.`` qualifier goes: a source-level AST prints ``fp.logb(x)`` where a
    transformed one prints ``logb(x)``, and the tests span both.
    """
    found: list[Expr] = []

    class _Collect(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if e.format().replace('fp.', '') == text:
                found.append(e)
            return super()._visit_expr(e, ctx)

    _Collect()._visit_function(ast, None)
    assert found, f'nothing prints as {text!r}'
    return found


def _find(ast, text: str) -> Expr:
    found = _find_all(ast, text)
    assert len(found) == 1, f'{len(found)} expressions print as {text!r}'
    return found[0]


def _cls(func, text: str) -> ValueClass:
    """The class the analysis gives the expression printing as *text*."""
    info = _analyze(func)
    return info.classify(_find(func.ast, text))


#####################################################################
# The lattice

class TestTheLattice:
    def test_the_atoms_partition_every_value(self):
        for x, want in (
            (_NAN, NAN), (_INF, INF), (-_INF, INF),
            (0.0, ZERO), (-0.0, ZERO), (1.5, FINITE), (-1e300, FINITE),
        ):
            assert class_of(fp.REAL.round(x)) is want, x

    def test_bottom_is_falsy_and_top_holds_every_atom(self):
        assert not ValueClass(0)
        assert all(a in TOP for a in _ATOMS)

    @pytest.mark.parametrize('ctx, want', [
        pytest.param(fp.REAL, TOP, id='real'),
        pytest.param(fp.FP64, TOP, id='fp64'),
        pytest.param(fp.FP32, TOP, id='fp32'),
        pytest.param(fp.MX_E4M3, NAN | ZERO | FINITE, id='e4m3_has_no_inf'),
        pytest.param(fp.SINT32, ZERO | FINITE, id='sint32'),
        pytest.param(fp.UINT16, ZERO | FINITE, id='uint16'),
        pytest.param(fp.INTEGER, ZERO | FINITE, id='integer'),
    ])
    def test_what_a_context_can_hold(self, ctx, want):
        assert representable_classes(ctx) == want

    def test_a_refusing_context_holds_neither_special(self):
        """The one the lowering pipeline produces: it states no NaN and no
        infinity, and rounding one raises rather than answering."""
        ctx = fp.MPBFixedContext(-1, fp.RealFloat(exp=10, c=1),
                                 overflow=fp.OverflowMode.ASSERT)
        assert representable_classes(ctx) == ZERO | FINITE

    def test_a_substituted_special_is_the_substitute_s_class(self):
        """``nan_value`` is a value the rounding produces, not a refusal, so the
        class it lands in is the substitute's -- here a finite one."""
        ctx = fp.MPBFixedContext(
            -1, fp.RealFloat(exp=10, c=1), enable_nan=False,
            nan_value=fp.Float(x=fp.RealFloat(exp=0, c=7), ctx=fp.REAL))
        assert representable_classes(ctx) == ZERO | FINITE
        assert class_of(ctx.round(fp.Float(isnan=True))) is FINITE


#####################################################################
# Transfer functions, against the interpreter

@fp.fpy(ctx=fp.REAL)
def _add(a: fp.Real, b: fp.Real) -> fp.Real:
    return a + b


@fp.fpy(ctx=fp.REAL)
def _sub(a: fp.Real, b: fp.Real) -> fp.Real:
    return a - b


@fp.fpy(ctx=fp.REAL)
def _mul(a: fp.Real, b: fp.Real) -> fp.Real:
    return a * b


@fp.fpy(ctx=fp.REAL)
def _logb(a: fp.Real) -> fp.Real:
    return fp.logb(a)


@fp.fpy(ctx=fp.REAL)
def _pow2(a: fp.Real) -> fp.Real:
    return 2 ** a


class TestTransferFunctionsAreSound:
    """Every observed result must be inside the predicted class.

    The tables are imported directly: they are the claim under test, and driving
    them through a program would need a distinct refinement ladder per operand
    pair.  ``TestRefinement`` covers the wiring.

    An operation the interpreter refuses contributes nothing -- the analysis
    describes executions in which every operation has a result.
    """

    def _sweep(self, predict, fn, arity: int):
        bad = []
        atoms = [(a,) for a in _ATOMS] if arity == 1 else [
            (a, b) for a in _ATOMS for b in _ATOMS]
        for combo in atoms:
            want = predict(*combo)
            for vals in _value_combos(combo):
                try:
                    got = class_of(fn(*vals))
                except Exception:
                    continue        # no result: says nothing about the class
                if not (got & want):
                    bad.append(f'{fn.name}{vals}: {got} not in {want}')
        assert not bad, '; '.join(bad[:6])

    def test_add(self):
        self._sweep(_exact_add, _add, 2)

    def test_sub(self):
        self._sweep(_exact_add, _sub, 2)

    def test_mul(self):
        self._sweep(_exact_mul, _mul, 2)

    def test_logb(self):
        self._sweep(lambda a: _map(_LOGB, a), _logb, 1)

    def test_pow_with_a_positive_base(self):
        self._sweep(lambda a: _map(_POW_POS_BASE, a), _pow2, 1)

    @pytest.mark.parametrize('table', [_exact_add, _exact_mul],
                             ids=['add', 'mul'])
    def test_the_tables_distribute_over_the_join(self, table):
        """Sweeping one atom at a time is only enough because a table applied to
        a union is the union of applying it to each atom."""
        for a in _every_class():
            for b in _every_class():
                parts = ValueClass(0)
                for x in _ATOMS:
                    for y in _ATOMS:
                        if x & a and y & b:
                            parts |= table(x, y)
                assert table(a, b) == parts, (a, b)

    def test_the_finite_atom_really_can_reach_a_zero(self):
        """``_exact_add(FINITE, FINITE)`` admits a zero because a sum of two
        non-zero values can be one; a table reading it as ``FINITE`` alone would
        be unsound."""
        assert class_of(_add(1e300, -1e300)) is ZERO
        assert ZERO & _exact_add(FINITE, FINITE)


def _value_combos(atoms: tuple[ValueClass, ...]):
    if len(atoms) == 1:
        return [(v,) for v in _SAMPLES[atoms[0]]]
    return [(x, y) for x in _SAMPLES[atoms[0]] for y in _SAMPLES[atoms[1]]]


def _every_class():
    for i in range(16):
        yield ValueClass(i)


#####################################################################
# Refinement

@fp.fpy(ctx=fp.REAL)
def _ladder(x: fp.Real) -> fp.Real:
    if fp.isnan(x):
        y = 0
    elif fp.isinf(x):
        y = 1
    elif x == 0:
        y = 2
    else:
        y = fp.logb(x)
    return y


@fp.fpy(ctx=fp.REAL)
def _zero_test_only(x: fp.Real) -> fp.Real:
    if x == 0:
        y = 0
    else:
        y = fp.logb(x)
    return y


@fp.fpy(ctx=fp.REAL)
def _both_arms(x: fp.Real) -> fp.Real:
    if fp.isnan(x):
        y = fp.logb(x)
    else:
        y = fp.fabs(x)
    return y


@fp.fpy(ctx=fp.REAL)
def _siblings(x: fp.Real) -> fp.Real:
    if fp.isnan(x):
        y = 0
    elif fp.isinf(x):
        y = fp.fabs(x)
    else:
        y = 1
    return y


class TestRefinement:
    def test_the_ladder_reaches_finite(self):
        """The chain from the module docstring: three tests intersect down to a
        finite non-zero, and only then is ``logb`` free of an infinity."""
        assert _cls(_ladder, 'logb(x)') == ZERO | FINITE

    def test_a_failed_zero_test_does_not_mean_non_zero(self):
        """The trap.  A NaN compares false to everything, so it takes the ``else``
        arm too, and ``logb`` there can still be a NaN."""
        assert not float('nan') == 0     # noqa: SIM201 -- the point of the test
        assert NAN & _cls(_zero_test_only, 'logb(x)')

    def test_each_arm_gets_its_own_refinement(self):
        assert _cls(_both_arms, 'logb(x)') == NAN
        assert _cls(_both_arms, 'abs(x)') == INF | ZERO | FINITE

    def test_a_sibling_arm_does_not_inherit_the_first_arm_s_mask(self):
        """``isinf(x)`` holds in the second arm of the ladder, and ``isnan(x)``
        failed to get there -- so ``x`` is an infinity, not nothing at all.
        Narrowing the second condition against the *first arm's* mask instead of
        the enclosing one intersected ``{NaN}`` with ``{Inf}`` and drove every
        later use to the empty class."""
        assert _cls(_siblings, 'abs(x)') == INF

    def test_a_phi_joins_the_arms(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isnan(x):
                y = fp.nan()
            else:
                y = 1
            return fp.fabs(y)

        assert _cls(f, 'abs(y)') == NAN | FINITE

    def test_isfinite_refines_both_ways(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isfinite(x):
                y = fp.fabs(x)
            else:
                y = fp.logb(x)
            return y

        assert _cls(f, 'abs(x)') == ZERO | FINITE
        assert _cls(f, 'logb(x)') == NAN | INF

    def test_isnormal_implies_finite_and_non_zero(self):
        """The premise the refinement rests on, checked against the interpreter:
        neither a special nor a zero nor a subnormal is normal."""
        @fp.fpy(ctx=fp.REAL)
        def normal(a: fp.Real) -> bool:
            return fp.isnormal(a)

        for v in (_NAN, _INF, -_INF, 0.0, -0.0, 5e-324):
            assert not normal(v), v
        assert normal(1.0)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isnormal(x):
                y = fp.fabs(x)
            else:
                y = 0
            return y

        assert _cls(f, 'abs(x)') == FINITE

    def test_a_conjunction_refines_by_every_conjunct(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isfinite(x) and x != 0:
                y = fp.fabs(x)
            else:
                y = 0
            return y

        assert _cls(f, 'abs(x)') == FINITE

    def test_a_failed_disjunction_refines_by_every_disjunct(self):
        """The natural spelling of the ladder in one test: neither disjunct held,
        so both are ruled out."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isnan(x) or fp.isinf(x):
                y = 0
            else:
                y = fp.fabs(x)
            return y

        assert _cls(f, 'abs(x)') == ZERO | FINITE

    def test_a_negation_swaps_the_arms(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if not fp.isnan(x):
                y = fp.fabs(x)
            else:
                y = 0
            return y

        assert _cls(f, 'abs(x)') == INF | ZERO | FINITE

    def test_an_ordered_comparison_rules_out_a_nan_where_it_holds(self):
        """A NaN compares false to everything, so a comparison that *holds*
        proves both sides ordered -- and its failure proves nothing."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if x < 1:
                y = fp.fabs(x)
            else:
                y = fp.logb(x)
            return y

        assert _cls(f, 'abs(x)') == INF | ZERO | FINITE
        assert _cls(f, 'logb(x)') == TOP

    def test_equality_against_a_non_zero_literal_pins_finite(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if x == 3:
                y = fp.fabs(x)
            else:
                y = 0
            return y

        assert _cls(f, 'abs(x)') == FINITE

    def test_an_inline_conditional_refines_its_branches(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return fp.logb(x) if fp.isinf(x) else fp.fabs(x)

        assert _cls(f, 'logb(x)') == INF
        assert _cls(f, 'abs(x)') == NAN | ZERO | FINITE


class TestLoops:
    def test_a_refinement_inside_a_body_does_not_escape_it(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            y = 0
            while y < n:
                if fp.isfinite(x):
                    y = y + 1
                else:
                    y = n
            return fp.fabs(x)

        assert _cls(f, 'abs(x)') == TOP

    def test_a_loop_phi_settles(self):
        """The lattice has height 4, so the fixpoint converges without widening;
        the accumulator ends up admitting the zero it starts at."""
        @fp.fpy(ctx=fp.REAL)
        def f(n: fp.Real) -> fp.Real:
            y = 0
            for i in range(n):
                y = y + 1
            return fp.fabs(y)

        assert _cls(f, 'abs(y)') == ZERO | FINITE


class TestNonScalars:
    def test_a_list_carries_no_class_and_its_elements_are_unconstrained(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs) -> fp.Real:
            return fp.fabs(xs[0])

        mono = st.monomorphize(f, args=[ListType(RealType(fp.FP64))])
        info = ValueClassInfer.analyze(mono.ast)
        assert info.by_expr[_find(mono.ast, 'xs')] is None
        assert info.classify(_find(mono.ast, 'abs(xs[0])')) == TOP


class TestArgumentsAndContexts:
    def test_an_integer_argument_is_neither_special(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return fp.fabs(x)

        mono = st.monomorphize(f, args=[RealType(fp.SINT32)])
        info = ValueClassInfer.analyze(mono.ast)
        assert info.classify(_find(mono.ast, 'abs(x)')) == ZERO | FINITE

    def test_a_float_argument_is_unconstrained(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return fp.fabs(x)

        mono = st.monomorphize(f, args=[RealType(fp.FP32)])
        info = ValueClassInfer.analyze(mono.ast)
        assert info.classify(_find(mono.ast, 'abs(x)')) == TOP

    def test_a_narrow_context_bounds_a_result_by_what_it_represents(self):
        """Rounding under a context yields a value that context holds, so an
        operation under an integer one is neither a NaN nor an infinity however
        unconstrained its operands."""
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.SINT32:
                z = x + y
            return z

        assert _cls(f, '(x + y)') == ZERO | FINITE

    def test_a_selection_is_not_bounded_by_its_context(self):
        """``min`` returns an operand unrounded, so it can carry a NaN out of a
        context that has none -- the class is the operands' join, not the
        context's."""
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.SINT32:
                z = fp.fmin(x, y)
            return z

        assert _cls(f, 'fmin(x, y)') == TOP

    def test_a_call_is_not_bounded_by_the_caller_s_context(self):
        @fp.fpy(ctx=fp.REAL)
        def g(x: fp.Real) -> fp.Real:
            return x

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.SINT32:
                y = g(x)
            return y

        assert _cls(f, 'g(x)') == TOP


class TestTheLoweredRounding:
    """The payoff, and the acceptance test for the consumers that follow.

    Every guard the lowered `FP16` rounding emits asks one of these four
    questions, and the answers are what the ``elif`` ladder three levels up
    already established.
    """

    @staticmethod
    def _lowered():
        @fp.fpy(ctx=fp.REAL)
        def q(x: fp.Real) -> fp.Real:
            with fp.FP16:
                y = fp.round(x)
            return y

        ref = st.monomorphize(q, args=[RealType(fp.FP32)])
        return st.rescale_fixed(st.float_to_fixed(
            st.unfold_overflow(ref, early_check=True)))

    @pytest.mark.parametrize('text', [
        '(16777216 * x)',       # asserted finite before rounding
        '((2 ** -exp) * x)',    # likewise
        '-exp',                 # ldexp's exponent, guarded by a branch
        'exp',
    ])
    def test_the_guarded_expression_is_provably_finite(self, text):
        """Every occurrence: ``exp`` is read twice, once per ``ldexp``."""
        low = self._lowered()
        info = ValueClassInfer.analyze(low.ast)
        assert all(info.is_finite(e) for e in _find_all(low.ast, text)), text

    def test_the_operand_of_the_ladder_is_not(self):
        """The refinement is what does the work, not a blanket answer: the same
        variable outside the ladder admits every class."""
        low = self._lowered()
        info = ValueClassInfer.analyze(low.ast)
        assert not info.is_finite(_find(low.ast, 'x >= 65536'))
