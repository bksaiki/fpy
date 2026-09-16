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
from fpy2.analysis.value_class import (
    _ATOMS, _LOGB, _POW_POS_BASE, _exact_add, _exact_mul, _exact_select,
    _exact_sub, _map,
)
from fpy2.ast.fpyast import Expr, Var
from fpy2.ast.visitor import DefaultVisitor
from fpy2.types import ListType, RealType

NAN, ZERO, FINITE = ValueClass.NAN, ValueClass.ZERO, ValueClass.FINITE
POS_INF, NEG_INF = ValueClass.POS_INF, ValueClass.NEG_INF
INF = ValueClass.INF        # the composite, `POS_INF | NEG_INF`
TOP = ValueClass.TOP

_NAN, _INF = float('nan'), float('inf')

_SAMPLES: dict[ValueClass, list[float]] = {
    NAN: [_NAN],
    POS_INF: [_INF],
    NEG_INF: [-_INF],
    ZERO: [0.0, -0.0],
    FINITE: [1.0, -1.0, 2.5, -0.5, 3.0, 1e300, -1e-300],
}


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
    info = ValueClassInfer.analyze(func.ast)
    return info.classify(_find(func.ast, text))


#####################################################################
# The lattice

class TestTheLattice:
    def test_the_atoms_partition_every_value(self):
        for x, want in (
            (_NAN, NAN), (_INF, POS_INF), (-_INF, NEG_INF),
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


@fp.fpy(ctx=fp.REAL)
def _max2(a: fp.Real, b: fp.Real) -> fp.Real:
    return max(a, b)


@fp.fpy(ctx=fp.REAL)
def _min2(a: fp.Real, b: fp.Real) -> fp.Real:
    return min(a, b)


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
        # its own table since the sign split: `_exact_add` sweeps against `-`
        # only while the infinities are one atom
        self._sweep(_exact_sub, _sub, 2)

    def test_mul(self):
        self._sweep(_exact_mul, _mul, 2)

    def test_max(self):
        self._sweep(lambda a, b: _exact_select([a, b], is_max=True), _max2, 2)

    def test_min(self):
        self._sweep(lambda a, b: _exact_select([a, b], is_max=False), _min2, 2)

    def test_logb(self):
        self._sweep(lambda a: _map(_LOGB, a), _logb, 1)

    def test_pow_with_a_positive_base(self):
        self._sweep(lambda a: _map(_POW_POS_BASE, a), _pow2, 1)

    @pytest.mark.parametrize('table', [
        pytest.param(_exact_add, id='add'),
        pytest.param(_exact_mul, id='mul'),
        pytest.param(lambda a, b: _exact_select([a, b], is_max=True), id='max'),
    ])
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
    for i in range(1 << len(_ATOMS)):
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


class TestTheSignedInfinities:
    """``POS_INF`` and ``NEG_INF`` are separate atoms; ``INF`` is their join."""

    def test_the_composite_reads_as_before(self):
        # every consumer outside this module asks `cls & INF`, i.e. "infinite
        # at all"; that question must not have changed meaning
        assert INF == POS_INF | NEG_INF
        assert POS_INF & INF and NEG_INF & INF
        assert not (ZERO & INF) and not (NAN & INF)

    def test_negation_swaps_them(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isinf(x):
                return -x
            return 0.0

        assert _cls(f, '-x') == INF          # either, since `x` is either

    def test_abs_has_no_negative_infinity(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return abs(x)

        assert _cls(f, 'abs(x)') == TOP & ~NEG_INF

    def test_logb_of_a_zero_is_only_negative(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if x == 0:
                return fp.logb(x)
            return 0.0

        assert _cls(f, 'logb(x)') == NEG_INF

    def test_logb_of_an_infinity_is_only_positive(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isinf(x):
                return fp.logb(x)
            return 0.0

        assert _cls(f, 'logb(x)') == POS_INF

    def test_adding_the_same_infinity_is_not_a_nan(self):
        """The split's point for arithmetic: ``inf + inf`` is an infinity, and
        only ``inf + -inf`` is a NaN.  One atom could not tell them apart."""
        assert _exact_add(POS_INF, POS_INF) == POS_INF
        assert _exact_add(NEG_INF, NEG_INF) == NEG_INF
        assert _exact_add(POS_INF, NEG_INF) == NAN
        assert _exact_add(INF, INF) == NAN | INF

    def test_subtraction_is_not_addition(self):
        """``inf - inf`` is a NaN where ``inf + inf`` is not, which is why the
        two have separate tables now."""
        assert _exact_sub(POS_INF, POS_INF) == NAN
        assert _exact_sub(POS_INF, NEG_INF) == POS_INF


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
        assert _cls(_both_arms, 'abs(x)') == POS_INF | ZERO | FINITE

    def test_a_sibling_arm_does_not_inherit_the_first_arm_s_mask(self):
        """``isinf(x)`` holds in the second arm of the ladder, and ``isnan(x)``
        failed to get there -- so ``x`` is an infinity, not nothing at all.
        Narrowing the second condition against the *first arm's* mask instead of
        the enclosing one intersected ``{NaN}`` with ``{Inf}`` and drove every
        later use to the empty class."""
        assert _cls(_siblings, 'abs(x)') == POS_INF

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
        assert _cls(f, 'logb(x)') == NAN | POS_INF

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

        assert _cls(f, 'abs(x)') == POS_INF | ZERO | FINITE

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

        assert _cls(f, 'abs(x)') == POS_INF | ZERO | FINITE
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

        assert _cls(f, 'logb(x)') == POS_INF
        assert _cls(f, 'abs(x)') == NAN | ZERO | FINITE


class TestALoweredChain:
    """A chain whose tail needs a statement is not an ``And`` by the time this
    analysis runs.

    :class:`~fpy2.transform.Hoistable` rewrites it into a flat ladder of guarded
    assignments, so the conjunction the refinement reads reaches it as a phi.
    :meth:`~fpy2.analysis.value_class._ValueClassInstance._implied_ladder` is
    where that is unpicked; these check it end to end.
    """

    def test_a_lowered_and_still_refines_both_operands(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = not fp.isnan(a)
            if t:
                t = not fp.isnan(b)
            if t:
                y = fp.fabs(a) + fp.fabs(b)
            else:
                y = 0
            return y

        assert not NAN & _cls(f, 'abs(a)')
        assert not NAN & _cls(f, 'abs(b)')

    def test_the_ladder_recurses(self):
        """A third operand joins a phi whose own operand is a phi."""

        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = not fp.isnan(a)
            if t:
                t = not fp.isnan(b)
            if t:
                t = a > 0
            if t:
                y = fp.fabs(a) + fp.fabs(b)
            else:
                y = 0
            return y

        assert not NAN & _cls(f, 'abs(a)')
        assert not NAN & _cls(f, 'abs(b)')

    def test_a_lowered_or_refines_only_when_it_fails(self):
        """An ``or`` guards on the negated accumulator, and says something about
        both operands exactly where the whole chain is false."""

        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = fp.isnan(a)
            if not t:
                t = fp.isnan(b)
            if t:
                y = 0
            else:
                y = fp.fabs(a) + fp.fabs(b)
            return y

        assert not NAN & _cls(f, 'abs(a)')
        assert not NAN & _cls(f, 'abs(b)')

    def test_an_unrelated_guard_refines_nothing(self):
        """``if p: t = q`` is the same *shape* but says nothing about ``t``:
        reaching the guard with ``t`` true only means the guard held or ``q``
        did, and here the guard tests something else entirely."""

        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            t = not fp.isnan(a)
            if b > 0:
                t = True
            if t:
                y = fp.fabs(a)
            else:
                y = 0
            return y

        assert NAN & _cls(f, 'abs(a)')

    def test_a_loop_phi_is_not_a_ladder(self):
        """A rotated `while` binds its condition to a name too, and its phi has
        the same two operands -- but a value carried round a loop says nothing
        about the arm below it."""

        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real) -> fp.Real:
            c = not fp.isnan(a)
            while c:
                a = a - 1
                c = not fp.isnan(a)
            return fp.fabs(a)

        assert NAN & _cls(f, 'abs(a)')


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

        assert _cls(f, 'abs(x)') == TOP & ~NEG_INF

    def test_a_loop_phi_settles(self):
        """The lattice is finite, so the fixpoint converges without widening;
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
        assert info.classify(_find(mono.ast, 'abs(xs[0])')) == TOP & ~NEG_INF


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
        assert info.classify(_find(mono.ast, 'abs(x)')) == TOP & ~NEG_INF

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


def _trackable(fn, name: str, n: int = 4) -> bool:
    """Whether a fact about the list *name*'s elements may be recorded."""
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=[ListType(RealType(fp.FP32), n)])
    for spec in CppCompiler().specialize(m):
        if spec.ast.name != fn.name:
            continue
        info = ValueClassInfer.analyze(spec.ast)
        for e in info.by_expr:
            if isinstance(e, Var) and str(e.name) == name:
                return info.element_region(e) is not None
    raise AssertionError(f'no `{name}` in {fn.name}')


def _amax_class(fn, n: int = 4, *, scalars: int = 0) -> ValueClass:
    """The class of the ``max(...)`` over a list in *fn*, after lowering."""
    from fpy2.ast.fpyast import AMax
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=(
        [ListType(RealType(fp.FP32), n)] + [RealType(fp.FP32)] * scalars
    ))
    for spec in CppCompiler().specialize(m):
        if spec.ast.name != fn.name:
            continue
        info = ValueClassInfer.analyze(spec.ast)
        for e, v in info.by_expr.items():
            if isinstance(e, AMax):
                return v
    raise AssertionError(f'no reduction in {fn.name}')


class TestListElementClasses:
    """What a list's elements are, keyed by the location they live in.

    ``abs`` never yields a negative infinity, so a list filled with it has none
    -- until a store puts one there, through *any* name for that location.
    """

    def test_a_list_built_here_carries_its_stores(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            return max(ys)

        assert not (_amax_class(f) & NEG_INF)

    def test_a_parameter_says_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            return max(xs)

        assert _amax_class(f) == TOP

    def test_a_store_is_seen(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            ys[0] = -fp.inf()
            return max(ys)

        assert _amax_class(f) & NEG_INF

    def test_a_store_through_another_name_is_seen(self):
        """The first attempt reported a class here that a run contradicted: the
        names differ, the location does not."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            zs = ys
            zs[0] = -fp.inf()
            return max(ys)

        assert _amax_class(f) & NEG_INF

    def test_a_store_in_one_arm_reaches_the_join(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs, c: fp.Real):
            ys = [abs(x) for x in xs]
            if c > 0:
                ys[0] = -fp.inf()
            return max(ys)

        assert _amax_class(f, scalars=1) & NEG_INF


class TestAGuardOverAWholeList:
    """``all(p(x) for x in xs)`` holding means every element satisfies ``p``:
    the loop covers the list, FPy having no ``break``.

    The fact is about the contents *at the loop's exit*, so a store anywhere
    between there and the read voids it.
    """

    def test_a_universal_reaches_the_elements(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            if all([fp.isfinite(x) for x in xs]):
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == ZERO | FINITE

    def test_an_existential_reaches_the_arm_it_fails_in(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            if any([fp.isnan(x) for x in xs]):
                return 0.0
            else:
                return max(xs)

        assert not (_amax_class(f) & NAN)

    def test_the_other_arm_learns_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            if all([fp.isfinite(x) for x in xs]):
                return 0.0
            else:
                return max(xs)

        assert _amax_class(f) == TOP

    def test_a_fold_written_by_hand(self):
        """Nothing here is `ReduceFusion`'s: the predicate is inlined rather
        than bound, and ``Hoistable`` leaves the fold as a guarded assignment
        rather than an ``and``."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ok = True
            for x in xs:
                ok = ok and fp.isfinite(x)
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == ZERO | FINITE

    def test_a_fold_that_is_not_one_says_nothing(self):
        """``ok`` is the *last* element's predicate, not every element's."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ok = True
            for x in xs:
                ok = fp.isfinite(x)
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == TOP

    def test_a_fold_rebuilt_each_round_says_nothing(self):
        """``ok`` is the last element's again: the guarded assignment is a fold
        only where what it guards on is what the loop carried in."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ok = True
            for x in xs:
                ok = x > 0
                if ok:
                    ok = fp.isfinite(x)
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == TOP

    def test_a_store_inside_the_scan(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ok = True
            for x in xs:
                xs[0] = fp.nan()
                ok = ok and fp.isfinite(x)
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == TOP

    def test_a_store_into_another_list_inside_the_scan(self):
        """The stamp is per region, so this keeps the fact."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [fp.nan() for _ in xs]
            ok = True
            for x in xs:
                ys[0] = fp.nan()
                ok = ok and fp.isfinite(x)
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == ZERO | FINITE

    def test_a_store_between_the_scan_and_the_guard(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ok = all([fp.isfinite(x) for x in xs])
            xs[0] = fp.nan()
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == TOP

    def test_a_store_under_a_branch_nested_in_the_arm(self):
        """The arm *restores* the mask, so without the stamp the inner branch
        would hand back a fact its own store had invalidated."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs, c: fp.Real):
            if all([fp.isfinite(x) for x in xs]):
                if c > 0:
                    xs[0] = fp.nan()
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f, scalars=1) == TOP


def _elt_classes(fn, n: int = 4) -> dict:
    """``by_elt``, keyed by variable name, after lowering."""
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=[ListType(RealType(fp.FP32), n)])
    for spec in CppCompiler().specialize(m):
        if spec.ast.name != fn.name:
            continue
        info = ValueClassInfer.analyze(spec.ast)
        return {str(d.name): cls for d, cls in info.by_elt.items()}
    raise AssertionError(f'no {fn.name}')


class TestElementClassesPerDefinition:
    """:attr:`ValueClassAnalysis.by_elt` -- what a list's elements are *ever*
    stored at, which is the question a storage choice asks.

    Where ``by_expr`` is what a read yields at a point, this is joined over the
    whole function: a buffer has to hold every value that ever lands in it.
    """

    def test_a_list_built_here(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            return max(ys)

        assert not (_elt_classes(f)['ys'] & NEG_INF)

    def test_every_store_counts(self):
        """Not the class at the end -- the join over all of them."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            ys[0] = -fp.inf()
            return max(ys)

        assert _elt_classes(f)['ys'] & NEG_INF

    def test_a_parameter_is_absent(self):
        """Nothing was stored through it, so nothing is known -- and absent is
        how a consumer reads that."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            return max(xs)

        assert 'xs' not in _elt_classes(f)

    def test_a_list_handed_to_a_call_is_absent(self):
        @fp.fpy(ctx=fp.REAL)
        def poison(ys):
            ys[0] = fp.nan()
            return 0

        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            zs = [abs(x) for x in xs]
            w = poison(zs)
            return max(zs) + w

        assert 'zs' not in _elt_classes(f)


class TestWhichListsCarryAFact:
    """:meth:`ValueClassAnalysis.element_region` -- where a fact about a list's
    elements may be recorded at all.

    An element class is a property of the *location*, and an FPy list is a
    reference, so the region is the key.  Two names for one share it; a
    list handed to a call has none, since the callee may store through it.
    """

    def test_a_read_only_list(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            t = xs
            return fp.logb(t[0])

        assert _trackable(f, 'xs')

    def test_a_list_built_here(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [fp.logb(x) for x in xs]
            return max(ys)

        assert _trackable(f, 'ys')

    def test_an_alias_is_the_same_region_not_a_refusal(self):
        """``ys = xs`` is one location under two names, so a store through either
        lands on the region both resolve to -- which is why the region is the
        key rather than something to refuse."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = xs
            ys[0] = fp.nan()
            return fp.logb(xs[0])

        assert _trackable(f, 'xs')

    def test_a_list_handed_to_a_call_carries_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def poison(ys):
            ys[0] = fp.nan()
            return 0

        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            z = poison(xs)
            return fp.logb(xs[0]) + z

        assert not _trackable(f, 'xs')


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
