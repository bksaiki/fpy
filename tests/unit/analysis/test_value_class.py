"""
Value-class analysis: can a value be a NaN, an infinity, a zero, or finite?

Two things need checking and they need it differently.  The **transfer
functions** are claims about arithmetic, so they are swept against the
interpreter: every combination of operand classes, every sample value in each,
and the observed result must fall inside what the table predicted.  The
**refinement** is a claim about control flow, so it is checked by reading the
class the analysis gives a marked expression in each arm.
"""

import math

import pytest

import fpy2 as fp
import fpy2.strategies as st
from fpy2.analysis import ValueClass, ValueClassInfer, class_of, representable_classes
from fpy2.analysis.value_class import (
    ListClass,
    TupleClass,
    join_class,
    _ATOMS, _LOGB, _POW_BIG_BASE, _POW_ONE_BASE, _POW_SMALL_BASE,
    _exact_add, _exact_mul, _exact_select, _exact_sum,
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


@fp.fpy(ctx=fp.FP64)
def _pow2_fp64(a: fp.Real) -> fp.Real:
    return 2 ** a


@fp.fpy(ctx=fp.FP64)
def _pow_half_fp64(a: fp.Real) -> fp.Real:
    return 0.5 ** a


@fp.fpy(ctx=fp.FP64)
def _pow_one_fp64(a: fp.Real) -> fp.Real:
    return 1 ** a


@fp.fpy(ctx=fp.REAL)
def _sum_list(xs: list[fp.Real]) -> fp.Real:
    return sum(xs)


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

    def _sweep(self, predict, fn, arity: int, *, rows: int, only=None):
        bad = []
        pool = _ATOMS if only is None else only
        atoms = [(a,) for a in pool] if arity == 1 else [
            (a, b) for a in pool for b in pool]
        covered = set()
        for combo in atoms:
            want = predict(*combo)
            for vals in _value_combos(combo):
                try:
                    got = class_of(fn(*vals))
                except Exception:
                    continue        # no result: says nothing about the class
                covered.add(combo)
                if not (got & want):
                    bad.append(f'{fn.name}{vals}: {got} not in {want}')
        assert not bad, '; '.join(bad[:6])
        # an operation the interpreter refuses is skipped, and a table whose
        # rows are *all* refused would sweep green having compared nothing
        assert len(covered) == rows, (
            f'{fn.name} compared {len(covered)} of {rows} rows; the rest were '
            f'skipped, so those rows of the table are untested'
        )

    def test_add(self):
        self._sweep(_exact_add, _add, 2, rows=25)

    def test_sub(self):
        # its own table since the sign split: `_exact_add` sweeps against `-`
        # only while the infinities are one atom
        self._sweep(_exact_sub, _sub, 2, rows=25)

    def test_mul(self):
        self._sweep(_exact_mul, _mul, 2, rows=25)

    def test_max(self):
        self._sweep(lambda a, b: _exact_select([a, b], is_max=True), _max2, 2, rows=25)

    def test_min(self):
        self._sweep(lambda a, b: _exact_select([a, b], is_max=False), _min2, 2, rows=25)

    def test_sum(self):
        """`_exact_sum` predicts from what the *elements* are, so this sweeps
        lists drawn from one atom and from two.  The mixed lists are where the
        closure earns itself: an infinity and its opposite make a NaN that no
        element was, and two finites a zero.

        The empty list is in the sweep because it is the one case with no
        elements to predict from -- it sums to zero whatever the atom says.
        """
        bad = []
        covered = set()
        for a in _ATOMS:
            for b in _ATOMS:
                want = _exact_sum(a | b)
                for va in _SAMPLES[a][:3]:
                    for vb in _SAMPLES[b][:3]:
                        for xs in ([], [va], [va, vb], [va, vb, va]):
                            try:
                                got = class_of(_sum_list(xs))
                            except Exception:   # noqa: BLE001
                                continue        # no result: says nothing
                            if xs:
                                covered.add((a, b))   # `[]` says nothing
                            if not (got & want):
                                bad.append(f'sum({xs}): {got} not in {want}')
        assert not bad, '; '.join(bad[:6])
        assert len(covered) == len(_ATOMS) ** 2

    def test_the_rows_the_closure_widens(self):
        """The row the rule exists for: `ts` holding only finites makes
        `sum(ts)` finite, where reading the elements' own class off the list
        would have said nothing at all."""
        assert _exact_sum(ZERO | FINITE) == ZERO | FINITE
        assert _exact_sum(FINITE) == ZERO | FINITE
        assert _exact_sum(INF) == NAN | ZERO | INF
        assert _exact_sum(ValueClass(0)) == ZERO      # only the empty list

    def test_logb(self):
        self._sweep(lambda a: _map(_LOGB, a), _logb, 1, rows=5)

    def test_pow_with_a_base_above_one(self):
        """Two rows under ``REAL``: the interpreter has no exact ``2 ** x`` for
        a NaN or an infinity."""
        self._sweep(lambda a: _map(_POW_BIG_BASE, a), _pow2, 1, rows=2)

    @pytest.mark.parametrize('table,fn', [
        pytest.param(_POW_BIG_BASE, _pow2_fp64, id='2 ** x'),
        pytest.param(_POW_SMALL_BASE, _pow_half_fp64, id='0.5 ** x'),
        pytest.param(_POW_ONE_BASE, _pow_one_fp64, id='1 ** x'),
    ])
    def test_pow_at_a_concrete_context(self, table, fn):
        """The three rows ``REAL`` cannot reach, and the ones the base literal
        tells apart: ``2 ** -inf`` is ``+0`` where ``0.5 ** -inf`` is ``+inf``.
        Sound against a *rounded* run only for these -- their results are
        exactly representable, where the finite row overflows (``2 ** 1e300``)
        and the exact table rightly does not say so."""
        self._sweep(
            lambda a: _map(table, a), fn, 1, rows=3,
            only=(NAN, POS_INF, NEG_INF),
        )

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

    def test_an_inline_conditional_does_not_refine_its_branches(self):
        """A backend may evaluate both arms on every input, as format
        inference assumes too."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return fp.logb(x) if fp.isinf(x) else fp.fabs(x)

        @fp.fpy(ctx=fp.REAL)
        def g(x: fp.Real) -> fp.Real:
            return fp.logb(x) + fp.fabs(x)

        assert _cls(f, 'logb(x)') == _cls(g, 'logb(x)')
        assert _cls(f, 'abs(x)') == _cls(g, 'abs(x)')


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
            p = b > 0
            if p:
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


def _arg_types(n: int, lists: int, scalars: int) -> list:
    return [ListType(RealType(fp.FP32), n)] * lists + [RealType(fp.FP32)] * scalars


def _ref_classes(fn, n: int = 4, *, lists: int = 0, scalars: int = 0) -> list:
    """The class of every ``xs[i]`` read in *fn*, after lowering."""
    from fpy2.ast.fpyast import ListRef
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=_arg_types(n, lists, scalars))
    for spec in CppCompiler().specialize(m):
        if spec.ast.name != fn.name:
            continue
        info = ValueClassInfer.analyze(spec.ast)
        return [v for e, v in info.by_expr.items() if isinstance(e, ListRef)]
    raise AssertionError(f'no {fn.name}')


def _amax_class(fn, n: int = 4, *, lists: int = 1, scalars: int = 0) -> ValueClass:
    """The class of the ``max(...)`` over a list in *fn*, after lowering."""
    from fpy2.ast.fpyast import AMax
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=_arg_types(n, lists, scalars))
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
        """One location, two names."""
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
        """The predicate is inlined and the fold is a guarded assignment, so
        nothing here is the shape `ReduceFusion` emits."""
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
        """``ok`` is the *last* element's predicate, not every element's.  An
        ``and`` that does not carry the accumulator is still not a fold."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ok = True
            for x in xs:
                p = fp.isfinite(x)
                q = x != 0
                ok = p and q
            if ok:
                return max(xs)
            else:
                return 0.0

        assert _amax_class(f) == TOP

    def test_a_fold_rebuilt_each_round_says_nothing(self):
        """A guarded assignment is a fold only where what it guards on is what
        the loop carried in."""
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

    def test_a_scan_walked_again_does_not_speak_early(self):
        """A `for` inside a loop is walked more than once, and inside it the
        accumulator covers only the part scanned so far."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs, n: fp.Real) -> fp.Real:
            y = 0.0
            i = 0.0
            while i < n:
                ok = True
                for x in xs:
                    ok = ok and fp.isfinite(x)
                    if ok:
                        y = max(xs)
                i = i + 1.0
            return y

        assert _amax_class(f, scalars=1) == TOP
        assert math.isnan(f([1.0, float('nan')], 2.0))

    def test_a_store_inside_the_scan_voids_the_fact(self):
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


def _bounds(fn, n: int = 4, *, lists: int = 1, scalars: int = 0) -> dict:
    """``bound_of`` per definition, keyed by variable name, after lowering."""
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=_arg_types(n, lists, scalars))
    for spec in CppCompiler().specialize(m):
        if spec.ast.name != fn.name:
            continue
        info = ValueClassInfer.analyze(spec.ast)
        return {
            str(d.name): info.bound_of(d)
            for d in info.type_info.def_use.defs
        }
    raise AssertionError(f'no {fn.name}')


def _elt_classes(fn, n: int = 4, *, lists: int = 1, scalars: int = 0) -> dict:
    """``by_elt``, keyed by variable name, after lowering."""
    from fpy2.backend.cpp.compiler import CppCompiler
    m = fp.Module()
    m.add(fn, arg_types=_arg_types(n, lists, scalars))
    for spec in CppCompiler().specialize(m):
        if spec.ast.name != fn.name:
            continue
        info = ValueClassInfer.analyze(spec.ast)
        return {str(d.name): cls for d, cls in info.by_elt.items()}
    raise AssertionError(f'no {fn.name}')


class TestAStructuralClass:
    """A class shaped like the value, so an aggregate narrows piece by piece."""

    def test_a_tuple_joins_field_by_field(self):
        a = TupleClass((NAN, ZERO))
        b = TupleClass((ZERO, ZERO))
        assert join_class(a, b) == TupleClass((NAN | ZERO, ZERO))

    def test_a_list_definition_carries_its_elements(self):
        """The shape `StorageInfer` consumes: a list narrows through its
        element, not through a class of its own."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            return max(ys)

        b = _bounds(f)['ys']
        assert isinstance(b, ListClass) and not (b.elt & NEG_INF)

    def test_a_list_nothing_was_stored_into_is_bottom(self):
        """``fp.empty`` holds nothing and reading it is undefined, so no
        execution contradicts any class of its elements."""
        @fp.fpy(ctx=fp.REAL)
        def f(n: fp.Real) -> fp.Real:
            ys = fp.empty(3)
            ys[0] = 1.0
            return ys[0]

        assert _bounds(f, lists=0, scalars=1)['ys'] == ListClass(FINITE)

    def test_a_list_joins_its_element(self):
        assert join_class(ListClass(NAN), ListClass(ZERO)) == ListClass(NAN | ZERO)

    def test_a_shape_mismatch_knows_nothing(self):
        """Not the top class -- the top is a fact about a *number*, and there is
        no such fact about a value whose shape is in question."""
        assert join_class(TupleClass((NAN,)), NAN) is None
        assert join_class(TupleClass((NAN,)), TupleClass((NAN, ZERO))) is None

    def test_an_unknown_side_stays_unknown(self):
        assert join_class(None, NAN) is None
        assert join_class(TupleClass((NAN, None)), TupleClass((ZERO, ZERO))) == \
            TupleClass((NAN | ZERO, None))


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

    def test_a_store_after_the_read_still_counts(self):
        """Joined over the whole function, where ``by_expr`` is read at a
        point: a buffer holds every value that ever lands in it."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            r = max(ys)
            ys[0] = -fp.inf()
            return r + ys[1]

        assert _elt_classes(f)['ys'] & NEG_INF

    def test_a_parameter_is_absent(self):
        """Nothing was stored through it, so nothing is known -- and absent is
        how a consumer reads that."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            return max(xs)

        assert 'xs' not in _elt_classes(f)


class TestTheBackendSharesItsAliasAnalysis:
    """`ValueClassInfer` builds an `Alias` when none is passed, and that one has
    no escape summaries -- so every list handed to a call reads as escaping and
    loses its element facts.  A caller holding a summarized one has to pass it,
    or it pays for two analyses and uses the weaker."""

    def test_the_compiler_passes_the_alias_it_built(self):
        from fpy2.backend.cpp.compiler import CppCompiler

        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            ys = [abs(x) for x in xs]
            return max(ys)

        m = fp.Module()
        m.add(f, arg_types=[ListType(RealType(fp.FP32), 4)])
        compiler = CppCompiler()
        specs = compiler.specialize(m)
        for spec, analyses in compiler._analyze_all(specs, {}):
            if spec.ast.name == 'f':
                assert analyses.class_info.alias is analyses.alias
                return
        raise AssertionError('no f')


class TestARegionMayHoldMoreThanOneList:
    """Alias analysis merges two lists into one region as soon as anything
    makes them may-alias, and then a fact about "the" list names neither.

    Each of these reported a class a run contradicts.
    """

    def test_an_allocation_does_not_wipe_a_list_it_shares_a_region_with(self):
        @fp.fpy(ctx=fp.REAL)
        def f(n: fp.Real) -> fp.Real:
            a = fp.empty(2)
            a[0] = 1.0
            b = fp.empty(2)
            b[0] = fp.nan()
            xss = fp.empty(2)
            xss[0] = a
            xss[1] = b
            return a[0]

        assert _ref_classes(f, scalars=1) == [TOP]
        assert f(1.0) == 1.0

    def test_a_universal_needs_the_region_to_be_one_list(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs, ys, c: fp.Real):
            ok = all([fp.isfinite(x) for x in xs])
            zs = xs if c > 0 else ys
            if ok:
                return max(zs)
            else:
                return 0.0

        assert _amax_class(f, lists=2, scalars=1) == TOP


class TestElementsThatArrivedWithoutAStore:
    """`_stored` is seeded where a list is seen *empty*, so a list built any
    other way keeps the top class however much is stored into it afterwards.

    Each of these reported a class a run contradicts.
    """

    def test_a_parameter_was_filled_by_the_caller(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            xs[0] = 1.0
            return max(xs)

        assert _elt_classes(f)['xs'] == TOP
        assert math.isnan(f([5.0, float('nan'), 2.0]))

    def test_a_literal_carries_its_own_elements(self):
        @fp.fpy(ctx=fp.REAL)
        def f(n: fp.Real) -> fp.Real:
            xs = [1.0, fp.nan()]
            xs[0] = 2.0
            return xs[1]

        assert _elt_classes(f, lists=0, scalars=1)['xs'] == TOP
        assert math.isnan(f(1.0))

    def test_a_nested_store_reaches_the_list_it_writes(self):
        @fp.fpy(ctx=fp.REAL)
        def f(n: fp.Real) -> fp.Real:
            row = fp.empty(2)
            row[0] = 1.0
            xss = fp.empty(2)
            xss[0] = row
            xss[0][0] = fp.nan()
            return row[0]

        assert _elt_classes(f, lists=0, scalars=1)['row'] == TOP
        assert math.isnan(f(1.0))


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

    def test_a_list_handed_to_a_call_has_no_element_class(self):
        """A callee may store through it, so neither accessor answers."""
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
        assert _bounds(f)['zs'] is None

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


def _lower(fn, args: list):
    """*fn* specialized to *args* and lowered only as far as
    :class:`CompToLoop`."""
    return st.comp_to_loop(st.monomorphize(fn, args=args))


def _amax_unfused(fn, n: int = 4, *, arg_types: list | None = None) -> ValueClass:
    """The class of the ``max(...)`` in *fn*, lowered only as far as
    :class:`CompToLoop`.

    That leaves the guard as a materialised mask, where :func:`_amax_class`'s
    full pipeline would have made a fold of it.  A hand-written fold reaches
    this unchanged, which is what lets the two spellings be compared.
    """
    from fpy2.ast.fpyast import AMax
    low = _lower(fn, arg_types if arg_types is not None else _arg_types(n, 1, 0))
    info = ValueClassInfer.analyze(low.ast)
    return next(v for e, v in info.by_expr.items() if isinstance(e, AMax))


@fp.fpy(ctx=fp.REAL)
def _clobber_mask(m) -> fp.Real:
    m[0] = True
    return 0.0


class TestAMaterialisedGuard:
    """The same fact as :class:`TestAGuardOverAWholeList`, in the spelling
    `CompToLoop` leaves when `ReduceFusion` has not run: the predicate lands in
    a list, and the guard reads `all` of it.

    What has to be proved is the same either way -- the loop covers the list,
    and nothing has stored into it since.  The mask needs the second half too,
    which the fold does not: it is a *list*, so a store through another name
    for it is invisible to the reaching def the guard reads.
    """

    def test_a_mask_reaches_the_elements(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            if all([fp.isfinite(x) for x in xs]):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == ZERO | FINITE

    def test_an_existential_mask_reaches_the_arm_it_fails_in(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            if any([fp.isnan(x) for x in xs]):
                return 0.0
            else:
                return max(xs)

        assert not (_amax_unfused(f) & NAN)

    def test_the_other_arm_learns_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            if all([fp.isfinite(x) for x in xs]):
                return 0.0
            else:
                return max(xs)

        assert _amax_unfused(f) == TOP

    def test_a_mask_written_by_hand(self):
        """Nothing above is the shape `CompToLoop` mints, and this is not it
        either -- the predicate is inlined and the mask is allocated here."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(4):
                x = xs[i]
                m[i] = fp.isfinite(x)
            if all(m):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == ZERO | FINITE

    def test_a_scan_over_a_prefix_says_nothing(self):
        """`all(m)` forces every element of the mask, but only the first two
        say anything about `xs`."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(2):
                x = xs[i]
                m[i] = fp.isfinite(x)
            if all(m):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == TOP

    def test_a_store_between_the_scan_and_the_guard_says_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(4):
                x = xs[i]
                m[i] = fp.isfinite(x)
            xs[0] = fp.nan()
            if all(m):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == TOP

    def test_an_unbound_element_says_nothing(self):
        """The refinement travels through the definition the predicate reads,
        and testing the read in place gives it none.  A limitation, not a
        soundness condition: every lowering binds the element."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(4):
                m[i] = fp.isfinite(xs[i])
            if all(m):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == TOP

    def test_a_guard_inside_the_scan_says_nothing(self):
        """`all(m)` on round `i` covers the rounds before it, not the list.
        The fold bails here because `_scanned` is dropped before the body; the
        mask needs `_scan_clocks` dropped for the same reason."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for j in range(4):
                m[j] = True         # so the guard passes on round 0
            r = 0.0
            for i in range(4):
                if all(m):
                    r = max(xs)
                x = xs[i]
                m[i] = fp.isfinite(x)
            return r

        assert _amax_unfused(f) == TOP
        assert math.isinf(f([float('inf'), 1.0, 1.0, 1.0]))

    def test_a_store_through_another_name_for_the_mask_says_nothing(self):
        """A direct `m[0] = True` redefines `m`, so the guard no longer reads
        the loop's phi and this never arises.  Through an alias it does."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(4):
                x = xs[i]
                m[i] = fp.isfinite(x)
            n = m
            n[0] = True
            if all(m):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == TOP
        assert math.isinf(f([float('inf'), 1.0, 1.0, 1.0]))

    def test_a_mask_handed_to_a_callee_says_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(4):
                x = xs[i]
                m[i] = fp.isfinite(x)
            t = _clobber_mask(m)
            if all(m):
                return max(xs)
            else:
                return t

        assert _amax_unfused(f) == TOP
        assert math.isinf(f([float('inf'), 1.0, 1.0, 1.0]))

    def test_a_second_write_earlier_in_the_round_says_nothing(self):
        """Only the mask's last definition in the body is read, so a store
        before it is invisible -- and one at the top of round `k` undoes round
        `k - 1`."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            m = fp.empty(4)
            for i in range(4):
                m[0] = True
                x = xs[i]
                m[i] = fp.isfinite(x)
            if all(m):
                return max(xs)
            else:
                return 0.0

        assert _amax_unfused(f) == TOP
        assert math.isinf(f([float('inf'), 1.0, 1.0, 1.0]))


_MATRIX = [ListType(ListType(RealType(fp.FP32), 4), 2)]


class TestAGuardOverARow:
    """The rows of one nested list share a region *and* an allocation site --
    so counting sites is not what "a single list" means, and a fact proved
    about one row, kept under the region, would land on every other.  It is
    kept under the definition naming the row instead, which is one list.

    Neither spelling is special here; the predicate is.
    """

    def test_a_fold_over_a_row_says_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xss):
            r0 = xss[0]
            r1 = xss[1]
            ok = True
            for x in r0:
                ok = ok and fp.isfinite(x)
            if ok:
                return max(r1)
            else:
                return 0.0

        assert _amax_unfused(f, arg_types=_MATRIX) == TOP
        assert math.isinf(f([[1.0, 2.0, 3.0, 4.0], [float('inf'), 0.0, 0.0, 0.0]]))

    def test_a_mask_over_a_row_says_nothing(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xss):
            r0 = xss[0]
            r1 = xss[1]
            m = fp.empty(4)
            for i in range(4):
                x = r0[i]
                m[i] = fp.isfinite(x)
            if all(m):
                return max(r1)
            else:
                return 0.0

        assert _amax_unfused(f, arg_types=_MATRIX) == TOP
        assert math.isinf(f([[1.0, 2.0, 3.0, 4.0], [float('inf'), 0.0, 0.0, 0.0]]))

    def test_the_row_the_guard_scanned_learns_it(self):
        """`r0` names one list, so what the scan says of its elements holds
        of them until a store into the rows' region."""
        @fp.fpy(ctx=fp.REAL)
        def f(xss):
            r0 = xss[0]
            m = fp.empty(4)
            for i in range(4):
                x = r0[i]
                m[i] = fp.isfinite(x)
            if all(m):
                return max(r0)
            else:
                return 0.0

        assert _amax_unfused(f, arg_types=_MATRIX) == ZERO | FINITE

    def test_a_store_into_another_row_voids_it(self):
        """A store through another name into the rows' region may be into
        `r0`, which is the stamp's to catch."""
        @fp.fpy(ctx=fp.REAL)
        def f(xss):
            r0 = xss[0]
            r1 = xss[1]
            m = fp.empty(4)
            for i in range(4):
                x = r0[i]
                m[i] = fp.isfinite(x)
            if all(m):
                r1[0] = fp.inf()
                return max(r0)
            else:
                return 0.0

        assert _amax_unfused(f, arg_types=_MATRIX) == TOP


#####################################################################
# Finiteness facts


@fp.fpy(ctx=fp.REAL)
def _literal_any_guard(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
    p0 = a * b
    t0 = not fp.isfinite(p0)
    if any([t0]) or not fp.isfinite(c):
        r = 0
    else:
        r = p0 * 3
    return r


@fp.fpy(ctx=fp.REAL)
def _backward_guard(a: fp.Real, b: fp.Real) -> fp.Real:
    p0 = a * b
    if not fp.isfinite(p0):
        r = 0
    else:
        r = a * 3
    return r


class TestFinitenessFacts:
    """A finiteness guard refines what it tests, and a finite product its factors."""

    def test_a_literal_any_guard_refines_what_it_tests(self):
        """`any` over a literal of named tests."""
        info = ValueClassInfer.analyze(_literal_any_guard.ast)
        read = _find(_literal_any_guard.ast, '(p0 * 3)').first
        assert info.classify(read) & (NAN | INF) == ValueClass(0)

    def test_a_finite_product_has_finite_factors(self):
        info = ValueClassInfer.analyze(_backward_guard.ast)
        read = _find(_backward_guard.ast, '(a * 3)').first
        assert info.classify(read) & (NAN | INF) == ValueClass(0)


class TestALiteralGuard:
    """`all` / `any` over a literal is the `and` / `or` of its elements."""

    def test_all_true_refines_each(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            if all([fp.isfinite(a), fp.isfinite(b)]):
                r = a * b
            else:
                r = 0
            return r

        assert _cls(f, '(a * b)') & (NAN | INF) == ValueClass(0)

    def test_any_true_refines_nothing(self):
        """A disjunction: either test may be the one that held."""
        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            if any([fp.isnan(a), fp.isnan(b)]):
                r = a * 3
            else:
                r = 0
            return r

        assert _cls(f, '(a * 3)') & NAN

    def test_all_over_some_says_nothing_of_the_rest(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            if all([fp.isfinite(a)]):
                r = b * 3
            else:
                r = 0
            return r

        assert _cls(f, '(b * 3)') & NAN


@fp.fpy(ctx=fp.REAL)
def _guard_all(xs, c):
    if all([fp.isfinite(x) for x in xs]):
        r = xs[0] * 3
    else:
        r = 0
    return r


@fp.fpy(ctx=fp.REAL)
def _guard_any(xs, c):
    if any([not fp.isfinite(x) for x in xs]) or not fp.isfinite(c):
        r = 0
    else:
        r = xs[0] * c
    return r


@fp.fpy(ctx=fp.REAL)
def _guard_not_any(xs, c):
    if not any([fp.isnan(x) for x in xs]):
        r = xs[1] * 3
    else:
        r = 0
    return r


_FIN = ZERO | FINITE


def _finite_after(p_of, ctx=fp.REAL):
    """The class of `a` where `p_of(a, b, c)`, computed under *ctx*, is
    known finite."""
    @fp.fpy(ctx=fp.REAL)
    def f(a: fp.Real, b: fp.Real, c: fp.Real) -> fp.Real:
        with ctx:
            p = p_of(a, b, c)
        if fp.isfinite(p):
            r = a * 3
        else:
            r = 0
        return r

    from fpy2.transform import FuncInline
    ast = FuncInline.apply(f.ast)
    info = ValueClassInfer.analyze(ast)
    return info.classify(_find(ast, '(a * 3)').first)


@fp.fpy
def _p_add(a, b, c):
    return a + b


@fp.fpy
def _p_sub(a, b, c):
    return b - a


@fp.fpy
def _p_mul(a, b, c):
    return a * b


@fp.fpy
def _p_neg(a, b, c):
    return -a


@fp.fpy
def _p_abs(a, b, c):
    return abs(a)


@fp.fpy
def _p_fma(a, b, c):
    return fp.fma(b, c, a)


@fp.fpy
def _p_num(a, b, c):
    return a / b


@fp.fpy
def _p_den(a, b, c):
    return b / a


@fp.fpy
def _p_round(a, b, c):
    return fp.round(a)


@fp.fpy
def _p_chain(a, b, c):
    q = a * b
    return q + c


class TestBackwardRefinement:
    """A finite result says its operands were finite, through the names they
    were computed from."""

    @pytest.mark.parametrize(
        'p_of', [_p_add, _p_sub, _p_mul, _p_neg, _p_abs, _p_fma, _p_num,
                 _p_round, _p_chain], ids=lambda f: f.name)
    def test_an_exact_operation(self, p_of):
        assert _finite_after(p_of) & (NAN | INF) == ValueClass(0)

    def test_not_a_denominator(self):
        """`b / inf` is zero."""
        assert _finite_after(_p_den) & INF

    def test_through_a_rounding_that_keeps_infinities(self):
        assert _finite_after(_p_mul, fp.FP32) & (NAN | INF) == ValueClass(0)

    def test_not_through_one_that_saturates(self):
        """`MX_E2M1` rounds an infinity, and a NaN, to 6."""
        assert _finite_after(_p_mul, fp.MX_E2M1) & INF

    def test_not_from_a_non_finite_result(self):
        @fp.fpy(ctx=fp.REAL)
        def f(a: fp.Real, b: fp.Real) -> fp.Real:
            p = a * b
            if fp.isinf(p):
                r = a * 3
            else:
                r = 0
            return r

        assert _cls(f, '(a * 3)') & NAN

    def test_which_contexts_keep_a_non_finite_value(self):
        from fpy2.analysis.value_class import keeps_non_finite
        assert keeps_non_finite(fp.FP32)
        assert keeps_non_finite(fp.MX_E4M3)      # an infinity becomes a NaN
        assert not keeps_non_finite(fp.MX_E2M1)  # saturates
        assert not keeps_non_finite(fp.SINT8)    # refuses


class TestEitherDisjunct:
    """One of several tests holding says something of a definition all of
    them test: it is in the union of what they say."""

    def test_an_or_of_one_value(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isnan(x) or fp.isinf(x):
                r = x * 3
            else:
                r = 0
            return r

        assert _cls(f, '(x * 3)') & (ZERO | FINITE) == ValueClass(0)

    def test_not_an_or_of_two(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            if fp.isnan(x) or fp.isinf(y):
                r = x * 3
            else:
                r = 0
            return r

        assert _cls(f, '(x * 3)') & FINITE

    def test_an_and_that_fails(self):
        """Either conjunct is false, and each says `x` is not finite."""
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isfinite(x) and not fp.isinf(x):
                r = 0
            else:
                r = x * 3
            return r

        assert _cls(f, '(x * 3)') & (ZERO | FINITE) == ValueClass(0)


def _typed_cls(f, text: str, arg_types) -> ValueClass:
    from fpy2.transform import Monomorphize
    ast = Monomorphize.apply(f.ast, fp.REAL, arg_types)
    return ValueClassInfer.analyze(ast).classify(_find(ast, text))


_L4 = ListType(RealType(fp.FP32), 4)


@fp.fpy(ctx=fp.REAL)
def _after_an_early_return(x):
    if not fp.isfinite(x):
        return 0
    return x * 3


@fp.fpy(ctx=fp.REAL)
def _a_mask_through_a_ladder(prods, c):
    m = fp.empty(4)
    for i in range(4):
        p = prods[i]
        m[i] = not fp.isfinite(p)
    t = any(m)
    if not t:
        t = not fp.isfinite(c)
    if t:
        r = 0
    else:
        r = prods[1] * 3
    return r


@fp.fpy(ctx=fp.REAL)
def _back_through_a_fill(A, B):
    prods = fp.empty(4)
    for i in range(4):
        prods[i] = A[i] * B[i]
    m = fp.empty(4)
    for i in range(4):
        p = prods[i]
        m[i] = not fp.isfinite(p)
    if any(m):
        r = 0
    else:
        r = A[1] * 3
    return r


class TestEarlyExitAndMasks:
    """Refinement past an early exit, and through a mask a loop fills."""

    def test_after_an_early_return(self):
        assert _typed_cls(_after_an_early_return, '(x * 3)',
                          [RealType(fp.FP32)]) & (NAN | INF) == ValueClass(0)

    def test_after_an_else_that_returns(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            if fp.isfinite(x):
                y = x
            else:
                return 0
            return x * 3

        assert _cls(f, '(x * 3)') & (NAN | INF) == ValueClass(0)

    def test_not_an_arm_that_returns_on_one_path(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, c: fp.Real) -> fp.Real:
            if not fp.isfinite(x):
                if c > 0:
                    return 0
            return x * 3

        assert _cls(f, '(x * 3)') & NAN

    def test_only_the_rest_of_its_block(self):
        """Inside a loop the return ends the call, but the refinement is read
        within the block; what follows the loop is not refined."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs, c):
            s = 0
            for x in xs:
                if not fp.isfinite(c):
                    return 0
                s = c * 3
            return c * 5

        L = ListType(RealType(fp.FP32), 2)
        R = RealType(fp.FP32)
        assert _typed_cls(f, '(c * 3)', [L, R]) & (NAN | INF) == ValueClass(0)
        assert _typed_cls(f, '(c * 5)', [L, R]) & NAN

    def test_a_mask_through_a_lowered_or(self):
        assert _typed_cls(_a_mask_through_a_ladder, '(prods[1] * 3)',
                          [_L4, RealType(fp.FP32)]) & (NAN | INF) == ValueClass(0)

    def test_back_through_a_fill(self):
        assert _typed_cls(_back_through_a_fill, '(A[1] * 3)',
                          [_L4, _L4]) & (NAN | INF) == ValueClass(0)


class TestBackThroughAFill:
    """Every element of a list finite, where one covering loop stored an exact
    operation of reads at its index, makes every element of those lists
    finite."""

    @staticmethod
    def _class_of_a(fill, *, n: int = 4, ctx=fp.REAL) -> ValueClass:
        @fp.fpy(ctx=fp.REAL)
        def f(A, B):
            prods = fp.empty(4)
            for i in range(n):
                with ctx:
                    prods[i] = fill(A[i], B[i], 0)
            m = fp.empty(4)
            for i in range(4):
                p = prods[i]
                m[i] = not fp.isfinite(p)
            if any(m):
                r = 0
            else:
                r = A[1] * 3
            return r

        from fpy2.transform import ConstFold, FreeVarElim, FuncInline, Monomorphize
        # as the backends do: `n` and `ctx` are closure values
        ast = Monomorphize.apply(f.ast, fp.REAL, [_L4, _L4])
        ast = FuncInline.apply(ConstFold.apply(FreeVarElim.apply(ast)))
        return ValueClassInfer.analyze(ast).classify(_find(ast, '(A[1] * 3)'))

    def test_a_sum(self):
        assert self._class_of_a(_p_add) & (NAN | INF) == ValueClass(0)

    def test_a_rounded_product(self):
        assert self._class_of_a(_p_mul, ctx=fp.FP32) & (NAN | INF) == ValueClass(0)

    def test_not_one_that_saturates(self):
        assert self._class_of_a(_p_mul, ctx=fp.MX_E2M1) & INF

    def test_not_a_denominator(self):
        assert self._class_of_a(_p_den) & INF

    def test_not_a_partial_fill(self):
        assert self._class_of_a(_p_mul, n=3) & INF

    def test_not_after_a_store_into_the_source(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A, B, c):
            prods = fp.empty(4)
            for i in range(4):
                prods[i] = A[i] * B[i]
            A[1] = c
            m = fp.empty(4)
            for i in range(4):
                p = prods[i]
                m[i] = not fp.isfinite(p)
            if any(m):
                r = 0
            else:
                r = A[1] * 3
            return r

        assert _typed_cls(f, '(A[1] * 3)', [_L4, _L4, RealType(fp.FP32)]) & INF

    def test_not_a_fill_that_may_not_run(self):
        @fp.fpy(ctx=fp.REAL)
        def under_if(A, B, c):
            prods = [0.0, 0.0, 0.0, 0.0]
            if c > 0:
                for i in range(4):
                    prods[i] = A[i] * B[i]
            m = fp.empty(4)
            for i in range(4):
                p = prods[i]
                m[i] = not fp.isfinite(p)
            if any(m):
                r = 0
            else:
                r = A[1] * 3
            return r

        @fp.fpy(ctx=fp.REAL)
        def under_loop(A, B, k):
            prods = [0.0, 0.0, 0.0, 0.0]
            for _ in range(k):
                for i in range(4):
                    prods[i] = A[i] * B[i]
            m = fp.empty(4)
            for i in range(4):
                p = prods[i]
                m[i] = not fp.isfinite(p)
            if any(m):
                r = 0
            else:
                r = A[1] * 3
            return r

        for f in (under_if, under_loop):
            assert _typed_cls(f, '(A[1] * 3)', [_L4, _L4, RealType(fp.FP32)]) & INF

    def test_a_fill_per_iteration(self):
        @fp.fpy(ctx=fp.REAL)
        def f(A, B, k):
            r = 0.0
            for _ in range(k):
                prods = fp.empty(4)
                for i in range(4):
                    prods[i] = A[i] * B[i]
                m = fp.empty(4)
                for i in range(4):
                    p = prods[i]
                    m[i] = not fp.isfinite(p)
                if any(m):
                    r = 0
                else:
                    r = A[1] * 3
            return r

        cls = _typed_cls(f, '(A[1] * 3)', [_L4, _L4, RealType(fp.FP32)])
        assert cls & (NAN | INF) == ValueClass(0)

    def test_not_a_second_store(self):
        """Inside the fill, so the stamp at its exit does not show it."""
        @fp.fpy(ctx=fp.REAL)
        def f(A, B, c):
            prods = fp.empty(4)
            for i in range(4):
                prods[i] = A[i] * B[i]
                prods[0] = c
            m = fp.empty(4)
            for i in range(4):
                p = prods[i]
                m[i] = not fp.isfinite(p)
            if any(m):
                r = 0
            else:
                r = A[1] * 3
            return r

        assert _typed_cls(f, '(A[1] * 3)', [_L4, _L4, RealType(fp.FP32)]) & INF


#####################################################################
# What a finite value says of what it was computed from

_F32 = RealType(fp.FP32)


def _lowered_cls(fn, text: str, args: list) -> ValueClass:
    """:func:`_cls`, after :class:`CompToLoop`, which leaves a list filled by a
    loop and a guard over a mask."""
    low = _lower(fn, args)
    return ValueClassInfer.analyze(low.ast).classify(_find(low.ast, text))


@fp.fpy(ctx=fp.REAL)
def _scaled_fill(xs, ys, s):
    ps = [(xs[i] * ys[i]) * s for i in range(len(xs))]
    if all([fp.isfinite(p) for p in ps]):  # noqa: C419
        return (s * 2) + max(xs)
    return 0.0


class TestFiniteSources:
    def test_through_a_join_only_one_side_can_be_finite(self):
        """An overflow lowered to a branch: the other arm is an infinity, so a
        finite join is the product, and its operands are finite."""
        @fp.fpy(ctx=fp.REAL)
        def f(a, b):
            p = a * b
            if abs(p) >= 1e30:
                r = fp.inf()
            else:
                r = p
            if fp.isfinite(r):
                return a * 2
            return 0.0

        assert _lowered_cls(f, '(a * 2)', [_F32, _F32]) == ZERO | FINITE

    def test_a_nested_fill_speaks_for_its_lists_and_scalars(self):
        assert _lowered_cls(_scaled_fill, 'max(xs)', [_L4, _L4, _F32]) == ZERO | FINITE
        assert _lowered_cls(_scaled_fill, '(s * 2)', [_L4, _L4, _F32]) == ZERO | FINITE

    def test_an_empty_fill_says_nothing_of_a_scalar(self):
        """No element, no product: `s` was never multiplied."""
        empty = ListType(_F32, 0)
        assert _lowered_cls(_scaled_fill, '(s * 2)', [empty, empty, _F32]) == TOP

    def test_a_scalar_the_loop_rebinds_is_not_the_one_it_stored(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs, t):
            s = t
            ps = fp.empty(4)
            for i in range(4):
                ps[i] = xs[i] * s
                if i == 3:
                    s = fp.inf()
            if all([fp.isfinite(p) for p in ps]):  # noqa: C419
                return s * 2
            return 0.0

        assert _lowered_cls(f, '(s * 2)', [_L4, _F32]) == TOP
        assert math.isinf(f([1.0, 2.0, 3.0, 4.0], 1.0))

    def test_a_list_parameter_is_in_its_format(self):
        """E4M3 has no infinity, so neither has an element of an E4M3 list,
        even past a loop that stores elsewhere."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            for _i in range(4):
                m = fp.empty(4)
                m[0] = 1.0
            return max(xs)

        e4m3 = ListType(RealType(fp.MX_E4M3), 4)
        assert _lowered_cls(f, 'max(xs)', [e4m3]) == NAN | ZERO | FINITE

    def test_the_second_rung_of_a_guard_scanned_inside_an_arm(self):
        """`ys` is scanned inside the arm; the join voids what the arm said,
        but nothing stored into the mask after its scan."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs, ys):
            bad = any([fp.isnan(x) for x in xs])  # noqa: C419
            if not bad:
                bad = any([fp.isnan(y) for y in ys])  # noqa: C419
            if bad:
                return 0.0
            else:
                return max(ys)

        assert not _lowered_cls(f, 'max(ys)', [_L4, _L4]) & NAN

    def test_a_loop_phi_is_not_one_side(self):
        """`r` finite at the guard says the *last* round's `a` was, under a
        definition this round's `a` shares."""
        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            r = fp.inf()
            out = 0.0
            for i in range(4):
                a = xs[i]
                if fp.isfinite(r):
                    out = a * 2
                r = a * 3
            return out

        assert _lowered_cls(f, '(a * 2)', [_L4]) == TOP
        assert math.isinf(f([1.0, 2.0, 3.0, math.inf]))

    def test_a_scalar_read_through_a_name_the_loop_rebinds(self):
        @fp.fpy(ctx=fp.REAL)
        def f(xs, t):
            s = t
            ps = fp.empty(4)
            for i in range(4):
                u = s
                ps[i] = xs[i] * u
                if i == 3:
                    s = fp.inf()
            if all([fp.isfinite(p) for p in ps]):  # noqa: C419
                return s * 2
            return 0.0

        assert _lowered_cls(f, '(s * 2)', [_L4, _F32]) == TOP
        assert math.isinf(f([1.0, 2.0, 3.0, 4.0], 1.0))
