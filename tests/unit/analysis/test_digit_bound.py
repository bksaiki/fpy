"""
Unit tests for the digit-bound constraint store and the pass that fills it.

The first classes exercise the store directly, on hand-built constraint sets;
the later ones run :class:`DigitBoundInfer` over an AST.
"""

import math

import pytest

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.digit_bound import (
    DigitBoundInfer,
    DigitBoundStore,
    Term,
    Z3Solver,
)
from fpy2.strategies import monomorphize
from fpy2.transform import CompToLoop
from fpy2.types import ListType, RealType


class TestTerm:
    """Affine arithmetic.  ``==`` is structural, never a constraint."""

    def test_arithmetic_collects_coefficients(self):
        s = DigitBoundStore()
        a, b = s.var('a'), s.var('b')
        assert a + a == a * 2
        assert a + b - b == a
        assert (a + 1) - 1 == a
        assert 3 - a == -a + 3

    def test_a_cancelled_variable_leaves_no_coefficient(self):
        s = DigitBoundStore()
        a = s.var('a')
        assert (a - a).coeffs == ()
        assert (a - a).const == 0

    def test_ordering_is_deterministic(self):
        """Terms order by creation index, not by ``id()``."""
        s = DigitBoundStore()
        a, b = s.var('a'), s.var('b')
        assert (b + a).coeffs == (a + b).coeffs


class TestQueries:

    def test_empty_store_is_unbounded(self):
        s = DigitBoundStore()
        assert s.maximum(s.var('x')) == math.inf

    def test_a_constant_term_needs_no_constraints(self):
        s = DigitBoundStore()
        a = s.var('a')
        assert s.maximum(a - a + 7) == 7

    def test_unsatisfiable_is_bottom(self):
        s = DigitBoundStore()
        a = s.var('a')
        s.le(a, 3)
        s.ge(a, 5)
        assert s.maximum(a) == -math.inf

    def test_precision_is_the_span(self):
        s = DigitBoundStore()
        l, g = s.var('l'), s.var('g')
        s.le(l, 15)
        s.ge(g, l - 10)      # an FP16 value: 11 significand bits
        assert s.prec(l, g) == 11


class TestFusedSum:
    """``nv.t_fdpa`` at ``F = 24``: the store the emission rules build."""

    @staticmethod
    def _store(F=24):
        s = DigitBoundStore()
        v = {n: s.var(n) for n in
             ('la', 'lb', 'lc', 'ea', 'eb', 'e_c', 'es', 'e_max', 'n', 'lp')}
        s.le(v['la'], 15)                             # exact_logb on FP16
        s.le(v['lb'], 15)
        s.le(v['lc'], 127)                            # ... and on FP32
        s.le(v['la'], v['ea'])                        # exponent(x,emin) >= logb x
        s.le(v['lb'], v['eb'])
        s.le(v['lc'], v['e_c'])
        s.le(v['lp'], v['la'] + v['lb'] + 1)          # logb(a*b) <= la + lb + 1
        s.eq(v['es'], v['ea'] + v['eb'])              # the comprehension body
        s.ge(v['e_max'], v['es'])                     # max(max(es), e_c)
        s.ge(v['e_max'], v['e_c'])
        s.eq(v['n'], v['e_max'] - F - 1)              # the call-site argument
        return s, v

    def test_product_summand_is_F_plus_2(self):
        s, v = self._store(F=24)
        assert s.prec(v['lp'], v['n'] + 1) == 26

    def test_the_accumulator_is_F_plus_1(self):
        """`c` is bounded by its own `e_c <= e_max`, so it reaches one binade
        less far than a product does."""
        s, v = self._store(F=24)
        assert s.prec(v['lc'], v['n'] + 1) == 25

    def test_the_rounding_mode_decides_the_carry(self):
        s, v = self._store(F=24)
        assert (s.prec(v['lp'], v['n'] + 1) + 1) == 27

    def test_F_is_symbolic_to_the_store(self):
        for F in (13, 24, 35):
            s, v = self._store(F)
            assert s.prec(v['lp'], v['n'] + 1) == F + 2

    def test_dropping_the_max_ordering_loses_everything(self):
        """The two `max` facts are the whole content: without them every other
        bound is still tight and the answer is still unbounded."""
        s = DigitBoundStore()
        v = {n: s.var(n) for n in ('la', 'lb', 'ea', 'eb', 'es', 'e_max', 'n', 'lp')}
        s.le(v['la'], 15)
        s.le(v['lb'], 15)
        s.le(v['la'], v['ea'])
        s.le(v['lb'], v['eb'])
        s.le(v['lp'], v['la'] + v['lb'] + 1)
        s.eq(v['es'], v['ea'] + v['eb'])
        s.eq(v['n'], v['e_max'] - 24 - 1)
        assert s.prec(v['lp'], v['n'] + 1) == math.inf


class TestSelfAnchored:
    """``with fp.MPFixedContext(logb(x) - k): y = fp.round(x)``.

    Measured on `FP32`: precision 12 at `k = 12`, and 13 under the default
    `RM.RNE`, whose carry reaches `2**(e+1)` exactly.
    """

    @pytest.mark.parametrize('k', [0, 1, 5, 12, 23])
    def test_prec_is_k(self, k):
        s = DigitBoundStore()
        lx, e, n = s.var('lx'), s.var('e'), s.var('n')
        s.le(lx, 127)
        s.eq(e, lx)                 # e = logb(x) names x's own exponent
        s.eq(n, e - k)
        assert s.prec(lx, n + 1) == k
        assert (s.prec(lx, n + 1) + 1) == k + 1


class TestPrecIsNeverNegative:
    """``msb - lsb + 1`` goes negative when the grid is coarser than the value's
    whole reach.  A count of digits cannot be, so the query floors at zero --
    and zero is below what a :class:`Format` admits, so a caller materializing
    one has to handle it.
    """

    @staticmethod
    def _store(k):
        s = DigitBoundStore()
        l, n = s.var('l'), s.var('n')
        s.le(l, 127)
        s.eq(n, l - k)
        return s, l, n

    def test_a_coarse_grid_floors_at_zero(self):
        for k in (0, -1, -5, -100):
            s, l, n = self._store(k)
            assert s.prec(l, n + 1) == 0, k

    def test_the_carry_still_applies_below_zero(self):
        """A mode that rounds away from zero reaches one quantum however
        coarse the grid is, so it keeps its digit."""
        for k in (0, -1, -5, -100):
            s, l, n = self._store(k)
            assert (s.prec(l, n + 1) + 1) == 1, k


class TestPrecAndPrecAt:
    """A context states the digit *below* its least significant one,
    so its grid is one higher.

    A context names the first *un*representable digit, so its grid sits one
    position higher -- an off-by-one worth having in exactly one place.
    """

    def test_they_agree(self):
        s = DigitBoundStore()
        l, n = s.var('l'), s.var('n')
        s.le(l, 100)
        s.eq(n, l - 7)
        assert s.prec(l, n + 1) == s.prec(l, n + 1) == 7


class TestDisjunctiveBounds:
    """``le_max`` and ``ge_min``: the shape every bound through a `max` takes.

    ``y <= max(p, q)`` is a disjunction, not anything affine, so the store needs
    a constraint form of its own for it.
    """

    def test_le_max_takes_the_larger(self):
        s = DigitBoundStore()
        a, b, y = s.var('a'), s.var('b'), s.var('y')
        s.le(a, 10)
        s.le(b, 3)
        s.le_max(y, [a, b])
        assert s.maximum(y) == 10

    def test_ge_min_takes_the_smaller(self):
        s = DigitBoundStore()
        a, b, g = s.var('a'), s.var('b'), s.var('g')
        s.ge(a, -5)
        s.ge(b, -20)
        s.ge_min(g, [a, b])
        assert -s.maximum(-g) == -20

    def test_one_operand_needs_no_disjunction(self):
        """A single alternative is an ordinary bound, and aligned summands
        sharing a term is exactly that case."""
        s = DigitBoundStore()
        a, y = s.var('a'), s.var('y')
        s.le(a, 7)
        s.le_max(y, [a])
        assert s.maximum(y) == 7

    def test_no_operands_state_nothing(self):
        s = DigitBoundStore()
        y = s.var('y')
        s.le_max(y, [])
        assert s.maximum(y) == math.inf


class TestBackendIsReplaceable:
    """The store must not be tied to z3 -- the point of the ``Solver``
    interface is that the backend choice stays reversible."""

    def test_a_stub_solver_is_honoured(self):
        class Stub:
            def assume(self, constraint): pass
            def maximize(self, objective, cutoff=None):
                return 5

        s = DigitBoundStore(solver=Stub())
        x = s.var('x')
        s.le(x, 3)   # constrain it, or the store answers `inf` without asking
        assert s.maximum(x) == 5

    def test_an_undecided_backend_degrades_to_unbounded(self):
        """`unknown` must read as the unbounded answer, which is what the
        analysis reports without a store at all -- never an exception."""
        class Undecided:
            def assume(self, constraint): pass
            def maximize(self, objective, cutoff=None):
                return math.inf

        s = DigitBoundStore(solver=Undecided())
        x = s.var('x')
        s.le(x, 3)
        assert s.maximum(x) == math.inf

    def test_a_free_variable_needs_no_backend(self):
        """Nothing constrains it, so it is unbounded by inspection -- asking
        a solver could only reach the same answer more slowly."""
        class Exploding:
            def assume(self, constraint): pass
            def maximize(self, objective, cutoff=None):
                raise AssertionError('should not have been asked')

        s = DigitBoundStore(solver=Exploding())
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)
        assert s.maximum(x) == math.inf
        assert s.maximum(-x) == math.inf
        assert s.maximum(x + y) == math.inf

    def test_every_constraint_reaches_the_backend(self):
        """The store states each constraint once, as it is made: a backend
        that only saw them at query time would have to be re-told the system
        on every objective."""
        seen = []

        class Recording:
            def assume(self, constraint):
                seen.append(constraint)
            def maximize(self, objective, cutoff=None):
                return 0

        s = DigitBoundStore(solver=Recording())
        x, y = s.var('x'), s.var('y')
        s.le(x, 3)
        s.eq(y, x)
        s.le_max(x, [y, 4])
        assert len(seen) == 3

    def test_a_backend_may_ignore_the_cutoff(self):
        """`cutoff` lets a backend stop early; the exact maximum is an upper
        bound whatever the caller asked for, so ignoring it stays correct."""
        asked = []

        class Exact:
            def assume(self, constraint): pass
            def maximize(self, objective, cutoff=None):
                asked.append(cutoff)
                return 3

        s = DigitBoundStore(solver=Exact())
        x = s.var('x')
        s.le(x, 3)
        assert s.reaches([(x, 3)]) is True
        assert s.reaches([(x, 4)]) is False
        assert asked == [3, 4]      # the threshold reached the backend

    def test_a_zero_timeout_is_not_an_error(self):
        s = DigitBoundStore(solver=Z3Solver(timeout_ms=1))
        x = s.var('x')
        s.le(x, 3)
        assert s.maximum(x) in (3, math.inf)


class TestReachesIsADecision:
    """`maximum` is an optimisation; where the answer is only compared
    against a threshold, one refutation settles it instead."""

    def test_it_agrees_with_the_maximum(self):
        s = DigitBoundStore()
        x, y = s.var('x'), s.var('y')
        s.le(x, 3)
        s.le(y, -2)
        for term, k in ((x, 3), (x, 4), (y, -2), (y, -1), (x + y, 1), (x + y, 2)):
            assert s.reaches([(term, k)]) == (s.maximum(term) >= k)

    def test_any_of_them(self):
        s = DigitBoundStore()
        x, y = s.var('x'), s.var('y')
        s.le(x, 3)
        s.le(y, 9)
        assert s.reaches([(x, 4), (y, 4)]) is True     # the second reaches
        assert s.reaches([(x, 4), (y, 10)]) is False
        assert s.reaches([]) is False

    def test_a_free_variable_reaches_anything(self):
        class Exploding:
            def assume(self, constraint): pass
            def maximize(self, objective, cutoff=None):
                raise AssertionError('should not have been asked')

        s = DigitBoundStore(solver=Exploding())
        x = s.var('x')
        assert s.reaches([(x, 10**9)]) is True


class TestUnboundedIsNotGuessed:
    """The upward probe needs somewhere to stop; where it stops must only ever
    loosen the answer, and a finite maximum must never be cut off."""

    def test_a_deeply_negative_maximum_is_still_exact(self):
        """It sits far below any plausible cutoff, and reporting it as
        unsatisfiable -- or as anything below itself -- would be unsound."""
        s = DigitBoundStore()
        x = s.var('x')
        s.le(x, -5_000_000)
        assert s.maximum(x) == -5_000_000

    def test_a_large_finite_maximum_is_still_exact(self):
        s = DigitBoundStore()
        x = s.var('x')
        s.le(x, 9_000_000)
        assert s.maximum(x) == 9_000_000

    def test_an_unbounded_objective_reads_as_inf(self):
        s = DigitBoundStore()
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)          # constrained, so the store does ask
        s.ge(x, y)          # x only bounded below
        assert s.maximum(x) == math.inf

    def test_an_unsatisfiable_store_reads_as_negative_inf(self):
        s = DigitBoundStore()
        x = s.var('x')
        s.le(x, 1)
        s.ge(x, 2)
        assert s.maximum(x) == -math.inf


class TestBisectionMatchesOptimization:
    """Bisection is the default because it is cheaper; it has to agree with
    `Optimize`, which is the reference."""

    CASES = (
        ('bounded above', lambda s, x, y: (s.le(x, 42), x)),
        ('deeply negative', lambda s, x, y: (s.le(x, -5_000_000), x)),
        ('large finite', lambda s, x, y: (s.le(x, 9_000_000), x)),
        ('through a chain', lambda s, x, y: ((s.le(y, 7), s.le(x, y)), x)),
        ('bounded below only', lambda s, x, y: ((s.le(y, 3), s.ge(x, y)), x)),
        ('over a max', lambda s, x, y: (
            (s.le(y, 3), s.le_max(x, [y, y + 5])), x)),
        ('unsatisfiable', lambda s, x, y: ((s.le(x, 1), s.ge(x, 2)), x)),
        # nothing restricts a coefficient to a unit, and the probe's ceiling
        # has to reach far enough for one that is not
        ('scaled objective', lambda s, x, y: (
            (s.le(x, 1), s.ge(x, 0)), x * 100)),
        ('scaled constraint', lambda s, x, y: (
            (s.le(x * 7, 70), s.ge(x, 0)), x)),
    )

    @pytest.mark.parametrize('name,build', CASES, ids=[c[0] for c in CASES])
    def test_both_backends_agree(self, name, build):
        answers = []
        for bisect in (True, False):
            s = DigitBoundStore(solver=Z3Solver(bisect=bisect))
            x, y = s.var('x'), s.var('y')
            _, obj = build(s, x, y)
            answers.append(s.maximum(obj))
        assert answers[0] == answers[1], f'{name}: {answers}'


class TestALoopCarriedScalarIsNotItsBody:
    """A loop's phi is the value an iteration *starts* from: the pre-loop value
    on the first pass, and on every pass if the loop never runs.  Giving it the
    body's terms claims neither."""

    def test_the_pre_loop_value_survives(self):

        @fp.fpy(ctx=fp.REAL)
        def f(c, xs):
            d = c
            for i in range(len(xs)):
                d = xs[i]       # never reads `d`, so nothing mints the phi
            return d

        g = monomorphize(
            f, args=[RealType(fp.FP64), ListType(RealType(fp.FP16), 4)],
        )
        b = DigitBoundInfer.analyze(g.ast, FormatInfer.analyze(g.ast))
        ret = b.by_expr[g.ast.body.stmts[-1].expr]
        # `xs` tops out at 2**15 and `c` at 2**1023; `d` is `c` on the way in
        assert b.store.maximum(ret.msb) == 1023


class TestAZeroArmStatesNoMagnitude:
    """A conditional's zero arm has no digits, so it neither raises the
    magnitude nor lowers the grid.  Joining it instead leaves the merge at its
    own seed, which spans the operand's whole reach."""

    def test_an_if_expression_takes_the_other_arm(self):

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            with fp.MPFixedContext(fp.logb(x) - 12, fp.RM.RTZ):
                r = fp.round(x)
            y = 0 if x == 0 else r
            return y + r

        g = monomorphize(f, args=[RealType(fp.FP32)])
        b = DigitBoundInfer.analyze(g.ast, FormatInfer.analyze(g.ast))
        got = {
            e.format(): b.store.prec(t.msb, t.lsb)
            for e, t in b.by_expr.items()
            if e.format() in ('(0 if x == 0 else r)', '(y + r)')
        }
        # `r` is 13 digits; the zero arm adds none, and the sum one more
        assert got == {'(0 if x == 0 else r)': 12, '(y + r)': 13}


@fp.fpy(ctx=fp.FP64)
def _ident(x: fp.Real) -> fp.Real:
    """A callee that hands back the caller's own term."""
    return x


class TestARefinedBoundStaysOnItsPath:
    """A guard narrows a value only where the guard holds.  Binding a
    definition to the expression that gave it its value must not carry that
    narrower bound back onto the expression, whose term every use shares."""

    @staticmethod
    def _max_logb_of_x(fn):

        g = monomorphize(fn, args=[RealType(fp.FP64)])
        b = DigitBoundInfer.analyze(g.ast, FormatInfer.analyze(g.ast))
        logb = next(t.msb for d, t in b.by_def.items()
                    if str(d.name) == 'x' and t.msb is not None)
        return b.store.maximum(logb)

    def test_an_assignment_does_not_export_the_guard(self):

        @fp.fpy(ctx=fp.FP64)
        def f(x):
            y = 0.0
            if x < 1:
                if x > -1:
                    y = x          # |x| <= 1 *here* only
            return x + y

        assert self._max_logb_of_x(f) == 1023   # x is an unrestricted FP64

    def test_a_call_returning_its_argument_does_not_either(self):

        ident = _ident

        @fp.fpy(ctx=fp.FP64)
        def f(x):
            y = 0.0
            if x < 1:
                if x > -1:
                    y = ident(x)   # the callee hands back the caller's term
            return x + y

        assert self._max_logb_of_x(f) == 1023


class TestRoundingAwayFromZeroLeavesTheBinade:
    """Every rule states the reach of the *exact* result, but an expression
    denotes that result rounded at its context -- and all but `trunc` can
    carry out of the binade the exact bound implies.

    The absolute numbers carry slack: a carry is charged at every operation
    under a rounding context, including one whose result is exact, and
    `x * exp2(-15)` collects two that way.  What the rule contributes is the
    *difference* between a carrying mode and `trunc`, which is what these
    assert.
    """

    @staticmethod
    def _max_logb(fn, text):

        g = monomorphize(fn, args=[RealType(fp.FP16), RealType(fp.FP16)])
        b = DigitBoundInfer.analyze(g.ast, FormatInfer.analyze(g.ast))
        t = next(t for e, t in b.by_expr.items()
                 if e.format() == text and t.msb is not None)
        return b.store.maximum(t.msb)

    def test_arithmetic_under_a_coarse_context_carries(self):

        @fp.fpy(ctx=fp.FP64)
        def f(x, y):
            a = x * fp.exp2(-15)          # a, b < 2
            b = y * fp.exp2(-15)
            with fp.MPFixedContext(0, fp.RM.RNE):   # quantum 2
                z = a + b
                p = abs(a)
            return z + p

        # at x = y = 57344 the operands are 1.75, bounded at 2: the sum
        # rounds a binade past its own exact bound and the absolute value
        # does not
        assert self._max_logb(f, '(x * fp.exp2(-15))') == 2
        assert self._max_logb(f, '(a + b)') == 4
        assert self._max_logb(f, 'abs(a)') == 3

    def test_ceil_leaves_the_binade_and_trunc_does_not(self):

        @fp.fpy(ctx=fp.FP64)
        def up(x, y):
            return fp.ceil(x * fp.exp2(-15)) + y * 0

        @fp.fpy(ctx=fp.FP64)
        def down(x, y):
            return fp.trunc(x * fp.exp2(-15)) + y * 0

        # the one binade between them is the whole claim
        assert self._max_logb(up, 'fp.ceil((x * fp.exp2(-15)))') == 4
        assert self._max_logb(down, 'fp.trunc((x * fp.exp2(-15)))') == 3


class TestAPartOfAListKeepsItsPairing:
    """`for i in range(a, b, s)` visits part of a list, so `xs[i]` is an
    element of that part.  Two parts taken over the same range stay paired --
    which is what lets a `max` over one bound the other -- and a part taken
    over a different range does not."""

    @staticmethod
    def _precs(fn):

        g = monomorphize(fn, args=[ListType(RealType(fp.FP32), 8)])
        # both forms: the comprehension and the gather loop it lowers to
        out = []
        for ast in (g.ast, CompToLoop.apply(g.ast)):
            b = DigitBoundInfer.analyze(ast, FormatInfer.analyze(ast))
            out.append(max(
                b.store.prec(t.msb, t.lsb)
                for e, t in b.by_expr.items()
                if e.format().startswith('fp.round')
            ))
        return out

    def test_a_max_over_the_evens_places_the_evens(self):

        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            n = len(xs)
            es = [fp.logb(xs[i]) for i in range(len(xs))]
            ev = max([es[i] for i in range(0, n, 2)])
            with fp.MPFixedContext(ev - 12, fp.RM.RTZ):
                ts = [fp.round(xs[i]) for i in range(0, n, 2)]
            return sum(ts)

        assert self._precs(f) == [12, 12]

    def test_and_says_nothing_about_the_odds(self):

        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            n = len(xs)
            es = [fp.logb(xs[i]) for i in range(len(xs))]
            ev = max([es[i] for i in range(0, n, 2)])
            with fp.MPFixedContext(ev - 12, fp.RM.RTZ):
                ts = [fp.round(xs[i]) for i in range(1, n, 2)]
            return sum(ts)

        # `ev` bounds no odd element, so the round spans fp32's whole reach
        assert self._precs(f) == [288, 288]


class TestReplayingAtAnIndexSet:
    """`instance` copies the facts that hold at every index onto fresh
    variables.  One naming anything else may be an aggregate over the whole
    list, true of the list and false of a part."""

    def test_an_element_fact_is_copied_and_an_aggregate_is_not(self):
        s = DigitBoundStore()
        elt, other, total = s.var('elt'), s.var('other'), s.var('total')
        s.le(elt, other + 1)         # per element
        s.le(total, elt + 2)         # an aggregate: `sum` over the list
        s.le(other, 7)

        subst: dict[int, Term] = {}
        s.instance({v.index for v, _ in (*elt.coeffs, *other.coeffs)}, subst, 0, '@0')
        part = elt.rename(subst)

        assert part != elt                       # a copy, not the same variable
        assert s.maximum(part) == 8              # `elt <= other + 1 <= 8`
        assert s.maximum(total - part) == math.inf   # the aggregate stayed put


@fp.fpy(ctx=fp.REAL)
def _exponent0(x, emin):
    """`exponent`, with the early return the AMD models spell."""
    if not fp.isfinite(x):
        return -1
    return max(fp.logb(x), emin)


@fp.fpy(ctx=fp.REAL)
def _exponent0_else(x, emin):
    if fp.isfinite(x):
        return max(fp.logb(x), emin)
    else:
        return -1


@fp.fpy(ctx=fp.REAL)
def _exponent0_zero(x, emin):
    if x == 0:
        return -1
    return max(fp.logb(x), emin)


@fp.fpy(ctx=fp.REAL)
def _exponent0_hoisted(x, emin):
    """The shape lowering leaves: the test in a temporary."""
    t = not fp.isfinite(x)
    if t:
        return -1
    return max(fp.logb(x), emin)


@fp.fpy(ctx=fp.REAL)
def _exponent0_or(x, emin):
    if fp.isnan(x) or fp.isinf(x):
        return -1
    return max(fp.logb(x), emin)


@fp.fpy(ctx=fp.REAL)
def _exponent0_and(x, emin):
    if fp.isnan(x) and emin < 0:
        return -1
    return max(fp.logb(x), emin)


class TestAPathOnlyANonFiniteValueReaches:
    """`logb` bounds the finite values, so no assignment describes a state
    holding a non-finite one.  A return only such a state reaches therefore
    states nothing -- the same reading `has_finite` already gives a return
    whose *value* has no finite values, applied to the guard instead."""

    @staticmethod
    def _round_prec(callee):

        @fp.fpy(ctx=fp.REAL)
        def f(x):
            e = callee(x, -126)
            with fp.MPFixedContext(e - 12, fp.RM.RTZ):
                return fp.round(x)

        g = monomorphize(f, args=[RealType(fp.FP32)])
        b = DigitBoundInfer.analyze(g.ast, FormatInfer.analyze(g.ast))
        t = next(t for e, t in b.by_expr.items() if e.format() == 'fp.round(x)')
        return b.store.prec(t.msb, t.lsb)

    def test_the_early_return_does_not_join_the_exponent(self):
        # `e >= logb(x)` survives the merge, so the round spans one binade
        assert self._round_prec(_exponent0) == 12

    def test_either_polarity(self):
        """The `else` of `if isfinite(x)` is the same path."""
        assert self._round_prec(_exponent0_else) == 12

    def test_a_zero_guard_is_not_one(self):
        """A zero *is* a state the store describes, so its arm still joins."""
        assert self._round_prec(_exponent0_zero) == 265

    def test_the_test_is_read_through_a_temporary(self):
        """Lowering hoists a condition, and a rule reading only the syntax
        would switch off where it matters most."""
        assert self._round_prec(_exponent0_hoisted) == 12

    def test_either_connective(self):
        """`or` needs every disjunct, `and` only one conjunct."""
        assert self._round_prec(_exponent0_or) == 12
        assert self._round_prec(_exponent0_and) == 12
