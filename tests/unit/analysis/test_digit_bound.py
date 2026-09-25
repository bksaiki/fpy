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


def _analyze(fn, args, lower=None):
    """*fn* monomorphized at *args*, lowered by *lower*, and its digit bounds."""
    ast = monomorphize(fn, args=args).ast
    if lower is not None:
        ast = lower(ast)
    return ast, DigitBoundInfer.analyze(ast, FormatInfer.analyze(ast))


def _term(b, text: str):
    """The expression printing as *text*, and its terms."""
    return next((e, t) for e, t in b.by_expr.items() if e.format() == text)


class _FakeSolver:
    """A backend answering every query with *answer*, recording what it is
    told and each query's cutoff."""

    def __init__(self, answer):
        self.answer = answer
        self.assumed = []
        self.cutoffs = []

    def assume(self, constraint):
        self.assumed.append(constraint)

    def maximize(self, objective, cutoff=None, assuming=frozenset()):
        self.cutoffs.append(cutoff)
        return self.answer


class TestTerm:
    """Affine arithmetic.  ``==`` is structural, never a constraint."""

    def test_arithmetic_collects_coefficients(self):
        s = DigitBoundStore()
        a, b = s.var('a'), s.var('b')
        assert a + a == a * 2
        assert a + b - b == a
        assert (a + 1) - 1 == a
        assert 3 - a == -a + 3

    def test_ordering_is_deterministic(self):
        """Terms order by creation index, not by ``id()``."""
        s = DigitBoundStore()
        a, b = s.var('a'), s.var('b')
        assert (b + a).coeffs == (a + b).coeffs


class TestQueries:

    def test_a_constant_term_needs_no_constraints(self):
        s = DigitBoundStore()
        a = s.var('a')
        assert s.maximum(a - a + 7) == 7

    def test_precision_is_the_span(self):
        s = DigitBoundStore()
        l, g = s.var('l'), s.var('g')
        s.le(l, 15)
        s.ge(g, l - 10)      # an FP16 value: 11 significand bits
        assert s.prec(l, g) == 11


class TestFusedSum:
    """``nv.t_fdpa``: the store the emission rules build, symbolic in `F`."""

    @staticmethod
    def _store(F):
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

    def test_a_product_is_F_plus_2_and_the_accumulator_F_plus_1(self):
        """`c` is bounded by its own `e_c <= e_max`, so it reaches one binade
        less far than a product does."""
        for F in (13, 24, 35):
            s, v = self._store(F)
            assert s.prec(v['lp'], v['n'] + 1) == F + 2
            assert s.prec(v['lc'], v['n'] + 1) == F + 1

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
    """``with fp.MPFixedContext(logb(x) - k): y = fp.round(x)`` keeps `k`
    digits.  A grid coarser than the value's whole reach (`k < 0`) keeps
    none: a count of digits floors at zero."""

    @pytest.mark.parametrize('k', [12, -5])
    def test_prec_is_k_floored_at_zero(self, k):
        s = DigitBoundStore()
        lx, n = s.var('lx'), s.var('n')
        s.le(lx, 127)
        s.eq(n, lx - k)
        assert s.prec(lx, n + 1) == max(k, 0)


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
        s = DigitBoundStore(solver=_FakeSolver(5))
        x = s.var('x')
        s.le(x, 3)   # constrain it, or the store answers `inf` without asking
        assert s.maximum(x) == 5

    def test_a_free_variable_needs_no_backend(self):
        """Nothing constrains it, so it is unbounded by inspection -- asking
        a solver could only reach the same answer more slowly."""
        solver = _FakeSolver(0)
        s = DigitBoundStore(solver=solver)
        x, y = s.var('x'), s.var('y')
        s.le(y, 3)
        assert s.maximum(x) == math.inf
        assert s.maximum(-x) == math.inf
        assert s.maximum(x + y) == math.inf
        assert solver.cutoffs == []

    def test_every_constraint_reaches_the_backend(self):
        """The store states each constraint once, as it is made: a backend
        that only saw them at query time would have to be re-told the system
        on every objective."""
        solver = _FakeSolver(0)
        s = DigitBoundStore(solver=solver)
        x, y = s.var('x'), s.var('y')
        s.le(x, 3)
        s.eq(y, x)
        s.le_max(x, [y, 4])
        assert len(solver.assumed) == 3

    def test_a_backend_may_ignore_the_cutoff(self):
        """`cutoff` lets a backend stop early; the exact maximum is an upper
        bound whatever the caller asked for, so ignoring it stays correct."""
        solver = _FakeSolver(3)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        s.le(x, 3)
        assert s.reaches([(x, 3)]) is True
        assert s.reaches([(x, 4)]) is False
        assert solver.cutoffs == [3, 4]      # the threshold reached the backend


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
        solver = _FakeSolver(0)
        s = DigitBoundStore(solver=solver)
        x = s.var('x')
        assert s.reaches([(x, 10**9)]) is True
        assert solver.cutoffs == []


class TestBisectionMatchesOptimization:
    """Bisection is the default because it is cheaper; it has to agree with
    `Optimize`, which is the reference.  Its upward probe needs somewhere to
    stop, and where it stops must only ever loosen the answer."""

    CASES = (
        # far below any plausible cutoff: reading it as unsatisfiable, or as
        # anything below itself, would be unsound
        ('deeply negative', lambda s, x, y: (s.le(x, -5_000_000), x), -5_000_000),
        ('large finite', lambda s, x, y: (s.le(x, 9_000_000), x), 9_000_000),
        ('through a chain', lambda s, x, y: ((s.le(y, 7), s.le(x, y)), x), 7),
        ('bounded below only', lambda s, x, y: (
            (s.le(y, 3), s.ge(x, y)), x), math.inf),
        ('over a max', lambda s, x, y: (
            (s.le(y, 3), s.le_max(x, [y, y + 5])), x), 8),
        ('unsatisfiable', lambda s, x, y: (
            (s.le(x, 1), s.ge(x, 2)), x), -math.inf),
        # nothing restricts a coefficient to a unit, and the probe's ceiling
        # has to reach far enough for one that is not
        ('scaled objective', lambda s, x, y: (
            (s.le(x, 1), s.ge(x, 0)), x * 100), 100),
        ('scaled constraint', lambda s, x, y: (
            (s.le(x * 7, 70), s.ge(x, 0)), x), 10),
    )

    @pytest.mark.parametrize('name,build,want', CASES, ids=[c[0] for c in CASES])
    def test_both_backends_find_the_maximum(self, name, build, want):
        for bisect in (True, False):
            s = DigitBoundStore(solver=Z3Solver(bisect=bisect))
            x, y = s.var('x'), s.var('y')
            _, obj = build(s, x, y)
            assert s.maximum(obj) == want, (name, bisect)


class TestAGuardedConstraint:
    """A constraint holding only where its literals do, and a query that says
    which it assumes."""

    @pytest.mark.parametrize('bisect', [True, False])
    def test_it_binds_only_when_assumed(self, bisect):
        s = DigitBoundStore(solver=Z3Solver(bisect=bisect))
        x = s.var('x')
        s.le(x, 10)
        g = s.literal()
        s.le(x, 3, guard=(g,))
        assert s.maximum(x) == 10
        assert s.maximum(x, frozenset({g})) == 3
        assert s.reaches([(x, 4)])
        assert not s.reaches([(x, 4)], frozenset({g}))

    def test_an_instance_copies_a_universal_one_under_its_guard(self):
        """"Every element of `xs` is finite" holds at every index or none --
        and at every index only where it holds at all."""
        s = DigitBoundStore()
        elt = s.var('elt')
        g = s.literal(universal=True)
        s.le(elt, 3, guard=(g,))
        s.le(elt, 9)
        subst: dict[int, Term] = {}
        s.instance({v.index for v, _ in elt.coeffs}, subst, 0, '@0')
        part = elt.rename(subst)
        assert s.maximum(part, frozenset({g})) == 3
        assert s.maximum(part) == 9

    def test_an_instance_does_not_copy_it(self):
        """A literal names one definition, not one per index."""
        s = DigitBoundStore()
        elt = s.var('elt')
        g = s.literal()
        s.le(elt, 3, guard=(g,))
        s.le(elt, 9)
        subst: dict[int, Term] = {}
        s.instance({v.index for v, _ in elt.coeffs}, subst, 0, '@0')
        assert s.maximum(elt.rename(subst), frozenset({g})) == 9


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

        ast, b = _analyze(f, [RealType(fp.FP64), ListType(RealType(fp.FP16), 4)])
        ret = b.by_expr[ast.body.stmts[-1].expr]
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

        _, b = _analyze(f, [RealType(fp.FP32)])
        _, sel = _term(b, '(0 if x == 0 else r)')
        _, total = _term(b, '(y + r)')
        # `r` is 12 digits; the zero arm adds none, and the sum one more
        assert b.store.prec(sel.msb, sel.lsb) == 12
        assert b.store.prec(total.msb, total.lsb) == 13


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
        _, b = _analyze(fn, [RealType(fp.FP64)])
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
        _, b = _analyze(fn, [RealType(fp.FP16), RealType(fp.FP16)])
        _, t = _term(b, text)
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
        # every form: the comprehension, the gather loop it lowers to, and
        # the counted loop `range(len(range(a, b, s)))` with index `a + s * k`
        out = []
        for lower in (
            None, CompToLoop.apply,
            lambda ast: CompToLoop.apply(ast, index_ranges=True),
        ):
            _, b = _analyze(fn, [ListType(RealType(fp.FP32), 8)], lower)
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

        assert self._precs(f) == [12, 12, 12]

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
        assert self._precs(f) == [288, 288, 288]

    def test_but_not_across_nested_counts(self):
        """`1 + 2 * k1` and `1 + 2 * k2` are one index set, but not one index:
        only the innermost count's element is element `k` of it."""

        @fp.fpy(ctx=fp.REAL)
        def f(xs):
            es = [fp.logb(xs[i]) for i in range(len(xs))]
            ys = fp.empty(4)
            for k1 in range(4):
                with fp.INTEGER:
                    i1 = 1 + 2 * k1
                for k2 in range(4):
                    with fp.INTEGER:
                        i2 = 1 + 2 * k2
                    with fp.MPFixedContext(es[i1] - 12, fp.RM.RTZ):
                        ys[k2] = fp.round(xs[i2])
            return sum(ys)

        assert all(p > 12 for p in self._precs(f))


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
def _exponent0_zero(x, emin):
    if x == 0:
        return -1
    return max(fp.logb(x), emin)


class TestAPathOnlyANonFiniteValueReaches:
    """A return reached only where `x` is non-finite is not taken where it is
    finite -- which the callee says of its result, and a caller that knows `x`
    finite resolves.  A caller that does not keeps the sentinel: `exponent0`'s
    `-1` is a finite value, and a position built from it is a real one."""

    @staticmethod
    def _round_prec(callee, guarded: bool = True):

        @fp.fpy(ctx=fp.REAL)
        def checked(x):
            if fp.isfinite(x):
                e = callee(x, -126)
                with fp.MPFixedContext(e - 12, fp.RM.RTZ):
                    t = fp.round(x)
            else:
                t = 0
            return t

        @fp.fpy(ctx=fp.REAL)
        def unchecked(x):
            e = callee(x, -126)
            with fp.MPFixedContext(e - 12, fp.RM.RTZ):
                return fp.round(x)

        _, b = _analyze(checked if guarded else unchecked, [RealType(fp.FP32)])
        e, t = _term(b, 'fp.round(x)')
        return b.store.prec(t.msb, t.lsb, b.assume_at(e))

    def test_the_early_return_does_not_join_the_exponent(self):
        # `e >= logb(x)` survives the merge, so the round spans one binade
        assert self._round_prec(_exponent0) == 12

    def test_not_where_the_caller_does_not_know(self):
        assert self._round_prec(_exponent0, guarded=False) > 12

    def test_a_zero_guard_is_not_one(self):
        """A zero *is* a state the store describes, so its arm still joins."""
        assert self._round_prec(_exponent0_zero) > 12
