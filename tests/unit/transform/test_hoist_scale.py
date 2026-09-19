"""
`HoistScale`: pulling an invariant factor out of a reduction.

The rewrite is `sum(c * eᵢ) -> c * sum(eᵢ)` in two shapes — a comprehension,
and a loop filling a list — and it is wrong without every one of its
conditions, so most of what is here is refusals.  `TestSoundness` holds the
ones that were once silent miscompiles; each names what stood between the
loop, the product, the write and the reduction.
"""

import fpy2 as fp
import fpy2.strategies as st
import pytest

from fpy2.analysis import ArraySizeInfer, LiveVars, ValueClassInfer
from fpy2.analysis.array_size import ListSize, is_size_eq
from fpy2.ast import IndexedAssign, Pow, Sum, Var
from fpy2.transform import HoistScale, TransformDeclined, walk_exprs
from fpy2.transform.hoist_invariant import _from_before, _Nodes
from fpy2.transform.utils import RoundingScopes

from .test_hoist_invariant import (
    _VALUES,
    _agrees_by_value,
    _loop_bodies_text,
    _loops,
    _scale_factor,
    _text,
    fused_sum,
)


def _scheduled():
    """The motivating schedule as PR 2 receives it — #307 included, since it is
    what discharges the factor's free variables."""
    f = st.simplify(st.rescale_fixed(st.comp_to_loop(st.fuse(fused_sum))))
    return st.hoist_invariant(f)


def _reductions(ast) -> list[Sum]:
    """Every `sum(...)` in *ast*, in visit order."""
    return [e for _, e in walk_exprs(ast) if isinstance(e, Sum)]


def _hoisted(func):
    return func.with_ast(HoistScale.apply(func.ast))


def _why(func) -> list[str]:
    return [why for _, why in HoistScale.refusals(func.ast)]


@fp.fpy(ctx=fp.REAL)
def guarded_comp(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        return sum([(2 ** k) * x for x in xs])
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def two_sums(xs: list[fp.Real], ys: list[fp.Real], k: fp.Real) -> fp.Real:
    """Two independent candidates, for aiming."""
    if fp.isfinite(k):
        a = sum([(2 ** k) * x for x in xs])
        b = sum([(2 ** k) * y for y in ys])
        return a + b
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def factor_reads_the_target(xs: list[fp.Real]) -> fp.Real:
    return sum([(2 ** x) * x for x in xs])


@fp.fpy(ctx=fp.REAL)
def factor_may_be_infinite(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    return sum([(2 ** k) * x for x in xs])


@fp.fpy(ctx=fp.FP32)
def rounds_between_adds(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        return sum([(2 ** k) * x for x in xs])
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def negative_factor(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        return sum([k * x for x in xs])
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def loop_may_not_cover(xs: list[fp.Real], ys: list[fp.Real], k: fp.Real) -> fp.Real:
    """`ts` is as long as `ys` but the loop runs `len(xs)` times, so an element
    may be left holding whatever `fp.empty` put there.  `k` is guarded so that
    coverage is the condition that refuses, not the factor."""
    if fp.isfinite(k):
        ts = fp.empty(len(ys))
        for i in range(len(xs)):
            t = (2 ** k) * xs[i]
            ts[i] = t
        return sum(ts)
    else:
        return 0.0


def _full_schedule(func):
    """What a user writes, through the strategies."""
    f = st.simplify(st.rescale_fixed(st.comp_to_loop(st.fuse(func))))
    return st.simplify(st.hoist_scale(st.hoist_invariant(f)))


# ----------------------------------------------------------------------
# Programs the rewrite must refuse


@fp.fpy(ctx=fp.REAL)
def list_rebound_after_the_loop(
    xs: list[fp.Real], ys: list[fp.Real], k: fp.Real
) -> fp.Real:
    if fp.isfinite(k):
        c = 2 ** k
        ts = fp.empty(len(xs))
        for i in range(len(xs)):
            t = c * xs[i]
            ts[i] = t
        ts = ys
        return sum(ts)
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def factor_rebound_before_the_reduction(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        c = 2 ** k
        ts = fp.empty(len(xs))
        for i in range(len(xs)):
            t = c * xs[i]
            ts[i] = t
        c = 1000.0
        return sum(ts) + (c - c)
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def writes_one_slot(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        c = 2 ** k
        ts = [1.0 for x in xs]
        for i in range(len(xs)):
            t = c * xs[i]
            ts[0] = t
        return sum(ts)
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def writes_twice(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        c = 2 ** k
        ts = [1.0 for x in xs]
        for i in range(len(xs)):
            t = c * xs[i]
            ts[i] = t
            if xs[i] > 1.5:
                ts[i] = 3.0
        return sum(ts)
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def list_read_in_the_body(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        c = 2 ** k
        ts = fp.empty(len(xs))
        acc = 0.0
        for i in range(len(xs)):
            t = c * xs[i]
            ts[i] = t
            acc = acc + ts[i]
        return sum(ts) - acc
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def product_read_in_the_body(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        c = 2 ** k
        ts = fp.empty(len(xs))
        acc = 0.0
        for i in range(len(xs)):
            t = c * xs[i]
            ts[i] = t
            acc = acc + t
        return sum(ts) + acc
    else:
        return 0.0


@fp.fpy(ctx=fp.REAL)
def product_rounds(xs: list[fp.Real], k: fp.Real) -> fp.Real:
    if fp.isfinite(k):
        c = 3 ** k
        ts = fp.empty(len(xs))
        for i in range(len(xs)):
            with fp.FP32:
                t = c * xs[i]
            ts[i] = t
        return sum(ts)
    else:
        return 0.0


UNSOUND = (
    (list_rebound_after_the_loop, [([1.0, 2.0], [3.0, 4.0], 1.0)]),
    (factor_rebound_before_the_reduction, [([1.0, 2.0], 1.0)]),
    (writes_one_slot, [([1.0, 2.0, 4.0], 1.0)]),
    (writes_twice, [([1.0, 2.0, 4.0], 1.0)]),
    (list_read_in_the_body, [([1.0, 2.0, 4.0], 1.0)]),
    (product_read_in_the_body, [([1.0, 2.0, 4.0], 1.0)]),
    (product_rounds, [([1.0, 2.0], 1.0)]),
)


class TestTheShapeTheRewriteMatches:

    def test_the_element_write_is_a_scaled_product(self):
        """`ts[i] = t` where `t = c * e`.  The write does not read the product
        syntactically — `rescale_fixed` binds it to a name first — so the pass
        reaches it through `defining_expr`, as `_scale_factor` does."""
        _, loop, _, factor = _scale_factor(_scheduled())
        # a name, not the gensym it happens to be: that renumbers upstream
        assert isinstance(factor, Var)
        writes = [s for s in loop.body.stmts if isinstance(s, IndexedAssign)]
        assert len(writes) == 1
        assert str(writes[0].var) == 'ts'

    def test_the_factor_is_a_name_and_the_power_is_one_step_further(self):
        """#307 hoisted `2 ** _k` above the loop, so the factor arrives as a
        *name*.  The sign check keys on a power with a positive literal base,
        which therefore needs `defining_expr` as well — one indirection for the
        product, and a second for the factor."""
        def_use, _, _, factor = _scale_factor(_scheduled())
        assert not isinstance(factor, Pow)
        base_pow = def_use.defining_expr(factor)
        assert isinstance(base_pow, Pow)
        assert base_pow.args[0].format() == '2'

    def test_there_are_two_reductions_and_only_one_is_a_candidate(self):
        out = _scheduled()
        assert [s.format() for s in _reductions(out.ast)] == ['sum(ts)', 'sum(xs)']
        site, = HoistScale.sites(out.ast)
        assert site.resolve().format() == 'sum(ts)'


class TestTheConditionsOnTheMotivatingSchedule:

    def test_the_then_branch_reduction_rounds_exactly(self):
        """Condition 1, and it is per-site: the same function holds a second
        reduction under `fp.FP32` that the pass must refuse."""
        out = _scheduled()
        scopes = RoundingScopes(out.ast)
        ts_sum, xs_sum = _reductions(out.ast)
        assert scopes.is_exact(ts_sum)
        assert not scopes.is_exact(xs_sum)

    def test_the_factor_reads_only_names_bound_before_the_loop(self):
        """Condition 2, established by #307 — asserted here because this is the
        PR that consumes it."""
        def_use, loop, stmt, factor = _scale_factor(_scheduled())
        body = _Nodes.of(loop.body)
        reaching = def_use.reach[stmt]
        assert all(
            _from_before(reaching.get(name), loop, body, set())
            for name in LiveVars.analyze(factor)
        )


    def test_the_factor_is_known_finite(self):
        """Condition 3.  `_k` may be `-inf` (an all-zero input makes
        `logb(0) = -inf`), and `2 ** -inf` is zero, not an infinity — which is
        exactly what reading the base literal establishes."""
        out = _scheduled()
        _, _, _, factor = _scale_factor(out)
        cls = ValueClassInfer.analyze(out.ast).classify(factor)
        assert str(cls) == 'ValueClass.ZERO|FINITE'

    def test_the_result_list_is_the_same_size_as_the_input(self):
        """Condition 5, with no length annotation anywhere: `xs` carries a
        fresh size symbol and `fp.empty(len(xs))` now keeps it, so the trip
        count and the list length are provably the same."""
        out = _scheduled()
        info = ArraySizeInfer.analyze(out.ast)
        xs, = [b for d, b in info.by_def.items()
               if b is not None and str(d.name) == 'xs']
        ts = [b for d, b in info.by_def.items()
              if b is not None and str(d.name) == 'ts']
        assert isinstance(xs, ListSize) and xs.size is not None
        assert ts and all(is_size_eq(xs, b) for b in ts)

    def test_the_loop_runs_once_per_element(self):
        """The other half of condition 5: the trip count is `len(xs)`, which
        the size above ties to the list."""
        out = _scheduled()
        loop = _loops(out.ast)[-1]
        assert loop.iterable.format() == 'range(len(xs))'
        assert 'ts = fp.empty(len(xs))' in _text(out, out.ast)


class TestTheBaseline:

    def test_the_schedule_computes_what_the_source_computes(self):
        """The empty list is excluded: `max([])` has no value."""
        assert _agrees_by_value(fused_sum, _scheduled().ast, values=_VALUES[1:])


class TestTheComprehensionForm:
    """The shape a program is written in, and the simpler of the two: a
    comprehension defines every element, so coverage needs no proof."""

    def test_it_hoists_the_factor(self):
        out = _hoisted(guarded_comp)
        assert '((2 ** k) * sum([x for x in xs]))' in _text(guarded_comp, out.ast)

    def test_the_values_are_unchanged(self):
        out = fp.Function(_hoisted(guarded_comp).ast, runtime=guarded_comp.runtime)
        for xs in ([], [1.0], [1.5, -2.0, 3.25], [1e300, -1e300, 1.0]):
            for k in (3.0, -5.0, 0.0):
                assert repr(out(xs, k)) == repr(guarded_comp(xs, k))

    def test_a_factor_reading_the_comprehension_target_is_refused(self):
        assert HoistScale.sites(factor_reads_the_target.ast) == []
        assert _why(factor_reads_the_target) == [
            'the factor reads `x`, which varies'
        ]

    def test_a_factor_that_may_be_infinite_is_refused(self):
        """`2 ** k` for an unconstrained `k` is `+inf` at `k = +inf`, and an
        infinite factor turns a cancellation into a survivor."""
        assert HoistScale.sites(factor_may_be_infinite.ast) == []
        assert _why(factor_may_be_infinite) == [
            'the factor may be an infinity or a NaN'
        ]

    def test_a_factor_that_may_be_negative_is_refused(self):
        assert HoistScale.sites(negative_factor.ast) == []
        assert 'may be negative' in _why(negative_factor)[0]

    def test_a_rounding_scope_is_refused(self):
        """The same program under `fp.FP32`: the partial sums round, so the
        two orders disagree."""
        assert HoistScale.sites(rounds_between_adds.ast) == []
        assert _why(rounds_between_adds) == [
            'the reduction does not round exactly'
        ]


class TestTheLoweredForm:
    """What the motivating schedule produces, where coverage is the difficulty."""

    def test_it_hoists_the_factor(self):
        out = _hoisted(_scheduled())
        src = _text(out, out.ast)
        assert 'return (t14 * sum(ts))' in src
        assert '(t14 * _t13)' not in src

    def test_the_values_are_unchanged(self):
        before = _scheduled()
        assert _agrees_by_value(before, _hoisted(before).ast, values=_VALUES[1:])

    def test_the_fp32_branch_is_refused(self):
        """`sum(xs)` reads an argument, not a list a loop filled."""
        assert 'no scaled list write fills the reduction' in _why(_scheduled())

    def test_a_loop_that_may_not_cover_the_list_is_refused(self):
        assert HoistScale.sites(loop_may_not_cover.ast) == []
        assert 'may not write every element' in _why(loop_may_not_cover)[0]


class TestAiming:

    def test_the_only_site_is_the_exact_reduction(self):
        out = _scheduled()
        site, = HoistScale.sites(out.ast)
        assert site.resolve().format() == 'sum(ts)'

    def test_a_cursor_on_a_refused_reduction_says_why(self):
        where = [c for c, _ in HoistScale.refusals(rounds_between_adds.ast)]
        with pytest.raises(TransformDeclined, match='round exactly'):
            HoistScale.apply(rounds_between_adds.ast, where[0])


class TestTheWholeSchedule:

    def test_it_reaches_the_target_form(self):
        """`docs/todos/algebraic-rewrites.md`: the loop body is one multiply,
        one round and one store, and the scaling is a single multiply after an
        integer accumulation."""
        out = _full_schedule(fused_sum)
        src = _text(out, out.ast)
        assert 'ts[t10] = _t13' in src
        assert 'return (t14 * sum(ts))' in src
        assert len(_loops(out.ast)[-1].body.stmts) == 4

    def test_the_elements_are_written_unscaled(self):
        out = _full_schedule(fused_sum)
        assert '2 **' not in _loop_bodies_text(out.ast)

    def test_the_fp32_branch_is_untouched(self):
        """Condition 1 is per-site: the `else` arm rounds, so its reduction is
        refused while the other is rewritten."""
        out = _full_schedule(fused_sum)
        assert 'return sum(xs)' in _text(out, out.ast)

    def test_the_values_are_unchanged(self):
        out = _full_schedule(fused_sum)
        assert _agrees_by_value(fused_sum, out.ast, values=_VALUES[1:])


class TestSoundness:
    """Each of these was a silent miscompile.  The shared mistake was letting
    something stand between the loop, the product, the write and the
    reduction — a rebinding, a second write, an extra read, or a rounding
    scope of its own."""

    @pytest.mark.parametrize('func,_args', UNSOUND, ids=lambda v: getattr(v, 'name', ''))
    def test_it_is_refused(self, func, _args):
        assert HoistScale.sites(func.ast) == []
        assert HoistScale.apply(func.ast).is_equiv(func.ast)

    @pytest.mark.parametrize('func,args', UNSOUND, ids=lambda v: getattr(v, 'name', ''))
    def test_the_values_would_have_changed(self, func, args):
        """Not just refused — refused for cause: the rewrite really is wrong
        on each of these."""
        out = fp.Function(HoistScale.apply(func.ast), runtime=func.runtime)
        for a in args:
            assert repr(out(*a)) == repr(func(*a))
