"""
Regression net for `HoistScale` — Phase 1 of `docs/todos/hoist-scale.md`.

Nothing hoists out of a reduction yet.  These assertions pin what the tree does
*today*, so that the diffs in Phase 2 (`value_class` inspects the base literal)
and Phase 3 (`array_size` carries a symbolic dimension) show exactly which
condition became answerable.  Every assertion that is meant to flip says so.

The rewrite is `sum(c * eᵢ) -> c * sum(eᵢ)`, under five conditions.  Two hold
already: the reduction's scope rounds exactly, and every name the factor reads
is bound outside the loop — the latter is what #307 established.  Two are
answered by the analysis upgrades below.  The fifth is the syntactic sign check,
which is the pass's own and has nothing to pin here.
"""

import fpy2.strategies as st

from fpy2.analysis import ArraySizeInfer, LiveVars, ValueClassInfer
from fpy2.analysis.array_size import ListSize, is_size_eq
from fpy2.ast import IndexedAssign, Pow, Sum, Var
from fpy2.transform import walk_exprs
from fpy2.transform.hoist_invariant import _from_before, _Nodes
from fpy2.transform.utils import RoundingScopes

from .test_hoist_invariant import (
    _VALUES,
    _agrees_by_value,
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
        sums = _reductions(out.ast)
        assert [s.format() for s in sums] == ['sum(ts)', 'sum(xs)']


class TestConditionsThatAlreadyHold:

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


class TestConditionsTheAnalysisUpgradesAnswer:

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
