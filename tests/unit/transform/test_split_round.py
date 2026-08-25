"""
Unit tests for the :class:`fpy2.transform.SplitRound` transform.

The candidates are *rounded* operations — the complement of
:class:`fpy2.transform.RoundInsert`'s — and the rewrite mints fresh ``_tN``
names, so these tests assert

1. **Structural shape**: the operation lands alone in a block of the
   intermediate, with an explicit ``round`` back to the target in the enclosing
   block, which is what applies the target's own mode.
2. **Semantic equivalence** via the interpreter: a split admitted by Figure 8
   changes no result, checked over random inputs and the edge cases the
   premises exist for (underflow, overflow, ``-0``).
3. **Declines**, each by the reason it gives.
4. **The `where` contract**: a listing reports exactly what ``where=None``
   rewrites.
"""

import random

import pytest

import fpy2 as fp
from fpy2.analysis.format_infer import derive_intermediate
from fpy2.ast.fpyast import ContextStmt, ForeignVal, FuncDef, Mul, Round
from fpy2.ast.visitor import DefaultVisitor
from fpy2.function import Function
from fpy2.number import RoundingMode as RM
from fpy2.transform import (
    ExprCursor,
    SplitRound,
    TransformDeclined,
    TransformReferenceError,
)
from fpy2.transform.cursor import expr_sites

VIA32 = derive_intermediate(fp.FP32)
"""The tightest RTO intermediate for an FP32 / RNE target."""


def _count(ast: FuncDef, kind) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal n
            if kind is ContextStmt:
                n += 1
            super()._visit_context(stmt, c)

        def _visit_unaryop(self, e, c):
            nonlocal n
            if kind is Round and isinstance(e, Round):
                n += 1
            super()._visit_unaryop(e, c)

    _C()._visit_function(ast, None)
    return n


def _via_blocks(ast: FuncDef, ctx) -> int:
    """Blocks in *ast* written as a `ForeignVal` of *ctx*."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, c):
            nonlocal n
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val == ctx:
                n += 1
            super()._visit_context(stmt, c)

    _C()._visit_function(ast, None)
    return n


def _cursor_at(ast: FuncDef, kind) -> ExprCursor:
    found = expr_sites(ast, lambda e: isinstance(e, kind))
    assert found, f'no {kind.__name__} in the program'
    return found[0]


def _agree(before: FuncDef, after: FuncDef, runtime, args_list) -> bool:
    fa: Function = Function(before, runtime=runtime)
    fb: Function = Function(after, runtime=runtime)
    return all(str(fa(*a)) == str(fb(*a)) for a in args_list)


@fp.fpy(ctx=fp.REAL)
def _product(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        t = x * y
    return t


@fp.fpy(ctx=fp.REAL)
def _two_ops(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.FP32:
        t = x * y
        s = x + y
    return t + s


@fp.fpy(ctx=fp.REAL)
def _exact(x: fp.Real, y: fp.Real) -> fp.Real:
    with fp.REAL:
        t = x * y
    return t


class TestShape:
    def test_the_operation_moves_under_the_intermediate(self):
        out = SplitRound.apply(_product.ast, VIA32)
        assert _via_blocks(out, VIA32) == 1
        # the re-rounding is explicit, since an assignment rounds nothing
        assert _count(out, Round) == 1

    def test_the_round_sits_outside_the_block(self):
        """The `round` has to be in the *enclosing* block to pick up the
        target's mode; inside the intermediate's block it would be a no-op."""
        out = SplitRound.apply(_product.ast, VIA32)
        text = Function(out, runtime=_product.runtime).format()
        via_line = next(i for i, ln in enumerate(text.splitlines()) if 'RTO' in ln)
        round_line = next(i for i, ln in enumerate(text.splitlines()) if 'fp.round' in ln)
        assert round_line > via_line
        # and at a shallower indent than the operation it rounds
        lines = text.splitlines()
        op_indent = len(lines[via_line + 1]) - len(lines[via_line + 1].lstrip())
        rd_indent = len(lines[round_line]) - len(lines[round_line].lstrip())
        assert rd_indent < op_indent

    def test_each_site_gets_its_own_block(self):
        out = SplitRound.apply(_two_ops.ast, VIA32)
        assert _via_blocks(out, VIA32) == 2
        assert _count(out, Round) == 2


class TestEquivalence:
    @staticmethod
    def _sweep(n: int = 2000):
        rng = random.Random(0)
        for i in range(n):
            a = rng.uniform(-1e3, 1e3) if i % 3 else rng.uniform(-1e-30, 1e-30)
            b = rng.uniform(-1e3, 1e3) if i % 2 else rng.uniform(-1e30, 1e30)
            yield a, b

    @staticmethod
    def _subnormal_sweep(n: int = 600):
        """Products landing in FP32's *gradual*-underflow band.

        The `exp - k` half of each premise exists for exactly this range, and
        nothing else here reaches it: the products in `_sweep` either flush to
        zero or stay normal.
        """
        rng = random.Random(1)
        for _ in range(n):
            k = rng.randint(-74, -60)
            yield rng.uniform(1.0, 2.0) * 2.0 ** k, rng.uniform(0.5, 2.0)

    @staticmethod
    def _overflow_sweep(n: int = 400):
        """Products above FP32's maxval, where a non-saturating intermediate
        sends to `inf` what the target clamps."""
        rng = random.Random(2)
        for _ in range(n):
            yield rng.uniform(1.0, 2.0) * 2.0 ** 127, rng.uniform(2.0, 8.0)

    def test_a_split_changes_no_value(self):
        out = SplitRound.apply(_product.ast, VIA32)
        assert _agree(_product.ast, out, _product.runtime, self._sweep())

    def test_the_edge_cases_the_premises_exist_for(self):
        """Underflow to zero, overflow to infinity, and `-0` — the cases where a
        careless intermediate would disagree."""
        out = SplitRound.apply(_product.ast, VIA32)
        edges = [
            (1e-40, 1e-40), (1e30, 1e30), (-1e30, 1e30),
            (0.0, -1.0), (-0.0, 1.0), (1.0, 1.0),
            (float('nan'), 1.0), (float('inf'), 0.0),
        ]
        assert _agree(_product.ast, out, _product.runtime, edges)

    def test_gradual_underflow(self):
        out = SplitRound.apply(_product.ast, VIA32)
        assert _agree(_product.ast, out, _product.runtime, self._subnormal_sweep())

    def test_overflow_saturates_rather_than_going_to_infinity(self):
        """The intermediate must saturate: overflowing to `inf` sends a value
        the target clamps to its maxval somewhere the re-rounding cannot bring
        it back from."""
        out = SplitRound.apply(_product.ast, VIA32)
        assert _agree(_product.ast, out, _product.runtime, self._overflow_sweep())

    def test_a_nested_rounded_operand(self):
        """An operand that is itself a rounded operation has to be bound under
        the *original* scope; left inline it would be re-rounded to the
        intermediate instead of to the target."""

        @fp.fpy(ctx=fp.FP32)
        def nested(x: fp.Real, y: fp.Real) -> fp.Real:
            return (x * y) + y

        assert len(SplitRound.sites(nested.ast, ctx=VIA32)) == 2
        out = SplitRound.apply(nested.ast, VIA32)
        assert _agree(nested.ast, out, nested.runtime, self._sweep(800))
        assert _agree(nested.ast, out, nested.runtime, self._subnormal_sweep(200))

    def test_a_fixed_point_target(self):
        """The premises are containment checks on `A`, indifferent to the format
        family."""
        target = fp.MPFixedContext(-8, fp.RoundingMode.RTZ)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        via = derive_intermediate(target)
        assert len(SplitRound.sites(f.ast, ctx=via)) == 1
        out = SplitRound.apply(f.ast, via)
        assert _agree(f.ast, out, f.runtime, self._sweep(400))

    def test_both_sites_together(self):
        out = SplitRound.apply(_two_ops.ast, VIA32)
        assert _agree(_two_ops.ast, out, _two_ops.runtime, self._sweep(500))

    @pytest.mark.parametrize('rm1', [RM.RTZ, RM.RAZ, RM.RTO])
    def test_a_directed_target(self, rm1):
        """Each admitted final mode, through its own derived intermediate."""
        target = fp.FP32.with_params(rm=rm1)

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with target:
                t = x * y
            return t

        via = derive_intermediate(target)
        assert len(SplitRound.sites(f.ast, ctx=via)) == 1
        out = SplitRound.apply(f.ast, via)
        assert _agree(f.ast, out, f.runtime, self._sweep(500))


class TestDeclines:
    def test_an_exact_operation(self):
        """Nothing to split — that direction is `insert_round`'s."""
        assert SplitRound.sites(_exact.ast, ctx=VIA32) == []
        why = SplitRound.refusals(_exact.ast, ctx=VIA32)
        assert len(why) == 1 and 'no rounding to split' in why[0][1]

    def test_an_unsound_mode_pair(self):
        """RNE over RNE: the pairing every `fp.FP*` context falls into, and
        Table 2's last row says it is unsound however wide the intermediate."""
        assert SplitRound.sites(_product.ast, ctx=fp.FP64) == []
        why = SplitRound.refusals(_product.ast, ctx=fp.FP64)
        assert len(why) == 1 and 'is not the same as' in why[0][1]

    def test_a_stochastic_intermediate(self):
        via = VIA32.with_params(num_randbits=2)
        assert via.is_stochastic()
        assert SplitRound.sites(_product.ast, ctx=via) == []
        why = SplitRound.refusals(_product.ast, ctx=via)
        assert len(why) == 1 and 'stochastic' in why[0][1]

    def test_a_symbolic_scope(self):
        """Without a pinned function context the scope stays symbolic, so the
        rounding it performs is unknown."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x * y

        assert SplitRound.sites(f.ast, ctx=VIA32) == []
        why = SplitRound.refusals(f.ast, ctx=VIA32)
        assert why and 'symbolic' in why[0][1]

    def test_naming_a_refused_operation_by_cursor(self):
        cursor = _cursor_at(_exact.ast, Mul)
        with pytest.raises(TransformDeclined, match='no rounding to split'):
            SplitRound.apply(_exact.ast, VIA32, where=cursor)

    def test_a_where_naming_nothing(self):
        with pytest.raises(TransformReferenceError):
            SplitRound.apply(_product.ast, VIA32, where=7)

    def test_a_non_context_intermediate(self):
        with pytest.raises(TypeError):
            SplitRound.sites(_product.ast, ctx=fp.FP32.format())  # type: ignore[arg-type]


class TestUnreachablePositions:
    def test_a_while_condition_is_left_alone(self):
        """The condition is re-evaluated every iteration, so a block hoisted
        before the loop computes it once — which does not terminate.  Measured:
        it hung before the suppression was added."""

        @fp.fpy(ctx=fp.FP32)
        def shrink(n: fp.Real) -> fp.Real:
            i = 0.0
            while i * 1.0 < n:
                i = i + 1.0
            return i

        out = SplitRound.apply(shrink.ast, VIA32)
        # the body's `i + 1.0` is the one site; the condition's `i * 1.0` is not
        assert len(SplitRound.sites(shrink.ast, ctx=VIA32)) == 1
        assert str(Function(out, runtime=shrink.runtime)(3.0)) == str(shrink(3.0))

    def test_an_if_expr_branch_is_left_alone(self):
        @fp.fpy(ctx=fp.FP32)
        def f(x: fp.Real, c: bool) -> fp.Real:
            return (x * x) if c else x

        assert SplitRound.sites(f.ast, ctx=VIA32) == []


class TestWhereContract:
    def test_sites_are_what_where_none_rewrites(self):
        listed = SplitRound.sites(_two_ops.ast, ctx=VIA32)
        assert len(listed) == 2
        every = SplitRound.apply(_two_ops.ast, VIA32)
        assert _via_blocks(every, VIA32) == len(listed)

    def test_each_index_rewrites_one(self):
        for j in range(len(SplitRound.sites(_two_ops.ast, ctx=VIA32))):
            out = SplitRound.apply(_two_ops.ast, VIA32, where=j)
            assert _via_blocks(out, VIA32) == 1

    def test_a_cursor_aims_the_same_as_its_index(self):
        listed = SplitRound.sites(_two_ops.ast, ctx=VIA32)
        for j, cursor in enumerate(listed):
            by_index = SplitRound.apply(_two_ops.ast, VIA32, where=j)
            by_cursor = SplitRound.apply(_two_ops.ast, VIA32, where=cursor)
            assert by_cursor.is_equiv(by_index)


class TestNotCandidates:
    def test_an_explicit_rounding_is_not_split(self):
        """Splitting a rounding is `merge_round`'s inverse; admitting `Round`
        and `Cast` would also make a second application grow the tree twice as
        fast."""

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                t = fp.round(x)
            return t

        assert SplitRound.sites(f.ast, ctx=VIA32) == []
        assert SplitRound.refusals(f.ast, ctx=VIA32) == []


class TestRepeatedApplication:
    def test_each_pass_terminates_and_splits_once_more(self):
        """Like the loop rewrites, this one is not idempotent: the operation is
        now under an RTO intermediate, and RTO-over-RTO is itself admitted.  One
        pass terminates; a schedule that wants a fixpoint has to bound it."""
        ast = _product.ast
        for expect in (1, 2, 3):
            assert len(SplitRound.sites(ast, ctx=VIA32)) == 1
            ast = SplitRound.apply(ast, VIA32)
            assert _via_blocks(ast, VIA32) == expect
            assert str(Function(ast, runtime=_product.runtime)(1.5, 2.5)) \
                == str(_product(1.5, 2.5))
