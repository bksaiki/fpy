"""`why_not_tileable`: may a loop body be evaluated as a tile?

Splitting a loop preserves semantics; evaluating the inner body as a tile is
what can change the answer, and only through the loop-carried variables.
"""

import pytest

import fpy2 as fp
from fpy2.ast.fpyast import ForStmt, If1Stmt, IfStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2 import Function
from fpy2.backend.triton import tile_loops, why_not_tileable


class _Loops(DefaultVisitor):
    def __init__(self):
        super().__init__()
        self.out: list[ForStmt] = []

    def _visit_for(self, s: ForStmt, ctx):
        self.out.append(s)
        return super()._visit_for(s, ctx)


def _count(ast, *types) -> int:
    n = 0

    class _V(DefaultVisitor):
        def _visit_statement(self, stmt, ctx):
            nonlocal n
            if isinstance(stmt, types):
                n += 1
            return super()._visit_statement(stmt, ctx)

    _V()._visit_function(ast, None)
    return n


def _innermost(func) -> ForStmt:
    v = _Loops()
    v._visit_function(func.ast, None)
    assert v.out, 'expected a loop'
    return v.out[-1]


def _why(func) -> str | None:
    return why_not_tileable(_innermost(func), func.ast)


class TestTileable:
    def test_a_max_fold_selects_an_operand(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            return m

        assert _why(f) is None

    def test_a_flag_cleared_under_a_guard_is_idempotent(self):
        """`matrix.is_diagonal`'s shape: an `and`-fold that does not look like
        one.  Which iteration cleared the flag does not matter."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            ok: bool = True
            for x in xs:
                if x != 0:
                    ok = False
            return ok

        assert _why(f) is None

    def test_an_element_write_at_the_loop_variable(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            out = [fp.round(0) for _ in xs]
            for i in range(len(xs)):
                out[i] = xs[i] * 2
            return out

        assert _why(f) is None

    def test_a_nested_write_mixes_the_loop_variable_with_an_invariant(self):
        """`out[i][j]` in the `j` loop: `i` is invariant there, so the
        elements are still distinct."""
        @fp.fpy(ctx=fp.FP64)
        def f(A: list[list[fp.Real]]):
            out = fp.empty(len(A), len(A[0]))
            for i in range(len(A)):
                for j in range(len(A[0])):
                    out[i][j] = A[i][j] * 2
            return out

        assert _why(f) is None

    def test_an_exact_accumulation_may_be_regrouped(self):
        """Under `INTEGER`, which is unbounded, no partial sum rounds."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs:
                acc = acc + fp.round(x)
            return acc

        assert _why(f) is None


class TestRefuses:
    def test_a_rounded_accumulation(self):
        """The case the whole predicate exists for: regrouping rounded adds
        moves bits, so this stays sequential per lane."""
        @fp.fpy(ctx=fp.FP32)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs:
                acc = acc + x
            return acc

        assert 'regrouping it moves bits' in (_why(f) or '')

    def test_two_iterations_writing_the_same_element(self):
        """`out[i % 2]` -- refused rather than sent to a dependence test."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real]):
            out = [fp.round(0) for _ in range(2)]
            for i in range(len(xs)):
                out[i % 2] = xs[i]
            return out

        assert 'cannot show distinct' in (_why(f) or '')

    def test_an_index_that_ignores_the_loop_variable(self):
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real], k: fp.Real):
            out = [fp.round(0) for _ in xs]
            for i in range(len(xs)):
                out[k] = xs[i]
            return out

        assert 'same index' in (_why(f) or '')

    def test_reading_the_list_back_while_writing_it(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            out = [fp.round(0) for _ in xs]
            for i in range(1, len(xs)):
                out[i] = out[i - 1] + xs[i]
            return out

        assert 'read back while being written' in (_why(f) or '')

    def test_a_write_that_ignores_the_carried_value(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            last = fp.round(0)
            for x in xs:
                last = x * 2
            return last

        assert 'which iteration wrote last' in (_why(f) or '')


class TestApi:
    def test_a_loop_carrying_nothing_is_tileable(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            for x in xs:
                y = x * 2
            return fp.round(0)

        assert _why(f) is None

    def test_a_non_for_is_a_type_error(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return x

        with pytest.raises(TypeError, match='ForStmt'):
            why_not_tileable(f.ast.body.stmts[-1], f.ast)


class TestTileLoops:
    """`tile_loops` splits what `why_not_tileable` accepts and leaves the rest
    sequential."""

    def test_a_refused_loop_is_left_alone(self):
        """The rounding fold stays a plain loop; the max fold is tiled."""
        @fp.fpy(ctx=fp.FP32)
        def mixed(xs: list[fp.Real]):
            acc = fp.round(0)          # rounded adds -- refused
            for x in xs:
                acc = acc + x
            m = fp.round(0)            # max selects -- tileable
            for x in xs:
                m = max(m, x)
            return (acc, m)

        out = tile_loops(mixed.ast, 4).func
        src = out.format()
        assert 'for x in xs:' in src, 'the refused loop was rewritten'
        assert _count(out, IfStmt, If1Stmt) == 1, 'exactly one mask'

    def test_a_tileable_loop_becomes_a_guarded_nest(self):
        @fp.fpy(ctx=fp.FP64)
        def largest(xs: list[fp.Real]):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            return m

        out = tile_loops(largest.ast, 4).func
        assert _count(out, ForStmt) == 2
        assert _count(out, IfStmt, If1Stmt) == 1

    def test_values_are_preserved(self):
        @fp.fpy(ctx=fp.FP64)
        def largest(xs: list[fp.Real]):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            return m

        tiled = Function(tile_loops(largest.ast, 4).func, runtime=largest.runtime)
        for n in range(0, 10):
            xs = [float(k) - 4 for k in range(n)]
            assert repr(tiled(xs)) == repr(largest(xs)), n

    def test_a_loop_free_function_is_unchanged(self):
        @fp.fpy(ctx=fp.FP64)
        def plain(x: fp.Real):
            return x * 2

        assert _count(tile_loops(plain.ast, 4).func, ForStmt) == 0

    def test_width_must_be_positive(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return x

        for bad in (0, -2):
            with pytest.raises(ValueError, match='positive width'):
                tile_loops(f.ast, bad).func

    def test_a_non_funcdef_is_a_type_error(self):
        with pytest.raises(TypeError, match='FuncDef'):
            tile_loops(42, 4)

    def test_only_the_innermost_of_a_nest_is_tiled(self):
        """Both loops are tileable, but the target wants one tiled dimension:
        the outer becomes the program instance, the inner the tile."""
        @fp.fpy(ctx=fp.FP64)
        def nested(A: list[list[fp.Real]]):
            out = fp.empty(len(A), len(A[0]))
            for i in range(len(A)):
                for j in range(len(A[0])):
                    out[i][j] = A[i][j] * 2
            return out

        assert why_not_tileable(_innermost(nested), nested.ast) is None
        out = tile_loops(nested.ast, 4).func
        # outer left alone + the inner split into a pair
        assert _count(out, ForStmt) == 3
        assert _count(out, IfStmt, If1Stmt) == 1

    def test_a_nest_preserves_values(self):
        @fp.fpy(ctx=fp.FP64)
        def nested(A: list[list[fp.Real]]):
            out = fp.empty(len(A), len(A[0]))
            for i in range(len(A)):
                for j in range(len(A[0])):
                    out[i][j] = A[i][j] * 2
            return out

        tiled = Function(tile_loops(nested.ast, 4).func, runtime=nested.runtime)
        for rows, cols in ((1, 1), (2, 3), (3, 5), (2, 8)):
            A = [[float(r * cols + c) for c in range(cols)] for r in range(rows)]
            assert repr(tiled(A)) == repr(nested(A)), (rows, cols)

    def test_it_reports_which_loops_it_tiled(self):
        """The emitter has to know which loop carries a tile; recognizing one
        by shape would be pattern-matching this pass's output."""
        @fp.fpy(ctx=fp.FP64)
        def two(xs: list[fp.Real], ys: list[fp.Real]):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            n = fp.round(0)
            for y in ys:
                n = max(n, y)
            return (m, n)

        r = tile_loops(two.ast, 4)
        assert len(r.tiled) == 2
        # each reported loop is an outer chunk loop of the rewritten function
        all_loops = []

        class _V(DefaultVisitor):
            def _visit_for(self, s, ctx):
                all_loops.append(s)
                return super()._visit_for(s, ctx)

        _V()._visit_function(r.func, None)
        assert all(any(t is loop for loop in all_loops) for t in r.tiled)

    def test_a_symbolic_width_is_a_free_variable(self):
        """A tile's width is a compile-time parameter of the kernel, chosen by
        the launcher -- so the name, not a literal."""
        @fp.fpy(ctx=fp.FP64)
        def largest(xs: list[fp.Real], BLOCK: fp.Real):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            return m

        r = tile_loops(largest.ast, 'BLOCK')
        assert 'BLOCK' in r.func.format()
        tiled = Function(r.func, runtime=largest.runtime)
        for n in (0, 1, 5, 8, 9):
            xs = [float(k) - 3 for k in range(n)]
            for b in (1, 4, 8):
                assert repr(tiled(xs, b)) == repr(largest(xs, b)), (n, b)

    def test_a_bad_width_is_rejected(self):
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real):
            return x

        with pytest.raises(TypeError, match='int.*str'):
            tile_loops(f.ast, 1.5)


class TestGuards:
    """`tile_loops` names the guard `SplitLoop` put on each tile, so the
    emitter can tell the tile's mask from a branch of the program's own."""

    @staticmethod
    def _if1s(func) -> list[If1Stmt]:
        out: list[If1Stmt] = []

        class _V(DefaultVisitor):
            def _visit_if1(self, s, ctx):
                out.append(s)
                return super()._visit_if1(s, ctx)

        _V()._visit_function(func, None)
        return out

    def test_a_symbolic_width_is_guarded(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        r = tile_loops(f.ast, 'BLOCK')
        assert len(r.guards) == 1
        assert any(g is r.guards[0] for g in self._if1s(r.func))

    def test_a_dividing_literal_width_is_not_guarded(self):
        from fpy2.transform import Monomorphize
        from fpy2.types import ListType, RealType

        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], out: list[fp.Real]):
            for i in range(len(xs)):
                if xs[i] < 0:
                    out[i] = xs[i]
            return out

        lt = ListType(RealType(fp.FP64), 8)
        r = tile_loops(Monomorphize.apply(f.ast, None, [lt, lt]), 4)
        assert len(r.tiled) == 1
        # the program's own `if1` is all that is left, and it is no guard
        assert len(self._if1s(r.func)) == 1
        assert r.guards == []


class TestReductionsOption:
    """A target with no lowering of a reduction across the tile asks for the
    loops carrying a scalar to stay sequential."""

    def test_a_carried_scalar_is_tiled_by_default(self):
        assert len(tile_loops(_largest_for_tiling.ast, 4).tiled) == 1

    def test_it_stays_sequential_without_reductions(self):
        r = tile_loops(_largest_for_tiling.ast, 4, reductions=False)
        assert r.tiled == []

    def test_a_loop_carrying_only_writes_by_element_still_tiles(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], out: list[fp.Real]):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        assert len(tile_loops(f.ast, 4, reductions=False).tiled) == 1


@fp.fpy(ctx=fp.FP64)
def _largest_for_tiling(xs: list[fp.Real]):
    m = fp.round(0)
    for x in xs:
        m = max(m, x)
    return m
