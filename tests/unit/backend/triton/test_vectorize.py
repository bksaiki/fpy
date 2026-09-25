"""`why_not_tileable`: may a loop body be evaluated as a tile?

Splitting a loop preserves semantics; evaluating the inner body as a tile is
what can change the answer, and only through the loop-carried variables.
"""

import pytest

import fpy2 as fp
from fpy2 import Function, Module
from fpy2.ast.fpyast import ForStmt, FuncDef, If1Stmt, IfStmt
from fpy2.backend.triton import (
    TileResult,
    normalize_module,
    tile_loops,
    why_not_tileable,
)
from fpy2.transform import Monomorphize, Simplify, Specialize, ZipElim
from fpy2.transform.path import walk_stmts
from fpy2.types import ListType, RealType, Type
from fpy2.utils import NamedId

_R = RealType(fp.FP64)
_M, _N = NamedId('m'), NamedId('n')


def _loops(ast: FuncDef) -> list[ForStmt]:
    return [s for _, s in walk_stmts(ast) if isinstance(s, ForStmt)]


def _count(ast: FuncDef, *types: type) -> int:
    return sum(isinstance(s, types) for _, s in walk_stmts(ast))


def _why(func: Function) -> str | None:
    """Why the last loop in visit order is not tileable."""
    return why_not_tileable(_loops(func.ast)[-1], func.ast)


def _normal(func: Function, arg_types: list[Type]) -> FuncDef:
    m = Module()
    m.add(func, arg_types=arg_types)
    m = m.map(lambda _m, fd: ZipElim.apply(fd))
    spec = Specialize.apply(m, size_key=True)
    return normalize_module(spec).get(func.name).func.ast


def _targets(loops: list[ForStmt]) -> list[str]:
    return [str(s.target) for s in loops]


def _tile(result: TileResult) -> str:
    """The target the tile loop binds, which is the row loop's."""
    loop = next(s for s in result.tiled[0].body.stmts if isinstance(s, ForStmt))
    guard = loop.body.stmts[0]
    return str(guard.body.stmts[0].target)


@fp.fpy(ctx=fp.FP64)
def _dot(A: list[fp.Real], B: list[fp.Real]) -> fp.Real:
    prods = [a * b for a, b in zip(A, B)]
    return max(prods)


@fp.fpy(ctx=fp.FP64)
def _mm(A, BT, out):
    for i in range(len(out)):
        row = out[i]
        for j in range(len(row)):
            row[j] = _dot(A[i], BT[j])
    return out


def _matmul(rows: int | NamedId, cols: int | NamedId, k: int = 4) -> FuncDef:
    return _normal(_mm, [
        ListType(ListType(_R, k), rows), ListType(ListType(_R, k), cols),
        ListType(ListType(_R, cols), rows),
    ])


@fp.fpy(ctx=fp.FP64)
def _nested(A: list[list[fp.Real]]):
    out = fp.empty(len(A), len(A[0]))
    for i in range(len(A)):
        for j in range(len(A[0])):
            out[i][j] = A[i][j] * 2
    return out


@fp.fpy(ctx=fp.FP64)
def _fixed_row(A: list[list[fp.Real]], out: list[list[fp.Real]]):
    """`row = out[0]`: every iteration writes the same row."""
    for i in range(len(A)):
        row = out[0]
        for j in range(len(row)):
            row[j] = A[i][j]
    return out


@fp.fpy(ctx=fp.FP64)
def _ident(x: fp.Real):
    return x


class TestTileable:
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
        assert _why(_nested) is None


class TestRefuses:
    def test_a_carried_scalar(self):
        """Even an exact `+`: the emitter lowers no reduction across a tile."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(xs: list[fp.Real]):
            acc = fp.round(0)
            for x in xs:
                acc = acc + fp.round(x)
            return acc

        assert 'carried whole' in (_why(f) or '')

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

    def test_reading_the_list_back_through_a_temporary(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], a: list[fp.Real]):
            for i in range(len(xs)):
                with fp.INTEGER:
                    j = len(xs) - 1 - i
                t = a[j]
                a[i] = t + xs[i]
            return a

        assert 'read back while being written' in (_why(f) or '')

    def test_reading_another_list_the_loop_writes(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], a: list[fp.Real], b: list[fp.Real]):
            for i in range(len(xs)):
                with fp.INTEGER:
                    j = len(xs) - 1 - i
                t = b[j]
                a[i] = t + xs[i]
                b[i] = xs[i]
            return a

        assert 'read back while being written' in (_why(f) or '')

    def test_a_scatter_through_a_list_of_indices(self):
        """`for k in ks` may repeat an index, so the last write is the answer."""
        @fp.fpy(ctx=fp.FP64)
        def f(ks: list[fp.Real], xs: list[fp.Real], out: list[fp.Real]):
            for k in ks:
                out[k] = xs[k]
            return out

        assert 'cannot show distinct' in (_why(f) or '')

    def test_a_write_through_an_alias_at_a_fixed_row(self):
        """`row = out[0]` names the same row each iteration."""
        @fp.fpy(ctx=fp.FP64)
        def f(A: list[list[fp.Real]], out: list[list[fp.Real]]):
            for i in range(len(A)):
                row = out[0]
                row[0] = A[i][0]
            return out

        assert _why(f) is not None

    def test_a_write_through_an_alias_in_a_nested_loop(self):
        """`row[j]` writes `out[0]`, whichever `i`."""
        assert why_not_tileable(_loops(_fixed_row.ast)[0], _fixed_row.ast) is not None


class TestApi:
    def test_a_loop_carrying_nothing_is_tileable(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real]):
            for x in xs:
                _y = x * 2
            return fp.round(0)

        assert _why(f) is None

    def test_a_non_for_is_a_type_error(self):
        with pytest.raises(TypeError, match='ForStmt'):
            why_not_tileable(_ident.ast.body.stmts[-1], _ident.ast)


@fp.fpy(ctx=fp.FP64)
def _double(xs: list[fp.Real], out: list[fp.Real]):
    for i in range(len(xs)):
        out[i] = xs[i] * 2
    return out


class TestTileLoops:
    """`tile_loops` splits what `why_not_tileable` accepts and leaves the rest
    sequential."""

    def test_a_refused_loop_is_left_alone(self):
        @fp.fpy(ctx=fp.FP64)
        def largest(xs: list[fp.Real]):
            m = fp.round(0)
            for x in xs:
                m = max(m, x)
            return m

        r = tile_loops(largest.ast, 4)
        assert r.tiled == []
        assert _count(r.func, ForStmt) == 1

    def test_a_tileable_loop_becomes_a_guarded_nest(self):
        out = tile_loops(_double.ast, 4).func
        assert _count(out, ForStmt) == 2
        assert _count(out, IfStmt, If1Stmt) == 1

    def test_values_are_preserved(self):
        tiled = Function(tile_loops(_double.ast, 4).func, runtime=_double.runtime)
        for n in range(10):
            xs = [float(k) - 4 for k in range(n)]
            assert repr(tiled(xs, [0.0] * n)) == repr(_double(xs, [0.0] * n)), n

    def test_a_loop_free_function_is_unchanged(self):
        @fp.fpy(ctx=fp.FP64)
        def plain(x: fp.Real):
            return x * 2

        assert _count(tile_loops(plain.ast, 4).func, ForStmt) == 0

    def test_width_must_be_positive(self):
        for bad in (0, -2):
            with pytest.raises(ValueError, match='positive width'):
                tile_loops(_ident.ast, bad)

    def test_a_non_funcdef_is_a_type_error(self):
        with pytest.raises(TypeError, match='FuncDef'):
            tile_loops(42, 4)

    def test_only_the_innermost_of_a_nest_is_tiled(self):
        """Both loops are tileable, but the target wants one tiled dimension:
        the outer becomes the program instance, the inner the tile."""
        assert why_not_tileable(_loops(_nested.ast)[0], _nested.ast) is None
        out = tile_loops(_nested.ast, 4).func
        # outer left alone + the inner split into a pair
        assert _count(out, ForStmt) == 3
        assert _count(out, IfStmt, If1Stmt) == 1

    def test_a_nest_preserves_values(self):
        tiled = Function(tile_loops(_nested.ast, 4).func, runtime=_nested.runtime)
        for rows, cols in ((1, 1), (2, 3), (3, 5), (2, 8)):
            A = [[float(r * cols + c) for c in range(cols)] for r in range(rows)]
            assert repr(tiled(A)) == repr(_nested(A)), (rows, cols)

    def test_it_reports_which_loops_it_tiled(self):
        """The emitter has to know which loop carries a tile; recognizing one
        by shape would be pattern-matching this pass's output."""
        @fp.fpy(ctx=fp.FP64)
        def two(xs: list[fp.Real], ys: list[fp.Real], a: list[fp.Real], b: list[fp.Real]):
            for i in range(len(xs)):
                a[i] = xs[i]
            for j in range(len(ys)):
                b[j] = ys[j]
            return a

        r = tile_loops(two.ast, 4)
        assert len(r.tiled) == 2
        # each reported loop is a loop of the rewritten function
        assert all(any(t is loop for loop in _loops(r.func)) for t in r.tiled)

    def test_a_symbolic_width_is_a_free_variable(self):
        """A tile's width is a compile-time parameter of the kernel, chosen by
        the launcher -- so the name, not a literal."""
        @fp.fpy(ctx=fp.FP64)
        def double(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i] * 2
            return out

        r = tile_loops(double.ast, 'BLOCK')
        assert 'BLOCK' in r.func.format()
        tiled = Function(r.func, runtime=double.runtime)
        for n in (0, 1, 5, 8, 9):
            xs = [float(k) - 3 for k in range(n)]
            for b in (1, 4, 8):
                assert repr(tiled(xs, [0.0] * n, b)) == repr(double(xs, [0.0] * n, b)), (n, b)

    def test_a_bad_width_is_rejected(self):
        with pytest.raises(TypeError, match='int.*str'):
            tile_loops(_ident.ast, 1.5)


class TestGuards:
    """`tile_loops` names the guard `SplitLoop` put on each tile, so the
    emitter can tell the tile's mask from a branch of the program's own."""

    def test_a_symbolic_width_is_guarded(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs: list[fp.Real], out: list[fp.Real], BLOCK: fp.Real):
            for i in range(len(xs)):
                out[i] = xs[i]
            return out

        r = tile_loops(f.ast, 'BLOCK')
        assert len(r.guards) == 1
        assert any(g is r.guards[0] for _, g in walk_stmts(r.func))

    def test_a_dividing_literal_width_is_not_guarded(self):
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
        assert _count(r.func, If1Stmt) == 1
        assert r.guards == []


class TestLanes:
    """The rows are the outputs, and the loops of static count beneath that
    write only their own elements run across a tile's lanes."""

    def test_a_matmul_tiles_its_columns_across_a_comprehension(self):
        out = tile_loops(_matmul(_M, _N), 4)
        assert _tile(out) == 'j'
        assert len(out.lanes) == 1
        f = Function(out.func, runtime=_mm.runtime)
        A, BT = [[1.0, -2.0, 3.0, 0.5]] * 3, [[2.0, 1.0, -1.0, 4.0]] * 5
        want = _mm(A, BT, [[0.0] * 5 for _ in range(3)])
        assert repr(f(A, BT, [[0.0] * 5 for _ in range(3)])) == repr(want)

    def test_with_no_runtime_count_the_outermost_is_the_row(self):
        assert _tile(tile_loops(_matmul(3, 5), 4)) == 'i'

    def test_a_runtime_count_is_a_row_not_a_lane(self):
        """A comprehension over a list of runtime length is the grid's."""
        @fp.fpy(ctx=fp.FP64)
        def f(xs, out):
            ys = [x * 2 for x in xs]
            for k in range(len(out)):
                out[k] = ys[k]
            return out

        ast = _normal(f, [ListType(_R, _N), ListType(_R, _N)])
        out = tile_loops(ast, 4)
        assert out.lanes == []

    def test_a_carried_value_is_not_a_lane(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xss, out):
            for j in range(len(out)):
                acc = fp.round(0)
                for x in xss[j]:
                    acc = acc + x
                out[j] = acc
            return out

        ast = _normal(f, [ListType(ListType(_R, 4), _N), ListType(_R, _N)])
        out = tile_loops(ast, 4)
        assert _tile(out) == 'j'
        assert out.lanes == []

    def test_a_loop_holding_a_lane_is_sequential(self):
        """nvfp4's group loop: each group's comprehension runs across the
        lanes, so the loop over groups cannot as well.  Under `REAL`, as
        nvfp4 is, so the slice's length is proven."""
        @fp.fpy(ctx=fp.REAL)
        def f(xss, out):
            for j in range(len(out)):
                xs = xss[j]
                ms = fp.empty(2)
                for g in range(2):
                    ms[g] = max([x * 2 for x in xs[g * 2:g * 2 + 2]])
                out[j] = ms[0] + ms[1]
            return out

        ast = _normal(f, [ListType(ListType(_R, 4), _N), ListType(_R, _N)])
        out = tile_loops(ast, 4)
        assert _tile(out) == 'j'
        assert 'g' not in _targets(out.lanes)
        assert len(out.lanes) == 1

    def test_a_lane_reading_its_tile_at_another_lane_is_not_one(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xs, out):
            for r in range(len(xs)):
                acc = [xs[r][k] for k in range(4)]
                for k in range(4):
                    with fp.INTEGER:
                        j = 3 - k
                    t = acc[j]
                    acc[k] = t + 1
                for k in range(4):
                    out[r][k] = acc[k]
            return out

        L = ListType(ListType(_R, 4), _M)
        out = tile_loops(_normal(f, [L, L]), 4)
        assert not any('3 - k' in s.format() for s in out.lanes)
        assert len(out.lanes) == 2

    def test_a_rewrite_finds_the_lanes_again(self):
        out = tile_loops(_matmul(_M, _N), 4)
        again = out.rewritten(Simplify.apply(out.func))
        assert _targets(again.lanes) == _targets(out.lanes)


class TestTheGridsSecondAxis:
    """The loop directly around a lone tile, carrying nothing and inside no
    other, is one program per iteration."""

    def test_a_matmul_takes_its_rows(self):
        out = tile_loops(_matmul(_M, _N), 4)
        assert _targets(out.grid) == ['i']
        again = out.rewritten(Simplify.apply(out.func))
        assert _targets(again.grid) == ['i']

    def test_a_carrying_loop_is_not_one(self):
        @fp.fpy(ctx=fp.FP64)
        def f(xss, out):
            acc = fp.round(0)
            for i in range(len(xss)):
                acc = acc + xss[i][0]
                for j in range(len(out)):
                    out[j] = xss[i][j] * 2
            return out

        ast = _normal(f, [ListType(ListType(_R, _N), _M), ListType(_R, _N)])
        assert tile_loops(ast, 4).grid == []

    def test_a_loop_writing_through_a_fixed_row_is_not_one(self):
        L = ListType(ListType(_R, _N), _M)
        assert tile_loops(_normal(_fixed_row, [L, L]), 4).grid == []

