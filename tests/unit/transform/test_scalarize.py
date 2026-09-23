"""
Unit tests for :class:`fpy2.transform.Scalarize`.

The pass replaces a proven-length comprehension with a **list literal of the
element names**, rather than deleting the list.  Every existing use therefore
keeps working -- `len`, a subscript, a fold, a call taking the whole list --
and what actually moves is where the element expressions are *evaluated*:
into statement positions.  That is the point, because `FuncInline` splices a
callee's body into the enclosing statement list and so cannot reach a call
inside a comprehension.

So the tests are in two halves: the meaning is preserved (interpreter), and
the calls ended up somewhere `FuncInline` can take them.
"""

import pytest

import fpy2 as fp
from fpy2 import Function, Module
from fpy2.ast import fpyast as A
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import FuncInline, Scalarize, Specialize
from fpy2.types import ListType, RealType


def _comps(ast) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            nonlocal n
            if isinstance(e, A.ListComp):
                n += 1
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return n


def _fpy_calls(ast) -> int:
    n = 0

    class _C(DefaultVisitor):
        def _visit_call(self, e, ctx):
            nonlocal n
            if isinstance(e.fn, Function):
                n += 1
            super()._visit_call(e, ctx)

    _C()._visit_function(ast, None)
    return n


def _agrees(f: Function, *args, **kw):
    out = Scalarize.apply(f.ast, **kw)
    assert repr(f.with_ast(out)(*args)) == repr(f(*args))
    return out


def _sized(f: Function, *lengths: int) -> Function:
    """*f* with its list arguments pinned, so lengths are proven."""
    m = Module()
    m.add(f, ctx=fp.FP32,
          arg_types=[ListType(RealType(fp.FP32), n) for n in lengths])
    return Specialize.apply(m, size_key=True).get(f.name).func


@fp.fpy(ctx=fp.FP32)
def _helper(x: fp.Real) -> fp.Real:
    t = x + 1
    return t


@fp.fpy(ctx=fp.FP32)
def _over_range(xs: list[fp.Real]):
    ys = [xs[i] * 2 for i in range(4)]
    return ys[0] + ys[3]


@fp.fpy(ctx=fp.FP32)
def _calls_inside(xs: list[fp.Real]):
    ys = [_helper(xs[i]) for i in range(3)]
    return max(ys)


@fp.fpy(ctx=fp.FP32)
def _over_seq(xs: list[fp.Real]):
    ys = [x * 2 for x in xs]
    return sum(ys)


@fp.fpy(ctx=fp.FP32)
def _lazy_arm(xs: list[fp.Real], c: fp.Real):
    return max([xs[i] for i in range(4)]) if c > 0 else c


class TestItUnrolls:
    def test_a_comprehension_over_a_range(self):
        out = _agrees(_over_range, [1.0, 2.0, 3.0, 4.0])
        assert _comps(out) == 0

    def test_a_comprehension_over_a_proven_sequence(self):
        g = _sized(_over_seq, 4)
        out = Scalarize.apply(g.ast)
        assert _comps(out) == 0
        args = [1.0, 2.0, 3.0, 4.0]
        assert repr(g.with_ast(out)(args)) == repr(g(args))

    def test_the_list_survives_as_a_literal(self):
        """Not deleted: `sum(ys)` still has a `ys` to fold."""
        g = _sized(_over_seq, 4)
        assert 'ys = [' in Function(Scalarize.apply(g.ast), runtime=None).format()


class TestTheCallsBecomeReachable:
    """The reason the pass exists."""

    def test_a_call_moves_into_a_statement(self):
        out = _agrees(_calls_inside, [1.0, 2.0, 3.0])
        assert _fpy_calls(out) == 3, 'one per element, all at statement level'

    def test_func_inline_can_then_take_them(self):
        """Before the pass it refuses; after it, nothing is left."""
        assert _fpy_calls(FuncInline.apply(_calls_inside.ast, recursive=True)) == 1
        out = Scalarize.apply(_calls_inside.ast)
        assert _fpy_calls(FuncInline.apply(out, recursive=True)) == 0


class TestItLeavesAlone:
    """Declining to unroll is not a refusal -- the program still compiles,
    by whatever path takes a sequence whose length it does not know."""

    def test_an_unproven_length(self):
        out = Scalarize.apply(_over_seq.ast)
        assert _comps(out) == 1

    def test_over_the_cap(self):
        g = _sized(_over_seq, 4)
        assert _comps(Scalarize.apply(g.ast, cap=2)) == 1
        assert _comps(Scalarize.apply(g.ast, cap=4)) == 0

    def test_a_lazily_evaluated_arm(self):
        """Hoisting out of an `IfExpr` would make it unconditional, which is
        the hazard `SimplifyIf` refuses over."""
        out = _agrees(_lazy_arm, [1.0, 2.0, 3.0, 4.0], -1.0)
        assert _comps(out) == 1


class TestTheInterface:
    def test_it_rejects_a_non_funcdef(self):
        with pytest.raises(TypeError, match='FuncDef'):
            Scalarize.apply(_over_range)

    def test_it_rejects_a_negative_cap(self):
        with pytest.raises(ValueError, match='non-negative'):
            Scalarize.apply(_over_range.ast, cap=-1)

    def test_a_function_with_no_sequences_is_untouched(self):
        @fp.fpy(ctx=fp.FP32)
        def plain(x: fp.Real):
            return x * 2

        assert Scalarize.apply(plain.ast).is_equiv(plain.ast)


@fp.fpy(ctx=fp.FP32)
def _join(xs: list[fp.Real], ys: list[fp.Real]):
    """Allocate-and-fill: how FPy builds a list of computed length, since it
    has no concatenation.  Taken from `examples/mmasim/models/utils.py`."""
    n = len(xs)
    m = len(ys)
    zs = fp.empty(n + m)
    for i in range(n):
        zs[i] = xs[i]
    for i in range(m):
        zs[n + i] = ys[i]
    return zs[0] + zs[3]


class TestAllocateAndFill:
    """`fp.empty(n)` plus the stores that fill it becomes plain values.

    Sound only under four conditions, so each is checked rather than
    assumed: the length is a constant within the cap, every store is at a
    constant index in range, each index is written exactly once, and nothing
    reads the list before the last store.
    """

    def test_it_unrolls(self):
        g = _sized(_join, 2, 3)
        out = Scalarize.apply(g.ast)
        assert 'fp.empty' not in Function(out, runtime=None).format()
        args = ([1.0, 2.0], [3.0, 4.0, 5.0])
        assert repr(g.with_ast(out)(*args)) == repr(g(*args))

    def test_the_length_may_come_from_len(self):
        """Inlining binds `n = len(xs)`, so the length is a `Len` rather than
        the literal `ConstFold` would have left."""
        g = _sized(_join, 2, 3)
        assert 'zs = [' in Function(Scalarize.apply(g.ast), runtime=None).format()

    def test_a_partial_fill_is_left_alone(self):
        @fp.fpy(ctx=fp.FP32)
        def partial(xs: list[fp.Real]):
            zs = fp.empty(3)
            zs[0] = xs[0]
            zs[1] = xs[1]
            return zs[0]

        out = Scalarize.apply(_sized(partial, 2).ast)
        assert 'fp.empty' in Function(out, runtime=None).format()

    def test_a_read_before_the_last_store_is_left_alone(self):
        @fp.fpy(ctx=fp.FP32)
        def reads_early(xs: list[fp.Real]):
            zs = fp.empty(2)
            zs[0] = xs[0]
            zs[1] = zs[0] + 1.0
            return zs[1]

        out = Scalarize.apply(_sized(reads_early, 2).ast)
        assert 'fp.empty' in Function(out, runtime=None).format()

    def test_over_the_cap_is_left_alone(self):
        g = _sized(_join, 2, 3)
        assert 'fp.empty' in Function(
            Scalarize.apply(g.ast, cap=4), runtime=None).format()


@fp.fpy(ctx=fp.FP32)
def _loop_over_values(xs: list[fp.Real]):
    ps = [xs[i] * 2 for i in range(3)]
    s = fp.round(0)
    for p in ps:
        s = s + p
    return s


@fp.fpy(ctx=fp.FP32)
def _loop_over_memory(xs: list[fp.Real]):
    s = fp.round(0)
    for x in xs:
        s = s + x
    return s


class TestLoopsOverValues:
    """A loop over a list of *values* has no iteration to perform: there is
    no object to step through, only that many values.  A loop over something
    in *memory* does, and stays a loop -- which is what keeps a reduction
    over a tile rolled.
    """

    def test_a_loop_over_values_unrolls(self):
        out = _agrees(_loop_over_values, [1.0, 2.0, 3.0])
        assert 'for ' not in Function(out, runtime=None).format()

    def test_a_loop_over_memory_is_left_alone(self):
        g = _sized(_loop_over_memory, 4)
        out = Scalarize.apply(g.ast)
        assert 'for x in xs' in Function(out, runtime=None).format()
        args = [1.0, 2.0, 3.0, 4.0]
        assert repr(g.with_ast(out)(args)) == repr(g(args))

    def test_a_copy_of_a_value_list_still_unrolls(self):
        """Inlining binds a callee's parameter to the caller's list by name,
        so the loop it came with sees a copy rather than the literal."""
        @fp.fpy(ctx=fp.FP32)
        def via_copy(xs: list[fp.Real]):
            ps = [xs[i] * 2 for i in range(3)]
            qs = ps
            s = fp.round(0)
            for q in qs:
                s = s + q
            return s

        out = _agrees(via_copy, [1.0, 2.0, 3.0])
        assert 'for ' not in Function(out, runtime=None).format()

    def test_over_the_cap_is_left_alone(self):
        assert 'for p in ps' in Function(
            Scalarize.apply(_loop_over_values.ast, cap=2), runtime=None).format()
