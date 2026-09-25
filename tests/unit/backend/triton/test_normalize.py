"""The Triton normal form: `fpy2.backend.triton.normalize`.

The form is calls inlined away and one exit.  An `if` statement stays, for the
emitter to flatten after the analyses have read its guard; a comprehension
becomes a loop, for the lanes.
"""

import pytest

import fpy2 as fp
from fpy2 import Function, Module
from fpy2.analysis import ArraySizeInfer, FormatInfer
from fpy2.ast.fpyast import (
    AssertStmt,
    Call,
    ForStmt,
    FuncDef,
    If1Stmt,
    IfStmt,
    ListComp,
    ReturnStmt,
    Sum,
)
from fpy2.backend.triton import (
    TritonNormalizeError,
    normalize,
    normalize_module,
)
from fpy2.backend.triton.storage import bound_fits_in_scalar
from fpy2.backend.triton.types import TritonScalar
from fpy2.transform import FreeVarElim, Specialize, ZipElim
from fpy2.transform.path import walk_exprs, walk_stmts
from fpy2.types import ListType, RealType


def _count(ast: FuncDef, *types: type) -> int:
    """Statements and expressions of *types*; a `Call` only to an FPy function."""
    nodes = [s for _, s in walk_stmts(ast)] + [e for _, e in walk_exprs(ast)]
    return sum(
        isinstance(n, types) and (not isinstance(n, Call) or isinstance(n.fn, Function))
        for n in nodes)


def _is_normal(ast: FuncDef) -> None:
    assert _count(ast, ReturnStmt) == 1, 'not a single exit'
    assert _count(ast, Call) == 0, 'a call remains'


@fp.fpy(ctx=fp.FP64)
def _scale(x: fp.Real) -> fp.Real:
    if x < 0:
        return -x
    return x


@fp.fpy(ctx=fp.FP64)
def _caller(x: fp.Real, y: fp.Real) -> fp.Real:
    a = _scale(x)
    if y > 0:
        a = a + y
    return a


def _normalized(func: Function) -> Function:
    """*func* through the module-level pass, which is the one that can reach a
    callee.  Returns the transformed entry."""
    m = Module()
    m.add(func)
    return normalize_module(m).get(func.name).func


class TestReachesTheForm:
    def test_all_three_at_once(self):
        """`_caller` needs every pass: a call, two exits in the callee, and an
        `if` of its own."""
        _is_normal(_normalized(_caller).ast)

    def test_values_are_preserved(self):
        out = _normalized(_caller)
        for x in (-2.0, 0.0, 3.0):
            for y in (-1.0, 0.0, 1.5):
                assert repr(out(x, y)) == repr(_caller(x, y)), (x, y)

    def test_a_multi_exit_callee_alone_is_not_reachable(self):
        """`normalize` holds only the caller, so it cannot make `_scale`
        single-exit and `FuncInline` keeps refusing it.  This is why the
        module-level pass exists."""
        with pytest.raises(TritonNormalizeError, match='call to `_scale` remains'):
            normalize(_caller.ast)

    def test_an_already_normal_function_is_unchanged_in_meaning(self):
        @fp.fpy(ctx=fp.FP64)
        def plain(x: fp.Real) -> fp.Real:
            return x * 2

        out = Function(normalize(plain.ast), runtime=plain.runtime)
        _is_normal(out.ast)
        assert repr(out(3.0)) == repr(plain(3.0))


class TestAnIfIsKept:
    """The emitter flattens an `if`; the normal form leaves it for the
    analyses to read its guard first."""

    def test_the_callees_exits_become_an_if(self):
        assert _count(_normalized(_caller).ast, IfStmt, If1Stmt) == 2

    def test_a_guarded_loop_stays_guarded(self):
        @fp.fpy(ctx=fp.FP64)
        def guarded(c: bool, xs: list[fp.Real]):
            total = fp.round(0)
            if c:
                for x in xs:
                    total = total + x
            return total

        out = normalize(guarded.ast)
        _is_normal(out)
        assert _count(out, If1Stmt) == 1
        g = Function(out, runtime=guarded.runtime)
        for c in (True, False):
            assert repr(g(c, [1.0, 2.0])) == repr(guarded(c, [1.0, 2.0]))


class TestAssertsAndLoops:
    def test_an_assert_under_a_guard_is_left_to_the_emitter(self):
        """Nothing is hoisted, so it runs only where the guard held; whether a
        kernel can spell it is `drop_asserts`'s question, at emission."""
        @fp.fpy(ctx=fp.FP64)
        def guarded(c: bool, xs: list[fp.Real]):
            total = fp.round(0)
            if c:
                for x in xs:
                    assert x > 0, 'positive'
                    total = total + x
            return total

        assert _count(normalize(guarded.ast), AssertStmt) == 1

    def test_a_varying_while_condition_is_not_normal(self):
        """A tile-wide loop runs a fixed number of times; a condition the body
        moves makes the trip count per-lane."""
        @fp.fpy(ctx=fp.FP64)
        def loop(n: fp.Real):
            i = fp.round(0)
            while i < n:
                i = i + 1
            return i

        with pytest.raises(TritonNormalizeError, match='`while` condition varies'):
            normalize(loop.ast)

    def test_a_fixed_while_condition_is_accepted(self):
        @fp.fpy(ctx=fp.FP64)
        def loop(c: bool, x: fp.Real):
            while c:
                x = x + 1
            return x

        _is_normal(normalize(loop.ast))

    def test_a_non_funcdef_is_a_type_error(self):
        with pytest.raises(TypeError, match='FuncDef'):
            normalize(_caller)


class TestLanes:
    """A comprehension becomes a loop, which gives a call in one a statement
    and keeps the iteration for the lanes."""

    def test_a_call_inside_a_comprehension_is_inlined(self):
        """`FuncInline` splices a callee's body into a statement list, so it
        cannot take a call inside a comprehension until that is a loop."""
        @fp.fpy(ctx=fp.FP64)
        def bump(x: fp.Real) -> fp.Real:
            t = x + 1
            return t

        @fp.fpy(ctx=fp.FP64)
        def uses(xs: list[fp.Real]):
            ys = [bump(x) for x in xs]
            return ys[0] + ys[2]

        out = _normalized(uses)
        _is_normal(out.ast)
        assert _count(out.ast, ForStmt) == 1
        assert _count(out.ast, ListComp) == 0
        args = [1.0, 2.0, 3.0]
        assert repr(out(args)) == repr(uses(args))

    def test_a_fused_sum_of_a_join(self):
        """The shape every T-FDPA ends in: products and an accumulator joined,
        each term truncated at one position, and summed.  The analyses read it
        without refusing."""
        @fp.fpy(ctx=fp.REAL)
        def join(xs, ys):
            n = len(xs)
            m = len(ys)
            zs = fp.empty(n + m)
            for i in range(n):
                zs[i] = xs[i]
            for i in range(m):
                zs[n + i] = ys[i]
            return zs

        @fp.fpy(ctx=fp.REAL)
        def fused(xs, n):
            with fp.MPFixedContext(n, fp.RM.RTZ):
                ts = [fp.round(x) for x in xs]
            return sum(ts)

        @fp.fpy(ctx=fp.REAL)
        def dpa(A: list[fp.Real], B: list[fp.Real], c: fp.Real):
            prods = [a * b for a, b in zip(A, B)]
            return fused(join(prods, [c]), -30)

        out = _normalized(dpa)
        _is_normal(out.ast)
        ArraySizeInfer.analyze(out.ast)
        FormatInfer.analyze(out.ast, use_digit_bounds=True)
        args = ([1.5, -2.25, 3.0], [0.5, 4.0, -1.0], 0.125)
        assert repr(out(*args)) == repr(dpa(*args))

    def test_an_early_return_scan(self):
        """A loop that returns from inside stays a loop, its exits flags."""
        @fp.fpy(ctx=fp.FP64)
        def first_neg(xs: list[fp.Real]) -> fp.Real:
            for x in xs:
                if x < 0:
                    return x
            return 0.0

        @fp.fpy(ctx=fp.FP64)
        def uses(xs: list[fp.Real]):
            neg = [-x for x in xs]
            return first_neg(neg)

        out = _normalized(uses)
        _is_normal(out.ast)
        for xs in ([1.0, -2.0, 3.0], [-1.0, 2.0], [0.0]):
            assert repr(out(xs)) == repr(uses(xs))


@fp.fpy(ctx=fp.REAL)
def _exponent0(x, emin):
    if not fp.isfinite(x):
        return -1
    return max(fp.logb(x), emin)


@fp.fpy(ctx=fp.REAL)
def _fused_sum(xs, n):
    with fp.MPFixedContext(n, fp.RM.RTZ):
        ts = [fp.round(x) for x in xs]
    return sum(ts)


@fp.fpy(ctx=fp.REAL)
def _evens(A: list[fp.Real], B: list[fp.Real]):
    """cdna3.bf8's shape: a truncated sum over the even-index products,
    aligned at the even exponents' greatest, past a finiteness guard."""
    prods = [a * b for a, b in zip(A, B)]
    es = [-40 if p == 0 else _exponent0(a, -15) + _exponent0(b, -15)
          for p, a, b in zip(prods, A, B)]
    if any([not fp.isfinite(p) for p in prods]):  # noqa: C419 -- FPy has no generators
        return 0.0
    e = max([es[i] for i in range(0, 8, 2)])
    return _fused_sum([prods[i] for i in range(0, 8, 2)], e - 25)


def test_a_gathered_sum_keeps_its_bound_past_a_guard():
    """The guard is stated on each element as it is read, and the evens are
    a part of the list: the part needs "every element is finite" to carry
    the bound, which is what makes the sum fit a `double`."""
    fp16 = RealType(fp.IEEEContext(5, 16))
    m = Module()
    m.add(_evens, arg_types=[ListType(fp16, 8), ListType(fp16, 8)])
    m = m.map(lambda _m, fd: ZipElim.apply(FreeVarElim.apply(fd)))
    ast = normalize_module(Specialize.apply(m, size_key=True)).get(
        '_evens').func.ast
    fmt = FormatInfer.analyze(ast, use_digit_bounds=True)
    sums = [fmt.by_expr.get(e) for _, e in walk_exprs(ast) if isinstance(e, Sum)]
    assert sums and all(bound_fits_in_scalar(b, TritonScalar.F64) for b in sums)
