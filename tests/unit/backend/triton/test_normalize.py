"""The Triton normal form: `fpy2.backend.triton.normalize`.

The form is calls inlined away, one exit, and every `if` a value;
comprehensions and derived iterables stay, since the vectorizer wants the
iteration written down.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.ast.fpyast import Call, If1Stmt, IfStmt, ReturnStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2 import Module
from fpy2.backend.triton import (
    TritonNormalizeError,
    normalize,
    normalize_module,
)
from fpy2.transform import TransformDeclined


def _count(ast, *types) -> int:
    n = 0

    class _V(DefaultVisitor):
        def _visit_statement(self, stmt, ctx):
            nonlocal n
            if isinstance(stmt, types):
                n += 1
            return super()._visit_statement(stmt, ctx)

        def _visit_call(self, e, ctx):
            nonlocal n
            if Call in types and isinstance(e.fn, Function):
                n += 1
            return super()._visit_call(e, ctx)

    _V()._visit_function(ast, None)
    return n


def _is_normal(ast) -> None:
    assert _count(ast, IfStmt, If1Stmt) == 0, 'an `if` statement remains'
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

    def test_a_comprehension_is_kept(self):
        """Item 2 wants the iteration written down, so nothing lowers it."""
        @fp.fpy(ctx=fp.FP64)
        def comp(xs: list[fp.Real]):
            return [x * 2 for x in xs]

        src = normalize(comp.ast).format()
        assert 'for' in src


class TestAGuardedLoopIsHoisted:
    """`SimplifyIf` hoists a `for` out of an arm: it terminates, and the
    merge discards what it computed on the side the guard did not take.  So
    the normal form is reached rather than refused."""

    def test_it_normalizes(self):
        @fp.fpy(ctx=fp.FP64)
        def guarded(c: bool, xs: list[fp.Real]):
            total = fp.round(0)
            if c:
                for x in xs:
                    total = total + x
            return total

        out = normalize(guarded.ast)
        _is_normal(out)
        g = Function(out, runtime=guarded.runtime)
        for c in (True, False):
            assert repr(g(c, [1.0, 2.0])) == repr(guarded(c, [1.0, 2.0]))


class TestRejects:
    def test_an_assert_under_a_guard_declines(self):
        """Hoisting it would make it run unconditionally, and it can abort.

        The pass's own `TransformDeclined` propagates rather than being
        rewrapped: it already names the construct.
        """
        @fp.fpy(ctx=fp.FP64)
        def guarded(c: bool, xs: list[fp.Real]):
            total = fp.round(0)
            if c:
                for x in xs:
                    assert x > 0, 'positive'
                    total = total + x
            return total

        with pytest.raises(TransformDeclined, match='`assert` would run'):
            normalize(guarded.ast)

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
