"""
Unit tests for :class:`fpy2.transform.SingleExit`.

An early return becomes an assignment to one result name, and the statements
that would have followed move into the branch that falls through.  Nothing is
duplicated: a branch that returns cannot reach what follows.

Consumers that need it -- `FPCoreCompiler` rejects multiple returns,
`FuncInline` refuses a callee with more than one, `SimplifyIf` refuses a
`return` in a branch.

A `return` under a loop is refused; such loops unroll away where the trip count
is known, which is every case in `examples/mmasim`.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import Reachability
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import SingleExit, TransformDeclined


def _returns(ast) -> int:
    return len(Reachability.analyze(ast).ret_stmts)


def _agrees(f: Function, *args):
    g = Function(SingleExit.apply(f.ast), runtime=f.runtime)
    assert repr(g(*args)) == repr(f(*args))


@fp.fpy
def early_return(x):
    if x > 0:
        return 1.0
    y = x * 2
    return y


@fp.fpy
def both_arms_return(x):
    if x > 0:
        return 1.0
    else:
        return 2.0


@fp.fpy
def nested(x, y):
    if x > 0:
        if y > 0:
            return 1.0
        z = 2.0
    else:
        return 3.0
    return z


@fp.fpy
def returns_in_a_context(x):
    if x > 100.0:
        return 0.0
    with fp.FP32:
        return fp.round(x)


@fp.fpy
def already_single(x):
    y = x * 2
    return y


@fp.fpy
def returns_in_a_for(xs: list[fp.Real]):
    for x in xs:
        if x < 0:
            return x
    return 0.0


@fp.fpy
def returns_in_a_while(x):
    while x > 0:
        if x > 100.0:
            return x
        x = x - 1.0
    return x


_REWRITTEN = [early_return, both_arms_return, nested, returns_in_a_context]
_ARGS = {
    'early_return': [(1.0,), (-1.0,), (0.0,)],
    'both_arms_return': [(1.0,), (-1.0,)],
    'nested': [(a, b) for a in (1.0, -1.0) for b in (1.0, -1.0)],
    'returns_in_a_context': [(1.0,), (1000.0,)],
}


class TestOneReturnSurvives:
    @pytest.mark.parametrize('f', _REWRITTEN, ids=lambda f: f.name)
    def test_the_input_has_several(self, f):
        """Otherwise the assertion below holds trivially."""
        assert _returns(f.ast) > 1

    @pytest.mark.parametrize('f', _REWRITTEN, ids=lambda f: f.name)
    def test_exactly_one_remains(self, f):
        assert _returns(SingleExit.apply(f.ast)) == 1

    def test_a_single_return_is_left_alone(self):
        assert SingleExit.apply(already_single.ast) is already_single.ast


class TestSemanticsArePreserved:
    @pytest.mark.parametrize('f', _REWRITTEN, ids=lambda f: f.name)
    def test_agrees_with_the_interpreter(self, f):
        for args in _ARGS[f.name]:
            _agrees(f, *args)


class TestTheContinuationIsMovedNotCopied:
    def test_the_tail_appears_once(self):
        """`y = x * 2` follows the `if`; it must end up in the `else` exactly
        once, not duplicated into both arms."""
        out = Function(SingleExit.apply(early_return.ast), runtime=None).format()
        assert out.count('x * 2') == 1


class TestLoopReturnsAreRefused:
    @pytest.mark.parametrize('f', [returns_in_a_for, returns_in_a_while],
                             ids=lambda f: f.name)
    def test_declines(self, f):
        with pytest.raises(TransformDeclined, match='inside a loop'):
            SingleExit.apply(f.ast)


class TestUnhandledShapesRefuse:
    def test_a_shape_that_is_not_moved_declines(self):
        """The postcondition, not a specific shape: `_sink` handles `if` and a
        `with` whose body always returns.  Anything else must refuse rather
        than hand back a function with several exits."""
        for f in (returns_in_a_for, returns_in_a_while):
            with pytest.raises(TransformDeclined):
                SingleExit.apply(f.ast)


@fp.fpy
def both_arms_fall_through(p, q):
    if p > 0:
        if q > 0:
            return 1.0
        z = 2.0
    else:
        z = 3.0
    return z


@fp.fpy
def one_armed_inner(x, y):
    z = 0.0
    if x > 0:
        if y > 0:
            return 1.0
        z = 2.0
    return z


class TestBothArmsFallThrough:
    """A guard nested inside a guard: neither arm of the outer `if` returns
    unconditionally, yet one contains a `return`.

    The continuation goes into *both* arms here, since no single place is
    reachable from exactly the non-returning paths.  The copy must be a fresh
    one -- the analyses key on node identity, and sharing raises from
    `DefineUse` rather than returning a wrong answer.
    """

    @pytest.mark.parametrize('f', [both_arms_fall_through, one_armed_inner],
                             ids=lambda f: f.name)
    def test_one_return_remains(self, f):
        assert _returns(SingleExit.apply(f.ast)) == 1

    @pytest.mark.parametrize('f', [both_arms_fall_through, one_armed_inner],
                             ids=lambda f: f.name)
    @pytest.mark.parametrize('p', [1.0, -1.0])
    @pytest.mark.parametrize('q', [1.0, -1.0])
    def test_agrees_with_the_interpreter(self, f, p, q):
        _agrees(f, p, q)

    def test_the_copies_share_no_nodes(self):
        """Sharing passes a shallow structural check but breaks def-use."""
        out = SingleExit.apply(both_arms_fall_through.ast)
        seen: set[int] = set()

        class _V(DefaultVisitor):
            def _visit_statement(self, stmt, ctx):
                assert id(stmt) not in seen, 'a statement node appears twice'
                seen.add(id(stmt))
                return super()._visit_statement(stmt, ctx)

        _V()._visit_function(out, None)
