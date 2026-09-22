"""
Unit tests for :class:`fpy2.transform.SingleExit`.

The result name comes from ``Gensym``, so a golden AST is brittle.  These
tests assert:

1. **Structural shape** -- exactly one `return` survives, a function that
   already has one is returned unchanged, and the copies share no nodes.
2. **The continuation is moved where it can be** -- the tail after an `if`
   appears once when only one arm falls through.
3. **Semantic equivalence** via the interpreter, on inputs taking each path.
4. **Refusals** -- a `return` inside a `while`, and a shape the rewrite
   leaves with several exits.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import Reachability
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import SingleExit, TransformDeclined


def _returns(ast) -> int:
    return len(Reachability.analyze(ast).ret_stmts)


class _LoopCount(DefaultVisitor):
    def __init__(self):
        self.n = 0

    def _visit_for(self, stmt, ctx):
        self.n += 1
        super()._visit_for(stmt, ctx)


def _loops(ast) -> int:
    v = _LoopCount()
    v._visit_function(ast, None)
    return v.n


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
def stores_after_returning(xs: list[fp.Real], out: list[fp.Real]):
    """A store *after* the returning `if`, which must stop at the return.

    This is what distinguishes guarding the whole body from guarding only the
    `return`: with `r` set once, a later iteration cannot overwrite it either
    way, so only a side effect makes the difference visible.
    """
    for i in range(len(xs)):
        if xs[i] < 0:
            return out
        out[i] = 1.0
    return out


@fp.fpy
def returns_in_a_while(x):
    while x > 0:
        if x > 100.0:
            return x
        x = x - 1.0
    return x


_CASES = [
    (early_return, [(1.0,), (-1.0,), (0.0,)]),
    (both_arms_return, [(1.0,), (-1.0,)]),
    (nested, [(a, b) for a in (1.0, -1.0) for b in (1.0, -1.0)]),
    (returns_in_a_context, [(1.0,), (1000.0,)]),
    (returns_in_a_for, [([1.0, 2.0],), ([1.0, -1.0, 2.0],), ([-1.0],), ([],)]),
    (stores_after_returning, [
        ([1.0, 2.0], [0.0, 0.0]),
        ([1.0, -1.0], [0.0, 0.0]),
        ([-1.0, 1.0], [0.0, 0.0]),
    ]),
]
_REWRITTEN = [f for f, _ in _CASES]


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
    @pytest.mark.parametrize('f,cases', _CASES, ids=lambda v: getattr(v, 'name', None))
    def test_agrees_with_the_interpreter(self, f, cases):
        for args in cases:
            _agrees(f, *args)


class TestTheContinuationIsMovedNotCopied:
    def test_the_tail_appears_once(self):
        """`y = x * 2` follows the `if`; it must end up in the `else` exactly
        once, not duplicated into both arms."""
        out = Function(SingleExit.apply(early_return.ast), runtime=None).format()
        assert out.count('x * 2') == 1


class TestAForLoopIsIfConverted:
    """A `return` in a `for` becomes a flag, not an unrolled loop.

    The continuation cannot move the way an `if`'s can -- the statements
    after the loop are reachable from the loop's own exit too -- so the
    return is recorded and the tail is tested against it.
    """

    def test_the_loop_survives(self):
        """If-converted, not unrolled: one loop in, one loop out."""
        out = SingleExit.apply(returns_in_a_for.ast)
        assert _loops(out) == _loops(returns_in_a_for.ast) == 1

    def test_the_whole_body_is_guarded(self):
        """Not just the `return`: a side effect after it must stop too.

        `out[i] = 1.0` follows the returning `if`, so guarding only the
        return would keep storing for every later element.
        """
        f = Function(SingleExit.apply(stores_after_returning.ast), runtime=None)
        assert repr(f([1.0, -1.0, 1.0], [0.0, 0.0, 0.0])) == repr(
            stores_after_returning([1.0, -1.0, 1.0], [0.0, 0.0, 0.0]))


class TestAWhileLoopIsStillRefused:
    """Predicating a `while` can change whether it terminates: the early
    return may be its only exit, where a `for` is bounded by its iterable."""

    def test_declines(self):
        with pytest.raises(TransformDeclined, match='inside a `while`'):
            SingleExit.apply(returns_in_a_while.ast)


@fp.fpy
def conditional_return_in_a_context(x):
    with fp.FP32:
        if x > 0:
            return fp.round(x)
        y = x
    return y


class TestThePostcondition:
    """A `with` that only sometimes returns is not moved -- the continuation
    would change rounding context.  `apply` counts what it left rather than
    handing back a function with several exits."""

    def test_declines(self):
        with pytest.raises(TransformDeclined, match='returns remain'):
            SingleExit.apply(conditional_return_in_a_context.ast)


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
