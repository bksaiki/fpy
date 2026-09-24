"""
Unit tests for :class:`fpy2.transform.SingleExit`.

The result name comes from ``Gensym``, so a golden AST is brittle.  These
tests assert:

1. **Structural shape** -- exactly one `return` survives, a function that
   already has one is returned unchanged, and the copies share no nodes.
2. **The continuation is moved where it can be** -- the tail after an `if`
   appears once when only one arm falls through.
3. **Semantic equivalence** via the interpreter, on inputs taking each path.
4. **Refusals** -- a shape the rewrite leaves with several exits.
"""

import pytest

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import Reachability, TypeInfer
from fpy2.ast.fpyast import ForStmt, WhileStmt
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import SingleExit, TransformDeclined


def _returns(ast) -> int:
    return len(Reachability.analyze(ast).ret_stmts)


class _Count(DefaultVisitor):
    """How many statements of *kind* a function holds."""

    kind: type
    n: int

    def __init__(self, kind: type):
        self.kind = kind
        self.n = 0

    def _visit_statement(self, stmt, ctx):
        if isinstance(stmt, self.kind):
            self.n += 1
        return super()._visit_statement(stmt, ctx)


def _count(ast, kind: type) -> int:
    v = _Count(kind)
    v._visit_function(ast, None)
    return v.n


def _whiles(ast) -> int:
    return _count(ast, WhileStmt)


def _loops(ast) -> int:
    return _count(ast, ForStmt)


class _FlagNames(DefaultVisitor):
    """The `done` flags a rewrite introduced."""

    names: set[str]

    def __init__(self):
        self.names = set()

    def _visit_var(self, e, ctx):
        if str(e.name).startswith('done'):
            self.names.add(str(e.name))


def _flag_names(ast) -> int:
    """How many distinct `done` flags the rewrite introduced."""
    v = _FlagNames()
    v._visit_function(ast, None)
    return len(v.names)


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


@fp.fpy(ctx=fp.REAL)
def guard_then_divide(x):
    """The condition would raise once `n` reaches 0, and the `return` is what
    stops it getting there.  Folding the flag in *after* `n != 0` would
    re-evaluate the division; short-circuiting means it does not."""
    n = x
    while n != 0 and 1 / n > 0:
        if n < 2:
            return n
        n = n - 1
    return 0


@fp.fpy
def returns_in_a_nested_loop(x, y):
    """A `return` in an inner loop must stop the *outer* one too.

    With a flag per loop it stops only the innermost: the outer body then
    does nothing, never changes the outer condition, and spins forever.
    """
    a = 0.0
    while x > 0:
        b = y
        while b > 0:
            if b > 5.0:
                return b
            b = b - 1.0
        a = a + 1.0
        x = x - 1.0
    return a


@fp.fpy
def returns_in_a_nested_for(xss: list[list[fp.Real]]):
    n = 0.0
    for xs in xss:
        for x in xs:
            if x < 0:
                return x
            n = n + x
    return n


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
    (returns_in_a_nested_loop, [
        (2.0, 8.0), (2.0, 3.0), (0.0, 9.0), (3.0, 0.0)]),
    (returns_in_a_nested_for, [
        ([[1.0, 2.0], [3.0]],), ([[1.0], [-1.0, 2.0]],),
        ([[-1.0]],), ([],), ([[]],)]),
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


class TestAWhileLoopFoldsTheFlagIntoItsCondition:
    """A `while` cannot be handled the way a `for` is.

    Guarding the body alone would spin forever: a dead body never makes the
    condition false, and unlike a `for` there is no iterable bounding it.  So
    the flag goes into the condition, which *stops* the loop rather than
    idling it.
    """

    def test_the_loop_survives(self):
        out = SingleExit.apply(returns_in_a_while.ast)
        assert _whiles(out) == _whiles(returns_in_a_while.ast) == 1

    def test_a_nested_return_stops_the_outer_loop(self):
        """One flag for the whole function, not one per loop.

        Caught by a differential rather than by construction: with a flag per
        loop this hangs instead of failing.
        """
        out = SingleExit.apply(returns_in_a_nested_loop.ast)
        assert _flag_names(out) == 1

    def test_the_flag_is_in_the_condition(self):
        """Checked structurally as well as behaviourally: a version that
        guards the body instead *hangs*, and a hanging test says much less
        than a failing one."""
        out = Function(SingleExit.apply(returns_in_a_while.ast),
                       runtime=None).format()
        cond = next(ln for ln in out.splitlines()
                    if ln.strip().startswith('while '))
        assert 'not ' in cond, f'the flag must stop the loop, not idle it: {cond}'

    def test_it_terminates_and_agrees(self):
        """`x > 100` returns early; without the condition fold this hangs."""
        for x in (150.0, 3.0, -1.0, 101.0):
            _agrees(returns_in_a_while, x)

    def test_the_condition_is_not_evaluated_after_returning(self):
        """`and` short-circuits, so a condition that would raise once the
        return has fired is never reached -- as in the original."""
        _agrees(guard_then_divide, 0.0)
        _agrees(guard_then_divide, 4.0)


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


@fp.fpy
def any_negative(xs: list[fp.Real]) -> bool:
    for x in xs:
        if x < 0:
            return True
    return False


@fp.fpy
def first_negative_row(xss: list[list[fp.Real]]) -> list[fp.Real]:
    for xs in xss:
        if xs[0] < 0:
            return xs
    return xss[0]


@fp.fpy
def first_negative_pair(xs: list[fp.Real]) -> tuple[fp.Real, bool]:
    for x in xs:
        if x < 0:
            return (x, True)
    return (fp.round(0), False)


class TestThePlaceholderHasTheReturnType:
    """A `return` in a loop gives the result a placeholder before the loop,
    which has to have the function's return type or the rewrite does not type
    check: `0` against a `bool` fails to unify."""

    @pytest.mark.parametrize('f, cases', [
        (any_negative, [[1.0, -2.0], [1.0, 2.0]]),
        (first_negative_row, [[[1.0], [-2.0]], [[1.0], [2.0]]]),
        (first_negative_pair, [[1.0, -2.0], [1.0, 2.0]]),
    ])
    def test_it_type_checks_and_agrees(self, f, cases):
        out = SingleExit.apply(f.ast)
        TypeInfer.check(out)
        g = Function(out, runtime=f.runtime)
        for xs in cases:
            assert repr(g(xs)) == repr(f(xs))
