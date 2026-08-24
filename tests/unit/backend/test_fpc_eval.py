"""
Differential evaluation tests for the FPCore backend.

The other ``test_fpc_*`` modules compare *emitted text*, which cannot catch a
core that prints correctly but does not evaluate — the failure mode of four
bugs in list-reduce and ``for``-loop lowering:

- a ``for`` accumulator init that referenced the loop index, which FPCore
  binds only *after* the inits are evaluated (``EvaluatorUnboundError``);
- ``:precision integer`` scoped over a whole combine rather than just the
  index arithmetic, truncating every partial ``sum``;
- raw Python ``str`` passed where an :class:`fpc.Expr` operand was expected,
  which prints as the bare identifier but has no evaluator dispatch;
- FPy's loop *target* used as the FPCore ``for`` *dimension* variable, binding
  it to the index instead of the element;

and of three more in list comprehensions (see :class:`TestListComp`).

So these tests run each program twice — once through the FPy interpreter, once
through titanfp's interpreter on the compiled core — and require agreement.

The same harness covers the boolean reductions (``any`` / ``all``): booleans
are the values most easily lost across the titanfp boundary, so they want
executing rather than string-matching too.
"""

import math

import pytest
import titanfp.fpbench.fpcast as fpc
from titanfp.arithmetic.mpmf import MPMF, Interpreter

import fpy2 as fp
from fpy2 import FPCoreCompiler
from fpy2.ast.fpyast import ListTypeAnn, RealTypeAnn
from fpy2.backend.fpc import FPCoreCompileError


def _size(fn, *dims: int):
    """Pin each list-typed argument to a fixed length (FPCore tensors need a
    known size).  ``dims`` is positional over the list arguments only."""
    it = iter(dims)
    for arg in fn.ast.args:
        if isinstance(arg.type, ListTypeAnn):
            arg.type = ListTypeAnn(RealTypeAnn(None, None), next(it), None)
    return fn


def _compile(fn) -> fpc.FPCore:
    return FPCoreCompiler(unsafe_int_cast=True).compile(fn)


def _to_mpmf(x):
    """Python value -> titanfp argument.  Routed through :class:`fp.Float` so
    NaN and infinity survive; ``MPMF(float('nan'))`` raises."""
    if isinstance(x, list):
        return [_to_mpmf(v) for v in x]
    f = x if isinstance(x, fp.Float) else fp.Float.from_float(float(x))
    return MPMF(negative=f.s, exp=f.exp, c=f.c, isinf=f.isinf, isnan=f.isnan)


def _both(fn, *args):
    """``(fpy_result, fpcore_result)`` as Python floats."""
    core = _compile(fn)
    fpcore = Interpreter().interpret(core, [_to_mpmf(a) for a in args])
    return float(fn(*args)), float(fpcore)


def _agree(fn, *args):
    want, got = _both(fn, *args)
    assert want == got, f'FPy {want} != FPCore {got}\n  {_compile(fn).e}'
    return want


def _agree_list(fn, *args):
    """``_agree`` for list-returning programs, comparing element by element:
    order is the whole point for a comprehension, and a sum over the elements
    would pass for any permutation of them."""
    core = _compile(fn)
    got = [float(v) for v in Interpreter().interpret(core, [_to_mpmf(a) for a in args])]
    want = [float(v) for v in fn(*args)]
    assert want == got, f'FPy {want} != FPCore {got}\n  {core.e}'
    return want


def _agree_bool(fn, *args):
    """``_agree`` for bool-returning programs.  Separate from ``_both``, which
    coerces through ``float`` and would let a numeric ``1.0`` pass as ``True``;
    here both sides must be a genuine ``bool``."""
    core = _compile(fn)
    got = Interpreter().interpret(core, [_to_mpmf(a) for a in args])
    want = fn(*args)
    assert isinstance(want, bool), f'expected bool from FPy, got {want!r}'
    assert isinstance(got, bool), f'expected bool from FPCore, got {got!r}'
    assert want == got, f'FPy {want} != FPCore {got}\n  {core.e}'
    return want


def _operands(e) -> list:
    """Every operand reachable from *e* in the emitted object graph.

    Traversal goes through titanfp's uniform ``Expr.subexprs()`` API, which
    excludes binding *names* (legitimately ``str``) and yields exactly the
    positions that must hold an :class:`fpc.Expr`.  Walking bespoke attributes
    by hand instead does not work: ``If`` stores ``cond`` / ``then_body`` /
    ``else_body``, ``While`` and ``For`` their own shapes, so an
    attribute-based walker silently stops at the first one and reports a clean
    tree for a graph that is full of bad operands.
    """
    out: list = []
    stack = [e]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, fpc.Expr):
            continue  # a bad operand: recorded by the parent, not descended into
        try:
            groups = cur.subexprs()
        except (NotImplementedError, TypeError, ValueError):
            continue  # leaf, or a node with no subexpressions to offer
        for group in groups:
            for sub in group:
                out.append(sub)
                stack.append(sub)
    return out


class TestNoBareStringOperands:
    """A ``str`` where an operand belongs prints as the identifier but has no
    evaluator dispatch, so it survives text-comparison tests and fails only at
    evaluation.  Assert the invariant directly over the emitted graph."""

    def test_list_reduce_operands_are_exprs(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return sum(xs)

        bad = [o for o in _operands(_compile(_size(f, 3)).e) if isinstance(o, str)]
        assert not bad, f'bare str operands in emitted core: {bad}'

    def test_minimum_wrapper_operands_are_exprs(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return min(x, y)

        bad = [o for o in _operands(_compile(f).e) if isinstance(o, str)]
        assert not bad, f'bare str operands in emitted core: {bad}'


class TestListReduce:
    """``sum`` / ``min`` / ``max`` over a list."""

    def test_sum_of_fractions(self):
        """Fractional values catch a ``:precision integer`` annotation scoped
        over the addition instead of just the ``(+ i 1)`` index."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return sum(xs)

        assert _agree(_size(f, 3), [0.5, 0.25, 0.125]) == 0.875

    def test_sum_of_integers(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return sum(xs)

        assert _agree(_size(f, 4), [1.0, 2.0, 4.0, 8.0]) == 15.0

    def test_amin(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return min(xs)

        assert _agree(_size(f, 3), [3.0, -1.0, 2.0]) == -1.0

    def test_amax(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return max(xs)

        assert _agree(_size(f, 3), [3.0, -1.0, 2.0]) == 3.0

    def test_reduce_init_does_not_reference_loop_index(self):
        """The accumulator init must be ``(ref t 0)``: FPCore evaluates
        ``for`` while-binding inits before the dimension variables exist."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                return sum(xs)

        core = _compile(_size(f, 3))
        fors = [e for e in _operands(core.e) + [core.e] if isinstance(e, fpc.For)]
        assert len(fors) == 1
        _, init, _ = fors[0].while_bindings[0]
        idx_names = {str(name) for name, _ in fors[0].dim_bindings}
        used = {str(o.value) for o in _operands(init) if isinstance(o, fpc.Var)}
        assert not (used & idx_names), (
            f'accumulator init {init} references loop index(es) {used & idx_names}'
        )


class TestVariadicMinMax:
    """The scalar ``min(x, y)`` / ``max(x, y)`` wrappers."""

    def test_min_two_args(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return min(x, y)

        assert _agree(f, 3.0, -1.0) == -1.0

    def test_max_three_args(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real, z: fp.Real) -> fp.Real:
            with fp.FP64:
                return max(x, y, z)

        assert _agree(f, 3.0, -1.0, 7.0) == 7.0

    def test_min_propagates_nan(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return min(x, y)

        want, got = _both(f, float('nan'), 1.0)
        assert math.isnan(want) and math.isnan(got)

    def test_min_signed_zero_tie(self):
        """``min(+0, -0)`` is ``-0`` — the tie-break the wrapper exists for."""
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return min(x, y)

        want, got = _both(f, 0.0, -0.0)
        assert math.copysign(1.0, want) == -1.0
        assert math.copysign(1.0, got) == -1.0


class TestForLoopBindsElements:
    """FPy's ``for`` target ranges over elements; FPCore's dimension variable
    ranges over indices.  Conflating them makes ``for x in [1, 2, 4]`` walk
    ``0, 1, 2`` — a core that evaluates cleanly to the wrong answer."""

    def test_element_iteration(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = fp.round(0)
                for x in xs:
                    acc = acc + x
                return acc

        # 1+2+4 = 7; index iteration would give 0+1+2 = 3
        assert _agree(_size(f, 3), [1.0, 2.0, 4.0]) == 7.0

    def test_range_iteration_still_indexes(self):
        """``range`` lowers to ``(tensor ([j n]) j)``, so element == index."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = fp.round(0)
                for i in range(3):
                    acc = acc + xs[i] * xs[i]
                return acc

        assert _agree(_size(f, 3), [1.0, 2.0, 4.0]) == 21.0

    def test_enumerate_iteration(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = fp.round(0)
                for i, x in enumerate(xs):
                    acc = acc + x * i
                return acc

        # 1*0 + 2*1 + 4*2 = 10
        assert _agree(_size(f, 3), [1.0, 2.0, 4.0]) == 10.0

    def test_zip_iteration(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = fp.round(0)
                for a, b in zip(xs, ys):
                    acc = acc + a * b
                return acc

        # 1*1 + 2*10 + 4*100 = 421
        assert _agree(_size(f, 3, 3), [1.0, 2.0, 4.0], [1.0, 10.0, 100.0]) == 421.0

    def test_loop_with_no_mutated_variable(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                for x in xs:
                    _ = x + x
                return xs[0]

        assert _agree(_size(f, 3), [5.0, 2.0, 4.0]) == 5.0

    def test_nested_element_iteration(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = fp.round(0)
                for x in xs:
                    for y in ys:
                        acc = acc + x * y
                return acc

        # (1+2) * (10+100) = 330
        assert _agree(_size(f, 2, 2), [1.0, 2.0], [10.0, 100.0]) == 330.0


class TestBooleanReduce:
    """``any`` / ``all`` lower to a ``for``-fold seeded with the operator's
    identity (``FALSE`` / ``TRUE``), combining with ``or`` / ``and``.

    Each program builds its boolean list *internally*: a boolean tensor does
    not survive titanfp's argument conversion (Python ``bool`` becomes a
    numeric ``MPMF``, which has no ``__bool__``, so every element reads
    truthy).  Passing a ``list[bool]`` in would make these tests vacuous.
    """

    @staticmethod
    def _reduce(op, n: int):
        """``op([x < y for x in xs])`` with ``xs`` pinned to length *n*."""
        if op is any:
            @fp.fpy
            def f(xs: list[fp.Real], y: fp.Real) -> bool:
                with fp.FP64:
                    return any([x < y for x in xs])
        else:
            @fp.fpy
            def f(xs: list[fp.Real], y: fp.Real) -> bool:
                with fp.FP64:
                    return all([x < y for x in xs])
        return _size(f, n)

    @pytest.mark.parametrize('op,xs,want', [
        (any, [1.0, -2.0, 3.0], True),      # some true
        (any, [1.0, 2.0, 3.0], False),      # none true
        (any, [-1.0, -2.0, -3.0], True),    # all true
        (all, [-1.0, -2.0, -3.0], True),    # all true
        (all, [-1.0, 2.0, -3.0], False),    # one false
        (all, [1.0, 2.0, 3.0], False),      # none true
        (any, [-1.0], True),                # single element
        (any, [1.0], False),
        (all, [-1.0], True),
        (all, [1.0], False),
    ])
    def test_matches_interpreter(self, op, xs, want):
        assert _agree_bool(self._reduce(op, len(xs)), xs, 0.0) is want

    @pytest.mark.parametrize('op,identity', [(any, 'FALSE'), (all, 'TRUE')])
    def test_fold_seeds_with_the_identity(self, op, identity):
        """The accumulator init is the identity constant, not ``(ref t 0)`` —
        which is what makes the empty list correct by construction.  (Empty
        itself is unreachable through titanfp, which cannot build a
        zero-length tensor, so assert on the shape.)"""
        core = _compile(self._reduce(op, 3))
        fors = [e for e in _operands(core.e) + [core.e] if isinstance(e, fpc.For)]
        assert len(fors) == 1
        _, init, _ = fors[0].while_bindings[0]
        assert isinstance(init, fpc.Constant) and str(init) == identity

    @pytest.mark.parametrize('op', [any, all])
    def test_operands_are_exprs(self, op):
        """No bare ``str`` in operand position (see TestNoBareStringOperands)."""
        bad = [o for o in _operands(_compile(self._reduce(op, 3)).e)
               if isinstance(o, str)]
        assert not bad, f'bare str operands in emitted core: {bad}'


class TestListComp:
    """A comprehension lowers to a ``tensor``; several clauses lower to one
    ``tensor`` over the product of the lengths, with the iteration variable
    de-linearized into an index per clause.

    Three bugs lived on that multi-clause path, all of which print cleanly:
    reference bindings renamed with ``gensym.refresh`` while the element kept
    the original target names; every index built against a hardcoded ``k``
    rather than the tensor's own variable; and the middle index divided by
    ``size_ids[1:]`` where its own comment said ``size_ids[i+1:]``.
    """

    def test_one_clause(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [x * x for x in xs]

        assert _agree_list(_size(f, 3), [1.0, 2.0, 4.0]) == [1.0, 4.0, 16.0]

    def test_two_clauses(self):
        """The cartesian product, last clause varying fastest.  This is where
        the element referenced the un-renamed target and went unbound."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [(x * 10) + y for x in xs for y in ys]

        got = _agree_list(_size(f, 2, 3), [1.0, 2.0], [1.0, 2.0, 3.0])
        assert got == [11.0, 12.0, 13.0, 21.0, 22.0, 23.0]

    def test_three_clauses_with_distinct_lengths(self):
        """Distinct lengths and a digit per clause, so the index arithmetic is
        readable in the result: a wrong divisor repeats or skips combinations.
        Equal lengths would not do -- the middle bug is a permutation there."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [((x * 100) + (y * 10)) + z for x in xs for y in ys for z in zs]

        got = _agree_list(
            _size(f, 2, 3, 4),
            [1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0],
        )
        assert got == [float(f'{x}{y}{z}')
                       for x in (1, 2) for y in (1, 2, 3) for z in (1, 2, 3, 4)]

    def test_a_program_variable_named_k(self):
        """The index expressions must use the tensor's own variable.  With `k`
        already taken the tensor gets `k2`, and a hardcoded `k` then reads the
        *program's* `k` -- a core that evaluates fine and is simply wrong."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real], k: fp.Real) -> list[fp.Real]:
            with fp.FP64:
                return [((x * 10) + y) * k for x in xs for y in ys]

        got = _agree_list(_size(f, 2, 2), [1.0, 2.0], [1.0, 2.0], 2.0)
        assert got == [22.0, 24.0, 42.0, 44.0]

    def test_underscore_target(self):
        """An underscore binds nothing, so its clause contributes a length and
        no reference binding."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [y for _ in xs for y in ys]

        got = _agree_list(_size(f, 2, 3), [0.0, 0.0], [7.0, 8.0, 9.0])
        assert got == [7.0, 8.0, 9.0, 7.0, 8.0, 9.0]

    @pytest.mark.parametrize('clauses', [1, 2])
    def test_tuple_binding_target(self, clauses):
        """A destructured target indexes the *element* position last: a list of
        pairs is `t[i][0]`, not `t[0][i]`.  Transposed, it read down one column
        instead of across each pair -- values, not an error."""
        if clauses == 1:
            @fp.fpy
            def f(ps: list[fp.Real], qs: list[fp.Real], zs: list[fp.Real]) -> list[fp.Real]:
                with fp.FP64:
                    return [a + b for a, b in zip(ps, qs)]
        else:
            @fp.fpy
            def f(ps: list[fp.Real], qs: list[fp.Real], zs: list[fp.Real]) -> list[fp.Real]:
                with fp.FP64:
                    return [(a + b) * z for a, b in zip(ps, qs) for z in zs]

        got = _agree_list(_size(f, 2, 2, 2), [1.0, 2.0], [10.0, 20.0], [1.0, 3.0])
        assert got == ([11.0, 22.0] if clauses == 1 else [11.0, 33.0, 22.0, 66.0])

    def test_dependent_clauses_are_refused(self):
        """`[b for a in xss for b in a]` has no FPCore form here: the iterables
        are hoisted into one `let` outside the tensor, so `a` would escape its
        binder, and the extent is the product of the lengths, which a ragged
        flatten does not have."""
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            with fp.FP64:
                return [b for a in xss for b in a]

        f.ast.args[0].type = ListTypeAnn(
            ListTypeAnn(RealTypeAnn(None, None), 2, None), 2, None,
        )
        with pytest.raises(FPCoreCompileError, match='mentions an earlier target'):
            _compile(f)
