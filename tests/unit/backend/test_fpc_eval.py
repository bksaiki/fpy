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
  it to the index instead of the element.

So these tests run each program twice — once through the FPy interpreter, once
through titanfp's interpreter on the compiled core — and require agreement.

The same differential harness now also covers the boolean reductions
(``any`` / ``all``), which are new rather than fixed — booleans are the values
most easily lost across the titanfp boundary, so they want executing rather
than string-matching too.
"""

import math

import titanfp.fpbench.fpcast as fpc
from titanfp.arithmetic.mpmf import MPMF, Interpreter

import fpy2 as fp
from fpy2 import FPCoreCompiler
from fpy2.ast.fpyast import ListTypeAnn, RealTypeAnn


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


def _agree_bool(fn, *args):
    """``_agree`` for bool-returning programs.

    Kept separate because ``_both`` coerces through ``float``, which would let
    a numeric ``1.0`` masquerade as ``True`` — exactly the confusion that makes
    boolean values delicate on the titanfp boundary (see ``_to_mpmf``).  Both
    sides are required to be genuine ``bool``.
    """
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

    The boolean list is built *inside* each program rather than passed in: a
    boolean tensor does not survive titanfp's argument conversion, which turns
    a Python ``bool`` into a numeric ``MPMF``, and ``MPMF`` has no
    ``__bool__`` — so every element would read as truthy and the fold would
    return ``True`` regardless.  That is a limitation of the interop, not of
    the lowering.
    """

    @staticmethod
    def _any():
        @fp.fpy
        def f(xs: list[fp.Real], y: fp.Real) -> bool:
            with fp.FP64:
                return any([x < y for x in xs])
        return f

    @staticmethod
    def _all():
        @fp.fpy
        def f(xs: list[fp.Real], y: fp.Real) -> bool:
            with fp.FP64:
                return all([x < y for x in xs])
        return f

    def test_any_some_true(self):
        assert _agree_bool(_size(self._any(), 3), [1.0, -2.0, 3.0], 0.0) is True

    def test_any_none_true(self):
        assert _agree_bool(_size(self._any(), 3), [1.0, 2.0, 3.0], 0.0) is False

    def test_any_all_true(self):
        assert _agree_bool(_size(self._any(), 3), [-1.0, -2.0, -3.0], 0.0) is True

    def test_all_all_true(self):
        assert _agree_bool(_size(self._all(), 3), [-1.0, -2.0, -3.0], 0.0) is True

    def test_all_one_false(self):
        assert _agree_bool(_size(self._all(), 3), [-1.0, 2.0, -3.0], 0.0) is False

    def test_all_none_true(self):
        assert _agree_bool(_size(self._all(), 3), [1.0, 2.0, 3.0], 0.0) is False

    def test_any_single_element(self):
        assert _agree_bool(_size(self._any(), 1), [-1.0], 0.0) is True
        assert _agree_bool(_size(self._any(), 1), [1.0], 0.0) is False

    def test_all_single_element(self):
        assert _agree_bool(_size(self._all(), 1), [-1.0], 0.0) is True
        assert _agree_bool(_size(self._all(), 1), [1.0], 0.0) is False

    def test_fold_seeds_with_the_identity(self):
        """The accumulator init is the identity constant, not ``(ref t 0)``.

        This is what makes the empty list correct by construction, and it is
        why the boolean fold needs no ``n - 1`` bound or ``(+ i 1)`` index
        arithmetic.  (The empty case itself is unreachable through titanfp,
        which cannot build a zero-length tensor, so assert on the shape.)
        """
        core = _compile(_size(self._all(), 3))
        fors = [e for e in _operands(core.e) + [core.e] if isinstance(e, fpc.For)]
        assert len(fors) == 1
        _, init, _ = fors[0].while_bindings[0]
        assert isinstance(init, fpc.Constant) and str(init) == 'TRUE'

        core = _compile(_size(self._any(), 3))
        fors = [e for e in _operands(core.e) + [core.e] if isinstance(e, fpc.For)]
        _, init, _ = fors[0].while_bindings[0]
        assert isinstance(init, fpc.Constant) and str(init) == 'FALSE'

    def test_operands_are_exprs(self):
        """No bare ``str`` in operand position (see TestNoBareStringOperands)."""
        for fn in (_size(self._any(), 3), _size(self._all(), 3)):
            bad = [o for o in _operands(_compile(fn).e) if isinstance(o, str)]
            assert not bad, f'bare str operands in emitted core: {bad}'
