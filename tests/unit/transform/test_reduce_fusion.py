"""
Unit tests for the :class:`fpy2.transform.ReduceFusion` transform.

The rewrite mints fresh ``_acc`` / ``_b`` names via ``Gensym``, so an
``is_equiv`` comparison against a hand-written golden AST is brittle.  As in
``test_zip_elim``, the tests assert:

1. **Structural shape** — the reduction is gone, replaced by a seeded
   accumulator and a ``for`` loop whose body binds the element before
   folding it.
2. **Semantic equivalence** — the fused function agrees with the original on
   concrete inputs, *including when the element expression raises*.  That
   last part is the whole reason the element is bound to a temp rather than
   inlined, so it gets its own test.

Negative tests (unchanged inputs) use ``is_equiv`` against the original.
"""

import pytest

import fpy2 as fp

from fpy2 import Function
from fpy2.ast.fpyast import (
    AllOf, AnyOf, Assign, BoolVal, ForStmt, ListComp, Or, Var,
)
from fpy2.transform import ReduceFusion


def _fuse(f) -> Function:
    return Function(ReduceFusion.apply(f.ast))


def _find(ast, cls):
    """First node of type *cls* reachable in *ast.body*, descending into
    compound statements."""
    def walk(stmts):
        for s in stmts:
            if isinstance(s, cls):
                return s
            body = getattr(s, 'body', None)
            if body is not None and hasattr(body, 'stmts'):
                hit = walk(body.stmts)
                if hit is not None:
                    return hit
        return None
    return walk(ast.body.stmts)


def _has_node(ast, cls) -> bool:
    """Whether any expression of type *cls* survives anywhere in *ast*."""
    from fpy2.ast import DefaultVisitor

    found = []

    class _V(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, cls):
                found.append(e)
            return super()._visit_expr(e, ctx)

    _V()._visit_function(ast, None)
    return bool(found)


def _agree(f, args_list):
    """The fused function matches the original on every input — same value,
    or the same exception type."""
    g = _fuse(f)

    def run(fn, args):
        try:
            return ('value', fn(*args))
        except Exception as e:  # noqa: BLE001 -- exception *type* is the check
            return ('raises', type(e).__name__)

    for args in args_list:
        assert run(f, args) == run(g, args), f'diverged on {args}'
    return g


class TestRewriteFires:
    """``any``/``all`` over a comprehension becomes a seeded loop."""

    def test_any_becomes_seeded_loop(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs])

        out = ReduceFusion.apply(f.ast)
        assert not _has_node(out, AnyOf), 'reduction should be gone'
        assert not _has_node(out, ListComp), 'intermediate list should be gone'
        seed = _find(out, Assign)
        assert isinstance(seed.expr, BoolVal) and seed.expr.val is False, (
            '`any` seeds the accumulator with False'
        )
        assert _find(out, ForStmt) is not None

    def test_all_seeds_true(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return all([x < 0 for x in xs])

        out = ReduceFusion.apply(f.ast)
        assert not _has_node(out, AllOf)
        seed = _find(out, Assign)
        assert isinstance(seed.expr, BoolVal) and seed.expr.val is True

    def test_loop_body_binds_element_before_folding(self):
        """The body must be ``_b = <elt>; _acc = _acc or _b`` — folding
        ``<elt>`` inline would let ``or`` short-circuit it away."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs])

        loop = _find(ReduceFusion.apply(f.ast), ForStmt)
        assert len(loop.body.stmts) == 2
        bind, fold = loop.body.stmts
        assert isinstance(bind, Assign)
        # the fold's operands are both plain variable references
        assert isinstance(fold.expr, Or)
        assert all(isinstance(a, Var) for a in fold.expr.args), (
            'element must be folded via a Var, not inlined'
        )
        # ...and one of them is the name the bind just defined
        assert str(bind.target) in {str(a.name) for a in fold.expr.args}

    def test_idempotent(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs])

        once = ReduceFusion.apply(f.ast)
        twice = ReduceFusion.apply(once)
        assert once.format() == twice.format()


class TestSemanticsPreserved:

    def test_any_matches_on_values(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs])

        g = _agree(f, [([],), ([1.0],), ([-1.0],), ([1.0, -2.0, 3.0],), ([1.0, 2.0],)])
        assert g([], ctx=fp.FP64) is False        # empty identity survives

    def test_all_matches_on_values(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return all([x < 0 for x in xs])

        g = _agree(f, [([],), ([1.0],), ([-1.0],), ([-1.0, -2.0],), ([-1.0, 2.0],)])
        assert g([], ctx=fp.FP64) is True

    def test_faulting_element_still_faults(self):
        """The motivating case for binding the element.

        ``xs[i]`` walks off the end.  Unfused, the comprehension evaluates
        every element and raises.  If the rewrite inlined the element into
        ``_acc or <elt>``, ``or`` would short-circuit once ``_acc`` is True
        and the program would return instead of raising.
        """
        @fp.fpy
        def f(xs: list[fp.Real], n: fp.Real) -> bool:
            with fp.FP64:
                return any([xs[i] < 0 for i in range(n)])

        # xs[0] < 0 decides `any` immediately, but n=4 overruns a length-3 list
        with pytest.raises(IndexError):
            f([-1.0, 2.0, 3.0], 4, ctx=fp.FP64)
        with pytest.raises(IndexError):
            _fuse(f)([-1.0, 2.0, 3.0], 4, ctx=fp.FP64)

        _agree(f, [([-1.0, 2.0, 3.0], 4), ([-1.0, 2.0, 3.0], 3), ([1.0, 2.0, 3.0], 4)])

    def test_nested_in_larger_expression(self):
        @fp.fpy
        def f(xs: list[fp.Real], y: fp.Real) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs]) and (y > 0)

        _agree(f, [([1.0, -2.0], 1.0), ([1.0, 2.0], 1.0), ([-1.0], -1.0), ([], 1.0)])

    def test_two_reductions_in_one_statement(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([x < 0 for x in xs]) and all([x < 10 for x in xs])

        out = ReduceFusion.apply(f.ast)
        assert not _has_node(out, AnyOf) and not _has_node(out, AllOf)
        _agree(f, [([1.0, -2.0],), ([1.0, 20.0],), ([-1.0, 2.0],), ([],)])

    def test_inside_loop_and_if_condition(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                n = fp.round(0)
                for x in xs:
                    if any([y < x for y in xs]):
                        n = n + 1
                return n

        _agree(f, [([1.0, 2.0, 3.0],), ([],), ([5.0],)])


class TestRewriteSuppressed:
    """Positions with no statement slot, and operands that aren't comps."""

    def test_non_comprehension_operand_untouched(self):
        @fp.fpy
        def f(bs: list[bool]) -> bool:
            return any(bs)

        out = ReduceFusion.apply(f.ast)
        assert out.is_equiv(f.ast)
        assert _has_node(out, AnyOf)

    def test_if_expr_branch_not_fused(self):
        """Hoisting a loop out of a conditional branch would run it
        unconditionally — observable when the element can fault."""
        @fp.fpy
        def f(xs: list[fp.Real], c: bool) -> bool:
            with fp.FP64:
                return any([xs[0] < 0 for _ in xs]) if c else False

        out = ReduceFusion.apply(f.ast)
        assert _has_node(out, AnyOf), 'branch reduction must be left in place'
        # `xs[0]` on an empty list would raise if the branch were hoisted
        _agree(f, [([], False), ([1.0], True), ([1.0], False)])

    def test_nested_inside_a_comprehension_not_fused(self):
        @fp.fpy
        def f(xss: list[fp.Real]) -> bool:
            with fp.FP64:
                inner = [any([y < x for y in xss]) for x in xss]
                return all(inner)

        out = ReduceFusion.apply(f.ast)
        # the inner reduction sits in a comp element -> no statement slot
        assert _has_node(out, AnyOf)
        _agree(f, [([1.0, 2.0],), ([],), ([3.0],)])

    def test_multi_stage_comprehension_left_alone(self):
        """``[e for a in xs for b in ys]`` has two targets; fusing it needs
        nested loops, so the pass declines."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([a < b for a in xs for b in ys])

        out = ReduceFusion.apply(f.ast)
        assert out.is_equiv(f.ast)
        _agree(f, [([1.0, 2.0], [3.0]), ([5.0], [3.0]), ([], [1.0]), ([1.0], [])])


class TestTupleBindingTarget:

    def test_zip_comprehension_fuses(self):
        """A ``zip`` comp is single-stage — one target, one iterable — so it
        fuses, with the tuple binding carried onto the ``for``."""
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> bool:
            with fp.FP64:
                return any([a < b for a, b in zip(xs, ys)])

        out = ReduceFusion.apply(f.ast)
        assert not _has_node(out, AnyOf)
        _agree(f, [([1.0, 2.0], [3.0, 1.0]), ([1.0], [0.0]), ([], [])])
