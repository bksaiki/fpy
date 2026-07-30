"""
Unit tests for the :class:`fpy2.transform.EnumerateElim` transform.

Structured like ``test_zip_elim.py``, and for the same reason: the rewrite
mints fresh ``_srcK`` / ``_iK`` names via ``Gensym``, whose suffix counts
depend on the existing in-scope names, so an ``is_equiv`` comparison against a
hand-written golden AST is too brittle.  Each positive test instead asserts

1. **Structural shape** of the rewritten AST (the iterable became a
   ``range(len(...))``, the body opens with the expected per-iteration
   assigns, no ``Enumerate`` / ``Zip`` survives) — "did the rewrite fire and
   is it well-formed?"
2. **Semantic equivalence** via the FPy interpreter on concrete sample
   inputs — this catches subtle errors that pass the shape checks.

For the negative tests (unchanged inputs), ``is_equiv`` against the original
AST is sufficient and stable.

One case is deliberately shape-only: an element slot whose arity disagrees
with the ``zip``'s is already ill-typed, so it cannot be evaluated.  The
transform is asserted to leave the ``zip`` in place rather than trade the
type error for an arity-mismatched destructure.
"""

import fpy2 as fp

from fpy2.ast.fpyast import (
    Assign, Enumerate, ForStmt, Fst, Len, ListComp, ListRef, NamedId, Range1,
    Snd, TupleBinding, TupleExpr, Var, Zip,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.transform import EnumerateElim


def _find_for(ast: fp.ast.FuncDef) -> ForStmt:
    """Return the first ``ForStmt`` reachable inside *ast.body*, descending
    into ``ContextStmt`` blocks.  Used to inspect the rewritten loop."""
    def walk(stmts):
        for s in stmts:
            if isinstance(s, ForStmt):
                return s
            body = getattr(s, 'body', None)
            if body is not None and hasattr(body, 'stmts'):
                hit = walk(body.stmts)
                if hit is not None:
                    return hit
        return None
    return walk(ast.body.stmts)


def _find_listcomp(ast: fp.ast.FuncDef) -> ListComp:
    """Return the first ``ListComp`` reachable in *ast*."""
    found: list[ListComp] = []

    class _C(DefaultVisitor):
        def _visit_list_comp(self, e, ctx):
            found.append(e)
            super()._visit_list_comp(e, ctx)

    _C()._visit_function(ast, None)
    return found[0] if found else None


def _contains(ast: fp.ast.FuncDef, types) -> bool:
    """True iff any sub-expression of *ast* is an instance of *types*."""
    hits: list = []

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, types):
                hits.append(e)
            return super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return bool(hits)


def _count_fors(ast: fp.ast.FuncDef) -> int:
    """How many ``ForStmt``s the function contains, at any depth."""
    n = 0

    class _C(DefaultVisitor):
        def _visit_for(self, stmt, ctx):
            nonlocal n
            n += 1
            super()._visit_for(stmt, ctx)

    _C()._visit_function(ast, None)
    return n


def _eval(ast: fp.ast.FuncDef, fn: fp.Function, *args):
    """Evaluate *ast* via the FPy interpreter using *fn*'s env."""
    return fn.with_ast(ast)(*args)


def _ref_assigns(loop: ForStmt) -> int:
    """Per-iteration ``x = src[i]`` assignments at the top of *loop*'s body."""
    return sum(
        1 for s in loop.body.stmts
        if isinstance(s, Assign) and isinstance(s.expr, ListRef)
    )


def _assert_same(new_ast, f, *args):
    """The rewrite preserves the interpreter's result on *args*."""
    before, after = f(*args), _eval(new_ast, f, *args)
    if hasattr(before, '__len__'):
        assert list(after) == list(before)
    else:
        assert after == before


# Sample inputs, reused across tests.
_XS = [1.0, 2.0, 3.0, 4.0]
_YS = [10.0, 20.0, 30.0, 40.0]
_ZS = [2.0, 2.0, 2.0, 2.0]
_PAIRS = [(1.0, 2.0), (3.0, 4.0)]
_ROWS = [[1.0, 2.0], [3.0, 4.0]]


class TestForLoopRewrite:
    """For-loops over ``enumerate`` become range-indexed loops."""

    def test_rewritten_to_indexed_loop(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, x in enumerate(xs):
                    acc = acc + x
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        # The index slot became the loop counter itself, keeping its name.
        assert isinstance(loop.target, NamedId)
        assert loop.target.base == 'i'
        # The iterable is range(len(_src0)).
        assert isinstance(loop.iterable, Range1)
        assert isinstance(loop.iterable.arg, Len)
        # One per-iteration assign, for the element slot only — the index
        # needs none, since range() already produces it.
        assert _ref_assigns(loop) == 1
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_index_read_in_body(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, x in enumerate(xs):
                    if i > 1:
                        acc = acc + x
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_discarded_index_gets_fresh_counter(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for _, x in enumerate(xs):
                    acc = acc + x
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        # An underscore index slot can't name the counter, so one is minted.
        assert isinstance(loop.target, NamedId)
        assert _ref_assigns(loop) == 1
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_discarded_element_emits_no_assign(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, _ in enumerate(xs):
                    acc = acc + 1
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        # Nothing reads the element, so no per-iteration assign is emitted;
        # the source is still bound before the loop for its length.
        assert _ref_assigns(loop) == 0
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_nested_element_slot_destructures(self):
        @fp.fpy
        def f(ps: list[tuple[fp.Real, fp.Real]]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, b) in enumerate(ps):
                    acc = acc + a * b
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        # The nested slot lowers to a destructuring assign; a statement
        # context needs no fst/snd.
        assert any(
            isinstance(s, Assign) and isinstance(s.target, TupleBinding)
            for s in loop.body.stmts
        )
        assert not _contains(new_ast, (Enumerate, Fst, Snd))
        _assert_same(new_ast, f, _PAIRS)

    def test_non_var_source_is_bound_once(self):
        """A slice source is evaluated once into the preamble temp, not
        re-evaluated per iteration."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, x in enumerate(xs[1:]):
                    acc = acc + x
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_body_rebinding_source_keeps_original_bound(self):
        """The loop bound comes from the preamble temp, so a body that
        rebinds the source name cannot change the iteration count."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, x in enumerate(xs):
                    acc = acc + x
                    xs = [1.0]
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        # The bound reads a name the body never touches.
        assert isinstance(loop.iterable, Range1)
        assert isinstance(loop.iterable.arg, Len)
        assert isinstance(loop.iterable.arg.arg, Var)
        assert loop.iterable.arg.arg.name.base == '_src'
        _assert_same(new_ast, f, _XS)

    def test_nested_loops_both_rewritten(self):
        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, row in enumerate(xss):
                    for j, x in enumerate(row):
                        acc = acc + x
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        assert not _contains(new_ast, Enumerate)
        assert _count_fors(new_ast) == 2
        _assert_same(new_ast, f, _ROWS)

    def test_non_enumerate_iterable_unchanged(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x in xs:
                    acc = acc + x
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        assert new_ast.is_equiv(f.ast)


class TestEnumerateOfZip:
    """``enumerate(zip(...))`` collapses *both* intermediate lists at once.

    This is why the transform can't defer to :class:`fpy2.transform.ZipElim`:
    after a plain enumerate rewrite the ``zip`` sits on the right-hand side of
    ``_src0 = zip(...)``, where a transform matching ``zip`` in *iterable*
    position can no longer reach it.
    """

    def test_two_arg_zip_fully_eliminated(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, b) in enumerate(zip(xs, ys)):
                    acc = acc + a * b
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        assert isinstance(loop.target, NamedId)
        assert isinstance(loop.iterable, Range1)
        # One assign per zip argument, indexing that argument directly.
        assert _ref_assigns(loop) == 2
        # Neither derived sequence survives.
        assert not _contains(new_ast, (Enumerate, Zip))
        _assert_same(new_ast, f, _XS, _YS)

    def test_three_arg_zip_fully_eliminated(self):
        @fp.fpy
        def f(
            xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]
        ) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, b, c) in enumerate(zip(xs, ys, zs)):
                    acc = acc + a * b * c
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        assert _ref_assigns(_find_for(new_ast)) == 3
        assert not _contains(new_ast, (Enumerate, Zip))
        _assert_same(new_ast, f, _XS, _YS, _ZS)

    def test_discarded_slot_inside_zip(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, _) in enumerate(zip(xs, ys)):
                    acc = acc + a
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        # Only the read slot gets an assign; both sources stay bound.
        assert _ref_assigns(_find_for(new_ast)) == 1
        assert not _contains(new_ast, (Enumerate, Zip))
        _assert_same(new_ast, f, _XS, _YS)

    def test_mismatched_arity_keeps_the_zip(self):
        """Shape-only: the program is ill-typed (a 2-slot binding over a
        3-argument ``zip``), so it can't be evaluated.  Keeping the ``zip``
        keeps its diagnostic instead of trading it for an arity-mismatched
        destructure."""

        @fp.fpy
        def f(
            xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]
        ) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, b) in enumerate(zip(xs, ys, zs)):
                    acc = acc + a * b
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        # The enumerate still goes; only the zip is left materialized.
        assert not _contains(new_ast, Enumerate)
        assert _contains(new_ast, Zip)


class TestWholeTupleSlot:
    """An element slot bound to the whole zipped tuple rebuilds it per
    iteration instead of materializing a list of them."""

    def test_for_loop_builds_tuple_per_iteration(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, p in enumerate(zip(xs, ys)):
                    acc = acc + fp.fst(p) * fp.snd(p)
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        # ``p = (_src0[i], _src1[i])`` — a tuple expression, not a ListRef.
        first = loop.body.stmts[0]
        assert isinstance(first, Assign)
        assert isinstance(first.expr, TupleExpr)
        assert len(first.expr.elts) == 2
        assert all(isinstance(e, ListRef) for e in first.expr.elts)
        assert not _contains(new_ast, (Enumerate, Zip))
        _assert_same(new_ast, f, _XS, _YS)

    def test_discarded_whole_slot_builds_nothing(self):
        """Previously this materialized the entire zip list purely to ask
        its length."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, _ in enumerate(zip(xs, ys)):
                    acc = acc + 1
                return acc

        new_ast = EnumerateElim.apply(f.ast)
        loop = _find_for(new_ast)
        assert _ref_assigns(loop) == 0
        assert not _contains(new_ast, (Enumerate, Zip, TupleExpr))
        _assert_same(new_ast, f, _XS, _YS)

    def test_list_comp_substitutes_a_tuple(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [fp.fst(p) * fp.snd(p) for i, p in enumerate(zip(xs, ys))]

        new_ast = EnumerateElim.apply(f.ast)
        comp = _find_listcomp(new_ast)
        assert isinstance(comp.iterables[0], Range1)
        # ``p`` was replaced by a tuple over one access path per source.  This
        # is not inlining the zip: it stays pure and O(1) per element.
        assert _contains(new_ast, TupleExpr)
        assert not _contains(new_ast, (Enumerate, Zip))
        _assert_same(new_ast, f, _XS, _YS)


class TestListCompRewrite:
    """List-comps over ``enumerate`` are rewritten when the sources are
    access paths."""

    def test_rewritten_to_indexed_comp(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [x * x for i, x in enumerate(xs)]

        new_ast = EnumerateElim.apply(f.ast)
        comp = _find_listcomp(new_ast)
        assert len(comp.targets) == 1
        # The index slot keeps its name and becomes the comp's counter.
        assert isinstance(comp.targets[0], NamedId)
        assert comp.targets[0].base == 'i'
        assert isinstance(comp.iterables[0], Range1)
        assert isinstance(comp.iterables[0].arg, Len)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_index_and_element_both_read(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [x + xs[i] for i, x in enumerate(xs)]

        new_ast = EnumerateElim.apply(f.ast)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_element_read_twice(self):
        """The substitution lands at every use of the bound name."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [x * x + x for i, x in enumerate(xs)]

        new_ast = EnumerateElim.apply(f.ast)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_discarded_index(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [x * x for _, x in enumerate(xs)]

        new_ast = EnumerateElim.apply(f.ast)
        comp = _find_listcomp(new_ast)
        assert isinstance(comp.targets[0], NamedId)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _XS)

    def test_zip_destructured_in_comp(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [a * b for i, (a, b) in enumerate(zip(xs, ys))]

        new_ast = EnumerateElim.apply(f.ast)
        comp = _find_listcomp(new_ast)
        assert isinstance(comp.iterables[0], Range1)
        assert not _contains(new_ast, (Enumerate, Zip))
        _assert_same(new_ast, f, _XS, _YS)

    def test_nested_slot_uses_fst_snd(self):
        @fp.fpy
        def f(ps: list[tuple[fp.Real, fp.Real]]) -> list[fp.Real]:
            with fp.FP64:
                return [a + b for i, (a, b) in enumerate(ps)]

        new_ast = EnumerateElim.apply(f.ast)
        # A comp has no statement context, so the nested slot's leaves are
        # reached by fst/snd chains over ``ps[i]``.
        assert _contains(new_ast, (Fst, Snd))
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _PAIRS)

    def test_non_access_path_source_unchanged(self):
        """A slice source would be re-evaluated once per iteration, turning
        O(n) into O(n^2), so the comp is left for the backend."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [x * x for i, x in enumerate(xs[1:])]

        new_ast = EnumerateElim.apply(f.ast)
        assert new_ast.is_equiv(f.ast)
        assert _contains(new_ast, Enumerate)

    def test_shadowed_target_in_nested_comp(self):
        """The inner comp re-binds ``x``; that reference must not be
        substituted by the outer rewrite."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [sum([x for x in ys]) + x for i, x in enumerate(xs)]

        new_ast = EnumerateElim.apply(f.ast)
        outer = _find_listcomp(new_ast)
        assert isinstance(outer.iterables[0], Range1)

        inner_found: list[ListComp] = []

        class _C(DefaultVisitor):
            def _visit_list_comp(self, e, ctx):
                if e is not outer:
                    inner_found.append(e)
                super()._visit_list_comp(e, ctx)

        _C()._visit_function(new_ast, None)
        assert len(inner_found) == 1
        inner = inner_found[0]
        # The inner elt is still a bare ``Var`` to ``x`` — not rewritten.
        assert isinstance(inner.elt, Var)
        assert inner.elt.name.base == 'x'
        assert isinstance(inner.iterables[0], Var)
        _assert_same(new_ast, f, _XS, _YS)

    def test_later_stage_iterable_references_rewritten_target(self):
        """A multi-stage comp whose second iterable reads the first stage's
        target: the substitution has to reach the iterable too, or the name
        is left dangling and ``SyntaxCheck`` rejects the output."""

        @fp.fpy
        def f(xss: list[list[fp.Real]]) -> list[fp.Real]:
            with fp.FP64:
                return [y for i, row in enumerate(xss) for y in row]

        new_ast = EnumerateElim.apply(f.ast)
        assert not _contains(new_ast, Enumerate)
        _assert_same(new_ast, f, _ROWS)


class TestProperties:
    """Cross-cutting properties of the transform."""

    def test_idempotent(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, (a, b) in enumerate(zip(xs, ys)):
                    acc = acc + a * b
                return acc

        once = EnumerateElim.apply(f.ast)
        twice = EnumerateElim.apply(once)
        assert once.is_equiv(twice)

    def test_syntax_check_passes(self):
        """``EnumerateElim.apply`` runs ``SyntaxCheck.check`` internally, so
        ill-formed output would make ``apply`` itself raise.  This exercises a
        representative mix of the shapes the transform emits."""

        @fp.fpy
        def f(
            xs: list[fp.Real], ys: list[fp.Real],
            ps: list[tuple[fp.Real, fp.Real]],
        ) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i, x in enumerate(xs):
                    acc = acc + x
                for j, (a, b) in enumerate(zip(xs, ys)):
                    acc = acc + a * b
                for k, p in enumerate(zip(xs, ys)):
                    acc = acc + fp.fst(p)
                for m, (c, d) in enumerate(ps):
                    acc = acc + c + d
                return acc + sum([z * z for n, z in enumerate(xs)])

        # Should not raise.
        out = EnumerateElim.apply(f.ast)
        assert not _contains(out, (Enumerate, Zip))

    def test_rejects_non_funcdef(self):
        import pytest

        with pytest.raises(TypeError):
            EnumerateElim.apply('not a funcdef')  # type: ignore[arg-type]
