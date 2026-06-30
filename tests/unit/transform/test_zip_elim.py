"""
Unit tests for the :class:`fpy2.transform.ZipElim` transform.

The rewrite mints fresh ``_srcK`` / ``_iK`` names via ``Gensym``,
whose suffix counts depend on the existing in-scope names — so an
``is_equiv`` comparison against a hand-written golden AST is too
brittle.  Instead, the tests assert two things:

1. **Structural shape** of the rewritten AST (the iterable is a
   ``range(len(...))``, the body opens with ``ListRef`` assigns,
   etc.).  This catches "did the rewrite fire and is it
   well-formed?"
2. **Semantic equivalence** via the FPy interpreter on concrete
   sample inputs.  This catches subtle errors in the rewrite that
   pass shape checks.

For the negative tests (unchanged inputs), ``is_equiv`` against
the original AST is sufficient and stable.
"""

import fpy2 as fp

from fpy2.ast.fpyast import (
    Assign, ForStmt, Fst, Len, ListComp, ListRef, NamedId, Range1, Snd,
    TupleBinding, Var, Zip,
)
from fpy2.transform import ZipElim


def _find_for(ast: fp.ast.FuncDef) -> ForStmt:
    """Return the first ``ForStmt`` reachable inside *ast.body*, descending
    into ``ContextStmt`` blocks.  Used to inspect the rewritten loop."""
    def walk(stmts):
        for s in stmts:
            if isinstance(s, ForStmt):
                return s
            # Descend into ContextStmt / other compound stmts via .body
            body = getattr(s, 'body', None)
            if body is not None and hasattr(body, 'stmts'):
                hit = walk(body.stmts)
                if hit is not None:
                    return hit
        return None
    return walk(ast.body.stmts)


def _find_listcomp(ast: fp.ast.FuncDef) -> ListComp:
    """Return the first ``ListComp`` reachable in *ast*."""
    from fpy2.ast.visitor import DefaultVisitor

    found: list[ListComp] = []

    class _C(DefaultVisitor):
        def _visit_list_comp(self, e, ctx):
            found.append(e)
            super()._visit_list_comp(e, ctx)

    _C()._visit_function(ast, None)
    return found[0] if found else None


def _eval(ast: fp.ast.FuncDef, fn: fp.Function, *args):
    """Evaluate *ast* via the FPy interpreter using *fn*'s env."""
    return fn.with_ast(ast)(*args)


class TestForLoopRewrite:
    """For-loops over zip get rewritten to range-indexed loops."""

    def test_two_arg_zip_rewritten(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for a, b in zip(xs, ys):
                    acc = acc + a * b
                return acc

        new_ast = ZipElim.apply(f.ast)
        loop = _find_for(new_ast)
        # The rewritten for-stmt's target is a NamedId (the fresh _i).
        assert isinstance(loop.target, NamedId)
        # The iterable is range(len(_src0)).
        assert isinstance(loop.iterable, Range1)
        assert isinstance(loop.iterable.arg, Len)
        # The first two body stmts are the per-iter ListRef assigns.
        body = loop.body.stmts
        assert isinstance(body[0], Assign)
        assert isinstance(body[0].expr, ListRef)
        assert isinstance(body[1], Assign)
        assert isinstance(body[1].expr, ListRef)
        # Semantic equivalence on sample inputs.
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [10.0, 20.0, 30.0, 40.0]
        assert _eval(new_ast, f, xs, ys) == f(xs, ys)

    def test_three_arg_zip_rewritten(self):
        @fp.fpy
        def f(
            xs: list[fp.Real], ys: list[fp.Real], zs: list[fp.Real]
        ) -> fp.Real:
            with fp.FP64:
                acc = 0
                for a, b, c in zip(xs, ys, zs):
                    acc = acc + a * b * c
                return acc

        new_ast = ZipElim.apply(f.ast)
        loop = _find_for(new_ast)
        # Three per-iter ListRef assigns, then the body.
        body = loop.body.stmts
        ref_count = sum(
            1 for s in body
            if isinstance(s, Assign) and isinstance(s.expr, ListRef)
        )
        assert ref_count == 3
        # Semantic equivalence.
        xs = [1.0, 2.0, 3.0]
        ys = [4.0, 5.0, 6.0]
        zs = [7.0, 8.0, 9.0]
        assert _eval(new_ast, f, xs, ys, zs) == f(xs, ys, zs)

    def test_underscore_target(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for _, b in zip(xs, ys):
                    acc = acc + b
                return acc

        new_ast = ZipElim.apply(f.ast)
        loop = _find_for(new_ast)
        body = loop.body.stmts
        # The underscore slot emits no per-iter assign, so only one
        # ListRef-assign appears (for ``b``).
        ref_count = sum(
            1 for s in body
            if isinstance(s, Assign) and isinstance(s.expr, ListRef)
        )
        assert ref_count == 1
        xs = [99.0, 99.0, 99.0]
        ys = [1.0, 2.0, 3.0]
        assert _eval(new_ast, f, xs, ys) == f(xs, ys)

    def test_non_zip_iterable_unchanged(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x in xs:
                    acc = acc + x
                return acc

        new_ast = ZipElim.apply(f.ast)
        # Untouched: AST equivalent to the original.
        assert new_ast.is_equiv(f.ast)


class TestListCompRewrite:
    """List-comps over zip are rewritten when all zip args are Vars."""

    def test_var_args_rewritten(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [a * b for a, b in zip(xs, ys)]

        new_ast = ZipElim.apply(f.ast)
        comp = _find_listcomp(new_ast)
        # Single iteration stage with a NamedId target.
        assert len(comp.targets) == 1
        assert isinstance(comp.targets[0], NamedId)
        # Iterable: range(len(...)).
        assert isinstance(comp.iterables[0], Range1)
        assert isinstance(comp.iterables[0].arg, Len)
        # Semantic equivalence.
        xs = [1.0, 2.0, 3.0]
        ys = [10.0, 20.0, 30.0]
        before = f(xs, ys)
        after = _eval(new_ast, f, xs, ys)
        assert list(after) == list(before)

    def test_non_var_args_unchanged(self):
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [a + b for a, b in zip(xs[1:], xs)]

        new_ast = ZipElim.apply(f.ast)
        # Non-Var zip arg (a slice) — comp is left alone.
        assert new_ast.is_equiv(f.ast)
        # And the comp still contains a Zip.
        comp = _find_listcomp(new_ast)
        assert isinstance(comp.iterables[0], Zip)

    def test_shadowed_target_in_nested_comp(self):
        """Inner comp re-binds ``a``; the inner reference should not be
        substituted by the outer rewrite."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [sum([a for a in ys]) + b for a, b in zip(xs, ys)]

        new_ast = ZipElim.apply(f.ast)
        outer = _find_listcomp(new_ast)
        # Outer iterable is a range now.
        assert isinstance(outer.iterables[0], Range1)
        # The outer ``b`` slot was substituted but inner ``a`` was not:
        # we can verify this by checking the inner ListComp's elt is
        # still a bare ``Var`` to ``a``, not a ListRef.
        from fpy2.ast.fpyast import ListComp
        # Find the inner list-comp.
        inner_found: list[ListComp] = []
        from fpy2.ast.visitor import DefaultVisitor

        class _C(DefaultVisitor):
            def _visit_list_comp(self, e, ctx):
                if e is not outer:
                    inner_found.append(e)
                super()._visit_list_comp(e, ctx)

        _C()._visit_function(new_ast, None)
        assert len(inner_found) == 1
        inner = inner_found[0]
        # The inner elt is still ``Var(a)`` — not rewritten.
        assert isinstance(inner.elt, Var)
        assert inner.elt.name.base == 'a'
        # And the inner iterable is still the original ``ys`` Var.
        assert isinstance(inner.iterables[0], Var)
        # Semantic equivalence.
        xs = [1.0, 2.0, 3.0]
        ys = [10.0, 20.0, 30.0]
        before = list(f(xs, ys))
        after = list(_eval(new_ast, f, xs, ys))
        assert before == after


class TestProperties:
    """Cross-cutting properties of the transform."""

    def test_idempotent(self):
        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for a, b in zip(xs, ys):
                    acc = acc + a * b
                return acc

        once = ZipElim.apply(f.ast)
        twice = ZipElim.apply(once)
        assert once.is_equiv(twice)

    def test_syntax_check_passes(self):
        """``ZipElim.apply`` runs ``SyntaxCheck.check`` internally; if
        the rewrite produced ill-formed output, ``apply`` itself would
        raise.  This test just exercises a representative input."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for a, b in zip(xs, ys):
                    acc = acc + a * b
                return acc

        # Should not raise.
        ZipElim.apply(f.ast)


def _contains(ast: fp.ast.FuncDef, types) -> bool:
    """True iff any sub-expression of *ast* is an instance of *types*."""
    from fpy2.ast.visitor import DefaultVisitor

    hits: list = []

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, types):
                hits.append(e)
            return super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return bool(hits)


class TestNestedTupleBinding:
    """Zip targets with a nested ``TupleBinding`` slot — newly supported
    via the ``fst``/``snd`` accessors (comp path) and a destructuring
    assignment (for-loop path)."""

    def test_for_loop_nested_binding(self):
        @fp.fpy
        def f(pairs: list[tuple[fp.Real, fp.Real]], xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for (a, b), c in zip(pairs, xs):
                    acc = acc + a + b + c
                return acc

        new_ast = ZipElim.apply(f.ast)
        loop = _find_for(new_ast)
        assert isinstance(loop.iterable, Range1)
        # The nested slot lowers to a destructuring assign (TupleBinding
        # target); the plain ``c`` slot to a ListRef assign.
        body = loop.body.stmts
        assert any(
            isinstance(s, Assign) and isinstance(s.target, TupleBinding)
            for s in body
        )
        assert not _contains(new_ast, Zip)
        pairs = [(1.0, 2.0), (3.0, 4.0)]
        xs = [10.0, 20.0]
        assert _eval(new_ast, f, pairs, xs) == f(pairs, xs)

    def test_list_comp_nested_binding(self):
        @fp.fpy
        def f(pairs: list[tuple[fp.Real, fp.Real]], xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [a + b + c for (a, b), c in zip(pairs, xs)]

        new_ast = ZipElim.apply(f.ast)
        comp = _find_listcomp(new_ast)
        # Rewritten to a range-indexed comp; the nested slot's elements are
        # reached by fst/snd accessors in the elt.
        assert isinstance(comp.iterables[0], Range1)
        assert isinstance(comp.targets[0], NamedId)
        assert not _contains(new_ast, Zip)
        assert _contains(new_ast, (Fst, Snd))
        pairs = [(1.0, 2.0), (3.0, 4.0)]
        xs = [10.0, 20.0]
        assert list(_eval(new_ast, f, pairs, xs)) == list(f(pairs, xs))

    def test_list_comp_deeply_nested_binding(self):
        @fp.fpy
        def f(
            ps: list[tuple[tuple[fp.Real, fp.Real], fp.Real]], xs: list[fp.Real]
        ) -> list[fp.Real]:
            with fp.FP64:
                return [a + b + c + d for ((a, b), c), d in zip(ps, xs)]

        new_ast = ZipElim.apply(f.ast)
        assert not _contains(new_ast, Zip)
        ps = [((1.0, 2.0), 3.0), ((4.0, 5.0), 6.0)]
        xs = [10.0, 20.0]
        assert list(_eval(new_ast, f, ps, xs)) == list(f(ps, xs))

    def test_underscore_inside_nested_binding(self):
        @fp.fpy
        def f(pairs: list[tuple[fp.Real, fp.Real]], xs: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [b + c for (_, b), c in zip(pairs, xs)]

        new_ast = ZipElim.apply(f.ast)
        assert not _contains(new_ast, Zip)
        pairs = [(99.0, 2.0), (99.0, 4.0)]
        xs = [10.0, 20.0]
        assert list(_eval(new_ast, f, pairs, xs)) == list(f(pairs, xs))
