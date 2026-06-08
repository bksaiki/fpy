"""
Unit tests for context-use analysis.

The handwritten ``TestContextUse`` suite covers basic invariants on small
fixed programs. ``TestContextUseOnGeneratedPrograms`` adds property tests
driven by the type-directed generator: it builds programs with random
``with``-block nesting and checks scope count, concrete-context
resolution, and the round-trip between ``uses`` and ``use_to_scope``.
"""

import fpy2 as fp

from hypothesis import given, settings, strategies as st

from fpy2.analysis.context_use import ContextUse
from fpy2.ast.fpyast import Ast, ContextStmt, ForeignVal, FuncDef
from fpy2.number import Context
from fpy2.utils import NamedId

from ..generators import fpy_real_funcdef


class TestContextUse:

    # ------------------------------------------------------------------
    # Symbolic contexts (no overriding context on the function)

    def test_no_ctx_single_op(self):
        """Function with no context: body uses one symbolic context."""
        @fp.fpy
        def f(x):
            return x + 1.0

        result = fp.analysis.ContextUse.analyze(f.ast)

        # Exactly one context scope introduced by the FuncDef
        assert len(result.scopes) == 1
        scope = result.scopes[0]
        assert scope.site is f.ast
        # No overriding context → symbolic variable
        assert isinstance(scope.ctx, fp.utils.NamedId)
        # The Add operation is a use under that scope
        assert len(result.uses[scope]) == 1

    def test_no_ctx_multiple_ops(self):
        """Multiple operations share the same symbolic context scope."""
        @fp.fpy
        def f(x, y):
            a = x + y
            b = a * 2.0
            return a - b

        result = fp.analysis.ContextUse.analyze(f.ast)

        assert len(result.scopes) == 1
        scope = result.scopes[0]
        assert isinstance(scope.ctx, fp.utils.NamedId)
        # Three binary ops (Add, Mul, Sub) are all uses
        assert len(result.uses[scope]) == 3

    # ------------------------------------------------------------------
    # Concrete function-level context

    def test_funcdef_concrete_ctx(self):
        """Function with a concrete overriding context."""
        @fp.fpy(ctx=fp.IEEEContext(11, 64, fp.RM.RNE))
        def f(x):
            return x + 1.0

        result = fp.analysis.ContextUse.analyze(f.ast)

        assert len(result.scopes) == 1
        scope = result.scopes[0]
        assert scope.site is f.ast
        assert isinstance(scope.ctx, fp.number.Context)

    # ------------------------------------------------------------------
    # ContextStmt with a statically-resolvable context

    def test_context_stmt_literal(self):
        """with-statement that uses a literal context constructor."""
        @fp.fpy
        def f(x):
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                return x + 1.0

        result = fp.analysis.ContextUse.analyze(f.ast)

        # One scope for the function body, one for the with-block
        assert len(result.scopes) == 2
        func_scope, with_scope = result.scopes

        # Function body has no operations of its own (just the with-stmt)
        assert len(result.uses[func_scope]) == 0

        # The with-block introduces a concrete context
        assert isinstance(with_scope.ctx, fp.number.Context)
        # The Add inside the with-block is attributed to with_scope
        assert len(result.uses[with_scope]) == 1
        use = next(iter(result.uses[with_scope]))
        assert isinstance(use, fp.ast.Add)

    def test_context_stmt_via_partial_eval(self):
        """with-statement where the context is reducible via partial evaluation."""
        @fp.fpy
        def f(x):
            ES = 11
            NB = 64
            with fp.IEEEContext(ES, NB, fp.RM.RNE):
                return x + 1.0

        result = fp.analysis.ContextUse.analyze(f.ast)

        # The with-block context should be resolved to a concrete value
        with_scope = result.scopes[-1]
        assert isinstance(with_scope.ctx, fp.number.Context)

    def test_context_stmt_kwargs(self):
        """with-statement using keyword arguments resolves to a concrete context."""
        @fp.fpy
        def f(x):
            with fp.IEEEContext(es=11, nbits=64, rm=fp.RM.RNE):
                return x + 1.0

        result = fp.analysis.ContextUse.analyze(f.ast)

        with_scope = result.scopes[-1]
        assert isinstance(with_scope.ctx, fp.number.Context)

    # ------------------------------------------------------------------
    # ContextStmt with a non-reducible context (symbolic fallback)

    def test_context_stmt_symbolic(self):
        """with-statement whose context depends on a runtime value."""
        @fp.fpy
        def f(x, ctx):
            with ctx:
                return x + 1.0

        result = fp.analysis.ContextUse.analyze(f.ast)

        assert len(result.scopes) == 2
        with_scope = result.scopes[-1]
        # Cannot be resolved statically → symbolic variable
        assert isinstance(with_scope.ctx, fp.utils.NamedId)

    # ------------------------------------------------------------------
    # Nested ContextStmt

    def test_nested_context_stmts(self):
        """Nested with-statements produce separate context scopes."""
        @fp.fpy
        def f(x):
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                a = x + 1.0
                with fp.IEEEContext(8, 32, fp.RM.RNE):
                    b = a * 2.0
                return a - b

        result = fp.analysis.ContextUse.analyze(f.ast)

        # Three scopes: function, outer with, inner with
        assert len(result.scopes) == 3
        func_scope, outer_scope, inner_scope = result.scopes

        # outer with: Add and Sub (a + 1, a - b)
        assert len(result.uses[outer_scope]) == 2
        # inner with: Mul (a * 2)
        assert len(result.uses[inner_scope]) == 1

    # ------------------------------------------------------------------
    # find_scope_from_use / use_to_scope

    def test_use_to_scope_mapping(self):
        """Every context-sensitive expression maps back to its scope."""
        @fp.fpy
        def f(x):
            a = x + 1.0
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                b = a * 2.0
            return a - b

        result = fp.analysis.ContextUse.analyze(f.ast)
        func_scope, with_scope = result.scopes

        # Every use maps back to the correct scope
        for u in result.uses[func_scope]:
            assert result.find_scope_from_use(u) is func_scope
        for u in result.uses[with_scope]:
            assert result.find_scope_from_use(u) is with_scope

    # ------------------------------------------------------------------
    # Accepting a pre-computed def_use

    def test_precomputed_def_use(self):
        """Passing an explicit DefineUseAnalysis should give the same result."""
        @fp.fpy
        def f(x):
            return x + 1.0

        def_use = fp.analysis.DefineUse.analyze(f.ast)
        result = fp.analysis.ContextUse.analyze(f.ast, def_use=def_use)

        assert len(result.scopes) == 1
        assert isinstance(result.scopes[0].ctx, fp.utils.NamedId)

    # ------------------------------------------------------------------
    # Error handling

    def test_invalid_input(self):
        """Passing a non-FuncDef raises TypeError."""
        import pytest
        with pytest.raises(TypeError):
            fp.analysis.ContextUse.analyze("not a func")


# ---------------------------------------------------------------------------
# Property tests driven by the type-directed generator
# ---------------------------------------------------------------------------

def _walk_ast(node):
    """Yield every AST node reachable from ``node`` via ``__slots__``."""
    if isinstance(node, Ast):
        yield node
    for slot in getattr(type(node), '__slots__', ()):
        try:
            val = getattr(node, slot)
        except AttributeError:
            continue
        if isinstance(val, Ast):
            yield from _walk_ast(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Ast):
                    yield from _walk_ast(item)


# Generator config used across the property tests. Compact wall-clock
# while still exercising nested ``with`` blocks.
_GEN_KWARGS = dict(
    num_args=st.integers(0, 2),
    max_depth=st.integers(1, 2),
    max_assigns=st.integers(0, 2),
    max_contexts=st.integers(0, 3),
    max_ifs=st.just(0),
    max_loops=st.just(0),
    max_whiles=st.just(0),
)


class TestContextUseOnGeneratedPrograms:
    """``ContextUse`` driven by the type-directed generator.

    Generated functions have:
    - no function-level ``ctx`` annotation (``FuncMeta.ctx is None``), so
      the function scope is always symbolic;
    - ``with``-block contexts that are always ``ForeignVal`` of a concrete
      :class:`Context` (drawn from ``_DEFAULT_CONTEXTS``), so they should
      always resolve concretely — never symbolic.
    """

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_scope_count_matches_context_stmt_count(self, fd: FuncDef) -> None:
        """``len(scopes) == 1 + (# of ContextStmts in body)``.

        The ``1`` is the function-level scope; each ``ContextStmt``
        introduces exactly one nested scope.
        """
        n_with = sum(1 for n in _walk_ast(fd.body) if isinstance(n, ContextStmt))
        result = ContextUse.analyze(fd)
        assert len(result.scopes) == 1 + n_with, (
            f'expected {1 + n_with} scopes (1 + {n_with} with-blocks), '
            f'got {len(result.scopes)}'
        )

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_function_scope_is_first_and_symbolic(self, fd: FuncDef) -> None:
        """``scopes[0]`` is the function-level scope; its ctx is a fresh
        symbolic ``NamedId`` (the generator never sets ``FuncMeta.ctx``)."""
        result = ContextUse.analyze(fd)
        assert result.scopes[0].site is fd
        assert isinstance(result.scopes[0].ctx, NamedId)

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_with_block_contexts_resolve_concretely(self, fd: FuncDef) -> None:
        """Every ``ContextStmt``'s scope should resolve to a concrete
        ``Context`` value (not a fresh symbolic ``NamedId``), because the
        generator only emits ``ForeignVal(<concrete_ctx>, None)`` for
        ``with``-block ctx expressions.

        If this ever fails, either partial-eval lost the value or the
        scope-lookup is mis-indexed.
        """
        result = ContextUse.analyze(fd)
        scope_by_site = {s.site: s for s in result.scopes}
        for node in _walk_ast(fd.body):
            if not isinstance(node, ContextStmt):
                continue
            # Sanity-check the generator's assumption first.
            assert isinstance(node.ctx, ForeignVal), (
                'generator unexpectedly emitted a non-literal with-ctx; '
                'this test relies on `ForeignVal(<Context>, None)`'
            )
            expected_ctx = node.ctx.val
            scope = scope_by_site[node]
            assert isinstance(scope.ctx, Context), (
                f'with-block ctx did not resolve to a concrete Context: '
                f'got {scope.ctx!r} for site {node}'
            )
            assert scope.ctx is expected_ctx, (
                f'with-block ctx resolved to wrong Context: '
                f'expected {expected_ctx!r}, got {scope.ctx!r}'
            )

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_use_to_scope_consistent_with_uses(self, fd: FuncDef) -> None:
        """``use_to_scope`` is the inverse of ``uses`` — every use in
        ``uses[s]`` should map back to ``s`` in ``use_to_scope``, and
        every key of ``use_to_scope`` should appear in exactly one
        ``uses`` set."""
        result = ContextUse.analyze(fd)
        # Forward direction: uses[s] ⇒ use_to_scope[u] is s
        for scope, uses in result.uses.items():
            for u in uses:
                assert result.use_to_scope[u] is scope
        # Reverse direction: every use_to_scope key appears in exactly one set
        seen: set = set()
        for uses in result.uses.values():
            for u in uses:
                assert id(u) not in seen, 'use appears in two scopes'
                seen.add(id(u))
        assert set(id(u) for u in result.use_to_scope) == seen
