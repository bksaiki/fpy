"""
Unit tests for context-use analysis.
"""

import fpy2 as fp


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
