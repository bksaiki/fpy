"""
Unit tests for context definition-use analysis.
"""

import fpy2 as fp


class TestCtxDefUse:

    # ------------------------------------------------------------------
    # Symbolic contexts (no overriding context on the function)

    def test_no_ctx_single_op(self):
        """Function with no context: body uses one symbolic context."""
        @fp.fpy
        def f(x):
            return x + 1.0

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        # Exactly one context def introduced by the FuncDef
        assert len(result.defs) == 1
        ctx_def = result.defs[0]
        assert ctx_def.site is f.ast
        # No overriding context → symbolic variable
        assert isinstance(ctx_def.ctx, fp.utils.NamedId)
        # The Add operation is a use under that def
        assert len(result.uses[ctx_def]) == 1

    def test_no_ctx_multiple_ops(self):
        """Multiple operations share the same symbolic context def."""
        @fp.fpy
        def f(x, y):
            a = x + y
            b = a * 2.0
            return a - b

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        assert len(result.defs) == 1
        ctx_def = result.defs[0]
        assert isinstance(ctx_def.ctx, fp.utils.NamedId)
        # Three binary ops (Add, Mul, Sub) are all uses
        assert len(result.uses[ctx_def]) == 3

    # ------------------------------------------------------------------
    # Concrete function-level context

    def test_funcdef_concrete_ctx(self):
        """Function with a concrete overriding context."""
        @fp.fpy(ctx=fp.IEEEContext(11, 64, fp.RM.RNE))
        def f(x):
            return x + 1.0

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        assert len(result.defs) == 1
        ctx_def = result.defs[0]
        assert ctx_def.site is f.ast
        assert isinstance(ctx_def.ctx, fp.number.Context)

    # ------------------------------------------------------------------
    # ContextStmt with a statically-resolvable context

    def test_context_stmt_literal(self):
        """with-statement that uses a literal context constructor."""
        @fp.fpy
        def f(x):
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                return x + 1.0

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        # One def for the function body, one for the with-block
        assert len(result.defs) == 2
        func_def, with_def = result.defs

        # Function body has no operations of its own (just the with-stmt)
        assert len(result.uses[func_def]) == 0

        # The with-block introduces a concrete context
        assert isinstance(with_def.ctx, fp.number.Context)
        # The Add inside the with-block is attributed to with_def
        assert len(result.uses[with_def]) == 1
        use = next(iter(result.uses[with_def]))
        assert isinstance(use, fp.ast.Add)

    def test_context_stmt_via_partial_eval(self):
        """with-statement where the context is reducible via partial evaluation."""
        @fp.fpy
        def f(x):
            ES = 11
            NB = 64
            with fp.IEEEContext(ES, NB, fp.RM.RNE):
                return x + 1.0

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        # The with-block context should be resolved to a concrete value
        with_def = result.defs[-1]
        assert isinstance(with_def.ctx, fp.number.Context)

    # ------------------------------------------------------------------
    # ContextStmt with a non-reducible context (symbolic fallback)

    def test_context_stmt_symbolic(self):
        """with-statement whose context depends on a runtime value."""
        @fp.fpy
        def f(x, ctx):
            with ctx:
                return x + 1.0

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        assert len(result.defs) == 2
        with_def = result.defs[-1]
        # Cannot be resolved statically → symbolic variable
        assert isinstance(with_def.ctx, fp.utils.NamedId)

    # ------------------------------------------------------------------
    # Nested ContextStmt

    def test_nested_context_stmts(self):
        """Nested with-statements produce separate context defs."""
        @fp.fpy
        def f(x):
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                a = x + 1.0
                with fp.IEEEContext(8, 32, fp.RM.RNE):
                    b = a * 2.0
                return a - b

        result = fp.analysis.CtxDefUse.analyze(f.ast)

        # Three defs: function, outer with, inner with
        assert len(result.defs) == 3
        func_def, outer_def, inner_def = result.defs

        # outer with: Add and Sub (a + 1, a - b)
        assert len(result.uses[outer_def]) == 2
        # inner with: Mul (a * 2)
        assert len(result.uses[inner_def]) == 1

    # ------------------------------------------------------------------
    # find_def_from_use / use_to_def

    def test_use_to_def_mapping(self):
        """Every context-sensitive expression maps back to its def."""
        @fp.fpy
        def f(x):
            a = x + 1.0
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                b = a * 2.0
            return a - b

        result = fp.analysis.CtxDefUse.analyze(f.ast)
        func_def, with_def = result.defs

        # Every use maps back to the correct def
        for u in result.uses[func_def]:
            assert result.find_def_from_use(u) is func_def
        for u in result.uses[with_def]:
            assert result.find_def_from_use(u) is with_def

    # ------------------------------------------------------------------
    # Accepting a pre-computed def_use

    def test_precomputed_def_use(self):
        """Passing an explicit DefineUseAnalysis should give the same result."""
        @fp.fpy
        def f(x):
            return x + 1.0

        def_use = fp.analysis.DefineUse.analyze(f.ast)
        result = fp.analysis.CtxDefUse.analyze(f.ast, def_use=def_use)

        assert len(result.defs) == 1
        assert isinstance(result.defs[0].ctx, fp.utils.NamedId)

    # ------------------------------------------------------------------
    # Error handling

    def test_invalid_input(self):
        """Passing a non-FuncDef raises TypeError."""
        import pytest
        with pytest.raises(TypeError):
            fp.analysis.CtxDefUse.analyze("not a func")
