"""
Unit tests for context-use analysis.

The handwritten ``TestContextUse`` suite covers basic invariants on small
fixed programs. ``TestContextUseOnGeneratedPrograms`` adds property tests
driven by the type-directed generator: it builds programs with random
``with``-block nesting and checks scope count, concrete-context
resolution, and the round-trip between ``uses`` and ``use_to_scope``.
"""

import fpy2 as fp

from fpy2 import dim, size
from hypothesis import given, settings, strategies as st

from fpy2.analysis.context_use import ContextUse
from fpy2.analysis.context_use import PartialContext
from fpy2.ast.fpyast import Ast, Call, ContextStmt, ForeignVal, FuncDef, Var
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
    # ContextStmt whose constructor is known but whose arguments are not

    def test_context_stmt_partial(self):
        """A context constructor with a runtime argument keeps its shape."""
        @fp.fpy
        def f(x, n):
            with fp.MPFixedContext(n):
                return fp.round(x)

        result = fp.analysis.ContextUse.analyze(f.ast)
        with_scope = result.scopes[-1]

        assert isinstance(with_scope.ctx, PartialContext)
        assert with_scope.ctx.cls is fp.MPFixedContext
        # the unresolved position survives as the expression itself
        (pos,) = with_scope.ctx.args
        assert isinstance(pos, Var)

    def test_context_stmt_partial_reduces_static_args(self):
        """Arguments that *do* reduce are recorded as values, not expressions."""
        @fp.fpy
        def f(x, n):
            with fp.MPFixedContext(n, fp.RM.RTN):
                return fp.round(x)

        result = fp.analysis.ContextUse.analyze(f.ast)
        ctx = result.scopes[-1].ctx

        assert isinstance(ctx, PartialContext)
        pos, rm = ctx.args
        assert isinstance(pos, Var)
        assert not isinstance(rm, Ast)

    def test_context_stmt_fully_static_is_concrete(self):
        """A constructor whose arguments all reduce is still a ``Context``."""
        @fp.fpy
        def f(x):
            with fp.MPFixedContext(5):
                return fp.round(x)

        result = fp.analysis.ContextUse.analyze(f.ast)
        assert isinstance(result.scopes[-1].ctx, Context)

    def test_partial_context_holes(self):
        """``holes`` names the arguments a caller still has to pin."""
        @fp.fpy
        def f(x, n, rm):
            with fp.MPFixedContext(n, rm):
                return fp.round(x)

        ctx = fp.analysis.ContextUse.analyze(f.ast).scopes[-1].ctx
        assert [h.format() for h in ctx.holes] == ['n', 'rm']

        @fp.fpy
        def g(x, n):
            with fp.MPFixedContext(n, fp.RM.RTN):
                return fp.round(x)

        ctx = fp.analysis.ContextUse.analyze(g.ast).scopes[-1].ctx
        assert [h.format() for h in ctx.holes] == ['n']

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
    # Operations that read no context

    def test_list_queries_are_not_uses(self):
        """``len`` / ``dim`` / ``size`` / ``range`` / ``enumerate`` give an
        integer answer whatever the scope, so they are not uses: a ``with``
        wrapping only these is unobservable."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                n = len(xs)
                d = dim(xs)
                m = size(xs, 0)
                r = range(n, m, d)
                e = enumerate(r)
            return xs[len(e) - 1]

        result = fp.analysis.ContextUse.analyze(f.ast)
        with_scope = result.scopes[-1]
        assert result.uses[with_scope] == set()

    def test_an_allocation_is_not_a_use(self):
        """``empty`` reserves list slots and rounds nothing, so a scope it
        allocates under is unobservable too."""
        @fp.fpy
        def f(n: fp.Real) -> list[fp.Real]:
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                ys = fp.empty(n)
            return ys

        result = fp.analysis.ContextUse.analyze(f.ast)
        with_scope = result.scopes[-1]
        assert result.uses[with_scope] == set()

    def test_a_rounding_beside_a_query_is_still_a_use(self):
        """The exclusion is per operator, not per block."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.IEEEContext(11, 64, fp.RM.RNE):
                n = len(xs)
                y = xs[0] + 1.0
            return y * n

        result = fp.analysis.ContextUse.analyze(f.ast)
        with_scope = result.scopes[-1]
        assert len(result.uses[with_scope]) == 1
        use = next(iter(result.uses[with_scope]))
        assert isinstance(use, fp.ast.Add)

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

    def test_a_context_expression_uses_real(self):
        """**E-Context** evaluates a ``with``'s context expression under
        ``REAL``, so the arithmetic inside it is used under ``REAL`` -- not
        under the enclosing context, and not under nothing at all."""
        @fp.fpy
        def f():
            ES = 2
            NB = 8
            with fp.IEEEContext(ES + 2, NB + 2):
                return fp.round(1)

        result = fp.analysis.ContextUse.analyze(f.ast)
        stmt = f.ast.body.stmts[2]
        assert isinstance(stmt, ContextStmt)

        assert isinstance(stmt.ctx, Call)
        adds = stmt.ctx.args           # `ES + 2` and `NB + 2`
        assert len(adds) == 2
        for a in adds:
            assert result.find_scope_from_use(a).ctx is fp.REAL

        # The `REAL` scope introduces no `with`, so it stays out of `scopes` --
        # consumers keying scopes by site would collapse it against the body's.
        assert [s.site for s in result.scopes] == [f.ast, stmt]
        assert all(s.ctx is not fp.REAL for s in result.scopes)

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

    Generated functions have no function-level ``ctx`` annotation, so the
    function scope is always symbolic, and every ``with``-block context is a
    ``ForeignVal`` of a concrete :class:`Context`, so it always resolves
    concretely.  One test, since generating the program is nearly all of the
    cost.
    """

    @given(fpy_real_funcdef(**_GEN_KWARGS))
    @settings(max_examples=80, deadline=None)
    def test_generated_programs(self, fd: FuncDef) -> None:
        result = ContextUse.analyze(fd)
        withs = [n for n in _walk_ast(fd.body) if isinstance(n, ContextStmt)]

        # one scope for the function, then one per `with`
        assert len(result.scopes) == 1 + len(withs), (
            f'expected {1 + len(withs)} scopes (1 + {len(withs)} with-blocks), '
            f'got {len(result.scopes)}'
        )
        assert result.scopes[0].site is fd
        assert isinstance(result.scopes[0].ctx, NamedId)

        # each `with` resolves to the very context it names
        scope_by_site = {s.site: s for s in result.scopes}
        for node in withs:
            assert isinstance(node.ctx, ForeignVal), (
                'generator unexpectedly emitted a non-literal with-ctx; '
                'this test relies on `ForeignVal(<Context>, None)`'
            )
            scope = scope_by_site[node]
            assert isinstance(scope.ctx, Context), (
                f'with-block ctx did not resolve to a concrete Context: '
                f'got {scope.ctx!r} for site {node}'
            )
            assert scope.ctx is node.ctx.val, (
                f'with-block ctx resolved to wrong Context: '
                f'expected {node.ctx.val!r}, got {scope.ctx!r}'
            )

        # `use_to_scope` inverts `uses`, and no use is in two scopes
        seen: set = set()
        for scope, uses in result.uses.items():
            for u in uses:
                assert result.use_to_scope[u] is scope
                assert id(u) not in seen, 'use appears in two scopes'
                seen.add(id(u))
        assert set(id(u) for u in result.use_to_scope) == seen
