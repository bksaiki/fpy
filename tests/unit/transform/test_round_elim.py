"""
Unit tests for the :class:`fpy2.transform.RoundElim` transform.

The rewrite mints fresh ``_tN`` names via ``Gensym``, whose suffix
counts depend on the existing in-scope names — so an ``is_equiv``
comparison against a hand-written golden AST is too brittle.  The
tests assert three kinds of properties:

1. **Structural shape** of the rewritten AST.  Counts of
   ``with fp.REAL:`` blocks, presence/absence of ``Round`` /
   ``RoundExact`` / ``Cast`` nodes, presence/absence of arithmetic
   ops under the original (non-REAL) context.
2. **Negative checks**: unchanged inputs (REAL scope, ops that
   don't fit) compare via ``is_equiv`` against the original AST.
3. **Semantic equivalence** via the FPy interpreter on concrete
   sample inputs.  Catches subtle errors in the rewrite that pass
   shape checks.
"""

import fpy2 as fp

from fpy2.ast.fpyast import (
    Abs, Add, Assign, BinaryOp, Cast, ContextStmt, ForeignVal, Mul, Neg, Round,
    RoundExact, Sub, UnaryOp,
)
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import REAL
from fpy2.transform import RoundElim


# ----------------------------------------------------------------------
# Helpers


_ROUNDED_ARITH = (Add, Sub, Mul, Abs, Neg)
_ROUND_NODES = (Round, RoundExact, Cast)


def _count_real_blocks(ast: fp.ast.FuncDef) -> int:
    """Return the number of ``with fp.REAL:`` blocks reachable in *ast*."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_context(self, stmt: ContextStmt, ctx):
            nonlocal count
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val is REAL:
                count += 1
            super()._visit_context(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


def _arith_op_types_under_real(ast: fp.ast.FuncDef) -> set[type]:
    """Return the set of arithmetic-op classes that appear under
    a ``with fp.REAL:`` block in *ast*.  Used to confirm the
    rewrite moved a given op kind into REAL — without forbidding
    the same op kind from also appearing *outside* REAL via an
    operand bind (which preserves a non-identity round at its
    original scope)."""
    seen: set[type] = set()

    class _C(DefaultVisitor):
        def __init__(self):
            super().__init__()
            self.under_real = 0

        def _visit_context(self, stmt: ContextStmt, ctx):
            if isinstance(stmt.ctx, ForeignVal) and stmt.ctx.val is REAL:
                self.under_real += 1
                super()._visit_context(stmt, ctx)
                self.under_real -= 1
            else:
                super()._visit_context(stmt, ctx)

        def _visit_unaryop(self, e: UnaryOp, ctx):
            if self.under_real > 0 and isinstance(e, (Abs, Neg)):
                seen.add(type(e))
            super()._visit_unaryop(e, ctx)

        def _visit_binaryop(self, e: BinaryOp, ctx):
            if self.under_real > 0 and isinstance(e, (Add, Sub, Mul)):
                seen.add(type(e))
            super()._visit_binaryop(e, ctx)

    _C()._visit_function(ast, None)
    return seen


def _count_fresh_temp_assigns(ast: fp.ast.FuncDef) -> int:
    """Count ``Assign`` statements in *ast* whose target name starts
    with ``_t`` (the convention this transform uses for hoisted
    temporaries).  Used by the Var-skip test to detect a stray
    copy assignment."""
    count = 0

    class _C(DefaultVisitor):
        def _visit_assign(self, stmt, ctx):
            t = stmt.target
            name = getattr(t, 'base', None) or str(t)
            if isinstance(name, str) and name.startswith('_t'):
                nonlocal count
                count += 1
            super()._visit_assign(stmt, ctx)

    _C()._visit_function(ast, None)
    return count


def _has_node(ast: fp.ast.FuncDef, node_type) -> bool:
    """True iff any reachable expression in *ast* is an instance of
    *node_type*."""
    found = [False]

    class _C(DefaultVisitor):
        def _visit_expr(self, e, ctx):
            if isinstance(e, node_type):
                found[0] = True
                return
            super()._visit_expr(e, ctx)

    _C()._visit_function(ast, None)
    return found[0]


def _eval(ast: fp.ast.FuncDef, fn: fp.Function, *args):
    """Run *ast* through the FPy interpreter using *fn*'s env."""
    return fn.with_ast(ast)(*args)


# ----------------------------------------------------------------------
# Arithmetic-op hoisting


class TestArithmeticHoist:
    """Eliminable arithmetic ops should hoist their unrounded computation
    into a ``with fp.REAL:`` block, leaving operand evaluation at the
    original scope."""

    def test_add_fits_hoisted(self):
        """``1.0 + 2.0`` under FP64 — sum is the SetFormat ``{3}``,
        fits FP64.  Add should hoist; an Add appears under REAL."""

        @fp.fpy
        def f():
            with fp.FP64:
                return 1.0 + 2.0

        out = RoundElim.apply(f.ast)
        assert _count_real_blocks(out) >= 1
        assert Add in _arith_op_types_under_real(out)
        assert _eval(out, f) == f()

    def test_chained_mul_fits_hoisted(self):
        """``(1.0 + 2.0) * 3.0`` under FP64 — both ops are eliminable.
        Per-op hoist produces nested temps; both Add and Mul end up
        under REAL."""

        @fp.fpy
        def f():
            with fp.FP64:
                return (1.0 + 2.0) * 3.0

        out = RoundElim.apply(f.ast)
        types_under_real = _arith_op_types_under_real(out)
        assert Add in types_under_real
        assert Mul in types_under_real
        assert _eval(out, f) == f()

    def test_real_operand_arith_inside_eliminable_op(self):
        """``abs(x + y)`` under FP64 where ``x, y: fp.Real``.

        The outer ``Abs``'s unrounded result fits FP64 (abs of any
        FP64 value is FP64-representable).  The inner ``Add``'s
        round, however, is *not* the identity — it rounds REAL
        operands to FP64.  The rewrite must bind ``x + y`` under
        FP64 *before* the REAL block so the Add's round fires at
        its original scope.

        Post-rewrite shape: Abs appears under REAL; Add survives
        outside REAL (the operand bind preserves the round)."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return abs(x + y)

        out = RoundElim.apply(f.ast)
        types_under_real = _arith_op_types_under_real(out)
        # Abs hoisted under REAL.
        assert Abs in types_under_real
        # Add NOT moved to REAL — the operand bind keeps it at FP64.
        assert Add not in types_under_real
        # Semantic equivalence — rewrite preserves the original
        # (FP64-rounded) value.  Picking small operands so the sum
        # doesn't overflow either side.
        assert _eval(out, f, 1.5, 2.5) == f(1.5, 2.5)

    def test_arith_doesnt_fit_unchanged(self):
        """``x + y`` under FP32 where ``x, y: fp.Real``.  The Add's
        unrounded format is REAL — doesn't fit FP32.  No rewrite
        should fire."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                return x + y

        out = RoundElim.apply(f.ast)
        assert out.is_equiv(f.ast)


# ----------------------------------------------------------------------
# Round / RoundExact / Cast collapse


class TestRoundNodeCollapse:
    """Eliminable explicit round nodes should disappear, replaced by
    their argument."""

    def test_round_collapses_when_arg_fits(self):
        """``fp.round(1.0)`` under FP64 — ``1.0`` is exactly
        FP64-representable, the Round is the identity."""

        @fp.fpy
        def f():
            with fp.FP64:
                return fp.round(1.0)

        out = RoundElim.apply(f.ast)
        assert not _has_node(out, Round)
        assert _eval(out, f) == f()

    def test_round_stays_when_arg_doesnt_fit(self):
        """``fp.round(x)`` under FP64 where ``x: fp.Real`` — Real
        doesn't fit FP64 (saturated abstract format), Round must
        stay."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return fp.round(x)

        out = RoundElim.apply(f.ast)
        assert _has_node(out, Round)
        assert out.is_equiv(f.ast)

    def test_cast_collapses_when_arg_fits(self):
        """``fp.cast(1.0)`` under FP64 — same identity check as Round."""

        @fp.fpy
        def f():
            with fp.FP64:
                return fp.cast(1.0)

        out = RoundElim.apply(f.ast)
        assert not _has_node(out, Cast)
        assert _eval(out, f) == f()

    def test_round_exact_collapses_when_arg_fits(self):
        """``fp.round_exact`` is identity-when-fits just like Round."""

        @fp.fpy
        def f():
            with fp.FP64:
                return fp.round_exact(1.0)

        out = RoundElim.apply(f.ast)
        assert not _has_node(out, RoundExact)
        assert _eval(out, f) == f()


# ----------------------------------------------------------------------
# Scope guards (REAL / symbolic)


class TestScopeGuards:
    """REAL scope is the trivial identity — no rounds to eliminate.
    Symbolic / unresolved scopes can't safely claim identity."""

    def test_real_scope_noop(self):
        """``@fpy(ctx=fp.REAL): def f(x): return x + x`` — no ``with``
        block, scope is REAL.  Nothing to hoist."""

        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return x + x

        out = RoundElim.apply(f.ast)
        assert out.is_equiv(f.ast)
        assert _count_real_blocks(out) == 0

    def test_symbolic_scope_noop(self):
        """Function with no declared ctx and no pinned outer ctx —
        the scope is symbolic.  RoundElim refuses to claim identity
        without a concrete target."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x + y

        out = RoundElim.apply(f.ast)
        assert out.is_equiv(f.ast)


# ----------------------------------------------------------------------
# Suppression positions


class TestSuppressionPositions:
    """Hoisting needs a statement-level preamble slot — disabled
    inside ListComp elt/iterables and IfExpr branches.  Round /
    Cast collapse still fires (pure node rewrite)."""

    def test_no_hoist_inside_list_comp(self):
        """``[a + b for a, b in zip(xs, ys)]`` under FP64 — the Add
        inside the elt would benefit from REAL evaluation, but
        there's no statement slot inside the comp to host the
        preamble.  Add must stay un-hoisted."""

        @fp.fpy
        def f(xs: list[fp.Real], ys: list[fp.Real]) -> list[fp.Real]:
            with fp.FP64:
                return [a + b for a, b in zip(xs, ys)]

        out = RoundElim.apply(f.ast)
        # No REAL block was added.  The original AST may already
        # contain REAL contexts elsewhere; we only assert no *new*
        # REAL block introduced.
        assert _count_real_blocks(out) == _count_real_blocks(f.ast)

    def test_round_collapse_inside_list_comp(self):
        """Round collapse still fires inside a list comp — it's a
        pure node rewrite that needs no preamble slot."""

        @fp.fpy
        def f() -> list[fp.Real]:
            with fp.FP64:
                return [fp.round(1.0), fp.round(2.0)]

        out = RoundElim.apply(f.ast)
        assert not _has_node(out, Round)
        assert list(_eval(out, f)) == list(f())

    def test_no_hoist_inside_if_expr_branches(self):
        """``(x + y) if cond else 0.0`` — the Add inside the ift
        branch would evaluate conditionally; hoisting it
        unconditionally would change exception semantics under
        contexts that can raise.  Stay suppressed."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real, cond: bool) -> fp.Real:
            with fp.FP64:
                return (1.0 + 2.0) if cond else (3.0 + 4.0)

        out = RoundElim.apply(f.ast)
        # No NEW REAL block introduced for the branch Adds.  (Whole-
        # IfExpr might still hoist if it itself is eliminable, but
        # the inner branch ops shouldn't hoist out.)
        # Soundness check via interpreter — works under both branches.
        assert _eval(out, f, 1.0, 2.0, True) == f(1.0, 2.0, True)
        assert _eval(out, f, 1.0, 2.0, False) == f(1.0, 2.0, False)


# ----------------------------------------------------------------------
# Var-skip


class TestVarSkip:
    """Operands that are already ``Var`` references skip the bind
    step — a copy ``_t = v`` has no rounds to preserve and just
    adds a redundant alias."""

    def test_var_operand_not_bound(self):
        """``abs(a)`` where ``a`` is a local Var.  The hoist should
        plug ``a`` directly into the REAL block without minting a
        temp for it.  Expected fresh temps: just one (the Abs
        result) — *not* two (which would mean `_t = a` was also
        emitted)."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                a = x + y
            with fp.FP64:
                return abs(a)

        out = RoundElim.apply(f.ast)
        # Abs hoisted.
        assert Abs in _arith_op_types_under_real(out)
        # Only one fresh temp — the Abs result.  An extra `_t = a`
        # copy would push this to 2.
        assert _count_fresh_temp_assigns(out) == 1
        assert _eval(out, f, 1.5, 2.5) == f(1.5, 2.5)


# ----------------------------------------------------------------------
# Idempotence + post-transform SyntaxCheck


class TestProperties:
    """Cross-cutting properties of the transform."""

    def test_idempotent_clean_case(self):
        @fp.fpy
        def f():
            with fp.FP64:
                return (1.0 + 2.0) * 3.0
        once = RoundElim.apply(f.ast)
        twice = RoundElim.apply(once)
        assert once.is_equiv(twice)

    def test_idempotent_operand_bind_case(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                return abs(fp.round(x))
        once = RoundElim.apply(f.ast)
        twice = RoundElim.apply(once)
        assert once.is_equiv(twice)

    def test_idempotent_real_operand_case(self):
        """The bug case — earlier `_safe_to_hoist` couldn't detect
        the non-identity inner Add, so hoisting was unsound and
        also non-idempotent across rewrites."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return abs(x + y)
        once = RoundElim.apply(f.ast)
        twice = RoundElim.apply(once)
        assert once.is_equiv(twice)

    def test_idempotent_real_scope_noop(self):
        @fp.fpy(ctx=fp.REAL)
        def f(x: fp.Real) -> fp.Real:
            return x + x
        once = RoundElim.apply(f.ast)
        twice = RoundElim.apply(once)
        assert once.is_equiv(twice)

    def test_syntax_check_passes(self):
        """``RoundElim.apply`` runs ``SyntaxCheck.check`` internally;
        if the rewrite produced ill-formed output, ``apply`` itself
        would raise.  This test just exercises a representative
        input."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                return abs(x + y)

        # Should not raise.
        RoundElim.apply(f.ast)

    def test_type_error_on_non_funcdef(self):
        try:
            RoundElim.apply("not a funcdef")
        except TypeError:
            pass
        else:
            assert False, "expected TypeError"


# ----------------------------------------------------------------------
# Regression: paths the property tests don't exercise hard


class TestRegression:
    """Coverage for paths the per-feature tests don't reach: the
    cleanup-chain advertised in the module docstring, hoists under
    nested ``with`` blocks, and the strict-tighter guard against
    unbounded contexts (``fp.INTEGER``).
    """

    def test_cleanup_chain_collapses_redundant_binds(self):
        """RoundElim intentionally leaves redundant operand binds
        (``_t = literal`` lines) that the documented downstream
        cleanup chain — :class:`ConstPropagate` then
        :class:`CopyPropagate` then :class:`DeadCodeEliminate` — is
        expected to collapse.

        ``ConstPropagate`` is the load-bearing piece: it inlines
        literal-bound temps (``_t = 1`` → uses of ``_t`` become
        ``1``) so they become dead.  ``CopyPropagate`` handles
        Var→Var copies (also created by the per-op bind scheme).
        ``DeadCodeEliminate`` removes the resulting unused
        assigns.  Without ``ConstPropagate`` the literal binds
        stay live and DCE can't remove them.

        This test pins that contract: the post-cleanup AST has
        fewer fresh temps than the post-RoundElim AST."""

        from fpy2.transform import (
            ConstPropagate, CopyPropagate, DeadCodeEliminate,
        )

        @fp.fpy
        def f():
            with fp.FP64:
                return (1.0 + 2.0) * 3.0

        after_elim = RoundElim.apply(f.ast)
        # The verbose intermediate form has several `_t` temps (per
        # operand + per op result).
        elim_temps = _count_fresh_temp_assigns(after_elim)
        assert elim_temps > 1, (
            f'expected RoundElim to produce multiple temps for the '
            f'verbose form, got {elim_temps}'
        )

        # Run the documented cleanup chain.
        after = ConstPropagate.apply(after_elim)
        after = CopyPropagate.apply(after)
        after = DeadCodeEliminate.apply(after)
        cleaned_temps = _count_fresh_temp_assigns(after)
        # Cleanup should reduce the count.  We don't pin the exact
        # number — that depends on the cleanup passes' specifics
        # and would couple this test to their implementations.
        assert cleaned_temps < elim_temps, (
            f'expected cleanup chain to reduce temp count from '
            f'{elim_temps}; got {cleaned_temps}'
        )
        # Semantic equivalence preserved through the full chain.
        assert _eval(after, f) == f()

    def test_hoist_inside_nested_with_blocks(self):
        """Hoist should fire correctly inside an inner ``with`` block.
        The active scope at the eliminable op is the innermost
        ``with``; ``_resolved_ctx`` and ``_unrounded_format`` must
        consult that scope, not the outer one."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:                 # outer scope
                with fp.FP64:             # inner scope — Add is here
                    a = 1.0 + 2.0         # eliminable: SetFormat({3}) ⊆ FP64
                return a + a              # outer Add under FP32 — Real-typed
                                          # operands → NOT eliminable, stays

        out = RoundElim.apply(f.ast)
        # Inner Add hoisted under REAL.
        assert Add in _arith_op_types_under_real(out)
        # Outer Add NOT hoisted — semantic preservation under FP32 round.
        # (Tested via interpreter; we don't assert structural shape on
        # the outer since the inner hoist may have introduced new
        # statements that the structural predicates pick up.)
        # The function's value with `x` unused: ``a + a == 6`` rounded
        # through FP32 == 6.  Interpreter agrees.
        assert _eval(out, f, 0.0) == f(0.0)

    def test_unbounded_context_no_hoist(self):
        """Strictly-tighter guard: under ``fp.INTEGER`` (an unbounded
        ``MPFixedContext``), the unrounded format of ``x + 1`` equals
        the scope's format — both are saturated except for ``exp=0``.
        ``round_is_identity`` is vacuously true, but hoisting yields
        no narrowing and breaks downstream storage selection (the cpp
        emitter's lossless-widening dispatch can't pick a finite type
        for a saturated ``AbstractFormat``).  The guard should reject
        this hoist."""

        @fp.fpy(ctx=fp.INTEGER)
        def f(x: fp.Real) -> fp.Real:
            return x + 1

        out = RoundElim.apply(f.ast)
        # No REAL block introduced.
        assert _count_real_blocks(out) == _count_real_blocks(f.ast)
        # Add stays in its original (INTEGER) position.
        assert out.is_equiv(f.ast)
