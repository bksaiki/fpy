"""Unit tests for :func:`fpy2.strategies.simplify`.

The expect-test fixtures use ``enable_const_fold_context=False`` so
the ``with fp.FP64:`` expression stays as an ``Attribute`` — the
concrete ``IEEEContext`` produced by the default context-fold can't
be written in FPy surface syntax, so we'd have no way to express
the expected AST.  Context-folding itself is tested separately.
"""

import fpy2 as fp

from fpy2.ast import Attribute, BinaryOp, ContextStmt, ForeignVal
from fpy2.ast.visitor import DefaultVisitor
from fpy2.strategies import simplify


# Module-level free vars referenced by the fixtures below.  Both
# should disappear after ``simplify`` folds them and DCE drops the
# gated code.
DEBUG = False
SCALE = 2.0


# ---------------------------------------------------------------------------
# Fixture pairs: (input, expected after simplify)
# ---------------------------------------------------------------------------


@fp.fpy
def _kitchen_sink(x: fp.Real) -> fp.Real:
    with fp.FP64:
        # Numeric folding bottom-up.
        a = 1.0 + 2.0
        b = a * a
        c = b - 1.0
        # List + reduce.
        coeffs = [c, c + 1.0, c + 2.0]
        peak = max(coeffs)
        # Tuple destructure carries values forward.
        pair = (a, peak)
        first, second = pair
        # Compare-folds → branch elimination.
        if peak < 0.0:
            return -1.0
        # IfExpr — Compare folds to True, IfExpr picks the ift branch.
        sign = 1.0 if first > 0.0 else -1.0
        # Copy chain.
        y = second
        z = y
        w = z
        # Foreign-var gated dead block.
        if DEBUG:
            assert w > 999.0, "impossible"
        # Pure effect.
        a + b + peak
        # Always-true assert.
        assert True
        return sign * (w + SCALE * x)


@fp.fpy
def _kitchen_sink_expect(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return 1 * (10 + 2 * x)


@fp.fpy
def _just_const_fold(x: fp.Real) -> fp.Real:
    with fp.FP64:
        a = 3.0
        b = a + 2.0
        return b * x


@fp.fpy
def _just_const_fold_expect(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return 5 * x


@fp.fpy
def _just_copy_prop(x: fp.Real) -> fp.Real:
    with fp.FP64:
        y = x
        z = y
        return z


@fp.fpy
def _just_copy_prop_expect(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return x


@fp.fpy
def _just_dead_branches(x: fp.Real) -> fp.Real:
    with fp.FP64:
        if False:
            return -1.0
        assert True
        return x


@fp.fpy
def _just_dead_branches_expect(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return x


@fp.fpy
def _if_expr_fold(x: fp.Real) -> fp.Real:
    with fp.FP64:
        # IfExpr with a foldable cond — picks the matching branch's
        # value even though the other branch isn't statically known.
        sign = 1.0 if True else x
        return sign * x


@fp.fpy
def _if_expr_fold_expect(x: fp.Real) -> fp.Real:
    with fp.FP64:
        return 1 * x


@fp.fpy
def _branch_merge(c: bool) -> fp.Real:
    with fp.FP64:
        # Both branches assign the same value — SCCP phi-merges to 5,
        # ConstFold substitutes ``return 5``, DCE prunes the if.
        if c:
            x = 5.0
        else:
            x = 5.0
        return x


@fp.fpy
def _branch_merge_expect(c: bool) -> fp.Real:
    with fp.FP64:
        return 5


@fp.fpy
def _loop_invariant(c: bool) -> fp.Real:
    with fp.FP64:
        # Loop-invariant variable — SCCP phi-merges 5 (pre-loop) and
        # 5 (body), ConstFold substitutes ``return 5``.
        x = 5.0
        while c:
            x = 5.0
        return x


@fp.fpy
def _loop_invariant_expect(c: bool) -> fp.Real:
    with fp.FP64:
        while c:
            pass
        return 5


_examples: list[tuple[fp.Function, fp.Function]] = [
    (_kitchen_sink, _kitchen_sink_expect),
    (_just_const_fold, _just_const_fold_expect),
    (_just_copy_prop, _just_copy_prop_expect),
    (_just_dead_branches, _just_dead_branches_expect),
    (_if_expr_fold, _if_expr_fold_expect),
    (_branch_merge, _branch_merge_expect),
    (_loop_invariant, _loop_invariant_expect),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_node(fd, predicate) -> bool:
    """True iff some AST node in *fd* satisfies *predicate*."""
    hit = [False]

    class _Search(DefaultVisitor):
        def _visit_statement(self, stmt, ctx):
            if predicate(stmt):
                hit[0] = True
            return super()._visit_statement(stmt, ctx)

        def _visit_expr(self, e, ctx):
            if predicate(e):
                hit[0] = True
            return super()._visit_expr(e, ctx)

    _Search()._visit_function(fd, None)
    return hit[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimplifyExpect:
    """Pin the exact simplified AST for each fixture pair."""

    def test_fixtures(self):
        for f, f_expect in _examples:
            simplified = simplify(f, enable_const_fold_context=False).ast
            simplified.name = f_expect.name
            assert simplified.is_equiv(f_expect.ast), (
                f'\nexpect:\n{f_expect.ast.format()}\n'
                f'actual:\n{simplified.format()}'
            )


class TestSimplifyRuntime:
    """Even when the AST shape drifts (e.g. future passes get more
    aggressive), runtime behaviour must be preserved."""

    def test_runtime_equivalent(self):
        for f, _ in _examples:
            simplified = simplify(f)
            for x in (0.0, 1.0, -1.5, 2.5, 100.0):
                assert float(f(x)) == float(simplified(x)), (
                    f'{f.name}: simplified differs at x={x}: '
                    f'{f(x)} vs {simplified(x)}'
                )


class TestSimplifyIdempotent:
    def test_idempotent(self):
        for f, _ in _examples:
            once = simplify(f)
            twice = simplify(once)
            assert once.ast.is_equiv(twice.ast), (
                f'{f.name}: simplify is not idempotent:\n'
                f'once:\n{once.ast.format()}\n'
                f'twice:\n{twice.ast.format()}'
            )


class TestSimplifyContextFold:
    """The default ``simplify`` folds ``fp.FP64`` (an ``Attribute``)
    to a ``ForeignVal`` carrying the concrete ``IEEEContext``.  The
    knob ``enable_const_fold_context=False`` opts out."""

    def test_default_folds_to_foreign_val(self):
        simplified = simplify(_kitchen_sink).ast
        assert _has_node(
            simplified,
            lambda n: isinstance(n, ContextStmt) and isinstance(n.ctx, ForeignVal),
        )
        # No Attribute remaining in context position.
        assert not _has_node(
            simplified,
            lambda n: isinstance(n, ContextStmt) and isinstance(n.ctx, Attribute),
        )

    def test_disable_keeps_attribute(self):
        simplified = simplify(_kitchen_sink, enable_const_fold_context=False).ast
        assert _has_node(
            simplified,
            lambda n: isinstance(n, ContextStmt) and isinstance(n.ctx, Attribute),
        )


class TestSimplifyOpFold:
    """``enable_const_fold_op=False`` preserves arithmetic, leaves the
    context-fold knob active."""

    def test_disable_keeps_arithmetic(self):
        simplified = simplify(_just_const_fold, enable_const_fold_op=False).ast
        # Some BinaryOp must remain (the un-folded ``a + 2.0`` etc.).
        assert _has_node(simplified, lambda n: isinstance(n, BinaryOp))
