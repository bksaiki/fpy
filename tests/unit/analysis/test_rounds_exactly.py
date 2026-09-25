"""`rounds_exactly`: does an operation's implicit round change anything?"""

import fpy2 as fp
from fpy2 import Function
from fpy2.analysis import ContextUse, FormatInfer
from fpy2.analysis.format_infer import rounds_exactly, unrounded_format
from fpy2.ast.fpyast import Add, Expr
from fpy2.number import Context
from fpy2.transform.path import walk_exprs


def _only_add(func: Function) -> tuple[Expr, Context | None, dict]:
    adds = [e for _, e in walk_exprs(func.ast) if isinstance(e, Add)]
    assert len(adds) == 1, f'expected one add, got {len(adds)}'
    e = adds[0]
    fmt = FormatInfer.analyze(func.ast)
    scope = ContextUse.analyze(func.ast).find_scope_from_use(e)
    c = scope.ctx if isinstance(scope.ctx, Context) else None
    return e, c, fmt.by_expr


class TestExact:
    def test_integers_accumulate_exactly(self):
        """`INTEGER` is unbounded, so a sum of two integers never rounds."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(a: fp.Real, b: fp.Real):
            x = fp.round(a)
            y = fp.round(b)
            return x + y

        e, c, by_expr = _only_add(f)
        assert rounds_exactly(e, by_expr, c)

    def test_unconstrained_operands_are_not_exact(self):
        """The operands must be *known* integers.  Two arbitrary reals under
        `INTEGER` round on the way in, so the sum is not exact."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(a: fp.Real, b: fp.Real):
            return a + b

        e, c, by_expr = _only_add(f)
        assert not rounds_exactly(e, by_expr, c)


class TestRounds:
    def test_a_float_sum_of_unknown_values_rounds(self):
        """Nothing bounds the operands, so the exact sum need not fit FP32."""
        @fp.fpy(ctx=fp.FP32)
        def f(a: fp.Real, b: fp.Real):
            return a + b

        e, c, by_expr = _only_add(f)
        assert not rounds_exactly(e, by_expr, c)


class TestIllPosed:
    def test_an_expression_with_no_round_has_no_unrounded_format(self):
        """A comparison carries no context-driven round."""
        @fp.fpy(ctx=fp.FP64)
        def f(a: fp.Real, b: fp.Real):
            return a < b

        fmt = FormatInfer.analyze(f.ast)
        ret = f.ast.body.stmts[-1]
        assert unrounded_format(ret.expr, fmt.by_expr) is None

    def test_an_unresolved_context_is_not_exact(self):
        """Without a concrete target there is nothing to claim identity
        against."""
        @fp.fpy(ctx=fp.INTEGER)
        def f(a: fp.Real, b: fp.Real):
            x = fp.round(a)
            y = fp.round(b)
            return x + y

        e, _c, by_expr = _only_add(f)
        assert not rounds_exactly(e, by_expr, None)
