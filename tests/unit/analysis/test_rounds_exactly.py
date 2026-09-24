"""`unrounded_format`: the exact result of a rounded operation."""

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.format_infer import unrounded_format


class TestIllPosed:
    def test_an_expression_with_no_round_has_no_unrounded_format(self):
        """A comparison carries no context-driven round."""
        @fp.fpy(ctx=fp.FP64)
        def f(a: fp.Real, b: fp.Real):
            return a < b

        fmt = FormatInfer.analyze(f.ast)
        ret = f.ast.body.stmts[-1]
        assert unrounded_format(ret.expr, fmt.by_expr) is None
