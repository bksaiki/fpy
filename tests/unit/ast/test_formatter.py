"""Unit tests for context resugaring in the AST formatter."""

import fpy2 as fp

from fpy2.transform import ConstFold


@fp.fpy(ctx=fp.FP64)
def _pinned(x: fp.Real) -> fp.Real:
    return x + 1.0


@fp.fpy(ctx=fp.IEEEContext(9, 40))
def _pinned_unnamed(x: fp.Real) -> fp.Real:
    return x + 1.0


@fp.fpy
def _with_attr(x: fp.Real) -> fp.Real:
    with fp.FP32:
        return x + 1.0


class TestContextResugar:
    """A ``ForeignVal`` carrying a named context formats as the
    ``fp.<NAME>`` surface syntax rather than the value's repr."""

    def test_decorator_ctx_named(self):
        assert 'ctx=fp.FP64' in _pinned.ast.format()

    def test_decorator_ctx_unnamed_falls_back(self):
        out = _pinned_unnamed.ast.format()
        assert 'IEEEContext(' in out

    def test_foreign_val_named(self):
        # const-folding `fp.FP32` produces a `ForeignVal` carrying the
        # concrete context, which resugars to the attribute syntax
        folded = ConstFold.apply(_with_attr.ast, enable_op=False)
        assert 'with fp.FP32:' in folded.format()
