"""Unit tests for the AST formatter."""

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


class TestCallArguments:
    """A call formats every argument it carries, keyword ones included."""

    def test_keyword_arguments(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FixedContext(signed=True, scale=-16, nbits=32):
                return fp.round(x)

        assert 'fp.FixedContext(signed=True, scale=-16, nbits=32)' in f.ast.format()

    def test_positional_and_keyword_arguments(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FixedContext(True, -16, 32, overflow=fp.OverflowMode.SATURATE):
                return fp.round(x)

        out = f.ast.format()
        assert 'fp.FixedContext(True, -16, 32, overflow=' in out


class TestNamedOperators:
    """An operator written by name keeps it — the name records how the source
    was written, which matters for translated benchmarks — while one written
    as an operator, or synthesized by a rewrite, formats as the operator."""

    def test_power_operator(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return x ** y

        assert '(x ** y)' in f.ast.format()

    def test_power_by_name(self):
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return fp.pow(x, y)

        assert 'fp.pow(x, y)' in f.ast.format()

    def test_named_operator_without_surface_syntax(self):
        """`min` has no operator form, so it formats by name either way."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            return min(x, y)

        assert 'min(x, y)' in f.ast.format()


class TestConditionals:
    """An `else` holding a lone conditional is an `elif`."""

    def test_elif(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            if x < 0:
                y = 1.0
            elif x < 1:
                y = 2.0
            else:
                y = 3.0
            return y

        out = f.ast.format()
        assert 'elif x < 1:' in out
        assert 'else:' in out
        # the chain stays flat: only the final `else` is indented as a branch
        assert '        if ' not in out

    def test_elif_without_else(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            y = 0.0
            if x < 0:
                y = 1.0
            elif x < 1:
                y = 2.0
            return y

        out = f.ast.format()
        assert 'elif x < 1:' in out
        assert 'else:' not in out

    def test_else_with_other_statements_stays_nested(self):
        """Only a *lone* conditional collapses; anything else keeps its
        `else` block."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            if x < 0:
                y = 1.0
            else:
                y = 2.0
                if x < 1:
                    y = 3.0
            return y

        out = f.ast.format()
        assert 'else:' in out
        assert 'elif' not in out
