"""
Unit tests for format analysis.
"""

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.number.context.real import REAL_FORMAT
from fpy2.transform import ConstFold


class TestFormatInfer:
    """Unit tests for :class:`FormatInfer`."""

    # ------------------------------------------------------------------
    # Helper

    @staticmethod
    def _run(func: fp.Function):
        ast = ConstFold.apply(func.ast, enable_op=True)
        return FormatInfer.analyze(ast)

    # ------------------------------------------------------------------
    # Simple cases – no explicit rounding context

    def test_real_argument(self):
        """Function arguments have REAL_FORMAT (top of lattice)."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x

        info = self._run(f)
        # All definitions should be REAL_FORMAT (no context)
        for fmt in info.by_def.values():
            assert fmt == REAL_FORMAT

    # ------------------------------------------------------------------
    # Single context block

    def test_fp32_context(self):
        """Operations inside an FP32 context produce FP32-format values."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = fp.round(x)
                return y

        info = self._run(f)
        expected = fp.FP32.format()
        # Every definition touched by the FP32 context should have FP32 format
        fmt_set = set(info.by_def.values())
        assert expected in fmt_set, f"expected FP32 format in by_def, got {fmt_set}"

    def test_fp64_context(self):
        """Operations inside an FP64 context produce FP64-format values."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = fp.round(x)
                return y

        info = self._run(f)
        expected = fp.FP64.format()
        fmt_set = set(info.by_def.values())
        assert expected in fmt_set, f"expected FP64 format in by_def, got {fmt_set}"

    # ------------------------------------------------------------------
    # Arithmetic operations inside a context

    def test_add_in_context(self):
        """Addition inside an FP32 context produces FP32-format result."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                return x + y

        info = self._run(f)
        expected = fp.FP32.format()
        # The return expression (x + y) is in by_expr; no assignment means no by_def entry
        fmt_set = set(info.by_expr.values())
        assert expected in fmt_set, f"expected FP32 format in by_expr, got {fmt_set}"

    # ------------------------------------------------------------------
    # Conditional branches

    def test_if_same_format(self):
        """
        When both branches of an ``if`` produce the same format, the merged
        variable should also have that format (join of equal formats).
        """

        @fp.fpy
        def f(x: fp.Real, cond: bool) -> fp.Real:
            with fp.FP32:
                if cond:
                    y = fp.round(x)
                else:
                    y = fp.round(x)
                return y

        info = self._run(f)
        expected = fp.FP32.format()
        fmt_set = set(info.by_def.values())
        assert expected in fmt_set, f"expected FP32 format in by_def, got {fmt_set}"

    # ------------------------------------------------------------------
    # Nested context blocks

    def test_nested_context(self):
        """Values computed in nested contexts carry the inner context's format."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = fp.round(x)
                with fp.FP32:
                    b = fp.round(a)
                return b

        info = self._run(f)
        fp32_fmt = fp.FP32.format()
        fmt_set = set(info.by_def.values())
        assert fp32_fmt in fmt_set, (
            f"expected FP32 format for value from inner context, got {fmt_set}"
        )

    # ------------------------------------------------------------------
    # Return-value format

    def test_return_format_fp32(self):
        """The return expression's format matches the enclosing context."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.round(x)

        info = self._run(f)
        expected = fp.FP32.format()
        # Find the ReturnStmt's expression in by_expr
        ret_fmts = [
            fmt for e, fmt in info.by_expr.items()
            if fmt != REAL_FORMAT
        ]
        assert expected in ret_fmts, (
            f"expected FP32 format among by_expr values, got {set(ret_fmts)}"
        )

    # ------------------------------------------------------------------
    # While loop

    def test_while_loop_same_format(self):
        """
        A loop variable that is always assigned within the same context
        should retain that context's format (join of equal formats = same format).
        """

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                i = fp.round(0)
                n = fp.round(10)
                acc = fp.round(0)
                while i < n:
                    acc = acc + x
                    i = i + fp.round(1)
                return acc

        info = self._run(f)
        fp32_fmt = fp.FP32.format()
        fmt_set = set(info.by_def.values())
        assert fp32_fmt in fmt_set, (
            f"expected FP32 format among loop variable definitions, got {fmt_set}"
        )

    # ------------------------------------------------------------------
    # Join lattice semantics

    def test_join_same_format(self):
        """
        ``join(f, f) == f``:
        formats from two sources with the same format join to that format.
        """
        from fpy2.analysis.format_infer import _join_formats

        fmt = fp.FP32.format()
        assert _join_formats(fmt, fmt) == fmt

    def test_join_different_formats(self):
        """
        ``join(f1, f2) == REAL_FORMAT`` when f1 != f2:
        formats from two sources with different formats widen to REAL_FORMAT.
        """
        from fpy2.analysis.format_infer import _join_formats

        fmt1 = fp.FP32.format()
        fmt2 = fp.FP64.format()
        assert _join_formats(fmt1, fmt2) == REAL_FORMAT

    def test_join_real_is_top(self):
        """
        ``join(REAL_FORMAT, f) == REAL_FORMAT``:
        REAL_FORMAT is the top element of the lattice.
        """
        from fpy2.analysis.format_infer import _join_formats

        fmt = fp.FP32.format()
        assert _join_formats(REAL_FORMAT, fmt) == REAL_FORMAT
        assert _join_formats(fmt, REAL_FORMAT) == REAL_FORMAT

    # ------------------------------------------------------------------
    # Error handling

    def test_type_error_on_non_funcdef(self):
        """``FormatInfer.analyze`` raises ``TypeError`` for non-FuncDef input."""
        import pytest

        with pytest.raises(TypeError):
            FormatInfer.analyze("not a FuncDef")  # type: ignore[arg-type]
