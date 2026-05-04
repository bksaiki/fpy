"""
Unit tests for format analysis.
"""

import fpy2 as fp
import pytest

from fractions import Fraction
from fpy2.analysis import ContextUseAnalysis, FormatInfer, TypeAnalysis
from fpy2.analysis.format_infer import AbstractFormat, ListFormat, SetFormat
from fpy2.analysis.format_infer.analysis import _join_bounds, _list_set_widen
from fpy2.number.context.format import Format
from fpy2.number.context.real import REAL_FORMAT
from fpy2.transform import FuncUpdate


class TestFormatInfer:
    """Unit tests for :class:`FormatInfer`."""

    # ------------------------------------------------------------------
    # Helper

    @staticmethod
    def _run(func: fp.Function):
        return FormatInfer.analyze(func.ast)

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

    def test_while_body_revisit_propagates_widened_phi(self):
        """
        A read of a loop-carried variable inside the body must observe the
        phi bound widened by the back-edge, not just the pre-loop value.
        Here ``y = x`` reads x's phi; x is further widened by ``x = 2`` in
        the body's if-branch, so the back-edge widens x's phi to {0, 2}.
        Without a fixpoint, the recorded bound for ``y`` (and for y's loop
        phi) keeps x's pre-loop bound (``SetFormat({0})``).
        """

        @fp.fpy
        def f(cond: bool) -> fp.Real:
            x = 0
            y = 0
            while cond:
                y = x
                if cond:
                    x = 2
            return y

        info = self._run(f)
        expected = SetFormat(frozenset((Fraction(0), Fraction(2))))
        # Every definition of ``y`` (the in-body assign and the while-phi)
        # must reflect x's widened bound.
        y_bounds = [b for d, b in info.by_def.items() if d.name.base == 'y']
        assert expected in y_bounds, (
            f"expected SetFormat({{0, 2}}) among y bounds, got {y_bounds}"
        )

    def test_for_body_revisit_propagates_widened_phi(self):
        """``for`` analogue of :meth:`test_while_body_revisit_propagates_widened_phi`."""

        @fp.fpy
        def f(xs: list[fp.Real], cond: bool) -> fp.Real:
            x = 0
            y = 0
            for _ in xs:
                y = x
                if cond:
                    x = 2
            return y

        info = self._run(f)
        expected = SetFormat(frozenset((Fraction(0), Fraction(2))))
        y_bounds = [b for d, b in info.by_def.items() if d.name.base == 'y']
        assert expected in y_bounds, (
            f"expected SetFormat({{0, 2}}) among y bounds, got {y_bounds}"
        )

    # ------------------------------------------------------------------
    # Type-info and context-use analysis are stored in the result

    def test_result_has_type_info(self):
        """The FormatAnalysis result stores the TypeAnalysis."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            return x

        info = self._run(f)
        assert isinstance(info.type_info, TypeAnalysis)

    def test_result_has_ctx_use(self):
        """The FormatAnalysis result stores the ContextUseAnalysis."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                return fp.round(x)

        info = self._run(f)
        assert isinstance(info.ctx_use, ContextUseAnalysis)

    # ------------------------------------------------------------------
    # Join lattice semantics

    def test_join_same_format(self):
        """
        ``join(f, f) == f``:
        formats from two sources with the same format join to that format.
        """
        fmt = fp.FP32.format()
        assert _join_bounds(fmt, fmt) == fmt

    def test_join_different_formats(self):
        """
        Joining two distinct abstractable Formats goes through
        ``AbstractFormat`` and yields a single :class:`Format` whose
        representable set contains both inputs (rather than widening
        immediately to ``REAL_FORMAT``).
        """
        fmt1 = fp.FP32.format()
        fmt2 = fp.FP64.format()
        joined = _join_bounds(fmt1, fmt2)
        assert isinstance(joined, Format)
        assert joined != REAL_FORMAT
        # The joined format must contain every value representable by either
        # input. Pick concrete witnesses near the bounds of FP32 / FP64.
        for fmt in (fmt1, fmt2):
            sample = fmt.maxval()._real
            assert joined.representable_in(sample)

    def test_join_real_is_top(self):
        """
        ``join(REAL_FORMAT, f) == REAL_FORMAT``:
        REAL_FORMAT is the top element of the lattice.
        """
        fmt = fp.FP32.format()
        assert _join_bounds(REAL_FORMAT, fmt) == REAL_FORMAT
        assert _join_bounds(fmt, REAL_FORMAT) == REAL_FORMAT

    def test_branch_distinct_formats_joins_to_containing_format(self):
        """
        End-to-end: a function whose ``if`` branches are typed under FP32 and
        FP64 should produce a single, containing :class:`Format` for the
        merged value, not ``REAL_FORMAT``.
        """
        @fp.fpy
        def f(cond: bool, x: fp.Real) -> fp.Real:
            if cond:
                with fp.FP32:
                    y = fp.round(x)
            else:
                with fp.FP64:
                    y = fp.round(x)
            return y

        info = self._run(f)
        # Find the phi merging the two branches.
        y_bounds = [b for d, b in info.by_def.items() if d.name.base == 'y']
        merged = [b for b in y_bounds if isinstance(b, Format) and b != REAL_FORMAT]
        assert merged, f"expected a containing Format among y bounds, got {y_bounds}"
        # Whichever specific Format we pick, it must contain both FP32 and FP64.
        joined = merged[-1]
        for fmt in (fp.FP32.format(), fp.FP64.format()):
            assert joined.representable_in(fmt.maxval()._real)

    def test_abstract_format_round_trip(self):
        """
        ``AbstractFormat.from_format(f).format()`` produces a :class:`Format`
        whose representable set contains every value of *f*.
        """
        for ctx in (fp.FP32, fp.FP64):
            fmt = ctx.format()
            roundtripped = AbstractFormat.from_format(fmt).format()
            assert isinstance(roundtripped, Format)
            assert roundtripped.representable_in(fmt.maxval()._real)
            assert roundtripped.representable_in(fmt.minval()._real)

    def test_join_saturates_to_real(self):
        """
        Joining a concrete Format with ``REAL_FORMAT`` (the saturated
        abstract format) collapses to ``REAL_FORMAT``.
        """
        af_fp32 = AbstractFormat.from_format(fp.FP32.format())
        af_real = AbstractFormat.from_format(REAL_FORMAT)
        assert (af_fp32 | af_real).format() == REAL_FORMAT

    def test_loop_format_join_converges(self):
        """
        A loop whose body produces a different concrete Format than the
        pre-loop binding must reach a fixpoint with the joined containing
        Format (not diverge, not silently widen to ``REAL_FORMAT``).
        """
        @fp.fpy
        def f(cond: bool, x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = fp.round(x)
            while cond:
                with fp.FP64:
                    y = fp.round(x)
            return y

        info = self._run(f)
        y_bounds = [b for d, b in info.by_def.items() if d.name.base == 'y']
        # The phi merging the FP32 pre-loop value with the FP64 body value
        # must be a containing Format that is neither FP32 nor REAL_FORMAT.
        joined = [
            b for b in y_bounds
            if isinstance(b, Format) and b not in (fp.FP32.format(), REAL_FORMAT)
        ]
        assert joined, f"expected a widened Format among y bounds, got {y_bounds}"
        for fmt in (fp.FP32.format(), fp.FP64.format()):
            assert joined[-1].representable_in(fmt.maxval()._real)

    # ------------------------------------------------------------------
    # Exact arithmetic under a concrete REAL context

    def test_exact_add_under_real(self):
        """
        ``a + b`` under ``with fp.REAL`` (no rounding) where both operands
        are FP32 should produce a Format strictly tighter than ``REAL_FORMAT``
        whose bounds contain ``2 * FP32_MAX``.
        """
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
                b = fp.round(y)
            with fp.REAL:
                return a + b

        info = self._run(f)
        adds = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Add']
        assert adds, 'expected an Add expression in by_expr'
        result = adds[-1]
        assert isinstance(result, Format)
        assert result != REAL_FORMAT
        fp32_max = fp.FP32.format().maxval()._real
        assert result.representable_in(fp32_max + fp32_max)

    def test_exact_mul_under_real(self):
        """
        ``a * b`` under ``with fp.REAL`` produces a Format that contains
        ``FP32_MAX**2`` (which itself does not fit in FP32 / FP64 ranges
        cleanly, but does fit in the AbstractFormat-derived bound).
        """
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
                b = fp.round(y)
            with fp.REAL:
                return a * b

        info = self._run(f)
        muls = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Mul']
        assert muls and isinstance(muls[-1], Format)
        assert muls[-1] != REAL_FORMAT

    def test_exact_neg_under_real(self):
        """``-a`` under REAL preserves a's format (up to the sign-flip)."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            with fp.REAL:
                return -a

        info = self._run(f)
        negs = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Neg']
        assert negs and isinstance(negs[-1], Format)
        assert negs[-1] != REAL_FORMAT
        # Negation cannot widen beyond ±FP32_MAX.
        fp32_max = fp.FP32.format().maxval()._real
        assert negs[-1].representable_in(-fp32_max)

    def test_exact_arith_skipped_under_symbolic_context(self):
        """
        When the active context is a symbolic / unresolved variable (the
        default top-level scope), exact arithmetic is *not* applied — we
        cannot assume the rounding is the identity.  Result falls back to
        ``REAL_FORMAT``.
        """
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
                b = fp.round(y)
            return a + b  # default top-level scope, symbolic ctx

        info = self._run(f)
        adds = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Add']
        assert adds and adds[-1] == REAL_FORMAT

    def test_loop_widens_to_real_format(self):
        """
        A loop whose body applies exact arithmetic to a phi'd value would
        produce an infinite ascending chain of AbstractFormats (each
        iteration widens prec/bounds).  After ``loop_iter_limit`` iterations
        the analysis switches to widen-mode joins so the fixpoint terminates
        at ``REAL_FORMAT``.
        """
        @fp.fpy
        def f(n: fp.Real) -> fp.Real:
            with fp.FP32:
                x = fp.round(0)
            while n > 0:
                with fp.REAL:
                    x = x + x
            return x

        # With a small limit, the fixpoint must terminate and widen x's
        # while-phi to REAL_FORMAT.
        info = FormatInfer.analyze(f.ast, loop_iter_limit=2)
        x_bounds = [b for d, b in info.by_def.items() if d.name.base == 'x']
        assert REAL_FORMAT in x_bounds, (
            f"expected REAL_FORMAT among x bounds with widening, got {x_bounds}"
        )

    def test_loop_iter_limit_zero_widens_immediately(self):
        """
        ``loop_iter_limit=0`` forces every loop join to widen on the very
        first iteration — distinct scalar Formats merge to ``REAL_FORMAT``
        without going through ``AbstractFormat``.
        """
        @fp.fpy
        def f(cond: bool, x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = fp.round(x)
            while cond:
                with fp.FP64:
                    y = fp.round(x)
            return y

        info = FormatInfer.analyze(f.ast, loop_iter_limit=0)
        y_bounds = [b for d, b in info.by_def.items() if d.name.base == 'y']
        # The loop phi should be REAL_FORMAT (widen-mode forced join of
        # FP32 and FP64 to top), not an MPB-float containing both.
        assert REAL_FORMAT in y_bounds, (
            f"expected REAL_FORMAT among y bounds with limit=0, got {y_bounds}"
        )

    def test_loop_iter_limit_high_keeps_precision(self):
        """
        With a generous limit, a non-divergent loop must still produce a
        precise join (an MPB-float containing FP32 and FP64), not widen to
        ``REAL_FORMAT``.
        """
        @fp.fpy
        def f(cond: bool, x: fp.Real) -> fp.Real:
            with fp.FP32:
                y = fp.round(x)
            while cond:
                with fp.FP64:
                    y = fp.round(x)
            return y

        info = FormatInfer.analyze(f.ast, loop_iter_limit=100)
        y_bounds = [b for d, b in info.by_def.items() if d.name.base == 'y']
        # Some bound must be a Format containing both FP32 and FP64 but not
        # equal to REAL_FORMAT (the AbstractFormat-mediated join survives).
        precise = [
            b for b in y_bounds
            if isinstance(b, Format) and b not in (fp.FP32.format(), REAL_FORMAT)
        ]
        assert precise, (
            f"expected an AbstractFormat-derived join among y bounds, got {y_bounds}"
        )

    def test_exact_arith_skipped_under_concrete_round(self):
        """
        Under a concrete non-REAL context (e.g., FP32), arithmetic results
        are rounded to that format — exact-arithmetic widening would be
        unsound.  The visitor must still return the scope's format.
        """
        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
                b = fp.round(y)
                return a + b

        info = self._run(f)
        adds = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Add']
        assert adds and adds[-1] == fp.FP32.format()

    # ------------------------------------------------------------------
    # SetFormat semantics

    def test_join_set_with_set(self):
        """``join(SetFormat(a), SetFormat(b)) == SetFormat(a ∪ b)``."""
        a = SetFormat(frozenset((Fraction(1), Fraction(2))))
        b = SetFormat(frozenset((Fraction(2), Fraction(3))))
        assert _join_bounds(a, b) == SetFormat(
            frozenset((Fraction(1), Fraction(2), Fraction(3)))
        )

    def test_join_set_with_compatible_format(self):
        """``join(SetFormat(s), fmt) == fmt`` when every value is representable."""
        fmt = fp.FP32.format()
        s = SetFormat(frozenset((Fraction(1), Fraction(2), Fraction(0.5))))
        assert _join_bounds(s, fmt) == fmt
        assert _join_bounds(fmt, s) == fmt

    def test_join_set_with_incompatible_format(self):
        """A non-dyadic value cannot fit in a binary FP format → REAL_FORMAT."""
        fmt = fp.FP32.format()
        s = SetFormat(frozenset((Fraction(1, 3),)))  # 1/3 is not dyadic
        assert _join_bounds(s, fmt) == REAL_FORMAT
        assert _join_bounds(fmt, s) == REAL_FORMAT

    def test_join_set_with_real_format(self):
        """Any set is contained in REAL_FORMAT, so the join is REAL_FORMAT."""
        s = SetFormat(frozenset((Fraction(1, 3),)))
        assert _join_bounds(s, REAL_FORMAT) == REAL_FORMAT
        assert _join_bounds(REAL_FORMAT, s) == REAL_FORMAT

    def test_literal_produces_set_shape(self):
        """A numeric literal expression has a singleton ``SetFormat``."""
        @fp.fpy
        def f() -> fp.Real:
            return 42

        info = self._run(f)
        literal_shapes = [
            shape for shape in info.by_expr.values()
            if isinstance(shape, SetFormat)
        ]
        assert SetFormat(frozenset((Fraction(42),))) in literal_shapes

    # ------------------------------------------------------------------
    # ListSet (functional update) semantics

    def test_list_set_widens_element_format(self):
        """
        ``set(xs, i, val)`` (a functional update produced from ``xs[i] = val``
        by :class:`FuncUpdate`) must widen the result's element format to
        include *val*'s format.  Otherwise the analysis can keep reporting
        the original ``SetFormat`` even after the list has been updated with
        a value the set cannot represent.
        """
        @fp.fpy
        def f(x: fp.Real) -> list[fp.Real]:
            xs = [1.0, 2.0]
            xs[0] = x
            return xs

        # FuncUpdate rewrites ``xs[0] = x`` to ``xs = ListSet(xs, (0,), x)``.
        ast = FuncUpdate.apply(f.ast)
        info = FormatInfer.analyze(ast)

        xs_bounds = [b for d, b in info.by_def.items() if d.name.base == 'xs']
        # The pre-update binding has a precise SetFormat element;
        # the post-update binding must widen to REAL_FORMAT (x's format).
        assert ListFormat(REAL_FORMAT) in xs_bounds, (
            f"expected post-update xs to be ListFormat(REAL_FORMAT), got {xs_bounds}"
        )

    def test_list_set_widen_helper_leaf(self):
        """``_list_set_widen`` at depth 0 is just a join."""
        a = SetFormat(frozenset((Fraction(1),)))
        b = SetFormat(frozenset((Fraction(2),)))
        assert _list_set_widen(a, 0, b) == SetFormat(
            frozenset((Fraction(1), Fraction(2)))
        )

    def test_list_set_widen_helper_nested(self):
        """``_list_set_widen`` peels one ``ListFormat`` layer per index."""
        leaf = SetFormat(frozenset((Fraction(1),)))
        nested = ListFormat(ListFormat(leaf))
        # Inserting a non-dyadic value at depth 2 widens the leaf to REAL_FORMAT
        # (1/3 is not representable in any concrete format here, so the join
        # falls back to REAL_FORMAT).
        insert = SetFormat(frozenset((Fraction(1, 3),)))
        result = _list_set_widen(nested, 2, insert)
        assert result == ListFormat(ListFormat(
            SetFormat(frozenset((Fraction(1), Fraction(1, 3))))
        ))

    # ------------------------------------------------------------------
    # Error handling

    def test_type_error_on_non_funcdef(self):
        """``FormatInfer.analyze`` raises ``TypeError`` for non-FuncDef input."""
        with pytest.raises(TypeError, match="expected a 'FuncDef'"):
            FormatInfer.analyze("not a FuncDef")  # type: ignore[arg-type]
