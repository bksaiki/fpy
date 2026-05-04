"""
Unit tests for format analysis.
"""

import fpy2 as fp
import pytest

from fractions import Fraction
from fpy2.analysis import ContextUseAnalysis, FormatInfer, TypeAnalysis
from fpy2.analysis.format_infer import AbstractFormat, ListFormat, SetFormat, TupleFormat
from fpy2.analysis.format_infer.analysis import _join_bounds, _list_set_widen
from fpy2.analysis.reaching_defs import AssignDef
from fpy2.ast.fpyast import IndexedAssign
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

    def test_join_subsumed_format_returns_wider_input(self):
        """
        When one Format is contained in the other (e.g., FP32 ⊆ FP64), the
        join short-circuits to the wider input Format directly rather than
        projecting through ``(af1 | af2).format()``.  This preserves the
        original Format's identity (e.g., ``IEEEFormat`` not
        ``MPBFloatFormat``) when no widening is needed.
        """
        fp32 = fp.FP32.format()
        fp64 = fp.FP64.format()
        # FP32 ⊆ FP64 → join returns FP64 in either order.
        assert _join_bounds(fp32, fp64) == fp64
        assert _join_bounds(fp64, fp32) == fp64

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

    def test_nested_loop_widen_propagates_inward(self):
        """
        When an outer loop's fixpoint enters widen-mode (its iteration
        count crosses ``loop_iter_limit``), the inner loop's joins also
        widen — the ``saved_widen or iter >= limit`` plumbing in
        ``_iterate_to_fixpoint`` must propagate the outer state through
        nested ``while``/``for``.

        The probe: an outer ``while`` whose body diverges (``x = x + x``
        under REAL) and contains an inner ``while`` whose body merges two
        Formats that would normally produce a precise containing Format.
        With a small outer limit, the outer enters widen-mode quickly;
        because that state propagates inward, the inner loop's phi must
        also widen to ``REAL_FORMAT`` rather than producing the
        AbstractFormat-mediated join.
        """
        @fp.fpy
        def f(n: fp.Real, m: fp.Real, c: bool) -> fp.Real:
            with fp.FP32:
                x = fp.round(0)
                z = fp.round(0)
            while n > 0:
                with fp.REAL:
                    x = x + x  # diverges → forces outer widen
                while m > 0:
                    if c:
                        with fp.FP32:
                            z = fp.round(0)
                    else:
                        with fp.FP64:
                            z = fp.round(0)
            return x + z

        # outer limit small → outer widens, propagates to inner.
        info = FormatInfer.analyze(f.ast, loop_iter_limit=2)
        z_bounds = [b for d, b in info.by_def.items() if d.name.base == 'z']
        # The inner loop's z-phi (which merges FP32 and FP64) must reach
        # REAL_FORMAT once the outer loop's widen state propagates in.
        assert REAL_FORMAT in z_bounds, (
            f"expected outer widen-mode to propagate into inner loop, got {z_bounds}"
        )

    # ------------------------------------------------------------------
    # Bounded-iteration mode for for-loops with statically known length

    def test_for_loop_with_known_count_is_unrolled_precisely(self):
        """
        ``for _ in range(0, N)`` where ``N`` is a static int: the analysis
        drives the phi update for exactly ``N`` body executions instead of
        iterating to a fixpoint.  Under a body that uses exact arithmetic
        (``z = z + a`` under ``with fp.REAL``) this avoids the
        widen-to-REAL_FORMAT fall-back and produces a precise containing
        Format — the AbstractFormat-mediated bound after exactly N
        iterations.
        """
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            z = a
            for _ in range(0, 3):
                with fp.REAL:
                    z = z + a
            return z

        info = self._run(f)
        z_bounds = [b for d, b in info.by_def.items() if d.name.base == 'z']
        # The loop phi must be a precise Format containing FP32, *not*
        # REAL_FORMAT (which would happen if widening kicked in).
        precise = [
            b for b in z_bounds
            if isinstance(b, Format) and b != REAL_FORMAT
        ]
        assert precise, (
            f'expected a precise (non-REAL_FORMAT) z bound, got {z_bounds}'
        )
        assert REAL_FORMAT not in z_bounds, (
            f'bounded iteration must not widen to REAL_FORMAT, got {z_bounds}'
        )

    def test_for_loop_with_unknown_count_still_widens(self):
        """
        Counterpart: when the iterable's length is *not* statically
        known (here: ``range(0, n)`` with symbolic ``n``), the analysis
        falls back to fixpoint+widening and the loop phi widens to
        ``REAL_FORMAT`` because exact arithmetic produces an
        infinite-height chain.
        """
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            z = a
            for _ in range(0, n):  # n symbolic → unknown size
                with fp.REAL:
                    z = z + a
            return z

        info = self._run(f)
        z_bounds = [b for d, b in info.by_def.items() if d.name.base == 'z']
        assert REAL_FORMAT in z_bounds, (
            f'expected fixpoint+widen to widen to REAL_FORMAT, got {z_bounds}'
        )

    def test_for_loop_with_zero_count_keeps_pre_loop_bound(self):
        """
        ``for _ in range(0, 0)`` runs 0 iterations.  The body never
        executes, so the loop phi stays at the pre-loop value.
        """
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            z = a
            for _ in range(0, 0):
                with fp.REAL:
                    z = z + a
            return z

        info = self._run(f)
        # All z-bounds must be FP32 (no widening since body didn't run).
        z_bounds = [b for d, b in info.by_def.items() if d.name.base == 'z']
        fp32 = fp.FP32.format()
        assert all(b == fp32 for b in z_bounds), (
            f'expected all z bounds to be FP32 with N=0, got {z_bounds}'
        )

    # ------------------------------------------------------------------
    # Exact arithmetic with SetFormat operands

    def test_exact_add_setformat_with_format(self):
        """
        ``acc + x`` under REAL where ``acc`` starts as ``SetFormat({0})``
        and ``x`` has a concrete Format.  The SetFormat operand must be
        lifted to an :class:`AbstractFormat`, allowing exact-arith to
        produce a precise containing Format instead of falling back to
        ``REAL_FORMAT``.
        """
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            with fp.REAL:
                acc = 0
                acc = acc + a
            return acc

        info = self._run(f)
        # The Add expression's format should be a precise Format
        # (containing FP32), not REAL_FORMAT.
        adds = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Add']
        assert adds, 'expected an Add expression'
        result = adds[-1]
        assert isinstance(result, Format)
        assert result != REAL_FORMAT, (
            f'expected SetFormat({{0}}) + FP32 to lift through AbstractFormat, '
            f'got {result}'
        )

    def test_exact_add_two_setformats_stays_precise(self):
        """
        ``SetFormat({a}) + SetFormat({b})`` under REAL stays a SetFormat
        with the pairwise sum — strictly more precise than going through
        AbstractFormat.
        """
        @fp.fpy
        def f() -> fp.Real:
            with fp.REAL:
                z = 1.0 + 2.0
            return z

        info = self._run(f)
        adds = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Add']
        assert adds, 'expected an Add expression'
        result = adds[-1]
        assert isinstance(result, SetFormat), f'expected SetFormat, got {result}'
        assert result.values == frozenset((Fraction(3),)), (
            f'expected SetFormat({{3}}), got {result}'
        )

    def test_exact_sub_setformat_with_format(self):
        """``a - SetFormat({0})`` under REAL preserves a's format."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            with fp.REAL:
                z = a - 0
            return z

        info = self._run(f)
        subs = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Sub']
        assert subs, 'expected a Sub expression'
        assert isinstance(subs[-1], Format) and subs[-1] != REAL_FORMAT

    def test_exact_mul_setformat_with_format(self):
        """``a * SetFormat({2})`` under REAL produces a containing Format."""
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP32:
                a = fp.round(x)
            with fp.REAL:
                z = a * 2.0
            return z

        info = self._run(f)
        muls = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Mul']
        assert muls, 'expected a Mul expression'
        assert isinstance(muls[-1], Format) and muls[-1] != REAL_FORMAT

    def test_exact_arith_setformat_accumulator_pattern(self):
        """
        End-to-end: the user's block-accumulator pattern.  ``block_acc =
        0; for j in range(K): block_acc += block[j]`` under REAL must
        produce a precise containing Format for ``block_acc``, not
        ``REAL_FORMAT`` (which would happen if ``SetFormat({0}) +
        EFloatFormat`` bailed to the scope's format).
        """
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.MX_E4M3:
                ys = [fp.round(x) for x in xs]
            block_acc = 0
            for j in range(0, 4):
                block_acc = block_acc + ys[j]
            return block_acc

        info = self._run(f)
        block_acc_bounds = [
            b for d, b in info.by_def.items() if d.name.base == 'block_acc'
        ]
        # The loop phi must be a precise (non-REAL_FORMAT) format
        # containing MX_E4M3 — the SetFormat({0}) initial value lifted
        # through AbstractFormat into the inner-loop accumulation.
        assert REAL_FORMAT not in block_acc_bounds, (
            f'expected block_acc to stay precise, got {block_acc_bounds}'
        )

    # ------------------------------------------------------------------
    # Sum reduction

    def test_sum_under_real_with_known_size(self):
        """
        ``sum(ys)`` under ``with fp.REAL`` over a known-size list of
        MX_E4M3 elements: simulate ``n - 1`` exact pairwise additions
        through :class:`AbstractFormat`.  Result is a precise containing
        Format, not ``REAL_FORMAT``.
        """
        @fp.fpy(ctx=fp.REAL)
        def f() -> fp.Real:
            with fp.MX_E4M3:
                ys = [fp.round(0.1), fp.round(0.2),
                      fp.round(0.3), fp.round(0.4)]
            return sum(ys)

        info = self._run(f)
        sums = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Sum']
        assert sums and isinstance(sums[-1], Format) and sums[-1] != REAL_FORMAT, (
            f'expected precise Sum bound under REAL, got {sums}'
        )
        # Bound must contain a single MX_E4M3 element.
        mx_e4m3 = fp.MX_E4M3.format()
        assert sums[-1].representable_in(mx_e4m3.maxval()._real)

    def test_sum_under_concrete_ctx_returns_scope_format(self):
        """
        Under a concrete non-REAL context, every pairwise add rounds to
        the scope's format, so ``sum(ys)`` reports the scope's format
        (the default ``_op_bound`` rule).
        """
        @fp.fpy
        def f() -> fp.Real:
            with fp.MX_E4M3:
                ys = [fp.round(0.1), fp.round(0.2),
                      fp.round(0.3), fp.round(0.4)]
            with fp.FP32:
                return sum(ys)

        info = self._run(f)
        sums = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Sum']
        assert sums and sums[-1] == fp.FP32.format(), (
            f'expected Sum to report FP32 (scope format), got {sums}'
        )

    def test_sum_unknown_size_falls_back(self):
        """
        ``sum(xs)`` over a symbolic-size list under REAL: the iteration
        count isn't known, so the analysis falls back to the default
        rule which yields ``REAL_FORMAT`` under REAL scope.
        """
        @fp.fpy(ctx=fp.REAL)
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.MX_E4M3:
                ys = [fp.round(x) for x in xs]
            return sum(ys)

        info = self._run(f)
        sums = [b for e, b in info.by_expr.items() if type(e).__name__ == 'Sum']
        assert sums and sums[-1] == REAL_FORMAT, (
            f'expected REAL_FORMAT fall-back when size is unknown, got {sums}'
        )

    def test_sum_grows_with_known_size(self):
        """
        Sanity: larger N produces a wider precise format under REAL.
        """
        @fp.fpy(ctx=fp.REAL)
        def small() -> fp.Real:
            with fp.FP32:
                ys = [fp.round(1.0), fp.round(1.0)]
            return sum(ys)

        @fp.fpy(ctx=fp.REAL)
        def big() -> fp.Real:
            with fp.FP32:
                ys = [fp.round(1.0), fp.round(1.0), fp.round(1.0),
                      fp.round(1.0), fp.round(1.0), fp.round(1.0),
                      fp.round(1.0), fp.round(1.0)]
            return sum(ys)

        small_sum = [
            b for e, b in self._run(small).by_expr.items()
            if type(e).__name__ == 'Sum'
        ][-1]
        big_sum = [
            b for e, b in self._run(big).by_expr.items()
            if type(e).__name__ == 'Sum'
        ][-1]
        # Both must be precise (non-REAL_FORMAT).
        assert isinstance(small_sum, Format) and small_sum != REAL_FORMAT
        assert isinstance(big_sum, Format) and big_sum != REAL_FORMAT
        # Size-8 sum's bound must contain size-2 sum's max value
        # (since both bounds are non-negative and big_sum is wider).
        # Concretely: big_sum.representable_in(small_sum.maxval) holds.
        small_max = small_sum.maxval()._real
        assert big_sum.representable_in(small_max)

    # ------------------------------------------------------------------
    # Integer-producing operations report INTEGER format

    def test_len_returns_integer_format_under_concrete_ctx(self):
        """
        ``len(xs)`` always returns an integer; the format must be
        ``INTEGER``, *not* the active rounding context's format.
        """
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP32:
                n = len(xs)
            return n

        info = self._run(f)
        # The Len expression must be the INTEGER format, not FP32.
        len_bounds = [
            b for e, b in info.by_expr.items() if type(e).__name__ == 'Len'
        ]
        integer_fmt = fp.INTEGER.format()
        assert len_bounds and len_bounds[0] == integer_fmt, (
            f'expected Len to report INTEGER format, got {len_bounds}'
        )
        # And the binding for ``n`` must inherit that.
        n_bounds = [b for d, b in info.by_def.items() if d.name.base == 'n']
        assert integer_fmt in n_bounds, (
            f'expected INTEGER among n bounds, got {n_bounds}'
        )

    def test_range1_returns_listformat_of_integer(self):
        """``range(n)`` produces a list of integers."""
        @fp.fpy
        def f(xs: list[fp.Real]) -> list[fp.Real]:
            ys = [0.0 for _ in range(len(xs))]
            return ys

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items() if type(e).__name__ == 'Range1'
        ]
        integer_fmt = fp.INTEGER.format()
        assert range_bounds, 'expected a Range1 expression'
        assert range_bounds[0] == ListFormat(integer_fmt), (
            f'expected ListFormat(INTEGER) for Range1, got {range_bounds}'
        )

    def test_range2_returns_listformat_of_integer(self):
        """``range(start, stop)`` produces a list of integers."""
        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(0, 10)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items() if type(e).__name__ == 'Range2'
        ]
        integer_fmt = fp.INTEGER.format()
        assert range_bounds and range_bounds[0] == ListFormat(integer_fmt)

    def test_range3_returns_listformat_of_integer(self):
        """``range(start, stop, step)`` produces a list of integers."""
        @fp.fpy
        def f() -> list[fp.Real]:
            return [0.0 for _ in range(0, 10, 2)]

        info = self._run(f)
        range_bounds = [
            b for e, b in info.by_expr.items() if type(e).__name__ == 'Range3'
        ]
        integer_fmt = fp.INTEGER.format()
        assert range_bounds and range_bounds[0] == ListFormat(integer_fmt)

    def test_enumerate_returns_listformat_of_int_value_tuple(self):
        """
        ``enumerate(xs)`` produces a list of (int, x) tuples — the int
        component is INTEGER, the value component preserves ``xs``'s
        element format.
        """
        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP32:
                ys = [fp.round(x) for x in xs]
            for i, y in enumerate(ys):
                pass
            return 0.0

        info = self._run(f)
        enum_bounds = [
            b for e, b in info.by_expr.items() if type(e).__name__ == 'Enumerate'
        ]
        assert enum_bounds, 'expected an Enumerate expression'
        bound = enum_bounds[0]
        # Outer: ListFormat. Element: TupleFormat((INTEGER, FP32)).
        integer_fmt = fp.INTEGER.format()
        fp32_fmt = fp.FP32.format()
        assert isinstance(bound, ListFormat)
        assert isinstance(bound.elt, TupleFormat)
        assert bound.elt.elts == (integer_fmt, fp32_fmt), (
            f'expected TupleFormat((INTEGER, FP32)), got {bound.elt}'
        )

    def test_blockwise_quantized_accumulator_end_to_end(self):
        """
        Regression: a realistic blockwise quantized-accumulator pattern.

        - ``ys`` is quantized to MX_E4M3.
        - Outer loop iterates blocks of K=32; each iteration takes a
          slice ``block = ys[i:i+K]`` and accumulates K elements under
          REAL into ``block_acc``.
        - The block accumulator is then added into ``acc`` under FP32.

        This exercises every load-bearing piece of the analysis at once:

        - ``ListSize`` resolution for ``range(K)`` (= 32) so the inner
          loop is unrolled exactly 32 times instead of fixpoint+widening.
        - ``SetFormat({0}) + EFloatFormat`` lifted through
          :class:`AbstractFormat` so the inner accumulator stays precise.
        - The FP32 ``with`` block forcing a clean rounded format on
          ``acc`` regardless of the symbolic outer-loop count.

        Before the SetFormat → AbstractFormat lift and the bounded-iter
        mode, ``block_acc`` widened to ``REAL_FORMAT`` on the first
        body pass and contaminated everything downstream.
        """
        @fp.fpy(ctx=fp.REAL)
        def foo(xs: list[fp.Real]):
            # block size
            K = 32

            # quantize to E4M3
            with fp.MX_E4M3:
                ys = [fp.round(x) for x in xs]
            # accumulate in blocks of K
            acc = 0
            for i in range(0, len(ys), K):
                block = ys[i : i + K]
                block_acc = 0
                for j in range(K):
                    block_acc += block[j]
                # quantize block accumulation to FP32
                with fp.FP32:
                    acc += block_acc
            # return final result in FP32
            return acc

        info = self._run(foo)

        # 1. block_acc must stay precise (no REAL_FORMAT contamination
        #    from the inner accumulator).
        block_acc_bounds = [
            b for d, b in info.by_def.items() if d.name.base == 'block_acc'
        ]
        assert REAL_FORMAT not in block_acc_bounds, (
            f'expected block_acc to stay precise, got {block_acc_bounds}'
        )
        # And it must reach a Format strictly tighter than REAL_FORMAT
        # at the loop phi (the bounded-iter mode produced an
        # MPB-Float-shaped accumulator that contains 32 × MX_E4M3_max).
        widened = [
            b for b in block_acc_bounds
            if isinstance(b, Format) and b != REAL_FORMAT
        ]
        assert widened, (
            f'expected a precise containing Format among block_acc bounds, '
            f'got {block_acc_bounds}'
        )

        # 2. acc must reach FP32 — every write is inside ``with fp.FP32``,
        #    so the FP32 round dominates regardless of the outer loop's
        #    symbolic iteration count.
        acc_bounds = [
            b for d, b in info.by_def.items() if d.name.base == 'acc'
        ]
        assert fp.FP32.format() in acc_bounds, (
            f'expected FP32 among acc bounds, got {acc_bounds}'
        )

        # 3. ys is the rounded list — its element format is MX_E4M3.
        ys_bounds = [
            b for d, b in info.by_def.items() if d.name.base == 'ys'
        ]
        mx_e4m3 = fp.MX_E4M3.format()
        assert any(
            isinstance(b, ListFormat) and b.elt == mx_e4m3
            for b in ys_bounds
        ), f'expected ListFormat(MX_E4M3) among ys bounds, got {ys_bounds}'

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

    def test_indexed_assign_widens_format_at_fresh_def(self):
        """
        ``xs[i] = x`` (raw ``IndexedAssign``, no ``FuncUpdate`` applied)
        creates a *fresh* SSA def of ``xs`` (per ``reaching_defs``'s
        treatment of indexed assignment as ``xs = update(xs, [i], x)``).
        The new def's format must be widened to match the inserted value's
        format — i.e., the same semantics as the post-FuncUpdate
        ``ListSet`` path tested above.
        """
        @fp.fpy
        def f(x: fp.Real) -> list[fp.Real]:
            xs = [1.0, 2.0]
            xs[0] = x
            return xs

        # NOTE: no FuncUpdate applied — IndexedAssign survives in the AST.
        info = FormatInfer.analyze(f.ast)

        # Two distinct defs for xs: the original Assign-sited binding and
        # the IndexedAssign-sited fresh def.
        xs_defs = [d for d in info.by_def if d.name.base == 'xs']
        assert len(xs_defs) == 2, f"expected 2 xs defs, got {xs_defs}"

        # Identify each by site kind.
        assign_defs = [
            d for d in xs_defs
            if isinstance(d, AssignDef) and not isinstance(d.site, IndexedAssign)
        ]
        idx_defs = [
            d for d in xs_defs
            if isinstance(d, AssignDef) and isinstance(d.site, IndexedAssign)
        ]
        assert len(assign_defs) == 1, f"expected 1 Assign-sited xs def, got {assign_defs}"
        assert len(idx_defs) == 1, f"expected 1 IndexedAssign-sited xs def, got {idx_defs}"

        # Pre-mutation: ListFormat over the literal element SetFormat.
        pre_bound = info.by_def[assign_defs[0]]
        assert isinstance(pre_bound, ListFormat)
        assert isinstance(pre_bound.elt, SetFormat)

        # Post-mutation (the fresh def at the IndexedAssign): the element
        # format must widen to REAL_FORMAT (the format of x), matching the
        # ListSet-path semantics asserted in
        # ``test_list_set_widens_element_format`` above.
        post_bound = info.by_def[idx_defs[0]]
        assert post_bound == ListFormat(REAL_FORMAT), (
            f"expected fresh IndexedAssign-sited xs def to widen to "
            f"ListFormat(REAL_FORMAT), got {post_bound}"
        )

        # Sanity: the IndexedAssign and post-FuncUpdate ListSet paths agree.
        funcupdate_info = FormatInfer.analyze(FuncUpdate.apply(f.ast))
        funcupdate_xs = [
            b for d, b in funcupdate_info.by_def.items() if d.name.base == 'xs'
        ]
        assert ListFormat(REAL_FORMAT) in funcupdate_xs

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
