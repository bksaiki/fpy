"""
Phase 3e tests for the cpp emitter — ``for`` loops over ``range(...)``.
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import RealType


def _compile(cc: CppCompiler, func, *, arg_ctx=None) -> str:
    arg_ctx = arg_ctx or fp.FP64
    arg_types = [RealType(arg_ctx) for _ in func.args]
    return cc.compile(func, ctx=arg_ctx, arg_types=arg_types)


class TestForRange:
    """Phase 3e — ``for i in range(...)`` loops."""

    def test_for_range1(self):
        """``range(N)`` becomes a counter loop starting at 0."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(10):
                    acc = acc + x
                return acc

        out = _compile(CppCompiler(), f)
        # ``i`` is single-writer; ``acc``'s loop phi has is_intro=False
        # (acc was assigned before the loop), so the pre-loop assign
        # declares and the body reassigns.
        assert 'double acc = 0;' in out
        assert 'for (int8_t i = 0; i < 10; ++i) {' in out
        assert 'acc = (acc + x);' in out

    def test_for_range2(self):
        """``range(start, stop)`` uses the start as the initialiser."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(2, 8):
                    acc = acc + x
                return acc

        out = _compile(CppCompiler(), f)
        assert 'for (int8_t i = 2; i < 8; ++i) {' in out

    def test_for_range3(self):
        """``range(start, stop, step)`` increments by ``step``."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(0, 12, 3):
                    acc = acc + x
                return acc

        out = _compile(CppCompiler(), f)
        assert 'for (int8_t i = 0; i < 12; i += 3) {' in out

    def test_two_loops_share_independent_counters(self):
        """Two for-loops over different counters each declare-on-assign
        in their own header."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(2, 8):
                    acc = acc + x
                for j in range(0, 12, 3):
                    acc = acc - x
                return acc

        out = _compile(CppCompiler(), f)
        assert 'for (int8_t i = 2; i < 8; ++i) {' in out
        assert 'for (int8_t j = 0; j < 12; j += 3) {' in out

    def test_nested_for(self):
        """A nested loop's counter declares in the inner header; the
        bodies indent normally."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(3):
                    for j in range(2):
                        acc = acc + x
                return acc

        out = _compile(CppCompiler(), f)
        # Inner loop sits inside the outer loop's body, both folded
        # into their own headers.
        assert (
            'for (int8_t i = 0; i < 3; ++i) {\n'
            '        for (int8_t j = 0; j < 2; ++j) {'
        ) in out

    def test_for_over_list(self):
        """A list-typed iterable becomes a range-based ``for`` loop."""

        @fp.fpy
        def f(xs: list[fp.Real]) -> fp.Real:
            with fp.FP64:
                acc = 0
                for x in xs:
                    acc = acc + x
                return acc

        from fpy2.types import ListType
        out = CppCompiler().compile(
            f, ctx=fp.FP64,
            arg_types=[ListType(RealType(fp.FP64))],
        )
        assert 'for (double x : xs) {' in out
        assert 'acc = (acc + x);' in out
