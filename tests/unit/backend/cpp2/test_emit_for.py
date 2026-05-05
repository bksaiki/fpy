"""
Phase 3e tests for the cpp2 emitter — ``for`` loops over ``range(...)``.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType


def _compile(cc: Cpp2Compiler, func, *, arg_ctx=None) -> str:
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

        out = _compile(Cpp2Compiler(), f)
        # The loop variable is hoisted at the function top alongside acc.
        assert 'double acc{};' in out
        assert 'int64_t i{};' in out
        assert 'for (i = 0; i < 10; ++i) {' in out
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

        out = _compile(Cpp2Compiler(), f)
        assert 'for (i = 2; i < 8; ++i) {' in out

    def test_for_range3(self):
        """``range(start, stop, step)`` increments by ``step``."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(0, 12, 3):
                    acc = acc + x
                return acc

        out = _compile(Cpp2Compiler(), f)
        assert 'for (i = 0; i < 12; i += 3) {' in out

    def test_two_loops_share_independent_counters(self):
        """Two for-loops over different counters get independent
        hoisted declarations."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(2, 8):
                    acc = acc + x
                for j in range(0, 12, 3):
                    acc = acc - x
                return acc

        out = _compile(Cpp2Compiler(), f)
        assert 'int64_t i{};' in out
        assert 'int64_t j{};' in out

    def test_nested_for(self):
        """A loop nested inside another emits indented bodies and
        hoists both counters at the function top."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(3):
                    for j in range(2):
                        acc = acc + x
                return acc

        out = _compile(Cpp2Compiler(), f)
        assert 'int64_t i{};' in out
        assert 'int64_t j{};' in out
        # Inner loop sits inside the outer loop's body.
        assert 'for (i = 0; i < 3; ++i) {\n        for (j = 0; j < 2; ++j) {' in out

    def test_non_range_iterable_rejects(self):
        """Iterables other than ``range(...)`` fall outside the slice."""

        from fpy2.backend.cpp2 import Cpp2CompileError
        import pytest

        @fp.fpy
        def f() -> fp.Real:
            with fp.FP64:
                acc = 0
                xs = [1, 2, 3]
                for x in xs:
                    acc = acc + x
                return acc

        with pytest.raises(Cpp2CompileError):
            Cpp2Compiler().compile(f, ctx=fp.FP64, arg_types=[])
