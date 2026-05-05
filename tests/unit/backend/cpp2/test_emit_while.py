"""
Phase 3d tests for the cpp2 emitter — ``while`` loops.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType


def _compile(cc: Cpp2Compiler, func, *, arg_ctx=None) -> str:
    arg_ctx = arg_ctx or fp.FP64
    arg_types = [RealType(arg_ctx) for _ in func.args]
    return cc.compile(func, ctx=arg_ctx, arg_types=arg_types)


class TestWhileStmt:
    """Phase 3d — ``while`` loops with phi-style accumulators."""

    def test_simple_countdown(self):
        """The pre-loop assign declares ``y``; the body reassigns
        across iterations.  The loop's phi is not is_intro because
        ``y`` already existed when the loop started."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while y > 0:
                    y = y - 1
                return y

        out = _compile(Cpp2Compiler(), f)
        assert out == (
            'double f(double x) {\n'
            '    double y = x;\n'
            '    while ((y > 0)) {\n'
            '        y = (y - 1);\n'
            '    }\n'
            '    return y;\n'
            '}\n'
        )

    def test_two_accumulators(self):
        """Multiple loop-carried variables each declare-on-first-assign
        before the loop and reassign in the body."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                i = x
                while i > 0:
                    acc = acc + i
                    i = i - 1
                return acc

        out = _compile(Cpp2Compiler(), f)
        assert 'double acc = 0;' in out
        assert 'double i = x;' in out
        assert 'while ((i > 0)) {' in out
        assert 'acc = (acc + i);' in out
        assert 'i = (i - 1);' in out

    def test_nested_while(self):
        """Nested loops indent and emit independently."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                a = x
                b = y
                while a > 0:
                    while b > 0:
                        b = b - 1
                    a = a - 1
                return a

        out = _compile(Cpp2Compiler(), f)
        # The inner while is properly nested under the outer one.
        assert 'while ((a > 0)) {\n        while ((b > 0)) {' in out
