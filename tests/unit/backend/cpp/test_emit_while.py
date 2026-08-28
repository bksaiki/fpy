"""
Phase 3d tests for the cpp emitter — ``while`` loops.

Every loop reaches the emitter *rotated*: `Hoistable` runs in
`CppCompiler.specialize()` and evaluates a non-atomic condition through a name,
once before the loop and once at the end of the body.  So the shape pinned here
is `bool c = cond; while (c) { body; c = cond; }` rather than `while (cond)`.
That is FPy's own evaluation order made explicit, and it costs nothing: the two
forms compile to identical machine code at ``-O2``, since loop rotation is a
transformation the C++ compiler performs itself.
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import RealType


def _compile(cc: CppCompiler, func, *, arg_ctx=None) -> str:
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

        out = _compile(CppCompiler(), f)
        assert out == (
            'double f(double x) {\n'
            '    double y = x;\n'
            '    bool c = (y > static_cast<double>(0));\n'
            '    while (c) {\n'
            '        y = (y - static_cast<double>(1));\n'
            '        c = (y > static_cast<double>(0));\n'
            '    }\n'
            '    return y;\n'
            '}'
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

        out = _compile(CppCompiler(), f)
        assert 'double acc = 0;' in out
        assert 'double i = x;' in out
        assert 'bool c = (i > static_cast<double>(0));' in out
        assert 'while (c) {' in out
        assert 'acc = (acc + i);' in out
        assert 'i = (i - static_cast<double>(1));' in out

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

        out = _compile(CppCompiler(), f)
        # The inner while is properly nested under the outer one, and each loop
        # rotates through a name of its own.
        assert (
            'while (c) {\n'
            '        bool c5 = (b > static_cast<double>(0));\n'
            '        while (c5) {'
        ) in out
