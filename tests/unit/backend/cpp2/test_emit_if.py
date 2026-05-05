"""
Phase 3c tests for the cpp2 emitter — ``if`` / ``if1`` statements.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType


def _compile(cc: Cpp2Compiler, func, *, arg_ctx=None) -> str:
    arg_ctx = arg_ctx or fp.FP64
    arg_types = [RealType(arg_ctx) for _ in func.args]
    return cc.compile(func, ctx=arg_ctx, arg_types=arg_types)


class TestIfStmt:
    """Phase 3c — ``if`` / ``else`` and the ``if1`` (no-else) form."""

    def test_if_else_assigns_into_phi(self):
        """``y`` is hoisted once at the top; both branches reassign it."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                if x < 0:
                    y = -x
                else:
                    y = x
                return y

        out = _compile(Cpp2Compiler(), f)
        assert out == (
            'double f(double x) {\n'
            '    double y{};\n'
            '    if ((x < 0)) {\n'
            '        y = (-x);\n'
            '    } else {\n'
            '        y = x;\n'
            '    }\n'
            '    return y;\n'
            '}\n'
        )

    def test_if1_no_else(self):
        """``if`` without an ``else`` emits a single guarded block."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                if x < 0:
                    y = -x
                return y

        out = _compile(Cpp2Compiler(), f)
        assert out == (
            'double f(double x) {\n'
            '    double y{};\n'
            '    y = x;\n'
            '    if ((x < 0)) {\n'
            '        y = (-x);\n'
            '    }\n'
            '    return y;\n'
            '}\n'
        )

    def test_nested_if(self):
        """Nesting indents correctly and reuses the same hoisted variable."""

        @fp.fpy
        def f(x: fp.Real, y: fp.Real) -> fp.Real:
            with fp.FP64:
                if x < y:
                    if x < 0:
                        z = -x
                    else:
                        z = x
                else:
                    z = y
                return z

        out = _compile(Cpp2Compiler(), f)
        assert out == (
            'double f(double x, double y) {\n'
            '    double z{};\n'
            '    if ((x < y)) {\n'
            '        if ((x < 0)) {\n'
            '            z = (-x);\n'
            '        } else {\n'
            '            z = x;\n'
            '        }\n'
            '    } else {\n'
            '        z = y;\n'
            '    }\n'
            '    return z;\n'
            '}\n'
        )

    def test_branch_local_does_not_leak(self):
        """A name assigned only inside one branch is still hoisted to the
        function top — the visitor walks every def regardless of position."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                if x < 0:
                    t = -x
                    y = t + 1
                else:
                    y = x
                return y

        out = _compile(Cpp2Compiler(), f)
        # Both ``t`` and ``y`` are declared at the top.
        assert 'double t{};' in out
        assert 'double y{};' in out
        # And the ``t`` reassignment lives inside the ``if`` body.
        assert '        t = (-x);' in out
        assert '        y = (t + 1);' in out
