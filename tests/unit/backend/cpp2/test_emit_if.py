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
        """``if`` without an ``else`` emits a single guarded block.
        The pre-if assign declares ``y``; the in-branch assign
        reassigns (the if1 phi has ``is_intro=False``)."""

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
            '    double y = x;\n'
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

    def test_hoist_lands_just_before_if(self):
        """When ``y`` is introduced fresh in both branches of an
        ``if/else`` and there's unrelated work before the ``if``, the
        ``double y{};`` hoist appears immediately before the
        ``if``, not at the function top."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = x + 1
                if a < 0:
                    y = -a
                else:
                    y = a
                return y

        out = _compile(Cpp2Compiler(), f)
        # Hoist is anchored to the if, after the unrelated ``a`` decl.
        assert out == (
            'double f(double x) {\n'
            '    double a = (x + 1);\n'
            '    double y{};\n'
            '    if ((a < 0)) {\n'
            '        y = (-a);\n'
            '    } else {\n'
            '        y = a;\n'
            '    }\n'
            '    return y;\n'
            '}\n'
        )

    def test_nested_if_else_anchors_at_outermost(self):
        """When both an outer and an inner ``if/else`` introduce the
        same name fresh, the hoist is anchored to the *outermost*
        responsible ``if`` so the variable's scope covers every
        branch."""

        @fp.fpy
        def g(c1: bool, c2: bool, x: fp.Real) -> fp.Real:
            with fp.FP64:
                a = x + 1
                if c1:
                    if c2:
                        y = 1
                    else:
                        y = 2
                else:
                    y = 3
                return y + a

        from fpy2.types import BoolType
        out = Cpp2Compiler().compile(
            g, ctx=fp.FP64,
            arg_types=[BoolType(), BoolType(), RealType(fp.FP64)],
        )
        # Single hoist, before the *outer* if — not at function top
        # and not duplicated for the inner if.
        assert out.count('y{};') == 1
        # The hoist sits between ``a``'s decl and the outer if.
        assert (
            '    double a = (x + 1);\n'
            '    uint8_t y{};\n'
            '    if (c1) {'
        ) in out

    def test_branch_local_does_not_leak(self):
        """``y`` is multi-writer (assigned in both branches → phi),
        so it stays hoisted at the function top.  ``t`` is single-
        writer (only the ``if`` branch writes it), so its type folds
        into the assign inside the branch."""

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
        # ``y`` is hoisted because both branches write it.
        assert 'double y{};' in out
        # ``t`` declares-on-assign inside the if-branch.
        assert '        double t = (-x);' in out
        assert '        y = (t + 1);' in out
