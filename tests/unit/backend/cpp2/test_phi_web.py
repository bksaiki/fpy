"""
Tests for the per-SSA-def renaming via phi-web equivalence classes.

The cpp2 emitter is free to map distinct SSA defs of the same source
name to distinct C++ variables.  Only defs joined by a phi node share
storage.  These tests pin that behavior.
"""

import fpy2 as fp

from fpy2.backend.cpp2 import Cpp2Compiler
from fpy2.types import RealType


def _compile(func, *, arg_ctx=None) -> str:
    arg_ctx = arg_ctx or fp.FP64
    arg_types = [RealType(arg_ctx) for _ in func.args]
    return Cpp2Compiler().compile(func, ctx=arg_ctx, arg_types=arg_types)


class TestPhiWebRenaming:
    """Per-SSA-def renaming."""

    def test_sequential_rebind_renames(self):
        """A rebind without a phi merge becomes its own C++ variable."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                x = x * 2
                return x

        out = _compile(f)
        # The arg is still ``x``; the rebind class picks ``x_1``.
        assert 'double f(double x)' in out
        assert 'double x_1{};' in out
        assert 'x_1 = (x * 2);' in out
        assert 'return x_1;' in out

    def test_rebind_in_if_merges_with_arg(self):
        """A branch-only rebind has a phi linking it to the arg, so
        the whole class shares the bare arg name — no rename."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                if x < 0:
                    x = -x
                return x

        out = _compile(f)
        # No ``x_1`` declaration — the phi binds arg + rebind together.
        assert 'double x_1{}' not in out
        assert 'x = (-x);' in out
        assert 'return x;' in out

    def test_two_independent_rebinds_each_get_a_class(self):
        """Sequential rebinds without phi merges each form their own
        class.  The third reads the second, the second reads the first."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                x = x + 1
                x = x + 1
                return x

        out = _compile(f)
        # Two non-arg classes for ``x``.
        assert 'double x_1{};' in out
        assert 'double x_2{};' in out
        assert 'x_1 = (x + 1);' in out
        assert 'x_2 = (x_1 + 1);' in out
        assert 'return x_2;' in out

    def test_loop_carried_var_keeps_one_class(self):
        """Loop phis pull the pre-loop init, the carry, and the
        body-rebind all into one class — single C++ variable."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                acc = 0
                for i in range(3):
                    acc = acc + x
                return acc

        out = _compile(f)
        # ``acc`` should appear exactly once as a hoisted decl.
        assert out.count('acc{};') == 1
        # No suffixed acc anywhere.
        assert 'acc_1' not in out
        assert 'acc_2' not in out
