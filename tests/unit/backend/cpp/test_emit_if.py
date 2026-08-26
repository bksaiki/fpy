"""
Phase 3c tests for the cpp emitter — ``if`` / ``if1`` statements.

Each ``CppCompiler`` construction passes ``optimize=False`` to
keep the bare-emitter output strings stable against optimizing
transforms (notably :class:`fpy2.transform.RoundElim`).
"""

import fpy2 as fp

from fpy2.backend.cpp import CppCompiler
from fpy2.types import RealType


def _compile(cc: CppCompiler, func, *, arg_ctx=None) -> str:
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

        out = _compile(CppCompiler(optimize=False), f)
        assert out == (
            'double f(double x) {\n'
            '    double y{};\n'
            '    if ((x < static_cast<double>(0))) {\n'
            '        y = (-x);\n'
            '    } else {\n'
            '        y = x;\n'
            '    }\n'
            '    return y;\n'
            '}'
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

        out = _compile(CppCompiler(optimize=False), f)
        assert out == (
            'double f(double x) {\n'
            '    double y = x;\n'
            '    if ((x < static_cast<double>(0))) {\n'
            '        y = (-x);\n'
            '    }\n'
            '    return y;\n'
            '}'
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

        out = _compile(CppCompiler(optimize=False), f)
        assert out == (
            'double f(double x, double y) {\n'
            '    double z{};\n'
            '    if ((x < y)) {\n'
            '        if ((x < static_cast<double>(0))) {\n'
            '            z = (-x);\n'
            '        } else {\n'
            '            z = x;\n'
            '        }\n'
            '    } else {\n'
            '        z = y;\n'
            '    }\n'
            '    return z;\n'
            '}'
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

        out = _compile(CppCompiler(optimize=False), f)
        # Hoist is anchored to the if, after the unrelated ``a`` decl.
        assert out == (
            'double f(double x) {\n'
            '    double a = (x + static_cast<double>(1));\n'
            '    double y{};\n'
            '    if ((a < static_cast<double>(0))) {\n'
            '        y = (-a);\n'
            '    } else {\n'
            '        y = a;\n'
            '    }\n'
            '    return y;\n'
            '}'
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
        out = CppCompiler(optimize=False).compile(
            g, ctx=fp.FP64,
            arg_types=[BoolType(), BoolType(), RealType(fp.FP64)],
        )
        # Single hoist, before the *outer* if — not at function top
        # and not duplicated for the inner if.
        assert out.count('y{};') == 1
        # The hoist sits between ``a``'s decl and the outer if.
        assert (
            '    double a = (x + static_cast<double>(1));\n'
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

        out = _compile(CppCompiler(optimize=False), f)
        # ``y`` is hoisted because both branches write it.
        assert 'double y{};' in out
        # ``t`` declares-on-assign inside the if-branch.
        assert '        double t = (-x);' in out
        assert '        y = (t + static_cast<double>(1));' in out


def _typechecks(src: str) -> tuple[bool, str]:
    """Does *src* compile?  The `else if` flattening moves declarations around,
    so a text assertion is not enough -- the emitted program has to build."""
    import shutil, subprocess, tempfile
    from pathlib import Path
    from fpy2.backend.cpp.utils import CPP_HEADERS
    cxx = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')
    if cxx is None:
        pytest.skip('no C++ compiler')
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'm.cpp'
        cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src)
        r = subprocess.run([cxx, '-std=c++17', '-fsyntax-only', str(cpp)],
                           capture_output=True, text=True)
    return r.returncode == 0, r.stderr


class TestElseIfChain:
    """``else { if ... }`` prints as ``else if``.

    An FPy ``elif`` chain arrives as one nesting level per arm, so without this
    a five-arm chain indents five times and the closing braces pile up.
    """

    def test_chain_is_flattened(self):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                if x > 3.0:
                    y = 1.0
                elif x > 2.0:
                    y = 2.0
                elif x > 1.0:
                    y = 3.0
                else:
                    y = 4.0
                return y

        out = CppCompiler().compile(f, arg_types=[RealType(fp.FP64)])
        assert out.count('else if') == 2
        # one `if`, two `else if`, one `else` — and so one closing brace
        assert out.count('} else {') == 1

    def test_an_arm_declaring_a_variable_still_compiles(self):
        """A class anchored to the nested ``if`` is declared *before* it, and
        the flattened form has no position for that declaration.

        Regression: flattening emitted `} else if (...) { z = 2; ... }` with no
        declaration of `z` anywhere, which does not compile.  A line-order
        assertion cannot catch this -- only building the output can.
        """
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            if x > 0:
                y = 1.0
            elif x > -1:
                z = 2.0
                y = z * 2
            else:
                z = 3.0
                y = z * 3
            return y

        out = CppCompiler().compile(f, arg_types=[RealType(fp.FP64)])
        ok, err = _typechecks(out)
        assert ok, f'emitted program does not compile:\n{err}\n--- emitted ---\n{out}'
        # the shared declaration forces the nesting to stay
        assert 'else if' not in out

    def test_a_flattened_chain_still_compiles(self):
        """The flattening itself must produce buildable output."""
        @fp.fpy(ctx=fp.FP64)
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                if x > 3.0:
                    y = 1.0
                elif x > 2.0:
                    y = 2.0
                elif x > 1.0:
                    y = 3.0
                else:
                    y = 4.0
                return y

        out = CppCompiler().compile(f, arg_types=[RealType(fp.FP64)])
        assert out.count('else if') == 2
        ok, err = _typechecks(out)
        assert ok, f'flattened chain does not compile:\n{err}'

    def test_setup_keeps_the_nesting(self):
        """A condition needing statements of its own must not be flattened.

        Those statements would land before the ``else`` and so run
        unconditionally.  Here the arm's exponent ``1 - n`` is not a bare name,
        so it is hoisted to a temporary from inside the condition, and the
        ``else { if ... }`` shape has to stay.
        """
        @fp.fpy
        def f(x: fp.Real, n: fp.Real) -> fp.Real:
            with fp.FP64:
                if x > 100.0:
                    y = 1.0
                elif ((2 ** (1 - n)) * x) > 0.0:
                    y = 2.0
                else:
                    y = 3.0
                return y

        out = CppCompiler().compile(
            f, arg_types=[RealType(fp.FP64), RealType(fp.SINT16)])
        assert 'else if' not in out
        # the hoisted temporaries sit inside the `else`, not before it
        lines = [ln.strip() for ln in out.splitlines()]
        i_else = lines.index('} else {')
        # any `TYPE _tN ...;` declaration -- which storage inference picks is
        # not what this test is about
        hoisted = [i for i, ln in enumerate(lines)
                   if len(p := ln.split()) >= 2 and p[1].startswith('_t')]
        assert hoisted, 'expected the condition to hoist a temporary'
        # the inner `if` is the one after the `else`; its condition reads a
        # name the setup bound, whichever name statement form gave it
        i_if = next(i for i, ln in enumerate(lines)
                    if i > i_else and ln.startswith('if ('))
        assert all(i_else < i < i_if for i in hoisted), 'setup escaped the else block'
