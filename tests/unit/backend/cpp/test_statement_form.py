"""The three positions a statement must never escape from, and the net.

A ``while`` condition, a ternary arm and a short-circuited operand are each
evaluated conditionally or repeatedly while the line they sit on is not, so an
operand needing a statement of its own would have it run where the operand does
not.  All three were live miscompiles -- a loop testing a value computed once,
and two assertions firing on a path FPy never takes -- recorded in
``docs/todos/backend-cpp.md``.

:class:`fpy2.transform.ANF` lowers all three before codegen, so these programs
now compile and *run*.  The runs are the point: each witness is one the emitter
got wrong by hanging or aborting, which no string comparison would have caught.

:class:`TestTheNet` checks the emitter refuses rather than miscompiles if a
program ever reaches it un-normalized -- the guarantee that survives a future
change to the pass.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import fpy2 as fp
from fpy2.backend.cpp import compiler as _compiler
from fpy2.backend.cpp.compiler import CppCompileError, CppCompiler
from fpy2.types import RealType

_CXX = shutil.which('c++') or shutil.which('g++')
_OPTS = ['-std=c++11', '-O0', '-Wall', '-Wextra', '-Werror=return-type']

pytestmark = pytest.mark.skipif(_CXX is None, reason='no C++ compiler')

_FP32 = fp.IEEEContext(8, 32)


def _run(func, arg_types, call: str, *, timeout: float = 10.0) -> str:
    """Compile *func*, run ``main`` printing ``call``, return its stdout.

    Fails on a non-zero exit, which is how an aborted assertion shows up, and on
    a timeout, which is how a loop that never terminates does.
    """
    cc = CppCompiler()
    src = (
        cc.prelude() + '\n'
        + cc.compile(func, ctx=fp.FP64, arg_types=arg_types) + '\n'
        + '#include <cstdio>\n'
        + f'int main() {{ printf("%g\\n", (double){call}); return 0; }}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        cpp, exe = Path(d) / 'w.cpp', Path(d) / 'w'
        cpp.write_text(src)
        assert _CXX is not None
        build = subprocess.run(
            [_CXX, *_OPTS, '-o', str(exe), str(cpp)],
            capture_output=True, text=True, check=False,
        )
        assert build.returncode == 0, build.stderr
        try:
            out = subprocess.run(
                [str(exe)], capture_output=True, text=True,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f'the emitted loop did not terminate\n{src}')
        assert out.returncode == 0, f'exit {out.returncode}\n{src}\n{out.stderr}'
        return out.stdout.strip()


class TestTheWitnesses:
    """Each ran wrong before statement form; each runs right now."""

    def test_a_while_condition_is_re_evaluated(self):
        """The reduction used to be hoisted out, so the loop tested a value
        computed once and never terminated."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        assert _run(f, [RealType(fp.FP64)], 'f(3.0)') == '0'
        assert repr(f(3.0)) == repr(fp.Function(f.ast, runtime=f.runtime)(3.0))

    def test_an_untaken_ternary_arm_does_not_assert(self):
        """`fp.cast`'s losslessness assertion used to be hoisted out of the
        conditional and abort for a value the taken arm never casts."""

        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with _FP32:
                    y = 0.0 if x > 1e30 else fp.cast(x)
                return y

        assert _run(f, [RealType(fp.FP64)], 'f(1e300)') == '0'

    def test_a_short_circuited_operand_does_not_assert(self):
        """The same assertion, past an `or` that already decided the answer."""

        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                with _FP32:
                    y = x > 1e30 or fp.cast(x) > 0.0
                return y

        assert _run(f, [RealType(fp.FP64)], 'f(1e300)') == '1'


@pytest.fixture
def anf_disabled(monkeypatch):
    """The pipeline with statement form turned off.

    Exactly the state a future change to the pass could leave a program in --
    and the state every one of these programs was compiled in before it existed.
    """
    class _Identity:
        @staticmethod
        def apply(func):
            return func

    monkeypatch.setattr(_compiler, 'ANF', _Identity)


class TestTheNet:
    """Un-normalized input is refused, not miscompiled."""

    @staticmethod
    def _emit_unnormalized(func):
        return CppCompiler().compile(
            func, ctx=fp.FP64, arg_types=[RealType(fp.FP64)],
        )

    def test_a_while_condition_is_refused(self, anf_disabled):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                y = x
                while max([y, 0.0]) > 0.0:
                    y = y - 1.0
                return y

        with pytest.raises(CppCompileError, match='while.*condition'):
            self._emit_unnormalized(f)

    def test_a_ternary_arm_is_refused(self, anf_disabled):
        @fp.fpy
        def f(x: fp.Real) -> fp.Real:
            with fp.FP64:
                with _FP32:
                    y = 0.0 if x > 1e30 else fp.cast(x)
                return y

        with pytest.raises(CppCompileError, match='ternary arm'):
            self._emit_unnormalized(f)

    def test_a_short_circuited_operand_is_refused(self, anf_disabled):
        @fp.fpy
        def f(x: fp.Real) -> bool:
            with fp.FP64:
                with _FP32:
                    y = x > 1e30 or fp.cast(x) > 0.0
                return y

        with pytest.raises(CppCompileError, match='short-circuited'):
            self._emit_unnormalized(f)
