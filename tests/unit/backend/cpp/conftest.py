"""Every emission these tests assert on is also compiled.

The assertions here grep emitted text, which a type error passes unnoticed: a
narrowed element type once produced ``std::max(int16_t, float)`` under a green
suite.  Wrapping the compiler rather than each call site means a new test gets
the check without asking for it.

A type error is an error everywhere, so the flags only have to promote the one
diagnostic that is not: a narrowing inside a braced initializer, ill-formed in
the standard but a warning on GCC.  Style is deliberately not promoted -- the
emitter parenthesizes every operand, which clang alone objects to, and that is
not what this is looking for.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.utils import CPP_HEADERS

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')

_NARROWS = 'int main() { double d = 1.5; int a[] = {d}; return a[0]; }'
"""A braced initializer that narrows: ill-formed, but only a warning by
default on GCC."""


def _narrowing_flag(cxx: str) -> list[str]:
    """``-Werror=`` for a narrowing conversion, in whichever spelling *cxx*
    knows -- GCC and clang name the diagnostic differently.

    Probed by compiling something that narrows and keeping the flag that turns
    it into an error.  Asking whether the *name* is accepted would not do: an
    unknown ``-Werror=`` is itself only a warning on clang, so a wrong guess
    would read as success and silently drop the check.
    """
    for flag in ('-Werror=narrowing', '-Werror=c++11-narrowing'):
        probe = subprocess.run(
            [cxx, flag, '-std=c++17', '-fsyntax-only', '-x', 'c++', '-'],
            input=_NARROWS, capture_output=True, text=True,
        )
        if probe.returncode != 0:
            return [flag]
    return []


_FLAGS = ['-std=c++17', '-fsyntax-only'] + (
    _narrowing_flag(_CXX) if _CXX else []
)


def _check(src: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        cpp = Path(td) / 'm.cpp'
        cpp.write_text('\n'.join(CPP_HEADERS) + '\n' + src)
        out = subprocess.run(
            [_CXX, *_FLAGS, str(cpp)], capture_output=True, text=True,
        )
    assert out.returncode == 0, (
        f'emitted C++ does not compile:\n{out.stderr[-1500:]}\n--- source ---\n{src}'
    )


@pytest.fixture(autouse=True)
def _compiles_what_it_emits(monkeypatch):
    if _CXX is None:
        return
    original = CppCompiler.compile

    def checked(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        _check(out)
        return out

    monkeypatch.setattr(CppCompiler, 'compile', checked)
