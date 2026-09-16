"""Every emission these tests assert on is also compiled.

The assertions here grep emitted text, which a type error passes unnoticed: a
narrowed element type once produced ``std::max(int16_t, float)`` under a green
suite.  Wrapping the compiler rather than each call site means a new test gets
the check without asking for it.

Warnings are errors -- a narrowing inside a braced initializer is only a warning
on GCC and ill-formed in the standard.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from fpy2.backend.cpp import CppCompiler
from fpy2.backend.cpp.utils import CPP_HEADERS

_CXX = shutil.which('c++') or shutil.which('g++') or shutil.which('clang++')

_FLAGS = [
    '-std=c++17', '-Wall', '-Wextra', '-Werror', '-fsyntax-only',
    # a program small enough to test often leaves an emitted name unread; that
    # is not what this is looking for
    '-Wno-unused-variable', '-Wno-unused-but-set-variable',
    '-Wno-unused-parameter',
]


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
