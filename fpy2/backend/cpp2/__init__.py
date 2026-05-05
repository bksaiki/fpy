"""
C++ backend (v2): format-inference-driven compilation to C++.

This is a re-implementation of :mod:`fpy2.backend.cpp` that uses the
:class:`FormatInfer` analysis to choose storage types per definition
and to decide where rounding is needed, separate from where storage
shape is decided.  See ``docs/todos/backend-cpp.md`` for design notes.
"""

from .compiler import Cpp2Compiler, Cpp2CompileError

__all__ = [
    'Cpp2Compiler',
    'Cpp2CompileError',
]
