"""
C++ backend: format-inference-driven compilation to C++.

Uses the :class:`FormatInfer` analysis to choose storage types per
definition and to decide where rounding is needed, separate from
where storage shape is decided.  See ``docs/todos/backend-cpp.md``
for design notes.
"""

from .compiler import CppCompileError, CppCompiler

__all__ = [
    'CppCompileError',
    'CppCompiler',
]
