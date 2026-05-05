"""
Public API for the cpp2 backend.

The compiler is built up across several phases (see
``docs/todos/backend-cpp.md``).  At present this is a stub: the public
surface is defined so callers can import :class:`Cpp2Compiler`, but
:meth:`Cpp2Compiler.compile` is not yet implemented.
"""

from typing import Collection

from ...ast.fpyast import FuncDef
from ...function import Function
from ...number import Context
from ...types import Type
from ..backend import Backend, CompileError


class Cpp2CompileError(CompileError):
    """Raised when cpp2 compilation fails."""
    pass


class Cpp2Compiler(Backend):
    """
    Format-inference-driven C++ compiler.

    This is the stub from Phase 0 of the cpp2 plan.  It exposes the
    public API surface (constructor, :meth:`compile`) so callers can
    import the class and so we can layer the compilation logic on top
    in subsequent phases.

    Subsequent phases will fill in:

    1. The pre-analysis pipeline (def-use, type-info, ctx-use,
       array-size, format-infer).
    2. Storage-type selection driven by the inferred format per def.
    3. Code emission for the language's expression and statement nodes,
       starting with scalar arithmetic under a concrete context and
       expanding to lists, tuples, control flow, and rounding boundaries.
    """

    def __init__(self):
        pass

    def compile(
        self,
        func: Function,
        *,
        ctx: Context | None = None,
        arg_types: Collection[Type | None] | None = None,
    ) -> str:
        """
        Compile *func* to a C++ source-code string.

        Args:
            func: The :class:`Function` to compile.
            ctx: Optional rounding context to monomorphize against.
            arg_types: Optional per-argument types to monomorphize against.

        Returns:
            The compiled C++ source as a single string.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')
        # Phase 0: not yet implemented.
        raise Cpp2CompileError(
            'cpp2 backend is under construction; compile() is not yet '
            'implemented.  See docs/todos/backend-cpp.md.'
        )
