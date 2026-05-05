"""
Public API for the cpp2 backend.

:class:`Cpp2Compiler` runs the analysis pipeline (monomorphization,
def-use, context-use, array-size, format inference, storage
inference) on a :class:`Function` and hands the result to
:class:`_Cpp2Emitter`, which produces a C++ source string.  Errors
surface as :class:`Cpp2CompileError`.

Phase notes live in ``docs/todos/backend-cpp.md``.
"""

from dataclasses import dataclass
from typing import Collection

from ...analysis import (
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    FormatInfer,
)
from ...analysis.format_infer import FormatAnalysis
from ...ast.fpyast import FuncDef
from ...function import Function
from ...number import Context
from ...transform import Monomorphize
from ...types import Type
from ..backend import Backend, CompileError

from .emitter import Cpp2EmitError, _Cpp2Emitter
from .storage import StorageSelectionError
from .storage_infer import StorageAnalysis, StorageInfer


class Cpp2CompileError(CompileError):
    """Raised when cpp2 compilation fails."""
    pass


@dataclass
class Cpp2PipelineResult:
    """
    Carrier for the per-function analysis state the emitter consumes.

    - ``ast``: the (post-monomorphization) :class:`FuncDef`.
    - ``format_info``: per-expression and per-definition format bounds.
    - ``storage``: per-SSA-def storage assignment.  Each def maps to a
      C++ identifier and storage type; two defs share storage iff they
      are connected either by phi edges *or* by an in-place mutation
      edge (the SSA-fresh def at an ``IndexedAssign`` site).
    """
    ast: FuncDef
    format_info: FormatAnalysis
    storage: StorageAnalysis


class Cpp2Compiler(Backend):
    """
    Format-inference-driven C++ compiler.

    The pipeline runs all pre-analyses and assigns each SSA def to a
    C++ variable; the emitter then walks the AST and produces source.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Pipeline — runs all pre-analyses and selects storage.

    def _run_pipeline(
        self,
        func: Function,
        ctx: Context | None,
        arg_types: Collection[Type | None] | None,
    ) -> Cpp2PipelineResult:
        ast = func.ast
        if arg_types is None:
            arg_types = [None for _ in func.args]
        if ast.ctx is not None:
            ctx = None

        # apply monomorphization to get concrete types
        ast = Monomorphize.apply_by_arg(ast, ctx, arg_types)

        # run program analyses to get the information the emitter needs
        def_use = DefineUse.analyze(ast)
        ctx_use = ContextUse.analyze(ast, def_use=def_use)
        array_size = ArraySizeInfer.analyze(ast)
        format_info = FormatInfer.analyze(
            ast,
            def_use=def_use,
            ctx_use=ctx_use,
            array_size=array_size,
        )

        # Per-SSA-def storage assignment.  Defs joined by phi edges or
        # by an in-place mutation edge (``IndexedAssign``) share a
        # C++ variable; anything else is free to rename.  See
        # ``storage_infer.py`` for the full contract.
        try:
            storage = StorageInfer.infer(
                format_info.type_info.def_use,
                format_info.by_def,
            )
        except StorageSelectionError as e:
            raise Cpp2CompileError(str(e)) from e

        return Cpp2PipelineResult(
            ast=ast,
            format_info=format_info,
            storage=storage,
        )

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
        result = self._run_pipeline(func, ctx, arg_types)
        emitter = _Cpp2Emitter(
            ast=result.ast,
            storage=result.storage,
            def_use=result.format_info.type_info.def_use,
            format_info=result.format_info,
        )
        try:
            return emitter.emit()
        except Cpp2EmitError as e:
            raise Cpp2CompileError(str(e)) from e
