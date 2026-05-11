"""
Public API for the cpp backend.

:class:`CppCompiler` runs the analysis pipeline (monomorphization,
def-use, context-use, array-size, format inference, storage
inference) on a :class:`Function` and hands the result to
:class:`_CppEmitter`, which produces a C++ source string.  Errors
surface as :class:`CppCompileError`.

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
from ...analysis.context_use import ContextUseAnalysis
from ...analysis.format_infer import FormatAnalysis
from ...ast.fpyast import FuncDef
from ...function import Function
from ...number import Context
from ...transform import Monomorphize
from ...types import Type
from ..backend import Backend, CompileError

from .emitter import CppEmitError, _CppEmitter
from .storage import StorageSelectionError
from .storage_infer import StorageAnalysis, StorageInfer
from .utils import CPP_HEADERS, CPP_HELPERS


class CppCompileError(CompileError):
    """Raised when cpp compilation fails."""
    pass


@dataclass
class CppPipelineResult:
    """
    Carrier for the per-function analysis state the emitter consumes.

    - ``ast``: the (post-monomorphization) :class:`FuncDef`.
    - ``format_info``: per-expression and per-definition format bounds.
    - ``ctx_use``: context scopes — used by the emitter to identify
      the active rounding context at every ``FuncDef`` / ``with``
      site (so it can validate / emit ``fesetround``).
    - ``storage``: per-SSA-def storage assignment.  Each def maps to a
      C++ identifier and storage type; two defs share storage iff they
      are connected either by phi edges *or* by an in-place mutation
      edge (the SSA-fresh def at an ``IndexedAssign`` site).
    """
    ast: FuncDef
    format_info: FormatAnalysis
    ctx_use: ContextUseAnalysis
    storage: StorageAnalysis


class CppCompiler(Backend):
    """
    Format-inference-driven C++ compiler.

    The pipeline runs all pre-analyses and assigns each SSA def to a
    C++ variable; the emitter then walks the AST and produces source.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Translation-unit preamble
    #
    # ``compile`` returns a function definition only, so single-function
    # tests can use exact-string equality.  Callers that want a full
    # translation unit pull these explicitly:
    #
    #     headers = '\\n'.join(cc.headers())
    #     unit = headers + '\\n' + cc.helpers() + cc.compile(f) + '\\n'

    def headers(self) -> list[str]:
        """C++ headers required by every emitted unit."""
        return list(CPP_HEADERS)

    def helpers(self) -> str:
        """Runtime helper definitions emitted alongside compiled
        functions.  Currently empty — cpp doesn't yet need custom
        runtime support beyond ``<cmath>`` / ``std::vector``."""
        return CPP_HELPERS

    def prelude(self) -> str:
        """Convenience: the headers and helpers concatenated as a
        single source-ready string.  Equivalent to
        ``'\\n'.join(self.headers()) + '\\n' + self.helpers()``."""
        return '\n'.join(self.headers()) + '\n' + self.helpers()

    # ------------------------------------------------------------------
    # Pipeline — runs all pre-analyses and selects storage.

    def _run_pipeline(
        self,
        func: Function,
        ctx: Context | None,
        arg_types: Collection[Type | None] | None,
    ) -> CppPipelineResult:
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
            raise CppCompileError(str(e)) from e

        return CppPipelineResult(
            ast=ast,
            format_info=format_info,
            ctx_use=ctx_use,
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
        emitter = _CppEmitter(
            ast=result.ast,
            storage=result.storage,
            def_use=result.format_info.type_info.def_use,
            format_info=result.format_info,
            ctx_use=result.ctx_use,
        )
        try:
            return emitter.emit()
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{func.name}`: {e}'
            ) from e
