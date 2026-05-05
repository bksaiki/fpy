"""
Public API for the cpp2 backend.

The compiler is built up across several phases (see
``docs/todos/backend-cpp.md``).  At present this is a stub: the public
surface is defined so callers can import :class:`Cpp2Compiler`, but
:meth:`Cpp2Compiler.compile` is not yet implemented.
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
from .phi_web import PhiWeb, compute_phi_web
from .storage import StorageSelectionError


class Cpp2CompileError(CompileError):
    """Raised when cpp2 compilation fails."""
    pass


@dataclass
class Cpp2PipelineResult:
    """
    Carrier for the per-function analysis state computed in Phase 1.

    The cpp2 emitter (Phase 2 onward) consumes this rather than re-running
    each pre-analysis itself.  The fields parallel the analyses listed in
    ``docs/todos/backend-cpp.md``:

    - ``ast``: the (post-monomorphization) :class:`FuncDef`.
    - ``format_info``: per-expression and per-definition format bounds.
    - ``phi_web``: phi-equivalence partition.  Each SSA def maps to a
      class; each class has a single C++ name and storage type.  Two
      defs share storage iff they are connected by phi edges.
    """
    ast: FuncDef
    format_info: FormatAnalysis
    phi_web: PhiWeb


class Cpp2Compiler(Backend):
    """
    Format-inference-driven C++ compiler.

    Phase 1 wires the pre-analysis pipeline and chooses storage types
    per definition.  Code emission lands in subsequent phases.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Pipeline (Phase 1) — runs all pre-analyses and selects storage.

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
        ast = Monomorphize.apply_by_arg(ast, ctx, arg_types)

        # The analyses are independent up to def_use.
        def_use = DefineUse.analyze(ast)
        ctx_use = ContextUse.analyze(ast, def_use=def_use)
        array_size = ArraySizeInfer.analyze(ast)
        format_info = FormatInfer.analyze(
            ast,
            def_use=def_use,
            ctx_use=ctx_use,
            array_size=array_size,
        )

        # Compute the phi-web partition: defs joined by phi edges share
        # a C++ variable, anything else is free to rename.  See
        # ``phi_web.py`` for the full contract.
        try:
            phi_web = compute_phi_web(
                format_info.type_info.def_use,
                format_info.by_def,
            )
        except StorageSelectionError as e:
            raise Cpp2CompileError(str(e)) from e

        return Cpp2PipelineResult(
            ast=ast,
            format_info=format_info,
            phi_web=phi_web,
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
            phi_web=result.phi_web,
            def_use=result.format_info.type_info.def_use,
            format_info=result.format_info,
        )
        try:
            return emitter.emit()
        except Cpp2EmitError as e:
            raise Cpp2CompileError(str(e)) from e
