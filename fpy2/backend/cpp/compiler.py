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
    TypeInfer,
)
from ...analysis.context_use import ContextUseAnalysis
from ...analysis.format_infer import FormatAnalysis
from ...ast.fpyast import Ast, Call, FuncDef
from ...function import Function
from ...number import Context, FP64
from ...transform import Monomorphize
from ...types import BoolType, ContextType, ListType, RealType, TupleType, Type, VarType
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
            The compiled C++ source.  When *func* invokes other
            FPy functions, each callee is emitted in topological
            order above *func* so the result is a self-contained
            sequence of definitions.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')

        # Compile transitive callees first so each is defined before
        # any user.  Callees are monomorphized at ``FP64`` with their
        # declared parameter types instantiated the same way the
        # infra harness does for top-level functions — symbolic Real
        # / Var slots collapse to ``RealType(FP64)``.
        defs: list[str] = []
        seen: set[str] = set()
        for callee in _collect_callees(func):
            if callee.name in seen:
                continue
            seen.add(callee.name)
            try:
                callee_arg_types = _default_arg_types(callee)
            except _UninstantiableType as e:
                raise CppCompileError(
                    f'cannot derive arg types for callee `{callee.name}`: {e}'
                )
            defs.append(self._compile_single(callee, FP64, callee_arg_types))

        defs.append(self._compile_single(func, ctx, arg_types))
        return '\n\n'.join(defs)

    def _compile_single(
        self,
        func: Function,
        ctx: Context | None,
        arg_types: Collection[Type | None] | None,
    ) -> str:
        """Single-function compile — runs the pipeline and the emitter
        without touching callees.  Used by :meth:`compile` to emit
        each transitive callee independently."""
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


# ---------------------------------------------------------------------
# Cross-function call support.
#
# When a function ``f`` invokes another FPy function ``g`` via a
# :class:`Call` node, the C++ output needs ``g``'s definition above
# ``f``'s.  We walk the AST for ``Call`` nodes, recursively collecting
# the callees in pre-order — duplicates are filtered at the caller side
# so the first occurrence in topological order wins.


class _UninstantiableType(Exception):
    """A callee parameter has a type the C++ backend can't
    instantiate (e.g., function or context types)."""


def _instantiate_type(ty: Type) -> Type:
    """Mirror of ``tests/infra/backend/cpp.py``'s ``_inst_type``:
    collapse symbolic ``VarType`` / ``RealType`` to ``RealType(FP64)``
    and recurse through aggregates.  Bool / Context types pass through
    unchanged.  Anything else is :class:`_UninstantiableType`."""
    match ty:
        case BoolType() | ContextType():
            return ty
        case VarType() | RealType():
            return RealType(FP64)
        case TupleType():
            return TupleType(*[_instantiate_type(elt) for elt in ty.elts])
        case ListType():
            return ListType(_instantiate_type(ty.elt))
        case _:
            raise _UninstantiableType(ty.format())


def _default_arg_types(func: Function) -> list[Type]:
    """Default per-arg types for a callee: run :class:`TypeInfer`
    over its body, then instantiate symbolic slots to ``FP64``."""
    ty_info = TypeInfer.check(func.ast)
    return [_instantiate_type(ty) for ty in ty_info.arg_types]


def _direct_callees(func: Function) -> list[Function]:
    """All :class:`Function` values reached by a :class:`Call` node
    inside *func*'s body, in source order (duplicates preserved).
    Walks every ``Ast``-typed slot recursively — works on the live
    AST without needing a fully-spelled visitor."""
    out: list[Function] = []

    def walk(node):
        if isinstance(node, Call) and isinstance(node.fn, Function):
            out.append(node.fn)
        if isinstance(node, Ast):
            for slot in node.__slots__:
                walk(getattr(node, slot, None))
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(func.ast.body)
    return out


def _collect_callees(func: Function) -> list[Function]:
    """Topologically-ordered (callee-before-caller) list of every
    :class:`Function` transitively reachable from *func* via
    :class:`Call`.  ``func`` itself is not included."""
    out: list[Function] = []
    queue: list[Function] = []
    enqueued: set[str] = set()

    def _enqueue(f: Function):
        for callee in _direct_callees(f):
            if callee.name in enqueued:
                continue
            enqueued.add(callee.name)
            queue.append(callee)

    _enqueue(func)
    while queue:
        f = queue.pop(0)
        out.append(f)
        _enqueue(f)
    # Reverse so leaf callees emit first: each function comes before
    # everyone that calls it.
    out.reverse()
    return out
