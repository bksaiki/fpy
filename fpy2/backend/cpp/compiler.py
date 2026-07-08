"""
Public API for the cpp backend.

:class:`CppCompiler` runs the analysis pipeline (monomorphization,
def-use, context-use, array-size, format inference, storage
inference) on a :class:`Function` and hands the result to
:class:`_CppEmitter`, which produces a C++ source string.  Errors
surface as :class:`CppCompileError`.
"""

from typing import Collection

from ...analysis import (
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    FormatInfer,
)
from ...ast.fpyast import Call, FuncDef
from ...ast.visitor import DefaultVisitor
from ...function import Function
from ...module import Module
from ...number import Context
from ...transform import FreeVarElim, RoundElim, Specialize, ZipElim
from ...transform.free_var_elim import inline_literal
from ...types import Type
from ..backend import Backend, CompileError

from .emitter import CppEmitError, CppEmitter
from .storage import StorageSelectionError
from .storage_infer import StorageInfer
from .utils import CPP_HEADERS, CPP_HELPERS


class CppCompileError(CompileError):
    """Raised when cpp compilation fails."""
    pass



# ---------------------------------------------------------------------
# Compiler


def _collect_call_names(ast: FuncDef) -> dict[Call, str]:
    """Build a ``Call → emit-name`` map for every Function-targeted Call in
    *ast*'s body.  After :class:`Specialize`, each such ``Call.fn`` points at
    the target spec's :class:`Function`, so the emit name is just
    ``call.fn.ast.name``."""
    out: dict[Call, str] = {}

    class _Collector(DefaultVisitor):
        def _visit_call(self, e: Call, ctx):
            if isinstance(e.fn, Function):
                out[e] = e.fn.ast.name
            super()._visit_call(e, ctx)

    _Collector()._visit_function(ast, None)
    return out


class CppCompiler(Backend):
    """
    Format-inference-driven C++ compiler.

    The pipeline runs all pre-analyses and assigns each SSA def to a
    C++ variable; the emitter then walks the AST and produces source.

    Args:
        unsafe_cast_int:
            When ``True`` (default), allow rounded arithmetic under
            an unbounded-integer context (``MPFixedContext(nmin=-1)``
            / ``fpy2.INTEGER``); the compiler compiles these by
            emitting casts to the widest built-in integer type
            (currently ``int64_t``) and assuming no overflow occurs.
            Set ``False`` to reject such programs at compile time.
        optimize:
            When ``True`` (default), apply optimizing program
            transformations to each :class:`FuncDef` before the rest
            of the pipeline runs:

            - :class:`fpy2.transform.ZipElim` (pre-monomorphize):
              skips materializing intermediate
              ``std::vector<std::tuple<...>>``s for ``zip`` iterables.
            - :class:`fpy2.transform.RoundElim` (post-monomorphize):
              hoists eliminable rounded operations into
              ``with fp.REAL:`` blocks so the cpp emitter's
              lossless-widening dispatch can pick tighter storage
              for them.

            The pipeline is sound either way.  Set ``False`` to
            compile the surface AST verbatim.
    """

    _unsafe_cast_int: bool
    _optimize: bool

    def __init__(
        self, *, unsafe_cast_int: bool = True, optimize: bool = True,
    ):
        self._unsafe_cast_int = unsafe_cast_int
        self._optimize = optimize

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
        return '\n'.join(self.headers()) + '\n\n' + self.helpers()

    # ------------------------------------------------------------------
    # Pipeline — runs all pre-analyses and selects storage.

    def compile(
        self,
        func: Function,
        *,
        ctx: Context | None = None,
        arg_types: Collection[Type | None] | None = None,
    ) -> str:
        """Compile *func* to a C++ source-code string.

        Thin wrapper around :meth:`compile_module` over a one-entry module,
        so the single-function and module paths share one pipeline.

        Args:
            func: The :class:`Function` to compile.
            ctx: Optional rounding context to monomorphize against.
            arg_types: Optional per-argument types to monomorphize against.
        """
        m = Module()
        m.add(func, ctx=ctx, arg_types=arg_types)
        return self.compile_module(m)

    def compile_module(self, module: Module) -> str:
        """Compile a :class:`~fpy2.Module` to a single C++ translation unit.

        Pipeline:
          1. **Pre-spec optimizations** (``ZipElim``) on every function in
             the module via ``map``.
          2. **Specialize** the module: each ``(FuncDef, ctx, arg_fmts)``
             becomes one entry; cross-function calls rewire to the
             appropriate spec.
          3. **Post-spec optimizations** (``RoundElim``) on each spec —
             monomorphic format inference is now available.
          4. **Per-spec codegen**, leaves-first, one C++ definition per entry.
        """
        if not isinstance(module, Module):
            raise TypeError(f'Expected `Module`, got {type(module)} for {module}')

        # Close each function over its captured *data* free variables
        # (materialize them as leading assignments) so codegen never sees an
        # undeclared closure value.  This is a correctness pass — always runs.
        module = module.map(lambda _m, fd: FreeVarElim.apply(fd))

        if self._optimize:
            module = module.map(lambda _m, fd: ZipElim.apply(fd))

        # Translate ``Monomorphize``'s bare ``RuntimeError`` (e.g. arg-type
        # mismatches) into ``CppCompileError`` so callers iterating over
        # candidate functions can catch a uniform error type.
        try:
            specialized = Specialize.apply(module)
        except RuntimeError as e:
            raise CppCompileError(f'specialization failed: {e}') from e

        if self._optimize:
            specialized = specialized.map(lambda _m, fd: RoundElim.apply(fd))

        cg = specialized.call_graph()
        return '\n\n'.join(self._compile_function(f) for f in cg.order)

    @staticmethod
    def _assert_free_vars_closed(ast: FuncDef) -> None:
        """After :class:`FreeVarElim`, no *data* free variable should remain —
        one would be emitted as an undeclared value.  Free variables bound to
        callables/modules (used only for call or attribute resolution) have no
        literal form and are left alone."""
        for fv in ast.free_vars:
            name = str(fv)
            if name in ast.env and inline_literal(ast.env[name]) is not None:
                raise CppCompileError(
                    f'unbound data free variable `{name}` = {ast.env[name]!r}'
                )

    def _compile_function(self, func: Function) -> str:
        """Emit one C++ function definition for a fully-specialized
        :class:`Function`.  ``func.ast.name`` is the final emitted name
        (set by :class:`Specialize` — public entries keep their user-given
        name, private specs get a mangled one)."""
        ast = func.ast
        self._assert_free_vars_closed(ast)

        # Per-spec analyses.
        def_use = DefineUse.analyze(ast)
        ctx_use = ContextUse.analyze(ast, def_use=def_use)
        array_size = ArraySizeInfer.analyze(ast)
        format_info = FormatInfer.analyze(
            ast,
            def_use=def_use,
            ctx_use=ctx_use,
            array_size=array_size,
        )

        # Per-spec storage selection.
        try:
            storage = StorageInfer.infer(
                format_info.type_info.def_use, format_info.by_def,
            )
        except StorageSelectionError as e:
            raise CppCompileError(
                f'storage selection failed for `{func.name}`: {e}'
            ) from e

        # Call.fn → emitted name.  ``Specialize`` rewired each Call.fn at
        # the source so call.fn.ast.name is the target spec's emit name.
        call_names = _collect_call_names(ast)

        emitter = CppEmitter(
            ast=ast,
            storage=storage,
            def_use=def_use,
            format_info=format_info,
            ctx_use=ctx_use,
            call_names=call_names,
            unsafe_cast_int=self._unsafe_cast_int,
        )
        try:
            return emitter.emit()
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{func.name}`: {e}'
            ) from e

    # ------------------------------------------------------------------
    # Call-graph walk

