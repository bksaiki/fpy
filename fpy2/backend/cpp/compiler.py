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
from ...analysis.format_infer import FormatAnalysis, PreAnalysisCache
from ...ast.fpyast import Call, FuncDef
from ...function import Function
from ...number import Context
from ...number.context.ieee754 import IEEEContext
from ...number.context.mp_fixed import MPFixedContext
from ...number.context.mpb_fixed import MPBFixedContext
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
        *,
        pre_cache: PreAnalysisCache,
    ) -> CppPipelineResult:
        """Monomorphize *func*, run pre-analyses, then run
        :class:`FormatInfer` and :class:`StorageInfer` for the
        top-level instantiation.

        The pre-analysis cache is shared with sub-analyses created
        by :class:`FormatInfer` when walking into :class:`Call`
        targets — each :class:`FuncDef` is structurally analyzed at
        most once across the whole call graph.
        """
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
            pre_cache=pre_cache,
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

        Cross-function calls are template-specialized: each unique
        ``(callee FuncDef, outer rounding context)`` pair produces
        one C++ definition with a mangled name.  Specializations are
        emitted before their callers so the result is a self-contained
        sequence of definitions.

        Args:
            func: The :class:`Function` to compile.
            ctx: Optional rounding context to monomorphize against.
            arg_types: Optional per-argument types to monomorphize against.

        Returns:
            The compiled C++ source.
        """
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')

        # Shared structural-analysis cache.  Every :class:`FuncDef`
        # encountered in the call graph gets def-use / type-infer /
        # context-use / array-size run exactly once.
        pre_cache = PreAnalysisCache()
        top_result = self._run_pipeline(
            func, ctx, arg_types, pre_cache=pre_cache,
        )

        # Walk the FormatAnalysis call graph in post-order so each
        # specialization is emitted before everyone that calls it.
        specs: dict[tuple, _Specialization] = {}
        emit_order: list[tuple] = []
        top_call_names = self._discover_specializations(
            top_result.format_info, specs, emit_order,
        )

        out: list[str] = [
            self._emit_specialization(specs[k]) for k in emit_order
        ]
        out.append(self._emit_top_level(func.name, top_result, top_call_names))
        return '\n\n'.join(out)

    # ------------------------------------------------------------------
    # Call-graph walk

    def _discover_specializations(
        self,
        format_info: FormatAnalysis,
        specs: dict[tuple, '_Specialization'],
        emit_order: list[tuple],
    ) -> dict[Call, str]:
        """Walk *format_info*'s ``by_call`` recursively.  For each
        :class:`Call` whose ``fn`` is a known :class:`Function`,
        materialize (or look up) the specialization for the
        ``(callee FuncDef, outer rounding context)`` pair.

        Returns a ``Call → mangled name`` mapping covering every Call
        in *format_info*'s function body.  Newly-discovered
        specializations append their key to *emit_order* after their
        own sub-calls finish — so the resulting order is
        callee-before-caller.
        """
        call_names: dict[Call, str] = {}
        for call_node, sub_fa in format_info.by_call.items():
            sig = sub_fa.fn_fmt
            seed_key = _param_seeds_fingerprint(sig.arg_fmts)
            key = (
                id(sub_fa.func),
                _ctx_fingerprint(sig.ctx),
                seed_key,
            )
            if key not in specs:
                inner_call_names = self._discover_specializations(
                    sub_fa, specs, emit_order,
                )
                try:
                    storage = StorageInfer.infer(
                        sub_fa.type_info.def_use,
                        sub_fa.by_def,
                    )
                except StorageSelectionError as e:
                    raise CppCompileError(
                        f'storage selection failed for callee '
                        f'`{sub_fa.func.name}`: {e}'
                    ) from e
                mangled = _mangle_with_seeds(
                    sub_fa.func.name,
                    sig.ctx,
                    seed_key,
                    specs,
                )
                specs[key] = _Specialization(
                    func=sub_fa.func,
                    name=mangled,
                    format_info=sub_fa,
                    storage=storage,
                    call_names=inner_call_names,
                )
                emit_order.append(key)
            call_names[call_node] = specs[key].name
        return call_names

    # ------------------------------------------------------------------
    # Emission

    def _emit_specialization(self, spec: '_Specialization') -> str:
        emitter = _CppEmitter(
            ast=spec.func,
            storage=spec.storage,
            def_use=spec.format_info.type_info.def_use,
            format_info=spec.format_info,
            ctx_use=spec.format_info.ctx_use,
            func_name_override=spec.name,
            call_names=spec.call_names,
        )
        try:
            return emitter.emit()
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{spec.func.name}` '
                f'(specialization `{spec.name}`): {e}'
            ) from e

    def _emit_top_level(
        self,
        name: str,
        result: CppPipelineResult,
        call_names: dict[Call, str],
    ) -> str:
        emitter = _CppEmitter(
            ast=result.ast,
            storage=result.storage,
            def_use=result.format_info.type_info.def_use,
            format_info=result.format_info,
            ctx_use=result.ctx_use,
            call_names=call_names,
        )
        try:
            return emitter.emit()
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{name}`: {e}'
            ) from e


# ---------------------------------------------------------------------
# Specialization machinery


@dataclass
class _Specialization:
    """A single C++ definition emitted for a
    ``(callee FuncDef, outer rounding context)`` pair.

    The ``func`` AST is shared across instantiations (the cpp backend
    does not monomorphize callees — formats and storage are derived
    from the per-instantiation :class:`FormatAnalysis` instead).  The
    ``call_names`` map gives each :class:`Call` in this body the
    mangled name of *its* specialization.
    """
    func: FuncDef
    name: str
    format_info: FormatAnalysis
    storage: StorageAnalysis
    call_names: dict[Call, str]


def _ctx_fingerprint(ctx: Context | None) -> str:
    """Stable, human-readable identifier for a rounding context.
    Used as part of the specialization key and the C++ mangled name."""
    if ctx is None:
        return 'any'
    if isinstance(ctx, IEEEContext):
        rm = ctx.rm.name.lower()
        return f'fp{ctx.nbits}_{rm}'
    if isinstance(ctx, MPFixedContext):
        return 'mpfixed'
    if isinstance(ctx, MPBFixedContext):
        return 'mpbfixed'
    # Fallback: hash the string repr to keep the name finite and
    # the key hashable without knowing every Context subclass.
    digest = hash(str(ctx)) & 0xFFFFFFFF
    return f'ctx{digest:08x}'


def _mangle(name: str, ctx: Context | None) -> str:
    """C++ name for a callee instantiated at *ctx*.  The double
    underscore matches a common templating convention and keeps the
    fingerprint visually separated from the user-chosen base name."""
    return f'{name}__{_ctx_fingerprint(ctx)}'


def _param_seeds_fingerprint(seeds) -> str:
    """Stable string identifying a tuple of parameter format seeds.

    Two specializations of the same callee that differ only in their
    parameter seeds (e.g. one called with FP32 args, the other with
    FP64 args, both under the same outer context) must map to
    different specializations.  The fingerprint becomes part of the
    cache key and — when ambiguous — part of the mangled name.

    Each seed is rendered via :func:`repr` and joined.  ``repr`` is
    sufficient because :class:`FormatBound` instances are
    ``dataclass(frozen=True)`` and have deterministic reprs.
    """
    return '|'.join(repr(s) for s in seeds)


def _mangle_with_seeds(
    name: str,
    ctx: Context | None,
    seed_key: str,
    specs: dict,
) -> str:
    """C++ name for a specialization.  Starts with the ctx-only
    mangle; if that name collides with an existing specialization of
    the same function at a different seed, falls back to a longer
    name that incorporates a seed-digest suffix.

    Most callees have a unique signature per ctx (parameter formats
    match the outer ctx), so the common case lands on the simple
    ``name__ctx`` form.  The longer form only fires when the corpus
    actually instantiates the same callee at the same outer ctx
    with different parameter shapes."""
    base = _mangle(name, ctx)
    # Detect collision: any existing spec for the same function and
    # ctx but a different seed_key needs a distinguishing suffix.
    for (fid, ctx_key, sk), spec in specs.items():
        if spec.func.name == name and ctx_key == _ctx_fingerprint(ctx):
            if sk != seed_key:
                digest = hash(seed_key) & 0xFFFFFFFF
                return f'{base}_{digest:08x}'
    return base
