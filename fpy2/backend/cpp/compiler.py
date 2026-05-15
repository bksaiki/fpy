"""
Public API for the cpp backend.

:class:`CppCompiler` runs the analysis pipeline (monomorphization,
def-use, context-use, array-size, format inference, storage
inference) on a :class:`Function` and hands the result to
:class:`_CppEmitter`, which produces a C++ source string.  Errors
surface as :class:`CppCompileError`.

Phase notes live in ``docs/todos/backend-cpp.md``.
"""

import hashlib

from dataclasses import dataclass
from typing import Collection, NamedTuple

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
from ...transform import Monomorphize, ZipElim
from ...types import Type
from ..backend import Backend, CompileError

from .emitter import CppEmitError, CppEmitter
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

# ---------------------------------------------------------------------
# Specialization machinery


class _SpecKey(NamedTuple):
    """Identity of a :class:`_Specialization` in the unit's table.

    Two call sites that share all three components reuse a single
    specialization (and therefore a single C++ definition); any
    difference forces a fresh specialization with a distinct
    mangled name.
    """
    func_fp: str
    """Run-stable short digest of the callee :class:`FuncDef`
    (see :func:`_func_fingerprint`).  Distinguishes specializations
    by AST identity rather than by source name, so two FPy
    functions that happen to share a name (e.g. across modules)
    don't collide.  Derived from the function's source location
    when available, so the resulting C++ mangled names are
    reproducible across runs."""
    ctx_fp: str
    """Fingerprint of the incoming rounding context (see
    :func:`_ctx_fingerprint`)."""
    params_fp: str
    """Fingerprint of the per-parameter format tuple (see
    :func:`_param_fingerprint`).  Two specializations under the
    same context but with different parameter formats — e.g.
    invoked with FP32 args vs. FP64 args — get distinct keys."""


class _TopLevel(NamedTuple):
    """A top-level function registered with a translation unit —
    rendered after all its specializations and under its declared
    name (not a mangled one)."""
    name: str
    result: CppPipelineResult
    call_names: dict[Call, str]


@dataclass(frozen=True)
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
    # Fallback: deterministic digest of the string repr keeps the
    # name finite without enumerating every Context subclass, and
    # stays the same across runs (Python's built-in ``hash`` for
    # strings randomizes per process).
    digest = hashlib.sha1(str(ctx).encode()).hexdigest()[:8]
    return f'ctx{digest}'


def _func_fingerprint(func: FuncDef) -> str:
    """Run-stable short identifier for *func*.

    Prefers source location (filename + start line) so the same
    program produces the same mangled name across runs and
    processes.  Falls back to ``(name, id())`` when no location is
    available — only synthesized ASTs from transforms typically
    lack one, and those are re-derived per compile so within a
    single run the fingerprint is still uniquely identifying.

    Uses :mod:`hashlib` rather than Python's built-in :func:`hash`,
    because the latter randomizes its seed per process by default
    (``PYTHONHASHSEED``) — which would make emitted C++ names
    differ across runs of the same input.
    """
    loc = func.loc
    if loc is not None:
        material = f'loc:{loc.source}:{loc.start_line}'
    else:
        # No source position to anchor to — fall back to identity,
        # which is at least process-stable.  Include the name so
        # the digest is correlated with the function for debugging.
        material = f'anon:{func.name}:{id(func)}'
    return hashlib.sha1(material.encode()).hexdigest()[:8]


def _param_fingerprint(param_fmts) -> str:
    """Stable string identifying a tuple of per-parameter formats.

    Two specializations of the same callee that differ only in
    their parameter formats (e.g. one called with FP32 args, the
    other with FP64 args, both under the same outer context) must
    map to different specializations.  The fingerprint becomes
    part of the cache key and — when ambiguous — part of the
    mangled name.

    Each format is rendered via :func:`repr` and joined.  ``repr``
    is sufficient because :class:`FormatBound` instances are
    ``dataclass(frozen=True)`` and have deterministic reprs.
    """
    return '|'.join(repr(f) for f in param_fmts)


def _mangle_with_params(
    name: str,
    key: _SpecKey,
    specs: dict[_SpecKey, _Specialization],
) -> str:
    """C++ name for a specialization.  Starts with the ctx-only
    mangle; appends a disambiguating digest suffix when an existing
    spec would otherwise collide on the emitted C++ name.

    Two distinct ways the base name ``{source name}__{ctx_fp}`` can
    collide within a single translation unit:

    1. **Different FuncDef, same source name.**  Two FPy functions
       in different modules share a ``.name`` (e.g.
       ``fpy2.libraries.vector.zeros`` and
       ``fpy2.libraries.matrix.zeros``).  Collision detection here
       must compare by :attr:`_SpecKey.func_fp` (a stable digest of
       the callee's source location) — comparing by
       ``spec.func.name`` alone would treat the two as the same
       callee and emit duplicate C++ definitions.
    2. **Same FuncDef, different parameter formats.**  The corpus
       invokes the same callee at the same outer ctx with
       differently-shaped operands.

    The new spec receives the suffix; specs already in the table
    keep their existing (shorter) names.  Common case — every
    ``(name, ctx_fp)`` corresponds to exactly one
    ``(func_fp, params_fp)`` — lands on the bare ``name__ctx_fp``
    form.  Disambiguating suffixes are derived from
    :func:`_func_fingerprint` and :func:`_param_fingerprint`, both
    of which produce run-stable digests, so emitted names are
    reproducible across runs.
    """
    base = f'{name}__{key.ctx_fp}'
    needs_func_disambig = False
    needs_params_disambig = False
    for other_key, spec in specs.items():
        if other_key == key:
            continue
        # Look for specs whose generated base name would clash.
        # ``spec.func.name`` is the source name the base mangle is
        # built from — a clash requires both that and ``ctx_fp`` to
        # match.
        if spec.func.name != name:
            continue
        if other_key.ctx_fp != key.ctx_fp:
            continue
        # Same source name + same ctx_fp.  Decide *why* it would
        # collide: different FuncDef (cross-module name reuse) vs.
        # same FuncDef with different parameter shapes.
        if other_key.func_fp != key.func_fp:
            needs_func_disambig = True
        elif other_key.params_fp != key.params_fp:
            needs_params_disambig = True
    suffix = ''
    if needs_func_disambig:
        # The ``func_fp`` is already a stable short digest derived
        # from the callee's source location — embed it directly,
        # truncated to keep names finite.
        suffix += f'_f{key.func_fp[:4]}'
    if needs_params_disambig:
        # ``params_fp`` is a long, deterministic string; hash to a
        # short hex digest.  ``hash()`` of a string isn't
        # cross-process-stable in general (Python's hash randomizes
        # by default), so use a deterministic digest instead.
        digest = hashlib.sha1(key.params_fp.encode()).hexdigest()[:8]
        suffix += f'_p{digest}'
    return base + suffix

# ---------------------------------------------------------------------
# Compiler


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
            transformations (currently :class:`fpy2.transform.ZipElim`)
            to each :class:`FuncDef` before the rest of the pipeline
            runs.  The pipeline is sound either way, but the
            optimized form skips materializing intermediate
            ``std::vector<std::tuple<...>>``s for ``zip`` iterables.
            Set ``False`` to compile the surface AST verbatim.
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

        # Optional pre-pipeline transforms.  Applied before
        # monomorphization since they only rewrite the surface AST
        # and don't depend on type information.
        if self._optimize:
            ast = ZipElim.apply(ast)

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
        unit = self.unit()
        unit.add(func, ctx=ctx, arg_types=arg_types)
        return unit.render()

    def unit(self) -> 'CppTranslationUnit':
        """Begin a new translation-unit build.  Each :class:`Function`
        :meth:`CppTranslationUnit.add`-ed to the result shares a
        single specialization cache, so a callee invoked from
        multiple top-level functions appears exactly once in the
        emitted source — avoiding the ODR violations that a naive
        per-function :meth:`compile` loop would produce when writing
        many functions into the same ``.cpp`` file."""
        return CppTranslationUnit(self)

    # ------------------------------------------------------------------
    # Call-graph walk

    def _discover_specializations(
        self,
        format_info: FormatAnalysis,
        specs: dict['_SpecKey', '_Specialization'],
        emit_order: list['_SpecKey'],
    ) -> dict[Call, str]:
        """Walk *format_info*'s ``by_call`` recursively.  For each
        :class:`Call` whose ``fn`` is a known :class:`Function`,
        materialize (or look up) the specialization for the
        ``(callee FuncDef, outer rounding context, param formats)``
        triple.

        Returns a ``Call → mangled name`` mapping covering every Call
        in *format_info*'s function body.  Newly-discovered
        specializations append their key to *emit_order* after their
        own sub-calls finish — so the resulting order is
        callee-before-caller.
        """
        call_names: dict[Call, str] = {}
        for call_node, sub_fa in format_info.by_call.items():
            sig = sub_fa.fn_fmt
            key = _SpecKey(
                func_fp=_func_fingerprint(sub_fa.func),
                ctx_fp=_ctx_fingerprint(sig.ctx),
                params_fp=_param_fingerprint(sig.arg_fmts),
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
                mangled = _mangle_with_params(
                    sub_fa.func.name, key, specs,
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
        emitter = CppEmitter(
            ast=spec.func,
            storage=spec.storage,
            def_use=spec.format_info.type_info.def_use,
            format_info=spec.format_info,
            ctx_use=spec.format_info.ctx_use,
            func_name_override=spec.name,
            call_names=spec.call_names,
            unsafe_cast_int=self._unsafe_cast_int,
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
        emitter = CppEmitter(
            ast=result.ast,
            storage=result.storage,
            def_use=result.format_info.type_info.def_use,
            format_info=result.format_info,
            ctx_use=result.ctx_use,
            call_names=call_names,
            unsafe_cast_int=self._unsafe_cast_int,
        )
        try:
            return emitter.emit()
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{name}`: {e}'
            ) from e


# ---------------------------------------------------------------------
# Translation-unit builder


class CppTranslationUnit:
    """Accumulator for a single C++ translation unit.

    Holds a shared :class:`PreAnalysisCache` and specialization
    table across multiple :meth:`add` calls so that a callee
    referenced from several top-level functions is emitted exactly
    once.  Naive looping ``CppCompiler.compile(...)`` over each
    function and concatenating the outputs would emit a fresh copy
    of every transitive callee per call and produce ODR
    redefinition errors when the C++ compiler links the unit.

    Typical use::

        unit = compiler.unit()
        for func in module_funcs:
            try:
                unit.add(func, ctx=fp.FP64, arg_types=arg_types)
            except CppCompileError as e:
                report(func.name, e)
        f.write(unit.render())

    Per-:meth:`add` failures are isolated — the exception unwinds
    only that call's contribution.  Successful adds accumulate.
    """

    _compiler: CppCompiler
    _pre_cache: PreAnalysisCache
    _specs: dict[_SpecKey, _Specialization]
    _emit_order: list[_SpecKey]
    _top_levels: list[_TopLevel]

    def __init__(self, compiler: CppCompiler):
        self._compiler = compiler
        self._pre_cache = PreAnalysisCache()
        self._specs = {}
        self._emit_order = []
        self._top_levels = []

    def add(
        self,
        func: Function,
        *,
        ctx: Context | None = None,
        arg_types: Collection[Type | None] | None = None,
    ) -> None:
        """Register a top-level function for this translation unit.
        Equivalent to one :meth:`CppCompiler.compile` call's worth
        of pipeline + specialization discovery, but the resulting
        specializations are merged into the unit's shared table
        instead of being re-emitted independently."""
        if not isinstance(func, Function):
            raise TypeError(f'Expected `Function`, got {type(func)} for {func}')
        top_result = self._compiler._run_pipeline(
            func, ctx, arg_types, pre_cache=self._pre_cache,
        )
        top_call_names = self._compiler._discover_specializations(
            top_result.format_info, self._specs, self._emit_order,
        )
        self._top_levels.append(_TopLevel(func.name, top_result, top_call_names))

    def render(self) -> str:
        """Return the unit's C++ source: every discovered
        specialization in callee-before-caller order, followed by
        each successfully :meth:`add`-ed top-level function in
        registration order."""
        out: list[str] = [
            self._compiler._emit_specialization(self._specs[k])
            for k in self._emit_order
        ]
        for top in self._top_levels:
            out.append(
                self._compiler._emit_top_level(
                    top.name, top.result, top.call_names,
                )
            )
        return '\n\n'.join(out)
