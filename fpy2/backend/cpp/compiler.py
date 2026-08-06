"""
Public API for the cpp backend.

:class:`CppCompiler` runs the analysis pipeline (monomorphization,
def-use, context-use, array-size, format inference, storage
inference) on a :class:`Function` and hands the result to
:class:`_CppEmitter`, which produces a C++ source string.  Errors
surface as :class:`CppCompileError`.
"""

from collections.abc import Collection
from dataclasses import dataclass

from ...analysis import (
    Alias,
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    Escape,
    FormatInfer,
)
from ...analysis.alias import AliasAnalysis
from ...analysis.context_use import ContextUseAnalysis
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.escape import EscapeSummary
from ...analysis.format_infer import FormatAnalysis
from ...ast.fpyast import Call, FuncDef, NamedId
from ...ast.visitor import DefaultVisitor
from ...function import Function
from ...module import Module
from ...number import Context
from ...transform import (
    EnumerateElim,
    FreeVarElim,
    ReduceFusion,
    RoundElim,
    Specialize,
    ZipElim,
)
from ...transform.free_var_elim import unclosed_data_free_vars
from ...types import Type
from ..backend import Backend, CompileError
from .emitter import CppEmitError, CppEmitter
from .storage import StorageSelectionError, choose_storage, return_storage
from .storage_infer import StorageAnalysis, StorageInfer
from .types import CppType
from .unbox import CalleeAbi, ParamAbi, Unbox, UnboxAnalysis
from .utils import CPP_HEADERS, CPP_HELPERS


class CppCompileError(CompileError):
    """Raised when cpp compilation fails."""


@dataclass
class SpecAnalyses:
    """The analyses one fully-specialized function is emitted from.

    ``ast`` is post-specialization, so its types are concrete — which is why a
    consumer cannot simply re-run the analyses on the user's original function.
    """

    ast: FuncDef
    def_use: DefineUseAnalysis
    ctx_use: ContextUseAnalysis
    format_info: FormatAnalysis
    storage: StorageAnalysis
    alias: AliasAnalysis
    summary: EscapeSummary
    unbox: UnboxAnalysis | None



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


def _find_spec(specs: list[Function], func: Function) -> Function:
    """The specialization of *func* among *specs*.

    A public entry keeps its user-given name through ``Specialize``; if it is
    the only spec there is nothing to match against anyway.
    """
    for s in specs:
        if s.ast.name == func.ast.name:
            return s
    if len(specs) == 1:
        return specs[0]
    raise CppCompileError(
        f'`{func.name}` has no specialization in this module'
    )


def _callee_abi(a: SpecAnalyses) -> CalleeAbi:
    """One spec's signature, as its callers must see it.

    All three facts cross a call edge: an argument must have the representation
    the callee declared and must be non-const if the callee writes it, and a
    result must have the representation the callee returns.
    """
    params: list[ParamAbi] = []
    for arg in a.ast.args:
        if not isinstance(arg.name, NamedId):
            raise CppCompileError(f'unnamed parameter in `{a.ast.name}`')
        d = a.def_use.find_def_from_site(arg.name, arg)
        ty = a.storage.storage_of(d)
        written = a.unbox is not None and a.unbox.writes_through(
            a.alias.region_of(d), ty,
        )
        params.append(ParamAbi(ty, written))
    return CalleeAbi(params, _return_storage(a))


def _return_storage(a: SpecAnalyses) -> CppType:
    return return_storage(a.format_info.fn_fmt.ret_fmt, a.unbox)


def _callees(ast: FuncDef) -> list[Function]:
    """Every :class:`Function` *ast* calls."""
    out: list[Function] = []

    class _Collector(DefaultVisitor):
        def _visit_call(self, e: Call, ctx):
            if isinstance(e.fn, Function):
                out.append(e.fn)
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

            - :class:`fpy2.transform.EnumerateElim` (pre-monomorphize):
              skips materializing intermediate
              ``std::vector<std::tuple<...>>``s for ``enumerate``
              iterables.  Must run before ``ZipElim``: it also handles
              ``enumerate(zip(...))``, collapsing both intermediate
              vectors at once, which ``ZipElim`` could no longer do
              once the ``zip`` sits in an assignment.
            - :class:`fpy2.transform.ZipElim` (pre-monomorphize):
              skips materializing intermediate
              ``std::vector<std::tuple<...>>``s for ``zip`` iterables.
            - :class:`fpy2.transform.ReduceFusion` (pre-monomorphize):
              folds ``any``/``all`` over a comprehension into one loop,
              skipping the intermediate ``std::vector<bool>``.
            - :class:`fpy2.transform.RoundElim` (post-monomorphize):
              hoists eliminable rounded operations into
              ``with fp.REAL:`` blocks so the cpp emitter's
              lossless-widening dispatch can pick tighter storage
              for them.

            The pipeline is sound either way.  Set ``False`` to
            compile the surface AST verbatim.
        unbox:
            When ``True`` (default), represent a list as a plain
            ``std::vector`` wherever :mod:`fpy2.analysis.alias` proves
            nothing can observe the difference (see :mod:`.unbox`).
            The choice is per list and per nesting level.  ``False``
            keeps every handle -- always correct, but slower at a
            native boundary.
    """

    _unsafe_cast_int: bool
    _optimize: bool
    _unbox: bool

    def __init__(
        self, *, unsafe_cast_int: bool = True, optimize: bool = True,
        unbox: bool = True,
    ):
        self._unsafe_cast_int = unsafe_cast_int
        self._optimize = optimize
        self._unbox = unbox

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
          1. **Pre-spec optimizations** (``EnumerateElim``, ``ZipElim``,
             ``ReduceFusion``) on every function in the module via ``map``.
          2. **Specialize** the module: each ``(FuncDef, ctx, arg_fmts)``
             becomes one entry; cross-function calls rewire to the
             appropriate spec.
          3. **Post-spec optimizations** (``RoundElim``) on each spec —
             monomorphic format inference is now available.
          4. **Per-spec codegen**, leaves-first, one C++ definition per entry.
        """
        specs = self.specialize(module)
        params: dict[FuncDef, CalleeAbi] = {}
        return '\n\n'.join(
            self._emit(f, a, params)
            for f, a in self._analyze_all(specs, params)
        )

    def _analyze_all(
        self,
        specs: list[Function],
        params: dict[FuncDef, CalleeAbi],
    ):
        """Analyze every spec leaves-first, filling *params* as it goes.

        One path, so :meth:`compile_module` and :meth:`signature` cannot reach
        different conclusions about the same function — they did once, and it
        was an ABI bug rather than a missed optimization.
        """
        called = {id(c.ast) for f in specs for c in _callees(f.ast)}
        summaries: dict[FuncDef, EscapeSummary] = {}
        for f in specs:
            a = self.analyze(
                f, is_called=id(f.ast) in called, summaries=summaries,
                callee_abis=params,
            )
            yield f, a
            summaries[f.ast] = a.summary
            params[f.ast] = _callee_abi(a)

    def specialize(self, module: Module) -> list[Function]:
        """Steps 1-3 of the pipeline: the fully-specialized functions, in
        leaves-first emission order."""
        if not isinstance(module, Module):
            raise TypeError(f'Expected `Module`, got {type(module)} for {module}')

        # Close each function over its captured data free variables (a
        # correctness pass — codegen has no closure environment).
        module = module.map(lambda _m, fd: FreeVarElim.apply(fd))

        if self._optimize:
            # before ZipElim: `enumerate(zip(...))` must be matched while the
            # `zip` is still in an iterable position
            module = module.map(lambda _m, fd: EnumerateElim.apply(fd))
            module = module.map(lambda _m, fd: ZipElim.apply(fd))
            # after both, so a `zip`/`enumerate` comp is already an indexed comp
            module = module.map(lambda _m, fd: ReduceFusion.apply(fd))

        # Translate ``Monomorphize``'s bare ``RuntimeError`` (e.g. arg-type
        # mismatches) into ``CppCompileError`` so callers iterating over
        # candidate functions can catch a uniform error type.
        try:
            specialized = Specialize.apply(module)
        except RuntimeError as e:
            raise CppCompileError(f'specialization failed: {e}') from e

        if self._optimize:
            specialized = specialized.map(lambda _m, fd: RoundElim.apply(fd))

        return list(specialized.call_graph().order)

    def analyze(
        self,
        func: Function,
        *,
        is_called: bool = False,
        summaries: dict[FuncDef, EscapeSummary] | None = None,
        callee_abis: dict[FuncDef, CalleeAbi] | None = None,
    ) -> SpecAnalyses:
        """The per-spec analyses one fully-specialized function is emitted
        from."""
        ast = func.ast
        if bad := unclosed_data_free_vars(ast):
            raise CppCompileError(
                f'unbound data free variable(s): {", ".join(bad)}'
            )

        def_use = DefineUse.analyze(ast)
        ctx_use = ContextUse.analyze(ast, def_use=def_use)
        array_size = ArraySizeInfer.analyze(ast)
        format_info = FormatInfer.analyze(
            ast,
            def_use=def_use,
            ctx_use=ctx_use,
            array_size=array_size,
        )

        try:
            du = format_info.type_info.def_use
            storage = StorageInfer.infer(du, format_info.by_def)
        except StorageSelectionError as e:
            raise CppCompileError(
                f'storage selection failed for `{func.name}`: {e}'
            ) from e
        except Exception as e:
            # An internal invariant failure in `storage.py` (e.g. an `assert`
            # in `_supremum`) would otherwise reach the caller as a bare
            # AssertionError naming neither the function nor the backend.
            raise CppCompileError(
                f'storage selection failed for `{func.name}`: '
                f'internal error: {e!r}'
            ) from e

        alias = Alias.analyze(ast, def_use=def_use, summaries=summaries)
        # This function's own summary, from the alias analysis it already has.
        # Its callers read it to decide whether they can stop treating an
        # argument as shared; it reads its own to decide the same about its
        # parameters, so both ends reach the same answer.
        summary = Escape.analyze(
            ast, summaries, def_use=def_use, alias=alias,
        )
        unbox = None
        if self._unbox:
            unbox = Unbox.decide(
                ast, storage, alias, def_use,
                is_called=is_called,
                summary=summary,
                callees=callee_abis,
            )
            # Rewrite each class's storage in place: the emitter reads a
            # declaration's representation straight off the type.
            storage.class_storage.update(unbox.storage)

        return SpecAnalyses(
            ast=ast,
            def_use=def_use,
            ctx_use=ctx_use,
            format_info=format_info,
            storage=storage,
            alias=alias,
            summary=summary,
            unbox=unbox,
        )

    def signature(
        self,
        func: Function,
        *,
        ctx: Context | None = None,
        arg_types: Collection[Type | None] | None = None,
        module: Module | None = None,
    ) -> tuple[list[CppType], CppType]:
        """The C++ storage types of *func*'s parameters and result.

        Not derivable from FPy types alone: how a list is represented depends on
        :mod:`.unbox`, so the same FPy signature can compile to ``fpy::list<T>``
        or ``std::vector<T>``.  Pass the *module* being compiled whenever there
        is one — a function another compiled function calls keeps its handles.
        """
        if module is None:
            module = Module()
            module.add(func, ctx=ctx, arg_types=arg_types)
        specs = self.specialize(module)
        entry = _find_spec(specs, func)
        emitted: dict[FuncDef, CalleeAbi] = {}
        a = next(
            an for f, an in self._analyze_all(specs, emitted) if f is entry
        )

        abi = _callee_abi(a)
        return [p.ty for p in abi.params], abi.ret

    def _compile_function(
        self, func: Function, *, is_called: bool = False,
    ) -> str:
        """Emit one C++ function definition for a fully-specialized
        :class:`Function`.  ``func.ast.name`` is the final emitted name
        (set by :class:`Specialize` — public entries keep their user-given
        name, private specs get a mangled one)."""
        return self._emit(func, self.analyze(func, is_called=is_called), {})

    def _emit(
        self,
        func: Function,
        a: SpecAnalyses,
        callee_params: dict[FuncDef, CalleeAbi],
    ) -> str:
        ast = a.ast

        # Call.fn → emitted name.  ``Specialize`` rewired each Call.fn at
        # the source so call.fn.ast.name is the target spec's emit name.
        call_names = _collect_call_names(ast)

        emitter = CppEmitter(
            ast=ast,
            storage=a.storage,
            def_use=a.def_use,
            format_info=a.format_info,
            ctx_use=a.ctx_use,
            call_names=call_names,
            unsafe_cast_int=self._unsafe_cast_int,
            unbox=a.unbox,
            callee_params=callee_params,
        )
        try:
            return emitter.emit()
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{func.name}`: {e}'
            ) from e

    # ------------------------------------------------------------------
    # Call-graph walk

