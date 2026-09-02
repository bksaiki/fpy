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
from typing import TypeAlias

from ...analysis import (
    Alias,
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    Escape,
    FormatInfer,
    ValueClassInfer,
)
from ...analysis.alias import AliasAnalysis
from ...analysis.context_use import ContextUseAnalysis
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.escape import EscapeSummary
from ...analysis.format_infer import FormatAnalysis
from ...analysis.storage_infer import StorageInfer
from ...analysis.value_class import ValueClassAnalysis
from ...ast.fpyast import Call, FuncDef, NamedId
from ...ast.visitor import DefaultVisitor
from ...function import Function
from ...module import Module
from ...number import Context
from ...transform import (
    CompToLoop,
    EnumerateElim,
    FreeVarElim,
    Hoistable,
    ReduceFusion,
    RoundElim,
    Simplify,
    Specialize,
    ZipElim,
)
from ...transform.free_var_elim import unclosed_data_free_vars
from ...types import Type
from ..backend import Backend, CompileError
from .emitter import CppEmitError, CppEmitter
from .storage import CppStorage, CppStorageDomain, StorageSelectionError
from .types import CppType
from .unbox import (
    CalleeAbi,
    ParamAbi,
    StrictUnboxError,
    Unbox,
    UnboxAnalysis,
    UnboxMode,
    check_strict,
    return_storage,
)
from .unfold_round import UnfoldMode
from .unfold_round import unfold as unfold_round
from .utils import CPP_HEADERS, CPP_HELPERS
from .variables import VariableAlloc, VariableAnalysis

_UnboxMode: TypeAlias = UnboxMode
_UnfoldMode: TypeAlias = UnfoldMode
"""Annotation-only aliases: ``CppCompiler.UnboxMode = UnboxMode`` shadows the
enum inside the class body.  Runtime resolves the shadow to the same object;
type checkers reject an attribute used as a type, so annotations there need
this name instead."""


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
    class_info: ValueClassAnalysis
    storage: CppStorage
    variables: VariableAnalysis
    alias: AliasAnalysis
    summary: EscapeSummary
    unbox: UnboxAnalysis | None



# ---------------------------------------------------------------------
# Compiler


def _to_statement_form(fd: FuncDef) -> FuncDef:
    """Hoistable form, with every comprehension lowered.

    Neither pass is a fixpoint alone and each supplies what the other lacks.
    ``Hoistable`` seals a comprehension's element -- it runs once per iteration,
    so the slot before the enclosing statement is no place for its temporaries --
    and ``CompToLoop`` makes the loop that *is* that slot; ``CompToLoop``
    declines a comprehension in a ternary arm or a ``while`` condition for want
    of a slot, and ``Hoistable`` gives it one.

    Iterating terminates without a cap, though not because the comprehension
    count falls -- lowering a dependent clause list *raises* it, peeling one
    comprehension into a row comprehension plus a nested one.  What falls is the
    clause count of the dependent one, by one per peel, and a single-clause
    comprehension cannot be dependent.  Everything else lowers outright, and
    ``Hoistable`` is idempotent over its own output.
    """
    while True:
        fd = Hoistable.apply(fd)
        log = CompToLoop.apply_with_edits(fd)
        if not log.edits:
            return fd
        fd = log.result


def _function_calls(ast: FuncDef) -> dict[Call, Function]:
    """Every ``Call`` in *ast*'s body that targets a :class:`Function`."""
    out: dict[Call, Function] = {}

    class _Collector(DefaultVisitor):
        def _visit_call(self, e: Call, ctx):
            if isinstance(e.fn, Function):
                out[e] = e.fn
            super()._visit_call(e, ctx)

    _Collector()._visit_function(ast, None)
    return out


def _collect_call_names(ast: FuncDef) -> dict[Call, str]:
    """``Call → emit-name``.  After :class:`Specialize` each ``Call.fn`` points
    at the target spec's :class:`Function`, so the name is ``fn.ast.name``."""
    return {c: fn.ast.name for c, fn in _function_calls(ast).items()}


def _reachable_asts(entry: Function) -> set[int]:
    """ids of the ``FuncDef``s on *entry*'s call path, *entry* included.

    ``Specialize`` rewired every ``Call.fn`` to the target spec's
    :class:`Function`, so the walk is over specs directly.
    """
    seen: set[int] = set()
    stack = [entry]
    while stack:
        f = stack.pop()
        if id(f.ast) not in seen:
            seen.add(id(f.ast))
            stack.extend(_function_calls(f.ast).values())
    return seen


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
    _check_signature_monomorphic(a)
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


def _check_signature_monomorphic(a: SpecAnalyses) -> None:
    """Refuse a spec whose signature still holds a type variable -- emitting it
    needs a template, which this backend does not do.

    Storage selection will not catch this: an unresolved kind reads as
    :class:`VarFormat`, whose storage is unobservable and so answers ``bool``
    rather than refusing -- which is right for an internal type variable and
    wrong only here.  Before ``VarFormat`` existed the two were indistinguishable
    (a :class:`VarType` and a ``bool`` shared the ``None`` bound), and
    ``return []`` became ``std::vector<bool> f()``.

    Scoped to the signature, which is where the template would be needed.  A
    type variable that stays internal belongs to a value no element is stored
    into or read from, so its storage is unobservable.
    """
    fn_type = a.format_info.type_info.fn_type
    if not fn_type.is_monomorphic():
        raise CppCompileError(
            f'`{a.ast.name}` has an unresolved type in its signature '
            f'({fn_type.format()}); emitting it would need a C++ template. '
            'Annotate the type, or give the value an element to infer from.'
        )


def _return_storage(a: SpecAnalyses) -> CppType:
    return return_storage(a.format_info.fn_fmt.ret_fmt, a.unbox)


class CppCompiler(Backend):
    """Format-inference-driven C++ compiler.

    Runs the pre-analyses, assigns each SSA def a C++ variable, then emits.

    Emitted code has one precondition on its callers: enter a kernel with the
    ``fesetround`` mode its top-level context names, or ``FE_TONEAREST`` when it
    names none (``REAL``, an integer context, or no annotation).  A kernel sets
    no mode at entry and restores what it found before returning.

    Args:
        unsafe_cast_int:
            Allow rounded arithmetic under an unbounded-integer context by
            casting to ``int64_t`` and assuming no overflow.  ``False`` rejects
            such programs instead.  Default ``True``.
        optimize:
            Run the optimizing transforms listed in :meth:`specialize`.  Sound
            either way; ``False`` skips them.  It does *not* mean the surface AST
            reaches the emitter untouched: ``FreeVarElim`` and
            :func:`_to_statement_form` run regardless.  Default ``True``.
        unbox:
            An :class:`~fpy2.backend.cpp.unbox.UnboxMode` (also reachable as
            ``CppCompiler.UnboxMode``).  ``ALLOW`` drops the handle where
            :mod:`.unbox` proves nothing observes the difference, per list and
            per nesting level; ``NEVER`` keeps every handle -- correct, but
            slower at a native boundary; ``STRICT`` is like ``ALLOW``, but a
            list that must keep its handle fails the compile.  Default
            ``STRICT``: a ``std::shared_ptr`` in numerical C++ is surprising
            enough that it has to be asked for.
        arrays:
            Compile an unboxed list whose length is statically proven to
            ``std::array<T, K>`` instead of ``std::vector<T>``.  Purely a
            representation choice -- same values, same element order -- but it
            does shape signatures: an entry whose ``arg_types`` carry a
            ``ListType`` *length* gets an array parameter, and a trusted
            ``assert len(xs) == K`` becomes a type-level commitment.  No effect
            under ``unbox=NEVER``, where nothing is a value.  Default ``True``.
        unfold:
            An :class:`~fpy2.backend.cpp.unfold_round.UnfoldMode` (also
            reachable as ``CppCompiler.UnfoldMode``).  ``ROUNDINGS`` lowers a
            rounding the op table cannot spell into integer arithmetic instead
            of refusing it; ``DOUBLE_ROUND`` also computes arithmetic under
            such a context at a native intermediate and re-rounds, where the
            correct-double-rounding rules say the two compose to what the one
            gave.  Default ``NONE``: the refusal is a checker's answer, and
            turning the compiler into a rewriter has to be asked for.
    """

    UnboxMode = UnboxMode
    UnfoldMode = UnfoldMode
    """The mode enums for ``unbox`` and ``unfold``, re-exported so callers
    holding the compiler need not import the modules defining them."""

    _unsafe_cast_int: bool
    _optimize: bool
    _unbox: _UnboxMode
    _unfold: _UnfoldMode
    _arrays: bool

    def __init__(
        self, *, unsafe_cast_int: bool = True, optimize: bool = True,
        unbox: _UnboxMode = UnboxMode.STRICT, arrays: bool = True,
        unfold: _UnfoldMode = UnfoldMode.NONE,
    ):
        if not isinstance(unbox, UnboxMode):
            raise TypeError(
                f'`unbox` must be an UnboxMode, got {unbox!r}; '
                'use UnboxMode.ALLOW / UnboxMode.NEVER instead of a bool'
            )
        if not isinstance(unfold, UnfoldMode):
            raise TypeError(
                f'`unfold` must be an UnfoldMode, got {unfold!r}; '
                'use UnfoldMode.ROUNDINGS / UnfoldMode.DOUBLE_ROUND '
                'instead of a bool'
            )
        self._unsafe_cast_int = unsafe_cast_int
        self._optimize = optimize
        self._unbox = unbox
        self._arrays = arrays
        self._unfold = unfold

    # ------------------------------------------------------------------
    # Translation-unit preamble.  ``compile`` returns a function definition
    # only, so single-function tests can use exact-string equality.

    def headers(self) -> list[str]:
        """C++ headers required by every emitted unit."""
        return list(CPP_HEADERS)

    def helpers(self) -> str:
        """Support code an emitted unit needs: currently none.

        See :data:`.utils.CPP_HELPERS`."""
        return CPP_HELPERS

    def prelude(self) -> str:
        """The headers and helpers concatenated, ready to prepend."""
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
        """Compile *func* to a C++ source string.

        A thin wrapper around :meth:`compile_module` over a one-entry module, so the
        single-function and module paths share one pipeline.
        """
        m = Module()
        m.add(func, ctx=ctx, arg_types=arg_types)
        return self.compile_module(m)

    def compile_module(self, module: Module) -> str:
        """Compile a :class:`~fpy2.Module` to a single C++ translation unit.

        Pre-spec optimizations, then ``Specialize`` -- one entry per
        ``(FuncDef, ctx, arg_fmts)``, with calls rewired -- then post-spec
        optimizations now that format inference is monomorphic, then codegen
        leaves-first.
        """
        try:
            return self._compile_module(module)
        except CppCompileError:
            if self._unfold is UnfoldMode.NONE:
                raise
            # The rewrite left a program that still does not compile, and it
            # fails further along: a rounding the emitter could name became a
            # temporary storage selection cannot place.  Report what the
            # unrewritten program says, so the flag never costs a diagnosis.
            self._without_unfold()._compile_module(module)
            raise   # it compiled unrewritten, so the rewrite's own error stands

    def _without_unfold(self) -> 'CppCompiler':
        """This compiler with the rewrite off, for a second opinion."""
        return CppCompiler(
            unsafe_cast_int=self._unsafe_cast_int, optimize=self._optimize,
            unbox=self._unbox, arrays=self._arrays, unfold=UnfoldMode.NONE,
        )

    def _compile_module(self, module: Module) -> str:
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
        only: set[int] | None = None,
    ):
        """Analyze every spec leaves-first, filling *params* as it goes.

        One path, so :meth:`compile_module` and :meth:`signature` cannot reach
        different conclusions about the same function — they did once, and it
        was an ABI bug rather than a missed optimization.

        *only* limits which specs are analyzed (by ``id(f.ast)``) without
        narrowing what counts as *called*, so :meth:`signature` can skip a
        spec the entry never reaches — whose failures are not the entry's —
        while an entry that other functions call keeps its boundary ABI.
        """
        called = {id(c.ast) for f in specs for c in _function_calls(f.ast).values()}
        summaries: dict[FuncDef, EscapeSummary] = {}
        for f in specs:
            if only is not None and id(f.ast) not in only:
                continue
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

        # required: codegen has no closure environment
        module = module.map(lambda _m, fd: FreeVarElim.apply(fd))

        if self._optimize:
            # before ZipElim: `enumerate(zip(...))` must be matched while the
            # `zip` is still in an iterable position
            module = module.map(lambda _m, fd: EnumerateElim.apply(fd))
            module = module.map(lambda _m, fd: ZipElim.apply(fd))
            # after both, so a `zip`/`enumerate` comp is already an indexed comp
            module = module.map(lambda _m, fd: ReduceFusion.apply(fd))

        # `size_key`: a spec per distinct argument-length vector, so a proven
        # length crosses the call edge as the callee's annotation and both ends
        # agree on `std::array` (see `.unbox`).  `Monomorphize` raises a bare
        # `RuntimeError`, which callers iterating over candidates cannot catch
        # uniformly.
        try:
            specialized = Specialize.apply(module, size_key=self._arrays)
        except RuntimeError as e:
            raise CppCompileError(f'specialization failed: {e}') from e

        # Before `RoundElim`, whose hoist is suppressed in two of the positions
        # this gives a slot.
        specialized = specialized.map(lambda _m, fd: _to_statement_form(fd))

        if self._optimize:
            specialized = specialized.map(lambda _m, fd: RoundElim.apply(fd))

        if self._unfold is not UnfoldMode.NONE:
            # after `RoundElim`, which removes roundings this would otherwise
            # lower, and re-normalized after: the lowering emits `with` blocks
            # and branches of its own.
            mode = self._unfold
            specialized = specialized.map(lambda _m, fd: unfold_round(fd, mode))
            specialized = specialized.map(lambda _m, fd: _to_statement_form(fd))

        if self._optimize:
            # Last, and after everything that names: the lowerings above leave
            # debris only a later pass can see -- a length read into a name
            # nothing goes on to use, a copy of an accumulator.
            specialized = specialized.map(lambda _m, fd: Simplify.apply(fd))

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
        class_info = ValueClassInfer.analyze(
            ast, type_info=format_info.type_info, ctx_use=ctx_use,
        )

        try:
            du = format_info.type_info.def_use
            chosen = StorageInfer.infer(
                du, format_info.by_def, format_info.by_expr,
                CppStorageDomain(),
            )
        except StorageSelectionError as e:
            raise CppCompileError(
                f'storage selection failed for `{func.name}`: {e}'
            ) from e
        except AssertionError as e:
            # An invariant the analysis expects an earlier phase to hold.
            # Named, so it does not reach the caller as a bare AssertionError
            # mentioning neither the function nor the backend.
            raise CppCompileError(
                f'storage selection failed for `{func.name}`: '
                f'internal error: {e!r}'
            ) from e

        # the analysis answers in formats; this is the target's spelling
        storage = CppStorage(chosen)
        variables = VariableAlloc.assign(du, storage)

        alias = Alias.analyze(ast, def_use=def_use, summaries=summaries)
        # This function's own summary, from the alias analysis it already has.
        # Its callers read it to decide whether they can stop treating an
        # argument as shared; it reads its own to decide the same about its
        # parameters, so both ends reach the same answer.
        summary = Escape.analyze(
            ast, summaries, def_use=def_use, alias=alias,
        )
        unbox = None
        if self._unbox is not UnboxMode.NEVER:
            unbox = Unbox.decide(
                ast, storage, variables, alias, def_use,
                is_called=is_called,
                summary=summary,
                callees=callee_abis,
                array_size=array_size if self._arrays else None,
            )
            unbox.strict = self._unbox is UnboxMode.STRICT
            # Rewrite each class's storage in place: the emitter reads a
            # declaration's representation straight off the type.
            storage.class_storage.update(unbox.storage)

        # The return type is one more place storage is chosen, and the only
        # one `StorageInfer` does not cover (e.g. a REAL-format return).
        # Checked here for every mode so each entry point reports the same
        # error -- `signature` has no emission step to catch it later.
        try:
            ret_ty = return_storage(format_info.fn_fmt.ret_fmt, unbox)
        except StorageSelectionError as e:
            raise CppCompileError(
                f'storage selection failed for `{func.name}`: {e}'
            ) from e

        if unbox is not None and unbox.strict:
            try:
                check_strict(unbox, storage, variables, ret_ty)
            except StrictUnboxError as e:
                raise CppCompileError(
                    f'strict unboxing failed for `{func.name}`: {e}'
                ) from e

        return SpecAnalyses(
            ast=ast,
            def_use=def_use,
            ctx_use=ctx_use,
            format_info=format_info,
            class_info=class_info,
            storage=storage,
            variables=variables,
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

        Not derivable from FPy types alone: representation depends on :mod:`.unbox`,
        so one FPy signature can compile to either a shared handle or
        ``std::vector<T>``.  Pass the *module* when there is one -- a function that
        compiled code calls keeps its handles.
        """
        if module is None:
            module = Module()
            module.add(func, ctx=ctx, arg_types=arg_types)
        specs = self.specialize(module)
        entry = _find_spec(specs, func)
        emitted: dict[FuncDef, CalleeAbi] = {}
        # Only the entry's call path: an unrelated spec's failure (routine
        # under STRICT) must not decide -- by `Module.add` order, no less --
        # whether the entry has a signature.
        a = next(
            an for f, an in self._analyze_all(
                specs, emitted, only=_reachable_asts(entry),
            ) if f is entry
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
            variables=a.variables,
            def_use=a.def_use,
            format_info=a.format_info,
            class_info=a.class_info,
            ctx_use=a.ctx_use,
            call_names=call_names,
            unsafe_cast_int=self._unsafe_cast_int,
            unbox=a.unbox,
            callee_params=callee_params,
        )
        try:
            # Under STRICT the emitter carries its own tripwire: every handle
            # spelling branches on `_is_boxed`, which refuses -- so a leak
            # past `check_strict` and `annotate` is a `CppEmitError` naming a
            # backend bug, not a `std::shared_ptr` in the output.
            return emitter.emit()
        except StrictUnboxError as e:
            # `annotate` refused an expression temporary -- a user program,
            # not a bug, so it presents like `check_strict`'s refusals.
            raise CppCompileError(
                f'strict unboxing failed for `{func.name}`: {e}'
            ) from e
        except CppEmitError as e:
            raise CppCompileError(
                f'compilation failed for `{func.name}`: {e}'
            ) from e
