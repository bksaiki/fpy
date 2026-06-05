"""
cpp backend: emitter.

Walks the (post-pipeline) :class:`FuncDef` and produces a C++ source
string.  Storage types and per-def C++ identifiers come from
:class:`CppPipelineResult.storage` (a :class:`StorageAnalysis`
produced by :class:`StorageInfer`); per-expression bounds come from
:class:`FormatAnalysis`.

Currently supported:

- Function signatures, ``Assign`` (with ``TupleBinding``
  destructuring), ``Return``, numeric literals, comparisons and
  booleans.
- Control flow: ``if`` / ``if1`` / ``while`` / ``for``-over-range
  and ``for``-over-list (with optional tuple-binding target).
- Lists: literals, indexing, slicing, comprehensions (with optional
  tuple-binding target), and ``IndexedAssign`` as direct in-place
  subscript-stores.
- Tuples: ``TupleExpr`` and tuple-binding destructuring everywhere.
- Built-ins: ``len``, ``sum``, ``enumerate``, ``zip``, ``range``.
- Primitive arithmetic / transcendental ops dispatched through
  the :class:`ScalarOpTable` from :mod:`.ops`, which validates
  the operand formats against the active rounding context and
  inserts explicit ``static_cast`` conversions.
- Rounding-context boundaries: ``with`` blocks emit save / set /
  restore around ``fesetround`` when the active mode changes;
  ``Round`` / ``Cast`` lower as
  ``static_cast`` (with a NaN-aware assertion for ``Cast``).

Anything else raises :class:`CppEmitError`, which the public
``CppCompiler`` re-wraps as :class:`CppCompileError`.
"""

from contextlib import contextmanager
from fractions import Fraction

from ...analysis import (
    ContextScope, ContextScopeSite, ContextUseSite, ContextUseAnalysis,
    DefineUseAnalysis, FormatAnalysis
)
from ...ast.fpyast import (
    And, Argument, Assign, AssertStmt, Ast, BinaryOp, BoolVal, Cast, Compare,
    ContextStmt, Decnum, Digits, Dim, EffectStmt, Empty, Enumerate, Expr,
    ForStmt, FuncDef, Hexnum, If1Stmt, IfStmt, IndexedAssign, Integer,
    IsFinite, IsInf, IsNan, IsNormal, Len, ListComp, ListExpr, ListRef,
    ListSlice, Max, Min, NamedId, NaryOp, Not, Or, Range1, Range2, Range3,
    Rational, ReturnStmt, Round, Signbit, Size, StmtBlock, Sum,
    TernaryOp, TupleBinding, TupleExpr, UnaryOp, UnderscoreId, Var,
    WhileStmt, Zip,
)
from ...ast.visitor import Visitor
from ...number import EFloatContext, MPFixedContext, MPBFixedContext, REAL, RM
from ...number.context.context import Context

from ...analysis.format_infer import (
    AbstractFormat, AbstractableFormat, SetFormat, round_is_identity,
)

from .ops import BinaryCppOp, ScalarOpTable, TernaryCppOp, UnaryCppOp
from .target import make_op_table
from .storage import (
    StorageSelectionError, choose_storage, scalar_fits_in, scalar_sup,
)
from .storage_infer import StorageAnalysis
from .types import CppList, CppScalar, CppTuple, CppType


# Map FPy rounding modes to ``<cfenv>`` macros.  Only the four modes
# in this table can be set via ``fesetround``.
_FE_RM_MACRO: dict[RM, str] = {
    RM.RNE: 'FE_TONEAREST',
    RM.RTZ: 'FE_TOWARDZERO',
    RM.RTP: 'FE_UPWARD',
    RM.RTN: 'FE_DOWNWARD',
}


def _list_depth(ty: CppType) -> int:
    """Number of nested ``CppList`` layers in *ty*.

    Implements the ``dim(xs)`` semantics: a flat ``vector<T>`` has
    depth 1, ``vector<vector<T>>`` has depth 2, etc.  Scalars and
    tuples count as 0 — ragged shapes are out of scope (the FPy
    semantics for ``dim`` assume a non-ragged tensor)."""
    depth = 0
    while isinstance(ty, CppList):
        depth += 1
        ty = ty.elt
    return depth


class _IndentedWriter:
    """Tiny line-oriented C++ source builder."""

    def __init__(self):
        self._lines: list[str] = []
        self._depth = 0

    def add_line(self, line: str = ''):
        if line:
            self._lines.append('    ' * self._depth + line)
        else:
            self._lines.append('')

    def indent(self):
        self._depth += 1

    def dedent(self):
        self._depth -= 1

    def render(self) -> str:
        return '\n'.join(self._lines)


class CppEmitError(Exception):
    """Raised for unsupported AST shapes during emission.

    Optionally carries an ``at`` :class:`Ast` node — the
    source-location of that node is prepended to the error message
    when present, so error output points at the offending FPy
    construct instead of leaving the user to guess.

    The wrapping :class:`CppCompileError` in :mod:`compiler` builds
    its message from ``str(e)`` of the underlying :class:`CppEmitError`,
    so the location prefix flows through untouched.
    """

    def __init__(self, msg: str, *, at: 'Ast | None' = None):
        self.msg = msg
        self.at = at
        loc = at.loc if at is not None else None
        if loc is not None:
            super().__init__(f'{loc.format()}: {msg}')
        else:
            super().__init__(msg)


class CppEmitter(Visitor):
    """Single-use visitor that produces a C++ source string."""

    ast: FuncDef
    storage: StorageAnalysis
    def_use: DefineUseAnalysis
    format_info: FormatAnalysis
    ctx_use: ContextUseAnalysis
    writer: _IndentedWriter

    def __init__(
        self,
        ast: FuncDef,
        storage: StorageAnalysis,
        def_use: DefineUseAnalysis,
        format_info: FormatAnalysis,
        ctx_use: ContextUseAnalysis,
        *,
        func_name_override: str | None = None,
        call_names: dict | None = None,
        unsafe_cast_int: bool = False,
    ):
        self.ast = ast
        self.storage = storage
        self.def_use = def_use
        self.format_info = format_info
        self.ctx_use = ctx_use
        # Optional C++ name to emit at the function-signature site
        # — used by the compiler to differentiate specializations of
        # the same callee at distinct rounding contexts (template-
        # style monomorphization).  When ``None``, the AST's declared
        # name is used.
        self._func_name_override = func_name_override
        # Mapping from each :class:`Call` AST node inside this
        # function to the mangled C++ name of its target.  The
        # compiler builds this map by walking ``format_info.by_call``
        # and dispensing a stable mangled name per
        # ``(callee FuncDef, outer_ctx)`` pair.  Falls back to the
        # callee's declared name when a Call isn't in the map (e.g.
        # foreign function values).
        self._call_names: dict = call_names or {}
        # When True, allow rounded arithmetic to dispatch under an
        # unbounded-integer context (truncating silently to
        # ``int64_t``).  Forwarded from
        # :attr:`CppCompiler.unsafe_cast_int`; defaults to ``False``.
        self._unsafe_cast_int = unsafe_cast_int
        self.writer = _IndentedWriter()
        self._tmp_counter = 0
        self.op_table: ScalarOpTable = make_op_table()
        # Build a site → scope lookup over the analysis's scope list.
        self._scope_by_site: dict[ContextScopeSite, ContextScope] = {
            scope.site: scope for scope in ctx_use.scopes
        }
        # Track the rounding mode in effect at the current emission
        # point.  ``None`` means *unknown* — that's the safe initial
        # state (the C++ caller's ``fenv`` could be anything) and the
        # state we restore to after a saved scope exits.  When we
        # transition from an unknown mode to any concrete mode, the
        # ``_fenv_scope`` helper always emits ``fesetround`` so we
        # never operate under a rounding mode we don't know.  When
        # the active mode is known and matches the target, we skip
        # the ``fesetround`` to keep simple ``with FP64:`` blocks
        # under an FP64-RNE function free of fenv noise.
        self._current_rm: RM | None = None

    def _fresh_temp(self) -> str:
        """Allocate a fresh emitter-only temporary identifier.

        Used by visitors that need to emit setup statements alongside
        an expression result — comprehensions, slices, tuple-binding
        destructure sources, ``enumerate``, ``zip``, etc.  The
        ``__cpp_tmp`` prefix keeps these distinct from any identifier
        the source program could introduce (FPy doesn't allow leading
        underscores in user-visible names).
        """
        self._tmp_counter += 1
        return f'__cpp_tmp{self._tmp_counter}'

    # ------------------------------------------------------------------
    # Public entry

    def emit(self) -> str:
        self._visit_function(self.ast, None)
        return self.writer.render()

    # ------------------------------------------------------------------
    # Helpers

    def _name_for_var_use(self, var: Var) -> str:
        d = self.def_use.find_def_from_use(var)
        return self.storage.def_to_name[d]

    def _name_for_def_at_site(self, name: NamedId, site) -> str:
        d = self.def_use.find_def_from_site(name, site)
        return self.storage.def_to_name[d]

    def _storage_for_arg(self, arg: Argument) -> CppType:
        assert isinstance(arg.name, NamedId)
        d = self.def_use.find_def_from_site(arg.name, arg)
        return self.storage.storage_of(d)

    def _emit_bind(self, name: NamedId, site, rhs: str) -> None:
        """Emit a single ``T name = rhs;`` (declare-on-assign) or
        ``name = rhs;`` (reassign) line for a NamedId target whose
        SSA def is registered at *site*.

        Whether to declare or reassign is decided by the
        :class:`StorageAnalysis`."""
        target_def = self.def_use.find_def_from_site(name, site)
        target_name = self.storage.def_to_name[target_def]
        if target_def in self.storage.declare_at_assign:
            storage = self.storage.storage_of(target_def)
            self.writer.add_line(f'{storage.format()} {target_name} = {rhs};')
        else:
            self.writer.add_line(f'{target_name} = {rhs};')

    def _destructure(
        self,
        binding: TupleBinding,
        src: str,
        site,
    ) -> None:
        """Emit destructuring assigns extracting each element of
        *binding* from the tuple-valued local *src*.  The SSA defs
        for every NamedId in *binding* are registered at *site* (the
        enclosing :class:`Assign` / :class:`ForStmt` / :class:`ListComp`
        node).  Underscore positions are skipped; nested tuple bindings
        recurse via a fresh temp."""
        for i, elt in enumerate(binding.elts):
            match elt:
                case UnderscoreId():
                    continue
                case NamedId():
                    access = f'std::get<{i}>({src})'
                    self._emit_bind(elt, site, access)
                case TupleBinding():
                    access = f'std::get<{i}>({src})'
                    sub_tmp = self._fresh_temp()
                    self.writer.add_line(f'auto {sub_tmp} = {access};')
                    self._destructure(elt, sub_tmp, site)
                case _:
                    raise CppEmitError(
                        f'unsupported tuple-binding element {elt!r}',
                        at=binding,
                    )

    def _storage_for_expr(self, e: Expr) -> CppType:
        """The C++ storage chosen for an expression's result.

        Falls back to ``CppEmitError`` if format inference produced
        nothing storable (e.g., a symbolic ``REAL_FORMAT``).
        """
        fmt = self.format_info.by_expr.get(e)
        try:
            return choose_storage(fmt)
        except StorageSelectionError as err:
            raise CppEmitError(
                f'cannot pick storage for {type(e).__name__}: {err}',
                at=e,
            ) from err

    # ------------------------------------------------------------------
    # Function emission

    def _visit_function(self, func: FuncDef, ctx):
        # Determine return type from the return statement's expression
        # bound (the slice assumes a single return, possibly nested in a
        # ``with`` block).
        ret_ty = self._infer_return_storage(func)
        ret_str = ret_ty.format() if ret_ty is not None else 'void'

        # Emit arg list.  Each argument's class is anchored to the bare
        # source name in ``StorageInfer``, so it's safe to use ``arg.name``
        # directly here (and any body-side reassignment that flows
        # through the arg's phi-class will write to the same variable).
        arg_strs: list[str] = []
        for arg in func.args:
            if not isinstance(arg.name, NamedId):
                raise CppEmitError(
                    f'unsupported arg pattern: {arg.name!r}', at=arg,
                )
            storage = self._storage_for_arg(arg)
            arg_strs.append(f'{storage.format()} {arg.name}')
        emitted_name = self._func_name_override or func.name
        sig = f'{ret_str} {emitted_name}({", ".join(arg_strs)})'

        self.writer.add_line(sig + ' {')
        self.writer.indent()
        # The function-level scope determines the rounding mode in
        # effect at function entry — but only when something actually
        # *uses* the context (a primitive op dispatching under it).
        # A function that does only context-free work (bool returns,
        # tuple shuffling, ops fully nested in inner ``with`` blocks)
        # doesn't need its outer scope to be supported.  See
        # :meth:`_resolve_used_ctx`.
        #
        # ``self._current_rm`` tracks the rounding mode the live
        # ``fenv`` is guaranteed to hold.  For a concrete FP
        # function-level scope the FPy contract says the caller
        # delivers that RM, so :meth:`_entry_rm` returns it and the
        # shared :meth:`_fenv_scope` skips an entry ``fesetround``
        # when the body's used context already matches.  Symbolic /
        # integer / unsupported function-level scopes leave
        # ``_current_rm`` at ``None`` (unknown), which forces
        # nested concrete contexts to emit ``fesetround``
        # unconditionally.
        self._current_rm = self._entry_rm(func)
        func_ctx = self._resolve_used_ctx(func)
        # ``REAL`` is unrepresentable in C++; it never sets an fenv
        # mode.  Ops inside a REAL scope succeed only via the
        # emitter's lossless-widening dispatch (see
        # :meth:`_try_widen_binary` and friends) — if widening is
        # unavailable the op-level error fires with a precise
        # location.  Validation here would fail prematurely, so we
        # treat REAL like an integer scope: descend without fenv
        # bookkeeping.
        if (
            func_ctx is None
            or func_ctx is REAL
            or self._validate_context_rm(func_ctx, at=func).is_integer()
        ):
            self._visit_block(func.body, None)
        else:
            assert isinstance(func_ctx, EFloatContext)
            with self._fenv_scope(func_ctx.rm):
                self._visit_block(func.body, None)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_hoist_for_class(self, c):
        """
        Emit a zero-initialised C++ variable declaration for a single
        storage class (used to anchor declarations just before the
        ``IfStmt`` that introduces a fresh-in-both-branches name).
        """
        name = self.storage.def_to_name[self.storage.class_members[c][0]]
        storage = self.storage.class_storage[c]
        # Zero-initialise via ``T name{};`` so reads-before-writes
        # are well-defined (FPy analyses ensure this can't happen,
        # but the initialiser also serves as a paper-trail).
        self.writer.add_line(f'{storage.format()} {name}{{}};')

    def _infer_return_storage(self, func: FuncDef) -> CppType | None:
        """Pull the storage type for the function's return value from
        the format analysis.

        ``format_info.fn_fmt.ret_fmt`` is the running join of every
        ``ReturnStmt`` expression's bound (see
        :meth:`FormatAnalysis._visit_return`); using it directly
        means multiple-return programs select a storage class wide
        enough to hold every path's value.

        A ``None`` bound is *not* a missing-return marker — it's the
        format inference convention for non-numeric return values
        (bool, e.g. a ``Compare`` result).  ``choose_storage(None)``
        correctly maps that to ``CppScalar.BOOL``.  FPy's
        reachability check rejects truly missing-return programs at
        decoration time, so we don't distinguish that case here."""
        fmt = self.format_info.fn_fmt.ret_fmt
        try:
            return choose_storage(fmt)
        except StorageSelectionError as e:
            raise CppEmitError(f'return type: {e}', at=func) from e

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_block(self, block: StmtBlock, ctx):
        for stmt in block.stmts:
            # Storage classes anchored to this stmt (currently only
            # ``IfStmt``s with is_intro phi merges) declare just before
            # the stmt, narrowing the variable's scope.
            for c in self.storage.hoists_before.get(stmt, ()):
                self._emit_hoist_for_class(c)
            self._visit_statement(stmt, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        match stmt.target:
            case NamedId():
                # ``StorageInfer`` maps this Assign's SSA def to a
                # C++ variable and tells us whether to declare (a
                # single-writer class) or just reassign into a
                # hoisted decl (multi-writer class).
                target_def = self.def_use.find_def_from_site(stmt.target, stmt)
                target_storage = self.storage.storage_of(target_def)
                rhs = self._emit_assign_rhs(stmt.expr, target_storage, ctx)
                self._emit_bind(stmt.target, stmt, rhs)
            case TupleBinding():
                # ``(a, b) = expr``: bind the rhs to a tuple-valued
                # temp once, then destructure.  Each NamedId in the
                # binding has its own SSA def registered at the
                # Assign statement.
                rhs = self._visit_expr(stmt.expr, ctx)
                tmp = self._fresh_temp()
                self.writer.add_line(f'auto {tmp} = {rhs};')
                self._destructure(stmt.target, tmp, stmt)
            case _:
                raise CppEmitError(
                    f'unsupported assignment target {stmt.target!r}',
                    at=stmt,
                )

    def _emit_assign_rhs(
        self, expr: Expr, target_ty: CppType, ctx,
    ) -> str:
        """Emit an assignment's RHS, coercing a list-literal wrapper to
        the target's storage when they disagree.

        Why this matters: ``_storage_for_expr`` picks the list's wrapper
        from format inference (the tight per-expression join), but
        :class:`StorageInfer` may have widened the variable's storage
        beyond that — e.g. when a subsequent ``x[i] = y`` (IndexedAssign)
        joins the def with a wider value.  Emitting
        ``vector<wide> x = vector<narrow>{...}`` is a hard C++ type
        error, and per-element narrowing inside the literal also
        triggers ``-Wnarrowing`` when the element expressions were
        emitted at the wider type (e.g. ``Round`` lowers to a
        ``static_cast`` into the active context's scalar).  Using the
        target's storage for the wrapper keeps the literal consistent
        with the variable's declared type.
        """
        if isinstance(expr, ListExpr) and isinstance(target_ty, CppList):
            return self._emit_list_expr_at(expr, target_ty, ctx)
        return self._visit_expr(expr, ctx)

    def _emit_list_expr_at(
        self, e: ListExpr, wrapper_ty: CppList, ctx,
    ) -> str:
        """Emit ``[a, b, c]`` as ``wrapper_ty{a, b, c}``, casting each
        element to ``wrapper_ty.elt`` when the per-element emitted
        storage disagrees.  Mirrors :meth:`_visit_list_expr` but with
        an externally-supplied wrapper type."""
        elt_target = wrapper_ty.elt
        parts: list[str] = []
        for elt in e.elts:
            elt_str = self._visit_expr(elt, ctx)
            if isinstance(elt_target, CppScalar):
                # Cast when the element's per-expression storage
                # differs from the wrapper's element type.  Defensive
                # ``try`` mirrors ``_storage_for_expr``'s exception
                # contract — non-storable formats (e.g., symbolic
                # REAL_FORMAT) just skip the cast.
                try:
                    elt_storage = self._storage_for_expr(elt)
                except CppEmitError:
                    elt_storage = None
                if (
                    isinstance(elt_storage, CppScalar)
                    and elt_storage != elt_target
                ):
                    elt_str = self._explicit_cast(elt_str, elt_target)
            parts.append(elt_str)
        return f'{wrapper_ty.format()}{{{", ".join(parts)}}}'

    def _visit_return(self, stmt: ReturnStmt, ctx):
        rhs = self._visit_expr(stmt.expr, ctx)
        self.writer.add_line(f'return {rhs};')

    def _resolve_scope_ctx(self, scope: ContextScope) -> Context | None:
        """Concrete :class:`Context` for *scope*, substituting the
        per-instantiation incoming context for symbolic scopes.

        Specializations of a callee are emitted without first
        monomorphizing the AST, so :class:`ContextUse` leaves the
        function-level scope's ctx as a symbolic ``NamedId``.  The
        ``ctx`` field of :attr:`FormatAnalysis.fn_fmt` is
        the concrete context the caller pinned at the call site —
        we substitute it in wherever the emitter would otherwise
        see a symbolic scope.
        """
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self.format_info.fn_fmt.ctx

    def _resolve_used_ctx(self, site: ContextScopeSite) -> Context | None:
        """Pull the scope's resolved context, *but only if some
        primitive op actually dispatches under it*.

        Returns ``None`` when the scope has no uses — in that case
        the caller skips validation, skips ``fesetround``, and just
        descends into the body.  Programs without rounding-context
        uses don't need a supported context.

        When uses exist, the scope's context must be statically
        resolvable (a concrete :class:`Context` or a symbolic one
        the caller has pinned via ``outer_ctx``); a remaining
        symbolic ctx is rejected here, with the error pointing at
        the ``with`` site rather than at the op that consumed it.
        """
        scope = self._scope_by_site.get(site)
        if scope is None or not self.ctx_use.uses.get(scope):
            return None
        resolved = self._resolve_scope_ctx(scope)
        if resolved is None:
            raise CppEmitError(
                'context expression must be a statically-resolvable '
                f'Context value; got symbolic `{scope.ctx}`',
                at=site,
            )
        return resolved

    def _entry_rm(self, site: ContextScopeSite) -> RM | None:
        """The rounding mode the C++ caller is contractually
        delivering when control enters *site*'s scope.

        Returns the scope's RM for a concrete, fesetround-supported
        FP context (the FPy annotation pins the caller's
        responsibility).  Returns ``None`` for symbolic, integer,
        or unsupported contexts — in those cases the caller's mode
        is treated as unknown and any nested concrete context must
        emit an explicit ``fesetround`` to recover certainty.
        """
        scope = self._scope_by_site.get(site)
        if scope is None:
            return None
        resolved = self._resolve_scope_ctx(scope)
        if not isinstance(resolved, EFloatContext):
            return None
        if resolved.rm not in _FE_RM_MACRO:
            return None
        return resolved.rm

    def _validate_context_rm(
        self, rctx: Context, at: Ast | None = None,
    ) -> CppScalar:
        """Validate *rctx* and return its scalar storage type.

        - Float storage (F32 / F64): rounding mode must be one of
          the four supported by ``fesetround`` (RNE / RTZ / RTP / RTN).
        - Integer storage: rounding mode must be RTZ — C++ integer
          arithmetic rounds toward zero, and other modes would need
          per-operation emulation.
        - Anything else (bool, list, tuple) is out of scope.

        *at* is an optional AST node to anchor error locations
        to — typically the function or ``with`` statement that
        introduced the context.
        """
        try:
            storage = choose_storage(rctx.format())
        except StorageSelectionError as e:
            raise CppEmitError(
                f'unsupported context `{rctx}`: {e}', at=at,
            ) from e
        if not isinstance(storage, CppScalar):
            raise CppEmitError(
                f'unsupported context storage `{storage!r}` for `{rctx}`',
                at=at,
            )
        if storage.is_float():
            assert isinstance(rctx, EFloatContext)
            if rctx.rm not in _FE_RM_MACRO:
                raise CppEmitError(
                    f'rounding mode {rctx.rm} for context `{rctx}` is not '
                    'supported by ``fesetround`` (need RNE, RTZ, RTP, or RTN)',
                    at=at,
                )
        elif storage.is_integer():
            assert isinstance(rctx, MPFixedContext | MPBFixedContext)
            if rctx.rm != RM.RTZ:
                raise CppEmitError(
                    f'integer context `{rctx}` must use RTZ rounding mode '
                    '(C++ integer arithmetic rounds toward zero); got '
                    f'{rctx.rm}',
                    at=at,
                )
            # Reject unbounded integer contexts unless the caller has
            # opted into the unsafe cast.  C++ has no
            # arbitrary-precision integer type, so any rounded
            # arithmetic landing in storage ``int64_t`` via the
            # unbounded-integer fallback may silently overflow.
            # ``MPFixedContext`` reports unboundedness via
            # ``nmin == -1`` (the lower-exponent bound).
            if (
                isinstance(rctx, MPFixedContext)
                and rctx.nmin == -1
                and not self._unsafe_cast_int
            ):
                raise CppEmitError(
                    f'rounding under unbounded integer context `{rctx}` '
                    'has no sound C++ analogue (no arbitrary-precision '
                    'integer type).  Pass `unsafe_cast_int=True` to '
                    '`CppCompiler` to allow truncation to int64_t.',
                    at=at,
                )
        else:
            raise CppEmitError(
                f'unsupported context storage `{storage!r}` for `{rctx}`',
                at=at,
            )
        return storage

    @contextmanager
    def _fenv_scope(self, target_rm: RM):
        """Wrap the contained emission in a ``fesetround`` save / set
        / restore unless the active mode is already *target_rm*.

        ``self._current_rm`` is ``None`` when the live mode is
        unknown (function entry, after restoring a previously-saved
        scope, etc.) — in that case we *always* emit ``fesetround``,
        never relying on a guess about what the C++ runtime is doing.
        When the active mode is known and equals *target_rm* we skip
        the save / set / restore so plain ``with FP64:`` blocks
        under an FP64-RNE function add no fenv noise.
        """
        if self._current_rm is not None and target_rm == self._current_rm:
            yield
            return
        fenv = self._fresh_temp()
        prev_rm = self._current_rm
        self.writer.add_line(f'const auto {fenv} = std::fegetround();')
        self.writer.add_line(f'std::fesetround({_FE_RM_MACRO[target_rm]});')
        self._current_rm = target_rm
        try:
            yield
        finally:
            self._current_rm = prev_rm
        self.writer.add_line(f'std::fesetround({fenv});')

    def _visit_context(self, stmt: ContextStmt, ctx):
        # ``with <ctx>:`` blocks.  The active rounding context is
        # taken from the :class:`ContextUseAnalysis` scope registered
        # at this statement's site (which has already resolved
        # attribute references / partial-eval'd the context
        # expression).  Validation only fires when something inside
        # the block actually uses the context — see
        # :meth:`_resolve_used_ctx`.  For used FP scopes we may emit
        # ``fesetround`` save / set / restore around the body.
        if not isinstance(stmt.target, UnderscoreId):
            raise CppEmitError(
                'binding the active context to a name is not yet supported',
                at=stmt,
            )
        rctx = self._resolve_used_ctx(stmt)
        # ``REAL`` doesn't correspond to any C++ rounding mode — see
        # the same comment in :meth:`_visit_function`.  Treat it as
        # a pass-through; per-op widening dispatch handles the body.
        if (
            rctx is None
            or rctx is REAL
            or self._validate_context_rm(rctx, at=stmt).is_integer()
        ):
            # No op uses this scope, the scope is REAL (no fenv
            # mode), or the scope is integer (no ``fenv`` to manage
            # either way).  Just descend.
            self._visit_block(stmt.body, ctx)
            return
        # Float context: validation above guarantees an
        # ``EFloatContext`` with a ``fesetround``-supported RM.
        assert isinstance(rctx, EFloatContext)
        with self._fenv_scope(rctx.rm):
            self._visit_block(stmt.body, ctx)

    # ------------------------------------------------------------------
    # Expression visitors — return a C++ source fragment

    def _visit_var(self, e: Var, ctx) -> str:
        # Resolve the use to its SSA def, then look up the C++
        # identifier ``StorageInfer`` assigned to that def's class.
        return self._name_for_var_use(e)

    def _visit_decnum(self, e: Decnum, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_hexnum(self, e: Hexnum, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_integer(self, e: Integer, ctx) -> str:
        # Integer literals print directly.
        return str(e.val)

    def _visit_rational(self, e: Rational, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _visit_digits(self, e: Digits, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational())

    def _emit_numeric_literal(self, v: Fraction) -> str:
        """
        Emit a numeric literal as a C++ expression.

        Integer-valued rationals → integer literal.
        Otherwise → ``(double)num / denom`` to force float division.
        """
        if v.denominator == 1:
            return str(v.numerator)
        return f'((double){v.numerator} / (double){v.denominator})'

    def _scalar_storage_for_expr(self, e: Expr) -> CppScalar:
        """Like :meth:`_storage_for_expr` but asserts the result is a
        scalar.  Used by op-table dispatch — primitive numeric ops
        only take/return scalars."""
        ty = self._storage_for_expr(e)
        if not isinstance(ty, CppScalar):
            raise CppEmitError(
                f'expected scalar storage for {type(e).__name__}, got {ty!r}',
                at=e,
            )
        return ty

    def _maybe_cast(
        self, arg: str, arg_ty: CppScalar, target_ty: CppScalar,
        *, at: Ast | None = None,
    ) -> str:
        """Emit *arg* in *target_ty* form, rejecting unsafe casts.

        Used by op-dispatch / comparison cast-to-active paths where
        the conversion is implicit from the user's perspective —
        format inference + storage selection together decided to
        rebind the operand into *target_ty*.  If the conversion
        isn't lossless (``scalar_fits_in``), we refuse to compile
        rather than silently emit a lossy ``static_cast``: the user
        should either narrow the active context or wrap the operand
        in ``fp.round(...)`` to acknowledge the precision change.

        For user-explicit casts (``Round`` / ``Cast``, vector
        subscript), use :meth:`_explicit_cast` instead — that path
        never refuses."""
        if arg_ty == target_ty:
            return arg
        if not scalar_fits_in(arg_ty, target_ty):
            raise CppEmitError(
                f'cannot implicitly cast `{arg_ty.format()}` to '
                f'`{target_ty.format()}`: conversion is lossy.  '
                f'Wrap the operand in ``fp.round(...)`` to make the '
                f'rounding explicit, or use a context whose format '
                f'contains the operand.',
                at=at,
            )
        return self._explicit_cast(arg, target_ty)

    def _explicit_cast(self, arg: str, target_ty: CppScalar) -> str:
        """Emit ``static_cast<target>(arg)`` unconditionally.

        Used at sites where the cast is part of the FPy semantics
        the user wrote — ``Round`` / ``Cast`` lower to a cast,
        ``xs[i]`` needs ``static_cast<size_t>(i)``.  These callers
        accept that the conversion may be lossy."""
        return f'static_cast<{target_ty.format()}>({arg})'

    def _active_ctx_for(self, e: ContextUseSite) -> Context:
        """Look up the rounding context active at expression *e*.

        Symbolic / unresolved scopes are rejected — the cpp backend
        only dispatches primitive ops under statically-known
        contexts.
        """
        try:
            scope = self.ctx_use.find_scope_from_use(e)
        except KeyError as err:
            raise CppEmitError(
                f'no context scope registered for {type(e).__name__}',
                at=e,
            ) from err
        resolved = self._resolve_scope_ctx(scope)
        if resolved is None:
            raise CppEmitError(
                f'cannot dispatch {type(e).__name__} under symbolic '
                f'context `{scope.ctx}`',
                at=e,
            )
        return resolved

    def _scalar_for_ctx(
        self, ctx: Context, at: Ast | None = None,
    ) -> CppScalar:
        """Resolve a context to its C++ scalar storage type."""
        try:
            storage = choose_storage(ctx.format())
        except StorageSelectionError as e:
            raise CppEmitError(
                f'unsupported context `{ctx}`: {e}', at=at,
            ) from e
        if not isinstance(storage, CppScalar):
            raise CppEmitError(
                f'context `{ctx}` resolves to non-scalar storage `{storage!r}`',
                at=at,
            )
        return storage

    def _dispatch_unary(self, e: UnaryOp, arg: str) -> str:
        """Dispatch a unary op via the op table.

        Three signature-selection paths, tried in order:

        1. **Direct match.**  Operand's C++ storage type and the
           active rounding context both equal a signature's input
           slots — no casts needed.
        2. **Cast-to-active.**  No direct match, but the active ctx
           has a same-type signature whose input width contains the
           operand losslessly; cast the operand to that width and
           emit.  Skipped when the active ctx has no C++ storage
           (e.g. ``REAL``).
        3. **Lossless widening.**  Used when (1) and (2) don't fire
           — typically because the active ctx is ``REAL`` or the
           operand can't safely cast into the active ctx.  Selects a
           signature whose output width matches the result-storage
           format inference chose for *e*, on the soundness premise
           that the wider C++ op produces the exact mathematical
           result and rounds to itself.  Soundness is gated by
           :meth:`_maybe_cast` rejecting any lossy operand widening.
        """
        sigs = self.op_table.unary.get(type(e))
        if sigs is None:
            raise CppEmitError(
                f'no signatures for unary op: {type(e).__name__}',
                at=e,
            )
        active = self._active_ctx_for(e)
        arg_storage = self._scalar_storage_for_expr(e.arg)

        # (1) Direct match.
        for sig in sigs:
            if sig.matches(arg_storage, active):
                return sig.format(arg)

        # (2) Cast-to-active fallback: pick the all-active signature
        # and cast the operand into the active context's storage.
        # Skipped silently when the active context has no concrete
        # storage (e.g. ``REAL``) — falls through to widening.
        try:
            target = self._scalar_for_ctx(active, at=e)
        except CppEmitError:
            target = None
        if target is not None:
            for sig in sigs:
                if sig.arg_ty == target and sig.out_ctx == active:
                    return sig.format(
                        self._maybe_cast(arg, arg_storage, target, at=e)
                    )

        # (3) Lossless-widening fallback — only sound when the active
        # context is ``REAL``.  The wider C++ op then produces the exact
        # mathematical result of the operand-typed op and rounds to
        # itself; under any other active context the wider op rounds
        # differently than the active ctx demands.  See
        # :meth:`_try_widen_unary`.
        if active is REAL:
            widened = self._try_widen_unary(e, sigs, arg, arg_storage)
            if widened is not None:
                return widened

        raise CppEmitError(
            f'no matching signature for {type(e).__name__} under '
            f'context `{active}`: arg type `{arg_storage.format()}`',
            at=e,
        )

    def _result_fits_ctx(self, e: Expr, ctx: Context) -> bool:
        """Is rounding the inferred result format of *e* under *ctx* an
        identity?  True iff the exact unrounded result of *e* (recorded
        in ``format_info.by_expr``) is representable in ``ctx.format()``
        — so the C++ op performed under *ctx* yields exactly the value
        format inference predicted."""
        fmt = self.format_info.by_expr.get(e)
        if isinstance(fmt, SetFormat):
            return round_is_identity(fmt, ctx)
        if isinstance(fmt, AbstractableFormat):
            return round_is_identity(AbstractFormat.from_format(fmt), ctx)
        return False

    def _try_widen_unary(
        self,
        e: UnaryOp,
        sigs: list[UnaryCppOp],
        arg: str,
        arg_storage: CppScalar,
    ) -> str | None:
        """Pick a unary signature whose output context contains the
        exact unrounded result of *e* (so the op-under-sig is identity)
        and whose input slot losslessly receives ``arg_storage``.  The
        sig's output is then losslessly cast down to ``result_ty``;
        that cast is sound because the runtime value lies in
        ``format_info.by_expr[e]`` which by storage selection fits in
        ``result_ty``.  ``None`` if no such signature exists.

        Two-pass to prefer narrower signatures: first sigs whose output
        already equals ``result_ty`` (no downcast), then sigs whose
        output is wider (downcast inserted).
        """
        try:
            result_ty = self._storage_for_expr(e)
        except CppEmitError:
            return None
        if not isinstance(result_ty, CppScalar):
            return None

        def _try(sig: UnaryCppOp, *, exact_out: bool) -> str | None:
            try:
                sig_out_ty = self._scalar_for_ctx(sig.out_ctx)
            except CppEmitError:
                return None
            if exact_out and sig_out_ty is not result_ty:
                return None
            if not scalar_fits_in(arg_storage, sig.arg_ty):
                return None
            if not self._result_fits_ctx(e, sig.out_ctx):
                return None
            try:
                cast = self._maybe_cast(arg, arg_storage, sig.arg_ty, at=e)
            except CppEmitError:
                return None
            out = sig.format(cast)
            if sig_out_ty is not result_ty:
                out = f'static_cast<{result_ty.format()}>({out})'
            return out

        for exact_out in (True, False):
            for sig in sigs:
                emitted = _try(sig, exact_out=exact_out)
                if emitted is not None:
                    return emitted
        return None

    def _dispatch_binary(self, e: BinaryOp, lhs: str, rhs: str) -> str:
        """Dispatch a binary op via the op table.

        Same three-phase selection as :meth:`_dispatch_unary`:
        direct match, cast-to-active, lossless-widening.  The
        widening fallback is the path that supports ``with fp.REAL:``
        ops on bounded-format operands: format inference picks a
        concrete wider result storage that contains the exact
        mathematical result, and we dispatch under that.
        """
        sigs = self.op_table.binary.get(type(e))
        if sigs is None:
            raise CppEmitError(
                f'no signatures for binary op: {type(e).__name__}',
                at=e,
            )
        active = self._active_ctx_for(e)
        lhs_storage = self._scalar_storage_for_expr(e.first)
        rhs_storage = self._scalar_storage_for_expr(e.second)

        # (1) Direct match.
        for sig in sigs:
            if sig.matches(lhs_storage, rhs_storage, active):
                return sig.format(lhs, rhs)

        # (2) Cast-to-active fallback.
        try:
            target = self._scalar_for_ctx(active, at=e)
        except CppEmitError:
            target = None
        if target is not None:
            for sig in sigs:
                if (sig.in1_ty == target
                        and sig.in2_ty == target
                        and sig.out_ctx == active):
                    return sig.format(
                        self._maybe_cast(lhs, lhs_storage, target, at=e),
                        self._maybe_cast(rhs, rhs_storage, target, at=e),
                    )

        # (3) Lossless-widening fallback — only sound when the active
        # context is ``REAL``.  See :meth:`_try_widen_binary` and the
        # corresponding note in :meth:`_dispatch_unary`.
        if active is REAL:
            widened = self._try_widen_binary(
                e, sigs, lhs, lhs_storage, rhs, rhs_storage,
            )
            if widened is not None:
                return widened

        raise CppEmitError(
            f'no matching signature for {type(e).__name__} under '
            f'context `{active}`: lhs `{lhs_storage.format()}`, '
            f'rhs `{rhs_storage.format()}`',
            at=e,
        )

    def _try_widen_binary(
        self,
        e: BinaryOp,
        sigs: list[BinaryCppOp],
        lhs: str,
        lhs_storage: CppScalar,
        rhs: str,
        rhs_storage: CppScalar,
    ) -> str | None:
        """Binary analogue of :meth:`_try_widen_unary`."""
        try:
            result_ty = self._storage_for_expr(e)
        except CppEmitError:
            return None
        if not isinstance(result_ty, CppScalar):
            return None

        def _try(sig: BinaryCppOp, *, exact_out: bool) -> str | None:
            try:
                sig_out_ty = self._scalar_for_ctx(sig.out_ctx)
            except CppEmitError:
                return None
            if exact_out and sig_out_ty is not result_ty:
                return None
            if not scalar_fits_in(lhs_storage, sig.in1_ty):
                return None
            if not scalar_fits_in(rhs_storage, sig.in2_ty):
                return None
            if not self._result_fits_ctx(e, sig.out_ctx):
                return None
            try:
                cast_lhs = self._maybe_cast(lhs, lhs_storage, sig.in1_ty, at=e)
                cast_rhs = self._maybe_cast(rhs, rhs_storage, sig.in2_ty, at=e)
            except CppEmitError:
                return None
            out = sig.format(cast_lhs, cast_rhs)
            if sig_out_ty is not result_ty:
                out = f'static_cast<{result_ty.format()}>({out})'
            return out

        for exact_out in (True, False):
            for sig in sigs:
                emitted = _try(sig, exact_out=exact_out)
                if emitted is not None:
                    return emitted
        return None

    def _visit_unaryop(self, e: UnaryOp, ctx) -> str:
        arg = self._visit_expr(e.arg, ctx)
        match e:
            case Cast():
                # ``Cast(arg)`` rounds ``arg`` into the active context
                # and asserts the round was lossless (it's an error to
                # cast a value the target format can't hold exactly).
                # Lowered as ``static_cast`` → bind to a temp →
                # ``assert(arg == tmp || (NaN-aware equality))``.
                return self._emit_exact_cast(e, arg)
            case Not():
                # Logical negation — operand is bool, result is bool.
                # No rounding context involved.
                return f'(!{arg})'
            case IsFinite() | IsInf() | IsNan() | IsNormal() | Signbit():
                return self._emit_fp_predicate(e, arg)
            case Range1():
                return self._emit_range(e, ctx)
            case Dim():
                # ``dim(xs)`` returns the nesting depth of a list — a
                # static property of the value's storage shape.  Read
                # it off format inference and emit the literal int.
                result_ty = self._storage_for_expr(e)
                depth = _list_depth(self._storage_for_expr(e.arg))
                return f'static_cast<{result_ty.format()}>({depth})'
            case Len():
                # ``len(xs)`` — result format is INTEGER, which storage
                # selection rounds to a concrete C++ integer type.
                # Casting ``size()`` (a ``size_t``) keeps the inferred
                # type stable across platforms where ``size_t``
                # differs from ``int64_t``.
                result_ty = self._storage_for_expr(e)
                return f'static_cast<{result_ty.format()}>({arg}.size())'
            case Sum():
                # ``sum(xs)`` → ``std::accumulate(begin, end, T(0))``
                # with ``T`` taken from format inference.
                result_ty = self._storage_for_expr(e)
                return (
                    f'std::accumulate({arg}.begin(), {arg}.end(), '
                    f'static_cast<{result_ty.format()}>(0))'
                )
            case Enumerate():
                return self._emit_enumerate(e, arg)
            case UnaryOp() if type(e) in self.op_table.unary:
                # Op-table-dispatched unary (Neg, Abs, all <cmath>).
                return self._dispatch_unary(e, arg)
            case _:
                raise CppEmitError(
                    f'unsupported unary op: {type(e).__name__}', at=e,
                )

    def _emit_enumerate(self, e: Enumerate, src_str: str) -> str:
        """``enumerate(xs)`` builds a ``std::vector<std::tuple<I, T>>``
        where ``I`` is the index integer type and ``T`` is the source
        element type — both come from format inference on the
        Enumerate node itself.
        """
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)
                and len(result_ty.elt.elts) == 2):
            raise CppEmitError(
                'expected list[(int, T)] for enumerate result, '
                f'got {result_ty!r}', at=e,
            )
        idx_ty = result_ty.elt.elts[0]

        src = self._fresh_temp()
        self.writer.add_line(f'auto {src} = {src_str};')
        result = self._fresh_temp()
        self.writer.add_line(
            f'{result_ty.format()} {result}({src}.size());'
        )
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 0; {i} < {src}.size(); ++{i}) {{'
        )
        self.writer.indent()
        self.writer.add_line(
            f'{result}[{i}] = std::make_tuple('
            f'static_cast<{idx_ty.format()}>({i}), {src}[{i}]);'
        )
        self.writer.dedent()
        self.writer.add_line('}')
        return result

    def _visit_binaryop(self, e: BinaryOp, ctx) -> str:
        match e:
            case Size():
                return self._emit_size(e, ctx)
            case Range2():
                return self._emit_range(e, ctx)
            case BinaryOp():
                # Op-table-dispatched binary (Add, Sub, Mul, Div, all
                # <cmath> two-arg functions).
                lhs = self._visit_expr(e.first, ctx)
                rhs = self._visit_expr(e.second, ctx)
                return self._dispatch_binary(e, lhs, rhs)
            case _:
                raise CppEmitError(
                    f'unsupported binary op: {type(e).__name__}', at=e,
                )

    def _emit_fp_predicate(self, e: UnaryOp, arg: str) -> str:
        """Bool-returning FP predicates: ``isnan`` / ``isinf`` /
        ``isfinite`` / ``isnormal`` / ``signbit``.  These take a float
        and return ``bool`` — they sit outside the op-table because
        the output isn't a rounding context."""
        match e:
            case IsFinite():
                return f'std::isfinite({arg})'
            case IsInf():
                return f'std::isinf({arg})'
            case IsNan():
                return f'std::isnan({arg})'
            case IsNormal():
                return f'std::isnormal({arg})'
            case Signbit():
                return f'std::signbit({arg})'
            case _:
                raise CppEmitError(
                    f'unsupported FP predicate: {type(e).__name__}', at=e,
                )

    def _emit_range(self, e: 'Range1 | Range2 | Range3', ctx) -> str:
        """``range(...)`` as an expression — materialise a vector via
        ``std::iota`` for unit-step ranges, or a manual fill loop for
        ``Range3``'s explicit step.  Used outside for-loop iterables,
        where the loop visitor handles the same shapes without
        materialising the vector."""
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppScalar)
                and result_ty.elt.is_integer()):
            raise CppEmitError(
                f'range(...) expected integer-list result, '
                f'got `{result_ty!r}`',
                at=e,
            )
        int_ty = result_ty.elt.format()
        tmp = self._fresh_temp()
        match e:
            case Range1():
                stop = self._visit_expr(e.arg, ctx)
                stop_ty = self._scalar_storage_for_expr(e.arg)
                stop_cast = self._maybe_cast(stop, stop_ty, result_ty.elt)
                self.writer.add_line(
                    f'{result_ty.format()} {tmp}(static_cast<size_t>({stop_cast}));'
                )
                self.writer.add_line(
                    f'std::iota({tmp}.begin(), {tmp}.end(), '
                    f'static_cast<{int_ty}>(0));'
                )
                return tmp
            case Range2():
                start = self._visit_expr(e.first, ctx)
                stop = self._visit_expr(e.second, ctx)
                start_ty = self._scalar_storage_for_expr(e.first)
                stop_ty = self._scalar_storage_for_expr(e.second)
                start_cast = self._maybe_cast(start, start_ty, result_ty.elt)
                stop_cast = self._maybe_cast(stop, stop_ty, result_ty.elt)
                size_expr = (
                    f'static_cast<size_t>({stop_cast} > {start_cast} '
                    f'? ({stop_cast} - {start_cast}) : 0)'
                )
                self.writer.add_line(
                    f'{result_ty.format()} {tmp}({size_expr});'
                )
                self.writer.add_line(
                    f'std::iota({tmp}.begin(), {tmp}.end(), {start_cast});'
                )
                return tmp
            case Range3():
                # Explicit step — emit a fill loop.
                start = self._visit_expr(e.args[0], ctx)
                stop = self._visit_expr(e.args[1], ctx)
                step = self._visit_expr(e.args[2], ctx)
                start_ty = self._scalar_storage_for_expr(e.args[0])
                stop_ty = self._scalar_storage_for_expr(e.args[1])
                step_ty = self._scalar_storage_for_expr(e.args[2])
                start_cast = self._maybe_cast(start, start_ty, result_ty.elt)
                stop_cast = self._maybe_cast(stop, stop_ty, result_ty.elt)
                step_cast = self._maybe_cast(step, step_ty, result_ty.elt)
                ctr = self._fresh_temp()
                self.writer.add_line(f'{result_ty.format()} {tmp};')
                self.writer.add_line(
                    f'for ({int_ty} {ctr} = {start_cast}; '
                    f'{ctr} < {stop_cast}; {ctr} += {step_cast}) {{'
                )
                self.writer.indent()
                self.writer.add_line(f'{tmp}.push_back({ctr});')
                self.writer.dedent()
                self.writer.add_line('}')
                return tmp
            case _:
                raise CppEmitError(
                    f'unsupported range op: {type(e).__name__}', at=e,
                )

    def _emit_size(self, e: Size, ctx) -> str:
        """``size(xs, d)`` returns the size of *xs* at dimension *d*.
        The compile-time shape of *xs* is known from format inference
        — we follow ``d`` ``[0]`` indices into the first element to
        reach the right ``vector``, then take ``.size()``.

        Requires a constant integer ``d``; symbolic / runtime ``d``
        would need a more sophisticated dispatch and isn't worth the
        complexity for the current corpus."""
        if not isinstance(e.second, Integer):
            raise CppEmitError(
                'size(xs, d) requires a constant integer dimension; '
                f'got `{type(e.second).__name__}`',
                at=e,
            )
        d = e.second.val
        if d < 0:
            raise CppEmitError(
                f'size(xs, d) needs d >= 0, got {d}', at=e,
            )
        xs_ty = self._storage_for_expr(e.first)
        # Walk d list layers; the d-th call is on a value of the
        # appropriate vector type.
        cur_ty = xs_ty
        for _ in range(d):
            if not isinstance(cur_ty, CppList):
                raise CppEmitError(
                    f'size(xs, {d}): xs is not deep enough '
                    f'(type `{xs_ty!r}`)',
                    at=e,
                )
            cur_ty = cur_ty.elt
        if not isinstance(cur_ty, CppList):
            raise CppEmitError(
                f'size(xs, {d}): not a list at depth {d} '
                f'(type `{xs_ty!r}`)',
                at=e,
            )
        xs = self._visit_expr(e.first, ctx)
        access = xs + ''.join(['[0]'] * d)
        result_ty = self._storage_for_expr(e)
        return f'static_cast<{result_ty.format()}>({access}.size())'

    # ------------------------------------------------------------------
    # Stubs for AST nodes not yet handled — classification ops
    # (``IsFinite`` / ``IsNan`` / etc.) and statement kinds
    # (``Assert`` / ``Effect`` / ``Pass``) raise a clean error
    # pointing at the node kind.

    def _unsupported(self, kind: str, at: Ast | None = None):
        raise CppEmitError(
            f'cpp emitter does not handle {kind}', at=at,
        )

    def _visit_bool(self, e: BoolVal, ctx) -> str:
        return 'true' if e.val else 'false'

    def _visit_compare(self, e: Compare, ctx) -> str:
        # Chained comparisons (``a < b < c``) expand to a conjunction
        # of pairwise comparisons.  FPy's evaluation order is
        # left-to-right with no short-circuit semantics that differ
        # from C++'s — the operands are pure expressions, so ``&&``
        # is fine.
        #
        # Each pair of operands is converted to its scalar supremum
        # via explicit ``static_cast`` (no implicit promotion) so the
        # comparison happens in a well-defined common type.
        args = [self._visit_expr(a, ctx) for a in e.args]
        arg_tys = [self._scalar_storage_for_expr(a) for a in e.args]
        clauses = []
        for i, op in enumerate(e.ops):
            common = scalar_sup([arg_tys[i], arg_tys[i + 1]])
            lhs = self._maybe_cast(args[i], arg_tys[i], common)
            rhs = self._maybe_cast(args[i + 1], arg_tys[i + 1], common)
            clauses.append(f'({lhs} {op.symbol()} {rhs})')
        if len(clauses) == 1:
            return clauses[0]
        return '(' + ' && '.join(clauses) + ')'

    def _visit_foreign(self, e, ctx):
        self._unsupported('ForeignVal', at=e)

    def _visit_attribute(self, e, ctx):
        self._unsupported('Attribute', at=e)

    def _visit_nullaryop(self, e, ctx):
        self._unsupported('NullaryOp', at=e)

    def _visit_ternaryop(self, e: TernaryOp, ctx) -> str:
        match e:
            case Range3():
                return self._emit_range(e, ctx)
            case TernaryOp():
                # Op-table-dispatched ternary (Fma).
                a1 = self._visit_expr(e.args[0], ctx)
                a2 = self._visit_expr(e.args[1], ctx)
                a3 = self._visit_expr(e.args[2], ctx)
                return self._dispatch_ternary(e, a1, a2, a3)
            case _:
                raise CppEmitError(
                    f'unsupported ternary op: {type(e).__name__}', at=e,
                )

    def _dispatch_ternary(
        self,
        e: TernaryOp,
        a1: str,
        a2: str,
        a3: str,
    ) -> str:
        """Dispatch a ternary op via the op table.

        Same three-phase selection as :meth:`_dispatch_binary`:
        direct match, cast-to-active, lossless-widening.
        """
        sigs = self.op_table.ternary.get(type(e))
        if sigs is None:
            raise CppEmitError(
                f'no signatures for ternary op: {type(e).__name__}', at=e,
            )
        active = self._active_ctx_for(e)
        in_storages = [self._scalar_storage_for_expr(a) for a in e.args]
        args = [a1, a2, a3]

        # (1) Direct match.
        for sig in sigs:
            if sig.matches(in_storages[0], in_storages[1], in_storages[2], active):
                return sig.format(a1, a2, a3)

        # (2) Cast-to-active fallback.
        try:
            target = self._scalar_for_ctx(active, at=e)
        except CppEmitError:
            target = None
        if target is not None:
            for sig in sigs:
                if (sig.in1_ty == target
                        and sig.in2_ty == target
                        and sig.in3_ty == target
                        and sig.out_ctx == active):
                    return sig.format(
                        self._maybe_cast(a1, in_storages[0], target, at=e),
                        self._maybe_cast(a2, in_storages[1], target, at=e),
                        self._maybe_cast(a3, in_storages[2], target, at=e),
                    )

        # (3) Lossless-widening fallback — only sound when the active
        # context is ``REAL``.  See :meth:`_try_widen_ternary` and the
        # corresponding note in :meth:`_dispatch_unary`.
        if active is REAL:
            widened = self._try_widen_ternary(e, sigs, args, in_storages)
            if widened is not None:
                return widened

        raise CppEmitError(
            f'no matching signature for {type(e).__name__} under '
            f'context `{active}`: '
            f'{[s.format() for s in in_storages]}',
            at=e,
        )

    def _try_widen_ternary(
        self,
        e: TernaryOp,
        sigs: list[TernaryCppOp],
        args: list[str],
        in_storages: list[CppScalar],
    ) -> str | None:
        """Ternary analogue of :meth:`_try_widen_unary`."""
        try:
            result_ty = self._storage_for_expr(e)
        except CppEmitError:
            return None
        if not isinstance(result_ty, CppScalar):
            return None

        def _try(sig: TernaryCppOp, *, exact_out: bool) -> str | None:
            try:
                sig_out_ty = self._scalar_for_ctx(sig.out_ctx)
            except CppEmitError:
                return None
            if exact_out and sig_out_ty is not result_ty:
                return None
            sig_in_tys = (sig.in1_ty, sig.in2_ty, sig.in3_ty)
            if not all(scalar_fits_in(s, t) for s, t in zip(in_storages, sig_in_tys)):
                return None
            if not self._result_fits_ctx(e, sig.out_ctx):
                return None
            try:
                casts = [
                    self._maybe_cast(a, s, t, at=e)
                    for a, s, t in zip(args, in_storages, sig_in_tys)
                ]
            except CppEmitError:
                return None
            out = sig.format(*casts)
            if sig_out_ty is not result_ty:
                out = f'static_cast<{result_ty.format()}>({out})'
            return out

        for exact_out in (True, False):
            for sig in sigs:
                emitted = _try(sig, exact_out=exact_out)
                if emitted is not None:
                    return emitted
        return None

    def _visit_naryop(self, e: NaryOp, ctx) -> str:
        match e:
            case Zip():
                return self._emit_zip(e, ctx)
            case And() | Or():
                return self._emit_bool_chain(e, ctx)
            case Min() | Max():
                return self._emit_min_max(e, ctx)
            case Empty():
                return self._emit_empty(e, ctx)
            case _:
                raise CppEmitError(
                    f'unsupported nary op: {type(e).__name__}', at=e,
                )

    def _emit_bool_chain(self, e: 'And | Or', ctx) -> str:
        """Reduce an ``And`` / ``Or`` to a fully-parenthesised chain
        of ``&&`` / ``||``.  C++'s short-circuit semantics match FPy's
        for pure expressions, and the operands are already bool —
        :class:`StorageInfer` chose ``BOOL`` storage for each one.
        Zero-arg ``and()`` / ``or()`` shouldn't reach here (the
        front-end rejects them), but we degenerate cleanly anyway."""
        if not e.args:
            return 'true' if isinstance(e, And) else 'false'
        args = [self._visit_expr(a, ctx) for a in e.args]
        if len(args) == 1:
            return args[0]
        op = '&&' if isinstance(e, And) else '||'
        return '(' + f' {op} '.join(args) + ')'

    def _emit_min_max(self, e: 'Min | Max', ctx) -> str:
        """Reduce an ``n``-ary ``min`` / ``max`` to a nested pairwise
        call.  We pick ``std::fmin`` / ``std::fmax`` for FP results
        and ``std::min`` / ``std::max`` for integer results — the
        active context's storage decides which.  Each operand is
        cast (losslessly) into the active context's storage so the
        pairwise calls have a single deduced template type."""
        if not e.args:
            raise CppEmitError(
                f'{type(e).__name__} requires at least one argument',
                at=e,
            )
        active = self._active_ctx_for(e)
        target = self._scalar_for_ctx(active, at=e)
        args = [self._visit_expr(a, ctx) for a in e.args]
        arg_storages = [self._scalar_storage_for_expr(a) for a in e.args]
        casted = [
            self._maybe_cast(a, s, target, at=e)
            for a, s in zip(args, arg_storages)
        ]
        if target.is_float():
            fn = 'std::fmin' if isinstance(e, Min) else 'std::fmax'
        else:
            fn = 'std::min' if isinstance(e, Min) else 'std::max'
        result = casted[0]
        for nxt in casted[1:]:
            result = f'{fn}({result}, {nxt})'
        return result

    def _emit_empty(self, e: Empty, ctx) -> str:
        """``empty(d1, ..., dN)`` builds an ``N``-dimensional zero-
        initialised vector.  The result's storage shape comes from
        format inference; we read the dimension sizes off the call
        site and emit nested ``std::vector<T>(d, ...)`` constructors
        right-to-left so the innermost element type bubbles out.

        ``empty()`` with zero args returns ``T()`` — a scalar, which
        format inference resolves to whatever storage the call site
        expects."""
        result_ty = self._storage_for_expr(e)
        dims = [self._visit_expr(a, ctx) for a in e.args]
        dim_storages = [
            self._scalar_storage_for_expr(a) for a in e.args
        ]
        # Each dimension index goes through size_t in the vector
        # constructor — cast explicitly so we don't rely on implicit
        # narrowing.
        dim_strs = [
            self._explicit_cast(d, CppScalar.U64) if s != CppScalar.U64 else d
            for d, s in zip(dims, dim_storages)
        ]
        if _list_depth(result_ty) != len(dim_strs):
            raise CppEmitError(
                f'empty(...) shape mismatch: result type `{result_ty!r}` '
                f'has depth {_list_depth(result_ty)}, but {len(dim_strs)} '
                f'dimensions were given',
                at=e,
            )
        # Build from the inside out: innermost is ``T()``-default,
        # each outer layer wraps it in ``vector<inner>(d, inner_val)``.
        ty: CppType = result_ty
        # Peel down to the innermost element type so we know what
        # default value to use at the leaf.
        peeled: list[CppType] = []
        while isinstance(ty, CppList):
            peeled.append(ty)
            ty = ty.elt
        # ``ty`` is now the scalar / tuple leaf.
        inner = f'{ty.format()}{{}}'
        for layer, d in zip(reversed(peeled), reversed(dim_strs)):
            inner = f'{layer.format()}({d}, {inner})'
        return inner

    def _emit_zip(self, e: Zip, ctx) -> str:
        """``zip(xs1, …, xsN)`` builds a
        ``std::vector<std::tuple<T1, …, TN>>`` whose length matches the
        first iterable.  Each iterable is bound to a temp once to
        evaluate side-effects in source order; the tuple type comes
        from format inference on the Zip node."""
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)):
            raise CppEmitError(
                f'expected list[tuple[...]] for zip result, got {result_ty!r}',
                at=e,
            )

        srcs: list[str] = []
        for arg in e.args:
            arg_str = self._visit_expr(arg, ctx)
            s = self._fresh_temp()
            self.writer.add_line(f'auto {s} = {arg_str};')
            srcs.append(s)

        result = self._fresh_temp()
        self.writer.add_line(
            f'{result_ty.format()} {result}({srcs[0]}.size());'
        )
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 0; {i} < {srcs[0]}.size(); ++{i}) {{'
        )
        self.writer.indent()
        elts = ', '.join(f'{s}[{i}]' for s in srcs)
        self.writer.add_line(f'{result}[{i}] = std::make_tuple({elts});')
        self.writer.dedent()
        self.writer.add_line('}')
        return result
    def _scalar_cast_types(self, e):
        """Source/target scalar storage for a round-like node ``e``.

        The argument's storage is only used to short-circuit
        same-type casts.  Non-dyadic numeric literals
        (e.g., ``fp.round(3.14159265359)``) have a SetFormat with
        no representable storage — that's fine, we always cast."""
        try:
            arg_ty = self._scalar_storage_for_expr(e.arg)
        except CppEmitError:
            arg_ty = None
        active = self._active_ctx_for(e)
        target_ty = self._scalar_for_ctx(active, at=e)
        return arg_ty, target_ty

    def _emit_exact_cast(self, e, arg: str) -> str:
        # ``Cast(arg)`` is a ``static_cast`` plus a runtime assertion
        # that the cast was lossless: cast → bind to a temp →
        # ``assert(arg == tmp || (NaN-aware equality))``.
        arg_ty, target_ty = self._scalar_cast_types(e)
        # Same-type is a guaranteed no-op, no assert.
        if arg_ty == target_ty:
            return arg
        # Bind the rounded value to a temp so the assertion
        # can name it without re-evaluating the source.
        tmp = self._fresh_temp()
        self.writer.add_line(
            f'{target_ty.format()} {tmp} = '
            f'{self._explicit_cast(arg, target_ty)};'
        )
        # NaN-aware comparison: ``NaN == NaN`` is false in
        # C++, so FP operands need an extra ``isnan`` guard
        # to avoid false asserts when both sides round to
        # NaN.  Skipped for purely integer operand pairs.
        if target_ty.is_float() or (arg_ty is not None and arg_ty.is_float()):
            check = (
                f'{arg} == {tmp} || '
                f'(std::isnan({arg}) && std::isnan({tmp}))'
            )
        else:
            check = f'{arg} == {tmp}'
        self.writer.add_line(f'assert({check});')
        return tmp

    def _visit_round(self, e, ctx) -> str:
        # ``Round(arg)`` rounds ``arg`` to the active rounding
        # context — emitted as a plain ``static_cast`` (the cast's
        # rounding mode is the active ``fesetround`` mode set by
        # Phase 5b at the surrounding ``with`` boundary).  The user
        # explicitly asked to round into the active context, so the
        # cast is emitted even when lossy.  Same-type short-circuits
        # to a no-op.
        arg = self._visit_expr(e.arg, ctx)
        arg_ty, target_ty = self._scalar_cast_types(e)
        if arg_ty == target_ty:
            return arg
        return self._explicit_cast(arg, target_ty)

    def _visit_round_at(self, e, ctx):
        self._unsupported('RoundAt', at=e)

    def _visit_call(self, e, ctx) -> str:
        # FPy ``Call(args, fn=Function(...), ...)`` lowers to a plain
        # C++ ``func(arg1, ..., argN)``.  The target's emitted name
        # is determined by the compiler's per-instantiation mangling:
        # at ``compile`` time we walk the FormatAnalysis call graph
        # and dispense a stable mangled name per
        # ``(callee FuncDef, outer_ctx)``, recorded in
        # :attr:`_call_names`.  Two call sites that instantiate the
        # same callee at the same context share an emitted name; two
        # that instantiate at distinct contexts get distinct
        # specializations.
        if e.kwargs:
            raise CppEmitError(
                f'unsupported call: kwargs are not supported '
                f'(call to `{e.func}`)',
                at=e,
            )
        args = ', '.join(self._visit_expr(a, ctx) for a in e.args)
        target = self._call_names.get(e, str(e.func))
        return f'{target}({args})'

    def _visit_tuple_expr(self, e: TupleExpr, ctx) -> str:
        # ``(a, b, c)`` → ``std::make_tuple(a, b, c)``.  Type deduction
        # from the argument types matches what the format-inference
        # pipeline assigns to the tuple slot — no explicit type prefix.
        elts = ', '.join(self._visit_expr(elt, ctx) for elt in e.elts)
        return f'std::make_tuple({elts})'

    def _visit_list_expr(self, e: ListExpr, ctx) -> str:
        # ``[a, b, c]`` → ``std::vector<T>{a, b, c}`` where ``T`` comes
        # from format inference on the list expression.
        elts = ', '.join(self._visit_expr(elt, ctx) for elt in e.elts)
        result_ty = self._storage_for_expr(e)
        return f'{result_ty.format()}{{{elts}}}'

    def _visit_list_comp(self, e: ListComp, ctx) -> str:
        # ``[elt for x1 in iter1 [for x2 in iter2 ...]]``
        #
        # Emitted as a temporary ``std::vector<T> tmp;`` followed by
        # nested for-loops that ``push_back`` the element expression
        # into ``tmp``.  Returns the temp's name as the comprehension's
        # value.  The temp shape mirrors the cpp/ backend; future
        # polish may inline directly into an ``Assign`` target to skip
        # the temp + copy.
        result_ty = self._storage_for_expr(e)
        tmp = self._fresh_temp()
        self.writer.add_line(f'{result_ty.format()} {tmp};')

        for target, iterable in zip(e.targets, e.iterables):
            self._open_comp_loop(target, iterable, e, ctx)

        elt = self._visit_expr(e.elt, ctx)
        self.writer.add_line(f'{tmp}.push_back({elt});')

        for _ in e.targets:
            self.writer.dedent()
            self.writer.add_line('}')

        return tmp

    def _open_comp_loop(
        self,
        target,
        iterable: Expr,
        comp_site: ListComp,
        ctx,
    ) -> None:
        """Emit one ``for`` line for a single comprehension stage,
        leaving the writer indented inside the loop body.

        The target's storage class is determined by ``StorageInfer``
        exactly as for any other AssignDef.  We always declare-on-
        assign in the for header — the comprehension's target lives
        only inside the loop's lexical scope.  The comp target's
        SSA site is the ``ListComp`` node, not the target id itself
        (see ``define_use._visit_list_comp``).
        """
        match target:
            case NamedId():
                target_def = self.def_use.find_def_from_site(target, comp_site)
                target_name = self.storage.def_to_name[target_def]
                target_ty = self.storage.storage_of(target_def).format()
                decl = f'{target_ty} {target_name}'
            case UnderscoreId():
                # ``_`` discards the loop variable — no SSA def, no
                # storage class.  Synthesize a fresh name and pick
                # the iterator type from the iterable: range
                # iterables use the stop bound's storage; value
                # iterables use ``auto`` and let the for-range loop
                # deduce.
                target_name = self._fresh_temp()
                match iterable:
                    case Range1():
                        stop_ty = self._scalar_storage_for_expr(iterable.arg)
                        decl = f'{stop_ty.format()} {target_name}'
                    case Range2() | Range3():
                        stop_ty = self._scalar_storage_for_expr(iterable.args[1])
                        decl = f'{stop_ty.format()} {target_name}'
                    case _:
                        decl = f'auto {target_name}'
            case TupleBinding():
                # ``for (a, b) in xs`` — bind the tuple element to
                # an anonymous temp via ``auto`` and destructure
                # inside the loop body.  Range iterables can't pair
                # with a tuple binding; reject early.
                if isinstance(iterable, (Range1, Range2, Range3)):
                    raise CppEmitError(
                        'tuple-binding comprehension target requires a '
                        'non-range iterable',
                        at=comp_site,
                    )
                tmp = self._fresh_temp()
                iter_str = self._visit_expr(iterable, ctx)
                self.writer.add_line(f'for (auto {tmp} : {iter_str}) {{')
                self.writer.indent()
                self._destructure(target, tmp, comp_site)
                return
            case _:
                raise CppEmitError(
                    f'unsupported comprehension target {target!r}',
                    at=comp_site,
                )

        match iterable:
            case Range1():
                stop = self._visit_expr(iterable.arg, ctx)
                self.writer.add_line(
                    f'for ({decl} = 0; '
                    f'{target_name} < {stop}; ++{target_name}) {{'
                )
            case Range2():
                start = self._visit_expr(iterable.first, ctx)
                stop = self._visit_expr(iterable.second, ctx)
                self.writer.add_line(
                    f'for ({decl} = {start}; '
                    f'{target_name} < {stop}; ++{target_name}) {{'
                )
            case Range3():
                start = self._visit_expr(iterable.args[0], ctx)
                stop = self._visit_expr(iterable.args[1], ctx)
                step = self._visit_expr(iterable.args[2], ctx)
                self.writer.add_line(
                    f'for ({decl} = {start}; '
                    f'{target_name} < {stop}; {target_name} += {step}) {{'
                )
            case _:
                iter_str = self._visit_expr(iterable, ctx)
                self.writer.add_line(f'for ({decl} : {iter_str}) {{')
        self.writer.indent()

    def _visit_list_ref(self, e: ListRef, ctx) -> str:
        # ``xs[i]`` — C++ ``operator[]`` takes ``size_t``, so we route
        # the index through an explicit ``static_cast<size_t>`` rather
        # than relying on implicit conversion.  Bounds-checking is
        # still TODO (FPy's interpreter is strict; we currently match
        # C++'s undefined-behaviour-on-out-of-range).
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        return f'{value}[static_cast<size_t>({index})]'

    def _visit_list_slice(self, e: ListSlice, ctx) -> str:
        # ``xs[start:stop]`` →
        #   auto __cpp_tmpN = <xs>;
        #   <result_ty>(__cpp_tmpN.begin() + start,
        #               __cpp_tmpN.begin() + stop)
        #
        # Binding the value to a temp avoids re-evaluating ``<xs>``
        # when it isn't a simple lvalue (and matches the interpreter,
        # which evaluates the value exactly once).  Indices are cast to
        # ``size_t`` to match the iterator-arithmetic API.  Strict
        # bounds-checking against the interpreter's behaviour is a TODO
        # (slice-out-of-range, negative-index handling, etc.).
        arr_tmp = self._fresh_temp()
        arr_str = self._visit_expr(e.value, ctx)
        self.writer.add_line(f'auto {arr_tmp} = {arr_str};')

        if e.start is None:
            start = '0'
        else:
            start = f'static_cast<size_t>({self._visit_expr(e.start, ctx)})'
        if e.stop is None:
            stop = f'{arr_tmp}.size()'
        else:
            stop = f'static_cast<size_t>({self._visit_expr(e.stop, ctx)})'

        result_ty = self._storage_for_expr(e)
        return (
            f'{result_ty.format()}('
            f'{arr_tmp}.begin() + {start}, '
            f'{arr_tmp}.begin() + {stop})'
        )
    def _visit_if_expr(self, e, ctx) -> str:
        # ``cond ? ift : iff`` — both branches must share a C++ type,
        # so when their storages differ we cast each to the IfExpr's
        # unified storage (chosen by format inference + storage
        # selection over the merged formats).  Non-scalar branches
        # must already match — there's no widening for lists/tuples.
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        out_ty = self._storage_for_expr(e)
        if isinstance(out_ty, CppScalar):
            ift_ty = self._scalar_storage_for_expr(e.ift)
            iff_ty = self._scalar_storage_for_expr(e.iff)
            ift = self._maybe_cast(ift, ift_ty, out_ty, at=e)
            iff = self._maybe_cast(iff, iff_ty, out_ty, at=e)
        else:
            ift_ty_ = self._storage_for_expr(e.ift)
            iff_ty_ = self._storage_for_expr(e.iff)
            if ift_ty_ != out_ty or iff_ty_ != out_ty:
                raise CppEmitError(
                    f'IfExpr branches have incompatible non-scalar '
                    f'storages: ift=`{ift_ty_!r}`, iff=`{iff_ty_!r}`, '
                    f'expected `{out_ty!r}`',
                    at=e,
                )
        return f'({cond} ? {ift} : {iff})'

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        # ``xs[i1]…[iN] = e`` is in-place mutation in C++.  The
        # post-mutation SSA def of ``xs`` shares a storage class with
        # its ``prev`` (see ``_is_in_place_assign`` in
        # ``storage_infer``), so the C++ name is the same on both
        # sides — emit a direct subscript-store.
        target_def = self.def_use.find_def_from_site(stmt.var, stmt)
        target_name = self.storage.def_to_name[target_def]
        idx_strs = [
            f'static_cast<size_t>({self._visit_expr(idx, ctx)})'
            for idx in stmt.indices
        ]
        chain = target_name + ''.join(f'[{i}]' for i in idx_strs)
        rhs = self._visit_expr(stmt.expr, ctx)
        self.writer.add_line(f'{chain} = {rhs};')

    def _visit_if1(self, stmt: If1Stmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'if ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _visit_if(self, stmt: IfStmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'if ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.ift, ctx)
        self.writer.dedent()
        self.writer.add_line('} else {')
        self.writer.indent()
        self._visit_block(stmt.iff, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _visit_while(self, stmt: WhileStmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'while ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _visit_for(self, stmt: ForStmt, ctx):
        match stmt.target:
            case NamedId():
                self._emit_for_named_target(stmt, ctx)
            case UnderscoreId():
                self._emit_for_underscore_target(stmt, ctx)
            case TupleBinding():
                self._emit_for_tuple_target(stmt, ctx)
            case _:
                raise CppEmitError(
                    f'unsupported for-loop target {stmt.target!r}',
                    at=stmt,
                )

    def _emit_for_underscore_target(self, stmt: ForStmt, ctx):
        """``for _ in iter:`` — the loop body never reads the counter,
        so we just emit a synthetic name and pick its type the same
        way :meth:`_open_comp_loop` does for an ``UnderscoreId``
        comprehension target."""
        target = self._fresh_temp()
        if isinstance(stmt.iterable, Range1):
            ty = self._scalar_storage_for_expr(stmt.iterable.arg).format()
        elif isinstance(stmt.iterable, (Range2, Range3)):
            ty = self._scalar_storage_for_expr(stmt.iterable.args[1]).format()
        else:
            ty = 'auto'
        decl = f'{ty} {target}'
        match stmt.iterable:
            case Range1():
                stop = self._visit_expr(stmt.iterable.arg, ctx)
                header = f'for ({decl} = 0; {target} < {stop}; ++{target})'
            case Range2():
                start = self._visit_expr(stmt.iterable.first, ctx)
                stop = self._visit_expr(stmt.iterable.second, ctx)
                header = (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; ++{target})'
                )
            case Range3():
                start = self._visit_expr(stmt.iterable.args[0], ctx)
                stop = self._visit_expr(stmt.iterable.args[1], ctx)
                step = self._visit_expr(stmt.iterable.args[2], ctx)
                header = (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; {target} += {step})'
                )
            case _:
                iter_str = self._visit_expr(stmt.iterable, ctx)
                header = f'for ({decl} : {iter_str})'
        self.writer.add_line(f'{header} {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_for_named_target(self, stmt: ForStmt, ctx):
        assert isinstance(stmt.target, NamedId)
        target_def = self.def_use.find_def_from_site(stmt.target, stmt)
        target = self.storage.def_to_name[target_def]
        # Fold the type into the for header iff the counter is a
        # single-writer class (the common case).  Otherwise the counter
        # was hoisted at the function top and we just reassign here.
        if target_def in self.storage.declare_at_assign:
            storage = self.storage.storage_of(target_def)
            decl = f'{storage.format()} {target}'
        else:
            decl = target
        match stmt.iterable:
            case Range1():
                stop = self._visit_expr(stmt.iterable.arg, ctx)
                header = f'for ({decl} = 0; {target} < {stop}; ++{target})'
            case Range2():
                start = self._visit_expr(stmt.iterable.first, ctx)
                stop = self._visit_expr(stmt.iterable.second, ctx)
                header = (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; ++{target})'
                )
            case Range3():
                start = self._visit_expr(stmt.iterable.args[0], ctx)
                stop = self._visit_expr(stmt.iterable.args[1], ctx)
                step = self._visit_expr(stmt.iterable.args[2], ctx)
                header = (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; {target} += {step})'
                )
            case _:
                iter_str = self._visit_expr(stmt.iterable, ctx)
                header = f'for ({decl} : {iter_str})'
        self.writer.add_line(f'{header} {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_for_tuple_target(self, stmt: ForStmt, ctx):
        # ``for (a, b) in xs:`` — a tuple-binding target only makes
        # sense for non-``range`` iterables (range produces ints).
        # Iterate via a tuple-typed temp, then destructure into the
        # binding's named SSA defs at the top of the loop body.
        if isinstance(stmt.iterable, (Range1, Range2, Range3)):
            raise CppEmitError(
                'tuple-binding for-loop target requires a non-range iterable',
                at=stmt,
            )
        iter_str = self._visit_expr(stmt.iterable, ctx)
        tmp = self._fresh_temp()
        self.writer.add_line(f'for (auto {tmp} : {iter_str}) {{')
        self.writer.indent()
        assert isinstance(stmt.target, TupleBinding)
        self._destructure(stmt.target, tmp, stmt)
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _visit_assert(self, stmt: AssertStmt, ctx):
        test = self._visit_expr(stmt.test, ctx)
        if stmt.msg is None:
            self.writer.add_line(f'assert({test});')
        else:
            msg = stmt.msg.format()
            escaped = msg.replace('\\', '\\\\').replace('"', '\\"')
            self.writer.add_line(f'assert({test} && "fpy assert: {escaped}");')

    def _visit_effect(self, stmt: EffectStmt, ctx):
        expr = self._visit_expr(stmt.expr, ctx)
        self.writer.add_line(f'{expr};')

    def _visit_pass(self, stmt, ctx):
        pass
