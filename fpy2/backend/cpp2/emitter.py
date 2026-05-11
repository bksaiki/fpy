"""
cpp2 backend: emitter.

Walks the (post-pipeline) :class:`FuncDef` and produces a C++ source
string.  Storage types and per-def C++ identifiers come from
:class:`Cpp2PipelineResult.storage` (a :class:`StorageAnalysis`
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
  ``Round`` / ``RoundExact`` / ``Cast`` lower as
  ``static_cast`` (with a NaN-aware assertion for ``RoundExact``).

Anything else raises :class:`Cpp2EmitError`, which the public
``Cpp2Compiler`` re-wraps as :class:`Cpp2CompileError`.
"""

from contextlib import contextmanager
from fractions import Fraction

from ...analysis import (
    ContextScope, ContextScopeSite, ContextUseSite, ContextUseAnalysis,
    DefineUseAnalysis, FormatAnalysis
)
from ...ast.fpyast import (
    Argument, Assign, BinaryOp, BoolVal, Cast, Compare, ContextStmt,
    Decnum, Digits, Enumerate, Expr, ForStmt, FuncDef, Hexnum, If1Stmt,
    IfStmt, IndexedAssign, Integer, Len, ListComp, ListExpr, ListRef,
    ListSlice, NamedId, NaryOp, Range1, Range2, Range3, Rational,
    ReturnStmt, Round, RoundExact, StmtBlock, Sum, TernaryOp, TupleBinding,
    TupleExpr, UnaryOp, UnderscoreId, Var, WhileStmt, Zip,
)
from ...ast.visitor import Visitor
from ...number import EFloatContext, MPFixedContext, MPBFixedContext, RM
from ...number.context.context import Context

from .ops import ScalarOpTable, make_op_table
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


class Cpp2EmitError(Exception):
    """Raised for unsupported AST shapes during emission."""
    pass


class _Cpp2Emitter(Visitor):
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
    ):
        self.ast = ast
        self.storage = storage
        self.def_use = def_use
        self.format_info = format_info
        self.ctx_use = ctx_use
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
        ``__cpp2_tmp`` prefix keeps these distinct from any identifier
        the source program could introduce (FPy doesn't allow leading
        underscores in user-visible names).
        """
        self._tmp_counter += 1
        return f'__cpp2_tmp{self._tmp_counter}'

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
            if isinstance(elt, UnderscoreId):
                continue
            access = f'std::get<{i}>({src})'
            if isinstance(elt, NamedId):
                self._emit_bind(elt, site, access)
            elif isinstance(elt, TupleBinding):
                sub_tmp = self._fresh_temp()
                self.writer.add_line(f'auto {sub_tmp} = {access};')
                self._destructure(elt, sub_tmp, site)
            else:
                raise Cpp2EmitError(
                    f'unsupported tuple-binding element {elt!r}'
                )

    def _storage_for_expr(self, e: Expr) -> CppType:
        """The C++ storage chosen for an expression's result.

        Falls back to ``Cpp2EmitError`` if format inference produced
        nothing storable (e.g., a symbolic ``REAL_FORMAT``).
        """
        fmt = self.format_info.by_expr.get(e)
        try:
            return choose_storage(fmt)
        except StorageSelectionError as err:
            raise Cpp2EmitError(
                f'cannot pick storage for {type(e).__name__}: {err}'
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
                raise Cpp2EmitError(f'unsupported arg pattern: {arg.name!r}')
            storage = self._storage_for_arg(arg)
            arg_strs.append(f'{storage.format()} {arg.name}')
        sig = f'{ret_str} {func.name}({", ".join(arg_strs)})'

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
        if func_ctx is None or self._validate_context_rm(func_ctx).is_integer():
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
        """Pull the type of the function's return expression from the
        format analysis.  Walks the body and any directly-nested
        ``with`` blocks; the AST shape FPy guarantees is that every
        path either falls off the end (rejected upstream) or hits a
        single ``Return`` whose expression's bound is the function's
        return type."""

        def find_return(block: StmtBlock) -> ReturnStmt | None:
            for stmt in block.stmts:
                if isinstance(stmt, ReturnStmt):
                    return stmt
                if isinstance(stmt, ContextStmt):
                    nested = find_return(stmt.body)
                    if nested is not None:
                        return nested
            return None

        ret = find_return(func.body)
        if ret is None:
            return None
        fmt = self.format_info.by_expr.get(ret.expr)
        try:
            return choose_storage(fmt)
        except StorageSelectionError as e:
            raise Cpp2EmitError(f'return type: {e}') from e

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
        rhs = self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.target, NamedId):
            # ``StorageInfer`` maps this Assign's SSA def to a C++
            # variable and tells us whether the assign should declare
            # the variable (single-writer class) or just reassign into
            # a hoisted decl (multi-writer class).
            self._emit_bind(stmt.target, stmt, rhs)
        elif isinstance(stmt.target, TupleBinding):
            # ``(a, b) = expr``: bind the rhs to a tuple-valued temp
            # once, then destructure.  Each NamedId in the binding has
            # its own SSA def registered at the Assign statement.
            tmp = self._fresh_temp()
            self.writer.add_line(f'auto {tmp} = {rhs};')
            self._destructure(stmt.target, tmp, stmt)
        else:
            raise Cpp2EmitError(
                f'unsupported assignment target {stmt.target!r}'
            )

    def _visit_return(self, stmt: ReturnStmt, ctx):
        rhs = self._visit_expr(stmt.expr, ctx)
        self.writer.add_line(f'return {rhs};')

    def _resolve_used_ctx(self, site: ContextScopeSite) -> Context | None:
        """Pull the scope's resolved context, *but only if some
        primitive op actually dispatches under it*.

        Returns ``None`` when the scope has no uses — in that case
        the caller skips validation, skips ``fesetround``, and just
        descends into the body.  Programs without rounding-context
        uses don't need a supported context.

        When uses exist, the scope's context must be statically
        resolvable (a concrete :class:`Context`); symbolic context
        variables are rejected here, with the error pointing at the
        ``with`` site rather than at the op that consumed it.
        """
        scope = self._scope_by_site.get(site)
        if scope is None or not self.ctx_use.uses.get(scope):
            return None
        rctx = scope.ctx
        if not isinstance(rctx, Context):
            raise Cpp2EmitError(
                'context expression must be a statically-resolvable '
                f'Context value; got symbolic `{rctx}`'
            )
        return rctx

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
        if scope is None or not isinstance(scope.ctx, EFloatContext):
            return None
        if scope.ctx.rm not in _FE_RM_MACRO:
            return None
        return scope.ctx.rm

    def _validate_context_rm(self, rctx: Context) -> CppScalar:
        """Validate *rctx* and return its scalar storage type.

        - Float storage (F32 / F64): rounding mode must be one of
          the four supported by ``fesetround`` (RNE / RTZ / RTP / RTN).
        - Integer storage: rounding mode must be RTZ — C++ integer
          arithmetic rounds toward zero, and other modes would need
          per-operation emulation.
        - Anything else (bool, list, tuple) is out of scope.
        """
        try:
            storage = choose_storage(rctx.format())
        except StorageSelectionError as e:
            raise Cpp2EmitError(
                f'unsupported context `{rctx}`: {e}'
            ) from e
        if not isinstance(storage, CppScalar):
            raise Cpp2EmitError(
                f'unsupported context storage `{storage!r}` for `{rctx}`'
            )
        if storage.is_float():
            assert isinstance(rctx, EFloatContext)
            if rctx.rm not in _FE_RM_MACRO:
                raise Cpp2EmitError(
                    f'rounding mode {rctx.rm} for context `{rctx}` is not '
                    'supported by ``fesetround`` (need RNE, RTZ, RTP, or RTN)'
                )
        elif storage.is_integer():
            assert isinstance(rctx, MPFixedContext | MPBFixedContext)
            if rctx.rm != RM.RTZ:
                raise Cpp2EmitError(
                    f'integer context `{rctx}` must use RTZ rounding mode '
                    '(C++ integer arithmetic rounds toward zero); got '
                    f'{rctx.rm}'
                )
        else:
            raise Cpp2EmitError(
                f'unsupported context storage `{storage!r}` for `{rctx}`'
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
            raise Cpp2EmitError(
                'binding the active context to a name is not yet supported'
            )
        rctx = self._resolve_used_ctx(stmt)
        if rctx is None or self._validate_context_rm(rctx).is_integer():
            # No op uses this scope, or the scope is integer (no
            # ``fenv`` to manage either way).  Just descend.
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
            raise Cpp2EmitError(
                f'expected scalar storage for {type(e).__name__}, got {ty!r}'
            )
        return ty

    def _maybe_cast(self, arg: str, arg_ty: CppScalar, target_ty: CppScalar) -> str:
        """Emit *arg* in *target_ty* form, rejecting unsafe casts.

        Used by op-dispatch / comparison cast-to-active paths where
        the conversion is implicit from the user's perspective —
        format inference + storage selection together decided to
        rebind the operand into *target_ty*.  If the conversion
        isn't lossless (``scalar_fits_in``), we refuse to compile
        rather than silently emit a lossy ``static_cast``: the user
        should either narrow the active context or wrap the operand
        in ``fp.round(...)`` to acknowledge the precision change.

        For user-explicit casts (``Round`` / ``RoundExact``, vector
        subscript), use :meth:`_explicit_cast` instead — that path
        never refuses."""
        if arg_ty == target_ty:
            return arg
        if not scalar_fits_in(arg_ty, target_ty):
            raise Cpp2EmitError(
                f'cannot implicitly cast `{arg_ty.format()}` to '
                f'`{target_ty.format()}`: conversion is lossy.  '
                f'Wrap the operand in ``fp.round(...)`` to make the '
                f'rounding explicit, or use a context whose format '
                f'contains the operand.'
            )
        return self._explicit_cast(arg, target_ty)

    def _explicit_cast(self, arg: str, target_ty: CppScalar) -> str:
        """Emit ``static_cast<target>(arg)`` unconditionally.

        Used at sites where the cast is part of the FPy semantics
        the user wrote — ``Round`` / ``RoundExact`` lower to a cast,
        ``xs[i]`` needs ``static_cast<size_t>(i)``.  These callers
        accept that the conversion may be lossy."""
        return f'static_cast<{target_ty.format()}>({arg})'

    def _active_ctx_for(self, e: ContextUseSite) -> Context:
        """Look up the rounding context active at expression *e*.

        Symbolic / unresolved scopes are rejected — the cpp2 backend
        only dispatches primitive ops under statically-known
        contexts.
        """
        try:
            scope = self.ctx_use.find_scope_from_use(e)
        except KeyError as err:
            raise Cpp2EmitError(
                f'no context scope registered for {type(e).__name__}'
            ) from err
        if not isinstance(scope.ctx, Context):
            raise Cpp2EmitError(
                f'cannot dispatch {type(e).__name__} under symbolic '
                f'context `{scope.ctx}`'
            )
        return scope.ctx

    def _scalar_for_ctx(self, ctx: Context) -> CppScalar:
        """Resolve a context to its C++ scalar storage type."""
        try:
            storage = choose_storage(ctx.format())
        except StorageSelectionError as e:
            raise Cpp2EmitError(
                f'unsupported context `{ctx}`: {e}'
            ) from e
        if not isinstance(storage, CppScalar):
            raise Cpp2EmitError(
                f'context `{ctx}` resolves to non-scalar storage `{storage!r}`'
            )
        return storage

    def _dispatch_unary(self, e: UnaryOp, arg: str) -> str:
        """Dispatch a unary op via the op table.

        Picks a signature whose ``out_ctx`` matches the active
        rounding context.  Direct match is preferred (operand's C++
        storage type equals the signature's ``arg_ty``); on
        mismatch we fall back to the all-active-context signature
        and explicit-cast the operand to that context's storage.
        """
        sigs = self.op_table.unary.get(type(e))
        if sigs is None:
            raise Cpp2EmitError(
                f'no signatures for unary op: {type(e).__name__}'
            )
        active = self._active_ctx_for(e)
        arg_storage = self._scalar_storage_for_expr(e.arg)

        for sig in sigs:
            if sig.matches(arg_storage, active):
                return sig.format(arg)
        # Cast-to-active fallback: pick the all-active signature and
        # cast the operand into the active context's storage.
        target = self._scalar_for_ctx(active)
        for sig in sigs:
            if sig.arg_ty == target and sig.out_ctx == active:
                return sig.format(self._maybe_cast(arg, arg_storage, target))
        raise Cpp2EmitError(
            f'no matching signature for {type(e).__name__} under '
            f'context `{active}`: arg type `{arg_storage.format()}`'
        )

    def _dispatch_binary(self, e: BinaryOp, lhs: str, rhs: str) -> str:
        """Dispatch a binary op via the op table.

        Picks a signature whose ``out_ctx`` matches the active
        rounding context.  Direct match is preferred (operand C++
        storage types equal the signature's ``in*_ty``); on
        mismatch we fall back to the all-active-context signature
        and explicit-cast both operands to that context's storage.
        """
        sigs = self.op_table.binary.get(type(e))
        if sigs is None:
            raise Cpp2EmitError(
                f'no signatures for binary op: {type(e).__name__}'
            )
        active = self._active_ctx_for(e)
        lhs_storage = self._scalar_storage_for_expr(e.first)
        rhs_storage = self._scalar_storage_for_expr(e.second)

        for sig in sigs:
            if sig.matches(lhs_storage, rhs_storage, active):
                return sig.format(lhs, rhs)
        # Cast-to-active fallback: pick the all-active signature and
        # cast both operands into the active context's storage.
        target = self._scalar_for_ctx(active)
        for sig in sigs:
            if (sig.in1_ty == target
                    and sig.in2_ty == target
                    and sig.out_ctx == active):
                return sig.format(
                    self._maybe_cast(lhs, lhs_storage, target),
                    self._maybe_cast(rhs, rhs_storage, target),
                )
        raise Cpp2EmitError(
            f'no matching signature for {type(e).__name__} under '
            f'context `{active}`: lhs `{lhs_storage.format()}`, '
            f'rhs `{rhs_storage.format()}`'
        )

    def _visit_unaryop(self, e: UnaryOp, ctx) -> str:
        arg = self._visit_expr(e.arg, ctx)
        if isinstance(e, Cast):
            # ``Cast`` is an analysis-only annotation: it asserts the
            # argument's value is contained in the active context's
            # format.  At the C++ level it's the identity — no
            # generated code, no static_cast (the explicit-cast policy
            # is enforced by the op-table dispatch elsewhere, not by
            # ``Cast`` itself).
            return arg
        if type(e) in self.op_table.unary:
            return self._dispatch_unary(e, arg)
        if isinstance(e, Len):
            # ``len(xs)`` — the result format is INTEGER, which storage
            # selection rounds to a concrete C++ integer type.  Casting
            # ``size()`` (a ``size_t``) keeps the inferred type stable
            # across platforms where ``size_t`` differs from ``int64_t``.
            result_ty = self._storage_for_expr(e)
            return f'static_cast<{result_ty.format()}>({arg}.size())'
        if isinstance(e, Sum):
            # ``sum(xs)`` → ``std::accumulate(begin, end, T(0))`` with
            # ``T`` taken from format inference (so the accumulator's
            # type matches the inferred bound on the result).
            result_ty = self._storage_for_expr(e)
            return (
                f'std::accumulate({arg}.begin(), {arg}.end(), '
                f'static_cast<{result_ty.format()}>(0))'
            )
        if isinstance(e, Enumerate):
            return self._emit_enumerate(e, arg)
        raise Cpp2EmitError(f'unsupported unary op: {type(e).__name__}')

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
            raise Cpp2EmitError(
                'expected list[(int, T)] for enumerate result, '
                f'got {result_ty!r}'
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
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)
        return self._dispatch_binary(e, lhs, rhs)

    # ------------------------------------------------------------------
    # Stubs for AST nodes not yet handled — classification ops
    # (``IsFinite`` / ``IsNan`` / etc.) and statement kinds
    # (``Assert`` / ``Effect`` / ``Pass``) raise a clean error
    # pointing at the node kind.

    def _unsupported(self, kind: str):
        raise Cpp2EmitError(f'cpp2 emitter does not handle {kind}')

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

    def _visit_foreign(self, e, ctx): self._unsupported('ForeignVal')
    def _visit_attribute(self, e, ctx): self._unsupported('Attribute')
    def _visit_nullaryop(self, e, ctx): self._unsupported('NullaryOp')
    def _visit_ternaryop(self, e: TernaryOp, ctx) -> str:
        a1 = self._visit_expr(e.args[0], ctx)
        a2 = self._visit_expr(e.args[1], ctx)
        a3 = self._visit_expr(e.args[2], ctx)
        return self._dispatch_ternary(e, a1, a2, a3)

    def _dispatch_ternary(
        self,
        e: TernaryOp,
        a1: str,
        a2: str,
        a3: str,
    ) -> str:
        """Dispatch a ternary op via the op table.

        Same shape as :meth:`_dispatch_binary` — direct match if
        every operand's C++ storage equals the signature's input
        slots; otherwise the all-active-context signature with
        explicit casts on each operand.
        """
        sigs = self.op_table.ternary.get(type(e))
        if sigs is None:
            raise Cpp2EmitError(
                f'no signatures for ternary op: {type(e).__name__}'
            )
        active = self._active_ctx_for(e)
        in_storages = [self._scalar_storage_for_expr(a) for a in e.args]

        for sig in sigs:
            if sig.matches(in_storages[0], in_storages[1], in_storages[2], active):
                return sig.format(a1, a2, a3)
        # Cast-to-active fallback.
        target = self._scalar_for_ctx(active)
        for sig in sigs:
            if (sig.in1_ty == target
                    and sig.in2_ty == target
                    and sig.in3_ty == target
                    and sig.out_ctx == active):
                return sig.format(
                    self._maybe_cast(a1, in_storages[0], target),
                    self._maybe_cast(a2, in_storages[1], target),
                    self._maybe_cast(a3, in_storages[2], target),
                )
        raise Cpp2EmitError(
            f'no matching signature for {type(e).__name__} under '
            f'context `{active}`: '
            f'{[s.format() for s in in_storages]}'
        )

    def _visit_naryop(self, e: NaryOp, ctx) -> str:
        if isinstance(e, Zip):
            return self._emit_zip(e, ctx)
        raise Cpp2EmitError(f'unsupported nary op: {type(e).__name__}')

    def _emit_zip(self, e: Zip, ctx) -> str:
        """``zip(xs1, …, xsN)`` builds a
        ``std::vector<std::tuple<T1, …, TN>>`` whose length matches the
        first iterable.  Each iterable is bound to a temp once to
        evaluate side-effects in source order; the tuple type comes
        from format inference on the Zip node."""
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)):
            raise Cpp2EmitError(
                f'expected list[tuple[...]] for zip result, got {result_ty!r}'
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
    def _visit_round(self, e, ctx) -> str:
        # ``Round(arg)`` rounds ``arg`` to the active rounding
        # context — emitted as a plain ``static_cast`` (the cast's
        # rounding mode is the active ``fesetround`` mode set by
        # Phase 5b at the surrounding ``with`` boundary).
        #
        # ``RoundExact(arg)`` is the same cast plus a runtime
        # assertion that the cast was lossless: cast → bind to a
        # temp → ``assert(arg == tmp || (NaN-aware equality))``.
        if not isinstance(e, (Round, RoundExact)):
            raise Cpp2EmitError(
                f'unsupported round op: {type(e).__name__}'
            )
        arg = self._visit_expr(e.arg, ctx)
        arg_ty = self._scalar_storage_for_expr(e.arg)
        active = self._active_ctx_for(e)
        target_ty = self._scalar_for_ctx(active)

        if isinstance(e, Round):
            # The user explicitly asked to round into the active
            # context — emit the cast even when lossy.  Same-type
            # short-circuits to a no-op.
            if arg_ty == target_ty:
                return arg
            return self._explicit_cast(arg, target_ty)

        # RoundExact: same-type is a guaranteed no-op, no assert.
        if arg_ty == target_ty:
            return arg
        # Bind the rounded value to a temp so the assertion can name
        # it without re-evaluating the source expression.
        tmp = self._fresh_temp()
        self.writer.add_line(
            f'{target_ty.format()} {tmp} = {self._explicit_cast(arg, target_ty)};'
        )
        # NaN-aware comparison: ``NaN == NaN`` is false in C++, so
        # FP operands need an extra ``isnan`` guard to avoid false
        # asserts when both sides round to NaN.  Skipped for purely
        # integer operand pairs.
        if target_ty.is_float() or arg_ty.is_float():
            check = (
                f'{arg} == {tmp} || '
                f'(std::isnan({arg}) && std::isnan({tmp}))'
            )
        else:
            check = f'{arg} == {tmp}'
        self.writer.add_line(f'assert({check});')
        return tmp

    def _visit_round_at(self, e, ctx):
        self._unsupported('RoundAt')

    def _visit_call(self, e, ctx):
        self._unsupported('Call')

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
        if isinstance(target, NamedId):
            target_def = self.def_use.find_def_from_site(target, comp_site)
            target_name = self.storage.def_to_name[target_def]
            target_ty = self.storage.storage_of(target_def).format()
            decl = f'{target_ty} {target_name}'
        elif isinstance(target, TupleBinding):
            # ``for (a, b) in xs`` — bind the tuple element to an
            # anonymous temp via ``auto`` and destructure inside the
            # loop body.  Range iterables can't pair with a tuple
            # binding; reject early.
            if isinstance(iterable, (Range1, Range2, Range3)):
                raise Cpp2EmitError(
                    'tuple-binding comprehension target requires a '
                    'non-range iterable'
                )
            tmp = self._fresh_temp()
            iter_str = self._visit_expr(iterable, ctx)
            self.writer.add_line(f'for (auto {tmp} : {iter_str}) {{')
            self.writer.indent()
            self._destructure(target, tmp, comp_site)
            return
        else:
            raise Cpp2EmitError(
                f'unsupported comprehension target {target!r}'
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
        #   auto __cpp2_tmpN = <xs>;
        #   <result_ty>(__cpp2_tmpN.begin() + start,
        #               __cpp2_tmpN.begin() + stop)
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
    def _visit_list_set(self, e, ctx):
        # ``ListSet`` is only produced by the ``FuncUpdate`` transform,
        # which the cpp2 pipeline does not run — C++ mutates vector
        # elements directly via ``IndexedAssign``.  Reaching this
        # visitor would be a pipeline bug.
        self._unsupported('ListSet')

    def _visit_if_expr(self, e, ctx):
        self._unsupported('IfExpr')

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
        if isinstance(stmt.target, NamedId):
            self._emit_for_named_target(stmt, ctx)
        elif isinstance(stmt.target, TupleBinding):
            self._emit_for_tuple_target(stmt, ctx)
        else:
            raise Cpp2EmitError(
                f'unsupported for-loop target {stmt.target!r}'
            )

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
            raise Cpp2EmitError(
                'tuple-binding for-loop target requires a non-range iterable'
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

    def _visit_assert(self, stmt, ctx):
        self._unsupported('AssertStmt')

    def _visit_effect(self, stmt, ctx):
        self._unsupported('EffectStmt')

    def _visit_pass(self, stmt, ctx):
        self._unsupported('PassStmt')
