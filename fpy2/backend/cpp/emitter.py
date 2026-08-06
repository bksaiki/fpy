"""
cpp backend: emitter.

Walks the post-pipeline :class:`FuncDef` and produces a C++ source string.
Storage types and per-def identifiers come from :class:`StorageAnalysis`;
per-expression bounds from :class:`FormatAnalysis`.  Primitive ops dispatch
through the :class:`ScalarOpTable` in :mod:`.ops`, which is where operand
formats meet the active rounding context.

An unsupported program raises :class:`CppEmitError`, which ``CppCompiler``
re-wraps as :class:`CppCompileError`.  A violated invariant from an earlier
phase raises :class:`CppInternalError` instead -- a backend bug rather than an
uncompilable program.
"""

import dataclasses
import math
from collections.abc import Sequence
from contextlib import contextmanager
from fractions import Fraction

from ...analysis import (
    AssignDef,
    ContextScope,
    ContextScopeSite,
    ContextUseAnalysis,
    ContextUseSite,
    DefineUseAnalysis,
    Definition,
    FormatAnalysis,
)
from ...analysis.format_infer import (
    AbstractableFormat,
    AbstractFormat,
    SetFormat,
    round_is_identity,
)
from ...ast.fpyast import (
    AllOf,
    AMax,
    AMin,
    And,
    AnyOf,
    Argument,
    AssertStmt,
    Assign,
    Ast,
    BinaryOp,
    BoolVal,
    Call,
    Cast,
    Compare,
    ContextStmt,
    Decnum,
    Digits,
    Dim,
    EffectStmt,
    Empty,
    Enumerate,
    Expr,
    ForStmt,
    Fst,
    FuncDef,
    Hexnum,
    If1Stmt,
    IfExpr,
    IfStmt,
    IndexedAssign,
    Integer,
    IsFinite,
    IsInf,
    IsNan,
    IsNormal,
    Len,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    Max,
    Min,
    NamedId,
    NaryOp,
    Not,
    Or,
    Range1,
    Range2,
    Range3,
    Rational,
    RationalVal,
    ReturnStmt,
    Round,
    Signbit,
    Size,
    Snd,
    StmtBlock,
    Sum,
    TernaryOp,
    TupleBinding,
    TupleExpr,
    UnaryOp,
    UnderscoreId,
    Var,
    WhileStmt,
    Zip,
)
from ...ast.visitor import Visitor
from ...function import Function
from ...number import (
    REAL,
    RM,
    EFloatContext,
    Float,
    MPBFixedContext,
    MPFixedContext,
    RealFloat,
)
from ...number.context.context import Context
from .ops import CppOp, ScalarOpTable
from .storage import (
    StorageSelectionError,
    choose_storage,
    scalar_fits_in,
    scalar_sup,
)
from .storage_infer import (
    StorageAnalysis,
    binds_by_reference,
    is_rebound,
)
from .target import make_op_table
from .types import CppList, CppScalar, CppTuple, CppType
from .unbox import UnboxAnalysis, return_storage

# Map FPy rounding modes to ``<cfenv>`` macros.  Only the four modes
# in this table can be set via ``fesetround``.
_FE_RM_MACRO: dict[RM, str] = {
    RM.RNE: 'FE_TONEAREST',
    RM.RTZ: 'FE_TOWARDZERO',
    RM.RTP: 'FE_UPWARD',
    RM.RTN: 'FE_DOWNWARD',
}

def _value_cpp_type(v: Fraction) -> 'CppScalar | None':
    """The C++ type of the token *v* prints as, or ``None`` if none can hold it.

    Not the same question as its *storage*, which comes from the value: ``1.5``
    is stored as a ``float`` while the token ``1.5`` is a ``double``.

    Bounds are on the *magnitude*, since C++ has no negative literal — ``-2**31``
    is unary minus applied to ``2**31``, so the expression is a ``long``.  A
    decimal literal takes the first of ``int`` / ``long`` / ``long long`` that
    fits and is ill-formed when none does, so ``None`` means the caller must
    spell it as a float or refuse.
    """
    if v.denominator != 1:
        return CppScalar.F64 if _as_exact_double(v) is not None else None
    n = abs(v.numerator)
    if n < 2 ** 31:
        return CppScalar.S32
    if n < 2 ** 63:
        return CppScalar.S64
    return None



def _as_exact_double(v: Fraction) -> float | None:
    """*v* as a ``double`` when binary64 holds it exactly, else ``None``."""
    try:
        x = float(v)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return x if Fraction(x) == v else None


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


@dataclasses.dataclass(frozen=True)
class _TupleAccess:
    """Resolved form of a folded ``fst``/``snd`` chain over a tuple.

    - ``off is None`` — a finished value: ``s`` is its C++ expression and
      ``ty`` its type (one ``std::get`` for an element, or a non-accessor
      base).
    - ``off`` an ``int`` — the not-yet-materialized suffix ``base[off:]`` of
      a tuple: ``s`` is the base expression and ``ty`` the base
      :class:`CppTuple`.  Materialized into a ``std::make_tuple`` only if the
      suffix is actually used (it never is when a ``fst`` reads one element
      out of it).
    """
    s: str
    ty: CppType
    off: int | None


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


class CppInternalError(CppEmitError):
    """An invariant an earlier phase was supposed to guarantee.

    Where :class:`CppEmitError` reports a *program* this backend cannot compile,
    this reports a *backend bug*: an upstream analysis handed the emitter
    something structurally impossible.  Kept distinct so such a bug does not
    reach the user as "your program is unsupported", sending them off to rewrite
    working code.

    A subclass, so existing handlers and the :class:`CppCompileError` wrapping
    are unchanged; the wrapping preserves the type on ``__cause__``, which is how
    ``test_internal_invariants.py`` tells the two apart.
    """

    def __init__(self, msg: str, *, at: 'Ast | None' = None):
        super().__init__(f'internal error (please report): {msg}', at=at)


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
        unbox: UnboxAnalysis | None = None,
        callee_params: dict | None = None,
    ):
        self.ast = ast
        self.storage = storage
        self.def_use = def_use
        self.format_info = format_info
        self.ctx_use = ctx_use
        # How each list is represented, or ``None`` to keep every handle.
        self.unbox = unbox
        # Emitted parameter types of the callees, so a call site can adapt.
        self._callee_params: dict = callee_params or {}
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
        # The storage this function returns, set once the signature is
        # emitted; see :meth:`_visit_return`.
        self._return_storage: CppType | None = None
        self.op_table: ScalarOpTable = make_op_table()
        # Build a site → scope lookup over the analysis's scope list.
        self._scope_by_site: dict[ContextScopeSite, ContextScope] = {
            scope.site: scope for scope in ctx_use.scopes
        }
        # The mode in effect at the current emission point; `None` is unknown,
        # which forces `_fenv_scope` to set it rather than assume.
        self._current_rm: RM | None = None
        self._fenv_saved: list[str] = []
        """Saved mode of each enclosing ``fesetround`` scope, outermost first.

        :meth:`_visit_return` restores from this, since a ``return`` jumps over
        the restore at the end of every scope it sits inside.  That covers every
        path: FPy has no ``break`` or ``continue``, so ``return`` is the only
        early exit.
        """

    # ------------------------------------------------------------------
    # List representation
    #
    # Purely syntactic: these take and return emitted C++ strings.  The point is
    # that how a list is stored and accessed is spelled out here and nowhere
    # else, so changing the representation is a change to this block plus
    # ``fpy::list`` in ``.utils``.

    @staticmethod
    def _elt_of(ty: CppType) -> str:
        """The element type of a list storage type."""
        assert isinstance(ty, CppList), f'not a list storage type: {ty!r}'
        return ty.elt.format()

    @staticmethod
    def _is_boxed(ty: CppType) -> bool:
        """Whether *ty* is a handle rather than the sequence itself."""
        assert isinstance(ty, CppList), f'not a list storage type: {ty!r}'
        return ty.boxed

    @classmethod
    def _list_seq(cls, ty: CppType, base: str) -> str:
        """*base* as the sequence itself — dereferenced if it is a handle.

        No parenthesis is needed unboxed: a list-valued C++ expression here is a
        name, an emitter temp, or a subscript chain, never an operator
        expression that could bind more loosely than ``[]``.
        """
        return f'(*{base})' if cls._is_boxed(ty) else base

    @classmethod
    def _member(cls, ty: CppType, base: str) -> str:
        """``base->`` or ``base.``, whichever reaches a member of the sequence."""
        return f'{base}->' if cls._is_boxed(ty) else f'{base}.'

    @classmethod
    def _list_len(cls, ty: CppType, base: str) -> str:
        """``len(xs)``."""
        return f'{cls._member(ty, base)}size()'

    @classmethod
    def _list_at(cls, ty: CppType, base: str, idx: str) -> str:
        """``xs[i]``.  The cast belongs here because C++ ``operator[]`` takes an
        unsigned index while FPy indices are signed."""
        return f'{cls._list_seq(ty, base)}[static_cast<size_t>({idx})]'

    @classmethod
    def _list_at_raw(cls, ty: CppType, base: str, idx: str) -> str:
        """``xs[i]`` where *idx* is already a ``size_t`` — an emitter-internal
        loop counter rather than an FPy index, so no cast is needed."""
        return f'{cls._list_seq(ty, base)}[{idx}]'

    def _bind_operand(self, expr: str) -> str:
        """A name for *expr*, so it can be read more than once.

        Already a name — evaluated once, no side effects — so nothing to bind.
        Otherwise bind it to a temp; a list is a handle, so this costs nothing.
        """
        if expr.isidentifier():
            return expr
        tmp = self._fresh_temp()
        self.writer.add_line(f'auto&& {tmp} = {expr};')
        return tmp

    def _list_range(self, ty: CppType, base: str) -> str:
        """The operand of a range-``for`` over a list.

        A temporary must be bound to a name: range-``for`` extends the lifetime
        of the range-init's own result, and dereferencing a handle yields a
        reference to the pointee -- so the handle would be freed before the
        first iteration.
        """
        bound = self._bind_operand(base)
        return f'*{bound}' if self._is_boxed(ty) else bound

    @classmethod
    def _list_begin(cls, ty: CppType, base: str) -> str:
        return f'{cls._member(ty, base)}begin()'

    @classmethod
    def _list_end(cls, ty: CppType, base: str) -> str:
        return f'{cls._member(ty, base)}end()'

    @classmethod
    def _list_push(cls, ty: CppType, base: str, elt: str) -> str:
        """Append to a list under construction."""
        return f'{cls._member(ty, base)}push_back({elt})'

    @classmethod
    def _list_new(cls, ty: CppType, args: str) -> str:
        """A new list from a parenthesised constructor argument list.

        The one place the boxed and unboxed spellings of construction are
        stated; the named wrappers below only supply the arguments.
        """
        if cls._is_boxed(ty):
            return f'fpy::make_list<{cls._elt_of(ty)}>({args})'
        return f'{ty.format()}({args})'

    @classmethod
    def _list_new_sized(cls, ty: CppType, n: str) -> str:
        return cls._list_new(ty, n)

    @classmethod
    def _list_empty(cls, ty: CppType) -> str:
        """A new empty list.  Never emit a bare declaration for a *boxed* list:
        an uninitialised ``fpy::list`` is a null handle, unlike an empty
        ``std::vector``."""
        return cls._list_new_sized(ty, '0')

    @classmethod
    def _list_new_filled(cls, ty: CppType, n: str, fill: str) -> str:
        return cls._list_new(ty, f'{n}, {fill}')

    @classmethod
    def _list_new_init(cls, ty: CppType, parts: list[str]) -> str:
        """The given elements.  Not :meth:`_list_new`: a braced list is the
        argument when boxed, and the whole initialiser when not."""
        joined = ', '.join(parts)
        if cls._is_boxed(ty):
            return f'fpy::make_list<{cls._elt_of(ty)}>({{{joined}}})'
        return f'{ty.format()}{{{joined}}}'

    @classmethod
    def _list_new_range(cls, ty: CppType, first: str, last: str) -> str:
        """The half-open iterator range ``[first, last)``, copied."""
        return cls._list_new(ty, f'{first}, {last}')

    def _fresh_temp(self) -> str:
        """A fresh emitter-only identifier.

        One leading underscore, deliberately: FPy forbids leading underscores in
        user names, so ``_tmp`` cannot collide, while any identifier containing
        ``__`` is reserved to the implementation at every scope.
        """
        self._tmp_counter += 1
        return f'_tmp{self._tmp_counter}'

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

    @staticmethod
    def _is_aggregate(storage: CppType) -> bool:
        """A list or a tuple — worth binding by reference rather than copying.

        For a tuple because a copy is O(size); for a list because a copy of the
        handle touches the refcount.  A scalar copy is free.
        """
        return isinstance(storage, (CppList, CppTuple))

    def _is_rebound(self, d: Definition) -> bool:
        return is_rebound(self.storage, d)

    def _arg_decl(self, arg: Argument, storage: CppType) -> str:
        """Parameter declaration; see :meth:`_binding_decl` for the rule."""
        assert isinstance(arg.name, NamedId)
        d = self.def_use.find_def_from_site(arg.name, arg)
        return self._binding_decl(d, storage, arg.name)

    def _binding_decl(self, d: Definition, storage: CppType, name) -> str:
        """``T name``, ``const T& name`` or ``T& name`` for the def *d*.

        A name that is never rebound binds by reference: that shares the object
        and leaves the refcount untouched, which measurement showed is the only
        case where the atomic control block costs anything.  ``const`` unless
        something in the region writes through it -- note ``const`` applies to
        the handle, not the elements, so a callee can still write ``xs[i] = e``
        and the caller sees it, which is FPy's parameter semantics.

        Rebinding must stay local, so such a name takes its own copy.
        """
        if not binds_by_reference(self.storage, self.def_use, d):
            return f'{storage.format()} {name}'
        if self._writes_through(d, storage):
            return f'{storage.format()}& {name}'
        return f'const {storage.format()}& {name}'

    def _writes_through(self, d: Definition, storage: CppType) -> bool:
        """Whether a ``const`` reference to *d* would reject a write FPy allows.

        Region-wide, and across calls: ``ys = xs; ys[0] = e`` writes through a
        different storage class than the one declared, and a callee that writes
        its parameter makes the caller's argument non-const too.
        """
        if self.unbox is None:
            return False
        return self.unbox.writes_through(
            self.unbox.alias.region_of(d), storage,
        )

    def _foreach_decl(
        self, target_def, name: str,
        *, elt: CppType | None = None, at: Expr | None = None,
    ) -> str:
        """Loop-variable declaration for a range-for.

        :meth:`_binding_decl`, plus: *elt* is the container element type when the
        caller knows it, and the two must agree -- a range-for has nowhere to put a
        conversion.
        """
        if target_def is None:
            return f'const auto& {name}'
        storage = self.storage.storage_of(target_def)
        if elt is not None and at is not None:
            self._require_bridgeable(elt, storage, at)
        return self._binding_decl(target_def, storage, name)

    def _is_readonly_alias(
        self, stmt: Assign, target_def, target_storage: CppType,
    ) -> bool:
        """Whether ``x = y`` can bind a ``const`` reference instead of copying.

        Copying a handle is O(1) and shares the elements, so this is only a
        saved refcount bump -- but a tuple copy is O(size).  Same condition as
        :meth:`_binding_decl`: a ``const`` reference cannot be rebound.
        """
        return self._binds_reference(target_def)

    def _emitted_storage_of(self, d) -> CppType:
        """The type *d*'s C++ name actually has.

        ``storage_of`` is the type the analysis *chose*; when the emitter binds
        the name as ``const auto&`` the type is the one C++ *deduced* from the
        initializer, and the two can differ.  Every caller reasoning about the
        emitted code wants this one.
        """
        src = self._reference_source(d)
        if src is not None:
            ty = self._storage_or_none(src)
            if ty is not None:
                return ty
        return self.storage.storage_of(d)

    def _reference_source(self, d) -> 'Expr | None':
        """The initializer *d*'s C++ name deduces its type from, if any.

        Asked of the whole storage class, not of *d*.  A use resolves to the
        latest def, and ``xs[i] = e`` makes a fresh one that is unioned with its
        ``prev`` — so a name declared as ``const auto& L3 = N2[1]`` and then
        written through is *read* through an ``IndexedAssign``-sited def whose own
        site says nothing about the binding.  One member of the class carries the
        declaration, and that is the one whose initializer fixed the type.
        """
        cls = self.storage.def_class[d]
        for m in self.storage.class_members[cls]:
            if isinstance(m.site, Assign) and self._binds_reference(m):
                return m.site.expr
        return None

    def _binds_reference(self, d) -> bool:
        """Whether the emitter binds *d*'s name as a reference.

        The single answer for the question, because two callers need the *same*
        one: this decides what to emit, and :meth:`_storage_or_none` decides what
        the emitted name's type therefore is.  Answering differently in the two
        places means the guard compares against a type the code does not have.
        """
        return binds_by_reference(
            self.storage, self.def_use, d,
            allow_projection=(
                self.unbox is not None
                and self.unbox.may_reference_projection(d)
            ),
        )

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
        src_ty: CppType | None = None,
        at: Expr | None = None,
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
                    # The name is declared at its own storage but initialized
                    # from the tuple's field, and there is no room for a
                    # conversion between the two.
                    if isinstance(src_ty, CppTuple) and i < len(src_ty.elts) and at is not None:
                        d = self.def_use.find_def_from_site(elt, site)
                        self._require_bridgeable(
                            src_ty.elts[i], self.storage.storage_of(d), at,
                        )
                    self._emit_bind(elt, site, access)
                case TupleBinding():
                    access = f'std::get<{i}>({src})'
                    sub_tmp = self._fresh_temp()
                    # read-only (destructured) -> reference, no copy
                    self.writer.add_line(f'auto&& {sub_tmp} = {access};')
                    sub_ty = (
                        src_ty.elts[i]
                        if isinstance(src_ty, CppTuple) and i < len(src_ty.elts)
                        else None
                    )
                    self._destructure(elt, sub_tmp, site, sub_ty, at)
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
            ty = choose_storage(fmt)
        except StorageSelectionError as err:
            raise CppEmitError(
                f'cannot pick storage for {type(e).__name__}: {err}',
                at=e,
            ) from err
        # `choose_storage` knows a list's *shape*, not its representation:
        # that is decided per alias region (see `.unbox`).
        return ty if self.unbox is None else self.unbox.annotate(e, ty)

    # ------------------------------------------------------------------
    # Function emission

    def _visit_function(self, func: FuncDef, ctx):
        # Determine return type from the return statement's expression
        # bound (the slice assumes a single return, possibly nested in a
        # ``with`` block).
        ret_ty = self._infer_return_storage(func)
        ret_str = ret_ty.format() if ret_ty is not None else 'void'
        # Read back by each `return` to convert a narrower value into it.
        self._return_storage = ret_ty

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
            arg_strs.append(self._arg_decl(arg, storage))
        emitted_name = self._func_name_override or func.name
        sig = f'{ret_str} {emitted_name}({", ".join(arg_strs)})'

        self.writer.add_line(sig + ' {')
        self.writer.indent()
        # `_current_rm` is the mode the live fenv is guaranteed to hold.  For a
        # concrete FP function-level scope the FPy contract says the caller
        # delivers it, so no entry `fesetround` is needed; symbolic, integer and
        # unsupported scopes leave it None, forcing nested contexts to set the
        # mode unconditionally.
        self._current_rm = self._entry_rm(func)
        func_ctx = self._resolve_used_ctx(func)
        # REAL sets no fenv mode; its ops succeed only via `_try_widen`, whose
        # failure reports a precise location.  Validating here would fire first
        # and worse, so descend like an integer scope.
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
        if isinstance(storage, CppList):
            # a bare ``fpy::list`` is a *null* handle, not an empty list, so a
            # hoisted list must be given one
            self.writer.add_line(
                f'{storage.format()} {name} = {self._list_empty(storage)};'
            )
        else:
            self.writer.add_line(f'{storage.format()} {name}{{}};')

    def _infer_return_storage(self, func: FuncDef) -> CppType | None:
        """The function's return storage, with a source location on failure.

        See :func:`return_storage`.  A ``None`` bound is not a missing return --
        FPy's reachability check rejects those at decoration time -- but format
        inference's convention for a non-numeric result, which maps to ``BOOL``.
        """
        try:
            return return_storage(self.format_info.fn_fmt.ret_fmt, self.unbox)
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
                if self._is_readonly_alias(stmt, target_def, target_storage):
                    # ``x = y`` where both are read-only aggregates: bind a
                    # const reference instead of copying the whole value.
                    src = self._visit_expr(stmt.expr, ctx)
                    target_name = self.storage.def_to_name[target_def]
                    # `auto&` when the alias must stay writable
                    ref = (
                        'auto&'
                        if self._writes_through(target_def, target_storage)
                        else 'const auto&'
                    )
                    self.writer.add_line(f'{ref} {target_name} = {src};')
                else:
                    rhs = self._emit_assign_rhs(stmt.expr, target_storage, ctx)
                    self._emit_bind(stmt.target, stmt, rhs)
            case TupleBinding():
                # ``(a, b) = expr``: bind the rhs to a tuple-valued
                # temp once, then destructure.  Each NamedId in the
                # binding has its own SSA def registered at the
                # Assign statement.
                rhs = self._visit_expr(stmt.expr, ctx)
                tmp = self._fresh_temp()
                # read-only (destructured) -> reference, no copy
                self.writer.add_line(f'auto&& {tmp} = {rhs};')
                self._destructure(
                    stmt.target, tmp, stmt,
                    self._storage_or_none(stmt.expr), stmt.expr,
                )
            case _:
                raise CppEmitError(
                    f'unsupported assignment target {stmt.target!r}',
                    at=stmt,
                )

    def _emit_at(self, e: Expr, want: CppType | None, ctx) -> str:
        """Emit *e* as a value of storage *want*.

        One place admits one C++ type, while ``format_infer`` bounds each expression
        by its own values -- correctly, since that is the question it answers.
        Reconciling them is a storage question, so it lives here: an expression that
        *constructs* its value is built at *want*, and anything with storage of its
        own goes to :meth:`_convert_storage`, where a shared list is refused.
        """
        if want is None:
            return self._visit_expr(e, ctx)
        if isinstance(want, CppScalar):
            # A scalar has no identity, so there is nothing to build at a
            # storage: C++ converts it at the point of use.
            return self._visit_expr(e, ctx)
        match e:
            case ListExpr() if isinstance(want, CppList):
                parts = [self._emit_deduced(elt, want.elt, ctx) for elt in e.elts]
                return self._list_new_init(want, parts)
            case TupleExpr() if (
                isinstance(want, CppTuple) and len(want.elts) == len(e.elts)
            ):
                parts = [
                    self._emit_deduced(elt, w, ctx)
                    for elt, w in zip(e.elts, want.elts)
                ]
                return f'std::make_tuple({", ".join(parts)})'
            case ListComp() if isinstance(want, CppList):
                return self._emit_list_comp_at(e, want, ctx)
            case IfExpr():
                cond = self._visit_expr(e.cond, ctx)
                ift = self._emit_at(e.ift, want, ctx)
                iff = self._emit_at(e.iff, want, ctx)
                return f'({cond} ? {ift} : {iff})'
        emitted = self._visit_expr(e, ctx)
        src = self._storage_or_none(e)
        if src is None:
            return emitted
        return self._convert_storage(emitted, src, want, at=e)

    def _emit_deduced(self, e: Expr, want: CppType, ctx) -> str:
        """:meth:`_emit_at` where C++ takes the type *from the argument*.

        A braced initializer and ``std::make_tuple`` both do: the first rejects
        a narrowing conversion outright, and the second silently deduces a
        different type — ``std::make_tuple(a, b)`` with a ``float`` ``a`` builds
        a ``std::tuple<float, double>`` that the surrounding declaration then
        will not accept.  Either way the scalar's cast has to be spelled, even
        though the same conversion is implicit anywhere else.
        """
        code = self._emit_at(e, want, ctx)
        if isinstance(want, CppScalar):
            src = self._storage_or_none(e)
            if isinstance(src, CppScalar) and src != want:
                return self._explicit_cast(code, want)
        return code

    def _emit_assign_rhs(
        self, expr: Expr, target_ty: CppType, ctx,
    ) -> str:
        """Emit an assignment's RHS at the *target's* storage, not its own.

        ``StorageInfer`` may have widened the variable past the RHS's own bound
        -- a later ``x[i] = y`` joins the def with a wider value -- and
        ``vector<wide> x = vector<narrow>{...}`` is a hard type error.
        """
        return self._emit_at(expr, target_ty, ctx)

    def _convert_storage(
        self, code: str, src: CppType, want: CppType, *, at: Expr,
    ) -> str:
        """*code*, of storage *src*, as storage *want*.

        Where :meth:`_emit_at` builds a constructor at the wanted bound, a variable
        cannot be built that way -- its storage was fixed by its definition -- so a
        narrower one flowing into a wider place is converted here.  Every conversion
        is spelled: ``std::make_tuple`` deduces from its arguments, and
        ``std::vector`` has no converting constructor, so a list is rebuilt
        element-wise.
        """
        if src == want:
            return code
        if isinstance(src, CppScalar) and isinstance(want, CppScalar):
            return self._explicit_cast(code, want)
        if isinstance(src, CppTuple) and isinstance(want, CppTuple):
            if len(src.elts) != len(want.elts):
                raise self._refuse_mismatch(src, want, at)
            base = self._bind_operand(code)
            fields = [
                self._convert_storage(f'std::get<{i}>({base})', s, w, at=at)
                for i, (s, w) in enumerate(zip(src.elts, want.elts))
            ]
            return f'std::make_tuple({", ".join(fields)})'
        if not (isinstance(src, CppList) and isinstance(want, CppList)):
            # Nothing bridges these, so handing the code back unchanged would
            # emit a type error inside generated C++ -- the worst place for a
            # user to meet one.  Refuse here instead.
            raise self._refuse_mismatch(src, want, at)
        if src.boxed:
            # A rebuilt list is a different object, and a handle exists exactly
            # so that FPy's aliasing survives.  Unsharing it here would be
            # silent, so refuse instead.
            raise self._refuse_unsharing(src, want, at)
        unboxed = CppList(want.elt, boxed=False)
        if src.elt != want.elt:
            code = self._rebuild_list(code, src, unboxed, at=at)
        if not want.boxed:
            return code
        # A value has no aliases to lose, so giving it a handle is free.
        return f'std::make_shared<{unboxed.format()}>({code})'

    def _refuse_mismatch(
        self, src: CppType, want: CppType, at: Expr,
    ) -> CppEmitError:
        """One place wants two types and nothing in C++ bridges them.

        Reached where a *representation* decision could not be reconciled --
        distinct ``std::vector`` instantiations are unrelated types, and so are a
        vector and a handle.  A limitation in the compiler is acceptable;
        emitting C++ that does not compile is not, so this refuses rather than
        letting the mismatch reach the C++ compiler.
        """
        return CppEmitError(
            f'unsupported: this value is `{src.format()}` where '
            f'`{want.format()}` is needed, and C++ has no conversion between '
            f'them.  Keeping the two formats the same at this point avoids it.',
            at=at,
        )

    def _require_no_narrowing(
        self, src: CppType | None, want: CppType | None, at: Expr,
    ) -> None:
        """Refuse a store C++ would narrow silently.

        The sibling of :meth:`_require_bridgeable`, for the one case where a
        format disagreement is a *wrong answer* instead of a compile error: C++
        accepts a narrowing store into a slot, and FPy says the list then holds
        the wider value.  Widening the container instead is not available --
        another name may already alias it, which ``format_infer`` does not track
        (see ``docs/todos/backend-cpp.md``).
        """
        if not (isinstance(src, CppScalar) and isinstance(want, CppScalar)):
            return
        if scalar_fits_in(src, want):
            return
        raise CppEmitError(
            f'unsupported: storing a `{src.format()}` into a slot of '
            f'`{want.format()}` would narrow it, and the list would then not '
            f'hold the value FPy says it does.  Round the value to the list\'s '
            f'format, or build the list at the wider one.',
            at=at,
        )

    def _require_bridgeable(
        self, src: CppType | None, want: CppType | None, at: Expr,
    ) -> None:
        """Refuse unless *src* can reach *want*, for a site that cannot convert.

        A slot store, a loop target and a destructured field all take their type
        from something else, so the emitter has nowhere to put a conversion --
        the only options are agreement or a refusal.
        """
        if src is None or want is None or src == want:
            return
        if isinstance(src, CppScalar) and isinstance(want, CppScalar):
            return                # C++ converts these
        raise self._refuse_mismatch(src, want, at)

    def _refuse_unsharing(
        self, src: CppList, want: CppList, at: Expr,
    ) -> CppEmitError:
        """The one representation change with no sound lowering.

        Points at the *definition* as well as the use: this is where the
        conflict surfaces, but the declaration is where it can be fixed — give
        the list the wider format and nothing needs converting.
        """
        what, where, fix = 'this list', '', 'define it at the wider format'
        if isinstance(at, Var):
            what = f'`{at.name}`'
            fix = f'define {what} at the wider format'
            site = self.def_use.find_def_from_use(at).site
            loc = getattr(site, 'loc', None)
            if loc is not None:
                # Line only: the file is already in the location prefix, and
                # ``Location.format`` opens a backtick it never closes.
                where = f' (defined on line {loc.start_line})'
        elif isinstance(at, Call):
            # The remaining reachable case: a callee's return representation is
            # fixed by its own body, so nothing on this side can raise it.
            callee = getattr(at.func, 'name', at.func)
            what = f'the list `{callee}` returns'
            fix = f'have `{callee}` return the wider format'
        if src.elt == want.elt:
            detail = (
                f'is `{src.format()}` where `{want.format()}` is needed'
            )
        else:
            detail = (
                f'holds `{src.elt.format()}` elements where '
                f'`{want.elt.format()}` is needed'
            )
        return CppEmitError(
            f'unsupported: {what}{where} {detail}.  Changing a list\'s element '
            f'type needs a new buffer, and this one is shared — so the copy '
            f'would not be the list its other references name.  Either {fix}, '
            f'or do not mix formats at this point.',
            at=at,
        )

    def _rebuild_list(
        self, code: str, src: CppList, want: CppList, *, at: Expr,
    ) -> str:
        """A new list of *want*, element-wise from *code*.

        Unboxed: :meth:`_convert_storage` adds the handle afterwards if one is
        wanted, so that the boxed case has exactly one place that allocates.
        """
        assert not want.boxed, 'the caller boxes; see _convert_storage'
        base = self._bind_operand(code)
        if not isinstance(src.elt, CppList):
            # Flat: the range constructor converts each element on the way in.
            return self._list_new_range(
                want, self._list_begin(src, base), self._list_end(src, base),
            )
        out, i = self._fresh_temp(), self._fresh_temp()
        n = self._list_len(src, base)
        self.writer.add_line(f'{want.format()} {out};')
        self.writer.add_line(f'{self._member(want, out)}reserve({n});')
        self.writer.add_line(f'for (size_t {i} = 0; {i} < {n}; ++{i}) {{')
        self.writer.indent()
        elt = self._convert_storage(
            self._list_at_raw(src, base, i), src.elt, want.elt, at=at,
        )
        self.writer.add_line(f'{self._list_push(want, out, elt)};')
        self.writer.dedent()
        self.writer.add_line('}')
        return out

    def _storage_or_none(self, e: Expr) -> CppType | None:
        """The storage *e* actually emits as, or ``None`` where unknown.

        A format with no ladder entry is not a disagreement to repair, so callers
        that only *adjust* a representation skip it rather than failing.

        A variable reads as its **declaration**: ``storage_infer`` aggregates a whole
        class, so asking ``by_expr`` would compare against a type the emitted name
        does not have.
        """
        if isinstance(e, Var):
            d = self.def_use.find_def_from_use(e)
            # A name the emitter binds as `const auto&` has the type C++
            # *deduced* from its initializer, not the one `storage_of` chose --
            # so follow the alias.  Missing this is how `[L3, L3]` came to hold
            # a `fpy::list<uint8_t>` in a `std::vector<fpy::list<float>>`.
            src = self._reference_source(d)
            if src is not None:
                return self._storage_or_none(src)
            ty = self.storage.storage_of(d)
            return ty if self.unbox is None else self.unbox.annotate(e, ty)
        if isinstance(e, ListRef):
            # ``xss[i]`` reads a *declared* element, so peel the container's
            # declaration rather than asking ``by_expr`` -- which answers from
            # the format, and can name a type the container does not hold.
            base = self._storage_or_none(e.value)
            if isinstance(base, CppList):
                return base.elt
        try:
            return self._storage_for_expr(e)
        except CppEmitError:
            return None

    def _visit_return(self, stmt: ReturnStmt, ctx):
        # A function has one return type, so every `return` produces it: built
        # at that storage where the value is constructed here, converted where
        # it comes from somewhere with storage of its own.
        rhs = self._emit_at(stmt.expr, self._return_storage, ctx)
        if self._fenv_saved:
            # A `return` jumps over each enclosing scope's restore, so the
            # mode would escape into the caller.  The outermost scope holds the
            # caller's.  C++ offers no point between evaluating a return
            # expression and returning, so bind the value first; non-const, so
            # the return can still move out of it.
            if not self._mode_independent(stmt.expr):
                ty = self._return_storage
                tmp = self._fresh_temp()
                decl = 'auto' if ty is None else ty.format()
                self.writer.add_line(f'{decl} {tmp} = {rhs};')
                rhs = tmp
            self.writer.add_line(f'std::fesetround({self._fenv_saved[0]});')
        self.writer.add_line(f'return {rhs};')

    def _mode_independent(self, e: Expr) -> bool:
        """Is the emitted value of *e* independent of the live rounding mode?

        Asked of the AST, not the emitted text: how a literal is *spelled* is
        not stable.  ``False`` costs a temp, never correctness.
        """
        if isinstance(e, Var | RationalVal):
            return True
        return isinstance(e, Round) and self._fold_rounded_literal(e) is not None

    def _resolve_scope_ctx(self, scope: ContextScope) -> Context | None:
        """Concrete :class:`Context` for *scope*.

        A callee's specializations are emitted without monomorphizing the AST,
        so ``ContextUse`` leaves the function-level scope symbolic; ``fn_fmt.ctx``
        is what the caller pinned, and substitutes in.
        """
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self.format_info.fn_fmt.ctx

    def _resolve_used_ctx(self, site: ContextScopeSite) -> Context | None:
        """The scope's resolved context, or ``None`` when no primitive op dispatches
        under it -- the caller then skips validation and ``fesetround`` entirely, so
        a program with no rounding-context use needs no supported context.

        Where uses exist the context must be statically resolvable; a symbolic one
        is rejected here, pointing at the ``with`` site rather than at the op that
        consumed it.
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

        Float storage needs an ``fesetround`` mode (RNE/RTZ/RTP/RTN); integer
        storage needs RTZ, which is what C++ integer arithmetic does -- anything
        else would need per-operation emulation.  Bool, list and tuple are out of
        scope.  *at* anchors the error location.
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
        self._fenv_saved.append(fenv)
        try:
            yield
        finally:
            self._fenv_saved.pop()
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
        return self._emit_real_literal(e.as_real(), at=e)

    def _visit_hexnum(self, e: Hexnum, ctx) -> str:
        return self._emit_real_literal(e.as_real(), at=e)

    def _visit_integer(self, e: Integer, ctx) -> str:
        # Through the shared path, which range-checks: `str(e.val)` here was a
        # second way to emit an integer too large for a C++ integer literal.
        return self._emit_numeric_literal(e.as_rational(), at=e)

    def _visit_rational(self, e: Rational, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational(), at=e)

    def _visit_digits(self, e: Digits, ctx) -> str:
        return self._emit_numeric_literal(e.as_rational(), at=e)

    def _emit_real_literal(self, v: 'Fraction | Float', *, at: Expr) -> str:
        """Emit an exact-real literal, preserving the sign of a negative zero
        (which has no `Fraction` form; see `RationalVal.as_real`)."""
        if isinstance(v, Float):
            # negative zero is the only non-`Fraction` real `as_real` produces
            return '-0.0'
        return self._emit_numeric_literal(v, at=at)

    def _emit_numeric_literal(self, v: Fraction, *, at: Expr) -> str:
        """Emit a numeric literal as a C++ expression.

        An FPy literal is an exact rational rounded where it is *used*; C++ has no
        such thing, so one binary64 holds exactly prints as itself and one it cannot
        is refused.  ``num / denom`` would be an *operation* where FPy has a
        constant, rounding under whatever mode happens to be set.  ``fp.round`` is
        how a program pins a constant to a format -- see
        :meth:`_fold_rounded_literal`.

        Digits only while a C++ integer literal holds the value
        (:func:`_value_cpp_type`); past ``long long`` gcc folds them to ``0``, so it
        falls through to the floating spelling below.
        """
        ty = _value_cpp_type(v)
        if ty is not None and ty.is_integer():
            return str(v.numerator)
        exact = _as_exact_double(v)
        if exact is not None:
            return repr(exact)
        raise CppEmitError(
            f'unsupported literal: `{v.numerator}/{v.denominator}` is not '
            f'representable in C++.  Wrap it in `fp.round(...)` to round it to '
            f'a format that is.',
            at=at,
        )

    def _scalar_storage_for_expr(self, e: Expr) -> CppScalar:
        """Like :meth:`_storage_for_expr` but asserts the result is a
        scalar.  Used by op-table dispatch — primitive numeric ops
        only take/return scalars."""
        ty = self._storage_for_expr(e)
        if not isinstance(ty, CppScalar):
            raise CppInternalError(
                f'expected scalar storage for {type(e).__name__}, got {ty!r}',
                at=e,
            )
        return ty

    def _maybe_cast(
        self, arg: str, arg_ty: CppScalar, target_ty: CppScalar,
        *, at: Ast | None = None,
    ) -> str:
        """Emit *arg* in *target_ty* form, rejecting unsafe casts.

        For conversions implicit from the user's perspective, where format inference
        and storage selection decided to rebind the operand.  A lossy one is refused
        rather than silently emitted: the user should narrow the active context or
        write ``fp.round(...)``.  Casts the user *did* write go through
        :meth:`_explicit_cast`, which never refuses.
        """
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

    @staticmethod
    def _literal_cpp_type(e: RationalVal) -> CppScalar | None:
        """The C++ type of the token *e* prints as, or ``None`` for no literal.

        Not the same question as its *storage*, which
        :class:`StorageAnalysis` picks from the literal's value.  A token has
        whatever type C++ gives it, which is what
        :func:`_value_cpp_type` answers.
        """
        return _value_cpp_type(e.as_rational())

    def _call_arg(self, code: str, e: Expr, want: CppScalar) -> str:
        """*code* as a call argument of type *want*, spelling a literal's type.

        A literal matches the op table on its *storage*, so ``1.5`` under FP32
        matches the ``float`` signature while the token is a ``double``, and
        nothing inserts a cast.  Harmless where a declaration supplies the type
        (and exact for ``+ - * /``, since double rounding equals single rounding
        when ``2p + 2 <= 53``).  Not harmless where the callee takes its type
        *from* the argument: ``fpy::min``/``max`` are templates, so mixed types
        fail to deduce, and the ``<cmath>`` overload sets pick the wider
        overload -- which for ``fma``, outside ``2p + 2``, rounds twice.
        """
        if not isinstance(e, RationalVal):
            return code
        if self._literal_cpp_type(e) == want:
            return code
        return self._explicit_cast(code, want)

    def _explicit_cast(self, arg: str, target_ty: CppScalar) -> str:
        """``static_cast<target>(arg)``, unconditionally.

        For casts the user wrote (``Round``/``Cast``) or the language requires
        (``size_t`` subscripts); these callers accept a lossy conversion, unlike
        :meth:`_maybe_cast`."""
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
            raise CppInternalError(
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

    def _dispatch(
        self,
        e: UnaryOp | BinaryOp | TernaryOp,
        table,
        operands: Sequence[tuple[str, Expr]],
    ) -> str:
        """Emit *e* via the op table, choosing a signature in three phases.

        1. **Direct match** -- every operand's storage and the active context
           already equal a signature's slots, so no conversion is needed.
        2. **Cast-to-active** -- no direct match, but the active context has an
           all-same-type signature whose width holds every operand losslessly.
           Skipped when the active context has no C++ storage, e.g. ``REAL``.
        3. **Lossless widening** -- sound only under ``REAL``, where the wider
           C++ op produces the exact mathematical result and rounds to itself.
           Under any other context the wider op rounds differently than that
           context demands.  See :meth:`_try_widen`.

        A literal operand of a call-form signature needs its type spelled even
        on a direct match: it matched on storage, not on the token's own type.
        """
        sigs = table.get(type(e))
        if sigs is None:
            raise CppEmitError(
                f'no signatures for op: {type(e).__name__}', at=e,
            )
        codes = [code for code, _ in operands]
        srcs = [src for _, src in operands]
        storages = [self._scalar_storage_for_expr(src) for src in srcs]
        active = self._active_ctx_for(e)

        def spell(args: list[str], slots) -> list[str]:
            return [
                self._call_arg(a, src, slot)
                for a, src, slot in zip(args, srcs, slots)
            ]

        # (1) direct match
        for sig in sigs:
            if sig.matches(tuple(storages), active):
                out = spell(codes, sig.in_tys) if sig.is_call else codes
                return sig.format(*out)

        # (2) cast every operand into the active context's storage
        try:
            target = self._scalar_for_ctx(active, at=e)
        except CppEmitError:
            target = None
        if target is not None:
            want = (target,) * len(operands)
            for sig in sigs:
                if sig.in_tys == want and sig.out_ctx == active:
                    casts = [
                        self._maybe_cast(code, have, target, at=e)
                        for code, have in zip(codes, storages)
                    ]
                    if sig.is_call:
                        casts = spell(casts, want)
                    return sig.format(*casts)

        # (3) widen, only under REAL
        if active is REAL:
            widened = self._try_widen(e, sigs, list(zip(codes, storages)))
            if widened is not None:
                return widened

        raise CppEmitError(
            f'no matching signature for {type(e).__name__} under context '
            f'`{active}`: {[s.format() for s in storages]}',
            at=e,
        )

    def _dispatch_unary(self, e: UnaryOp, arg: str) -> str:
        return self._dispatch(e, self.op_table.unary, [(arg, e.arg)])

    def _dispatch_binary(self, e: BinaryOp, lhs: str, rhs: str) -> str:
        return self._dispatch(
            e, self.op_table.binary, [(lhs, e.first), (rhs, e.second)],
        )

    def _dispatch_ternary(
        self, e: TernaryOp, a1: str, a2: str, a3: str,
    ) -> str:
        return self._dispatch(
            e, self.op_table.ternary, list(zip([a1, a2, a3], e.args)),
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

    def _try_widen(
        self,
        e: Expr,
        sigs: Sequence[CppOp],
        operands: Sequence[tuple[str, CppScalar]],
    ) -> str | None:
        """Pick a signature whose output context contains the exact unrounded
        result of *e* -- so the op under that signature is the identity -- and
        whose input slots losslessly receive *operands*.  Its output is cast down
        to the expression's own storage, sound because the runtime value lies in
        ``format_info.by_expr[e]``, which fits ``result_ty`` by storage
        selection.  ``None`` when no signature qualifies.

        Two passes, to prefer a narrower signature: first those whose output
        already *is* ``result_ty``, then those needing the downcast.
        """
        try:
            result_ty = self._storage_for_expr(e)
        except CppEmitError:
            return None
        if not isinstance(result_ty, CppScalar):
            return None

        def _try(sig, *, exact_out: bool) -> str | None:
            try:
                sig_out_ty = self._scalar_for_ctx(sig.out_ctx)
            except CppEmitError:
                return None
            if exact_out and sig_out_ty is not result_ty:
                return None
            slots = sig.in_tys
            if not all(
                scalar_fits_in(have, want)
                for (_, have), want in zip(operands, slots)
            ):
                return None
            if not self._result_fits_ctx(e, sig.out_ctx):
                return None
            try:
                casts = [
                    self._maybe_cast(code, have, want, at=e)
                    for (code, have), want in zip(operands, slots)
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




    def _visit_unaryop(self, e: UnaryOp, ctx) -> str:
        if isinstance(e, (Fst, Snd)):
            # Tuple accessors fold as a chain (e.g. ``fst(snd(e))``) into a
            # single ``std::get``, so they bypass the eager operand visit
            # below — see ``_emit_tuple_accessor``.
            return self._emit_tuple_accessor(e, ctx)
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
                arg_ty = self._storage_for_expr(e.arg)
                return (
                    f'static_cast<{result_ty.format()}>'
                    f'({self._list_len(arg_ty, arg)})'
                )
            case Sum():
                return self._emit_sum(e, arg)
            case AMin() | AMax():
                return self._emit_amin_amax(e, arg)
            case AnyOf() | AllOf():
                return self._emit_any_all(e, arg)
            case Enumerate():
                return self._emit_enumerate(e, arg)
            case UnaryOp() if type(e) in self.op_table.unary:
                # Op-table-dispatched unary (Neg, Abs, all <cmath>).
                return self._dispatch_unary(e, arg)
            case _:
                raise CppEmitError(
                    f'unsupported unary op: {type(e).__name__}', at=e,
                )

    def _emit_tuple_accessor(self, e: UnaryOp, ctx) -> str:
        """Emit a (possibly nested) ``fst``/``snd`` chain.

        The chain is folded to a single ``std::get`` whenever it reads one
        element of a tuple — e.g. ``fst(snd(e))`` over a 3+-tuple is
        ``std::get<1>(e)`` rather than ``std::get<0>`` of a freshly built
        tail tuple.  Only a chain that genuinely yields a shorter tuple (a
        ``snd`` whose result is still multi-element and is *not* consumed by
        an outer ``fst``) materializes a ``std::make_tuple``.
        """
        acc = self._tuple_access(e, ctx)
        if acc.off is None:
            return acc.s
        # Unconsumed tail: materialize the remaining elements as a new tuple,
        # binding the base to a temp so it is evaluated once across the gets.
        assert isinstance(acc.ty, CppTuple)
        tmp = self._bind_operand(acc.s)
        gets = ', '.join(
            f'std::get<{i}>({tmp})' for i in range(acc.off, len(acc.ty.elts))
        )
        return f'std::make_tuple({gets})'

    def _tuple_access(self, e: Expr, ctx) -> _TupleAccess:
        """Resolve a ``fst``/``snd`` chain over a tuple into a
        :class:`_TupleAccess`, peeling the nesting and tracking the net
        offset into the underlying base tuple."""
        match e:
            case Fst():
                inner = self._tuple_access(e.arg, ctx)
                base_s, base_ty = self._as_tuple(inner, e)
                off = 0 if inner.off is None else inner.off
                return _TupleAccess(f'std::get<{off}>({base_s})', base_ty.elts[off], None)
            case Snd():
                inner = self._tuple_access(e.arg, ctx)
                base_s, base_ty = self._as_tuple(inner, e)
                off = (0 if inner.off is None else inner.off) + 1
                if len(base_ty.elts) - off == 1:
                    # the tail is a single (bare) element
                    return _TupleAccess(f'std::get<{off}>({base_s})', base_ty.elts[off], None)
                return _TupleAccess(base_s, base_ty, off)
            case _:
                # opaque base: emit it once, treat as a finished value
                return _TupleAccess(self._visit_expr(e, ctx), self._storage_for_expr(e), None)

    def _as_tuple(self, acc: _TupleAccess, e: Expr) -> tuple[str, CppTuple]:
        """The (base string, tuple type) of *acc*, which must be a tuple."""
        if not isinstance(acc.ty, CppTuple):
            raise CppInternalError(
                f'tuple accessor expects a tuple, got {acc.ty.format()}', at=e,
            )
        return acc.s, acc.ty

    def _emit_enumerate(self, e: Enumerate, src_str: str) -> str:
        """``enumerate(xs)`` builds a ``fpy::list<std::tuple<I, T>>``
        where ``I`` is the index integer type and ``T`` is the source
        element type — both come from format inference on the
        Enumerate node itself.
        """
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)
                and len(result_ty.elt.elts) == 2):
            raise CppInternalError(
                'expected list[(int, T)] for enumerate result, '
                f'got {result_ty!r}', at=e,
            )
        idx_ty = result_ty.elt.elts[0]

        src_ty = self._storage_for_expr(e.args[0])
        src = self._bind_operand(src_str)
        result = self._fresh_temp()
        self.writer.add_line(
            f'{result_ty.format()} {result} = '
            f'{self._list_new_sized(result_ty, self._list_len(src_ty, src))};'
        )
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 0; {i} < '
            f'{self._list_len(src_ty, src)}; ++{i}) {{'
        )
        self.writer.indent()
        self.writer.add_line(
            f'{self._list_at_raw(result_ty, result, i)} = std::make_tuple('
            f'static_cast<{idx_ty.format()}>({i}), '
            f'{self._list_at_raw(src_ty, src, i)});'
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
            raise CppInternalError(
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
                # ``range(stop)`` with ``stop <= 0`` is empty; clamp before
                # the unsigned cast so a negative stop doesn't wrap to a
                # huge allocation.
                size_expr = (
                    f'static_cast<size_t>({stop_cast} > 0 ? {stop_cast} : 0)'
                )
                self.writer.add_line(
                    f'{result_ty.format()} {tmp} = '
                    f'{self._list_new_sized(result_ty, size_expr)};'
                )
                self.writer.add_line(
                    f'std::iota({self._list_begin(result_ty, tmp)}, '
                    f'{self._list_end(result_ty, tmp)}, '
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
                    f'{result_ty.format()} {tmp} = '
                    f'{self._list_new_sized(result_ty, size_expr)};'
                )
                self.writer.add_line(
                    f'std::iota({self._list_begin(result_ty, tmp)}, '
                    f'{self._list_end(result_ty, tmp)}, {start_cast});'
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
                self.writer.add_line(
                    f'{result_ty.format()} {tmp} = '
                    f'{self._list_empty(result_ty)};'
                )
                self.writer.add_line(
                    f'for ({int_ty} {ctr} = {start_cast}; '
                    f'{ctr} < {stop_cast}; {ctr} += {step_cast}) {{'
                )
                self.writer.indent()
                self.writer.add_line(
                    f'{self._list_push(result_ty, tmp, ctr)};'
                )
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
        access = self._visit_expr(e.first, ctx)
        level = xs_ty
        for _ in range(d):
            assert isinstance(level, CppList)   # validated by the walk above
            access = self._list_at(level, access, '0')
            level = level.elt
        result_ty = self._storage_for_expr(e)
        return (
            f'static_cast<{result_ty.format()}>'
            f'({self._list_len(level, access)})'
        )

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
        # `a < b < c` expands to pairwise `&&`, sound because the operands are
        # pure.  Each pair is cast to its scalar supremum so the comparison
        # happens in a defined common type.
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
            raise CppInternalError(
                f'{type(e).__name__} requires at least one argument',
                at=e,
            )
        active = self._active_ctx_for(e)
        target = self._scalar_for_ctx(active, at=e)
        args = [self._visit_expr(a, ctx) for a in e.args]
        arg_storages = [self._scalar_storage_for_expr(a) for a in e.args]
        casted = [
            self._call_arg(self._maybe_cast(a, s, target, at=e), src, target)
            for a, s, src in zip(args, arg_storages, e.args)
        ]
        if target.is_float():
            # NaN-propagating wrapper around ``std::fmin`` / ``std::fmax``
            # (see :data:`CPP_HELPERS`).  ±0 ordering is delegated to the
            # underlying ``std::fmin`` / ``std::fmax``.
            fn = 'fpy::min' if isinstance(e, Min) else 'fpy::max'
        else:
            fn = 'std::min' if isinstance(e, Min) else 'std::max'
        result = casted[0]
        for nxt in casted[1:]:
            result = f'{fn}({result}, {nxt})'
        return result

    def _emit_sum(self, e: Sum, arg: str) -> str:
        """``sum(xs)`` as the fold the interpreter performs.

        ``_eval_sum`` seeds with ``xs[0]`` **unrounded** and does *n-1* additions;
        an empty list is an exact ``+0``.  ``accumulate`` takes seed and range
        separately, so that is a range starting one past ``begin``.  The empty guard
        is not optional -- both ``begin() + 1`` and ``xs[0]`` are undefined there.

        The accumulator may be *wider* than the element but not narrower:
        ``init + *first`` converts to the common type, which is the accumulator only
        when the element fits it exactly, making the fold uni-precision there as the
        interpreter is.  Otherwise the seed would round.
        """
        result_ty = self._storage_for_expr(e)
        arg_ty = self._storage_for_expr(e.arg)
        elt_ty = arg_ty.elt if isinstance(arg_ty, CppList) else None
        if not isinstance(elt_ty, CppScalar) or not isinstance(result_ty, CppScalar):
            raise CppInternalError(
                f'expected scalar element and result for `sum`, got '
                f'{elt_ty!r} and {result_ty!r}',
                at=e,
            )
        if not scalar_fits_in(elt_ty, result_ty):
            raise CppEmitError(
                f'unsupported: `sum` over `{elt_ty.format()}` elements '
                f'accumulating in `{result_ty.format()}`, which cannot hold one '
                f'exactly.  Use a context whose format contains the element '
                f'format.',
                at=e,
            )
        # A prvalue operand (a list literal) would otherwise give begin() and
        # end() into *different* temporaries; binding also keeps the three uses
        # below to one evaluation.
        src = self._bind_operand(arg)
        seed = self._list_at(arg_ty, src, '0')
        if elt_ty != result_ty:
            # `accumulate` deduces `T` from its seed, so an uncast one would run
            # the whole fold in the element type.  Exact, by the check above.
            seed = self._explicit_cast(seed, result_ty)
        return (
            f'({self._list_len(arg_ty, src)} == 0'
            f' ? static_cast<{result_ty.format()}>(0)'
            f' : std::accumulate({self._list_begin(arg_ty, src)} + 1, '
            f'{self._list_end(arg_ty, src)}, {seed}))'
        )

    def _emit_any_all(self, e: 'AnyOf | AllOf', arg_str: str) -> str:
        """``any(bs)`` / ``all(bs)`` -> ``std::any_of`` / ``std::all_of`` with an
        identity predicate.  The empty range agrees with FPy (``all_of`` is
        ``true``, ``any_of`` is ``false``), so unlike ``min``/``max`` there is
        no unguarded ``xs[0]``."""
        arg_storage = self._storage_for_expr(e.arg)
        # element type only: a bool list qualifies whichever way it is
        # represented, and `CppList.__eq__` distinguishes those
        if not (
            isinstance(arg_storage, CppList)
            and arg_storage.elt == CppScalar.BOOL
        ):
            raise CppInternalError(
                f'expected list[bool] arg for {type(e).__name__}, '
                f'got {arg_storage!r}',
                at=e,
            )
        fn = 'std::any_of' if isinstance(e, AnyOf) else 'std::all_of'
        # Bind first, as ``Sum`` does: on a prvalue operand ``begin()`` and
        # ``end()`` would name iterators into different temporaries.
        src = self._bind_operand(arg_str)
        pred = self._fresh_temp()
        return (
            f'{fn}({self._list_begin(arg_storage, src)}, '
            f'{self._list_end(arg_storage, src)}, '
            f'[](bool {pred}) {{ return {pred}; }})'
        )

    def _emit_amin_amax(self, e: 'AMin | AMax', arg_str: str) -> str:
        """Reduce ``min(xs)`` / ``max(xs)`` to a hoisted for-loop.

        Combiner mirrors :meth:`_emit_min_max`: ``std::fmin``/``fmax``
        for FP results, ``std::min``/``max`` (templated) for integer
        ones.  Both demand uniform operand types — we cast each element
        to ``result_ty`` via :meth:`_maybe_cast`, which raises on a
        lossy cast.

        Empty-list behavior is undefined (matches the interpreter's
        ``min([]) → ValueError`` contract); the emit indexes ``xs[0]``
        without a guard.
        """
        result_ty = self._storage_for_expr(e)
        if not isinstance(result_ty, CppScalar):
            raise CppInternalError(
                f'expected scalar result for {type(e).__name__}, got {result_ty!r}',
                at=e,
            )
        arg_storage = self._storage_for_expr(e.arg)
        if not (isinstance(arg_storage, CppList)
                and isinstance(arg_storage.elt, CppScalar)):
            raise CppInternalError(
                f'expected list[scalar] arg for {type(e).__name__}, '
                f'got {arg_storage!r}',
                at=e,
            )
        elt_ty = arg_storage.elt
        if result_ty.is_float():
            # NaN-propagating wrapper around ``std::fmin`` / ``std::fmax``;
            # see :meth:`_emit_min_max` and :data:`CPP_HELPERS`.
            fn = 'fpy::min' if isinstance(e, AMin) else 'fpy::max'
        else:
            fn = 'std::min' if isinstance(e, AMin) else 'std::max'

        src = self._bind_operand(arg_str)
        acc = self._fresh_temp()
        first = self._list_at_raw(arg_storage, src, '0')
        init = self._maybe_cast(first, elt_ty, result_ty, at=e)
        self.writer.add_line(f'{result_ty.format()} {acc} = {init};')
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 1; {i} < '
            f'{self._list_len(arg_storage, src)}; ++{i}) {{'
        )
        self.writer.indent()
        elt = self._maybe_cast(
            self._list_at_raw(arg_storage, src, i), elt_ty, result_ty, at=e,
        )
        self.writer.add_line(f'{acc} = {fn}({acc}, {elt});')
        self.writer.dedent()
        self.writer.add_line('}')
        return acc

    def _emit_empty(self, e: Empty, ctx) -> str:
        """``empty(d1, ..., dN)`` builds an ``N``-dimensional zero-
        initialised vector.  The result's storage shape comes from
        format inference; we read the dimension sizes off the call
        site and emit nested ``fpy::make_list<T>(d, ...)`` calls
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
            inner = self._list_new_filled(layer, d, inner)
        return inner

    def _emit_zip(self, e: Zip, ctx) -> str:
        """``zip(xs1, …, xsN)`` builds a
        ``fpy::list<std::tuple<T1, …, TN>>`` whose length matches the
        first iterable.  Each iterable is bound to a temp once to
        evaluate side-effects in source order; the tuple type comes
        from format inference on the Zip node."""
        result_ty = self._storage_for_expr(e)
        if not (isinstance(result_ty, CppList)
                and isinstance(result_ty.elt, CppTuple)):
            raise CppInternalError(
                f'expected list[tuple[...]] for zip result, got {result_ty!r}',
                at=e,
            )

        srcs: list[tuple[CppType, str]] = []
        for arg in e.args:
            arg_str = self._visit_expr(arg, ctx)
            srcs.append(
                (self._storage_for_expr(arg), self._bind_operand(arg_str)),
            )

        head_ty, head = srcs[0]
        result = self._fresh_temp()
        self.writer.add_line(
            f'{result_ty.format()} {result} = '
            f'{self._list_new_sized(result_ty, self._list_len(head_ty, head))};'
        )
        i = self._fresh_temp()
        self.writer.add_line(
            f'for (size_t {i} = 0; {i} < '
            f'{self._list_len(head_ty, head)}; ++{i}) {{'
        )
        self.writer.indent()
        elts = ', '.join(
            self._list_at_raw(ty, src, i) for ty, src in srcs
        )
        self.writer.add_line(
            f'{self._list_at_raw(result_ty, result, i)} = '
            f'std::make_tuple({elts});'
        )
        self.writer.dedent()
        self.writer.add_line('}')
        return result
    def _scalar_cast_types(self, e):
        """Source/target scalar storage for a round-like node ``e``.

        The argument's storage is only used to short-circuit same-type casts.
        A non-dyadic literal has a ``SetFormat`` with no representable storage
        — fine, we always cast.  ``Round`` folds those before getting here
        (:meth:`_fold_rounded_literal`); ``Cast`` does not, and refuses them at
        emission instead, which is right for a cast that asserts exactness."""
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

    def _fold_rounded_literal(self, e) -> str | None:
        """``Round(<literal>)`` as a C++ literal, or ``None`` to emit a cast.

        The only way an inexact constant reaches C++ at all, since the argument
        has no representation of its own (:meth:`_emit_numeric_literal`).
        Rounding at compile time also uses the mode the program asked for
        rather than whatever ``fesetround`` last left behind.
        """
        if not isinstance(e.arg, RationalVal):
            return None
        active = self._active_ctx_for(e)
        if not isinstance(active, EFloatContext):
            return None
        rounded = active.round(e.arg.as_rational())
        if rounded.isinf or rounded.isnan:
            # An overflowing literal is a value the target format does have,
            # but ``HUGE_VAL``/``NAN`` are a separate spelling; leave it.
            return None
        if rounded.is_zero() and rounded.s:
            # Underflow to *negative* zero.  It has to be spelled here: a
            # `Fraction` has no signed zero, so going through `as_rational`
            # below would emit `0` and turn `x / -0.0` from -inf into +inf.
            lit = '-0.0'
        else:
            v = rounded.as_rational()
            if _as_exact_double(v) is None:
                return None
            lit = self._emit_numeric_literal(v, at=e)
        # The literal prints as a ``double``.  A narrower target still needs
        # its cast, but not a rounding one: the value is already in that
        # format, so no mode can change it.
        target_ty = self._scalar_for_ctx(active, at=e)
        if target_ty == CppScalar.F64:
            return lit
        return self._explicit_cast(lit, target_ty)

    def _visit_round(self, e, ctx) -> str:
        # ``Round(arg)`` rounds ``arg`` to the active rounding
        # context — emitted as a plain ``static_cast`` (the cast's
        # rounding mode is the active ``fesetround`` mode set by
        # Phase 5b at the surrounding ``with`` boundary).  The user
        # explicitly asked to round into the active context, so the
        # cast is emitted even when lossy.  Same-type short-circuits
        # to a no-op.  A literal argument is rounded here instead.
        folded = self._fold_rounded_literal(e)
        if folded is not None:
            return folded
        arg = self._visit_expr(e.arg, ctx)
        arg_ty, target_ty = self._scalar_cast_types(e)
        if arg_ty == target_ty:
            return arg
        return self._explicit_cast(arg, target_ty)

    def _visit_round_at(self, e, ctx):
        self._unsupported('RoundAt', at=e)

    def _visit_call(self, e, ctx) -> str:
        # The emitted name comes from the compiler's per-instantiation
        # mangling in :attr:`_call_names`: one name per (callee, outer_ctx), so
        # call sites at the same context share a specialization.
        if e.kwargs:
            raise CppEmitError(
                f'unsupported call: kwargs are not supported '
                f'(call to `{e.func}`)',
                at=e,
            )
        abi = None
        if isinstance(e.fn, Function):
            abi = self._callee_params.get(e.fn.ast)
        params = abi.params if abi is not None else None
        parts = []
        for i, a in enumerate(e.args):
            want = params[i].ty if params and i < len(params) else None
            parts.append(self._adapt_arg(self._visit_expr(a, ctx), a, want))
        target = self._call_names.get(e, str(e.func))
        call = f'{target}({", ".join(parts)})'
        return self._adapt_result(call, e, abi.ret if abi is not None else None)

    def _adapt_result(self, emitted: str, e: Expr, got: CppType | None) -> str:
        """*emitted* as the caller needs the result.

        A callee's return representation is fixed by its own body, so a caller
        that keeps a handle for a local reason has to make one.  The result is a
        prvalue whose ownership the callee handed over — that is why it unboxed
        — so this moves the elements rather than copying them.

        Only this direction: :func:`~fpy2.backend.cpp.unbox._boxed_by_callees`
        gives the caller a handle whenever the callee returns one.
        """
        if got is None or self.unbox is None:
            return emitted
        want = self._storage_for_expr(e)
        if not (isinstance(got, CppList) and isinstance(want, CppList)):
            return emitted
        if got.boxed == want.boxed and got.elt == want.elt:
            return emitted
        if got.boxed or got.elt != want.elt:
            raise CppEmitError(
                f'cannot hand back `{got.format()}` where `{want.format()}` '
                f'is wanted',
                at=e,
            )
        return f'std::make_shared<{got.format()}>({emitted})'

    def _adapt_arg(self, emitted: str, e: Expr, want: CppType | None) -> str:
        """*emitted* as the callee spelled its parameter.

        A callee's signature is fixed by its own body, so a caller holding a handle
        the callee does not want hands over the pointee -- same elements, no copy,
        and a write still reaches the caller.  Only this direction arises: ``unbox``
        gives an argument a handle whenever the callee declared one.  Just as well,
        since the reverse needs ``fpy::borrow``, which cannot bind a ``const``
        reference.
        """
        if want is None or self.unbox is None:
            return emitted
        have = self._storage_or_none(e)
        if not (isinstance(have, CppList) and isinstance(want, CppList)):
            self._require_bridgeable(have, want, e)
            return emitted
        if have.elt != want.elt:
            # The callee declared a different element type; nothing at the call
            # site can bridge two `std::vector` instantiations.
            raise self._refuse_mismatch(have, want, e)
        if have.boxed == want.boxed:
            return emitted
        if not have.boxed:
            raise CppEmitError(
                f'passing an unboxed list where `{want.format()}` is declared',
                at=e,
            )
        return f'*{self._bind_operand(emitted)}'

    def _visit_tuple_expr(self, e: TupleExpr, ctx) -> str:
        # Through `_emit_at`: `std::make_tuple` deduces from its arguments, so
        # a narrower field must be converted or deduction yields a tuple type
        # nothing accepts.
        return self._emit_at(e, self._storage_for_expr(e), ctx)

    def _visit_list_expr(self, e: ListExpr, ctx) -> str:
        # Through `_emit_at`: one vector means one element type, so each
        # element is emitted at the list's `T`, not at its own bound.
        return self._emit_at(e, self._storage_for_expr(e), ctx)

    def _visit_list_comp(self, e: ListComp, ctx) -> str:
        return self._emit_list_comp_at(e, self._storage_for_expr(e), ctx)

    def _emit_list_comp_at(self, e: ListComp, result_ty: CppType, ctx) -> str:
        # A temp plus nested loops that `push_back` the element expression.
        # *result_ty* is supplied because a comprehension builds its vector
        # element by element, making it a contributor to whatever place it
        # reaches -- see :meth:`_emit_at`.
        tmp = self._fresh_temp()
        self.writer.add_line(
                    f'{result_ty.format()} {tmp} = '
                    f'{self._list_empty(result_ty)};'
                )

        for target, iterable in zip(e.targets, e.iterables):
            self._open_comp_loop(target, iterable, e, ctx)

        elt_want = result_ty.elt if isinstance(result_ty, CppList) else None
        elt = self._emit_at(e.elt, elt_want, ctx)
        self.writer.add_line(f'{self._list_push(result_ty, tmp, elt)};')

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
        # ``loop_def`` drives the value-iterable element binding (const&
        # for read-only aggregates); ``None`` => discarded/anonymous.
        loop_def = None
        match target:
            case NamedId():
                target_def = self.def_use.find_def_from_site(target, comp_site)
                target_name = self.storage.def_to_name[target_def]
                # For a range counter, size to the exit-test overshoot (see
                # :meth:`_range_counter_scalar`); the element storage would be
                # too narrow.  Non-range iterables fall back to element storage.
                counter_scalar = self._range_counter_scalar(iterable)
                storage = counter_scalar or self.storage.storage_of(target_def)
                decl = f'{storage.format()} {target_name}'
                loop_def = target_def
            case UnderscoreId():
                # ``_`` discards the loop variable — no SSA def, no
                # storage class.  Synthesize a fresh name and pick the
                # iterator type from the range counter's trajectory (or the
                # stop bound's storage when the range isn't concrete); value
                # iterables use ``auto`` and let the for-range loop deduce.
                target_name = self._fresh_temp()
                counter_scalar = self._range_counter_scalar(iterable)
                match iterable:
                    case Range1() if counter_scalar is not None:
                        decl = f'{counter_scalar.format()} {target_name}'
                    case Range1():
                        stop_ty = self._scalar_storage_for_expr(iterable.arg)
                        decl = f'{stop_ty.format()} {target_name}'
                    case (Range2() | Range3()) if counter_scalar is not None:
                        decl = f'{counter_scalar.format()} {target_name}'
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
                iter_ty = self._storage_for_expr(iterable)
                # element read-only (only destructured) -> bind by const&
                self.writer.add_line(
                    f'for (const auto& {tmp} : '
                    f'{self._list_range(iter_ty, iter_str)}) {{'
                )
                self.writer.indent()
                self._destructure(
                    target, tmp, comp_site,
                    iter_ty.elt if isinstance(iter_ty, CppList) else None,
                    iterable,
                )
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
                iter_ty = self._storage_for_expr(iterable)
                decl = self._foreach_decl(
                    loop_def, target_name,
                    elt=iter_ty.elt if isinstance(iter_ty, CppList) else None,
                    at=iterable,
                )
                self.writer.add_line(
                    f'for ({decl} : {self._list_range(iter_ty, iter_str)}) {{'
                )
        self.writer.indent()

    def _visit_list_ref(self, e: ListRef, ctx) -> str:
        # ``xs[i]`` — C++ ``operator[]`` takes ``size_t``, so we route
        # the index through an explicit ``static_cast<size_t>`` rather
        # than relying on implicit conversion.  Bounds-checking is
        # still TODO (FPy's interpreter is strict; we currently match
        # C++'s undefined-behaviour-on-out-of-range).
        value = self._visit_expr(e.value, ctx)
        index = self._visit_expr(e.index, ctx)
        return self._list_at(self._storage_for_expr(e.value), value, index)

    def _visit_list_slice(self, e: ListSlice, ctx) -> str:
        # `auto&& _tmpN = <xs>` binds an lvalue without copying and
        # lifetime-extends a prvalue, so `<xs>` is evaluated exactly once, as
        # the interpreter does.  Bounds-checking is a TODO; see backend-cpp.md.
        arr_ty = self._storage_for_expr(e.value)
        arr_tmp = self._bind_operand(self._visit_expr(e.value, ctx))

        if e.start is None:
            start = '0'
        else:
            start = f'static_cast<size_t>({self._visit_expr(e.start, ctx)})'
        if e.stop is None:
            stop = self._list_len(arr_ty, arr_tmp)
        else:
            stop = f'static_cast<size_t>({self._visit_expr(e.stop, ctx)})'

        result_ty = self._storage_for_expr(e)
        begin = self._list_begin(arr_ty, arr_tmp)
        return (
            self._list_new_range(
                result_ty,
                f'{begin} + {start}',
                f'{begin} + {stop}',
            )
        )
    def _visit_if_expr(self, e, ctx) -> str:
        # ``cond ? ift : iff`` — both branches must share a C++ type,
        # so when their storages differ we cast each to the IfExpr's
        # unified storage (chosen by format inference + storage
        # selection over the merged formats).  Non-scalar branches
        # must already match — there's no widening for lists/tuples.
        out_ty = self._storage_for_expr(e)
        if not isinstance(out_ty, CppScalar):
            # A list or tuple arm is built at the ternary's storage, or
            # converted into it — see :meth:`_emit_at`.
            return self._emit_at(e, out_ty, ctx)
        cond = self._visit_expr(e.cond, ctx)
        ift = self._visit_expr(e.ift, ctx)
        iff = self._visit_expr(e.iff, ctx)
        ift_ty = self._scalar_storage_for_expr(e.ift)
        iff_ty = self._scalar_storage_for_expr(e.iff)
        ift = self._maybe_cast(ift, ift_ty, out_ty, at=e)
        iff = self._maybe_cast(iff, iff_ty, out_ty, at=e)
        return f'({cond} ? {ift} : {iff})'

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        # ``xs[i1]…[iN] = e`` is in-place mutation in C++.  The
        # post-mutation SSA def of ``xs`` shares a storage class with
        # its ``prev`` (see ``same_object_defs`` in
        # ``reaching_defs``), so the C++ name is the same on both
        # sides — emit a direct subscript-store.
        target_def = self.def_use.find_def_from_site(stmt.var, stmt)
        target_name = self.storage.def_to_name[target_def]
        idxs = [self._visit_expr(idx, ctx) for idx in stmt.indices]
        chain = target_name
        level = self._emitted_storage_of(target_def)
        for idx in idxs:
            if not isinstance(level, CppList):
                raise CppInternalError(
                    f'`{stmt.var}` is indexed {len(idxs)} deep but its storage '
                    f'is `{level!r}`',
                    at=stmt,
                )
            chain = self._list_at(level, chain, idx)
            level = level.elt
        # The slot's type comes from the container, so there is nowhere to put
        # a conversion: the value has to already fit.
        src = self._storage_or_none(stmt.expr)
        self._require_bridgeable(src, level, stmt.expr)
        self._require_no_narrowing(src, level, stmt.expr)
        rhs = self._emit_at(stmt.expr, level, ctx)
        self.writer.add_line(f'{chain} = {rhs};')

    def _emit_guarded_block(self, keyword: str, cond, body, ctx) -> None:
        """``<keyword> (<cond>) { <body> }``."""
        self.writer.add_line(f'{keyword} ({self._visit_expr(cond, ctx)}) {{')
        self.writer.indent()
        self._visit_block(body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _visit_if1(self, stmt: If1Stmt, ctx):
        self._emit_guarded_block('if', stmt.cond, stmt.body, ctx)

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
        self._emit_guarded_block('while', stmt.cond, stmt.body, ctx)

    def _concrete_int_of(self, e: Expr) -> int | None:
        """Concrete integer value of *e* per format inference, or ``None``."""
        fmt = self.format_info.by_expr.get(e)
        if isinstance(fmt, SetFormat) and len(fmt.values) == 1:
            (v,) = fmt.values
            # A `SetFormat` value may be `NEG_ZERO`, which is not an integer --
            # no `range` bound may be one, and it has no `denominator`.
            if isinstance(v, Fraction) and v.denominator == 1:
                return v.numerator
        return None

    def _range_counter_scalar(self, iterable: Expr) -> 'CppScalar | None':
        """Counter type for a ``range`` for-loop, or ``None`` when the bounds
        aren't statically concrete.

        The C-style counter transiently reaches the first value past ``stop``
        (the exit-test overshoot), which the loop variable's *element* format
        (the values actually taken) excludes — so a counter typed from the
        element storage can overflow at a type boundary (e.g. ``range(128)``
        would overflow ``int8_t``).  We size it to cover ``start`` and that
        overshoot instead.
        """
        start: int | None
        stop: int | None
        step: int | None
        match iterable:
            case Range1():
                start, stop, step = 0, self._concrete_int_of(iterable.arg), 1
            case Range2():
                start = self._concrete_int_of(iterable.first)
                stop = self._concrete_int_of(iterable.second)
                step = 1
            case Range3():
                start = self._concrete_int_of(iterable.args[0])
                stop = self._concrete_int_of(iterable.args[1])
                step = self._concrete_int_of(iterable.args[2])
            case _:
                return None
        if start is None or stop is None or step is None or step == 0:
            return None
        overshoot = start + len(range(start, stop, step)) * step
        b = RealFloat.from_int(max(abs(start), abs(overshoot)))
        scalar = choose_storage(AbstractFormat(float('inf'), 0, b).format())
        assert isinstance(scalar, CppScalar)
        return scalar

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

    def _emit_for_loop(self, stmt: ForStmt, ctx, target: str, decl: str,
                       target_def) -> None:
        """``for (<header>) { body }`` given the loop variable and its
        declaration.

        A ``Range*`` iterable is a counter loop and uses *decl* as given;
        anything else is a range-for over a container, where the element
        binding is :meth:`_foreach_decl`'s decision instead -- ``const&`` for a
        read-only aggregate, so no per-element copy.  A ``None`` *target_def*
        marks a discarded element, which that method binds ``const auto&``.
        """
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
                iter_ty = self._storage_for_expr(stmt.iterable)
                elt_decl = self._foreach_decl(
                    target_def, target,
                    elt=iter_ty.elt if isinstance(iter_ty, CppList) else None,
                    at=stmt.iterable,
                )
                header = (
                    f'for ({elt_decl} : '
                    f'{self._list_range(iter_ty, iter_str)})'
                )
        self.writer.add_line(f'{header} {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')

    def _emit_for_underscore_target(self, stmt: ForStmt, ctx):
        """``for _ in iter:`` -- the body never reads the counter, so emit a
        synthetic name and type it the way :meth:`_open_comp_loop` types an
        ``UnderscoreId`` comprehension target."""
        target = self._fresh_temp()
        counter_scalar = self._range_counter_scalar(stmt.iterable)
        if counter_scalar is not None:
            ty = counter_scalar.format()
        elif isinstance(stmt.iterable, Range1):
            ty = self._scalar_storage_for_expr(stmt.iterable.arg).format()
        elif isinstance(stmt.iterable, (Range2, Range3)):
            ty = self._scalar_storage_for_expr(stmt.iterable.args[1]).format()
        else:
            ty = 'auto'
        self._emit_for_loop(stmt, ctx, target, f'{ty} {target}', None)

    def _emit_for_named_target(self, stmt: ForStmt, ctx):
        assert isinstance(stmt.target, NamedId)
        target_def = self.def_use.find_def_from_site(stmt.target, stmt)
        target = self.storage.def_to_name[target_def]
        # Fold the type into the for header iff the counter is a
        # single-writer class (the common case).  Otherwise the counter
        # was hoisted at the function top and we just reassign here.
        if target_def in self.storage.declare_at_assign:
            # A range counter transiently reaches the exit-test overshoot,
            # which the loop variable's element storage may be too narrow to
            # hold; size it from the counter's real trajectory instead.
            counter_scalar = self._range_counter_scalar(stmt.iterable)
            storage = counter_scalar or self.storage.storage_of(target_def)
            decl = f'{storage.format()} {target}'
        else:
            decl = target
        self._emit_for_loop(stmt, ctx, target, decl, target_def)

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
        iter_ty = self._storage_for_expr(stmt.iterable)
        tmp = self._fresh_temp()
        # element read-only (only destructured) -> bind by const&
        self.writer.add_line(
            f'for (const auto& {tmp} : '
            f'{self._list_range(iter_ty, iter_str)}) {{'
        )
        self.writer.indent()
        assert isinstance(stmt.target, TupleBinding)
        self._destructure(
            stmt.target, tmp, stmt,
            iter_ty.elt if isinstance(iter_ty, CppList) else None,
            stmt.iterable,
        )
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
