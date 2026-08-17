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
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from fractions import Fraction
from typing import NoReturn

from ... import ops as fpy_ops
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
    exact_exp2,
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
    Const1_Pi,
    Const2_Pi,
    Const2_SqrtPi,
    ConstE,
    ConstInf,
    ConstLn2,
    ConstLog2E,
    ConstLog10E,
    ConstNan,
    ConstPi,
    ConstPi_2,
    ConstPi_4,
    ConstSqrt1_2,
    ConstSqrt2,
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
    Mul,
    NamedId,
    NaryOp,
    Not,
    NullaryOp,
    Or,
    Pow,
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
    OverflowMode,
    RealFloat,
)
from ...number.context.context import Context
from .ops import CppOp, ScalarOpTable
from .storage import (
    StorageSelectionError,
    bound_fits_in_scalar,
    choose_storage,
    scalar_fits_in,
    scalar_sup,
)
from .storage_infer import (
    StorageAnalysis,
    binds_by_reference,
    is_rebound,
)
from .target import is_native_ctx, make_op_table
from .types import CppList, CppScalar, CppTuple, CppType
from .unbox import ParamAbi, UnboxAnalysis, contains_boxed, return_storage

# Map FPy rounding modes to ``<cfenv>`` macros.  Only the four modes
# in this table can be set via ``fesetround``.
_FE_RM_MACRO: dict[RM, str] = {
    RM.RNE: 'FE_TONEAREST',
    RM.RTZ: 'FE_TOWARDZERO',
    RM.RTP: 'FE_UPWARD',
    RM.RTN: 'FE_DOWNWARD',
}

# Finite nullary constants.  C++11 has no spelling for these, but each is fixed
# once the active context is known, so it is evaluated and emitted as a literal.
# `ConstNan`/`ConstInf` have no literal form; `_visit_nullaryop` spells those.
_NULLARY_CONSTS: dict[type[NullaryOp], Callable[..., Float]] = {
    ConstPi: fpy_ops.const_pi,
    ConstE: fpy_ops.const_e,
    ConstLog2E: fpy_ops.const_log2e,
    ConstLog10E: fpy_ops.const_log10e,
    ConstLn2: fpy_ops.const_ln2,
    ConstPi_2: fpy_ops.const_pi_2,
    ConstPi_4: fpy_ops.const_pi_4,
    Const1_Pi: fpy_ops.const_1_pi,
    Const2_Pi: fpy_ops.const_2_pi,
    Const2_SqrtPi: fpy_ops.const_2_sqrt_pi,
    ConstSqrt2: fpy_ops.const_sqrt2,
    ConstSqrt1_2: fpy_ops.const_sqrt1_2,
}

def _value_cpp_type(v: Fraction) -> 'CppScalar | None':
    """The C++ type of the token *v* prints as, or ``None`` if none can hold it.

    Not its *storage*, which comes from the value: ``1.5`` is stored as a
    ``float`` while the token is a ``double``.  Bounds are on magnitude, since
    C++ has no negative literal -- ``-2**31`` is unary minus on ``2**31``, so the
    expression is a ``long``.  ``None`` means no integer literal fits and the
    caller must spell a float or refuse.
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
    """A folded ``fst``/``snd`` chain over a tuple.

    ``off is None`` means a finished value in ``s``; an ``int`` means the
    unmaterialized suffix ``base[off:]``, built into a ``std::make_tuple`` only
    if used -- which it is not when a ``fst`` reads one element out of it.
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
    """Raised for a program this backend cannot emit.

    An optional ``at`` node prefixes the message with a source location, which
    the wrapping :class:`CppCompileError` passes through untouched.
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

    A *backend bug*, not a program this backend cannot compile -- kept distinct
    so it does not reach the user as "your program is unsupported".  A subclass,
    so handlers are unchanged and the type survives on ``__cause__``, which is
    how ``test_internal_invariants.py`` tells the two apart.
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
        # Call -> mangled target name, one per (callee, outer_ctx).  A Call not
        # in the map falls back to the callee's declared name.
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
    # the boxed spelling in ``.types``.

    @staticmethod
    def _elt_of(ty: CppType) -> str:
        """The element type of a list storage type."""
        assert isinstance(ty, CppList), f'not a list storage type: {ty!r}'
        return ty.elt.format()

    def _strict_tripwire(self) -> CppEmitError:
        """The error for a handle reaching emission under STRICT.

        `check_strict` and `annotate` guarantee no boxed type survives to the
        emitter, so raising here means a backend bug, not a user program.
        """
        return CppEmitError(
            'internal error: strict unboxing let a shared handle reach '
            'emission'
        )

    def _is_boxed(self, ty: CppType) -> bool:
        """Whether *ty* is a handle rather than the sequence itself.

        Under STRICT this is the tripwire: every handle spelling branches on
        this predicate, so answering "yes" is refused at the source."""
        assert isinstance(ty, CppList), f'not a list storage type: {ty!r}'
        if ty.boxed and self._strict:
            raise self._strict_tripwire()
        return ty.boxed

    @property
    def _strict(self) -> bool:
        return self.unbox is not None and self.unbox.strict

    def _list_seq(self, ty: CppType, base: str) -> str:
        """*base* as the sequence itself — dereferenced if it is a handle.

        No parenthesis is needed unboxed: a list-valued C++ expression here is a
        name, an emitter temp, or a subscript chain, never an operator
        expression that could bind more loosely than ``[]``.
        """
        return f'(*{base})' if self._is_boxed(ty) else base

    def _member(self, ty: CppType, base: str) -> str:
        """``base->`` or ``base.``, whichever reaches a member of the sequence."""
        return f'{base}->' if self._is_boxed(ty) else f'{base}.'

    def _list_len(self, ty: CppType, base: str) -> str:
        """``len(xs)``."""
        return f'{self._member(ty, base)}size()'

    def _list_at(self, ty: CppType, base: str, idx: str) -> str:
        """``xs[i]``.  The cast belongs here because C++ ``operator[]`` takes an
        unsigned index while FPy indices are signed."""
        return f'{self._list_seq(ty, base)}[static_cast<size_t>({idx})]'

    def _list_at_raw(self, ty: CppType, base: str, idx: str) -> str:
        """``xs[i]`` where *idx* is already a ``size_t`` — an emitter-internal
        loop counter rather than an FPy index, so no cast is needed."""
        return f'{self._list_seq(ty, base)}[{idx}]'

    def _bind_operand(self, expr: str) -> str:
        """A name for *expr*, so it can be read more than once.

        Already a name — evaluated once, no side effects — so nothing to bind.
        Otherwise bind it to a temp; ``auto&&`` binds a reference, so this
        copies nothing whatever the representation.
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

    def _list_begin(self, ty: CppType, base: str) -> str:
        return f'{self._member(ty, base)}begin()'

    def _list_end(self, ty: CppType, base: str) -> str:
        return f'{self._member(ty, base)}end()'

    @staticmethod
    def _is_sized(ty: CppType) -> bool:
        """Whether *ty* is a fixed-length value list -- a ``std::array``."""
        return (
            isinstance(ty, CppList) and not ty.boxed and ty.size is not None
        )

    @classmethod
    def _all_sized(cls, ty: CppType) -> bool:
        """Whether every list level of *ty* is fixed-length."""
        while isinstance(ty, CppList):
            if not cls._is_sized(ty):
                return False
            ty = ty.elt
        return True

    def _list_push(self, ty: CppType, base: str, elt: str) -> str:
        """Append to a list under construction.  ``std::array`` has no growth
        operation; :meth:`_open_list_build` spells an index store instead."""
        if self._is_sized(ty):
            raise CppInternalError(
                f'push_back on a fixed-size list `{ty.format()}`'
            )
        return f'{self._member(ty, base)}push_back({elt})'

    def _list_new(self, ty: CppType, args: str) -> str:
        """A new list from a parenthesised constructor argument list.

        The one place the boxed and unboxed spellings of construction are
        stated; the named wrappers below only supply the arguments, which
        ``make_shared`` forwards to ``std::vector``'s constructor unchanged.
        ``std::array`` has none of these constructors, so each wrapper handles
        the fixed-size case before coming here.
        """
        if self._is_sized(ty):
            raise CppInternalError(
                f'no constructor-argument form for `{ty.format()}`'
            )
        if self._is_boxed(ty):
            return f'std::make_shared<std::vector<{self._elt_of(ty)}>>({args})'
        return f'{ty.format()}({args})'

    def _list_new_sized(self, ty: CppType, n: str) -> str:
        """*n* value-initialised elements.  A fixed-size list has its length in
        the type, so *n* -- the same ``K`` by construction -- says nothing."""
        if self._is_sized(ty):
            return f'{ty.format()}{{}}'
        return self._list_new(ty, n)

    def _list_empty(self, ty: CppType) -> str:
        """A new empty list.  Never emit a bare declaration for a *boxed* list:
        an uninitialised handle is a null ``shared_ptr``, unlike an empty
        ``std::vector``."""
        return self._list_new_sized(ty, '0')

    def _list_new_filled(self, ty: CppType, n: str, fill: str) -> str:
        """*n* copies of *fill*.  A fixed-size list has no ``(n, fill)``
        constructor, but ``K`` is a compile-time count, so the fill is simply
        repeated -- safe because FPy expressions are pure and ``_emit_empty``
        binds a dimension to a name before it can appear in one."""
        if self._is_sized(ty):
            assert isinstance(ty, CppList) and ty.size is not None
            return self._list_new_init(ty, [fill] * ty.size)
        return self._list_new(ty, f'{n}, {fill}')

    def _list_new_init(self, ty: CppType, parts: list[str]) -> str:
        """The given elements.  Not :meth:`_list_new`: a braced init-list is a
        non-deduced context, so ``make_shared`` cannot take one -- the boxed
        form spells the inner vector and moves it in.  A fixed-size list needs
        double braces; single braces rely on brace elision, which
        ``-Wmissing-braces`` flags."""
        joined = ', '.join(parts)
        if self._is_sized(ty):
            assert isinstance(ty, CppList)
            if len(parts) != ty.size:
                raise CppInternalError(
                    f'{len(parts)} elements for `{ty.format()}`'
                )
            if not parts:
                return f'{ty.format()}{{}}'
            return f'{ty.format()}{{{{{joined}}}}}'
        if self._is_boxed(ty):
            elt = self._elt_of(ty)
            return (
                f'std::make_shared<std::vector<{elt}>>'
                f'(std::vector<{elt}>{{{joined}}})'
            )
        return f'{ty.format()}{{{joined}}}'

    def _declare_empty_list(self, ty: CppType) -> str:
        """Declare a fresh empty list of *ty* and return its name.

        The one spelling of an empty declaration: a fixed-size list is
        value-initialised in place, everything else needs an initialiser.
        """
        out = self._fresh_temp()
        if self._is_sized(ty):
            self.writer.add_line(f'{ty.format()} {out}{{}};')
        else:
            self.writer.add_line(f'{ty.format()} {out} = {self._list_empty(ty)};')
        return out

    def _list_new_range(self, ty: CppType, first: str, last: str) -> str:
        """The half-open iterator range ``[first, last)``, copied.  A
        fixed-size list has no iterator-pair constructor, so it is
        value-initialised and filled with ``std::copy`` instead."""
        if self._is_sized(ty):
            out = self._declare_empty_list(ty)
            self.writer.add_line(f'std::copy({first}, {last}, {out}.begin());')
            return out
        return self._list_new(ty, f'{first}, {last}')

    def _open_fill_loop(self, ty: CppType, src_len: str) -> tuple[str, str]:
        """A fresh list of *ty* with a ``for`` opened over its indices:
        ``(name, index)``.  The caller writes the element stores and closes the
        loop.

        The count is *src_len* -- except for a fixed-size result, which counts
        by its own ``K``: the analysis proved the two agree on defined
        executions, and on one FPy leaves undefined (a mismatched ``zip`` that
        runs) ``K`` truncates where the runtime length would write out of
        bounds.
        """
        if self._is_sized(ty):
            assert isinstance(ty, CppList)
            n = str(ty.size)
        else:
            n = src_len
        out = self._fresh_temp()
        self.writer.add_line(
            f'{ty.format()} {out} = {self._list_new_sized(ty, n)};'
        )
        i = self._fresh_temp()
        self.writer.add_line(f'for (size_t {i} = 0; {i} < {n}; ++{i}) {{')
        self.writer.indent()
        return out, i

    def _open_list_build(self, ty: CppType) -> tuple[str, Callable[[str], str]]:
        """A named list under element-at-a-time construction: ``(name,
        append)``, where ``append(elt)`` is the statement text adding one
        element.

        A vector grows with ``push_back``; a fixed-size list is filled through
        a running index, its length being the ``K`` the analysis proved matches
        the number of appends.
        """
        out = self._declare_empty_list(ty)
        if self._is_sized(ty):
            idx = self._fresh_temp()
            self.writer.add_line(f'size_t {idx} = 0;')
            return out, lambda elt: f'{out}[{idx}++] = {elt}'
        return out, lambda elt: self._list_push(ty, out, elt)

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

        A name never rebound binds by reference, sharing the object and skipping the
        atomic refcount; ``const`` unless something in its region writes through it.
        That ``const`` is on the handle, not the elements, so a callee can still
        write ``xs[i] = e`` and the caller sees it.  A rebind must stay local, so it
        takes its own copy.
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

        Asked of the storage class, not of *d*: a use resolves to the latest def, and
        ``xs[i] = e`` makes a fresh one unioned with its ``prev``, whose own site says
        nothing about the binding.  One member carries the declaration.
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
        """``T name = rhs;`` or ``name = rhs;``, per :class:`StorageAnalysis`.
        """
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
        """Emit assigns extracting each element of *binding* from the tuple *src*.

        Underscore positions are skipped; nested bindings recurse via a fresh temp.
        """
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
        if self._strict and ret_ty is not None and contains_boxed(ret_ty):
            raise self._strict_tripwire()
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
            if self._strict and contains_boxed(storage):
                raise self._strict_tripwire()
            arg_strs.append(self._arg_decl(arg, storage))
        emitted_name = self._func_name_override or func.name
        sig = f'{ret_str} {emitted_name}({", ".join(arg_strs)})'

        self.writer.add_line(sig + ' {')
        self.writer.indent()
        # `_current_rm` is the mode the live fenv is guaranteed to hold on entry,
        # which the caller delivers -- see `_entry_rm` for what it is and when it
        # is unknown.  Either way no entry `fesetround` is emitted.
        self._current_rm = self._entry_rm(func)
        func_ctx = self._resolve_used_ctx(func)
        # REAL sets no fenv mode; its ops succeed only via `_try_widen`, whose
        # failure reports a precise location.  Validating here would fire first
        # and worse, so descend like an integer scope.
        entry_rm = None
        if func_ctx is not None and func_ctx is not REAL:
            storage = self._validate_context_rm(func_ctx, at=func)
            # only a floating-point context in float storage rounds through
            # `fesetround`; integer storage and the libm fixed-point lowering
            # both round by other means and want no scope
            if storage.is_float() and isinstance(func_ctx, EFloatContext):
                entry_rm = func_ctx.rm
        if entry_rm is None:
            self._visit_block(func.body, None)
        else:
            with self._fenv_scope(entry_rm):
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
            # a bare handle is a *null* ``shared_ptr``, not an empty list, so a
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
            case Empty() if isinstance(want, CppList):
                # Its own bound is the lattice bottom, so its own storage is
                # the ladder's first rung; building at the target saves
                # rebuilding a `vector<uint8_t>` element-wise into it.
                return self._emit_empty(e, want, ctx)
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
        """:meth:`_emit_at` where C++ takes the type *from* the argument.

        A braced initializer rejects a narrowing conversion outright, and
        ``std::make_tuple`` silently deduces a different type -- a ``float`` first
        argument gives ``std::tuple<float, double>``, which the declaration then
        rejects.  Either way the scalar's cast must be spelled.
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
        if self._is_sized(want) and src.size != want.size:
            # Nothing bridges a sizeless source or a different `K`; a tripwire,
            # since one region has one size.
            raise self._refuse_mismatch(src, want, at)
        # A boxed `want` carries no size, so this is the size drop too -- and it
        # has to precede the boxing: `make_shared<vector>(array)` does not
        # compile.
        unboxed = CppList(want.elt, boxed=False, size=want.size)
        if src != unboxed:
            code = self._rebuild_list(code, src, unboxed, at=at)
        if not self._is_boxed(want):
            return code
        # A value has no aliases to lose, so giving it a handle is free.
        return f'std::make_shared<{unboxed.format()}>({code})'

    def _refuse_mismatch(
        self, src: CppType, want: CppType, at: Expr,
    ) -> CppEmitError:
        """One place wants two types and nothing in C++ bridges them.

        Distinct ``std::vector`` instantiations are unrelated types, as are a
        vector and a handle, and differently-sized ``std::array``\\s.  A
        limitation is acceptable; emitting C++ that does not compile is not.
        """
        return CppEmitError(
            f'unsupported: this value is `{src.format()}` where '
            f'`{want.format()}` is needed, and C++ has no conversion between '
            f'them.  Keeping the two representations the same at this point '
            f'avoids it.',
            at=at,
        )

    def _require_no_narrowing(
        self, src: CppType | None, want: CppType | None, at: Expr,
    ) -> None:
        """Refuse a store C++ would narrow silently.

        The one case where a format disagreement is a *wrong answer* rather than a
        compile error: C++ accepts a narrowing store into a slot, and FPy says the
        list then holds the wider value.  Widening the container is not available --
        another name may alias it.  See ``docs/todos/backend-cpp.md``.
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
        sized = self._is_sized(want)
        if not isinstance(src.elt, CppList) and not sized:
            # Flat vector: the range constructor converts each element on the
            # way in.  A fixed-size target has no such constructor, so the loop
            # below stores by index instead.
            return self._list_new_range(
                want, self._list_begin(src, base), self._list_end(src, base),
            )
        i = self._fresh_temp()
        n = self._list_len(src, base)
        if sized:
            out = self._declare_empty_list(want)
        else:
            out = self._fresh_temp()
            self.writer.add_line(f'{want.format()} {out};')
            self.writer.add_line(f'{self._member(want, out)}reserve({n});')
        self.writer.add_line(f'for (size_t {i} = 0; {i} < {n}; ++{i}) {{')
        self.writer.indent()
        elt = self._convert_storage(
            self._list_at_raw(src, base, i), src.elt, want.elt, at=at,
        )
        self.writer.add_line(
            f'{self._list_at_raw(want, out, i)} = {elt};' if sized
            else f'{self._list_push(want, out, elt)};'
        )
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
            # a boxed `uint8_t` list in a vector of boxed `float` lists.
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
        """The rounding mode the caller contractually delivers at *site*.

        The scope's own RM for a concrete, ``fesetround``-supported FP context --
        the FPy annotation pins that on the caller.  A scope naming no FP mode
        (absent, ``REAL``, integer) falls back to ``RNE``: it is the same caller
        either way, so a ``REAL``-topped kernel has no reason to assume less than
        an ``FP64``-topped one, and a nested RNE ``with`` stays a no-op instead of
        a save/set/restore per execution.

        ``None`` is for the cases with no answer -- a context nothing resolved, an
        RM ``fesetround`` cannot express -- where a nested ``with`` must set the
        mode unconditionally.
        """
        scope = self._scope_by_site.get(site)
        if scope is None:
            return RM.RNE
        resolved = self._resolve_scope_ctx(scope)
        if resolved is None:
            return None
        if not isinstance(resolved, EFloatContext):
            return RM.RNE
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
        if not isinstance(storage, CppScalar) or not (
            storage.is_integer() or storage.is_float()
        ):
            raise CppEmitError(
                f'unsupported context storage `{storage!r}` for `{rctx}`',
                at=at,
            )
        if isinstance(rctx, MPFixedContext | MPBFixedContext):
            # A fixed-point context rounds by a libm call (float storage) or a
            # cast (integer storage).  Neither goes through ``fesetround``, so
            # its rounding mode is checked against what that lowering can do
            # rather than against the ``fenv`` modes.
            if storage.is_integer():
                if rctx.rm != RM.RTZ:
                    raise CppEmitError(
                        f'integer context `{rctx}` must use RTZ rounding mode '
                        '(C++ integer arithmetic rounds toward zero); got '
                        f'{rctx.rm}',
                        at=at,
                    )
                # C++ has no arbitrary-precision integer, so the int64_t
                # fallback may silently overflow; `nmin == -1` is how
                # MPFixedContext reports unboundedness.
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
            else:  # float storage, by the check above
                if rctx.nmin != -1:
                    raise CppEmitError(
                        f'fixed-point context `{rctx}` in floating-point '
                        'storage must have its digits at position zero; run '
                        '`fpy2.strategies.rescale_fixed` first',
                        at=at,
                    )
                if self._integral_fn(rctx.rm) is None:
                    raise CppEmitError(
                        f'rounding mode {rctx.rm} for context `{rctx}` has no '
                        'libm function rounding to an integral value',
                        at=at,
                    )
                # Everything the libm lowering refuses has to be refused *here*
                # too, or `_visit_round` falls back to a cast and the rounding
                # disappears.  These mirror `_emit_integral_round`'s guards.
                if rctx.num_randbits != 0:
                    raise CppEmitError(
                        f'stochastic rounding under `{rctx}` has no C++ '
                        'analogue: no libm function draws random bits',
                        at=at,
                    )
                if not rctx.enable_neg_zero:
                    raise CppEmitError(
                        f'context `{rctx}` drops the negative zero that its '
                        'floating-point storage keeps, and the libm rounding '
                        'functions preserve',
                        at=at,
                    )
                if rctx.nan_value is not None or rctx.inf_value is not None:
                    raise CppEmitError(
                        f'context `{rctx}` substitutes a value for NaN or an '
                        'infinity; a libm rounding passes both through '
                        'unchanged, and an assertion cannot stand in for a '
                        'substitution',
                        at=at,
                    )
        elif storage.is_float():
            if not isinstance(rctx, EFloatContext):
                raise CppEmitError(
                    f'context `{rctx}` resolves to floating-point storage but '
                    'is not a floating-point context, so it states no '
                    '``fesetround`` mode',
                    at=at,
                )
            if rctx.rm not in _FE_RM_MACRO:
                raise CppEmitError(
                    f'rounding mode {rctx.rm} for context `{rctx}` is not '
                    'supported by ``fesetround`` (need RNE, RTZ, RTP, or RTN)',
                    at=at,
                )
        else:
            raise CppEmitError(
                f'context `{rctx}` resolves to integer storage but is not a '
                'fixed-point context; only `MPFixedContext` and '
                '`MPBFixedContext` lower to integer arithmetic',
                at=at,
            )
        return storage

    @contextmanager
    def _fenv_scope(self, target_rm: RM):
        """Wrap the contained emission in a ``fesetround`` save/set/restore, unless the
        active mode is already *target_rm*.

        ``_current_rm`` is ``None`` when the live mode is unknown, and then the set is
        unconditional rather than a guess about what the runtime holds.
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
        # The context comes from the ContextUse scope at this site, already
        # resolved.  Validation fires only if something inside uses it.
        if not isinstance(stmt.target, UnderscoreId):
            raise CppEmitError(
                'binding the active context to a name is not yet supported',
                at=stmt,
            )
        rctx = self._resolve_used_ctx(stmt)
        # ``REAL`` doesn't correspond to any C++ rounding mode — see
        # the same comment in :meth:`_visit_function`.  Treat it as
        # a pass-through; per-op widening dispatch handles the body.
        if rctx is None or rctx is REAL:
            # No op uses this scope, or the scope is REAL (no fenv mode).
            self._visit_block(stmt.body, ctx)
            return
        storage = self._validate_context_rm(rctx, at=stmt)
        if storage.is_integer() or isinstance(rctx, MPFixedContext | MPBFixedContext):
            # A fixed-point scope sets no ``fenv`` mode whichever storage it
            # lands in: it rounds by a cast or by a libm call naming its mode.
            # The one exception is RNE, whose ``std::nearbyint`` reads the live
            # mode instead -- `_emit_integral_round` checks that separately.
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

        An FPy literal is an exact rational rounded where it is *used*, which C++
        has no spelling for -- so a value binary64 holds exactly prints as itself,
        and one it cannot is refused rather than emitted as ``num / denom``, which
        would be an operation where FPy has a constant.  Digits only while an
        integer literal holds the value (:func:`_value_cpp_type`).
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
        *, at: Ast | None = None, src: Expr | None = None,
    ) -> str:
        """Emit *arg* in *target_ty* form, rejecting unsafe casts.

        For conversions implicit from the user's perspective, where format inference
        and storage selection decided to rebind the operand.  A lossy one is refused
        rather than silently emitted: the user should narrow the active context or
        write ``fp.round(...)``.  Casts the user *did* write go through
        :meth:`_explicit_cast`, which never refuses.

        Pass *src* to fall back on :func:`bound_fits_in_scalar` when the
        type-level test refuses.
        """
        if arg_ty == target_ty:
            return arg
        if not scalar_fits_in(arg_ty, target_ty) and not (
            src is not None
            and bound_fits_in_scalar(
                self.format_info.by_expr.get(src), target_ty,
            )
        ):
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
        matches the ``float`` signature while the token is a ``double``.  Harmless
        where a declaration supplies the type, and exact for ``+ - * /`` since
        ``2p + 2 <= 53``.  Not where the callee deduces from the argument:
        ``fpy::min``/``max`` fail to deduce, and the ``<cmath>`` overload sets pick
        the wider one -- which for ``fma``, outside that bound, rounds twice.
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

        A **direct match** needs no conversion.  Failing that, **cast-to-active**
        takes an all-same-type signature of the active context whose width holds
        every operand losslessly -- skipped when that context has no C++ storage.
        Failing that, **widening**, sound only under ``REAL``, where the wider op
        gives the exact result and rounds to itself; see :meth:`_try_widen`.

        A literal operand of a call-form signature needs its type spelled even on a
        direct match: it matched on storage, not on the token's type.
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

        Folded to a single ``std::get`` whenever it reads one element, so
        ``fst(snd(e))`` over a 3-tuple is ``std::get<1>(e)``.  Only a chain genuinely
        yielding a shorter tuple materializes a ``std::make_tuple``.
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
        """``enumerate(xs)`` builds a list of ``std::tuple<I, T>``
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
        result, i = self._open_fill_loop(
            result_ty, self._list_len(src_ty, src),
        )
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
            case Mul():
                scaled = self._emit_scale_by_pow2(e, ctx)
                if scaled is not None:
                    return scaled
                lhs = self._visit_expr(e.first, ctx)
                rhs = self._visit_expr(e.second, ctx)
                return self._dispatch_binary(e, lhs, rhs)
            case Pow():
                scaled = self._emit_pow2(e, ctx)
                if scaled is not None:
                    return scaled
                lhs = self._visit_expr(e.first, ctx)
                rhs = self._visit_expr(e.second, ctx)
                return self._dispatch_binary(e, lhs, rhs)
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

    _LDEXP_EXP_LIMIT = 1 << 30
    """How large a scale exponent may be before ``std::ldexp``'s ``int``
    parameter is in doubt.  Far above any real format's range, so this only
    rules out a bound the analysis could not narrow."""

    def _ldexp_exponent(self, e: Expr) -> bool | None:
        """Whether *e* can be ``std::ldexp``'s exponent, and what it costs.

        - ``None`` -- no: not known to be an integer within ``int``'s range, so
          the caller must emit a product instead.
        - ``True`` -- yes, unconditionally.
        - ``False`` -- yes, but only under a runtime finiteness guard: the
          format admits a NaN or an infinity, neither of which has an ``int``,
          and the branches that rule them out are not something the analysis
          reads.
        """
        fmt = self.format_info.by_expr.get(e)
        if isinstance(fmt, SetFormat):
            ok = all(
                isinstance(v, Fraction) and v.denominator == 1
                and abs(v) < self._LDEXP_EXP_LIMIT
                for v in fmt.values
            )
            return True if ok else None
        if not isinstance(fmt, AbstractableFormat):
            return None
        af = AbstractFormat.from_format(fmt)
        # a grid whose finest digit sits at or above position zero holds
        # integers only
        if not isinstance(af.exp, int) or af.exp < 0:
            return None
        bounds = (af.pos_bound, af.neg_bound)
        if any(not isinstance(b, RealFloat) for b in bounds):
            return None
        if any(abs(int(b)) >= self._LDEXP_EXP_LIMIT for b in bounds):
            return None
        return not (af.has_nan or af.has_pos_inf or af.has_neg_inf)

    def _emit_scale_by_pow2(self, e: Mul, ctx) -> str | None:
        """``2 ** n * v`` as ``std::ldexp(v, n)``, or `None` to emit a product.

        ``ldexp`` is IEEE 754's ``scaleB``: multiplication by an integral power
        of two, exact but for overflow and underflow.  The product it replaces
        rounds twice and rests on ``std::pow`` returning ``2 ** n`` exactly,
        which C11 F.10 does not require of any math function and IEEE 754 only
        *recommends* for ``exp2`` -- so a conforming libm within one ulp would
        put the scale on the wrong grid.  This is the shape `rescale_fixed`
        emits for every rounding it moves.

        Because ``ldexp`` computes the *exact* product, it stands in for
        ``round_C`` only where that rounding is the identity -- otherwise it
        would skip a rounding the context asked for, which is what the op-table
        dispatch refuses by matching storage against the context.  The operand
        must also reach the result's storage losslessly, for the same reason.
        """
        active = self._active_ctx_for(e)
        if active is None or not self._result_fits_ctx(e, active):
            # the context rounds this product; `ldexp` would not
            return None
        for scale, value in ((e.first, e.second), (e.second, e.first)):
            if not isinstance(scale, Pow):
                continue
            base, exp = scale.first, scale.second
            if not (isinstance(base, RationalVal) and base.as_rational() == 2):
                continue
            finite = self._ldexp_exponent(exp)
            if finite is None:
                continue
            # One call replaces *two* rounded steps, so the intermediate has to
            # be unrounded as well: under `FP64`, `2 ** -1080` is already zero
            # before it reaches the product, and `ldexp` would never form it.
            if not self._pow2_is_exact(exp, active):
                continue
            target = self._storage_for_expr(e)
            if not (isinstance(target, CppScalar) and target.is_float()):
                continue
            # the value has to reach `target` without being rounded on the way
            value_ty = self._storage_for_expr(value)
            if not (
                isinstance(value_ty, CppScalar)
                and (value_ty == target or scalar_fits_in(value_ty, target))
            ):
                continue
            v = self._visit_expr(value, ctx)
            if value_ty != target:
                # `std::ldexp` is overloaded on its first argument, so a narrower
                # operand would scale -- and overflow -- in the narrower type
                v = self._explicit_cast(v, target)
            return self._ldexp_call(v, exp, ctx, finite=finite)
        return None

    def _pow2_is_exact(self, exp: Expr, ctx: Context) -> bool:
        """Whether `ctx` rounds `2 ** exp` at all.

        Asks :func:`exact_exp2` rather than reading the recorded bound: what
        inference records is the result *after* the scope clipped it, so a power
        too wide for the context comes back looking like the context's own
        format.  ``ldexp`` forms the exact value, so a clipped one is exactly
        the case that must decline.
        """
        return round_is_identity(
            exact_exp2(self.format_info.by_expr.get(exp)), ctx,
        )

    def _emit_pow2(self, e: Pow, ctx) -> str | None:
        """``2 ** n`` as ``std::ldexp(1, n)``, or `None` to emit a ``pow``.

        The same accuracy argument as :meth:`_emit_scale_by_pow2`, for a power
        that is not an operand of a multiply.  Where it *is*, that method is
        preferred: ``ldexp(v, n)`` scales in one step, while
        ``ldexp(1, n) * v`` can underflow the intermediate to zero and lose a
        product that was representable.
        """
        if not (isinstance(e.first, RationalVal) and e.first.as_rational() == 2):
            return None
        active = self._active_ctx_for(e)
        if active is None or not self._pow2_is_exact(e.second, active):
            return None
        finite = self._ldexp_exponent(e.second)
        if finite is None:
            return None
        target = self._storage_for_expr(e)
        if not (isinstance(target, CppScalar) and target.is_float()):
            return None
        return self._ldexp_call('1', e.second, ctx, finite=finite)

    def _ldexp_call(
        self, value: str, exp: Expr, ctx, *, finite: bool,
    ) -> str:
        """``std::ldexp(value, exp)``, falling back to a product where the
        exponent may not be finite.

        ``ldexp`` takes its exponent as an ``int``, and converting a NaN or an
        infinity to one is undefined -- on x86-64 it yields ``INT_MIN``, so
        ``2 ** inf`` would come back ``0`` instead of an infinity.  FPy defines
        all three (``inf``, ``0``, NaN), and so does ``std::pow``, so the
        product is the faithful lowering exactly where the exponent is not
        finite.  An assertion would not do: it compiles out under ``NDEBUG``,
        leaving the undefined conversion in a release build.

        The analysis proves finiteness whenever the exponent's format admits no
        specials -- a bounded integer one, say -- and then this costs nothing.
        """
        n = self._bind_operand(self._visit_expr(exp, ctx))
        if finite:
            return f'std::ldexp({value}, static_cast<int>({n}))'
        # both arms name the value, so it has to be bound before either
        v = self._bind_operand(value)
        scaled = f'std::ldexp({v}, static_cast<int>({n}))'
        return f'(std::isfinite({n}) ? {scaled} : std::pow(2.0, {n}) * {v})'

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

    def _range_bound(self, e: Expr, elt_ty: CppScalar, ctx) -> str:
        """A ``range`` bound, cast into the element type."""
        return self._maybe_cast(
            self._visit_expr(e, ctx), self._scalar_storage_for_expr(e), elt_ty,
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
        match e:
            case Range1():
                tmp = self._fresh_temp()
                stop_cast = self._range_bound(e.arg, result_ty.elt, ctx)
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
                tmp = self._fresh_temp()
                start_cast = self._range_bound(e.first, result_ty.elt, ctx)
                stop_cast = self._range_bound(e.second, result_ty.elt, ctx)
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
                start_cast = self._range_bound(e.args[0], result_ty.elt, ctx)
                stop_cast = self._range_bound(e.args[1], result_ty.elt, ctx)
                step_cast = self._range_bound(e.args[2], result_ty.elt, ctx)
                ctr = self._fresh_temp()
                out, append = self._open_list_build(result_ty)
                self.writer.add_line(
                    f'for ({int_ty} {ctr} = {start_cast}; '
                    f'{ctr} < {stop_cast}; {ctr} += {step_cast}) {{'
                )
                self.writer.indent()
                self.writer.add_line(f'{append(ctr)};')
                self.writer.dedent()
                self.writer.add_line('}')
                return out
            case _:
                raise CppEmitError(
                    f'unsupported range op: {type(e).__name__}', at=e,
                )

    def _emit_size(self, e: Size, ctx) -> str:
        """``size(xs, d)`` -- follow *d* ``[0]`` indices into the shape format
        inference knows, then ``.size()``.

        Requires a constant *d*; a runtime one would need a dispatch the corpus does
        not justify.
        """
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

    def _unsupported(self, kind: str, at: Ast | None = None) -> NoReturn:
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
        arg_tys = []
        for a in e.args:
            ty = self._storage_for_expr(a)
            if not isinstance(ty, CppScalar):
                # `==` is `a -> a -> bool`, so an aggregate arrives well-typed
                raise CppEmitError(
                    f'cannot compare `{ty.format()}`: the cpp backend compares '
                    f'scalars only',
                    at=e,
                )
            arg_tys.append(ty)
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

    def _visit_nullaryop(self, e: NullaryOp, ctx) -> str:
        ty = self._scalar_storage_for_expr(e)
        if isinstance(e, ConstInf | ConstNan):
            # No literal spells these, and no integer type holds them.
            if ty.is_integer():
                raise CppEmitError(
                    f'{type(e).__name__} is not representable in integer '
                    f'storage `{ty.format()}`',
                    at=e,
                )
            member = 'infinity' if isinstance(e, ConstInf) else 'quiet_NaN'
            return f'std::numeric_limits<{ty.format()}>::{member}()'

        fn = _NULLARY_CONSTS.get(type(e))
        if fn is None:
            self._unsupported(type(e).__name__, at=e)

        active = self._active_ctx_for(e)
        try:
            val = fn(ctx=active)
        except NotImplementedError as err:
            # e.g. an irrational constant under `REAL`, which rounds to nothing
            raise CppEmitError(
                f'cannot evaluate {type(e).__name__} under context `{active}`',
                at=e,
            ) from err
        return self._emit_numeric_literal(val.as_rational(), at=e)

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
                return self._emit_empty(e, self._storage_for_expr(e), ctx)
            case _:
                raise CppEmitError(
                    f'unsupported nary op: {type(e).__name__}', at=e,
                )

    def _emit_bool_chain(self, e: 'And | Or', ctx) -> str:
        """Reduce ``And``/``Or`` to a parenthesised ``&&``/``||`` chain.

        C++'s short-circuit matches FPy's for pure expressions, and the operands are
        already ``BOOL``.  A zero-arg form is rejected by the front end, but
        degenerates cleanly here.
        """
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

        Combiner as in :meth:`_emit_min_max`.  Both demand uniform operands, so each
        element is cast to ``result_ty``.  The empty list is undefined, matching the
        interpreter's ``ValueError``: the emit indexes ``xs[0]`` unguarded.
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

    def _emit_empty(self, e: Empty, result_ty: CppType, ctx) -> str:
        """``empty(d1, ..., dN)``: an N-dimensional zero-initialised list of
        storage *result_ty*.

        Sizes are read off the call site and emitted as nested constructor calls
        right-to-left, so the innermost element type bubbles out.  ``empty()`` is a
        scalar ``T()``, which format inference resolves at the call site.

        Each non-constant dimension is bound to a name first, so that a
        fixed-size layer repeating its fill ``K`` times does not evaluate a
        dimension more than once.
        """
        dims = []
        for a in e.args:
            d = self._visit_expr(a, ctx)
            dims.append(d if d.isdigit() else self._bind_operand(d))
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
        if self._all_sized(result_ty):
            # Every dimension is in the type, and value-initialising a nested
            # `std::array` zeroes it recursively -- no per-layer fill needed.
            return self._list_empty(result_ty)
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
        list of ``std::tuple<T1, …, TN>`` whose length matches the
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
        result, i = self._open_fill_loop(
            result_ty, self._list_len(head_ty, head),
        )
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

    def _require_cast_is_round(self, e) -> None:
        """Refuse a `Round`/`Cast` whose context a ``static_cast`` cannot perform.

        Storage is chosen to *contain* a format, not to equal it, so a cast into
        it rounds to the storage's format rather than the context's: `FP16` gets
        ``float``, and ``static_cast<float>(1024.5)`` is ``1024.5`` where
        `FP16.round` says ``1024``.  Arithmetic never had this problem because the
        op table matches on whole contexts (:meth:`CppOp.matches`); these two
        bypass it, so the same discipline is applied here.

        Fixed-point contexts are exempt: `_validate_context_rm` has already
        checked that either the libm lowering or an integer cast reproduces the
        rounding.
        """
        active = self._active_ctx_for(e)
        if isinstance(active, MPFixedContext | MPBFixedContext):
            return
        # resolved first: a context with no storage at all -- ``REAL``, or a
        # format wider than the ladder -- has a more specific complaint than this
        storage = self._scalar_for_ctx(active, at=e)
        if not is_native_ctx(active):
            raise CppEmitError(
                f'rounding under `{active}` has no C++ analogue: its storage '
                f'`{storage.format()}` rounds to that type\'s own format, not '
                'to this one.  Lower it first with `monomorphize -> '
                'unfold_overflow -> float_to_fixed -> rescale_fixed`.',
                at=e,
            )

    def _scalar_cast_types(self, e):
        """Source/target scalar storage for a round-like node *e*.

        The argument's storage only short-circuits same-type casts, so a non-dyadic
        literal with no representable storage is fine.  ``Round`` folds those earlier;
        ``Cast`` refuses them, which is right for a cast asserting exactness.
        """
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
        #
        # That assertion tests exactness in the *storage*, which is the context's
        # own question only where the two formats agree -- so the same guard as
        # `Round` has to pass before the equality below means anything.
        self._require_cast_is_round(e)
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
        # A `static_cast`, whose rounding mode is the one the surrounding
        # `with` set.  The user asked for it, so it is emitted even when lossy;
        # a literal argument is folded instead.
        folded = self._fold_rounded_literal(e)
        if folded is not None:
            return folded
        arg = self._visit_expr(e.arg, ctx)
        integral = self._emit_integral_round(e, arg)
        if integral is not None:
            return integral
        # only now that the exact paths above have declined does the rounding
        # fall to a cast, which not every context's `round` agrees with
        self._require_cast_is_round(e)
        arg_ty, target_ty = self._scalar_cast_types(e)
        if arg_ty == target_ty:
            return arg
        return self._explicit_cast(arg, target_ty)

    def _integral_fn(self, rm: RM) -> str | None:
        """The libm function rounding to an integral value under *rm*.

        Each is exact and stays in the floating-point type, so unlike a cast to
        an integer it keeps a signed zero and needs no integer wide enough for
        the value.  ``RAZ`` has no single spelling and ``RTO``/``RTE`` none at
        all.
        """
        return {
            RM.RTZ: 'std::trunc',
            RM.RTN: 'std::floor',
            RM.RTP: 'std::ceil',
            RM.RNA: 'std::round',
            # follows the *current* mode, so it is RNE only under FE_TONEAREST
            RM.RNE: 'std::nearbyint',
        }.get(rm)

    def _undefined_guard(
        self, ctx: MPFixedContext | MPBFixedContext, operand: str,
    ) -> str | None:
        """A test that *operand* is a value `ctx` can round, or `None`.

        A context states which values it has no result for; each such statement
        compiles to an assertion.  A stated *substitute* (``nan_value`` /
        ``inf_value``) is a value rather than a refusal, so it needs an emitted
        branch instead -- callers decline those.
        """
        tests = []
        if not ctx.enable_nan and ctx.nan_value is None:
            tests.append(f'!std::isnan({operand})')
        if not ctx.enable_inf and ctx.inf_value is None:
            tests.append(f'!std::isinf({operand})')
        if len(tests) == 2:
            # both refused: one call says it
            return f'std::isfinite({operand})'
        return tests[0] if tests else None

    def _emit_integral_round(self, e, arg: str) -> str | None:
        """``round(v)`` under an integer context, as a libm call.

        `None` when this rounding is not of that shape, leaving the caller's
        cast in place.  Applies when the context's digits sit at position zero
        and its storage is a floating-point type: the result is then an integral
        *value* in a float, which is what C++ rounds to natively.

        The context's edges become assertions bracketing the call -- an operand
        it cannot round, and a result past its bound.  Both vanish under
        ``NDEBUG``, as every FPy assertion does.
        """
        active = self._active_ctx_for(e)
        if not isinstance(active, MPFixedContext | MPBFixedContext):
            return None
        # position zero: `nmin` is the last unrepresentable digit, so -1 is the
        # integer grid.  Any other position needs the operand scaled first,
        # which is `rescale_fixed`'s job rather than the backend's.
        if active.nmin != -1 or active.num_randbits != 0:
            return None
        # libm keeps a signed zero; a format without one would disagree
        if not active.enable_neg_zero:
            return None
        fn = self._integral_fn(active.rm)
        if fn is None:
            return None
        target_ty = self._scalar_for_ctx(active, at=e)
        if not target_ty.is_float():
            # integer storage: the existing cast is the better lowering
            return None
        if active.rm is RM.RNE and self._current_rm not in (None, RM.RNE):
            raise CppEmitError(
                f'rounding to nearest under `{active}` needs `std::nearbyint` '
                f'in FE_TONEAREST, but the enclosing scope set {self._current_rm}',
                at=e,
            )

        operand = self._bind_operand(arg)
        guard = self._undefined_guard(active, operand)
        if guard is not None:
            self.writer.add_line(
                f'assert({guard} && "fpy: rounding is undefined for this value");'
            )
        out = self._fresh_temp()
        self.writer.add_line(
            f'{target_ty.format()} {out} = {fn}({operand});'
        )
        if isinstance(active, MPBFixedContext) and active.overflow is OverflowMode.ASSERT:
            hi = active.maxval().as_real().as_rational()
            lo = active.maxval(s=True).as_real().as_rational()
            if lo == -hi:
                test = f'std::fabs({out}) <= {self._emit_numeric_literal(hi, at=e)}'
            else:
                # the two bounds are independent -- a two's-complement grid runs
                # to -128 but only to 127 -- and `fabs` cannot say that
                test = (
                    f'{self._emit_numeric_literal(lo, at=e)} <= {out} && '
                    f'{out} <= {self._emit_numeric_literal(hi, at=e)}'
                )
            self.writer.add_line(
                f'assert(({test}) '
                f'&& "fpy: overflow occurred so rounding is undefined");'
            )
        return out

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
            param = params[i] if params and i < len(params) else None
            parts.append(self._adapt_arg(self._visit_expr(a, ctx), a, param))
        target = self._call_names.get(e, str(e.func))
        call = f'{target}({", ".join(parts)})'
        return self._adapt_result(call, e, abi.ret if abi is not None else None)

    def _adapt_result(self, emitted: str, e: Expr, got: CppType | None) -> str:
        """*emitted* as the caller needs the result.

        A callee's return representation is fixed by its own body, so a caller
        keeping a handle for a local reason has to make one.  The result is a
        prvalue the callee handed over -- which is why it unboxed -- so
        re-boxing a vector moves rather than copies; a *sized* result headed for
        a vector or a handle is rebuilt, which copies.  Only this direction
        arises; ``unbox`` gives the caller a handle whenever the callee returns
        one.
        """
        if got is None or self.unbox is None:
            return emitted
        want = self._storage_for_expr(e)
        if not (isinstance(got, CppList) and isinstance(want, CppList)):
            return emitted
        if got == want:
            return emitted
        if got.boxed or got.elt != want.elt:
            raise CppEmitError(
                f'cannot hand back `{got.format()}` where `{want.format()}` '
                f'is wanted',
                at=e,
            )
        # What remains is a value adjustment -- a size drop, a boxing, or both.
        return self._convert_storage(emitted, got, want, at=e)

    def _adapt_arg(self, emitted: str, e: Expr, param: ParamAbi | None) -> str:
        """*emitted* as the callee spelled its parameter.

        A callee's signature is fixed by its own body, so a caller holding a handle
        the callee does not want hands over the pointee -- same elements, no copy,
        and a write still reaches the caller.  Only this direction arises: ``unbox``
        gives an argument a handle whenever the callee declared one.  Just as well,
        since the reverse needs an aliasing ``shared_ptr`` (the test-side
        ``fpy::borrow`` interop helper), which cannot bind a ``const``
        reference.
        """
        if param is None or self.unbox is None:
            return emitted
        want = param.ty
        have = self._storage_or_none(e)
        if not (isinstance(have, CppList) and isinstance(want, CppList)):
            self._require_bridgeable(have, want, e)
            return emitted
        if have.elt != want.elt:
            # The callee declared a different element type; nothing at the call
            # site can bridge two `std::vector` instantiations.
            raise self._refuse_mismatch(have, want, e)
        if have.size != want.size:
            if have.boxed and self._is_sized(want) and not param.written:
                # The spec was keyed at the argument *value*'s proven length,
                # which the caller's representation joined away (its region is
                # shared, so the value lives behind a handle).  For a read-only
                # parameter a copy is observationally equal, and its length is
                # `K` by the key's construction.
                base = self._bind_operand(emitted)
                return self._list_new_range(
                    want, self._list_begin(have, base),
                    self._list_end(have, base),
                )
            # Otherwise nothing bridges: a *written* sized parameter cannot take
            # a copy, since the write has to reach the caller.
            raise self._refuse_mismatch(have, want, e)
        if have.boxed == want.boxed:
            return emitted
        if not have.boxed:
            raise CppEmitError(
                f'passing an unboxed list where `{want.format()}` is declared',
                at=e,
            )
        self._is_boxed(have)  # strict tripwire: a handle is handed over
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
        # A temp plus nested loops that append the element expression.
        # *result_ty* is supplied because a comprehension builds its list
        # element by element, making it a contributor to whatever place it
        # reaches -- see :meth:`_emit_at`.
        tmp, append = self._open_list_build(result_ty)

        for target, iterable in zip(e.targets, e.iterables):
            self._open_comp_loop(target, iterable, e, ctx)

        elt_want = result_ty.elt if isinstance(result_ty, CppList) else None
        elt = self._emit_at(e.elt, elt_want, ctx)
        self.writer.add_line(f'{append(elt)};')

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
        """Emit one ``for`` line for a comprehension stage, leaving the writer
        indented inside the body.

        Always declare-on-assign: the target lives only in the loop scope.  Its
        SSA site is the ``ListComp`` node, not the target id.
        """
        # `loop_def` drives the element binding; None means discarded.
        loop_def = None
        match target:
            case NamedId():
                target_def = self.def_use.find_def_from_site(target, comp_site)
                target_name = self.storage.def_to_name[target_def]
                # A range counter is sized to the exit-test overshoot; the
                # element storage would be too narrow.
                counter = self._range_counter_scalar(iterable)
                storage = counter or self.storage.storage_of(target_def)
                decl = f'{storage.format()} {target_name}'
                loop_def = target_def
            case UnderscoreId():
                target_name = self._fresh_temp()
                decl = f'{self._underscore_counter_ty(iterable)} {target_name}'
            case TupleBinding():
                # A tuple binding cannot pair with a range.
                if isinstance(iterable, (Range1, Range2, Range3)):
                    raise CppEmitError(
                        'tuple-binding comprehension target requires a '
                        'non-range iterable',
                        at=comp_site,
                    )
                tmp = self._fresh_temp()
                iter_str = self._visit_expr(iterable, ctx)
                iter_ty = self._storage_for_expr(iterable)
                # read-only (only destructured) -> bind by const&
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

        header = self._for_header(iterable, target_name, decl, loop_def, ctx)
        self.writer.add_line(f'{header} {{')
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
        ift = self._maybe_cast(ift, ift_ty, out_ty, at=e, src=e.ift)
        iff = self._maybe_cast(iff, iff_ty, out_ty, at=e, src=e.iff)
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

    #: Expression kinds that emit no statement of their own, so a condition
    #: built only from these can move onto an ``else if`` line.  Deliberately a
    #: whitelist: anything absent keeps the nesting, which is always correct.
    _PURE_COND_OPS = (
        Var, Integer, Decnum, Rational, Hexnum, BoolVal,
        Compare, Not, And, Or,
        IsNan, IsInf, IsFinite, IsNormal, Signbit,
    )

    def _is_pure_cond(self, e: Expr) -> bool:
        """Whether *e* compiles to an expression with no statements before it.

        An operand needing a temporary, and the assertions the rounding
        lowerings emit, are both statements -- and on an ``else if`` line they
        would sit before the ``else`` and run unconditionally.  Decided
        structurally rather than by emitting and undoing, since a rewind cannot
        take back the emitter state an expression visitor may also have touched.
        """
        if not isinstance(e, self._PURE_COND_OPS):
            return False
        subs: list[Expr] = [
            v for name in ('arg', 'first', 'second')
            if isinstance(v := getattr(e, name, None), Expr)
        ]
        subs += [v for v in getattr(e, 'args', ()) if isinstance(v, Expr)]
        return all(self._is_pure_cond(s) for s in subs)

    def _flattenable(self, block: StmtBlock) -> 'IfStmt | None':
        """The ``if`` this ``else`` block can collapse onto, or ``None``.

        An ``else`` holding one ``if`` and nothing else is what ``else if`` is
        for; a chain of FPy ``elif``s arrives as exactly that, one nesting level
        per arm.  Two things keep the nesting: a storage class anchored to the
        inner ``if``, which declares *before* it and has nowhere to go on an
        ``else if`` line, and a condition that needs statements of its own.
        """
        if len(block.stmts) != 1:
            return None
        only = block.stmts[0]
        if not isinstance(only, IfStmt):
            return None
        if self.storage.hoists_before.get(only):
            return None
        return only if self._is_pure_cond(only.cond) else None

    def _visit_if(self, stmt: IfStmt, ctx):
        cond = self._visit_expr(stmt.cond, ctx)
        self.writer.add_line(f'if ({cond}) {{')
        self.writer.indent()
        self._visit_block(stmt.ift, ctx)
        self.writer.dedent()

        # Flatten ``else { if ... }`` into ``else if``, which is what the
        # nesting means and keeps a long chain readable.
        while (nested := self._flattenable(stmt.iff)) is not None:
            nested_cond = self._visit_expr(nested.cond, ctx)
            self.writer.add_line(f'}} else if ({nested_cond}) {{')
            self.writer.indent()
            self._visit_block(nested.ift, ctx)
            self.writer.dedent()
            stmt = nested

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
        """Counter type for a ``range`` loop, or ``None`` when the bounds are not
        statically concrete.

        A C-style counter transiently reaches the first value past ``stop``, which
        the loop variable's *element* format excludes -- so typing it from the
        element storage overflows at a boundary (``range(128)`` in an ``int8_t``).
        Sized to cover ``start`` and that overshoot instead.
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

    def _for_header(self, iterable: Expr, target: str, decl: str,
                    target_def, ctx) -> str:
        """The ``for (...)`` header for *iterable*, without the brace.

        A ``Range*`` is a counter loop using *decl* as given; anything else is a
        range-for, where :meth:`_foreach_decl` decides the element binding and a
        ``None`` *target_def* marks a discarded element.

        Shared with :meth:`_open_comp_loop`, which needs the header alone -- it
        leaves the writer open for its caller to fill the body.
        """
        match iterable:
            case Range1():
                stop = self._visit_expr(iterable.arg, ctx)
                return f'for ({decl} = 0; {target} < {stop}; ++{target})'
            case Range2():
                start = self._visit_expr(iterable.first, ctx)
                stop = self._visit_expr(iterable.second, ctx)
                return (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; ++{target})'
                )
            case Range3():
                start = self._visit_expr(iterable.args[0], ctx)
                stop = self._visit_expr(iterable.args[1], ctx)
                step = self._visit_expr(iterable.args[2], ctx)
                return (
                    f'for ({decl} = {start}; '
                    f'{target} < {stop}; {target} += {step})'
                )
            case _:
                iter_str = self._visit_expr(iterable, ctx)
                iter_ty = self._storage_for_expr(iterable)
                elt_decl = self._foreach_decl(
                    target_def, target,
                    elt=iter_ty.elt if isinstance(iter_ty, CppList) else None,
                    at=iterable,
                )
                return (
                    f'for ({elt_decl} : '
                    f'{self._list_range(iter_ty, iter_str)})'
                )

    def _underscore_counter_ty(self, iterable: Expr) -> str:
        """The C++ type for a discarded loop variable.

        A range counter is sized from its trajectory (:meth:`_range_counter_scalar`),
        falling back to the stop bound's storage; a value iterable uses ``auto``
        and lets the range-for deduce.
        """
        counter = self._range_counter_scalar(iterable)
        if counter is not None:
            return counter.format()
        match iterable:
            case Range1():
                return self._scalar_storage_for_expr(iterable.arg).format()
            case Range2() | Range3():
                return self._scalar_storage_for_expr(iterable.args[1]).format()
            case _:
                return 'auto'

    def _emit_for_loop(self, stmt: ForStmt, ctx, target: str, decl: str,
                       target_def) -> None:
        """``for (<header>) { body }``."""
        header = self._for_header(stmt.iterable, target, decl, target_def, ctx)
        self.writer.add_line(f'{header} {{')
        self.writer.indent()
        self._visit_block(stmt.body, ctx)
        self.writer.dedent()
        self.writer.add_line('}')


    def _emit_for_underscore_target(self, stmt: ForStmt, ctx):
        """``for _ in iter:`` -- the body never reads the counter, so emit a
        synthetic name."""
        target = self._fresh_temp()
        ty = self._underscore_counter_ty(stmt.iterable)
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
