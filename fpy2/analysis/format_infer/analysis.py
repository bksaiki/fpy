"""
Format analysis for FPy programs.

This analysis determines the number format (bounding the set of representable
values) for each expression and variable definition in an FPy program.

Analysis pipeline
-----------------
1. **Type checking** (:class:`TypeInfer`) — establishes basic types for every
   expression and variable (``bool``, ``real``, ``list``, ``tuple``, …).
2. **Context-use analysis** (:class:`ContextUse`) — maps every context-sensitive
   operation (arithmetic, rounding, etc.) to the rounding context scope that is
   active when it is evaluated.
3. **Format inference** — propagates number format information through the
   program.  For each operation the format is derived from its active rounding
   context; for variables it is propagated from the expression they are bound
   to; at control-flow merge points (``if``/``while``/``for`` phi nodes) formats
   are joined.

Format lattice
--------------
The analysis tracks a **format** that mirrors the basic-type structure::

    FormatBound ::= None                                  # non-numeric values
                  | SetFormat(values: frozenset)          # known finite real values
                  | Format                                # scalar real
                  | TupleFormat(elts: tuple[FormatBound]) # heterogeneous tuple
                  | ListFormat(elt: FormatBound)          # homogeneous list

- ``None`` is used for booleans, contexts, foreign values, function values, and
  any other expression for which a number format is not meaningful.
- :class:`SetFormat` describes an expression with a known, finite set of real
  values (e.g. a numeric literal or a join of numeric literals).  It is more
  precise than any :class:`Format` containing all of its values.
- A scalar :class:`Format` (e.g. ``IEEEFormat(es=8, nbits=32)``) describes a
  real-valued expression.  ``REAL_FORMAT`` is the **scalar top** — unrestricted
  real values (the format is unknown or unconstrained).
- :class:`TupleFormat` tracks per-element formats and is **not** weakened to a
  single scalar.
- :class:`ListFormat` tracks a single element format; the formats of all
  elements in a list expression are joined to obtain it (lists are homogeneous).

**Join rule** (least upper bound)::

    join(None, None)                 = None
    join(SetFormat(a), SetFormat(b)) = SetFormat(a ∪ b)
    join(SetFormat(s), fmt)          = fmt   if every v in s is representable in fmt
                                     = REAL_FORMAT   otherwise
    join(f, f)                       = f                 (scalar Format)
    join(f1, f2)                     = (AbstractFormat.from_format(f1)
                                        | AbstractFormat.from_format(f2)).format()
                                                          (when both abstractable)
                                     = REAL_FORMAT        (otherwise)
    join(Tuple(a..), Tuple(b..))     = Tuple(join(ai, bi)..)
    join(List(a), List(b))           = List(join(a, b))

For loops, phi nodes are initialised from the pre-loop definition's format and
the body is iterated until the phi bounds stop changing.  The lattice has
finite height (``REAL_FORMAT`` tops the scalar sub-lattice, ``SetFormat``
values are drawn from the program's finitely many literals, and the structural
lattice is shape-preserving), so the fixpoint terminates.

Format inference rules
----------------------
- **Context-sensitive operations** (``NullaryOp``, ``UnaryOp``, ``BinaryOp``,
  ``TernaryOp``, ``NaryOp``): the result is rounded to the active rounding
  context's format, i.e. ``scope.ctx.format()`` when the context is concrete,
  or ``REAL_FORMAT`` when it is a symbolic variable.
- **Exact arithmetic** (``Neg``, ``Abs``, ``Add``, ``Sub``, ``Mul`` under a
  concrete :class:`RealContext`): tighter than the default rule.  When all
  operand formats are abstractable, the result is computed via
  :class:`AbstractFormat`'s arithmetic
  (``(AbstractFormat.from_format(f1) ⊕ AbstractFormat.from_format(f2)).format()``)
  rather than widening to ``REAL_FORMAT``.
- **Function calls** (``Call``): conservatively the top format of the callee's
  return type.
- **Variable references** (``Var``): the format of the variable's definition.
- **Numeric literals** (``Decnum``, ``Integer``, ``Rational``, …):
  ``SetFormat({v})`` — the singleton set containing the literal's exact value.
- **Booleans, comparisons, foreign values, attributes**: ``None``.
- **Tuple expressions**: ``TupleFormat`` of the per-element formats.
- **List expressions**: ``ListFormat`` of the join of element formats; the top
  format of the list's element type when the list is empty.
- **List comprehensions**: ``ListFormat`` of the body expression's format; the
  loop target is bound to the iterable's element format.
- **Indexing/slicing**: list indexing returns the list's element format; list
  slicing returns the same ``ListFormat`` as the value.
- **Inline conditionals** (``IfExpr``): ``join(then_fmt, else_fmt)``.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import TypeAlias

from ...ast.fpyast import *
from ...ast.visitor import Visitor
from ...number import Context
from ...number.context.format import Format
from ...number.context.real import REAL_FORMAT, RealContext
from ...number.number.reals import RealFloat
from ...utils import is_dyadic
from ...types import (
    Type,
    BoolType,
    RealType,
    ContextType,
    TupleType,
    ListType,
    FunctionType,
    VarType,
)

from ..context_use import ContextUse, ContextUseAnalysis, ContextScope, ContextUseSite
from ..define_use import DefineUse, DefineUseAnalysis
from ..reaching_defs import PhiDef, Definition, DefSite
from ..type_infer import TypeInfer, TypeAnalysis
from .format import AbstractFormat, AbstractableFormat

__all__ = [
    'FormatInfer',
    'FormatAnalysis',
    'FormatBound',
    'SetFormat',
    'TupleFormat',
    'ListFormat',
]


#####################################################################
# Format lattice

@dataclass(frozen=True)
class SetFormat:
    """
    Format for a real-valued expression with a known, finite set of values.

    Strictly more precise than any :class:`Format` that contains every value;
    when joined with such a format the format is returned (otherwise the join
    widens to ``REAL_FORMAT``).
    """
    values: frozenset[Fraction]


@dataclass(frozen=True)
class TupleFormat:
    """Format for a tuple-valued expression."""
    elts: tuple['FormatBound', ...]


@dataclass(frozen=True)
class ListFormat:
    """Format for a list-valued expression (homogeneous element format)."""
    elt: 'FormatBound'


FormatBound: TypeAlias = 'None | SetFormat | Format | TupleFormat | ListFormat'
"""
Inferred format for an expression or variable definition.

- ``None`` — no numeric format (booleans, contexts, foreign values, …).
- :class:`SetFormat` — known finite set of real values; more precise than any
  format containing them.
- :class:`Format` — scalar format; ``REAL_FORMAT`` is the top of the scalar
  lattice.
- :class:`TupleFormat` — heterogeneous tuple, per-element formats preserved.
- :class:`ListFormat` — homogeneous list, single element format (the join of
  all element formats).
"""


def _all_representable_in(values: frozenset[Fraction], fmt: Format) -> bool:
    """
    Returns true iff every value in *values* is representable under *fmt*.

    ``REAL_FORMAT`` represents all reals, so it admits everything.  For any
    other format we require each value to be a dyadic rational (binary FP
    formats only represent dyadic rationals) and to satisfy
    :py:meth:`Format.representable_in`.
    """
    if fmt == REAL_FORMAT:
        return True
    for v in values:
        if not is_dyadic(v):
            return False
        if not fmt.representable_in(RealFloat.from_rational(v)):
            return False
    return True


def _top_bound(ty: Type) -> FormatBound:
    """
    Returns the top of the format lattice for *ty*.

    For scalar real values this is ``REAL_FORMAT``; for tuples and lists it is
    a structural format with ``REAL_FORMAT`` (or ``None``) at the leaves.  For
    non-numeric types (bool, context, function, type variable) it is ``None``.
    """
    match ty:
        case RealType():
            return REAL_FORMAT
        case TupleType():
            return TupleFormat(tuple(_top_bound(t) for t in ty.elts))
        case ListType():
            return ListFormat(_top_bound(ty.elt))
        case BoolType() | ContextType() | FunctionType() | VarType():
            return None
        case _:
            raise RuntimeError(f'unreachable: unknown type {ty!r}')


def _join_bounds(s1: FormatBound, s2: FormatBound) -> FormatBound:
    """
    Returns the join (least upper bound) of two formats.

    Joins are structural and only defined for formats of matching kind — the
    type checker guarantees this for well-typed programs.
    """
    match s1, s2:
        case None, None:
            return None
        case SetFormat(values=a), SetFormat(values=b):
            return SetFormat(a | b)
        case SetFormat(values=vals), Format() as fmt:
            return fmt if _all_representable_in(vals, fmt) else REAL_FORMAT
        case Format() as fmt, SetFormat(values=vals):
            return fmt if _all_representable_in(vals, fmt) else REAL_FORMAT
        case Format(), Format():
            if s1 == s2:
                return s1
            if isinstance(s1, AbstractableFormat) and isinstance(s2, AbstractableFormat):
                return (AbstractFormat.from_format(s1) | AbstractFormat.from_format(s2)).format()
            return REAL_FORMAT
        case TupleFormat(elts=a), TupleFormat(elts=b) if len(a) == len(b):
            return TupleFormat(tuple(_join_bounds(x, y) for x, y in zip(a, b)))
        case ListFormat(elt=a), ListFormat(elt=b):
            return ListFormat(_join_bounds(a, b))
        case _:
            raise RuntimeError(
                f'unreachable: cannot join incompatible formats {s1!r}, {s2!r}'
            )


def _list_set_widen(
    value_fmt: FormatBound,
    depth: int,
    insert_fmt: FormatBound,
) -> FormatBound:
    """
    Widen a list format for a functional update at *depth* levels of nesting.

    A ``ListSet`` expression ``set(xs, i1, …, iN, val)`` produces a new list
    in which ``xs[i1]…[iN]`` is replaced by *val*.  The result's format is the
    join of the original format with *insert_fmt* at the leaf level after
    peeling *depth* layers of :class:`ListFormat` (one per index).
    """
    if depth == 0:
        return _join_bounds(value_fmt, insert_fmt)
    assert isinstance(value_fmt, ListFormat), \
        f'expected ListFormat at depth {depth}, got {value_fmt!r}'
    return ListFormat(_list_set_widen(value_fmt.elt, depth - 1, insert_fmt))


def _format_of_scope(scope: ContextScope) -> Format:
    """
    Returns the number format associated with a context scope.

    If the scope carries a concrete :class:`Context`, the format is
    ``ctx.format()``.  For symbolic (unresolved) context variables the
    analysis falls back to ``REAL_FORMAT``.
    """
    if isinstance(scope.ctx, Context):
        return scope.ctx.format()
    return REAL_FORMAT


#####################################################################
# Analysis result

@dataclass
class FormatAnalysis:
    """
    Result of format analysis for an FPy function.

    Maps each variable definition site and expression to its inferred
    format.  For real-valued expressions the format is a :class:`Format`;
    for booleans and other non-numeric expressions it is ``None``; for tuples
    and lists it is a structural :class:`TupleFormat` or :class:`ListFormat`.
    """

    type_info: TypeAnalysis
    """Underlying basic-type analysis (bool, real, list, …)."""

    ctx_use: ContextUseAnalysis
    """Underlying context-use analysis (maps operations to rounding scopes)."""

    by_def: dict[Definition, FormatBound]
    """
    Format inferred for each variable definition site.

    Keys are ``AssignDef`` or ``PhiDef`` objects from the definition-use
    analysis.  For phi nodes the format is the join of the two incoming
    control-flow edge formats.
    """

    by_expr: dict[Expr, FormatBound]
    """
    Format inferred for each expression.

    For context-sensitive operations the format is that of the active
    rounding context.  For variable references it is the definition's format.
    For non-numeric expressions it is ``None``.
    """


#####################################################################
# Internal analysis visitor

class _FormatInferInstance(Visitor):
    """
    Single-use visitor that performs format inference.

    The visitor walks the entire function body in a single forward pass
    (loops are iterated to a fixpoint over their phi bounds) and populates
    ``by_def`` and ``by_expr``.
    """

    func: FuncDef
    type_info: TypeAnalysis
    ctx_use: ContextUseAnalysis

    by_def: dict[Definition, FormatBound]
    by_expr: dict[Expr, FormatBound]

    def __init__(
        self,
        func: FuncDef,
        type_info: TypeAnalysis,
        ctx_use: ContextUseAnalysis,
    ):
        self.func = func
        self.type_info = type_info
        self.ctx_use = ctx_use
        self.by_def = {}
        self.by_expr = {}

    # ------------------------------------------------------------------
    # Helpers

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.type_info.def_use

    def _set_def_bound(self, d: Definition, fmt: FormatBound):
        self.by_def[d] = fmt

    def _bound_of_def(self, d: Definition) -> FormatBound:
        return self.by_def[d]

    def _scope_format(self, e: ContextUseSite) -> Format:
        """Returns the format of the rounding context scope for *e*."""
        return _format_of_scope(self.ctx_use.find_scope_from_use(e))

    def _op_bound(self, e: ContextUseSite) -> FormatBound:
        """
        Format of a rounded operation's result.

        Real-valued operations are rounded to the active context's format.
        Operations that return any other type (bool, list, context, …) are
        not numerical computations — their format is derived from the type.
        """
        ret_ty = self.type_info.by_expr[e]
        if isinstance(ret_ty, RealType):
            return self._scope_format(e)
        return _top_bound(ret_ty)

    def _is_real_scope(self, e: ContextUseSite) -> bool:
        """Returns True iff *e*'s active scope is a concrete :class:`RealContext`.

        Distinct from ``_scope_format(e) == REAL_FORMAT`` because the latter
        also fires for symbolic (unresolved) context variables, where we
        cannot assume the rounding is the identity.
        """
        return isinstance(self.ctx_use.find_scope_from_use(e).ctx, RealContext)

    def _exact_arith_bound(
        self,
        e: ContextUseSite,
        operands: tuple[FormatBound, ...],
    ) -> FormatBound | None:
        """
        Tries to compute a tight format for an exact arithmetic operation.

        Under a concrete :class:`RealContext` (no rounding), negation, abs,
        addition, subtraction, and multiplication preserve enough structure
        that a containing :class:`Format` can be derived from the operand
        formats via :class:`AbstractFormat`'s arithmetic.

        Returns ``None`` to signal that the caller should fall back to
        :meth:`_op_bound` (e.g., the operator is not exact-arithmetic, the
        scope is not concretely REAL, or some operand is not abstractable).
        """
        if not self._is_real_scope(e):
            return None
        afs: list[AbstractFormat] = []
        for f in operands:
            if not isinstance(f, AbstractableFormat):
                return None
            afs.append(AbstractFormat.from_format(f))
        match e, afs:
            case Neg(), [af]:
                result = -af
            case Abs(), [af]:
                result = abs(af)
            case Add(), [a, b]:
                result = a + b
            case Sub(), [a, b]:
                result = a - b
            case Mul(), [a, b]:
                result = a * b
            case _:
                return None
        return result.format()

    def _visit_binding(
        self,
        site: DefSite,
        binding: Id | TupleBinding,
        fmt: FormatBound,
    ):
        """Records *fmt* for every variable introduced by *binding* at *site*."""
        match binding:
            case NamedId():
                d = self.def_use.find_def_from_site(binding, site)
                self._set_def_bound(d, fmt)
            case UnderscoreId():
                pass
            case TupleBinding():
                if not isinstance(fmt, TupleFormat) or len(fmt.elts) != len(binding.elts):
                    raise RuntimeError(f'expected TupleFormat of length {len(binding.elts)}, got {fmt!r}')
                for sub_binding, sub_fmt in zip(binding.elts, fmt.elts):
                    self._visit_binding(site, sub_binding, sub_fmt)
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    # ------------------------------------------------------------------
    # Expression visitors — return the inferred FormatBound for *e*

    def _visit_expr(self, e: Expr, ctx: None) -> FormatBound:  # type: ignore[override]
        """Dispatch, record in ``by_expr``, and return the inferred format."""
        fmt: FormatBound = super()._visit_expr(e, ctx)
        self.by_expr[e] = fmt
        return fmt

    # Numeric literals: exact real values bounded by the singleton set {v}
    def _visit_decnum(self, e: Decnum, ctx: None) -> FormatBound:
        return SetFormat(frozenset((e.as_rational(),)))

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> FormatBound:
        return SetFormat(frozenset((e.as_rational(),)))

    def _visit_integer(self, e: Integer, ctx: None) -> FormatBound:
        return SetFormat(frozenset((e.as_rational(),)))

    def _visit_rational(self, e: Rational, ctx: None) -> FormatBound:
        return SetFormat(frozenset((e.as_rational(),)))

    def _visit_digits(self, e: Digits, ctx: None) -> FormatBound:
        return SetFormat(frozenset((e.as_rational(),)))

    # Non-real-valued leaves
    def _visit_bool(self, e: BoolVal, ctx: None) -> FormatBound:
        return None

    def _visit_foreign(self, e: ForeignVal, ctx: None) -> FormatBound:
        return None

    def _visit_attribute(self, e: Attribute, ctx: None) -> FormatBound:
        self._visit_expr(e.value, ctx)
        return None

    # Variable reference: propagate from the definition
    def _visit_var(self, e: Var, ctx: None) -> FormatBound:
        d = self.def_use.find_def_from_use(e)
        return self._bound_of_def(d)

    # Context-sensitive operations: real-valued results take the active scope's
    # format; non-real results (bool predicates, range, declcontext, …) take
    # the top format of their inferred type.
    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> FormatBound:
        return self._op_bound(e)

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> FormatBound:
        arg_fmt = self._visit_expr(e.arg, ctx)
        tight = self._exact_arith_bound(e, (arg_fmt,))
        return tight if tight is not None else self._op_bound(e)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> FormatBound:
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)
        tight = self._exact_arith_bound(e, (lhs, rhs))
        return tight if tight is not None else self._op_bound(e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> FormatBound:
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        return self._op_bound(e)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> FormatBound:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return self._op_bound(e)

    # Function calls: conservatively the top format of the return type
    def _visit_call(self, e: Call, ctx: None) -> FormatBound:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)
        return _top_bound(self.type_info.by_expr[e])

    # Comparison: produces a bool, so no numeric format
    def _visit_compare(self, e: Compare, ctx: None) -> FormatBound:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        return None

    # Compound expressions
    def _visit_tuple_expr(self, e: TupleExpr, ctx: None) -> FormatBound:
        return TupleFormat(tuple(self._visit_expr(elt, ctx) for elt in e.elts))

    def _visit_list_expr(self, e: ListExpr, ctx: None) -> FormatBound:
        elt_fmts = [self._visit_expr(elt, ctx) for elt in e.elts]
        if elt_fmts:
            joined = reduce(_join_bounds, elt_fmts)
        else:
            # Empty list: derive the element format from the inferred list type.
            list_ty = self.type_info.by_expr[e]
            assert isinstance(list_ty, ListType)
            joined = _top_bound(list_ty.elt)
        return ListFormat(joined)

    def _visit_list_comp(self, e: ListComp, ctx: None) -> FormatBound:
        for target, iterable in zip(e.targets, e.iterables):
            iter_fmt = self._visit_expr(iterable, ctx)
            assert isinstance(iter_fmt, ListFormat)
            self._visit_binding(e, target, iter_fmt.elt)
        body_fmt = self._visit_expr(e.elt, ctx)
        return ListFormat(body_fmt)

    def _visit_list_ref(self, e: ListRef, ctx: None) -> FormatBound:
        value_fmt = self._visit_expr(e.value, ctx)
        self._visit_expr(e.index, ctx)
        assert isinstance(value_fmt, ListFormat)
        return value_fmt.elt

    def _visit_list_slice(self, e: ListSlice, ctx: None) -> FormatBound:
        value_fmt = self._visit_expr(e.value, ctx)
        if e.start is not None:
            self._visit_expr(e.start, ctx)
        if e.stop is not None:
            self._visit_expr(e.stop, ctx)
        # Slice of a list has the same format as the list itself.
        return value_fmt

    def _visit_list_set(self, e: ListSet, ctx: None) -> FormatBound:
        # Functional update: the result is a new list with the inserted
        # element joined into the original element format at depth len(indices).
        value_fmt = self._visit_expr(e.value, ctx)
        for s in e.indices:
            self._visit_expr(s, ctx)
        insert_fmt = self._visit_expr(e.expr, ctx)
        return _list_set_widen(value_fmt, len(e.indices), insert_fmt)

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> FormatBound:
        self._visit_expr(e.cond, ctx)
        then_fmt = self._visit_expr(e.ift, ctx)
        else_fmt = self._visit_expr(e.iff, ctx)
        return _join_bounds(then_fmt, else_fmt)

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_assign(self, stmt: Assign, ctx: None):
        fmt = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, fmt)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        # The list variable's definition format is unchanged by element mutation.

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
            rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
            self._set_def_bound(phi, _join_bounds(lhs, rhs))

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
            rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
            self._set_def_bound(phi, _join_bounds(lhs, rhs))

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        # Initialise phi formats from pre-loop (lhs) definitions so that the
        # loop body can reference them before the back-edge has been processed.
        for phi in self.def_use.phis[stmt]:
            self._set_def_bound(phi, self._bound_of_def(self.def_use.defs[phi.lhs]))
        # Iterate body + join until phi bounds stop changing.
        while True:
            prev = {phi: self.by_def[phi] for phi in self.def_use.phis[stmt]}
            self._visit_expr(stmt.cond, ctx)
            self._visit_block(stmt.body, ctx)
            for phi in self.def_use.phis[stmt]:
                lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
                rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
                self._set_def_bound(phi, _join_bounds(lhs, rhs))
            if all(self.by_def[phi] == prev[phi] for phi in self.def_use.phis[stmt]):
                break

    def _visit_for(self, stmt: ForStmt, ctx: None):
        iter_fmt = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_fmt, ListFormat)
        self._visit_binding(stmt, stmt.target, iter_fmt.elt)
        # Initialise phi formats from pre-loop (lhs) definitions
        for phi in self.def_use.phis[stmt]:
            self._set_def_bound(phi, self._bound_of_def(self.def_use.defs[phi.lhs]))
        # Iterate body + join until phi bounds stop changing.
        while True:
            prev = {phi: self.by_def[phi] for phi in self.def_use.phis[stmt]}
            self._visit_block(stmt.body, ctx)
            for phi in self.def_use.phis[stmt]:
                lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
                rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
                self._set_def_bound(phi, _join_bounds(lhs, rhs))
            if all(self.by_def[phi] == prev[phi] for phi in self.def_use.phis[stmt]):
                break

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        # The context expression itself is not a numerical computation.
        # Record the context variable (if named) with format ``None``.
        if isinstance(stmt.target, NamedId):
            d = self.def_use.find_def_from_site(stmt.target, stmt)
            self._set_def_bound(d, None)
        self._visit_block(stmt.body, ctx)

    def _visit_assert(self, stmt: AssertStmt, ctx: None):
        self._visit_expr(stmt.test, ctx)
        if stmt.msg is not None:
            self._visit_expr(stmt.msg, ctx)

    def _visit_effect(self, stmt: EffectStmt, ctx: None):
        self._visit_expr(stmt.expr, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx: None):
        self._visit_expr(stmt.expr, ctx)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        pass

    def _visit_block(self, block: StmtBlock, ctx: None):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        # Arguments: top format of the declared type (REAL_FORMAT for reals,
        # None for non-numeric, structural for tuples/lists).
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._set_def_bound(d, _top_bound(self.type_info.by_def[d]))
        # Free variables (captured from outer scope): top format of inferred type.
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            self._set_def_bound(d, _top_bound(self.type_info.by_def[d]))
        self._visit_block(func.body, ctx)

    def analyze(self) -> FormatAnalysis:
        self._visit_function(self.func, None)
        return FormatAnalysis(
            type_info=self.type_info,
            ctx_use=self.ctx_use,
            by_def=self.by_def,
            by_expr=self.by_expr,
        )


#####################################################################
# Public API

class FormatInfer:
    """
    Format inference for FPy functions.

    This analysis bounds the number format for each expression and variable
    definition in an FPy program.  It is an **alternative** to
    :class:`ContextInfer`: rather than threading context information through a
    Hindley-Milner style type system, it uses two simpler pre-analyses
    (:class:`TypeInfer` and :class:`ContextUse`) and then performs a single
    forward pass over the AST.

    **Format lattice**::

        FormatBound ::= None
                      | SetFormat(values)     (known finite set of reals)
                      | Format               (REAL_FORMAT is the scalar top)
                      | TupleFormat(elts)
                      | ListFormat(elt)

    **Join rule**::

        join(None, None)                 = None
        join(SetFormat(a), SetFormat(b)) = SetFormat(a ∪ b)
        join(SetFormat(s), fmt)          = fmt if every value in s fits in fmt
                                         = REAL_FORMAT otherwise
        join(f, f)                       = f
        join(f1, f2)                     = (AbstractFormat.from_format(f1)
                                            | AbstractFormat.from_format(f2)).format()
                                                          (when both abstractable)
                                         = REAL_FORMAT     (otherwise)
        join(Tuple(a..), Tuple(b..))     = Tuple(join(ai, bi)..)
        join(List(a), List(b))           = List(join(a, b))

    This rule is applied at all control-flow merge points (phi nodes), including
    branch merges (``if``/``if1``) and loop back-edges (``while``/``for``).
    Loops iterate body + join until phi bounds stop changing; the lattice has
    finite height, so the fixpoint terminates.

    **Usage**::

        from fpy2.analysis import FormatInfer

        info = FormatInfer.analyze(func)
        for d, fmt in info.by_def.items():
            print(d.name, '->', fmt)
    """

    @staticmethod
    def analyze(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
    ) -> FormatAnalysis:
        """
        Performs format analysis on a compiled FPy function.

        All three pre-analyses (``def_use``, ``type_info``, ``ctx_use``) are
        computed automatically when not supplied.  Pre-computing them externally
        is useful when they are also needed for other purposes.

        Args:
            func:      The :class:`FuncDef` AST node to analyse.
            def_use:   Optional pre-computed definition-use analysis.
            type_info: Optional pre-computed type analysis.
            ctx_use:   Optional pre-computed context-use analysis.

        Returns:
            A :class:`FormatAnalysis` result whose ``by_def`` and ``by_expr``
            maps contain the inferred formats for every definition site
            and expression in *func*.

        Raises:
            TypeError: If *func* is not a :class:`FuncDef`.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {type(func)}")

        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        if ctx_use is None:
            ctx_use = ContextUse.analyze(func, def_use=def_use)

        return _FormatInferInstance(func, type_info, ctx_use).analyze()
