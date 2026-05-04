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
**infinite ascending chains** in the scalar sub-lattice when exact arithmetic
(``+``/``-``/``*`` under :class:`RealContext`, see below) is applied to a
phi'd value: each iteration widens the resulting :class:`AbstractFormat`'s
precision and bounds without bound.

When a ``for`` loop's iterable has a **statically-known length** (per
:class:`ArraySizeAnalysis`), the analysis drives the phi update for
*exactly* that many body executions instead of iterating to a fixpoint.
This matches the runtime semantics exactly and avoids any widening fall-back
— important for the exact-arithmetic lattice.

For ``for`` loops with symbolic-length iterables and for ``while`` loops, the
iteration count isn't known statically.  These fall back to the fixpoint
loop, which runs at most ``loop_iter_limit`` iterations before switching joins
to **widen-mode** — distinct scalar :class:`Format` inputs fall back to
``REAL_FORMAT`` instead of going through :class:`AbstractFormat` union, at
which point the fixpoint converges in one more iteration.

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
- **Sum reduction** (``Sum`` under a concrete :class:`RealContext` over a
  list whose size is statically known via :class:`ArraySizeAnalysis`):
  simulates ``n - 1`` exact pairwise additions through
  :class:`AbstractFormat` instead of widening.  Under non-REAL contexts
  every pairwise add rounds to the scope's format, so the result is just
  the scope's format (the default rule).
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
from typing import Callable, Iterable, TypeAlias

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

from ..array_size import ArraySizeAnalysis, ArraySizeInfer, ListSize
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


ScalarFormatBound: TypeAlias = None | SetFormat | Format
FormatBound: TypeAlias = None | SetFormat | Format | TupleFormat | ListFormat
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


def _setformat_to_abstract(s: SetFormat) -> AbstractFormat | None:
    """
    Returns a tight :class:`AbstractFormat` containing every value in *s*,
    or ``None`` when any value is non-dyadic (and therefore can't be
    pinned by an :class:`AbstractFormat`'s integer ``prec``/``exp``).

    Used by :meth:`_exact_arith_bound` to bridge a :class:`SetFormat`
    operand into the :class:`AbstractFormat` arithmetic path so that
    e.g. ``SetFormat({0}) + EFloatFormat`` doesn't bail to
    ``REAL_FORMAT`` under exact-arithmetic.
    """
    if not s.values:
        return None
    if not all(is_dyadic(v) for v in s.values):
        return None
    rfs = [RealFloat.from_rational(v) for v in s.values]
    # Bounds: max non-negative value and min non-positive value, with
    # zero used when no value falls on the corresponding side.  This
    # matches AbstractFormat's convention of pos_bound >= 0 and
    # neg_bound <= 0.
    pos_rfs = [rf for rf in rfs if not rf.s and not rf.is_zero()]
    neg_rfs = [rf for rf in rfs if rf.s]
    pos_bound = max(pos_rfs, default=RealFloat.from_int(0))
    neg_bound = min(neg_rfs, default=RealFloat.from_int(0))
    # Precision and exponent: tight enough to represent every value.
    # ``RealFloat.p`` is 0 for zero values; AbstractFormat requires
    # ``prec >= 1``, so floor at 1.
    prec = max((rf.p for rf in rfs), default=1)
    if prec < 1:
        prec = 1
    exp = min((rf.exp for rf in rfs), default=0)
    return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound)


def _to_abstract(f: ScalarFormatBound) -> AbstractFormat | None:
    """
    Lifts an abstractable :class:`FormatBound` to an
    :class:`AbstractFormat`.  Returns ``None`` for variants that the
    abstract arithmetic doesn't accept (``None``, ``TupleFormat``,
    ``ListFormat``, non-dyadic ``SetFormat``).
    """
    if isinstance(f, AbstractableFormat):
        return AbstractFormat.from_format(f)
    if isinstance(f, SetFormat):
        return _setformat_to_abstract(f)
    return None


def _join_bounds(
    s1: FormatBound,
    s2: FormatBound,
    *,
    widen: bool = False,
) -> FormatBound:
    """
    Returns the join (least upper bound) of two formats.

    Joins are structural and only defined for formats of matching kind — the
    type checker guarantees this for well-typed programs.

    When ``widen`` is true, the scalar ``Format ⊔ Format`` case for distinct
    inputs falls back to ``REAL_FORMAT`` instead of going through
    :class:`AbstractFormat`-mediated union.  This is used inside loop
    fixpoints once an iteration cap is reached, to force convergence on
    AbstractFormat lattices that have infinite ascending chains under
    arithmetic.
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
            if widen:
                return REAL_FORMAT
            if isinstance(s1, AbstractableFormat) and isinstance(s2, AbstractableFormat):
                af1 = AbstractFormat.from_format(s1)
                af2 = AbstractFormat.from_format(s2)
                # Subsumption: if one input contains the other, return the
                # original (un-canonicalized) Format directly.  This is
                # important for **fixpoint idempotence**: ``(af | af).format()``
                # may return an MPB-float that is value-equivalent to the
                # input but compares unequal under ``==`` (e.g.,
                # ``IEEEFormat(FP64)`` vs the corresponding
                # ``MPBFloatFormat``), which would prevent loop phis from
                # detecting convergence.
                if af1 <= af2:
                    return s2
                if af2 <= af1:
                    return s1
                return (af1 | af2).format()
            return REAL_FORMAT
        case TupleFormat(elts=a), TupleFormat(elts=b) if len(a) == len(b):
            return TupleFormat(tuple(_join_bounds(x, y, widen=widen) for x, y in zip(a, b)))
        case ListFormat(elt=a), ListFormat(elt=b):
            return ListFormat(_join_bounds(a, b, widen=widen))
        case _:
            raise RuntimeError(
                f'unreachable: cannot join incompatible formats {s1!r}, {s2!r}'
            )


def _list_set_widen(
    value_fmt: FormatBound,
    depth: int,
    insert_fmt: FormatBound,
    *,
    widen: bool = False,
) -> FormatBound:
    """
    Widen a list format for a functional update at *depth* levels of nesting.

    A ``ListSet`` expression ``set(xs, i1, …, iN, val)`` produces a new list
    in which ``xs[i1]…[iN]`` is replaced by *val*.  The result's format is the
    join of the original format with *insert_fmt* at the leaf level after
    peeling *depth* layers of :class:`ListFormat` (one per index).

    *widen* propagates to the leaf-level :func:`_join_bounds` call so that
    ListSet expressions inside a saturated loop iteration also widen.
    """
    if depth == 0:
        return _join_bounds(value_fmt, insert_fmt, widen=widen)
    assert isinstance(value_fmt, ListFormat), \
        f'expected ListFormat at depth {depth}, got {value_fmt!r}'
    return ListFormat(_list_set_widen(value_fmt.elt, depth - 1, insert_fmt, widen=widen))


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
    array_size: ArraySizeAnalysis

    by_def: dict[Definition, FormatBound]
    by_expr: dict[Expr, FormatBound]

    def __init__(
        self,
        func: FuncDef,
        type_info: TypeAnalysis,
        ctx_use: ContextUseAnalysis,
        array_size: ArraySizeAnalysis,
        loop_iter_limit: int,
    ):
        self.func = func
        self.type_info = type_info
        self.ctx_use = ctx_use
        self.array_size = array_size
        self.by_def = {}
        self.by_expr = {}
        self._loop_iter_limit = loop_iter_limit
        # When True, joins forced inside a saturated loop fixpoint widen
        # distinct scalar Formats to ``REAL_FORMAT`` instead of going through
        # an :class:`AbstractFormat` union (which has infinite ascending
        # chains under arithmetic).  Saved/restored around each loop so that
        # nested loops manage their own state.
        self._widen: bool = False

    # ------------------------------------------------------------------
    # Helpers

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.type_info.def_use

    def _set_def_bound(self, d: Definition, fmt: FormatBound):
        self.by_def[d] = fmt

    def _bound_of_def(self, d: Definition) -> FormatBound:
        return self.by_def[d]

    def _join(self, s1: FormatBound, s2: FormatBound) -> FormatBound:
        """Join two formats, respecting the visitor's current widen state."""
        return _join_bounds(s1, s2, widen=self._widen)

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
    ) -> Format | SetFormat | None:
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
        match e:
            case Neg():
                assert len(operands) == 1
                op_fmt, = operands
                assert isinstance(op_fmt, ScalarFormatBound), f'expected scalar format for operand of Neg, got {op_fmt!r}'
                if isinstance(op_fmt, AbstractableFormat):
                    return (-AbstractFormat.from_format(op_fmt)).format()
                elif isinstance(op_fmt, SetFormat):
                    return SetFormat(frozenset(-v for v in op_fmt.values))
                else:
                    return None
            case Abs():
                assert len(operands) == 1
                op_fmt, = operands
                assert isinstance(op_fmt, ScalarFormatBound), f'expected scalar format for operand of Abs, got {op_fmt!r}'
                if isinstance(op_fmt, AbstractableFormat):
                    return abs(AbstractFormat.from_format(op_fmt)).format()
                elif isinstance(op_fmt, SetFormat):
                    return SetFormat(frozenset(abs(v) for v in op_fmt.values))
                else:
                    return None
            case Add():
                assert len(operands) == 2
                a, b = operands
                assert isinstance(a, ScalarFormatBound), f'expected scalar format for first operand of Add, got {a!r}'
                assert isinstance(b, ScalarFormatBound), f'expected scalar format for second operand of Add, got {b!r}'
                # Both SetFormat: keep precision by pairwise sum.
                if isinstance(a, SetFormat) and isinstance(b, SetFormat):
                    return SetFormat(frozenset(va + vb for va in a.values for vb in b.values))
                # Mixed or both AbstractableFormat: lift to AbstractFormat.
                af_a = _to_abstract(a)
                af_b = _to_abstract(b)
                if af_a is None or af_b is None:
                    return None
                return (af_a + af_b).format()
            case Sub():
                assert len(operands) == 2
                a, b = operands
                assert isinstance(a, ScalarFormatBound), f'expected scalar format for first operand of Sub, got {a!r}'
                assert isinstance(b, ScalarFormatBound), f'expected scalar format for second operand of Sub, got {b!r}'
                if isinstance(a, SetFormat) and isinstance(b, SetFormat):
                    return SetFormat(frozenset(va - vb for va in a.values for vb in b.values))
                af_a = _to_abstract(a)
                af_b = _to_abstract(b)
                if af_a is None or af_b is None:
                    return None
                return (af_a - af_b).format()
            case Mul():
                assert len(operands) == 2
                a, b = operands
                assert isinstance(a, ScalarFormatBound), f'expected scalar format for first operand of Mul, got {a!r}'
                assert isinstance(b, ScalarFormatBound), f'expected scalar format for second operand of Mul, got {b!r}'
                if isinstance(a, SetFormat) and isinstance(b, SetFormat):
                    return SetFormat(frozenset(va * vb for va in a.values for vb in b.values))
                af_a = _to_abstract(a)
                af_b = _to_abstract(b)
                if af_a is None or af_b is None:
                    return None
                return (af_a * af_b).format()
            case _:
                # unsupported operation
                return None

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
        if isinstance(e, Sum):
            assert isinstance(arg_fmt, ListFormat), f'expected ListFormat for argument of Sum, got {arg_fmt!r}'
            return self._sum_bound(e, arg_fmt)
        tight = self._exact_arith_bound(e, (arg_fmt,))
        return tight if tight is not None else self._op_bound(e)

    def _sum_bound(self, e: Sum, arg_fmt: ListFormat) -> FormatBound:
        """
        Format of ``Sum(xs)`` — reduce-by-pairwise-addition with rounding
        under the active context at each step.

        - **Non-REAL scope**: each pairwise add rounds to the scope's
          format, so the accumulator equals the scope's format
          throughout.  Falls back to :meth:`_op_bound`.
        - **REAL scope**: no rounding.  Simulate ``n - 1`` exact
          additions through :class:`AbstractFormat` to produce a precise
          containing format.  Requires the list size to be statically
          known via :class:`ArraySizeAnalysis`; otherwise falls back to
          ``REAL_FORMAT`` via :meth:`_op_bound`.
        """
        if not self._is_real_scope(e):
            return self._op_bound(e)
        n = self._known_iter_count(e.arg)
        if n is None:
            return self._op_bound(e)
        elt_fmt = arg_fmt.elt
        if n == 0:
            # ``sum([])`` is conventionally 0.
            return SetFormat(frozenset((Fraction(0),)))
        if n == 1:
            # No addition occurs; the lone element passes through.
            return elt_fmt
        # Lift the element format once and accumulate ``n - 1`` times.
        assert isinstance(elt_fmt, ScalarFormatBound), f'expected scalar format for elements of sum operand, got {elt_fmt!r}'
        af_elt = _to_abstract(elt_fmt)
        if af_elt is None:
            return self._op_bound(e)
        af_acc = af_elt
        for _ in range(n - 1):
            af_acc = af_acc + af_elt
        return af_acc.format()

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
            joined = reduce(self._join, elt_fmts)
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
        return _list_set_widen(value_fmt, len(e.indices), insert_fmt, widen=self._widen)

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> FormatBound:
        self._visit_expr(e.cond, ctx)
        then_fmt = self._visit_expr(e.ift, ctx)
        else_fmt = self._visit_expr(e.iff, ctx)
        return self._join(then_fmt, else_fmt)

    # ------------------------------------------------------------------
    # Statement visitors

    def _visit_assign(self, stmt: Assign, ctx: None):
        fmt = self._visit_expr(stmt.expr, ctx)
        self._visit_binding(stmt, stmt.target, fmt)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        # ``xs[i1]…[iN] = expr`` is treated as ``xs = update(xs, …, expr)``
        # — a fresh SSA def of ``xs`` (per ``reaching_defs``).  The new
        # def's format is the original element format widened with the
        # inserted value's format at depth ``len(indices)``, matching the
        # semantics of ``ListSet`` (see :func:`_list_set_widen`).
        d_use = self.def_use.find_def_from_use(stmt)
        value_fmt = self._bound_of_def(d_use)
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        insert_fmt = self._visit_expr(stmt.expr, ctx)
        new_fmt = _list_set_widen(
            value_fmt, len(stmt.indices), insert_fmt, widen=self._widen
        )
        d_def = self.def_use.find_def_from_site(stmt.var, stmt)
        self._set_def_bound(d_def, new_fmt)

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.body, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
            rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
            self._set_def_bound(phi, self._join(lhs, rhs))

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        self._visit_block(stmt.ift, ctx)
        self._visit_block(stmt.iff, ctx)
        for phi in self.def_use.phis[stmt]:
            lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
            rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
            self._set_def_bound(phi, self._join(lhs, rhs))

    def _fixpoint(self, phis: Iterable[PhiDef], run_body: Callable[[], None]):
        """
        Drive a loop's phi-bound fixpoint to convergence.

        Initialises each phi from its pre-loop (lhs) definition, then
        repeatedly runs *run_body* and joins the post-body (rhs) into each
        phi.  After ``_loop_iter_limit`` iterations without convergence,
        switches joins to widen-mode (``self._widen = True``) to force
        termination on infinite-height AbstractFormat chains (e.g., from
        exact arithmetic in the body).  Save/restore semantics mean an
        outer loop already in widen-mode propagates that into nested
        iterations.
        """
        for phi in phis:
            self._set_def_bound(phi, self._bound_of_def(self.def_use.defs[phi.lhs]))
        saved_widen = self._widen
        iter_count = 0
        while True:
            prev = {phi: self.by_def[phi] for phi in phis}
            self._widen = saved_widen or iter_count >= self._loop_iter_limit
            run_body()
            for phi in phis:
                lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
                rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
                self._set_def_bound(phi, self._join(lhs, rhs))
            if all(self.by_def[phi] == prev[phi] for phi in phis):
                break
            iter_count += 1
        self._widen = saved_widen

    def _unroll(
        self,
        phis: Iterable[PhiDef],
        run_body: Callable[[], None],
        n: int,
    ):
        """
        Drive a loop's phi update for *exactly* ``n`` body executions.

        Used when the iterable's length is statically known (via
        :class:`ArraySizeAnalysis`).  At runtime the loop walks exactly
        ``n`` body iterations, so the analysis can mirror that walk and
        avoid widening — important for the exact-arithmetic
        (``+``/``-``/``*`` under :class:`RealContext`) lattice, which has
        infinite ascending chains.

        Iter-by-iter visit produces a precise (if potentially wide)
        bound after exactly ``n`` joins; the result is sound and
        strictly more precise than the fixpoint+widening fall-back when
        ``n`` is small.
        """
        phis = list(phis)
        for phi in phis:
            self._set_def_bound(phi, self._bound_of_def(self.def_use.defs[phi.lhs]))
        if n == 0:
            # Body never runs; the phi stays at the pre-loop value.
            return
        for _ in range(n):
            run_body()
            for phi in phis:
                lhs = self._bound_of_def(self.def_use.defs[phi.lhs])
                rhs = self._bound_of_def(self.def_use.defs[phi.rhs])
                self._set_def_bound(phi, self._join(lhs, rhs))

    def _known_iter_count(self, iterable: Expr) -> int | None:
        """
        Returns the iterable's statically-known length, or ``None``.

        Consults :class:`ArraySizeAnalysis`.  Recognises ``ListSize``
        expressions whose ``size`` is a concrete ``int`` (e.g. ``range``
        with static bounds, list literals, comprehensions over
        known-size iterables).
        """
        bound = self.array_size.by_expr.get(iterable)
        if isinstance(bound, ListSize) and isinstance(bound.size, int):
            return bound.size
        return None

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        def iterate():
            self._visit_expr(stmt.cond, ctx)
            self._visit_block(stmt.body, ctx)

        self._fixpoint(self.def_use.phis[stmt], iterate)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        iter_fmt = self._visit_expr(stmt.iterable, ctx)
        assert isinstance(iter_fmt, ListFormat)
        self._visit_binding(stmt, stmt.target, iter_fmt.elt)

        def iterate():
            self._visit_block(stmt.body, ctx)

        # If the iterable's length is statically known, drive the phi
        # update for exactly that many body executions instead of
        # iterating to a fixpoint.  This matches the runtime semantics
        # exactly and avoids the widening fall-back for the
        # exact-arithmetic lattice.
        n = self._known_iter_count(stmt.iterable)
        if n is not None:
            self._unroll(self.def_use.phis[stmt], iterate, n)
        else:
            self._fixpoint(self.def_use.phis[stmt], iterate)

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

    **Loops** are handled in one of two modes:

    - **Bounded iteration**: when a ``for`` loop's iterable has a
      statically-known length (per :class:`ArraySizeAnalysis`), the
      analysis drives the phi update for *exactly* that many body
      executions.  This mirrors runtime semantics and avoids any
      widening fall-back — important for the exact-arithmetic lattice,
      which has infinite ascending chains.
    - **Fixpoint + widening**: ``while`` loops and ``for`` loops over
      symbolic-length iterables iterate body + join until phi bounds
      stop changing.  The AbstractFormat-mediated scalar join introduces
      infinite ascending chains when exact arithmetic
      (``+``/``-``/``*`` under :class:`RealContext`) is applied to a
      phi'd value, so the fixpoint runs at most ``loop_iter_limit``
      iterations before switching joins to widen-mode (distinct scalar
      Formats fall back to ``REAL_FORMAT``) to force convergence.

    **Usage**::

        from fpy2.analysis import FormatInfer

        info = FormatInfer.analyze(func)
        for d, fmt in info.by_def.items():
            print(d.name, '->', fmt)
    """

    DEFAULT_LOOP_ITER_LIMIT = 10
    """
    Default number of body+join iterations a loop fixpoint runs before
    switching to widen-mode (joins of distinct scalar Formats fall back to
    ``REAL_FORMAT`` to force convergence).  Tunable via the ``loop_iter_limit``
    parameter of :meth:`analyze`.
    """

    @staticmethod
    def analyze(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
        array_size: ArraySizeAnalysis | None = None,
        loop_iter_limit: int = DEFAULT_LOOP_ITER_LIMIT,
    ) -> FormatAnalysis:
        """
        Performs format analysis on a compiled FPy function.

        All four pre-analyses (``def_use``, ``type_info``, ``ctx_use``,
        ``array_size``) are computed automatically when not supplied.
        Pre-computing them externally is useful when they are also needed
        for other purposes.

        Args:
            func:        The :class:`FuncDef` AST node to analyse.
            def_use:     Optional pre-computed definition-use analysis.
            type_info:   Optional pre-computed type analysis.
            ctx_use:     Optional pre-computed context-use analysis.
            array_size:  Optional pre-computed array-size analysis.
                Used to recognise ``for`` loops whose iterable has a
                statically-known length, so the phi update can be driven
                exactly that many times instead of iterating to a
                fixpoint.  This is strictly more precise — important for
                the exact-arithmetic lattice (``+``/``-``/``*`` under
                :class:`RealContext`) which has infinite ascending
                chains.
            loop_iter_limit:
                Number of loop body+join iterations to run before forcing
                joins of distinct scalar Formats to widen to ``REAL_FORMAT``.
                Only applies to loops without a known iteration count
                (``while`` loops, and ``for`` loops over symbolic-size
                iterables).  Defaults to :attr:`DEFAULT_LOOP_ITER_LIMIT`.

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
        if array_size is None:
            array_size = ArraySizeInfer.analyze(func, type_info=type_info)

        return _FormatInferInstance(
            func, type_info, ctx_use, array_size, loop_iter_limit=loop_iter_limit
        ).analyze()
