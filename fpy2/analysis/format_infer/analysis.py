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
(``+``/``-``/``*`` under :data:`REAL`, see below) is applied to a phi'd value:
each iteration widens the resulting :class:`AbstractFormat`'s precision and
bounds without bound.

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
- **Exact arithmetic** (``Neg``, ``Abs``, ``Add``, ``Sub``, ``Mul`` under
  the :data:`REAL` singleton): tighter than the default rule.  When all
  operand formats are abstractable, the result is computed via
  :class:`AbstractFormat`'s arithmetic
  (``(AbstractFormat.from_format(f1) ⊕ AbstractFormat.from_format(f2)).format()``)
  rather than widening to ``REAL_FORMAT``.  The check is identity
  against the singleton — other :class:`RealContext` instances are not
  recognised.
- **Sum reduction** (``Sum`` under :data:`REAL` over a list whose size is
  statically known via :class:`ArraySizeAnalysis`): simulates ``n - 1``
  exact pairwise additions through :class:`AbstractFormat` instead of
  widening.  Under non-REAL contexts every pairwise add rounds to the
  scope's format, so the result is just the scope's format (the default
  rule).
- **Integer-producing operations** (``Len``, ``Dim``, ``Size``,
  ``Range1``/``Range2``/``Range3``, the integer projection of
  ``Enumerate``): the result format is ``INTEGER`` regardless of the
  active rounding context.  These ops always produce integer values, so
  reporting the active context's format would be unnecessarily loose
  (e.g., ``len(xs)`` under ``with fp.FP32`` is still an integer, not an
  arbitrary FP32 value).  ``Range`` returns ``ListFormat(INTEGER)``;
  ``Enumerate(xs)`` returns ``ListFormat(TupleFormat(INTEGER, xs.elt))``.
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
- **Selection ops** (``Min`` / ``Max`` / ``AMin`` / ``AMax``): the result
  is exactly one operand, so the format is the join of operand formats
  (for the variadic forms) or the list element format (for the reduce
  forms) — no widening to the active scope.
"""

import operator

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from typing import Any, Callable, Iterable, TypeAlias

from ...ast.fpyast import *
from ...ast.visitor import Visitor
from ...function import Function
from ...number import Context, Float, RealFloat, INTEGER, REAL
from ...number.format import Format, REAL_FORMAT
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
from ..call_graph import CallGraph, CallGraphError
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
    'exact_binop',
    'exact_unop',
    'round_is_identity',
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

    @staticmethod
    def from_value(x: Fraction):
        return SetFormat(frozenset((x,)))


def _free_var_set_format(val: object) -> 'SetFormat | None':
    """Singleton :class:`SetFormat` for a captured numeric free variable, or
    ``None`` if *val* has no exact finite rational form (non-numeric, or a
    non-finite ``Float``/``float``)."""
    match val:
        case Fraction():
            return SetFormat.from_value(val)
        case Float():
            return SetFormat.from_value(val.as_rational()) if val.is_finite() else None
        case RealFloat():
            return SetFormat.from_value(Fraction(val))
        case int() | float():
            try:
                return SetFormat.from_value(Fraction(val))
            except (ValueError, OverflowError):
                return None  # non-finite float
        case _:
            return None


@dataclass(frozen=True)
class TupleFormat:
    """Format for a tuple-valued expression."""
    elts: tuple['FormatBound', ...]


@dataclass(frozen=True)
class ListFormat:
    """Format for a list-valued expression (homogeneous element format)."""
    elt: 'FormatBound'


@dataclass(frozen=True)
class FunctionFormat:
    """Format-level signature of a function instantiation — mirrors
    :class:`fpy2.types.FunctionType`.

    Captures the function's interface: the active rounding context
    it runs under, a per-parameter format bound, and the format
    bound for its return value.

    Two roles:

    - **Input** to :meth:`FormatInfer.analyze` — the caller pins
      ``ctx`` and ``arg_fmts`` to instantiate the function at a
      particular signature.  The ``ret_fmt`` slot is treated as
      expected-but-unverified and is overwritten by whatever the
      body analysis derives.
    - **Output** recorded on :attr:`FormatAnalysis.fn_fmt` — all
      three fields populated, describing the function as analyzed.
    """

    ctx: 'Context | None'
    """Incoming rounding context (mirrors :attr:`FunctionType.ctx`).
    Pins the function's outermost scope when :class:`ContextUse`
    produced a symbolic context variable there.  ``None`` means no
    substitution — symbolic scopes stay symbolic."""

    arg_fmts: tuple['FormatBound', ...]
    """Per-parameter format bound, in source order (mirrors
    :attr:`FunctionType.arg_types`).  One entry per
    :class:`Argument`.  These bounds become the initial format for
    each parameter SSA def at the start of the analysis; bypasses
    the declared parameter types."""

    ret_fmt: 'FormatBound'
    """Format bound for the function's return value (mirrors
    :attr:`FunctionType.return_type`)."""


ScalarFormatBound: TypeAlias = None | SetFormat | Format
AbstractableFormatBound: TypeAlias = SetFormat | AbstractableFormat
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


_INTEGER_FORMAT: Format = INTEGER.format()
"""
Cached format for the :data:`INTEGER` context — used as the result
format of integer-producing operations (``Len``, ``Dim``, ``Size``,
``Range1``/``Range2``/``Range3``, the integer projection of
``Enumerate``).  These ops always produce integer values regardless of
the active rounding context, so reporting the active context's format
would be unnecessarily loose.
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


def _instantiate_real_at_ctx(ty: Type, ctx: Context) -> Type:
    """Replace every free :class:`Real` slot in *ty* with
    ``RealType(ctx)``.  Aggregate shapes (tuple, list) are preserved;
    non-numeric leaves (bool, context) pass through unchanged.  Used
    to derive a concrete parameter format from a declared type when
    a callee is analyzed under a caller's active context, so the
    callee body sees concrete inputs instead of ``REAL_FORMAT``
    everywhere.

    A :class:`RealType` with an *already-concrete* ``ctx`` keeps its
    own context — explicit annotations win over caller defaults.
    Unknown leaf types pass through unchanged (the caller absorbs the
    fallback)."""
    match ty:
        case RealType():
            if isinstance(ty.ctx, Context):
                return ty
            return RealType(ctx)
        case VarType():
            return RealType(ctx)
        case TupleType():
            return TupleType(*[_instantiate_real_at_ctx(elt, ctx) for elt in ty.elts])
        case ListType():
            return ListType(_instantiate_real_at_ctx(ty.elt, ctx))
        case _:
            return ty




def _bound_of_type(ty: Type) -> FormatBound:
    """
    Derives the most precise :data:`FormatBound` we can from a static
    type alone — used to derive initial format bounds for argument
    bindings, free variables, and other places where no
    expression-level format is yet available.

    For scalar real values, the result depends on whether the type
    carries a concrete rounding context (``RealType.ctx``):

    - **No context** (the default for un-monomorphized programs): the
      format is unknown, so we return the scalar top ``REAL_FORMAT``.
    - **Concrete context** (typical after a monomorphizing pass): the
      format is pinned to ``ctx.format()`` directly.

    For tuples and lists this is applied recursively to the element
    types, so a monomorphized ``list[Real[FP32]]`` becomes
    ``ListFormat(IEEEFormat(es=8, nbits=32))``.  For non-numeric types
    (bool, context, function, type variable) the result is ``None``.

    Note: the function is no longer always the *top* of the lattice —
    when a concrete ctx is present the result is a pinned, possibly
    non-top format.  The name reflects that it derives a bound *from a
    type* (parallel to :meth:`_bound_of_def`, which retrieves a bound
    by definition site).
    """
    match ty:
        case RealType():
            # If the type carries a concrete rounding context, the format
            # is pinned by that context — typically the case after a
            # monomorphization pass.  Otherwise (symbolic or absent ctx)
            # the format is unknown and we report the scalar top.
            if isinstance(ty.ctx, Context):
                return ty.ctx.format()
            return REAL_FORMAT
        case TupleType():
            return TupleFormat(tuple(_bound_of_type(t) for t in ty.elts))
        case ListType():
            return ListFormat(_bound_of_type(ty.elt))
        case BoolType() | ContextType() | FunctionType() | VarType():
            return None
        case _:
            raise RuntimeError(f'unreachable: unknown type {ty!r}')


def _setformat_to_abstract(s: SetFormat) -> AbstractFormat | None:
    """
    Returns a tight :class:`AbstractFormat` containing every value in *s*,
    or ``None`` when any value is non-dyadic (and therefore can't be
    pinned by an :class:`AbstractFormat`'s integer ``prec``/``exp``).

    Used by the exact-arithmetic cases in :meth:`_visit_unaryop` and
    :meth:`_visit_binaryop` to bridge a :class:`SetFormat` operand into
    the :class:`AbstractFormat` arithmetic path so that e.g.
    ``SetFormat({0}) + EFloatFormat`` doesn't bail to ``REAL_FORMAT``
    under exact-arithmetic.
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


def _to_abstract(f: AbstractableFormatBound | SetFormat) -> AbstractFormat | None:
    """
    Lifts an abstractable :class:`FormatBound` to an
    :class:`AbstractFormat`.  Returns ``None`` for variants that the
    abstract arithmetic doesn't accept (``None``, ``TupleFormat``,
    ``ListFormat``, non-dyadic ``SetFormat``).
    """
    if isinstance(f, AbstractableFormat):
        return AbstractFormat.from_format(f)
    else:
        return _setformat_to_abstract(f)

_ZERO_SET: 'SetFormat' = SetFormat.from_value(Fraction(0))


def _is_zero_set(f: 'FormatBound') -> bool:
    """``f`` is the precise singleton ``{0}``."""
    return isinstance(f, SetFormat) and f.values == _ZERO_SET.values


def exact_binop(
    lhs: 'FormatBound',
    rhs: 'FormatBound',
    op: Callable[[Any, Any], Any],
) -> 'SetFormat | AbstractFormat | None':
    """Compute the exact unrounded result of an arithmetic binary op.

    *op* is applied either pointwise to a pair of :class:`SetFormat`
    operands (over their :class:`Fraction` values) or directly to
    operands lifted to :class:`AbstractFormat`.  Both paths produce
    a sound, non-rounded result.  Returns ``None`` when either
    operand isn't abstractable (e.g. ``None``, ``TupleFormat``,
    non-dyadic ``SetFormat``).

    When one operand is the precise zero singleton ``SetFormat({0})``,
    the abstract path would produce a zero-bounded ``AbstractFormat``
    that loses the singleton precision and can drive subsequent
    ``prec`` computations to ``0``.  Short-circuit through the
    algebraic identities (``0 * x = 0``, ``0 + x = x``, ``x - 0 = x``)
    so the precise format survives.

    Used by :meth:`_FormatInferInstance._visit_binaryop` to compute
    the candidate ``F`` that :meth:`_bound_if_fits` then checks
    against the active scope.  Also part of the public API
    consumed by downstream transforms (e.g.,
    :class:`fpy2.transform.RoundElim`) that need the unrounded
    image of a rounded operation.
    """
    if not (
        isinstance(lhs, AbstractableFormatBound)
        and isinstance(rhs, AbstractableFormatBound)
    ):
        return None
    if isinstance(lhs, SetFormat) and isinstance(rhs, SetFormat):
        return SetFormat(frozenset(
            op(va, vb) for va in lhs.values for vb in rhs.values
        ))
    lhs_zero = _is_zero_set(lhs)
    rhs_zero = _is_zero_set(rhs)
    if op is operator.mul and (lhs_zero or rhs_zero):
        return _ZERO_SET
    if op is operator.add:
        if lhs_zero:
            return rhs if isinstance(rhs, SetFormat) else _to_abstract(rhs)
        if rhs_zero:
            return lhs if isinstance(lhs, SetFormat) else _to_abstract(lhs)
    if op is operator.sub and rhs_zero:
        return lhs if isinstance(lhs, SetFormat) else _to_abstract(lhs)
    af_a = _to_abstract(lhs)
    af_b = _to_abstract(rhs)
    if af_a is None or af_b is None:
        return None
    return op(af_a, af_b)


def exact_unop(
    arg: 'FormatBound',
    op: Callable[[Any], Any],
) -> 'SetFormat | AbstractFormat | None':
    """Unary analogue of :func:`exact_binop` — apply *op* either
    pointwise to a :class:`SetFormat`'s values or directly to a
    lifted :class:`AbstractFormat`.  Returns ``None`` when *arg* is
    not abstractable.
    """
    if not isinstance(arg, AbstractableFormatBound):
        return None
    if isinstance(arg, SetFormat):
        return SetFormat(frozenset(op(v) for v in arg.values))
    af = _to_abstract(arg)
    if af is None:
        return None
    return op(af)


def round_is_identity(
    unrounded: 'SetFormat | AbstractFormat | None',
    ctx: Context | None,
) -> bool:
    """Decide whether ``round_{ctx}(unrounded) == unrounded`` — i.e.,
    every value representable in *unrounded* is also representable
    in *ctx*'s format, so the round leaves the value-set unchanged.

    *unrounded* is the unrounded value-set, typically the result of
    :func:`exact_binop` / :func:`exact_unop` (or, for explicit
    ``Round``/``Cast`` nodes, the argument's inferred
    bound).  *ctx* is the concrete rounding context whose
    round-identity behavior we care about.

    Returns ``True`` when *ctx* is :data:`REAL` (the trivial identity
    context) and *unrounded* is non-``None``.  Returns ``False`` when
    *ctx* is ``None`` (symbolic / unresolved scope — we can't claim
    identity without a concrete target) or *unrounded* is ``None``.

    Used by :meth:`_FormatInferInstance._bound_if_fits` internally and
    exposed publicly for transforms that key off the same identity
    notion (e.g., :class:`fpy2.transform.RoundElim`).
    """
    if unrounded is None or ctx is None:
        return False
    if ctx is REAL:
        return True
    ctx_fmt = ctx.format()
    if isinstance(unrounded, SetFormat):
        return _all_representable_in(unrounded.values, ctx_fmt)
    if not isinstance(ctx_fmt, AbstractableFormat):
        return False
    return unrounded <= AbstractFormat.from_format(ctx_fmt)


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
            # Widening: SetFormat unions form a lattice of unbounded
            # height (each loop iteration can add a fresh value), so a
            # naive phi-join over them never converges.  When the
            # caller signals it's running a saturated loop fixpoint,
            # collapse to ``REAL_FORMAT`` so the next iteration falls
            # back to the abstract path and reaches a fixed point.
            if widen:
                return REAL_FORMAT
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

    An ``IndexedAssign`` ``xs[i1]…[iN] = val`` produces a new list in which
    ``xs[i1]…[iN]`` is replaced by *val*.  The result's format is the join
    of the original format with *insert_fmt* at the leaf level after
    peeling *depth* layers of :class:`ListFormat` (one per index).

    *widen* propagates to the leaf-level :func:`_join_bounds` call so that
    indexed assignments inside a saturated loop iteration also widen.
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
# Pre-analysis cache.
#
# The three structural analyses :class:`DefineUse`, :class:`TypeInfer`,
# :class:`ContextUse`, :class:`ArraySizeInfer` are independent of the
# active rounding context — they describe the function's structure, not
# its numerical formats.  When the same :class:`FuncDef` is reached from
# multiple call sites (potentially under different contexts), we run
# those analyses once and share the result; only :class:`FormatInfer`
# itself re-runs per instantiation.


@dataclass(frozen=True)
class PreAnalyses:
    """Bundle of the four structural pre-analyses a
    :class:`FormatAnalysis` depends on.  Identical for every
    instantiation of the same :class:`FuncDef`."""
    def_use: DefineUseAnalysis
    type_info: TypeAnalysis
    ctx_use: ContextUseAnalysis
    array_size: ArraySizeAnalysis


class PreAnalysisCache:
    """Memoizes :class:`PreAnalyses` per :class:`FuncDef`.

    Shared across a single top-level :meth:`FormatInfer.analyze` call
    and all the recursive sub-analyses that walk into the call graph.
    Each ``FuncDef`` is analyzed structurally exactly once, no matter
    how many call sites reach it or under how many contexts.
    """

    def __init__(self):
        self._table: dict[int, PreAnalyses] = {}

    def get(
        self,
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
        array_size: ArraySizeAnalysis | None = None,
    ) -> PreAnalyses:
        """Return the pre-analyses for *func*, computing them on the
        first request.  Pre-supplied analyses (via the keyword args)
        populate the cache for *func* if it hasn't been computed
        yet — used by the public :meth:`FormatInfer.analyze` to
        thread externally-computed analyses in.  Subsequent lookups
        ignore the keyword args."""
        key = id(func)
        cached = self._table.get(key)
        if cached is not None:
            return cached
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        if ctx_use is None:
            ctx_use = ContextUse.analyze(func, def_use=def_use)
        if array_size is None:
            array_size = ArraySizeInfer.analyze(func, type_info=type_info)
        pre = PreAnalyses(def_use, type_info, ctx_use, array_size)
        self._table[key] = pre
        return pre


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

    func: FuncDef
    """The function whose body was analyzed."""

    fn_fmt: FunctionFormat
    """
    The format-level signature of this instantiation (mirror of the
    function's :class:`FunctionType`).

    For top-level analyses, ``fn_fmt.ctx`` and
    ``fn_fmt.arg_fmts`` reflect whatever the caller
    pinned (or sensible defaults derived from the declared types).
    For sub-analyses recorded in :attr:`by_call`, they reflect the
    active rounding context at the call site and the formats of
    the call-site argument expressions, respectively.  In every
    case ``fn_fmt.ret_fmt`` is the joined bound
    across the function's :class:`ReturnStmt` expressions, as
    derived by this analysis.
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

    by_call: dict[Call, 'FormatAnalysis']
    """
    Per-call-site sub-analyses — the :class:`FormatAnalysis` graph
    mirrors the call graph.

    For each :class:`Call` whose ``fn`` is a known :class:`Function`,
    :class:`FormatInfer` recurses into the callee with the call site's
    active rounding context threaded through as the callee's
    ``outer_ctx``.  Structural analyses (def-use, types, context
    scopes, array sizes) are shared across instantiations via a
    :class:`PreAnalysisCache`; only :class:`FormatInfer` itself re-runs
    per call site.

    Calls into foreign / unknown functions are absent from this map —
    in those cases the call's format falls back to a type-based bound.
    Recursive call graphs (direct or mutual) are out of scope for this
    analysis; the caller is expected to rule them out with a separate
    pre-check.
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
    by_call: dict[Call, FormatAnalysis]

    _pre_cache: PreAnalysisCache
    _fn_fmt: FunctionFormat | None
    _return_fmt: FormatBound | None
    _loop_iter_limit: int
    _widen: bool

    def __init__(
        self,
        func: FuncDef,
        pre: PreAnalyses,
        pre_cache: PreAnalysisCache,
        fn_fmt: FunctionFormat | None,
        loop_iter_limit: int,
    ):
        self.func = func
        self.type_info = pre.type_info
        self.ctx_use = pre.ctx_use
        self.array_size = pre.array_size
        self.by_def = {}
        self.by_expr = {}
        self.by_call = {}
        self._pre_cache = pre_cache
        # The instantiation signature the caller pinned.  ``None``
        # means "no substitution" — the function is analyzed
        # standalone, with declared parameter types and any symbolic
        # function-level scope degrading to ``REAL_FORMAT`` at
        # affected op sites.  Read by :meth:`_visit_function` to
        # initialize parameter formats and by
        # :meth:`_resolve_active_ctx` for scope substitution.
        self._fn_fmt = fn_fmt
        # Running join of every :class:`ReturnStmt` format the body
        # walk has seen so far.  :meth:`_visit_return` folds each
        # new return into this slot; :meth:`_visit_function` reads
        # it after the body walk.  ``None`` doubles as both the
        # "no returns visited yet" sentinel and the legitimate
        # bound for void/bool returns — they coincide operationally
        # since :class:`TypeInfer` already guarantees every return
        # has the same static type, so the analysis can't see a
        # mix of bool and non-bool returns in a single function.
        self._return_fmt = None
        self._loop_iter_limit = loop_iter_limit
        # When True, joins forced inside a saturated loop fixpoint widen
        # distinct scalar Formats to ``REAL_FORMAT`` instead of going through
        # an :class:`AbstractFormat` union (which has infinite ascending
        # chains under arithmetic).  Saved/restored around each loop so that
        # nested loops manage their own state.
        self._widen: bool = False

    @property
    def _outer_ctx(self) -> Context | None:
        """Convenience: incoming rounding context, or ``None`` if no
        ``fn_fmt`` was supplied."""
        return self._fn_fmt.ctx if self._fn_fmt is not None else None

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

    def _resolve_active_ctx(
        self, e: ContextUseSite
    ) -> Context | None:
        """Concrete rounding context active at *e*, or ``None`` if
        unresolvable.

        Looks up *e*'s active scope and returns the scope's context
        when it's a concrete :class:`Context`.  Symbolic scopes are
        substituted with :attr:`_outer_ctx` when the caller provided
        one — this is how the recursive call-site instantiation flows
        the caller's active context into the callee's outer scope."""
        scope = self.ctx_use.find_scope_from_use(e)
        if isinstance(scope.ctx, Context):
            return scope.ctx
        return self._outer_ctx

    def _scope_format(self, e: ContextUseSite) -> Format:
        """Returns the format of the rounding context scope for *e*.

        Symbolic scopes resolve through :attr:`_outer_ctx` when one
        was provided; otherwise the legacy ``REAL_FORMAT`` fallback
        applies."""
        resolved = self._resolve_active_ctx(e)
        if resolved is not None:
            return resolved.format()
        return REAL_FORMAT

    def _op_bound(self, e: ContextUseSite) -> FormatBound:
        """
        Format of a rounded operation's result.

        Real-valued operations are rounded to the active context's
        format.  Operations that return any other type (bool, list,
        context, …) are not numerical computations — their format
        is derived from the declared type.
        """
        ret_ty = self.type_info.by_expr[e]
        if isinstance(ret_ty, RealType):
            return self._scope_format(e)
        return _bound_of_type(ret_ty)

    def _is_real_scope(self, e: ContextUseSite) -> bool:
        """Returns True iff *e*'s active scope is the :data:`REAL` singleton.

        Distinct from ``_scope_format(e) == REAL_FORMAT`` because the latter
        also fires for symbolic (unresolved) context variables, where we
        cannot assume the rounding is the identity.  ``REAL`` is a unique
        singleton, so identity comparison is the right check (and avoids
        accidentally matching other ``RealContext`` instances that this
        analysis does not support).
        """
        return self.ctx_use.find_scope_from_use(e).ctx is REAL

    def _bound_if_fits(
        self,
        e: ContextUseSite,
        exact: SetFormat | AbstractFormat | None,
    ) -> FormatBound | None:
        """Return the format of ``round_C(exact)``, where ``C`` is *e*'s
        active scope and *exact* is the statically-known unrounded
        result.  Returns ``None`` only when the helper can't compute a
        usable bound (no *exact*, symbolic scope, non-abstractable
        scope format, or :class:`SetFormat` that doesn't fit — the
        last case becomes precise in Phase B).

        Two regimes:

        - ``exact ⊆ C``: every value is C-representable, so ``round_C``
          is the identity and the inferred bound *is* ``exact`` — strict
          improvement over the scope's format.  This is the case the
          helper handled exclusively before; the name reflects that
          legacy intent.

        - ``exact ⊄ C`` (some dimension exceeds the scope but others
          may be tighter): the image ``round_C(exact)`` is bounded by
          the intersection ``exact & C`` at the
          :class:`AbstractFormat` level (precision clipped down to
          ``C``'s, quantum coarsened to ``max(F.exp, C.exp)``, magnitude
          clipped to ``min(F.bound, C.bound)``).  The intersection is
          a sound over-approximation of the image and is at most as
          wide as ``C``.  Phase A widens to ``F & C`` here; Phase B
          will additionally compute the precise value-image for
          :class:`SetFormat` operands via :meth:`Context.round`.

        :data:`REAL` is the case where ``C.F`` trivially contains every
        value (intersection equals *exact*).  Symbolic scopes return
        ``None`` — we can't apply rounding to an unknown context.
        """
        if exact is None:
            return None
        resolved = self._resolve_active_ctx(e)
        if resolved is None:
            return None
        # Under loop-fixpoint widening, bail to the scope format via
        # :meth:`_op_bound`.  Both the REAL identity shortcut and the
        # intersection branch below grow the inferred format every
        # iteration as the unrounded chain grows, which prevents
        # convergence — same intent as the widen-to-REAL branches in
        # :func:`_join_bounds`.
        if self._widen:
            return None
        if resolved is REAL:
            return exact if isinstance(exact, SetFormat) else exact.format()
        # Identity-round fast path: when ``round_C(F) == F``, the
        # rounded image equals *exact* itself.  Both the SetFormat
        # fits-everywhere case and the AbstractFormat
        # ``F ⊆ scope_af`` case collapse into this single check —
        # single-sourced via :func:`round_is_identity` so external
        # consumers (e.g. ``RoundElim``) see the same notion.
        if round_is_identity(exact, resolved):
            return exact if isinstance(exact, SetFormat) else exact.format()
        scope_fmt = resolved.format()
        if isinstance(exact, SetFormat):
            # TODO: we can round each value in the set individually
            # to produce a precise image even when some values exceed the scope
            # It is unclear how this affects the overal algorithm.
            return None
        if not isinstance(scope_fmt, AbstractableFormat):
            return None
        scope_af = AbstractFormat.from_format(scope_fmt)
        if scope_af <= exact:
            return scope_fmt

        # Mixed-overlap branch.  Tighten prec/exp unconditionally —
        # both are sound under any rounding mode.  Tighten bounds
        # only when F's precision fits in C's.
        #
        # Soundness pitfall on bounds: ``round_C(F.pos_bound)`` can
        # land up to one ulp_C *above* F.pos_bound (round-up, or
        # round-to-nearest with a tie pointing away from zero) when
        # F.pos_bound isn't exactly C-representable.  A naive
        # ``min(F.pos_bound, C.pos_bound)`` would then under-claim
        # the image's bound and be unsound.
        #
        # The gate: when ``F.prec <= C.prec``, F.pos_bound has
        # precision ≤ F.prec ≤ C.prec and is therefore exactly
        # C-representable, so the intersection's bounds are sound.
        # When ``F.prec > C.prec`` we fall back to C's bounds.
        # ``int | float`` comparison works directly with the
        # ``float('inf')`` sentinel used for unbounded prec.
        prec = min(exact.prec, scope_af.prec)
        exp = max(exact.exp, scope_af.exp)
        if exact.prec > scope_af.prec:
            pos_bound = scope_af.pos_bound
            neg_bound = scope_af.neg_bound
        else:
            pos_bound = min(exact.pos_bound, scope_af.pos_bound)
            neg_bound = max(exact.neg_bound, scope_af.neg_bound)
        return AbstractFormat(prec, exp, pos_bound, neg_bound=neg_bound).format()

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
        return SetFormat.from_value(e.as_rational())

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> FormatBound:
        return SetFormat.from_value(e.as_rational())

    def _visit_integer(self, e: Integer, ctx: None) -> FormatBound:
        return SetFormat.from_value(e.as_rational())

    def _visit_rational(self, e: Rational, ctx: None) -> FormatBound:
        return SetFormat.from_value(e.as_rational())

    def _visit_digits(self, e: Digits, ctx: None) -> FormatBound:
        return SetFormat.from_value(e.as_rational())

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
        match e:
            case Len() | Dim():
                # len(...) / dim(...) -> result format is INTEGER regardless
                # of the active rounding context.
                return _INTEGER_FORMAT
            case Range1():
                # range(...) -> result is a list of integers.
                return ListFormat(_INTEGER_FORMAT)
            case Enumerate():
                # enumerate(xs) -> list of (int, elt) tuples; the int projection
                # is INTEGER, the original element format is preserved.
                assert isinstance(arg_fmt, ListFormat), \
                    f'expected ListFormat for argument of Enumerate, got {arg_fmt!r}'
                return ListFormat(TupleFormat((_INTEGER_FORMAT, arg_fmt.elt)))
            case Fst():
                # tuple head — the first element's format.
                assert isinstance(arg_fmt, TupleFormat), \
                    f'expected TupleFormat for argument of fst, got {arg_fmt!r}'
                return arg_fmt.elts[0]
            case Snd():
                # tuple tail — the second element's format for a pair, else
                # the format of the remaining elements.
                assert isinstance(arg_fmt, TupleFormat), \
                    f'expected TupleFormat for argument of snd, got {arg_fmt!r}'
                rest = arg_fmt.elts[1:]
                return rest[0] if len(rest) == 1 else TupleFormat(rest)
            case Sum():
                assert isinstance(arg_fmt, ListFormat), f'expected ListFormat for argument of Sum, got {arg_fmt!r}'
                return self._sum_bound(e, arg_fmt)
            case AMin() | AMax():
                # ``min(xs)`` / ``max(xs)`` selects one element of ``xs``
                # exactly — no rounding.  Result format is the list's
                # element format (which already joins all elements).
                assert isinstance(arg_fmt, ListFormat), \
                    f'expected ListFormat for argument of {type(e).__name__}, got {arg_fmt!r}'
                return arg_fmt.elt
            case Round() | Cast():
                # Identity-when-fits.  ``Round`` rounds the argument to
                # the active scope; ``Cast`` rounds and asserts the
                # result is exact.  When the argument's
                # inferred format is contained in the scope's format,
                # every one of these operations is the identity at the
                # value level — so the result's tight format equals
                # the argument's, not the scope's.  Falls back to
                # :meth:`_op_bound` (the scope format) otherwise.
                if isinstance(arg_fmt, SetFormat):
                    fitted = self._bound_if_fits(e, arg_fmt)
                elif isinstance(arg_fmt, AbstractableFormat):
                    fitted = self._bound_if_fits(
                        e, AbstractFormat.from_format(arg_fmt),
                    )
                else:
                    fitted = None
                if fitted is not None:
                    return fitted
            case Abs():
                # abs: precision-preserving.  Use the unrounded result
                # whenever the active scope's format contains it; see
                # :meth:`_bound_if_fits`.
                fitted = self._bound_if_fits(e, exact_unop(arg_fmt, abs))
                if fitted is not None:
                    return fitted
            case Neg():
                # negation: precision-preserving (same logic as Abs).
                fitted = self._bound_if_fits(
                    e, exact_unop(arg_fmt, operator.neg),
                )
                if fitted is not None:
                    return fitted

        return self._op_bound(e)


    def _sum_bound(self, e: Sum, arg_fmt: ListFormat) -> FormatBound:
        """
        Format of ``Sum(xs)`` — reduce-by-pairwise-addition with rounding
        under the active context at each step.

        Strategy: simulate ``n - 1`` exact pairwise additions through
        :class:`AbstractFormat`.  The simulated accumulator's
        representable set is a superset of every intermediate partial
        sum's value set (this matters for signed reductions, where
        intermediate magnitudes can exceed the final result).  If the
        simulated accumulator fits under the active scope's format,
        every per-step ``round_C`` is the identity, so the precise
        format is sound to report — regardless of whether the scope
        is :data:`REAL` or a concrete rounding context.  Otherwise
        the result widens to the scope's format via :meth:`_op_bound`.

        Requires the list size to be statically known via
        :class:`ArraySizeAnalysis`; otherwise falls back to
        :meth:`_op_bound`.

        The branches below are ordered so that the trivial cases
        (``n == 0`` and ``n == 1``) are decided before any
        abstractability check on the element format.
        """
        n = self._known_iter_count(e.arg)
        if n is None:
            return self._op_bound(e)

        # ``sum([])`` is conventionally 0 — fits trivially in any scope.
        if n == 0:
            return SetFormat.from_value(Fraction(0))

        elt_fmt = arg_fmt.elt
        # Single-element reduction: ``sum([x])`` evaluates to
        # ``round_C(x)``.  Tighten to ``elt_fmt`` when it fits under
        # the scope; otherwise the round may not be the identity.
        if n == 1:
            if isinstance(elt_fmt, SetFormat):
                fitted = self._bound_if_fits(e, elt_fmt)
            elif isinstance(elt_fmt, AbstractableFormat):
                fitted = self._bound_if_fits(e, AbstractFormat.from_format(elt_fmt))
            else:
                fitted = None
            return fitted if fitted is not None else self._op_bound(e)

        # ``n >= 2``: simulate ``n - 1`` pairwise additions through
        # AbstractFormat, then check the final accumulator against the
        # scope.  Requires the element to be abstractable.
        if not isinstance(elt_fmt, AbstractableFormatBound):
            return self._op_bound(e)
        af_elt = _to_abstract(elt_fmt)
        if af_elt is None:
            return self._op_bound(e)
        af_acc = af_elt
        for _ in range(n - 1):
            af_acc = af_acc + af_elt
        fitted = self._bound_if_fits(e, af_acc)
        return fitted if fitted is not None else self._op_bound(e)

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> FormatBound:
        lhs = self._visit_expr(e.first, ctx)
        rhs = self._visit_expr(e.second, ctx)
        match e:
            case Size():
                # size(...) -> result format is INTEGER regardless of scope.
                return _INTEGER_FORMAT
            case Range2():
                # range(...) -> result is a list of integers.
                return ListFormat(_INTEGER_FORMAT)
            case Add():
                # Exact addition.  Use the unrounded result whenever
                # the active scope's format contains it; see
                # :meth:`_bound_if_fits`.  Subsumes the legacy REAL-only
                # fast path (REAL_FORMAT contains everything).
                fitted = self._bound_if_fits(
                    e, exact_binop(lhs, rhs, operator.add),
                )
                if fitted is not None:
                    return fitted
            case Sub():
                fitted = self._bound_if_fits(
                    e, exact_binop(lhs, rhs, operator.sub),
                )
                if fitted is not None:
                    return fitted
            case Mul():
                fitted = self._bound_if_fits(
                    e, exact_binop(lhs, rhs, operator.mul),
                )
                if fitted is not None:
                    return fitted

        return self._op_bound(e)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> FormatBound:
        self._visit_expr(e.first, ctx)
        self._visit_expr(e.second, ctx)
        self._visit_expr(e.third, ctx)
        if isinstance(e, Range3):
            return ListFormat(_INTEGER_FORMAT)
        return self._op_bound(e)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> FormatBound:
        arg_fmts = [self._visit_expr(arg, ctx) for arg in e.args]
        match e:
            case Min() | Max():
                # Selection, not arithmetic — ``min(a, b, …)`` is
                # ``a < b ? a : b`` chained.  The result is exactly one
                # operand, so the format joins operand formats (no scope
                # widening; mirrors :meth:`_visit_if_expr`).
                return reduce(self._join, arg_fmts)
            case Zip():
                # ``zip(xs1, ..., xsN)`` yields a list of N-tuples whose
                # element formats are taken directly from the input lists'
                # element formats — *not* from the active rounding context.
                # This matches the runtime semantics: zip is a structural
                # rearrangement and never rounds.
                elts = [
                    fmt.elt if isinstance(fmt, ListFormat) else REAL_FORMAT
                    for fmt in arg_fmts
                ]
                return ListFormat(TupleFormat(tuple(elts)))
            case _:
                return self._op_bound(e)

    # Function calls: conservatively the top format of the return type
    def _visit_call(self, e: Call, ctx: None) -> FormatBound:
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)

        # Template-style recursion: when the callee is a known FPy
        # :class:`Function`, instantiate it at the call site's active
        # rounding context and recursively analyze its body.  The
        # sub-result is recorded so downstream consumers can walk the
        # full call graph; the call's own format becomes the joined
        # bound across the callee's return expressions, which is
        # strictly tighter than the type-based bound used as a
        # fallback below.
        sub = self._analyze_callee(e)
        if sub is not None:
            self.by_call[e] = sub
            return sub.fn_fmt.ret_fmt
        return _bound_of_type(self.type_info.by_expr[e])

    def _analyze_callee(self, e: Call) -> 'FormatAnalysis | None':
        """Recursive sub-analysis on the callee at this site.

        The callee's structural pre-analyses (def-use, types, context
        scopes, array sizes) are shared via the :class:`PreAnalysisCache`
        — analyzed once per :class:`FuncDef`.  Only :class:`FormatInfer`
        re-runs per instantiation, with the call site's active
        rounding context threaded through as ``outer_ctx`` so symbolic
        scopes inside the callee resolve correctly and parameter
        formats get pinned.

        Recursion assumes no cycles in the call graph — enforced by
        the :class:`CallGraph` acyclicity check at the public
        :meth:`FormatInfer.analyze` entry.  Returns ``None`` when the
        callee isn't a concrete :class:`Function` — the caller falls
        back to a static type bound.
        """
        fn = e.fn
        if not isinstance(fn, Function):
            return None
        callee_outer = self._resolve_active_ctx(e)
        # Build the callee's instantiation signature from this call
        # site: the active rounding context, plus one format per
        # parameter taken straight from the call-site argument
        # expression.  This is what gives the callee the right
        # signature: a callee invoked with ``FP64`` arguments under
        # an ``FP32`` outer scope gets ``a: FP64, b: FP64, → FP32``
        # — and any FP32-rounded operation on those doubles will
        # trip the op-table's lossy-implicit-cast guard at emission
        # time.  The ``ret_fmt`` slot is a placeholder; the
        # sub-analysis will compute the real value from the body.
        arg_fmts = tuple(self.by_expr.get(arg) for arg in e.args)
        callee_signature = FunctionFormat(
            ctx=callee_outer,
            arg_fmts=arg_fmts,
            ret_fmt=REAL_FORMAT,
        )
        pre = self._pre_cache.get(fn.ast)
        return _FormatInferInstance(
            fn.ast,
            pre=pre,
            pre_cache=self._pre_cache,
            fn_fmt=callee_signature,
            loop_iter_limit=self._loop_iter_limit,
        ).analyze()

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
            joined = _bound_of_type(list_ty.elt)
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
        # inserted value's format at depth ``len(indices)``; see
        # :func:`_list_set_widen` for the recursive widening rule.
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

        # Widening is only a device to converge the phi fixpoint.  If it had
        # to engage (the loop hit the iteration limit), the body's recorded
        # ``by_expr`` came from a *widened* pass, which collapses every
        # rounding/exact chain — including loop-invariant ones that don't
        # depend on a widened phi — to the scope format.  Re-run the body
        # once with widening off, using the now-fixed phi bounds: a single
        # pass terminates (the phis are constants) and records precise
        # formats for everything that isn't genuinely tied to a widened phi.
        # (Skip when an outer loop is still widening — it wants the coarse
        # bounds.)
        if not saved_widen and iter_count >= self._loop_iter_limit:
            run_body()

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
        (``+``/``-``/``*`` under :data:`REAL`) lattice, which has
        infinite ascending chains.

        Iter-by-iter visit produces a precise (if potentially wide)
        bound after exactly ``n`` joins; the result is sound and
        strictly more precise than the fixpoint+widening fall-back when
        ``n`` is small.
        """
        phis = list(phis)
        for phi in phis:
            self._set_def_bound(phi, self._bound_of_def(self.def_use.defs[phi.lhs]))
        if n <= 0:
            # The body never executes at runtime, so the phi stays at the
            # pre-loop value.  But the backend still emits the body, so its
            # inner definitions need format bounds: visit the body once to
            # populate them, without folding the result into the phi.
            run_body()
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
        # Fold each return's format into the running join here, as
        # the body walk encounters it — saves a second AST
        # traversal at function-end.  The first ``ReturnStmt`` sets
        # ``_return_fmt`` directly; subsequent ones join with
        # whatever's been seen so far.
        fmt = self._visit_expr(stmt.expr, ctx)
        if self._return_fmt is None:
            self._return_fmt = fmt
        else:
            self._return_fmt = self._join(self._return_fmt, fmt)

    def _visit_pass(self, stmt: PassStmt, ctx: None):
        pass

    def _visit_block(self, block: StmtBlock, ctx: None):
        for stmt in block.stmts:
            self._visit_statement(stmt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None) -> FunctionFormat:
        """Initialize parameter / free-variable formats, visit the
        body, then return the :class:`FunctionFormat` describing this
        instantiation.

        Initial parameter formats come from three sources, in
        priority order:

        1. ``self._fn_fmt.arg_fmts``: caller-supplied per-parameter
           formats — used when we're analyzing a callee at a
           specific call site, so the body sees the *caller's actual
           argument formats* rather than the declared parameter
           types.
        2. ``self._fn_fmt.ctx``: when no caller-supplied format is
           available, pin free ``Real`` slots in the declared type
           to the outer context.  Useful for standalone analyses
           that want to pretend the function is being run under a
           particular ctx without faking up every operand format.
        3. Fallback: declared type, untouched — yields
           ``REAL_FORMAT`` for free ``Real`` slots.

        After the body is visited, the returned :class:`FunctionFormat`
        echoes the caller's ``ctx``, reads ``arg_fmts`` back from
        the parameter defs (so it reflects what the body actually
        saw), and exposes ``ret_fmt`` as the running join of every
        :class:`ReturnStmt` expression's bound that
        :meth:`_visit_return` accumulated during the walk.
        """
        fn_fmt = self._fn_fmt
        outer_ctx = fn_fmt.ctx if fn_fmt is not None else None
        params = fn_fmt.arg_fmts if fn_fmt is not None else None

        def param_from_type(ty: Type) -> FormatBound:
            if outer_ctx is not None:
                ty = _instantiate_real_at_ctx(ty, outer_ctx)
            return _bound_of_type(ty)

        # Parameter defs
        for i, arg in enumerate(func.args):
            if not isinstance(arg.name, NamedId):
                continue
            d = self.def_use.find_def_from_site(arg.name, arg)
            if params is not None and i < len(params):
                self._set_def_bound(d, params[i])
            else:
                self._set_def_bound(d, param_from_type(self.type_info.by_def[d]))
        # Free-variable defs (captured from an outer scope).  A finite numeric
        # capture pins the def to the singleton set {value}; anything else
        # falls back to the type-derived bound.
        for v in func.free_vars:
            d = self.def_use.find_def_from_site(v, func)
            fmt = _free_var_set_format(self.func.env[str(v)])
            if fmt is None:
                fmt = param_from_type(self.type_info.by_def[d])
            self._set_def_bound(d, fmt)

        # Walk the body — populates ``by_def`` / ``by_expr`` / ``by_call``.
        self._visit_block(func.body, ctx)
        assert isinstance(self._return_fmt, FormatBound), 'expected at least one ReturnStmt in function body'

        # Build this instantiation's signature.  ``arg_fmts`` is the
        # initial format each parameter held entering the body
        # (caller-supplied or declared-type fallback); ``ret_fmt``
        # is the running join of every ``ReturnStmt`` expression's
        # format accumulated during the walk.
        arg_fmts: list[FormatBound] = []
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                arg_fmts.append(self.by_def.get(d))
            else:
                arg_fmts.append(None)
        return FunctionFormat(
            ctx=outer_ctx,
            arg_fmts=tuple(arg_fmts),
            ret_fmt=self._return_fmt,
        )


    def analyze(self) -> FormatAnalysis:
        fn_fmt = self._visit_function(self.func, None)
        return FormatAnalysis(
            func=self.func,
            fn_fmt=fn_fmt,
            type_info=self.type_info,
            ctx_use=self.ctx_use,
            by_def=self.by_def,
            by_expr=self.by_expr,
            by_call=self.by_call,
        )


#####################################################################
# Public API

class FormatInfer:
    """
    Format inference for FPy functions.

    This analysis bounds the number format for each expression and variable
    definition in an FPy program.  It uses two pre-analyses
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
      (``+``/``-``/``*`` under :data:`REAL`) is applied to a phi'd
      value, so the fixpoint runs at most ``loop_iter_limit`` iterations
      before switching joins to widen-mode (distinct scalar Formats fall
      back to ``REAL_FORMAT``) to force convergence.

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
        pre_cache: PreAnalysisCache | None = None,
        fn_fmt: FunctionFormat | None = None,
        loop_iter_limit: int = DEFAULT_LOOP_ITER_LIMIT,
    ) -> FormatAnalysis:
        """
        Performs format analysis on an FPy function.

        Walks into every :class:`Call` whose ``fn`` is a known
        :class:`Function` and recurses to produce a graph of
        :class:`FormatAnalysis` mirroring the call graph
        (:attr:`FormatAnalysis.by_call`).  Structural pre-analyses
        (def-use, types, context scopes, array sizes) are shared
        across all sub-analyses via a :class:`PreAnalysisCache` —
        each :class:`FuncDef` is structurally analyzed at most once
        for the lifetime of a top-level call to :meth:`analyze`.
        :class:`FormatInfer` itself re-runs per instantiation so each
        callee sees the caller's active rounding context.

        The call graph must be acyclic; recursion (direct or mutual)
        is not handled by the per-call-site sub-analysis.  A
        :class:`CallGraph` acyclicity check runs at this entry and
        raises :class:`CallGraphError` if a cycle is reachable.

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
                :data:`REAL`) which has infinite ascending chains.
            pre_cache:
                Optional :class:`PreAnalysisCache` shared with
                sub-analyses.  Useful when a caller wants to
                amortize structural analysis across multiple
                independent :meth:`analyze` invocations on a
                shared corpus.  When ``None``, a fresh cache is
                allocated for this call (and its recursive
                descents).
            fn_fmt:
                Format-level signature to instantiate the function
                at.  Supplies the incoming rounding context (used
                to pin the function's outermost scope when
                symbolic) and a per-parameter initial format (used
                to bypass the declared parameter types — typically
                derived from the caller's call-site argument
                formats).  The ``ret_fmt`` slot is treated as
                expected and overwritten in the result.  ``None``
                falls back to the legacy behaviour: symbolic outer
                scopes degrade to ``REAL_FORMAT`` and parameter
                formats come from the declared types.
            loop_iter_limit:
                Number of loop body+join iterations to run before
                forcing joins of distinct scalar Formats to widen to
                ``REAL_FORMAT``.  Only applies to loops without a known
                iteration count (``while`` loops, and ``for`` loops
                over symbolic-size iterables).

        Returns:
            A :class:`FormatAnalysis` whose ``by_def``, ``by_expr``,
            and ``by_call`` describe the formats inferred at every
            definition site, expression, and reachable sub-call.

        Raises:
            TypeError: If *func* is not a :class:`FuncDef`.
            CallGraphError: If the call graph reachable from *func*
                contains a cycle (FPy forbids recursion).
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {type(func)}")

        # Guard against recursion: the per-call-site sub-analysis in
        # `_analyze_callee` would otherwise diverge on a cyclic call
        # graph.  `CallGraph` raises on any cycle reachable from `func`.
        # The recursive descents go through `_FormatInferInstance`
        # directly, not this public entry, so the check runs exactly
        # once per top-level analysis — before any instance is built.
        CallGraph.analyze(func)

        if pre_cache is None:
            pre_cache = PreAnalysisCache()

        pre = pre_cache.get(
            func,
            def_use=def_use,
            type_info=type_info,
            ctx_use=ctx_use,
            array_size=array_size,
        )

        inst = _FormatInferInstance(
            func,
            pre=pre,
            pre_cache=pre_cache,
            fn_fmt=fn_fmt,
            loop_iter_limit=loop_iter_limit,
        )
        return inst.analyze()
