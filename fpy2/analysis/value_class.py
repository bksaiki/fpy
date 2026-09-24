"""
Path-sensitive value-class analysis.

One question per expression: can this value be a NaN, an infinity of either
sign, a zero, or a finite non-zero?  The five atoms form a finite lattice —
union is the join, intersection the meet, so no widening is needed — and it is
*refined* at every ``if`` statement that tests a value's class.  Not at an
``IfExpr``, as format inference does not: a backend may evaluate both arms.

Format inference cannot answer this, and this is deliberately not a fourth flag
on :class:`~fpy2.analysis.format_infer.AbstractFormat`, which already carries
``has_nan`` / ``has_pos_inf`` / ``has_neg_inf``.  A format bounds a value's
magnitude and says whether the *format* has a NaN, not whether *this* value is
one, and it structurally cannot say **not zero**: ``pos_bound >= 0 >= neg_bound``
holds by convention there, so every format represents a ``+0``.  A no-zero bit
would have to be threaded through ``__add__``, ``__mul__``, the join and storage
selection to buy nothing those need.

That bit is the load-bearing one: the emitted guards a consumer wants to drop
follow from ``x`` being finite *and non-zero*, which is what an ``elif`` ladder
establishes::

    if fp.isnan(x):   ...
    elif fp.isinf(x): ...
    elif x == 0:      ...
    else:                     # x : {Finite}, so logb(x) : {Zero, Finite}
        e = fp.logb(x)

Results are keyed per *expression*, not per definition: the same definition of
``x`` is every class at the ``isnan`` test and only ``Finite`` three arms later.
Expression keys are identities, so **any rewrite of the AST invalidates the
result** — a transform must query the AST it was handed, before rewriting it.

Soundness assumption
--------------------
Only executions in which every operation *has* a result are described.  An
operation handed a value its rounding context refuses has none — the interpreter
raises, the C++ backend asserts — so it contributes no class.  That is what makes
a guard removable at all, and it stays honest: a consumer drops a guard only
where no class reaching the operation is refused, so the abort survives wherever
FPy has no answer.

Precision
---------
Sound by default, precise where it has been taught to be.  An operation with no
rule here reports the classes its rounding context can represent, which for an
unbounded or symbolic context is every class — so adding a rule can only narrow,
never correct.

Scalars only: a list or tuple carries no class, and reading an element gives the
top class.

Not yet taught: the sign of a zero, which would let ``signbit`` refine;
magnitudes (``x > 1``), which is `FormatInfer`'s question; ``assert`` as a
refinement; the class of a numeric free variable; a ``for`` target over
``range``; and the code after an early return, since :meth:`_visit_if1` refines
only its body.
"""

import enum
import functools
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..number import REAL, Context, Float
from ..number.context.format import Format
from ..types import RealType, Type
from .alias import Alias, AliasAnalysis, Region
from .array_size import (
    ArraySizeAnalysis,
    ArraySizeInfer,
    ListSize,
    size_eq,
    trip_count,
)
from .context_use import ContextUse, ContextUseAnalysis, ContextUseSite
from .define_use import (
    AssignDef,
    DefineUse,
    DefineUseAnalysis,
    Definition,
    DefSite,
    PhiDef,
)
from .type_infer import TypeAnalysis, TypeInfer

__all__ = [
    'ClassBound',
    'ListClass',
    'TupleClass',
    'ValueClass',
    'ValueClassAnalysis',
    'ValueClassInfer',
    'class_of',
    'is_positive_literal',
    'representable_classes',
]


#####################################################################
# Lattice

class ValueClass(enum.Flag):
    """Which kinds of value an expression can evaluate to.

    A :class:`enum.Flag`, so the lattice operations are the bitwise ones: ``|``
    joins, ``&`` meets, ``in`` tests membership, and the empty flag is bottom.
    """

    NAN = enum.auto()
    POS_INF = enum.auto()
    NEG_INF = enum.auto()
    ZERO = enum.auto()
    """Either signed zero -- the sign is not tracked."""
    FINITE = enum.auto()
    """Finite and **non-zero**, of either sign."""

    INF = POS_INF | NEG_INF
    """Either infinity.  A composite: ``cls & INF`` asks "infinite at all"."""

    TOP = NAN | INF | ZERO | FINITE


_NAN = ValueClass.NAN
_POS_INF = ValueClass.POS_INF
_NEG_INF = ValueClass.NEG_INF
_INF = ValueClass.INF
_ZERO = ValueClass.ZERO
_FINITE = ValueClass.FINITE
_TOP = ValueClass.TOP
_BOT = ValueClass(0)

_ATOMS = (_NAN, _POS_INF, _NEG_INF, _ZERO, _FINITE)
"""The join-irreducible classes.  ``INF`` is not one: it is the composite a
consumer uses to ask "infinite at all"."""


@dataclass(frozen=True)
class TupleClass:
    """What each field of a tuple can be."""
    elts: 'tuple[ClassBound, ...]'


@dataclass(frozen=True)
class ListClass:
    """What every element of a list can be."""
    elt: 'ClassBound'


ClassBound: TypeAlias = 'ValueClass | TupleClass | ListClass | None'
"""A class shaped like the value it describes, mirroring
:data:`~fpy2.analysis.format_infer.FormatBound`.

A storage choice is structural -- a tuple's is field-wise, a list's is its
element's -- so a class that is not loses at the first aggregate: one
:class:`ValueClass` for a whole tuple is the top class and narrows nothing.
``None`` says nothing about the value at that position, and so does a shape
that does not match the bound's.
"""


def join_class(a: ClassBound, b: ClassBound) -> ClassBound:
    """Either of *a* and *b*, structurally.  ``None`` where they disagree about
    the shape, since nothing is then known of the whole."""
    match a, b:
        case ValueClass(), ValueClass():
            return a | b
        case TupleClass(), TupleClass() if len(a.elts) == len(b.elts):
            return TupleClass(tuple(
                join_class(x, y) for x, y in zip(a.elts, b.elts)
            ))
        case ListClass(), ListClass():
            return ListClass(join_class(a.elt, b.elt))
        case _:
            return None


def _negate(a: ValueClass) -> ValueClass:
    """*a* under negation: the infinities swap, the rest are sign-blind."""
    out = a & ~_INF
    if a & _POS_INF:
        out |= _NEG_INF
    if a & _NEG_INF:
        out |= _POS_INF
    return out


def _magnitude(a: ValueClass) -> ValueClass:
    """*a* under ``abs``: a negative infinity becomes a positive one."""
    out = a & ~_NEG_INF
    if a & _NEG_INF:
        out |= _POS_INF
    return out


def class_of(x: Float) -> ValueClass:
    """The class *x* belongs to."""
    if x.isnan:
        return _NAN
    if x.isinf:
        return _NEG_INF if x.s else _POS_INF
    return _ZERO if x.is_zero() else _FINITE


_PROBES = (Float(isnan=True), Float(isinf=True), Float(isinf=True, s=True))


@functools.cache
def representable_classes(ctx: Context) -> ValueClass:
    """Which classes *ctx* can hold.

    Every zero and every finite non-zero real has a nearest representable value,
    so those two are always in.  A NaN or an infinity is in only where rounding
    one gives one back: a context that refuses the value raises instead, and one
    that substitutes answers with the substitute's class.
    """
    out = _ZERO | _FINITE
    for x in _PROBES:
        out |= _rounded_class(ctx, x)
    return out


@functools.cache
def representable_classes_of(fmt: Format) -> ValueClass:
    """Which classes *fmt* can hold.

    The format-level counterpart of :func:`representable_classes`, for a value
    that is already in the format rather than being rounded into it -- an
    argument binding.  Nothing rounds, so a context's substitution and refusal
    rules cannot apply and membership settles it.
    """
    out = _ZERO | _FINITE
    for x in _PROBES:
        if fmt.representable_in(x):
            out |= class_of(x)
    return out


def _rounded_class(ctx: Context, x: Float) -> ValueClass:
    """The class of ``ctx.round(x)``, or bottom where *ctx* has no result for
    *x* -- it raises, and the analysis describes executions that do not."""
    try:
        return class_of(ctx.round(x))
    except Exception:  # noqa: BLE001 -- a refusal is not a representable class
        return _BOT


#####################################################################
# Transfer functions -- the class of an *exact* real result

def _map(table: dict[ValueClass, ValueClass], a: ValueClass) -> ValueClass:
    out = _BOT
    for atom, res in table.items():
        if atom & a:
            out |= res
    return out


_LOGB = {
    _NAN: _NAN,
    _POS_INF: _POS_INF, _NEG_INF: _POS_INF,   # `logb` reads a magnitude
    _ZERO: _NEG_INF,
    _FINITE: _ZERO | _FINITE,
}
"""``logb(0)`` is ``-inf``, ``logb(inf)`` is ``+inf``, ``logb(1.5)`` is ``0``."""

_POW_BIG_BASE = {
    _NAN: _NAN,
    _POS_INF: _POS_INF, _NEG_INF: _ZERO,
    _ZERO: _FINITE, _FINITE: _FINITE,
}
"""``b ** y`` for a literal ``b > 1``: ``b ** +inf`` is ``+inf`` and
``b ** -inf`` is ``0``.  Which way round is what reading the literal buys --
the two infinities are not alike, and a caller asking whether ``2 ** k`` can be
infinite gets an answer only here."""

_POW_SMALL_BASE = {
    _NAN: _NAN,
    _POS_INF: _ZERO, _NEG_INF: _POS_INF,
    _ZERO: _FINITE, _FINITE: _FINITE,
}
"""``b ** y`` for a literal ``0 < b < 1``: the mirror of :data:`_POW_BIG_BASE`."""

_POW_ONE_BASE = dict.fromkeys(_ATOMS, _FINITE)
"""``1 ** y`` is ``1`` whatever ``y`` is -- a NaN exponent included, which is
what IEEE 754 says and what the sweep against the interpreter confirms."""


def _exact_add(a: ValueClass, b: ValueClass) -> ValueClass:
    """``a + b``; :func:`_exact_sub` negates *b* and reuses this.

    An infinity survives unless the opposite one is added to it, which is where
    the NaN comes from.  ``FINITE`` is sign-blind, so a finite operand can
    neither create nor cancel an infinity.
    """
    if not (a and b):
        return _BOT             # an operand nothing reaches produces nothing
    out = _BOT
    if (a | b) & _NAN:
        out |= _NAN
    for x, y in ((a, b), (b, a)):
        if x & _POS_INF and y & (_POS_INF | _ZERO | _FINITE):
            out |= _POS_INF
        if x & _NEG_INF and y & (_NEG_INF | _ZERO | _FINITE):
            out |= _NEG_INF
    if (a & _POS_INF and b & _NEG_INF) or (a & _NEG_INF and b & _POS_INF):
        out |= _NAN                      # inf - inf
    if a & _ZERO and b & _ZERO:
        out |= _ZERO
    if (a & _ZERO and b & _FINITE) or (a & _FINITE and b & _ZERO):
        out |= _FINITE
    if a & _FINITE and b & _FINITE:
        out |= _ZERO | _FINITE
    return out


def _exact_sub(a: ValueClass, b: ValueClass) -> ValueClass:
    """``a - b``, as ``a + (-b)``."""
    return _exact_add(a, _negate(b))


def _exact_sum(elt: ValueClass) -> ValueClass:
    """``sum(xs)`` for a list whose every element is in *elt*.

    An accumulation, not a selection: how many additions there are is not
    known, so this is the closure of :func:`_exact_add` over *elt*.  That is
    wider than *elt* in both directions a single add is -- two finites cancel
    to a zero, two opposite infinities make a NaN.

    The zero is there whatever the elements are, an empty list summing to one.
    """
    out = _ZERO | elt
    while True:
        grown = out | _exact_add(out, elt)
        if grown == out:
            return out
        out = grown


def _exact_mul(a: ValueClass, b: ValueClass) -> ValueClass:
    if not (a and b):
        return _BOT
    out = _BOT
    if (a | b) & _NAN:
        out |= _NAN
    for x, y in ((a, b), (b, a)):
        if x & _INF and y & (_INF | _FINITE):
            # a product's sign needs both operands', and `FINITE` is sign-blind
            out |= _INF
        if x & _INF and y & _ZERO:
            out |= _NAN                  # 0 * inf
        if x & _ZERO and y & (_ZERO | _FINITE):
            out |= _ZERO
    if a & _FINITE and b & _FINITE:
        out |= _FINITE
    return out


def _exact_select(args: list[ValueClass], *, is_max: bool) -> ValueClass:
    """``max(...)`` or ``min(...)`` over operands of classes *args*.

    A selection knows which operand it picks, which the join does not: ``max``
    is ``+inf`` when *some* operand can be, and ``-inf`` only when *every* one
    can, since an operand that is provably greater is already a larger maximum.
    ``min`` is the dual.  So ``max(logb(x), -126)`` cannot be ``-inf``.

    NaN propagates from any operand, and the finite atoms are joined.
    """
    if not args or not all(args):
        return _BOT             # an operand nothing reaches produces nothing
    near, far = (_POS_INF, _NEG_INF) if is_max else (_NEG_INF, _POS_INF)
    out = _BOT
    for a in args:
        out |= a & (_NAN | _ZERO | _FINITE | near)
    if all(a & far for a in args):
        out |= far
    return out


def _trackable(alias: AliasAnalysis, region: 'Region | None') -> 'Region | None':
    """*region*, unless no fact about its elements may be recorded.

    ``None`` where the list escapes -- handed to a call, which may store
    through it after this analysis has stopped looking.
    """
    if region is None or alias.escapes_at(region):
        return None
    return region


def is_positive_literal(e: Expr) -> bool:
    """Whether *e* is a literal greater than zero."""
    return isinstance(e, RationalVal) and e.as_rational() > 0


def _pow_table(base: Expr) -> dict[ValueClass, ValueClass] | None:
    """Which ``b ** y`` table *base* selects, or `None` where it is not a
    positive literal.  A negative or symbolic base gets no rule.

    None of the three yields ``-inf``: a positive base has no negative power.
    """
    if not is_positive_literal(base):
        return None
    assert isinstance(base, RationalVal)
    b = base.as_rational()
    if b > 1:
        return _POW_BIG_BASE
    if b < 1:
        return _POW_SMALL_BASE
    return _POW_ONE_BASE


#####################################################################
# Result

@dataclass
class ValueClassAnalysis:
    """Result of value-class analysis for an FPy function."""

    func: FuncDef
    """The function whose body was analyzed."""

    by_expr: dict[Expr, ValueClass | None]
    """Class of each expression, refined by the branches that dominate it.
    ``None`` for a non-real-valued expression."""

    by_elt: dict[Definition, ValueClass]
    """For a list definition, every class its elements are ever stored at.

    Where :attr:`by_def` is what a *name* holds, this is what the list behind it
    holds, so a consumer picking storage can narrow the element type.  Joined
    over the whole function rather than read at a point -- storage holds what a
    list ever held.  The top class stands for "anything", which is what a list
    built where no store was walked gets: a literal, a parameter, a callee's
    result.  A definition is absent where no fact may be recorded at all --
    a list that escapes to a callee."""

    by_def: dict[Definition, ValueClass | None]
    """Class of each variable definition, *unrefined* -- the class the defining
    expression had, joined across incoming edges at a phi.  A consumer wants
    :attr:`by_expr`, which is where a branch's refinement shows up; this is the
    per-definition view the other analyses expose, and what
    ``tests/infra/analysis/value_class.py`` dumps."""

    alias: AliasAnalysis
    """Underlying alias analysis: which lists may be the same location.  A class
    for a list's *elements* is a property of that location rather than of a
    name, so a `Region` -- the set of locations a place may hold -- is the key.
    See :meth:`element_region`."""

    type_info: TypeAnalysis
    """Underlying basic-type analysis, which decides what carries a class."""

    ctx_use: ContextUseAnalysis
    """Underlying context-use analysis, which supplies each operation's context."""

    def element_region(self, e: Expr) -> 'Region | None':
        """The region whose elements a fact about the list *e* belongs to, or
        ``None`` where no fact may be recorded.

        A list is a reference, so ``ys = xs`` is one location under two names
        and a store through either is visible through both; the region is what
        both resolve to.
        """
        return _trackable(self.alias, self.alias.region_of_expr(e))

    def bound_of(self, d: Definition) -> ClassBound:
        """*d*'s class, shaped like the value it holds.

        :attr:`by_def` and :attr:`by_elt` answer for a scalar and for a list's
        elements; this is the one a structural consumer wants, and the only
        place that knows both.
        """
        elt = self.by_elt.get(d)
        if elt is not None:
            return ListClass(elt)
        cls = self.by_def.get(d)
        return cls if isinstance(cls, ValueClass) else None

    def classify(self, e: Expr) -> ValueClass:
        """The class of *e*, or the top class where nothing is known."""
        cls = self.by_expr.get(e)
        return cls if isinstance(cls, ValueClass) else _TOP

    def excludes(self, e: Expr, cls: ValueClass) -> bool:
        """Can *e* be none of *cls*?"""
        return not (self.classify(e) & cls)

    def is_finite(self, e: Expr) -> bool:
        """Is *e* neither a NaN nor an infinity?"""
        return self.excludes(e, _NAN | _INF)


#####################################################################
# Analysis

class _ValueClassInstance(DefaultVisitor):
    """Single-use instance of value-class analysis."""

    _ROUNDS_PER_PHI = len(_ATOMS)
    """A phi gains at least one atom per round until it stops growing, so this
    many rounds per phi is enough to reach a fixpoint.  Exceeding it means a
    transfer function is not monotone -- a bug -- and the phis drop to the top
    class rather than the loop running forever."""

    func: FuncDef
    type_info: TypeAnalysis
    ctx_use: ContextUseAnalysis

    by_def: dict[Definition, ValueClass | None]
    by_expr: dict[Expr, ValueClass | None]

    alias: AliasAnalysis

    _elt: dict[Region, ValueClass]
    """What every element of each list currently is.  Keyed by `Region` because
    an element class is a property of the *location*, not of the name reaching
    it, and flow-sensitive because a store changes it -- the same shape as
    :attr:`_refine`, merged at a branch and iterated at a loop."""

    _clock: int
    _touched: dict[Region, int]
    """When each region's elements last changed.  Monotone and never restored,
    so a store anywhere already walked voids a fact taken before it."""

    _scanned: dict[ForStmt, int]
    """For a loop that did *not* store into the list it iterates, that list's
    :attr:`_touched` stamp at the exit.  Absent means the loop's accumulator
    says nothing; see :meth:`_implied_universal`."""

    _sizes_cache: 'ArraySizeAnalysis | None'

    _scan_clocks: dict[ForStmt, tuple[int, int]]
    """The :attr:`_clock` each loop began and ended at, for
    :meth:`_implied_mask`: a list read *inside* a loop is not the one
    :attr:`_scanned` speaks for, so the entry clock is what says nothing has
    stored into it since before the scan, and the exit clock the same for the
    mask the loop filled.  Dropped before the body for the reason
    :attr:`_scanned` is -- a guard *inside* the scan reads a half-filled
    mask."""

    _stored: dict[Region, ValueClass]
    """Every class ever stored into each region, for a consumer choosing
    storage: a buffer holds what a list *ever* held.  Seeded at bottom only
    where a region holding one list is seen empty, so a list built any other
    way -- a literal, a parameter, a callee's result -- stays at the top."""

    _refine: dict[Definition, ValueClass]
    """Per-definition mask the enclosing branches imply, intersected into every
    read of that definition.  Saved and restored around each arm."""

    _refine_elt: dict[Region, tuple[ValueClass, int]]
    """The same, per region, with the :attr:`_touched` stamp it was taken at.

    :attr:`_refine` needs no stamp, a rebind making a new `Definition` where a
    list's contents change under a fixed one.  An arm *restores* this map, so
    without the stamp a store in a branch nested inside it would come back
    undone on the way out."""

    def __init__(
        self,
        func: FuncDef,
        type_info: TypeAnalysis,
        ctx_use: ContextUseAnalysis,
        alias: AliasAnalysis,
    ):
        self.func = func
        self.type_info = type_info
        self.ctx_use = ctx_use
        self.alias = alias
        self._elt = {}
        self._stored = {}
        self._clock = 0
        self._touched = {}
        self._scanned = {}
        self._scan_clocks = {}
        self._sizes_cache = None
        self.by_def = {}
        self.by_expr = {}
        self._refine = {}
        self._refine_elt = {}

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.type_info.def_use

    @property
    def sizes(self) -> ArraySizeAnalysis:
        """Array sizes, computed on first use.  Only :meth:`_implied_mask` wants
        them, and only a program that guards on a mask reaches it."""
        if self._sizes_cache is None:
            self._sizes_cache = ArraySizeInfer.analyze(self.func)
        return self._sizes_cache

    def _by_elt(self) -> dict[Definition, ValueClass]:
        """:attr:`ValueClassAnalysis.by_elt`, per definition rather than per
        region."""
        # every key of `_stored` went through `_trackable`, so an escaping
        # region is already absent
        out: dict[Definition, ValueClass] = {}
        for d in self.def_use.defs:
            region = self.alias.region_of(d)
            if region is not None and region in self._stored:
                out[d] = self._stored[region]
        return out

    def analyze(self) -> ValueClassAnalysis:
        self._visit_function(self.func, None)
        return ValueClassAnalysis(
            func=self.func,
            by_expr=self.by_expr,
            by_def=self.by_def,
            by_elt=self._by_elt(),
            alias=self.alias,
            type_info=self.type_info,
            ctx_use=self.ctx_use,
        )

    # ------------------------------------------------------------------
    # Definitions

    def _region_of(self, e: Expr) -> 'Region | None':
        """See :meth:`ValueClassAnalysis.element_region`."""
        return _trackable(self.alias, self.alias.region_of_expr(e))

    def _one_list(self, region: Region) -> bool:
        """Whether *region* abstracts a single list.

        Two lists share a region as soon as anything makes them may-alias, and
        then "every element of *the* list" names neither of them.  Only a count
        answers it, and a region with no site is as unanswerable as one with
        two.

        A count of allocations is not enough on its own: the rows of one
        nested list share a region *and* a site, so `inside_at` rules out a
        region that is a place within a container, which stands for one list
        per element of it.
        """
        return (len(self.alias.sites_at(region)) == 1
                and not self.alias.inside_at(region))

    def _sole_region(self, e: Expr) -> 'Region | None':
        """*e*'s region, where it abstracts exactly one list.

        Two lists sharing a region means a fact about one says nothing about
        the other, so a caller refining "every element of *the* list" has
        nothing to key on.
        """
        region = self._region_of(e)
        return region if region is not None and self._one_list(region) else None

    def _elements_of(self, e: Expr) -> ValueClass:
        """What every element of the list *e* names currently is."""
        region = self._region_of(e)
        if region is None:
            return _TOP
        return self._elt.get(region, _TOP) & self._mask_of(region)

    def _stamp(self, region: Region) -> int:
        return self._touched.get(region, 0)

    def _touch(self, region: Region):
        """Record that *region*'s elements changed."""
        self._clock += 1
        self._touched[region] = self._clock

    def _mask_of(self, region: Region) -> ValueClass:
        """What the enclosing branches imply about *region*'s elements, or the
        top class once a store has landed since that was taken."""
        cls, stamp = self._refine_elt.get(region, (_TOP, 0))
        return cls if stamp == self._stamp(region) else _TOP

    def _store_element(self, region: 'Region | None', cls: ValueClass):
        """Record a store of *cls* into *region*, which joins: the elements the
        store did not reach are still there."""
        if region is not None:
            self._elt[region] = self._elt.get(region, _TOP) | cls
            self._stored[region] = self._stored.get(region, _TOP) | cls
            self._touch(region)

    @staticmethod
    def _join_elements(
        a: 'dict[Region, ValueClass]', b: 'dict[Region, ValueClass]',
    ) -> 'dict[Region, ValueClass]':
        """Two paths' element maps.  Absent means *unknown*, so it joins to the
        top rather than being skipped."""
        return {
            r: a.get(r, _TOP) | b.get(r, _TOP)
            for r in (*a, *b)
        }

    def _set_def(self, d: Definition, cls: ValueClass | None):
        if not isinstance(self.type_info.by_def.get(d), RealType):
            cls = None
        self.by_def[d] = cls

    def _def_class(self, d: Definition) -> ValueClass:
        cls = self.by_def.get(d)
        return cls if isinstance(cls, ValueClass) else _TOP

    def _bind(self, site: DefSite, binding: Id | TupleBinding, cls: ValueClass | None):
        """Records *cls* for every variable *binding* introduces at *site*."""
        match binding:
            case NamedId():
                self._set_def(self.def_use.find_def_from_site(binding, site), cls)
            case UnderscoreId():
                pass
            case TupleBinding():
                # no structural classes: an unpacked element is unconstrained
                for sub in binding.elts:
                    self._bind(site, sub, _TOP)
            case _:
                raise RuntimeError(f'unreachable: {binding}')

    def _merge_phis(self, stmt: Stmt):
        """Joins each phi's incoming classes.

        The masks the two arms were walked under are gone by now, and deliberately
        so: a definition made *inside* an arm already has that arm's refinement
        folded into its class, and reading an outer definition unrefined is the
        sound direction.
        """
        for phi in self.def_use.phis[stmt]:
            lhs = self._def_class(self.def_use.defs[phi.lhs])
            rhs = self._def_class(self.def_use.defs[phi.rhs])
            self._set_def(phi, lhs | rhs)

    # ------------------------------------------------------------------
    # Refinement

    @contextmanager
    def _refined(self, cond: Expr, truth: bool) -> Iterator[None]:
        """Walk an arm with *cond* known to be *truth*.

        Both arms narrow the *enclosing* mask.  Narrowing whatever
        ``self._refine`` happens to hold would carry the first arm's refinement
        into its sibling -- intersecting ``{NaN}`` with ``{Inf}`` down an ``elif``
        ladder and driving every later use to the empty class.
        """
        saved, saved_elt = self._refine, self._refine_elt
        out = dict(saved)
        for d, cls in self._implied(cond, truth):
            out[d] = out.get(d, _TOP) & cls
        # re-stamped, so an entry a store has already invalidated reads as the
        # top class here rather than coming back as this arm's starting point
        out_elt = {r: (self._mask_of(r), self._stamp(r)) for r in saved_elt}
        for region, cls in self._implied_elements(cond, truth):
            prev, _ = out_elt.get(region, (_TOP, 0))
            out_elt[region] = (prev & cls, self._stamp(region))
        self._refine, self._refine_elt = out, out_elt
        try:
            yield
        finally:
            self._refine, self._refine_elt = saved, saved_elt

    def _implied(self, cond: Expr, truth: bool) -> list[tuple[Definition, ValueClass]]:
        """What *cond* being *truth* says about the definitions it tests."""
        match cond:
            case Not():
                return self._implied(cond.arg, not truth)
            case And() if truth:
                return [i for a in cond.args for i in self._implied(a, True)]
            case Or() if not truth:
                return [i for a in cond.args for i in self._implied(a, False)]
            case IsNan():
                return self._at(cond.arg, _NAN if truth else _INF | _ZERO | _FINITE)
            case IsInf():
                return self._at(cond.arg, _INF if truth else _NAN | _ZERO | _FINITE)
            case IsFinite():
                return self._at(cond.arg, _ZERO | _FINITE if truth else _NAN | _INF)
            case IsNormal() if truth:
                return self._at(cond.arg, _FINITE)   # normal implies non-zero
            case Compare():
                return self._implied_compare(cond, truth)
            case Var():
                # through a name: a test bound to one still says what it tests
                src = self.def_use.defining_expr(cond)
                if src is not cond:
                    return self._implied(src, truth)
                # no single defining expression -- but a lowered chain is a
                # phi, and still a conjunction
                return self._implied_ladder(
                    self.def_use.use_to_def.get(cond), truth,
                )
            case _:
                return []

    def _implied_ladder(
        self, d: 'Definition | None', truth: bool
    ) -> list[tuple[Definition, ValueClass]]:
        """What a *lowered* ``and``/``or`` being *truth* says.

        :class:`~fpy2.transform.Hoistable` rewrites a chain whose tail needs a
        statement into a flat ladder of guarded assignments, which the ``And``
        case above cannot match:

        .. code-block:: python

            t = not isnan(a)        # `not isnan(a) and not isnan(b)`
            if t: t = not isnan(b)
            if t: ...               # `t` here is a phi of the two

        The conjunction is only moved.  ``t`` takes the incoming value where the
        guard failed and the body's where it held, and the guard *is* the
        incoming value -- so ``t`` true forces the guard true, hence both
        operands true.  Dually for an ``or``, which guards on the negation and
        says something only when the whole is false.

        Recursion terminates: a phi's operands are defined before it, so each
        step moves strictly earlier in :attr:`DefineUseAnalysis.defs`.  A loop's
        phi never matches, its site being a ``while`` rather than an ``if``.
        """
        if not isinstance(d, PhiDef) or not isinstance(d.site, If1Stmt):
            return []
        # `or` guards on the accumulator's negation, `and` on the accumulator
        guard = d.site.cond
        if isinstance(guard, Not):
            negated, test = True, guard.arg
        else:
            negated, test = False, guard
        if not isinstance(test, Var):
            return []
        # the guard must test exactly the value the phi joins, or this is some
        # other `if p: t = q` that says nothing about `t`
        guard_def = self.def_use.use_to_def.get(test)
        if guard_def is None or self.def_use.def_to_idx.get(guard_def) != d.lhs:
            return []
        if truth is negated:      # `and` speaks when true, `or` when false
            return []
        return [
            i for idx in (d.lhs, d.rhs)
            for i in self._implied_at(self.def_use.defs[idx], truth)
        ]

    def _implied_at(
        self, d: Definition, truth: bool
    ) -> list[tuple[Definition, ValueClass]]:
        """What the definition *d* holding *truth* says, one rung of a ladder.

        The expression belongs to its own assignment, so its variables resolve
        to the definitions reaching *there* -- which is what makes the fact
        correct, and why a later redefinition simply goes unrefined.
        """
        if isinstance(d, AssignDef) and isinstance(d.site, Assign):
            return self._implied(d.site.expr, truth)
        return self._implied_ladder(d, truth)     # a longer ladder

    def _implied_elements(
        self, cond: Expr, truth: bool
    ) -> 'list[tuple[Region, ValueClass]]':
        """What *cond* being *truth* says about the elements of a list."""
        match cond:
            case Not():
                return self._implied_elements(cond.arg, not truth)
            case And() if truth:
                return [i for a in cond.args
                        for i in self._implied_elements(a, True)]
            case Or() if not truth:
                return [i for a in cond.args
                        for i in self._implied_elements(a, False)]
            case AllOf() if truth:
                return self._implied_mask(cond.arg, True)
            case AnyOf() if not truth:
                return self._implied_mask(cond.arg, False)
            case Var():
                src = self.def_use.defining_expr(cond)
                if src is not cond:
                    return self._implied_elements(src, truth)
                return self._implied_universal(
                    self.def_use.use_to_def.get(cond), truth,
                )
            case _:
                return []

    def _implied_universal(
        self, d: 'Definition | None', truth: bool
    ) -> 'list[tuple[Region, ValueClass]]':
        """What a *lowered* ``all`` / ``any`` being *truth* says about the list
        it scanned.

        :class:`~fpy2.transform.ReduceFusion` leaves the reduction as a
        loop-carried fold, which the ``And`` case above cannot match:

        .. code-block:: python

            acc = True                  # `all([isfinite(x) for x in xs])`
            for x in xs:
                b = isfinite(x)
                acc = acc and b

        The exit value is ``seed and b_1 and ... and b_n``, so ``acc`` true
        forces every ``b`` -- whatever the seed, with an empty list vacuous.
        The loop covers the list, FPy having no ``break``, so what the fold's
        other operands say about the target they say about every element.
        Dually for ``any``, an ``Or`` that speaks when it is false.

        Nothing here names what the lowering minted: an inlined predicate and a
        hand-written fold match too.
        """
        if not isinstance(d, PhiDef) or not isinstance(d.site, ForStmt):
            return []
        stmt = d.site
        if not isinstance(stmt.target, NamedId):
            return []
        region = self._region_of(stmt.iterable)
        # a store since the exit -- or one the loop made itself, which leaves
        # no entry -- means the list read is not the list scanned, and a region
        # holding two lists means scanning one says nothing about the other
        if region is None or not self._one_list(region):
            return []
        if self._scanned.get(stmt) != self._stamp(region):
            return []
        target = self.def_use.find_def_from_site(stmt.target, stmt)
        return [
            (region, cls)
            for td, cls in self._implied_fold(d, truth)
            if td == target
        ]

    def _implied_fold(
        self, d: PhiDef, truth: bool
    ) -> list[tuple[Definition, ValueClass]]:
        """What one round of the fold *d* accumulates says, given the exit value
        is *truth*.

        The accumulator must be an operand of its own new value.  That is what
        makes the fold monotone, and so what lets the exit value speak for every
        round: ``acc = p(x)`` speaks for the last element alone.
        """
        step = self.def_use.defs[d.rhs]
        if isinstance(step, PhiDef):
            # `Hoistable` moves the fold into a guarded assignment, and the
            # guard it checks for is the accumulator -- but only where that is
            # what the loop carried in.  `ok = x > 0; if ok: ok = p(x)` rebuilds
            # `ok` each round, so it too speaks for the last element alone.
            if self.def_use.def_to_idx.get(d) != step.lhs:
                return []
            return self._implied_ladder(step, truth)
        if not isinstance(step, AssignDef) or not isinstance(step.site, Assign):
            return []
        fold = step.site.expr
        if not isinstance(fold, And if truth else Or):
            return []
        rest = [
            a for a in fold.args
            if not (isinstance(a, Var) and self.def_use.use_to_def.get(a) == d)
        ]
        if len(rest) == len(fold.args):
            return []
        return [i for a in rest for i in self._implied(a, truth)]

    def _implied_mask(
        self, mask: Expr, truth: bool
    ) -> 'list[tuple[Region, ValueClass]]':
        """What a *materialised* ``all`` / ``any`` being *truth* says about the
        list the mask was computed from.

        :class:`~fpy2.transform.CompToLoop` leaves the predicate in a list
        rather than in a fold, which :meth:`_implied_universal` cannot match:

        .. code-block:: python

            m = fp.empty(32)            # `all([isfinite(x) for x in xs])`
            for i in range(32):
                x = xs[i]
                m[i] = fp.isfinite(x)
            if all(m): ...

        ``all(m)`` true forces every element of ``m``, and the loop wrote
        ``p(xs[i])`` into element ``i`` -- so where it ran once per element of
        ``xs``, what ``p`` says about the element it bound it says about every
        one.  Dually for ``any``, which speaks when it is false.

        Nothing here names what the lowering minted: a hand-written loop over
        a mask matches too, so long as it *binds* the element -- the
        refinement travels through the definition the predicate reads, and
        `m[i] = isfinite(xs[i])` gives it none.
        """
        if not isinstance(mask, Var):
            return []
        d = self.def_use.use_to_def.get(mask)
        if not isinstance(d, PhiDef) or not isinstance(d.site, ForStmt):
            return []
        loop = d.site
        clocks = self._scan_clocks.get(loop)
        if clocks is None:
            return []       # the guard sits inside the scan, over a half-filled mask
        entry, exited = clocks
        if not self._holds_the_scan(mask, exited):
            return []
        # the mask's last definition must be the store the loop makes, or the
        # value `all` reads is not the one the predicate wrote
        write = self.def_use.defs[d.rhs]
        if not isinstance(write, AssignDef) or not isinstance(write.site, IndexedAssign):
            return []
        # and that store must read what the loop carried in: an earlier store in
        # the same round leaves an element no later round rewrites
        idx = self.def_use.def_to_idx.get(d)
        if idx is None or write.prev != idx:
            return []
        store = write.site
        if len(store.indices) != 1 or not self._is_target(store.indices[0], loop):
            return []
        return [
            (region, cls)
            for td, cls in self._implied(store.expr, truth)
            if (region := self._scanned_by(td, loop, entry)) is not None
        ]

    def _holds_the_scan(self, mask: Expr, exited: int) -> bool:
        """Whether *mask* still holds what the loop wrote into it, the loop
        having ended at the clock read *exited*.

        A redefinition of the name is caught by reaching defs -- the guard
        would not read the loop's phi at all -- but a store through another
        name for the same list is not, and neither is a callee's.
        """
        region = self._sole_region(mask)
        return region is not None and self._stamp(region) <= exited

    def _is_target(self, e: Expr, loop: ForStmt) -> bool:
        """Whether *e* names *loop*'s target."""
        if not isinstance(e, Var) or not isinstance(loop.target, NamedId):
            return False
        target = self.def_use.find_def_from_site(loop.target, loop)
        return self.def_use.use_to_def.get(e) == target

    def _scanned_by(
        self, d: Definition, loop: ForStmt, entry: int
    ) -> 'Region | None':
        """The region *d* reads an element of, where *loop* covers it and
        nothing has stored into it since the clock read *entry*.  ``None``
        where *d* is not such a read, or where any of that is unproven."""
        if not isinstance(d, AssignDef) or not isinstance(d.site, Assign):
            return None
        ref = d.site.expr
        if not isinstance(ref, ListRef) or not isinstance(ref.value, Var):
            return None
        if not self._is_target(ref.index, loop):
            return None
        region = self._sole_region(ref.value)
        if region is None:
            return None
        # a store since before the scan leaves the elements the predicate
        # tested different from the ones the list holds now
        if self._stamp(region) > entry:
            return None
        # and the loop must have run once per element, not over a prefix
        size = self.sizes.by_def.get(self.def_use.find_def_from_use(ref.value))
        if not isinstance(size, ListSize):
            return None
        if not size_eq(trip_count(loop.iterable, self.sizes), size.size):
            return None
        return region

    def _implied_compare(
        self, cond: Compare, truth: bool
    ) -> list[tuple[Definition, ValueClass]]:
        """A comparison's refinements.

        A comparison that *holds* rules out a NaN on both sides, since a NaN
        compares false to everything, and one against a literal pins the class
        outright.  **A comparison that fails rules out nothing** -- the trap the
        ``x == 0`` row exists for: ``not (x == 0)`` does not mean non-zero,
        because a NaN takes that arm too.  The exception is an equality against
        zero, whose failure rules out a zero and nothing else.
        """
        args = cond.args
        if truth:
            pairs = list(zip(cond.ops, args, args[1:]))
        elif len(cond.ops) != 1:
            return []           # a failed chain does not say which link broke
        elif cond.ops[0] is CompareOp.NE:
            pairs = [(CompareOp.EQ, args[0], args[1])]   # `not (a != b)` is `a == b`
        elif cond.ops[0] is CompareOp.EQ:
            return [i for x, y in _both(args) if _is_zero_literal(y)
                    for i in self._at(x, _NAN | _INF | _FINITE)]
        else:
            return []           # a failed ordering admits a NaN

        out: list[tuple[Definition, ValueClass]] = []
        for op, a, b in pairs:
            for x, y in _both((a, b)):
                v = _literal_value(y)
                if op is CompareOp.NE:
                    # `x != 0` rules out a zero; `x != 1` rules out nothing,
                    # since a NaN is unequal to everything
                    if v == 0:
                        out += self._at(x, _NAN | _INF | _FINITE)
                elif op is CompareOp.EQ and v is not None:
                    out += self._at(x, _ZERO if v == 0 else _FINITE)
                else:
                    out += self._at(x, _INF | _ZERO | _FINITE)
        return out

    def _at(self, e: Expr, cls: ValueClass) -> list[tuple[Definition, ValueClass]]:
        """*cls*, against the definition *e* names -- nothing unless *e* is a
        real-valued variable, since only a definition can be refined."""
        if not isinstance(e, Var):
            return []
        if not isinstance(self.type_info.by_expr.get(e), RealType):
            return []
        return [(self.def_use.find_def_from_use(e), cls)]

    # ------------------------------------------------------------------
    # Expressions

    def _visit_expr(self, e: Expr, ctx: None) -> ValueClass | None:  # type: ignore[override]
        cls = super()._visit_expr(e, ctx)
        if not isinstance(self.type_info.by_expr.get(e), RealType):
            cls = None
        elif not isinstance(cls, ValueClass):
            cls = _TOP
        self.by_expr[e] = cls
        return cls

    def _operand(self, e: Expr, ctx: None) -> ValueClass:
        cls = self._visit_expr(e, ctx)
        return cls if isinstance(cls, ValueClass) else _TOP

    def _rounded(self, e: ContextUseSite, exact: ValueClass) -> ValueClass:
        """*exact* as the operation's rounding context leaves it.

        Rounding under :data:`REAL` is the identity, so the exact class stands.
        Under any other concrete context the result is a value that context
        represents -- which is all that can be said without modelling overflow,
        underflow and substitution per context.

        Only for an operation that really does round its result: a selection
        (``min``) or a projection (``fst``) passes an operand through untouched,
        and could carry a NaN out of a context with no NaN.
        """
        scope = self.ctx_use.use_to_scope.get(e)
        if scope is None or not isinstance(scope.ctx, Context):
            return _TOP
        return exact if scope.ctx is REAL else representable_classes(scope.ctx)

    def _visit_list_ref(self, e: ListRef, ctx: None) -> ValueClass:
        """``xs[i]``: whatever every element of ``xs`` currently is."""
        self._visit_expr(e.index, ctx)
        self._visit_expr(e.value, ctx)
        return self._elements_of(e.value)

    def _visit_var(self, e: Var, ctx: None) -> ValueClass:
        d = self.def_use.find_def_from_use(e)
        return self._def_class(d) & self._refine.get(d, _TOP)

    def _visit_decnum(self, e: Decnum, ctx: None) -> ValueClass:
        return _literal_class(e)

    def _visit_hexnum(self, e: Hexnum, ctx: None) -> ValueClass:
        return _literal_class(e)

    def _visit_integer(self, e: Integer, ctx: None) -> ValueClass:
        return _literal_class(e)

    def _visit_rational(self, e: Rational, ctx: None) -> ValueClass:
        return _literal_class(e)

    def _visit_digits(self, e: Digits, ctx: None) -> ValueClass:
        return _literal_class(e)

    def _visit_nullaryop(self, e: NullaryOp, ctx: None) -> ValueClass:
        match e:
            case ConstNan():
                exact = _NAN
            case ConstInf():
                exact = _POS_INF        # `-inf` is a `Neg` of this
            case _:
                exact = _FINITE      # pi, e, sqrt2, ...
        return self._rounded(e, exact)

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> ValueClass:
        a = self._operand(e.arg, ctx)
        match e:
            case Neg():
                return self._rounded(e, _negate(a))
            case Abs():
                return self._rounded(e, _magnitude(a))
            case Cast():
                return self._rounded(e, a)
            case Logb():
                return self._rounded(e, _map(_LOGB, a))
            case AMin() | AMax():
                # the result *is* one element, so it is bounded by them
                return self._elements_of(e.arg)
            case Sum():
                # an accumulation *of* the elements rather than one of them,
                # so the bound they give has to be closed under adding
                return self._rounded(e, _exact_sum(self._elements_of(e.arg)))
            case Fst() | Snd():
                return _TOP          # passes an operand through; see `_rounded`
            case _:
                return self._rounded(e, _TOP)

    # `RoundAt` needs no case of its own: the base visitor sends it to
    # `_visit_binaryop`, whose fallback is what it would get anyway -- it rounds
    # digits away even under `REAL`, so the operand's class does not carry over.
    def _visit_round(self, e: Round, ctx: None) -> ValueClass:
        return self._rounded(e, self._operand(e.arg, ctx))

    def _visit_binaryop(self, e: BinaryOp, ctx: None) -> ValueClass:
        a = self._operand(e.first, ctx)
        b = self._operand(e.second, ctx)
        match e:
            case Add():
                return self._rounded(e, _exact_add(a, b))
            case Sub():
                return self._rounded(e, _exact_sub(a, b))
            case Mul():
                return self._rounded(e, _exact_mul(a, b))
            case Pow() if (table := _pow_table(e.first)) is not None:
                return self._rounded(e, _map(table, b))
            case _:
                return self._rounded(e, _TOP)

    def _visit_ternaryop(self, e: TernaryOp, ctx: None) -> ValueClass:
        for arg in (e.first, e.second, e.third):
            self._visit_expr(arg, ctx)
        return self._rounded(e, _TOP)

    def _visit_naryop(self, e: NaryOp, ctx: None) -> ValueClass:
        args = [self._operand(arg, ctx) for arg in e.args]
        match e:
            case Min() | Max():
                # a selection, not a rounding: no `_rounded`
                return _exact_select(args, is_max=isinstance(e, Max))
            case _:
                return self._rounded(e, _TOP)

    def _visit_compare(self, e: Compare, ctx: None) -> None:
        for arg in e.args:
            self._visit_expr(arg, ctx)

    def _visit_call(self, e: Call, ctx: None) -> ValueClass:
        super()._visit_call(e, ctx)
        # the callee produces the result, so the caller's context says nothing
        return _TOP

    def _visit_if_expr(self, e: IfExpr, ctx: None) -> ValueClass:
        # arms unrefined: a backend may evaluate both on every input
        self._visit_expr(e.cond, ctx)
        return self._operand(e.ift, ctx) | self._operand(e.iff, ctx)

    # ------------------------------------------------------------------
    # Statements

    def _visit_assign(self, stmt: Assign, ctx: None):
        self._bind(stmt, stmt.target, self._visit_expr(stmt.expr, ctx))
        if isinstance(stmt.expr, Empty):
            # A fresh allocation holds nothing yet, which is what lets the
            # stores that follow say anything.  Only where the region is one
            # list: it is the sole *strong* update here, and wiping a region
            # two lists share would drop the other one's elements.  Every other
            # way of building a list -- a literal, a parameter, a callee's
            # result -- puts elements there without a store, so a region never
            # seen empty keeps the top class however much is stored into it.
            region = self._region_of_def(stmt.target, stmt)
            if region is not None:
                if self._one_list(region):
                    self._elt[region] = _BOT
                    self._stored[region] = _BOT
                self._touch(region)

    def _region_of_def(self, target, site, depth: int = 0) -> 'Region | None':
        """The region *depth* list levels inside what *target* binds at *site*.
        ``depth`` is what a nested store writes through."""
        if not isinstance(target, NamedId):
            return None
        return _trackable(self.alias, self.alias.region_of(
            self.def_use.find_def_from_site(target, site), depth,
        ))

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        stored = self._visit_expr(stmt.expr, ctx)
        # a fresh def of a list, which carries no *scalar* class; what the store
        # says is about the region's elements
        self._bind(stmt, stmt.var, None)
        # `xss[i][j] = v` writes the region one level in, which is where the
        # alias analysis puts it too; the level above holds lists, and what a
        # store says about those is nothing.
        depth = len(stmt.indices) - 1
        self._store_element(
            self._region_of_def(stmt.var, stmt, depth),
            stored if isinstance(stored, ValueClass) else _TOP,
        )
        for above in range(depth):
            self._store_element(
                self._region_of_def(stmt.var, stmt, above), _TOP,
            )

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        entry = dict(self._elt)
        with self._refined(stmt.cond, True):
            self._visit_block(stmt.body, ctx)
        # the body may not have run, so its stores only *may* have happened
        self._elt = self._join_elements(entry, self._elt)
        self._merge_phis(stmt)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        entry = dict(self._elt)
        with self._refined(stmt.cond, True):
            self._visit_block(stmt.ift, ctx)
        taken, self._elt = self._elt, entry
        with self._refined(stmt.cond, False):
            self._visit_block(stmt.iff, ctx)
        self._elt = self._join_elements(taken, self._elt)
        self._merge_phis(stmt)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        def body():
            self._visit_expr(stmt.cond, ctx)
            with self._refined(stmt.cond, True):
                self._visit_block(stmt.body, ctx)

        self._fixpoint(stmt, body)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        self._visit_expr(stmt.iterable, ctx)
        region = self._region_of(stmt.iterable)
        before = None if region is None else self._stamp(region)

        def body():
            # the target *is* an element, so it is whatever they are
            self._bind(stmt, stmt.target, self._elements_of(stmt.iterable))
            self._visit_block(stmt.body, ctx)

        # dropped before the body runs, not after: a `for` inside a loop is
        # walked again, and inside it the accumulator covers only the part
        # scanned so far -- an entry left from the previous walk would speak
        # for the whole list
        self._scanned.pop(stmt, None)
        self._scan_clocks.pop(stmt, None)
        entry = self._clock
        self._fixpoint(stmt, body)
        self._scan_clocks[stmt] = (entry, self._clock)
        if region is not None and self._stamp(region) == before:
            self._scanned[stmt] = before

    def _fixpoint(self, stmt: Stmt, run_body: Callable[[], None]):
        """Drives a loop's phi classes to convergence.

        A phi's class starts at what reached the loop and is re-joined with the
        body's result until two rounds agree.  Joining only ever adds atoms and
        there are finitely many, so the sequence stops on its own; if it has not
        stopped after :attr:`_ROUNDS_PER_PHI` rounds per phi, a transfer
        function is not monotone and every phi is dropped to the top class.
        """
        phis = self.def_use.phis[stmt]
        for phi in phis:
            self._set_def(phi, self._def_class(self.def_use.defs[phi.lhs]))
        entry = dict(self._elt)
        for _ in range(self._ROUNDS_PER_PHI * len(phis) + 1):
            prev = ({phi: self.by_def[phi] for phi in phis}, dict(self._elt))
            run_body()
            for phi in phis:
                lhs = self._def_class(self.def_use.defs[phi.lhs])
                rhs = self._def_class(self.def_use.defs[phi.rhs])
                self._set_def(phi, lhs | rhs)
            # the body runs zero or more times, so its stores join with the
            # state that reached the loop as well as with the round before
            self._elt = self._join_elements(entry, self._elt)
            if (({phi: self.by_def[phi] for phi in phis}, self._elt)) == prev:
                return
        for phi in phis:
            self._set_def(phi, _TOP)
        self._elt = {r: _TOP for r in self._elt}
        run_body()

    def _visit_context(self, stmt: ContextStmt, ctx: None):
        self._visit_expr(stmt.ctx, ctx)
        self._bind(stmt, stmt.target, None)
        self._visit_block(stmt.body, ctx)

    def _visit_list_comp(self, e: ListComp, ctx: None) -> None:
        for target, iterable in zip(e.targets, e.iterables):
            self._visit_expr(iterable, ctx)
            self._bind(e, target, _TOP)
        self._visit_expr(e.elt, ctx)

    def _visit_function(self, func: FuncDef, ctx: None):
        for arg in func.args:
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._set_def(d, _arg_class(self.type_info.by_def.get(d)))
        for v in func.free_vars:
            self._set_def(self.def_use.find_def_from_site(v, func), _TOP)
        self._visit_block(func.body, ctx)


def _both(pair: Sequence[Expr]) -> tuple[tuple[Expr, Expr], ...]:
    """Both orderings of a comparison's operands: either side may be the literal
    and either side may be the variable to refine."""
    a, b = pair
    return ((a, b), (b, a))


def _literal_value(e: Expr) -> Fraction | None:
    """The exact value of a numeric literal, or `None` if *e* is not one."""
    return e.as_rational() if isinstance(e, RationalVal) else None


def _is_zero_literal(e: Expr) -> bool:
    return _literal_value(e) == 0


def _literal_class(e: RationalVal) -> ValueClass:
    """A literal is exact -- an enclosing operation rounds it, not the binding."""
    return _ZERO if e.as_rational() == 0 else _FINITE


def _arg_class(ty: Type | None) -> ValueClass:
    """A parameter's class, from the format its declared type pins it to."""
    if isinstance(ty, RealType) and ty.fmt is not None:
        return representable_classes_of(ty.fmt)
    return _TOP


class ValueClassInfer:
    """
    Path-sensitive value-class analysis.

    Computes, for every expression, which of ``{NaN, Inf, Zero, Finite}`` it can
    evaluate to, refining at each branch that tests a value's class.  See the
    module docstring for the lattice, the soundness assumption, and what the
    analysis has not been taught.
    """

    @staticmethod
    def analyze(
        func: FuncDef,
        *,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        ctx_use: ContextUseAnalysis | None = None,
        alias: AliasAnalysis | None = None,
    ) -> ValueClassAnalysis:
        """
        Runs value-class analysis on a function.

        The pre-analyses are accepted as keyword arguments so a caller that
        already holds them does not recompute them.

        *alias* is computed here when absent rather than the analysis going
        without: it costs about half what this analysis does, and without it a
        list fact would have to be dropped silently.  Computed here it has no
        escape summaries, which is the *conservative* reading -- every list
        handed to a call is marked as escaping.  A caller holding a summarized
        one should pass it: the C++ backend does, and gets the element facts
        the conservative reading throws away.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        if ctx_use is None:
            ctx_use = ContextUse.analyze(func, def_use=def_use)
        if alias is None:
            alias = Alias.analyze(func, def_use=def_use, type_info=type_info)
        return _ValueClassInstance(func, type_info, ctx_use, alias).analyze()
