"""
Digit-bound inference over an FPy program.

Each real-valued expression and definition gets two integer variables:
``msb``, an upper bound on ``logb`` of its value, and ``lsb``, a lower bound
on the position of its least significant digit.  Constraints relate them, and
their difference answers the one question this analysis exists for --

    precision of v  =  max(msb(v) - lsb(v) + 1)

-- which for a rounded value is the precision of the fixed-point result.  See
``docs/todos/digit-bound-inference.md``.

A reduced product with :class:`fpy2.analysis.FormatInfer`: that pass supplies
the ranges this seeds from and consumes the precisions this derives.  Nothing
here builds a :class:`Format`, and nothing in the format lattice names a term.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from ...ast.fpyast import *
from ...ast.visitor import DefaultVisitor
from ...function import Function
from ...interpret.value import unwrap_foreign
from ...number import RoundingMode
from ...number.context.mp_fixed import MPFixedContext
from ...types import ListType, RealType
from ..array_size import ArraySizeAnalysis, ArraySizeBound, ListSize, concrete_size
from ..context_use import ContextUseAnalysis, PartialContext
from ..reaching_defs import Definition, PhiDef
from ..type_infer import TypeAnalysis
from .store import DigitBoundStore, Term

__all__ = [
    'Bounds',
    'DigitBoundAnalysis',
    'DigitBoundInfer',
    'FormatView',
    'Terms',
]


@dataclass
class _IndexSet:
    """One renaming of the per-element variables, for one range."""

    tag: str
    subst: dict[int, Term] = field(default_factory=dict)
    mark: int = 0


@dataclass
class Terms:
    """What a value names in the store.

    ``logb`` and ``grid`` bracket it from above and below; their difference is
    its precision.  ``value`` is the exact value, for the integer-valued
    expressions a rounding position is built from, and is filled lazily since a
    context expression is not part of the walk.

    For a *list* all three describe an arbitrary element, as ``ListFormat`` does
    for formats: a fact stated about one is a claim about every element.
    """

    msb: Term | None = None
    lsb: Term | None = None
    value: Term | None = None


@dataclass(frozen=True)
class Bounds:
    """What the store can say about a value.  Turning these into a
    :class:`Format` is the format domain's business."""

    prec: int
    """significant digits, never more than *exp* and *mag* leave room for"""

    exp: int
    """position of the least significant digit"""

    mag: int
    """the value is below ``2 ** (mag + 1)``"""


@dataclass(frozen=True)
class DigitBoundParams:
    """A callee's parameters as a caller bound them: the constraint system,
    and one term per parameter in it.

    The digit-bound analogue of :attr:`FunctionFormat.arg_fmts`, and it cannot
    take that shape.  A relation *between* two arguments -- ``n`` is ``xs``'s
    greatest exponent less twelve -- lives in the store and reduces to no
    per-argument bound, so the terms travel with the system that gives them
    meaning and a spec analyzed on its own cannot see it.
    """

    store: DigitBoundStore
    args: tuple[Terms, ...]


@dataclass
class DigitBoundAnalysis:
    """Result of digit-bound inference: the constraint system, and what names it."""

    store: DigitBoundStore
    args: tuple[Terms, ...] = ()
    """The terms a caller bound this function's parameters to, so the same
    binding can be replayed on a specialized copy of it."""
    by_expr: dict[Expr, Terms] = field(default_factory=dict)
    by_def: dict[Definition, Terms] = field(default_factory=dict)
    by_call: dict[Call, 'DigitBoundAnalysis'] = field(default_factory=dict)
    ret: Terms = field(default_factory=Terms)

    def escapes(self, e: Expr, exp: int | None, mag: int | None) -> bool:
        """Can *e* sit below digit position *exp*, or reach past ``2 ** (mag
        + 1)``?  ``None`` for either asks nothing of it.

        The question to ask before :meth:`bounds` is worth asking: a format
        contains what the store says only if the store's grid is no finer
        and its magnitude no larger, so a `True` here settles that nothing
        will be used -- in one satisfiability question where `bounds` costs
        three optimisations.
        """
        terms = self.by_expr.get(e)
        if terms is None or terms.msb is None or terms.lsb is None:
            return False
        # magnitude first: `reaches` stops at the first question that settles it
        asks: list[tuple[Term, int]] = []
        if mag is not None:
            asks.append((terms.msb, mag + 1))
        if exp is not None:
            asks.append((-terms.lsb, 1 - exp))
        return self.store.reaches(asks)

    def bounds(self, e: Expr) -> Bounds | None:
        """*e*'s precision, digit position and magnitude.

        ``None`` whenever the store leaves any of the three open.
        """
        terms = self.by_expr.get(e)
        if terms is None or terms.msb is None or terms.lsb is None:
            return None
        # One open value discards all three, so each is asked only once the
        # one before it came back closed -- every query costs a solve.
        prec = self.store.prec(terms.msb, terms.lsb)
        if not isinstance(prec, int):
            return None
        exp = -self.store.maximum(-terms.lsb)
        if not isinstance(exp, int):
            return None
        mag = self.store.maximum(terms.msb)
        if not isinstance(mag, int):
            return None
        # The three are independent maxima and need not be attained together,
        # so a precision can exceed what the other two leave room for.  Each is
        # sound alone, but only the narrowed precision names a real format.
        return Bounds(min(prec, mag - exp + 1), exp, mag)


class FormatView(Protocol):
    """What digit-bound inference reads from the format domain, and all it reads.

    Three readings of an inferred format, plus the structural analyses that
    pass already computed.  :class:`fpy2.analysis.FormatAnalysis` satisfies it
    structurally, so neither module imports the other.
    """

    type_info: TypeAnalysis
    ctx_use: ContextUseAnalysis
    array_size: ArraySizeAnalysis

    def logb_range(self, of: Expr | Definition) -> tuple[int | None, int | None]:
        """``logb``'s least and greatest value, where each is finite."""
        ...

    scopes: dict
    """Each context scope's resolved context, concrete or partial."""

    def has_finite(self, of: Expr | Definition) -> bool:
        """Does *of*'s format hold any finite value?"""
        ...

    def int_range(self, of: Expr | Definition) -> tuple[int, int] | None:
        """The least and greatest values, when only integers are held."""
        ...

    def int_value(self, of: Expr | Definition) -> int | None:
        """The value, when it is exactly one integer."""
        ...

    def view_of_call(self, e: Call) -> 'FormatView | None':
        """The view for *e*'s callee, or ``None`` when it was not analyzed."""
        ...


def _round_will_carry(rm: RoundingMode | None) -> bool:
    """Whether rounding under *rm* can carry out of the top binade.

    Only ``RTZ`` never increases a magnitude; every other mode carries on at
    least one sign, and this domain does not track sign.  An unresolved mode
    -- including a stochastic context, whose effective direction is not its
    ``rm`` -- has to be assumed to carry.

    ``RTO`` is not excepted: at precision zero the candidates are zero and one
    quantum, and *one* is odd.  See ``TestRoundCarry``.
    """
    return rm is not RoundingMode.RTZ


def _is_empty_alloc(d: Definition) -> bool:
    """Was *d* defined by `fp.empty(...)` -- holding no element to describe?"""
    return isinstance(d.site, Assign) and isinstance(d.site.expr, Empty)


def _all(ts: Iterable[Term | None]) -> list[Term] | None:
    """*ts* as a list, or `None` if any is missing."""
    out: list[Term] = []
    for t in ts:
        if t is None:
            return None
        out.append(t)
    return out


def _size(bound: ArraySizeBound) -> int | None:
    """*bound*'s length, when it is statically known."""
    return concrete_size(bound.size) if isinstance(bound, ListSize) else None


class _DigitBoundInferInstance(DefaultVisitor):
    """One function, instantiated at one call site's terms.

    Shares its caller's store, so a callee's rounding can cancel against a
    position the caller built.  The body is walked once: a term is per
    definition and the rules are structural, so another iteration would restate
    the same constraints.  A loop-carried value leaves a cyclic constraint,
    which the solver answers as unbounded.
    """

    def __init__(
        self,
        func: FuncDef,
        view: FormatView,
        store: DigitBoundStore,
        args: tuple[Terms, ...] = (),
    ):
        self.func = func
        self.view = view
        self.store = store
        self.args = args
        self.type_info = view.type_info
        self.def_use = view.type_info.def_use
        self.ctx_use = view.ctx_use
        self.array_size = view.array_size
        self.scopes = view.scopes
        self.out = DigitBoundAnalysis(store, args)
        self._loop_index: dict[Definition, int] = {}
        self._partial: dict[Definition, list[Terms] | None] = {}
        self._fields: dict[Definition, tuple[Terms | None, ...]] = {}
        self._elt_expr: dict[Definition, Expr] = {}
        self._covered_write: set[Definition] = set()
        self._elt_depth = 0
        self._elt_vars: set[int] = set()
        self._index_sets: dict[tuple, _IndexSet] = {}
        self._gather: tuple[tuple, Definition] | None = None
        self._gathered: set[Expr] = set()
        self._non_finite_path = 0
        self._seen_return = False
        self._vacuous_used: list[tuple[Term, list[Term]]] = []

    def analyze(self) -> DigitBoundAnalysis:
        # positional, so params that do not match the signature would bind
        # every later parameter to the term before it -- and `zip` would hide
        # it.  A caller passes one term per parameter or none at all.
        if self.args and len(self.args) != len(self.func.args):
            raise RuntimeError(
                f'digit-bound params for `{self.func.name}` have {len(self.args)} '
                f'terms for {len(self.func.args)} parameters'
            )
        for arg, src in zip(self.func.args, self.args):
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._share(d, src)
                # the caller minted these, and `_share` skips the seeding
                # `_fresh_interval` does -- but a list is a list in both
                # frames, so its summary describes an element here too
                self._mark_elt(d, src.msb, src.lsb, src.value)
        self._visit_block(self.func.body, None)
        self._check_vacuous()
        return self.out

    # -- terms ---------------------------------------------------------

    def _var(self, name: str) -> Term:
        """A fresh variable, per-element where the walk is inside a loop."""
        t = self.store.var(name)
        if self._elt_depth:
            self._elt_vars.update(v.index for v, _ in t.coeffs)
        return t

    def _mark_elt(self, of: Expr | Definition, *ts: Term | None) -> None:
        """Record *ts* as per-element where *of* is a list, whoever minted
        them: a list's summary describes an element wherever it came from."""
        if isinstance(self._type_of(of), ListType):
            self._elt_vars.update(
                v.index for t in ts if t is not None for v, _ in t.coeffs
            )

    def _fresh_interval(self, terms: Terms, of: Expr | Definition, tag: str) -> tuple[Term, Term]:
        """Give *terms* a bracketing pair, seeded from *of*'s inferred range.

        ``logb`` gets no lower seed: it bounds a magnitude from above, and a
        zero has no magnitude to bound.  Pinning it above the format's least
        non-zero value would make a zero's own reading contradictory.
        """
        lo, hi = self.view.logb_range(of)
        terms.msb = self._var(f'msb{tag}')
        terms.lsb = self._var(f'lsb{tag}')
        self._mark_elt(of, terms.msb, terms.lsb)
        if hi is not None:
            self.store.le(terms.msb, hi)
        if lo is not None:
            self.store.ge(terms.lsb, lo)
        return terms.msb, terms.lsb

    def _def(self, d: Definition) -> Terms:
        """*d*'s terms, minted on first reference.

        A definition, not its defining expression, is what uses share: a
        parameter has no defining expression, and two reads of one have to land
        on the same variable or nothing cancels.
        """
        terms = self.out.by_def.get(d)
        if terms is None:
            terms = self.out.by_def[d] = Terms()
            self._fresh_interval(terms, d, f'd{len(self.out.by_def)}')
        return terms

    def _share(self, d: Definition, src: Terms) -> None:
        """Adopt *src*'s terms for *d*, field by field, where *d* has none.

        States nothing.  The two end up naming the *same* store variables, so
        every constraint on either binds both -- which is what lets a callee's
        rounding cancel against a position its caller built, and is a stronger
        claim than `_join`'s one-directional bracket.

        A field *d* already has is kept and *src*'s dropped, leaving the two
        unrelated: an opportunity, not a guarantee.  That is why the lookup
        mints nothing -- `_fresh_interval` would seed a `logb` and the share
        could never fire -- and why anything reaching *d* first takes the
        opportunity away.
        """
        dst = self.out.by_def.setdefault(d, Terms())
        dst.msb = dst.msb if dst.msb is not None else src.msb
        dst.lsb = dst.lsb if dst.lsb is not None else src.lsb
        dst.value = dst.value if dst.value is not None else src.value

    def _join(self, d: Definition, srcs: list[Terms] | None) -> None:
        """Bracket *d* by every source at once, where `_share` hands over one.

        One source without a term leaves the join unstated: the bound is a
        claim about all of them.
        """
        if not srcs:
            return
        terms = self._def(d)
        msbs = [s.msb for s in srcs if s.msb is not None]
        lsbs = [s.lsb for s in srcs if s.lsb is not None]
        if terms.msb is not None and len(msbs) == len(srcs):
            self.store.le_max(terms.msb, msbs)
        if terms.lsb is not None and len(lsbs) == len(srcs):
            self.store.ge_min(terms.lsb, lsbs)

    def _type_of(self, of: Expr | Definition) -> Any:
        return (self.type_info.by_def if isinstance(of, Definition)
                else self.type_info.by_expr).get(of)  # type: ignore[arg-type]

    def _msb_of(self, e: Expr) -> Term | None:
        terms = self.out.by_expr.get(e)
        return terms.msb if terms is not None else None

    def _lsb_of(self, e: Expr) -> Term | None:
        terms = self.out.by_expr.get(e)
        return terms.lsb if terms is not None else None

    def _msbs(self, *es: Expr) -> list[Term] | None:
        """Every operand's `logb`, or `None` where any lacks one.

        All-or-nothing, as `_join` is: a bound over a strict subset of the
        operands is one the dropped operand never had to satisfy.  No
        operands at all is a list of none, which the store reads as no claim.
        """
        return _all(self._msb_of(e) for e in es)

    def _lsbs(self, *es: Expr) -> list[Term] | None:
        """Every operand's grid, or `None` where any lacks one; see
        :meth:`_msbs`."""
        return _all(self._lsb_of(e) for e in es)

    # -- emission ------------------------------------------------------

    def _visit_expr(self, e: Expr, ctx):
        super()._visit_expr(e, ctx)
        if not isinstance(self.type_info.by_expr.get(e), RealType | ListType):
            return
        terms = self.out.by_expr.setdefault(e, Terms())
        if terms.msb is not None:
            return

        # A variable reads what its definition wrote, and `xs[i]` over the
        # whole list *is* an element, so both share terms rather than
        # getting their own.
        src: Terms | None
        match e:
            case Var():
                src = self._def(self.def_use.find_def_from_use(e))
            case ListRef(value=Var() as lst, index=idx) if self._covers(
                idx, self._len_of(lst)
            ):
                src = self._def(self.def_use.find_def_from_use(lst))
            case ListRef(value=Var() as lst, index=Var() as idx) if (
                inst := self._at_index_set(lst, idx)
            ) is not None:
                # an element of a *part* of the list: the value comes too,
                # since what a gathered exponent goes on to build cancels
                # against it
                self._gathered.add(e)
                terms.msb, terms.lsb, terms.value = inst.msb, inst.lsb, inst.value
                return
            case ListComp():
                src = self.out.by_expr.get(e.elt)
            case _:
                src = None
        if src is not None:
            terms.msb, terms.lsb = src.msb, src.lsb
            return

        logb, grid = self._fresh_interval(terms, e, f'e{len(self.out.by_expr)}')
        self._emit_logb(e, self._ambient(e, logb, grid))
        self._emit_grid(e, grid)

    def _ambient(self, e: Expr, logb: Term, grid: Term) -> Term:
        """Account for the context *e* is evaluated under, and return the term
        the exact rules should bound.

        Every rule below states the reach of the *exact* result, but an
        expression denotes that result **rounded at the active context**, and
        rounding away from zero carries out of the binade the exact bound
        implies.  So where the context resolves and its mode can carry, the
        rules bound a fresh term and the expression reaches one further.

        The same context puts a floor under the grid: a rounded value is a
        multiple of the quantum, so it has no digit below it.  That is a
        tightening, and it is what pays for the carry.

        `Round` and `Cast` name that context as their own operation and model
        it themselves; applying it here too would count it twice.  A selection
        or a projection hands an operand back untouched -- it resolves a
        context without rounding at it -- so the floor would claim a quantum
        its result never sat on.
        """
        if isinstance(e, Round | Cast | Min | Max | AMin | AMax | Fst | Snd):
            return logb
        found = self._rounding(e)
        if found is None:
            return logb
        pos, rm = found
        self.store.ge(grid, pos + 1)
        if not _round_will_carry(rm):
            return logb
        exact = self._var(f'X{len(self.out.by_expr)}')
        self.store.le_max(logb, [exact + 1, pos + 1])
        return exact

    def _operands(self, e: Expr) -> tuple[Expr, ...]:
        """The operands *e*'s own bound is taken over, for the rules that
        just take a bound over all of them."""
        match e:
            case Neg() | Abs() | AMin() | AMax() | Sum():
                return (e.arg,)
            case Copysign():
                return (e.first,)   # magnitude from the first, sign from the second
            case Mod() | Fmod() | Remainder():
                return (e.second,)  # a remainder is smaller than its divisor
            case Add() | Sub():
                return (e.first, e.second)
            case IfExpr():
                return tuple(self._live_arms(e))
            case ListExpr():
                return tuple(e.elts)
            case Min() | Max():
                return tuple(e.args)
            case _:
                return ()

    def _emit_logb(self, e: Expr, t: Term) -> None:
        """Bound ``logb(e)`` from its operands'."""
        store = self.store
        match e:
            case (Neg() | Abs() | Copysign() | IfExpr() | ListExpr()
                  | Min() | Max() | AMin() | AMax()
                  | Mod() | Fmod() | Remainder()):
                if (ts := self._msbs(*self._operands(e))) is not None:
                    store.le_max(t, ts)
            case Mul():
                lhs, rhs = self._msb_of(e.first), self._msb_of(e.second)
                if lhs is not None and rhs is not None:
                    # Two mantissas in `[1, 2)` multiply below 4, hence the
                    # extra binade -- unless one is exactly 1, i.e. a power of
                    # two.
                    exact = self._exp2_arg(e.first) is not None \
                        or self._exp2_arg(e.second) is not None
                    store.le(t, lhs + rhs + (0 if exact else 1))
            case Add() | Sub() | Hypot() | Fdim():
                # `|a + b| <= 2 * max(|a|, |b|)`.  No alignment needed: in
                # exponent space addition loses only the *lower* bound.
                if (ts := self._msbs(e.first, e.second)) is not None:
                    store.le_max(t, [v + 1 for v in ts])
            case Trunc():
                # toward zero, so the binade cannot grow; a magnitude below
                # one lands at one
                if (ts := self._msbs(e.arg)) is not None:
                    store.le_max(t, [*ts, 0])
            case Floor() | Ceil() | RoundInt() | NearbyInt():
                # the others round *away* from zero on one sign or both, and
                # that carries out of the binade: `ceil(1.75) == 2`
                if (ts := self._msbs(e.arg)) is not None:
                    store.le_max(t, [v + 1 for v in ts] + [0])
            case Sum():
                # `n` terms below `2 ** (L + 1)` sum below `2 ** (L + 1 + ceil(log2 n))`
                elt, n = self._msb_of(e.arg), self._len_of(e.arg)
                if elt is not None and n is not None and n > 0:
                    store.le(t, elt + math.ceil(math.log2(n)))
            case Round() | Cast():
                src, found = self._msb_of(e.args[0]), self._rounding(e)
                if src is not None and found is not None:
                    pos, rm = found
                    if _round_will_carry(rm):
                        # A carrying mode reaches one binade further, and one
                        # quantum however coarse the grid is -- the larger of
                        # the two once the grid passes the value's own reach.
                        store.le_max(t, [src + 1, pos + 1])
                    else:
                        store.le(t, src)
            case Exp2() | Pow() if (k := self._exp2_arg(e)) is not None:
                store.le(t, k)

    def _emit_grid(self, e: Expr, g: Term) -> None:
        """Bound *e*'s least significant digit from its operands'."""
        store = self.store
        match e:
            # `Add`/`Sub` take the finer of the two, which aligned summands share
            case (Neg() | Abs() | Copysign() | IfExpr() | ListExpr()
                  | Add() | Sub() | Sum()):
                if (gs := self._lsbs(*self._operands(e))) is not None:
                    store.ge_min(g, gs)
            case Mul():
                lhs, rhs = self._lsb_of(e.first), self._lsb_of(e.second)
                if lhs is not None and rhs is not None:
                    store.ge(g, lhs + rhs)
            case Round() | Cast():
                # the result is a multiple of the quantum it rounded at
                found = self._rounding(e)
                if found is not None:
                    store.ge(g, found[0] + 1)
            case Exp2() | Pow() if (k := self._exp2_arg(e)) is not None:
                store.ge(g, k)   # one digit, at position `k`

    # -- the value channel ---------------------------------------------
    #
    # `logb` bridges the two sorts: the value of `logb(x)` is the variable
    # bounding `x`'s exponent, which is what lets a rounding position and the
    # value being rounded cancel.

    def value_of(self, e: Expr) -> Term | None:
        """*e*'s exact value as an affine term, for an expression used as a
        rounding position.  ``None`` when it is not affine over ``logb``\\ s and
        literals.  Memoized: computing it mints variables and states
        constraints.
        """
        terms = self.out.by_expr.setdefault(e, Terms())
        if terms.value is None:
            terms.value = self._value_uncached(e)
        return terms.value

    def _value_uncached(self, e: Expr) -> Term | None:
        store = self.store
        # A literal, a captured constant, and anything partial evaluation
        # folded all arrive the same way: as one integer.
        const = self.view.int_value(e)
        if const is not None:
            return Term((), const)
        match e:
            # A context expression is not part of the walk, so its literals and
            # variables may have no recorded term of their own.
            case Integer():
                return Term((), int(e.as_rational()))
            case Logb():
                t = self._msb_of(e.arg)
                of: Expr | Definition = e.arg
                if isinstance(e.arg, Var):
                    d = self.def_use.find_def_from_use(e.arg)
                    of = d
                    t = t if t is not None else self._def(d).msb
                # Reading a `logb` is what anchors the term absolutely, and is
                # where the argument may be taken as non-zero: a rounding
                # steered by `logb(0)` has no position at all.  A *precondition*
                # rather than a fact -- `logb(x)` is `msb(x)`, so this lands on
                # every path -- and `_usable` is where a program that hands the
                # zero path a position of its own is caught leaving it.
                lo, _ = self.view.logb_range(of)
                if t is not None and lo is not None:
                    store.ge(t, lo)
                return t
            case Neg():
                inner = self.value_of(e.arg)
                return None if inner is None else -inner
            case Add() | Sub():
                lhs, rhs = self.value_of(e.first), self.value_of(e.second)
                if lhs is None or rhs is None:
                    return None
                return lhs + rhs if isinstance(e, Add) else lhs - rhs
            case IfExpr():
                return self._merge_arms(
                    f'V{len(self.out.by_expr)}', e.cond,
                    self.value_of(e.ift), self.value_of(e.iff),
                )
            case AMax() | AMin() | Max() | Min():
                # `max(...) >= every operand` is the whole content; the value
                # itself is never needed.  An operand with no term contributes
                # none, and the ordering still holds for the rest.
                m = self._fresh_value(e)
                operands = [e.arg] if isinstance(e, AMax | AMin) else list(e.args)
                for t in (self.value_of(a) for a in operands):
                    if t is None:
                        continue
                    if isinstance(e, AMax | Max):
                        store.ge(m, t)
                    else:
                        store.le(m, t)
                return m
            case ListComp():
                return self.value_of(e.elt)
            case Var():
                # a list built by a loop has no defining expression at all
                return self._def_value(self.def_use.find_def_from_use(e))
            case _:
                # No rule, but a range still keeps a `max` over it bounded --
                # enough for a position built from it to materialize.
                return self._fresh_value(e) if self.view.int_range(e) else None

    def _live_arms(self, e: IfExpr) -> list[Expr]:
        """The arms of *e* that state a magnitude.

        A zero arm has no digits, so it neither raises the magnitude nor
        lowers the grid -- the same reading `_visit_branches` gives the
        statement form this lowers to.
        """
        return [a for a in (e.ift, e.iff) if self.view.int_value(a) != 0]

    def _merge_arms(
        self, tag: str, cond: Expr, ift: Term | None, iff: Term | None,
    ) -> Term | None:
        """A value that is whichever of two arms *cond* selects.

        Where *cond* tests something against zero, the `then` arm is reached
        only where that value *is* zero, and a zero's `logb` is below every
        bound.  So either `iff` is the answer, or nothing the subject's
        magnitude goes on to bound has to hold -- which is one disjunction,
        and is what ties a rounding position built from an exponent back to
        the value it came from.
        """
        if ift is None or iff is None:
            return None
        m = self._var(tag)
        # `m` is one of the two arms, which is what keeps it placed at all ...
        self.store.le_max(m, [ift, iff])
        self.store.ge_min(m, [ift, iff])
        # ... and, where the `then` arm carries no magnitude, `m` tracks the
        # other one.  Both hold; the second alone would leave `m` free below,
        # and an absolute position is as necessary as a relative one.
        vacuous = self._vacuous(cond)
        if vacuous is not None and self._usable(vacuous, ift):
            self.store.ge_min(m, [iff, *vacuous])
            self._vacuous_used.append((ift, vacuous))
        return m

    def _usable(self, vacuous: list[Term], ift: Term) -> bool:
        """Whether any vacuous disjunct can still sit at or below the `then` arm.

        Such a disjunct stands in for "the `then` arm carries no magnitude",
        and is worth stating only while the store admits it there.  It need
        not: `value_of(Logb)` floors a `logb`'s term at its argument's minimum
        exponent on the precondition that the argument is non-zero, and that
        lands on every path -- including one where the program itself supplies
        a position for the zero case.  Offering one then pushes the merge up to
        that floor, and a value rounded at the `then` arm's own position loses
        every digit below it.

        Skipping is sound: a disjunct only ever tightens, so dropping one costs
        precision.  Read off the store as it stands, which is why
        :meth:`_check_vacuous` re-asks once the walk is over.

        Why the check lives here and not at the floor.  `logb(x)` *is*
        `msb(x)` -- the read denotes no new quantity -- so "x is non-zero on
        the path through this read" has no term of its own to sit on, and a
        store with no notion of path states it about every path or not at all.
        Dropping it instead is not an option either: it is what anchors a
        position absolutely, and without it self-anchored rounding, anchors
        across a call, and the guarded-exponent shape all go (measured: the
        corpus still reaches 14/16, and 30 unit tests do not).  So the floor
        stays, stated too widely, and the one place its overreach is
        *detectable* -- a merge that hands the zero path a position of its own,
        which is the program saying the floor's premise does not hold there --
        refuses to build on it.  Making it localisable instead needs a
        path-sensitive store; see `docs/todos/digit-bound-inference.md`.
        """
        keep = self._floor(ift)
        return any(self._floor(t) <= keep for t in vacuous)

    def _floor(self, t: Term) -> int | float:
        """The least value *t* can take; `-inf` where nothing floors it."""
        return -self.store.maximum(-t)

    def _check_vacuous(self) -> None:
        """:meth:`_usable` reads the store mid-walk, so a `ge` stated *after* a
        merge could floor a disjunct that was free when it was taken -- and the
        constraint it justified is already in the store, where nothing can
        retract it.

        Every one of them on the corpus is admissible at both points, so this has
        never fired; it is here because the alternative to noticing is a
        silently over-narrow integer.  Deferring the whole emission instead
        costs a design, the mid-walk queries wanting the constraint.
        """
        for ift, vacuous in self._vacuous_used:
            if not self._usable(vacuous, ift):
                raise RuntimeError(
                    f'digit-bound: a vacuous disjunct of `{ift}` was floored '
                    f'after the merge that used it, in `{self.func.name}`'
                )

    def _vacuous(self, cond: Expr) -> list[Term] | None:
        """`logb`s that every path into the arm *cond* guards leaves free.

        The store satisfies a `ge_min` one way, so a disjunct naming a value
        only *one* path zeroes lets it drive the merge down while the value the
        bound is actually about stays high.  What is usable is a value zero on
        every path -- `sum(group) * scale` is, where neither the group nor the
        scale is.  It is all-or-nothing: a path zeroing nothing would leave the
        disjunction unsatisfiable, which reads as `-inf`.
        """
        paths = self._zero_paths(cond)
        if not paths:
            return None
        every = self._universal_zeros(cond)
        out: list[Term] = []
        seen: set[Term] = set()
        for e in list(self.out.by_expr):
            t = self._msb_of(e)
            if t is None or t in seen:
                continue
            if all(self._forced_zero(e, atoms, every) for atoms in paths):
                seen.add(t)
                out.append(t - 1)
        return out or None

    def _forced_zero(
        self, e: Expr, atoms: set[Term], every: set[Term],
    ) -> bool:
        """Is *e* zero wherever every term in *atoms* names a zero?

        Only what annihilates: a product with a zero factor, and a sum whose
        every summand is zero.

        *every* is the subset of *atoms* a test made about the whole list --
        `all(x == 0 for x in xs)` -- rather than about one element.  A list's
        term describes an arbitrary element either way, so the two are
        indistinguishable once in the store, but only the universal one makes
        a `Sum` zero: `xs[i] == 0` says nothing about the other summands.
        """
        t = self._msb_of(e)
        if t is not None and t in atoms:
            return True
        match e:
            case Mul():
                return (self._forced_zero(e.first, atoms, every)
                        or self._forced_zero(e.second, atoms, every))
            case Sum():
                # zero on *this* path, and zero for every element rather than
                # the one the guard tested
                inner = self._msb_of(e.arg)
                return inner is not None and inner in atoms and inner in every
            case Neg() | Abs():
                return self._forced_zero(e.arg, atoms, every)
            case _:
                return False

    def _universal_zeros(self, cond: Expr) -> set[Term]:
        """The list summaries *cond* zeroes for **every** element, via an
        `all(...)`.  A bare `xs[i] == 0` names the same term and is not here.
        """
        match cond:
            case Or() | And():
                out: set[Term] = set()
                for a in cond.args:
                    out |= self._universal_zeros(a)
                return out
            case AllOf():
                # `_zero_paths` already reads both shapes an `all` takes --
                # over a comprehension, and over the list a loop filled
                return {t for ts in (self._zero_paths(cond) or ()) for t in ts}
            case Var():
                return self._universal_zeros_of(
                    self.def_use.find_def_from_use(cond)
                )
        return set()

    def _universal_zeros_of(self, d: Definition) -> set[Term]:
        """:meth:`_universal_zeros`, reached through a definition."""
        if isinstance(d, PhiDef):
            if d.is_loop:
                return set()
            return (self._universal_zeros_of(self.def_use.defs[d.lhs])
                    | self._universal_zeros_of(self.def_use.defs[d.rhs]))
        return (self._universal_zeros(d.site.expr)
                if isinstance(d.site, Assign) else set())

    def _zero_paths(self, cond: Expr) -> list[set[Term]] | None:
        """One set of zeroed `logb`s per path making *cond* true.

        ``None`` where some path zeroes nothing.  Lowering leaves `a or b` as a
        phi over two conditions and `all(xs)` as a reduce over a list an inner
        loop filled, so both are read through.
        """
        match cond:
            case Compare(ops=(CompareOp.EQ,), args=(lhs, rhs)):
                for value, other in ((rhs, lhs), (lhs, rhs)):
                    if self.view.int_value(value) == 0:
                        t = self._msb_of(other)
                        return None if t is None else [{t}]
            case Or():
                # `And` is the dual and is not handled
                return self._zero_paths_all(cond.args)
            case AllOf(arg=ListComp() as comp):
                # every element true, so whatever the element tests is zero
                return self._zero_paths(comp.elt)
            case AllOf(arg=Var() as xs):
                # ... and the same after the comprehension became a loop
                elt = self._elt_expr.get(self.def_use.find_def_from_use(xs))
                return None if elt is None else self._zero_paths(elt)
            case Var():
                return self._zero_paths_of(self.def_use.find_def_from_use(cond))
        return None

    def _zero_paths_of(self, d: Definition) -> list[set[Term]] | None:
        """*d*'s paths, taking a merge as the paths that reach it."""
        if isinstance(d, PhiDef):
            if d.is_loop:
                return None     # carried, so not the `a or b` this reads
            return self._zero_paths_all(
                [self.def_use.defs[i] for i in (d.lhs, d.rhs)]
            )
        return self._zero_paths(d.site.expr) if isinstance(d.site, Assign) else None

    def _zero_paths_all(
        self, branches: Sequence[Expr] | Sequence[Definition]
    ) -> list[set[Term]] | None:
        """Every branch's paths together, or `None` if any branch has none."""
        found: list[set[Term]] = []
        for b in branches:
            part = (
                self._zero_paths(b) if isinstance(b, Expr)
                else self._zero_paths_of(b)
            )
            if part is None:
                return None
            found.extend(part)
        return found

    def _def_value(self, d: Definition) -> Term | None:
        """*d*'s value, looking through to whatever expression defines it."""
        const = self.view.int_value(d)
        if const is not None:
            return Term((), const)
        known = self.out.by_def.get(d)
        if known is not None and known.value is not None:
            return known.value
        return self.value_of(d.site.expr) if isinstance(d.site, Assign) else None

    def _visit_branches(self, stmt: If1Stmt | IfStmt, cond: Expr) -> None:
        """Merge what the two paths through *stmt* leave behind."""
        for phi in self.def_use.phis[stmt]:
            # `lhs` is the `then` arm of an `if`/`else`, but the *untaken*
            # path of a one-armed `if`, where the body is `rhs`.
            ift, iff = self.def_use.defs[phi.rhs], self.def_use.defs[phi.lhs]
            if isinstance(stmt, IfStmt):
                ift, iff = iff, ift
            # A zero arm has no digits, so it neither raises the magnitude nor
            # lowers the grid: it states nothing and is dropped rather than
            # joined, the way `_visit_return` drops a path that returns only
            # infinities.  Joining it instead leaves the merge at its own seed.
            live = [
                self.out.by_def.get(d)
                for d in (ift, iff) if self.view.int_value(d) != 0
            ]
            if live and all(t is not None for t in live):
                self._join(phi, live)  # type: ignore[arg-type]
            value = self._merge_arms(
                f'V{len(self.out.by_def)}', cond,
                self._def_value(ift), self._def_value(iff),
            )
            if value is not None:
                self.out.by_def.setdefault(phi, Terms()).value = value

    def _only_non_finite(self, cond: Expr, negated: bool) -> bool:
        """Does *cond*, read *negated* or not, hold only where some value is
        non-finite?
        """
        match cond:
            case Not():
                return self._only_non_finite(cond.arg, not negated)
            case IsFinite():
                return negated
            case IsInf() | IsNan():
                return not negated
            case Or() | And():
                # `or` needs every disjunct to say so where `and` needs only
                # one, and a negation swaps which: `not (a and b)` is
                # `not a or not b`.  A stronger hypothesis must not do worse.
                parts = [self._only_non_finite(a, negated) for a in cond.args]
                return any(parts) if isinstance(cond, And) != negated else all(parts)
            case Var():
                # lowering hoists a condition into a temporary, and a rule
                # that reads the syntax would silently switch off
                d = self.def_use.find_def_from_use(cond)
                return (isinstance(d.site, Assign)
                        and self._only_non_finite(d.site.expr, negated))
            case _:
                return False

    def _guarded(self, block: StmtBlock, cond: Expr, negated: bool, ctx) -> None:
        """Walk *block*, noting a path only a non-finite value reaches."""
        off = self._only_non_finite(cond, negated)
        self._non_finite_path += off
        self._visit_block(block, ctx)
        self._non_finite_path -= off

    def _visit_if1(self, stmt: If1Stmt, ctx):
        self._visit_expr(stmt.cond, ctx)
        self._guarded(stmt.body, stmt.cond, False, ctx)
        self._visit_branches(stmt, stmt.cond)

    def _visit_if(self, stmt: IfStmt, ctx):
        self._visit_expr(stmt.cond, ctx)
        self._guarded(stmt.ift, stmt.cond, False, ctx)
        self._guarded(stmt.iff, stmt.cond, True, ctx)
        self._visit_branches(stmt, stmt.cond)

    def _fresh_value(self, e: Expr) -> Term:
        """A value variable for *e*, bounded by the range its format gives."""
        v = self._var(f'V{len(self.out.by_expr)}')
        rng = self.view.int_range(e)
        if rng is not None:
            self.store.ge(v, rng[0])
            self.store.le(v, rng[1])
        return v

    def _exp2_arg(self, e: Expr) -> Term | None:
        """*e*'s exponent when it is a power of two.  Only base two: a general
        power is irrational, and nothing states it exactly."""
        match e:
            case Exp2():
                return self.value_of(e.arg)
            case Pow() if self.view.int_value(e.first) == 2:
                return self.value_of(e.second)
            case _:
                return None

    def _rounding(self, e: Expr) -> tuple[Term, RoundingMode | None] | None:
        """The digit position *e* rounds at, and the mode it rounds with.

        A *concrete* fixed-point context counts too: it is unbounded above, so
        what limits the result is the operand's own reach.  A rescaled program
        is entirely of that shape.

        ``None`` for an expression that uses no context at all -- a bare
        literal, say -- which `_ambient` asks about as readily as a rounding.
        """
        if e not in self.ctx_use.use_to_scope:
            return None
        scope = self.ctx_use.find_scope_from_use(e)
        ctx = self.scopes.get(scope, scope.ctx)
        pos: object
        rm: object
        if isinstance(ctx, MPFixedContext):
            pos = ctx.nmin
            # a stochastic context rounds away from zero whatever `rm` says,
            # so it reports no mode and is assumed to carry
            rm = None if ctx.is_stochastic() else ctx.rm
        elif isinstance(ctx, PartialContext) and ctx.cls is MPFixedContext and ctx.args:
            pos = ctx.args[0]
            # partial evaluation hands back foreign values wrapped
            arg_rm = ctx.args[1] if len(ctx.args) > 1 else None
            given: Any = next((v for k, v in ctx.kwargs if k == 'rm'), arg_rm)
            rm = unwrap_foreign(given)
        else:
            return None
        if not isinstance(rm, RoundingMode):
            rm = None
        if isinstance(pos, int):
            return Term((), pos), rm
        if isinstance(pos, Expr):
            n = self.value_of(pos)
            return None if n is None else (n, rm)
        return None

    # -- index sets ------------------------------------------------------
    #
    # A list has one set of terms -- its *summary* -- describing an arbitrary
    # element, so a fact stated about it is a claim about every element.  Two
    # comprehensions over the same list may share that summary, which is what
    # lets `max([logb(x) for x in xs])` bound a rounding in a later
    # `[... for x in xs]`: sharing makes the two the same variable, and the
    # relation cancels.
    #
    # Sound only while both range over the *same* elements, so every rule below
    # checks that, and a subset gets its own terms and no relation.
    #
    def _len_of(self, e: Expr) -> int | None:
        return _size(self.array_size.by_expr.get(e))

    def _covers(self, index: Expr, length: int | None) -> bool:
        """Does *index* run over every element of a list of *length*?

        True when it reads a loop variable this walk registered and the list is
        exactly as long as the loop has iterations.  That check separates an
        element access from an arbitrary index: ``xs[i]`` under
        ``for i in range(len(xs))`` visits all of ``xs``, and the same
        expression under ``range(16)`` visits some of it.
        """
        if not isinstance(index, Var) or length is None:
            return False
        return self._loop_index.get(self.def_use.find_def_from_use(index)) == length

    def _range_key(self, e: Expr) -> tuple | None:
        """An identity for a range, so two loops over the same one land on
        the same index set.  Structural: a literal by its value, a variable by
        its definition.
        """
        match e:
            case Range2():
                parts = (e.first, e.second, None)
            case Range3():
                parts = (e.first, e.second, e.third)
            case _:
                return None
        key: list = []
        for part in parts:
            const = None if part is None else self.view.int_value(part)
            if part is None:
                key.append(('c', 1))
            elif const is not None:
                key.append(('c', const))
            elif isinstance(part, Var):
                key.append(('d', self.def_use.find_def_from_use(part)))
            else:
                return None
        return tuple(key)

    def _at_index_set(self, lst: Var, idx: Var) -> Terms | None:
        """*lst*'s element summary restricted to the range the enclosing loop
        runs over, or ``None`` where there is no such range.

        ``for i in range(a, b, s)`` visits *part* of a list, so ``xs[i]`` is an
        element of that part; a bound the part goes on to build -- a ``max``
        over it -- says nothing about the rest, which is why sharing the
        whole list's variables here would be unsound.  The part gets variables
        of its own and the store replays onto them every fact that holds at
        every index.  Two parts are related exactly when the two loops run
        over the same range, which is what the key is for: one renaming per
        index set, so ``es`` and ``prods`` gathered over the evens keep the
        pairing they had.
        """
        if self._gather is None:
            return None
        key, index = self._gather
        if self.def_use.find_def_from_use(idx) is not index:
            return None
        d_lst = self.def_use.find_def_from_use(lst)
        src = self._def(d_lst)
        if src.value is None:
            # the part is what a rounding position is built from, and the
            # whole list's value is minted lazily, so force it here
            src.value = self._def_value(d_lst)
        fields = (src.msb, src.lsb, src.value)
        names = [v for t in fields if t is not None for v, _ in t.coeffs]
        if any(v.index not in self._elt_vars for v in names):
            # a variable this frame cannot vouch for as per-element would be
            # left shared rather than copied, which is the two-way claim
            return None
        inst = self._index_sets.get(key)
        if inst is None:
            inst = self._index_sets[key] = _IndexSet(f'@{len(self._index_sets)}')
        for v in names:
            if v.index not in inst.subst:
                inst.subst[v.index] = self.store.var(v.name + inst.tag)
        inst.mark = self.store.instance(
            self._elt_vars, inst.subst, inst.mark, inst.tag
        )
        return Terms(*(None if t is None else t.rename(inst.subst) for t in fields))

    def _carried(self, d: Definition) -> list[Terms] | None:
        """What bounds an arbitrary element of the list *d*, if anything does.

        A partial write leaves the rest of a list alone, so the reach of the
        writes that came before it is part of the answer.  An `empty` is
        bounded by nothing at all -- every element of one is uninitialized, so
        a read that gets there is undefined and no bound has to cover it.
        """
        if _is_empty_alloc(d):
            return []
        if d in self._partial:
            return self._partial[d]
        terms = self.out.by_def.get(d)
        return None if terms is None else [terms]

    def _bind_elt(self, target: NamedId, site, source: Var) -> None:
        d_src = self.def_use.find_def_from_use(source)
        self._share(self.def_use.find_def_from_site(target, site), self._def(d_src))

    def _bind_iter(self, target: Id | TupleBinding, iterable: Expr, site) -> None:
        """What iterating *iterable* as *target* binds, at *site*.

        Shared by the comprehension and the loop it lowers to, which is where
        the two forms of every rule here have drifted apart before.
        """
        if not isinstance(target, NamedId):
            return
        match iterable:
            case Var():
                # iterating a list directly binds the target to its element
                self._bind_elt(target, site, iterable)
            case Range1():
                # ... while `range(len(xs))` makes the target an index.  Only
                # `Range1`: a range with a start or a step does not visit
                # every index from zero, and covering is the claim that it
                # does ...
                n = self._len_of(iterable)
                if n is not None:
                    self._loop_index[self.def_use.find_def_from_site(target, site)] = n
            case Range2() | Range3():
                # ... it visits a *part*, which `_at_index_set` gives
                # variables of its own.
                key = self._range_key(iterable)
                if key is not None:
                    self._gather = (key, self.def_use.find_def_from_site(target, site))

    def _visit_list_comp(self, e: ListComp, ctx):
        outer = self._gather
        for target, iterable in zip(e.targets, e.iterables):
            self._visit_expr(iterable, ctx)
            # a `zip` has no counterpart in the lowered loop, so it stays here
            if isinstance(target, TupleBinding) and isinstance(iterable, Zip):
                for sub, part in zip(target.elts, iterable.args):
                    if isinstance(sub, NamedId) and isinstance(part, Var):
                        self._bind_elt(sub, e, part)
            else:
                self._bind_iter(target, iterable, e)
        self._elt_depth += 1
        self._visit_expr(e.elt, ctx)
        self._elt_depth -= 1
        self._gather = outer

    def _visit_for(self, stmt: ForStmt, ctx):
        self._visit_expr(stmt.iterable, ctx)
        outer = self._gather
        self._bind_iter(stmt.target, stmt.iterable, stmt)
        # A list the body fills enters the loop holding whatever it already
        # does, which a partial write leaves in place.
        for phi in self.def_use.phis[stmt]:
            self._partial[phi] = self._carried(self.def_use.defs[phi.lhs])
        self._elt_depth += 1
        self._visit_block(stmt.body, ctx)
        self._elt_depth -= 1
        self._gather = outer
        for phi in self.def_use.phis[stmt]:
            body_def = self.def_use.defs[phi.rhs]
            body = self.out.by_def.get(body_def)
            pre = self.out.by_def.get(self.def_use.defs[phi.lhs])
            if body is None:
                self._join(phi, self._partial.get(body_def))
            elif (_is_empty_alloc(self.def_use.defs[phi.lhs])
                  or body_def in self._covered_write):
                # Every iteration writes the same expression, so the body's
                # terms describe the loop's element too.  Zero trips is
                # vacuous in both these cases and only these: an `empty` has
                # no element to describe, and a *covering* write runs as many
                # times as the list is long, so no trips means no elements.
                self._share(phi, body)
            elif pre is not None:
                # Otherwise the phi is what an iteration *starts* from -- the
                # pre-loop value on the first pass, and on every pass if the
                # loop never runs -- so only the join covers both.  A list
                # that arrived with elements keeps them.
                self._join(phi, [pre, body])
            self._partial[phi] = self._partial.get(body_def)
            if body_def in self._fields:
                self._fields[phi] = self._fields[body_def]
            if body_def in self._elt_expr:
                self._elt_expr[phi] = self._elt_expr[body_def]

    def _visit_assign(self, stmt: Assign, ctx):
        self._visit_expr(stmt.expr, ctx)
        if isinstance(stmt.target, TupleBinding):
            # Unpacking an element of a materialized `zip` is what puts the
            # names back in touch with the lists they came from -- comprehension
            # elimination leaves exactly this behind.
            self._unpack_elt(stmt, stmt.target)
            return
        if not isinstance(stmt.target, NamedId):
            return
        d = self.def_use.find_def_from_site(stmt.target, stmt)
        match stmt.expr:
            case ListRef(value=Var() as lst, index=Var() as idx) if self._covers(
                idx, self._len_of(lst)
            ):
                d_src = self.def_use.find_def_from_use(lst)
                if isinstance(d_src.site, Assign) and isinstance(d_src.site.expr, Range1):
                    # `range(n)[i] == i`, so indexing a materialized range with
                    # a full index yields another one.  Comprehension
                    # elimination emits exactly this for a `zip`.
                    self._loop_index[d] = self._loop_index[
                        self.def_use.find_def_from_use(idx)
                    ]
                else:
                    self._share(d, self._def(d_src))
            case _:
                src = self.out.by_expr.get(stmt.expr)
                if src is None:
                    return
                terms = self._def(d)
                # One-directional, not an equality.  `_def` seeds this
                # definition from *its* format, which branch refinement
                # narrows to the path the assignment sits on -- and the
                # expression's terms are shared by every use of it, so an
                # equality would state that narrower bound everywhere.  The
                # useful direction survives: the definition reaches no
                # further than the expression, and is no finer.
                if terms.msb is not None and src.msb is not None:
                    self.store.le(terms.msb, src.msb)
                if terms.lsb is not None and src.lsb is not None:
                    self.store.ge(terms.lsb, src.lsb)
                terms.value = terms.value if terms.value is not None else src.value

    def _zip_fields(self, d: Definition) -> tuple[Terms | None, ...] | None:
        """A `zip`'s element field by field, each an element of one operand.

        The same list reaches the walk both as a `zip` and, once comprehension
        elimination has run, as one materialized by a loop.
        """
        if not isinstance(d.site, Assign) or not isinstance(d.site.expr, Zip):
            return None
        return tuple(
            self._def(self.def_use.find_def_from_use(a)) if isinstance(a, Var) else None
            for a in d.site.expr.args
        )

    def _unpack_elt(self, stmt: Assign, target: TupleBinding) -> None:
        match stmt.expr:
            case ListRef(value=Var() as lst, index=Var() as idx) if self._covers(
                idx, self._len_of(lst)
            ):
                d_lst = self.def_use.find_def_from_use(lst)
            case _:
                return
        fields = self._fields.get(d_lst) or self._zip_fields(d_lst)
        if fields is None:
            return
        for sub, part in zip(target.elts, fields):
            if isinstance(sub, NamedId) and part is not None:
                self._share(self.def_use.find_def_from_site(sub, stmt), part)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        d_in = self.def_use.find_def_from_use(stmt)
        d = self.def_use.find_def_from_site(stmt.var, stmt)
        src = self.out.by_expr.get(stmt.expr)
        length = _size(self.array_size.by_def.get(d_in))
        covers = len(stmt.indices) == 1 and self._covers(stmt.indices[0], length)
        # A gather covers too: every element it writes is an element of the
        # source at an index in the loop's range, and one it never writes is
        # uninitialized, so no bound has to reach it.
        gathers = stmt.expr in self._gathered and self._carried(d_in) == []
        if not covers and not gathers:
            prior = self._carried(d_in)
            self._partial[d] = None if src is None or prior is None else [*prior, src]
            return
        if src is not None:
            self._share(d, src)
        self._covered_write.add(d)
        self._elt_expr[d] = stmt.expr
        if isinstance(stmt.expr, TupleExpr):
            # A list of tuples has no element term of its own, so the fields
            # are kept beside it for whatever destructures them.
            self._fields[d] = tuple(
                self.out.by_expr.get(el) for el in stmt.expr.elts
            )
        # A list of exponents needs its element's *value*, not its `logb`: that
        # is what a rounding position built from it cancels against.
        val = self.value_of(stmt.expr)
        if val is not None:
            self.out.by_def.setdefault(d, Terms()).value = val

    def _visit_return(self, stmt: ReturnStmt, ctx):
        self._visit_expr(stmt.expr, ctx)
        # A caller can use the result as a rounding position, and `value_of`
        # is lazy, so the value has to be named before the walk leaves.
        self.value_of(stmt.expr)
        # A path returning nothing but infinities and NaNs states no
        # magnitude: `logb` bounds the finite values, and this one has none.
        # Nor does a path only a non-finite *argument* reaches, whatever it
        # returns -- no assignment describes that call, so a sentinel returned
        # there must not join the exponent the other return states.
        if not self.view.has_finite(stmt.expr) or self._non_finite_path:
            return
        now = self.out.by_expr.get(stmt.expr) or Terms()
        if not self._seen_return:
            self._seen_return = True
            self.out.ret = Terms(now.msb, now.lsb, now.value)
            return
        self.out.ret = self._merge_returns(self.out.ret, now)

    def _merge_returns(self, prev: Terms, now: Terms) -> Terms:
        """Two returns describe two values, so the result reaches as far as
        the wider of them and is no finer than the coarser.

        An exact value survives only where both agree: a join of two values is
        not one.
        """
        value = prev.value if prev.value is not None and prev.value == now.value else None
        merged = Terms(value=value)
        if prev.msb is not None and now.msb is not None:
            merged.msb = self.store.var(f'msbR{len(self.out.by_expr)}')
            self.store.le_max(merged.msb, [prev.msb, now.msb])
        if prev.lsb is not None and now.lsb is not None:
            merged.lsb = self.store.var(f'lsbR{len(self.out.by_expr)}')
            self.store.ge_min(merged.lsb, [prev.lsb, now.lsb])
        return merged

    def _visit_call(self, e: Call, ctx):
        for arg in e.args:
            self._visit_expr(arg, ctx)
        for _, kwarg in e.kwargs:
            self._visit_expr(kwarg, ctx)
        view = self.view.view_of_call(e)
        if not isinstance(e.fn, Function) or view is None:
            return
        # The callee shares this store, so its rounding can cancel against a
        # position built here.  A rounding position is an argument like any
        # other, so `value_of` here is what threads one across the edge.
        args = tuple(
            Terms(self._msb_of(a), self._lsb_of(a), self.value_of(a))
            for a in e.args
        )
        before = self.store.n_vars
        sub = _DigitBoundInferInstance(e.fn.ast, view, self.store, args).analyze()
        if self._elt_depth:
            # called once per element, so everything it minted is per-element
            self._elt_vars.update(range(before, self.store.n_vars))
        self.out.by_call[e] = sub
        if sub.ret.msb is not None or sub.ret.value is not None:
            # The result's terms are the callee's, so a bound this site knows
            # and the callee does not -- a branch refined away here -- is
            # stated on them.  Only on a term the callee *minted*, though: a
            # callee that returns an argument verbatim hands back the
            # caller's own variable, shared by every use of that name, and
            # this site's bound may hold only on the path reaching the call.
            passed = {a.msb for a in args if a.msb is not None}
            _, hi = self.view.logb_range(e)
            if (sub.ret.msb is not None and hi is not None
                    and sub.ret.msb not in passed):
                self.store.le(sub.ret.msb, hi)
            self.out.by_expr[e] = Terms(sub.ret.msb, sub.ret.lsb, sub.ret.value)


class DigitBoundInfer:
    """Digit-bound inference.

    Runs after format inference, which supplies the ranges it seeds from, and
    before the pass that consumes the precisions it derives.
    """

    @staticmethod
    def analyze(
        func: FuncDef,
        view: FormatView,
        params: DigitBoundParams | None = None,
    ) -> DigitBoundAnalysis:
        """Infer digit-bound relations for *func*, seeded from *view*.

        *params* continues a caller's constraint system instead of starting a
        fresh one, binding *func*'s parameters to the terms the caller passed.
        A relation between two arguments lives in that system and reduces to no
        per-argument format, so replaying it is the only way a separately
        analyzed copy of *func* can see it.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')
        if params is None:
            params = DigitBoundParams(DigitBoundStore(), ())
        return _DigitBoundInferInstance(
            func, view, params.store, params.args
        ).analyze()
