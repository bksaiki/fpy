"""
Observed values against the bounds format inference gives them.

A site is a name an assignment binds, on one call path (`tracing`).  Its
observed values abstract, as a set, to the least abstract format containing
them, `(max prec, min lsb, signed extremes, specials)` -- the maximally
precise result of Taneja et al. (CGO '20) -- so any gap to the static bound is
the analysis's.

Static bounds come from the compiler's view (`compiler_view`): the program
specialized per call site, each spec analyzed with its caller's digit bounds.
"""

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from tracing import tracer

import fpy2 as fp
from fpy2.analysis import FormatInfer
from fpy2.analysis.format_infer import (
    AbstractFormat,
    FormatAnalysis,
    FormatBound,
    ListFormat,
    SetFormat,
    TupleFormat,
    to_abstract,
)
from fpy2.analysis.format_infer.analysis import NEG_ZERO, SetValue, Special
from fpy2.analysis.value_class import ValueClassInfer
from fpy2.ast.fpyast import Assign, Call, FuncDef, TupleBinding
from fpy2.ast.visitor import DefaultVisitor
from fpy2.number import Float, RealFloat
from fpy2.number.format import REAL_FORMAT, Format
from fpy2.strategies import monomorphize
from fpy2.transform.specialize import Specialize
from fpy2.types import Type
from fpy2.utils import Location, NamedId

Site = tuple[tuple[Call, ...], Assign, NamedId]
"""A name an assignment binds, and the calls the assignment is reached through."""

Key = tuple[tuple[Location | None, ...], Location | None, str]
"""A site by source location, the same in every specialization of a program."""

Static = Format | SetFormat
"""A site's static bound on each of its numbers: `REAL_FORMAT` is top."""

_DISTINCT_CAP = 4096
"""Distinct values kept per site, to check them against a `SetFormat`."""


@dataclass
class Observed:
    """The join of every number a site took, each element of a list alone."""

    runs: int = 0
    """Times the site was assigned."""
    prec: int = 0
    """Most significant digits of a finite nonzero value."""
    lsb: int | None = None
    """Lowest nonzero digit position of any finite nonzero value."""
    msb: int | None = None
    """Highest digit position of any finite nonzero value."""
    pos: Fraction | None = None
    """Largest positive finite value."""
    neg: Fraction | None = None
    """Smallest negative finite value."""
    pos_inf: bool = False
    neg_inf: bool = False
    nan: bool = False
    neg_zero: bool = False
    nondyadic: bool = False
    """Some value had no finite binary expansion."""
    distinct: set[SetValue] | None = field(default_factory=set)
    """Every value, until there are more than `_DISTINCT_CAP`."""

    def add(self, v: Any) -> None:
        """Joins *v*'s numbers in: each leaf of a list; anything not a number
        (booleans, contexts, unset elements) is skipped."""
        if isinstance(v, list):
            for x in v:
                self.add(x)
            return
        if isinstance(v, Fraction):
            if v.denominator & (v.denominator - 1):
                self.nondyadic = True
                self._keep(v)
                return
            v = Float.from_rational(v)
        if not isinstance(v, Float):
            return
        if v.isnan:
            self.nan = True
            self._keep(Special.NAN)
        elif v.isinf:
            self.neg_inf |= v.s
            self.pos_inf |= not v.s
            self._keep(Special.NEG_INF if v.s else Special.POS_INF)
        elif v.is_zero():
            self.neg_zero |= v.s
            self._keep(NEG_ZERO if v.s else Fraction(0))
        else:
            tz = (v.c & -v.c).bit_length() - 1
            self.prec = max(self.prec, v.c.bit_length() - tz)
            self.lsb = v.exp + tz if self.lsb is None else min(self.lsb, v.exp + tz)
            self.msb = v.e if self.msb is None else max(self.msb, v.e)
            q = v.as_rational()
            if q > 0:
                self.pos = q if self.pos is None else max(self.pos, q)
            else:
                self.neg = q if self.neg is None else min(self.neg, q)
            self._keep(q)

    def _keep(self, x: SetValue) -> None:
        if self.distinct is not None:
            self.distinct.add(x)
            if len(self.distinct) > _DISTINCT_CAP:
                self.distinct = None


@dataclass(frozen=True)
class SiteResult:
    """A site's static bound against its observed values.  Slacks are in
    bits, `None` where no finite nonzero value was observed or the bound has
    no abstract format; 0 is tight."""

    top: bool
    executed: bool
    sound: bool
    digits: int | float | None
    lsb: int | float | None
    msb: int | float | None


class _Assigns(DefaultVisitor):
    """The assignments in one function's body, not its callees'."""

    def __init__(self) -> None:
        self.out: list[Assign] = []

    def _visit_assign(self, stmt: Assign, ctx: Any) -> None:
        self.out.append(stmt)
        super()._visit_assign(stmt, ctx)


def _leaf(b: FormatBound) -> Static | None:
    """The bound on each number of a value bounded by *b*, or `None` where
    its values are not numbers."""
    if isinstance(b, ListFormat):
        return _leaf(b.elt)
    return b if isinstance(b, (Format, SetFormat)) else None


def _bind(target: Any, b: FormatBound, v: Any = None) -> Iterator[tuple[NamedId, FormatBound, Any]]:
    """Each name *target* binds, with its part of bound *b* and value *v*."""
    if isinstance(target, TupleBinding):
        bs = b.elts if isinstance(b, TupleFormat) else (None,) * len(target.elts)
        vs = v if isinstance(v, tuple) else (None,) * len(target.elts)
        for t, bi, vi in zip(target.elts, bs, vs, strict=True):
            yield from _bind(t, bi, vi)
    elif isinstance(target, NamedId):
        yield target, b, v


def key(site: Site) -> Key:
    calls, stmt, name = site
    return tuple(c.loc for c in calls), stmt.loc, str(name)


def compiler_view(
    fn: fp.Function, arg_types: list[Type], digit_bounds: bool,
) -> tuple[fp.Function, dict[FuncDef, FormatAnalysis]]:
    """*fn* specialized per call site as the compiler does (`Specialize`),
    and each spec's analysis, with its caller's digit bounds where
    *digit_bounds*."""
    mod = fp.Module()
    mod.add(monomorphize(fn, args=arg_types))
    params: dict = {}
    # `size_key`: as the compiler does, so a callee knows its lists' lengths
    out = Specialize.apply(
        mod, size_key=True, bound_params=params, use_digit_bounds=digit_bounds)
    analyses = {
        f.ast: FormatInfer.analyze(
            f.ast, use_digit_bounds=digit_bounds, digit_bound_params=params.get(f.name),
            value_classes=ValueClassInfer.analyze(f.ast),
        )
        for f in out.functions()
    }
    return out.public()[0], analyses


def statics(
    entry: fp.Function, analyses: dict[FuncDef, FormatAnalysis], call: tuple[Call, ...] = (),
) -> dict[Site, Static]:
    """Each numeric site of *entry*, and of every function it calls, with its
    static bound from that function's analysis in *analyses*."""
    analysis = analyses[entry.ast]
    v = _Assigns()
    v._visit_function(entry.ast, None)
    out: dict[Site, Static] = {}
    for stmt in v.out:
        for name, b, _ in _bind(stmt.target, analysis.by_expr.get(stmt.expr)):
            if (leaf := _leaf(b)) is not None:
                out[(call, stmt, name)] = leaf
    for c in analysis.by_call:
        out |= statics(c.fn, analyses, (*call, c))
    return out


def observe(fn: fp.Function, inputs: Iterable[Sequence[Any]], sites: Iterable[Site]) -> dict[Site, Observed]:
    """The join of each site's values over a run of *fn* on each input."""
    obs = {s: Observed() for s in sites}
    by_rhs: dict[tuple[tuple[Call, ...], Any], Assign] = {(c, a.expr): a for c, a, _ in obs}

    def record(site: tuple[tuple[Call, ...], Any], v: Any) -> None:
        stmt = by_rhs.get(site)
        if stmt is None:
            return
        for name, _, vi in _bind(stmt.target, None, v):
            if (o := obs.get((site[0], stmt, name))) is not None:
                o.runs += 1
                o.add(vi)

    run = tracer(fn, record)
    for args in inputs:
        run(*args)
    return obs


def _rat(b: RealFloat | float) -> Fraction | float:
    return b if isinstance(b, float) else b.as_rational()


def _within(a: AbstractFormat, o: Observed) -> bool:
    """Whether every observed value lies in *a*: a finite `m * 2^e` does
    exactly when its precision is at most `a.prec`, its lowest digit at least
    `a.exp`, and it lies between the bounds."""
    finite = o.lsb is None or (
        o.prec <= a.prec and o.lsb >= a.exp
        and (o.pos is None or o.pos <= _rat(a.pos_bound))
        and (o.neg is None or o.neg >= _rat(a.neg_bound))
    )
    return (finite and not o.nondyadic
            and (a.has_pos_inf or not o.pos_inf) and (a.has_neg_inf or not o.neg_inf)
            and (a.has_nan or not o.nan) and (a.has_neg_zero or not o.neg_zero))


def _holds(static: Static, o: Observed) -> bool:
    if isinstance(static, SetFormat):
        if o.distinct is None:
            if len(static.values) > _DISTINCT_CAP:
                raise ValueError('too many distinct values to check against the set')
            return False
        return o.distinct <= static.values
    a = to_abstract(static)
    return a is not None and _within(a, o)


def compare(static: Static, o: Observed) -> SiteResult:
    """*static* against what *o* observed."""
    top = static == REAL_FORMAT
    sound = top or _holds(static, o)
    a = None if top else to_abstract(static)
    if a is None or o.lsb is None or o.msb is None:
        return SiteResult(top, o.runs > 0, sound, None, None, None)
    hi = a.bound
    return SiteResult(
        top, o.runs > 0, sound,
        digits=a.effective_prec() - o.prec,
        lsb=o.lsb - a.exp,
        msb=(math.inf if isinstance(hi, float) else hi.e) - o.msb,
    )
