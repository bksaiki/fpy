"""
Path-sensitive value-class analysis.

One question per expression: can this value be a NaN, an infinity of either
sign, a zero, or a finite non-zero?  The five atoms form a finite lattice —
union is the join, intersection the meet, so no widening is needed — and it is
*refined* at every branch that tests a value's class.

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

A *list* carries a class for its elements, read by an element access, a ``for``
target over it, and ``min``/``max`` over it.  It comes from the stores that build
the list and from a reduction over it (:meth:`_implied_elements`).  A tuple
carries nothing.

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

from ..ast.fpyast import *
from ..ast.visitor import DefaultVisitor
from ..number import REAL, Context, Float
from ..types import RealType, Type
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
    'ValueClass',
    'ValueClassAnalysis',
    'ValueClassInfer',
    'class_of',
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

_POW_INF = _POS_INF | _ZERO | _FINITE
"""``b ** (+-inf)`` for a positive literal ``b``: ``+inf`` when ``b > 1``, ``0``
when ``b < 1``, ``1`` when ``b`` is ``1``.  The literal is not inspected, so all
three stand; never ``-inf``, since a positive base has no negative power."""

_POW_POS_BASE = {
    _NAN: _NAN,
    _POS_INF: _POW_INF, _NEG_INF: _POW_INF,
    _ZERO: _FINITE, _FINITE: _FINITE,
}
"""``b ** y`` for a positive constant ``b``: ``b ** 0`` is ``1``."""


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


def _exact_select(args: list[ValueClass], is_max: bool) -> ValueClass:
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


def _shared_and_mutated(func: FuncDef) -> set[NamedId]:
    """List names whose elements cannot be tracked.

    An element class is a property of the *object*, but definitions are per
    name, and an FPy list is a reference: ``ys = xs`` makes one object with two
    names, so ``ys[0] = e`` changes what ``xs[0]`` reads while leaving ``xs``'s
    definition -- and anything recorded against it -- untouched.

    Neither half alone is a problem.  Aliasing without mutation is what
    `CompToLoop` emits when it binds an iterable; mutation without aliasing goes
    through the one name, which the per-definition join follows.  A name is
    dropped when its copy group holds a store, or when it reaches somewhere a
    store could happen unseen -- an argument of a call, a cell of an aggregate,
    or a call's result, which may alias anything the callee held.

    `Alias` is the real answer for the first half, but its ``written_regions``
    is intraprocedural, so a callee storing through a list it was handed does
    not appear; that needs escape summaries, which a standalone caller cannot
    build.
    """
    group: dict[NamedId, set[NamedId]] = {}
    stored: set[NamedId] = set()
    escaped: set[NamedId] = set()

    def join(a: NamedId, b: NamedId):
        merged = group.setdefault(a, {a}) | group.setdefault(b, {b})
        for n in merged:
            group[n] = merged

    def escape(e: Expr):
        if isinstance(e, Var):
            escaped.add(e.name)

    class _Scan(DefaultVisitor):
        def _visit_assign(self, stmt: Assign, ctx):
            if isinstance(stmt.target, NamedId):
                if isinstance(stmt.expr, Var):
                    join(stmt.target, stmt.expr.name)
                elif isinstance(stmt.expr, Call):
                    escaped.add(stmt.target)
            super()._visit_assign(stmt, ctx)

        def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
            if isinstance(stmt.var, NamedId):
                stored.add(stmt.var)
            escape(stmt.expr)           # `outer[i] = xs` shares `xs`'s cells
            super()._visit_indexed_assign(stmt, ctx)

        def _visit_call(self, e: Call, ctx):
            for arg in e.args:
                escape(arg)
            super()._visit_call(e, ctx)

        def _visit_list_expr(self, e: ListExpr, ctx):
            for elt in e.elts:
                escape(elt)
            super()._visit_list_expr(e, ctx)

        def _visit_tuple_expr(self, e: TupleExpr, ctx):
            for elt in e.elts:
                escape(elt)
            super()._visit_tuple_expr(e, ctx)

    _Scan()._visit_function(func, None)
    out = set(escaped)
    for g in group.values():
        if len(g) > 1 and (g & stored or g & escaped):
            out |= g
    return out


def _positive_literal(e: Expr) -> bool:
    return isinstance(e, RationalVal) and e.as_rational() > 0


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

    elt_by_def: dict[Definition, ValueClass]
    """Class of every *element* of each list definition, unrefined.  The list
    analogue of :attr:`by_def`; absent means nothing is known."""

    elt_by_expr: dict[Expr, ValueClass]
    """Class of every element of the list each expression names, refined by the
    branches dominating it.  The list analogue of :attr:`by_expr`."""

    by_def: dict[Definition, ValueClass | None]
    """Class of each variable definition, *unrefined* -- the class the defining
    expression had, joined across incoming edges at a phi.  A consumer wants
    :attr:`by_expr`, which is where a branch's refinement shows up; this is the
    per-definition view the other analyses expose, and what
    ``tests/infra/analysis/value_class.py`` dumps."""

    type_info: TypeAnalysis
    """Underlying basic-type analysis, which decides what carries a class."""

    ctx_use: ContextUseAnalysis
    """Underlying context-use analysis, which supplies each operation's context."""

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

    def classify_elements(self, e: Expr) -> ValueClass:
        """The class every element of the list *e* names belongs to, or the top
        class where nothing is known."""
        return self.elt_by_expr.get(e, _TOP)


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
    elt_by_def: dict[Definition, ValueClass]
    elt_by_expr: dict[Expr, ValueClass]

    _refine: dict[Definition, ValueClass]
    _refine_elt: dict[Definition, ValueClass]

    _shared: set[NamedId]
    """Lists no element class may be recorded against; see
    :func:`_shared_and_mutated`."""
    """Per-definition mask the enclosing branches imply, intersected into every
    read of that definition.  Saved and restored around each arm."""

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
        self.elt_by_def = {}
        self.elt_by_expr = {}
        self._refine = {}
        self._refine_elt = {}
        self._shared = set()

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.type_info.def_use

    def analyze(self) -> ValueClassAnalysis:
        self._shared = _shared_and_mutated(self.func)
        self._visit_function(self.func, None)
        return ValueClassAnalysis(
            func=self.func,
            by_expr=self.by_expr,
            by_def=self.by_def,
            elt_by_def=self.elt_by_def,
            elt_by_expr=self.elt_by_expr,
            type_info=self.type_info,
            ctx_use=self.ctx_use,
        )

    # ------------------------------------------------------------------
    # Definitions

    def _set_def(self, d: Definition, cls: ValueClass | None):
        if not isinstance(self.type_info.by_def.get(d), RealType):
            cls = None
        self.by_def[d] = cls

    def _def_class(self, d: Definition) -> ValueClass:
        cls = self.by_def.get(d)
        return cls if isinstance(cls, ValueClass) else _TOP

    def _elt_class(self, e: Expr) -> ValueClass:
        """The class every element of the list *e* names belongs to.

        The list analogue of :meth:`_visit_var`: what the definition holds, met
        with what the enclosing branches proved.  A list that is not a plain
        name carries no definition, so it is the top class.
        """
        if not isinstance(e, Var) or e.name in self._shared:
            return _TOP
        d = self.def_use.find_def_from_use(e)
        stored = self.elt_by_def.get(d, _TOP)
        out = stored & self._refine_elt.get(d, _TOP)
        self.elt_by_expr[e] = out
        return out

    def _set_elt(self, d: Definition, cls: ValueClass):
        if d.name not in self._shared:
            self.elt_by_def[d] = cls

    def _join_elt_phi(self, phi: Definition, lhs: Definition, rhs: Definition):
        """A phi's element class, where either arm has one.

        An absent arm joins as the top, or a store on one path would look like
        a promise about the other.
        """
        if lhs not in self.elt_by_def and rhs not in self.elt_by_def:
            return
        self._set_elt(
            phi,
            self.elt_by_def.get(lhs, _TOP) | self.elt_by_def.get(rhs, _TOP),
        )

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
            lhs, rhs = self.def_use.defs[phi.lhs], self.def_use.defs[phi.rhs]
            self._set_def(phi, self._def_class(lhs) | self._def_class(rhs))
            self._join_elt_phi(phi, lhs, rhs)

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
        out_elt = dict(saved_elt)
        for d, cls in self._implied_elements(cond, truth):
            out_elt[d] = out_elt.get(d, _TOP) & cls
        self._refine, self._refine_elt = out, out_elt
        try:
            yield
        finally:
            self._refine, self._refine_elt = saved, saved_elt

    def _implied_elements(
        self, cond: Expr, truth: bool,
    ) -> list[tuple[Definition, ValueClass]]:
        """What *cond* being *truth* says about the *elements* of a list.

        One shape: ``any``/``all`` over a comprehension, as `CompToLoop` leaves
        it::

            acc = False                 # `True` for `all`
            for x in xs:
                b = <pred>(x)
                acc = acc or b          # `and` for `all`

        ``not acc`` after the ``or`` form, and ``acc`` after the ``and`` form,
        both say the predicate's verdict holds for *every* element.  FPy has no
        ``break``, so the loop runs the whole iterable and the quantifier is
        universal.

        The match is strict -- a literal seed, a step that is exactly
        ``acc <op> b`` naming this phi, and a refinement the loop target itself
        carries -- and anything else returns nothing.  A store into the list
        needs no check: it is a new definition, and a refinement recorded
        against the old one does not reach it.
        """
        if not isinstance(cond, Var):
            return []
        d = self.def_use.find_def_from_use(cond)
        if not (isinstance(d, PhiDef) and d.is_loop):
            return []
        loop = d.site
        if not isinstance(loop, ForStmt) or not isinstance(loop.iterable, Var):
            return []
        if not isinstance(loop.target, NamedId):
            return []

        seed = self._assigned_expr(self.def_use.defs[d.lhs])
        step = self._assigned_expr(self.def_use.defs[d.rhs])
        if not isinstance(seed, BoolVal) or not isinstance(step, (And, Or)):
            return []
        # `acc = acc <op> b`, with the accumulator being this very phi
        if len(step.args) != 2:
            return []
        carried, probe = step.args
        if not isinstance(carried, Var) or not isinstance(probe, Var):
            return []
        if self.def_use.find_def_from_use(carried) is not d:
            return []

        # `all` refines where it holds, `any` where it does not, and each needs
        # the seed that makes it a fold rather than a constant
        if isinstance(step, And):
            if not (seed.val and truth):
                return []
            want = True
        else:
            if seed.val or truth:
                return []
            want = False

        pred = self.def_use.defining_expr(probe)
        if pred is probe:
            return []
        target_def = self.def_use.find_def_from_site(loop.target, loop)
        cls = dict(self._implied(pred, want)).get(target_def)
        if cls is None or loop.iterable.name in self._shared:
            return []
        return [(self.def_use.find_def_from_use(loop.iterable), cls)]

    def _assigned_expr(self, d: Definition) -> Expr | None:
        """The expression *d* is assigned, where *d* is a plain assignment."""
        return d.site.expr if isinstance(d.site, Assign) else None

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
        """``xs[i]``: whatever every element of ``xs`` is."""
        self._visit_expr(e.index, ctx)
        self._visit_expr(e.value, ctx)
        return self._elt_class(e.value)

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
                return self._elt_class(e.arg)
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
            case Pow() if _positive_literal(e.first):
                return self._rounded(e, _map(_POW_POS_BASE, b))
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
        self._visit_expr(e.cond, ctx)
        with self._refined(e.cond, True):
            ift = self._operand(e.ift, ctx)
        with self._refined(e.cond, False):
            iff = self._operand(e.iff, ctx)
        return ift | iff

    # ------------------------------------------------------------------
    # Statements

    def _visit_assign(self, stmt: Assign, ctx: None):
        self._bind(stmt, stmt.target, self._visit_expr(stmt.expr, ctx))
        if isinstance(stmt.target, NamedId):
            elt = self._elt_of_list_expr(stmt.expr)
            if elt is not None:
                self._set_elt(
                    self.def_use.find_def_from_site(stmt.target, stmt), elt,
                )

    def _elt_of_list_expr(self, e: Expr) -> ValueClass | None:
        """The element class of the list *e* builds, or ``None`` where *e* is
        not a list this can say anything about.

        ``None`` and bottom are different answers: bottom is a fresh
        ``empty(...)``, whose elements no store has reached yet, and ``None``
        leaves the definition with no entry at all, which reads as the top.
        """
        match e:
            case Empty():
                return _BOT
            case ListExpr():
                out = _BOT
                for elt in e.elts:
                    cls = self.by_expr.get(elt)
                    out |= cls if isinstance(cls, ValueClass) else _TOP
                return out
            case Var():
                # `_elt_class`, not the stored class: a copy made inside a
                # refined branch carries that branch's refinement, and
                # `CompToLoop` binds the iterable to a name before reading it
                return self._elt_class(e)
            case _:
                return None

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        stored = self._visit_expr(stmt.expr, ctx)
        # The list itself still carries no *scalar* class; what the store says
        # is about its elements, and joins with whatever was there before --
        # the definition this one supersedes still happened.
        self._bind(stmt, stmt.var, None)
        d = self.def_use.find_def_from_site(stmt.var, stmt)
        # the definition this one supersedes still happened, so its elements
        # are still reachable and join in
        was = _BOT if d.prev is None else self.elt_by_def.get(
            self.def_use.defs[d.prev], _TOP,
        )
        # a nested store (`xs[i][j] = e`) constrains the *inner* list, which
        # this channel does not reach, so the outer one falls back to the top
        cls = stored if isinstance(stored, ValueClass) else _TOP
        self._set_elt(d, was | (cls if len(stmt.indices) == 1 else _TOP))

    def _visit_if1(self, stmt: If1Stmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        with self._refined(stmt.cond, True):
            self._visit_block(stmt.body, ctx)
        self._merge_phis(stmt)

    def _visit_if(self, stmt: IfStmt, ctx: None):
        self._visit_expr(stmt.cond, ctx)
        with self._refined(stmt.cond, True):
            self._visit_block(stmt.ift, ctx)
        with self._refined(stmt.cond, False):
            self._visit_block(stmt.iff, ctx)
        self._merge_phis(stmt)

    def _visit_while(self, stmt: WhileStmt, ctx: None):
        def body():
            self._visit_expr(stmt.cond, ctx)
            with self._refined(stmt.cond, True):
                self._visit_block(stmt.body, ctx)

        self._fixpoint(stmt, body)

    def _visit_for(self, stmt: ForStmt, ctx: None):
        self._visit_expr(stmt.iterable, ctx)

        def body():
            # the target *is* an element, so it inherits the list's class
            self._bind(stmt, stmt.target, self._elt_class(stmt.iterable))
            self._visit_block(stmt.body, ctx)

        self._fixpoint(stmt, body)

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
            lhs = self.def_use.defs[phi.lhs]
            self._set_def(phi, self._def_class(lhs))
            if lhs in self.elt_by_def:
                self._set_elt(phi, self.elt_by_def[lhs])
        for _ in range(self._ROUNDS_PER_PHI * len(phis) + 1):
            prev = {phi: (self.by_def[phi], self.elt_by_def.get(phi))
                    for phi in phis}
            run_body()
            for phi in phis:
                lhs, rhs = self.def_use.defs[phi.lhs], self.def_use.defs[phi.rhs]
                self._set_def(phi, self._def_class(lhs) | self._def_class(rhs))
                self._join_elt_phi(phi, lhs, rhs)
            if all((self.by_def[phi], self.elt_by_def.get(phi)) == prev[phi]
                   for phi in phis):
                return
        for phi in phis:
            self._set_def(phi, _TOP)
            self._set_elt(phi, _TOP)
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
    """A parameter's class, from the context its declared type pins it to."""
    if isinstance(ty, RealType) and isinstance(ty.ctx, Context):
        return representable_classes(ty.ctx)
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
    ) -> ValueClassAnalysis:
        """
        Runs value-class analysis on a function.

        The pre-analyses are accepted as keyword arguments so a caller that
        already holds them -- the C++ compiler holds all three -- does not
        recompute them.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        if ctx_use is None:
            ctx_use = ContextUse.analyze(func, def_use=def_use)
        return _ValueClassInstance(func, type_info, ctx_use).analyze()
