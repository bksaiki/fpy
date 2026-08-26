"""
Path-sensitive value-class analysis.

One question per expression: can this value be a NaN, an infinity, a zero, or a
finite non-zero?  The four atoms form a 16-element lattice — union is the join,
intersection the meet, height 4, so no widening is needed — and it is *refined*
at every branch that tests a value's class.

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
never correct.  Scalars only: a list or tuple carries no class, and reading an
element gives the top class.

Not yet taught: sign (splitting ``±0`` and ``±Inf`` would let ``signbit`` refine),
magnitudes (``x > 1`` says nothing here), ``assert`` statements as refinements,
a bool-valued variable holding a test's result, the class of a numeric free
variable, and a ``for`` target -- a loop counter over ``range`` is an integer and
so neither special, but it reports the top class.
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
from .define_use import DefineUse, DefineUseAnalysis, Definition, DefSite
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
    INF = enum.auto()
    ZERO = enum.auto()
    """Either signed zero -- the sign is not tracked."""
    FINITE = enum.auto()
    """Finite and **non-zero**."""

    TOP = NAN | INF | ZERO | FINITE


_NAN = ValueClass.NAN
_INF = ValueClass.INF
_ZERO = ValueClass.ZERO
_FINITE = ValueClass.FINITE
_TOP = ValueClass.TOP
_BOT = ValueClass(0)


def class_of(x: Float) -> ValueClass:
    """The class *x* belongs to."""
    if x.isnan:
        return _NAN
    if x.isinf:
        return _INF
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


_LOGB = {_NAN: _NAN, _INF: _INF, _ZERO: _INF, _FINITE: _ZERO | _FINITE}
"""``logb(0)`` is an infinity; ``logb(1.5)`` is ``0``."""

_POW_POS_BASE = {_NAN: _NAN, _INF: _INF | _ZERO, _ZERO: _FINITE, _FINITE: _FINITE}
"""``b ** y`` for a positive constant ``b``: ``b ** 0`` is ``1``, and an infinite
exponent gives an infinity or a zero depending on signs neither tracked here."""


def _exact_add(a: ValueClass, b: ValueClass) -> ValueClass:
    """``a + b`` and ``a - b``: the atoms are sign-blind, so one table serves."""
    if not (a and b):
        return _BOT             # an operand nothing reaches produces nothing
    out = _BOT
    if (a | b) & _NAN:
        out |= _NAN
    if (a | b) & _INF:
        out |= _INF
    if a & _INF and b & _INF:
        out |= _NAN                      # inf - inf
    if a & _ZERO and b & _ZERO:
        out |= _ZERO
    if (a & _ZERO and b & _FINITE) or (a & _FINITE and b & _ZERO):
        out |= _FINITE
    if a & _FINITE and b & _FINITE:
        out |= _ZERO | _FINITE
    return out


def _exact_mul(a: ValueClass, b: ValueClass) -> ValueClass:
    if not (a and b):
        return _BOT
    out = _BOT
    if (a | b) & _NAN:
        out |= _NAN
    for x, y in ((a, b), (b, a)):
        if x & _INF and y & (_INF | _FINITE):
            out |= _INF
        if x & _INF and y & _ZERO:
            out |= _NAN                  # 0 * inf
        if x & _ZERO and y & (_ZERO | _FINITE):
            out |= _ZERO
    if a & _FINITE and b & _FINITE:
        out |= _FINITE
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


#####################################################################
# Analysis

class _ValueClassInstance(DefaultVisitor):
    """Single-use instance of value-class analysis."""

    _ROUNDS_PER_PHI = 4
    """A phi can grow once per atom, so a loop settles within this many rounds
    per phi.  Exceeding the bound means a transfer function is not monotone -- a
    bug -- and the phis drop to the top class rather than the walk spinning."""

    func: FuncDef
    type_info: TypeAnalysis
    ctx_use: ContextUseAnalysis

    by_def: dict[Definition, ValueClass | None]
    by_expr: dict[Expr, ValueClass | None]

    _refine: dict[Definition, ValueClass]
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
        self._refine = {}

    @property
    def def_use(self) -> DefineUseAnalysis:
        return self.type_info.def_use

    def analyze(self) -> ValueClassAnalysis:
        self._visit_function(self.func, None)
        return ValueClassAnalysis(
            func=self.func,
            by_expr=self.by_expr,
            by_def=self.by_def,
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
        saved = self._refine
        out = dict(saved)
        for d, cls in self._implied(cond, truth):
            out[d] = out.get(d, _TOP) & cls
        self._refine = out
        try:
            yield
        finally:
            self._refine = saved

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
                # A test bound to a name says what the test says; see
                # `DefineUseAnalysis.defining_expr`.
                src = self.def_use.defining_expr(cond)
                return [] if src is cond else self._implied(src, truth)
            case _:
                return []

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
                exact = _INF
            case _:
                exact = _FINITE      # pi, e, sqrt2, ...
        return self._rounded(e, exact)

    def _visit_unaryop(self, e: UnaryOp, ctx: None) -> ValueClass:
        a = self._operand(e.arg, ctx)
        match e:
            case Neg() | Abs() | Cast():
                return self._rounded(e, a)
            case Logb():
                return self._rounded(e, _map(_LOGB, a))
            case AMin() | AMax() | Fst() | Snd():
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
            case Add() | Sub():
                return self._rounded(e, _exact_add(a, b))
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
                # the result *is* one operand, unrounded
                out = _BOT
                for a in args:
                    out |= a
                return out
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

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: None):
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        # a fresh def of a list, which carries no class
        self._bind(stmt, stmt.var, None)

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
            # no structural classes, so an element is unconstrained
            self._bind(stmt, stmt.target, _TOP)
            self._visit_block(stmt.body, ctx)

        self._fixpoint(stmt, body)

    def _fixpoint(self, stmt: Stmt, run_body: Callable[[], None]):
        """Drives a loop's phi classes to convergence.

        Each phi starts at its pre-loop class and only ever joins, so the walk
        ascends a height-4 lattice and settles without widening.
        """
        phis = self.def_use.phis[stmt]
        for phi in phis:
            self._set_def(phi, self._def_class(self.def_use.defs[phi.lhs]))
        for _ in range(self._ROUNDS_PER_PHI * len(phis) + 1):
            prev = {phi: self.by_def[phi] for phi in phis}
            run_body()
            for phi in phis:
                lhs = self._def_class(self.def_use.defs[phi.lhs])
                rhs = self._def_class(self.def_use.defs[phi.rhs])
                self._set_def(phi, lhs | rhs)
            if all(self.by_def[phi] == prev[phi] for phi in phis):
                return
        for phi in phis:
            self._set_def(phi, _TOP)
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
