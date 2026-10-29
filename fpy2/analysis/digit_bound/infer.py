"""
Digit-bound inference over an FPy program.

Each real-valued expression and definition gets two integer variables:
``msb``, an upper bound on ``logb`` of its value, and ``lsb``, a lower bound
on the position of its least significant digit.  Constraints relate them, and
their difference answers the one question this analysis exists for --

    precision of v  =  max(msb(v) - lsb(v) + 1)

-- which for a rounded value is the precision of the fixed-point result.  See
``docs/todos/digit-bound.md``.

A reduced product with :class:`fpy2.analysis.FormatInfer`: that pass supplies
the ranges this seeds from and consumes the precisions this derives.  Nothing
here builds a :class:`Format`, and nothing in the format lattice names a term.
"""

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, TypeAlias

from ...ast.accessors import subblocks, subexprs
from ...ast.fpyast import *
from ...ast.visitor import DefaultVisitor
from ...function import Function
from ...interpret.value import unwrap_foreign
from ...number import (
    Context,
    EFloatContext,
    MPBFixedContext,
    MPBFloatContext,
    MPFloatContext,
    MPSFloatContext,
    OverflowMode,
    RoundingMode,
)
from ...number.context.context import SizedContext
from ...number.context.mp_fixed import MPFixedContext
from ...number.context.real import RealContext
from ...types import ListType, RealType
from ..alias import Alias, Region
from ..array_size import ArraySizeAnalysis, ArraySizeBound, ListSize, concrete_size
from ..context_use import ContextScope, ContextUseAnalysis, PartialContext
from ..define_use import DefineUse, DefineUseAnalysis
from ..reaching_defs import AssignDef, Definition, PhiDef
from ..type_infer import TypeAnalysis
from ..value_class import ValueClass, ValueClassAnalysis, ValueClassInfer
from .spaces import Binder, Elements, IndexSpaces, Key
from .store import DigitBoundStore, Term

__all__ = [
    'Bounds',
    'DigitBoundAnalysis',
    'DigitBoundInfer',
    'DigitBoundParams',
    'FormatView',
    'Terms',
]

_Operand: TypeAlias = int | Definition
"""A range operand's identity: a literal by its value, a variable by its
definition."""


@dataclass(frozen=True)
class _Range:
    """`range(start, stop, step)`."""

    start: _Operand
    stop: _Operand
    step: _Operand = 1


@dataclass(frozen=True)
class _Elts:
    """The elements `start + step * k` for `k` over `range(n)`."""

    start: _Operand
    step: _Operand
    n: _Operand


@dataclass(frozen=True)
class _Witness:
    """An index an existential picks -- an aggregate's worst element, a loop's
    iteration before, an element's writer -- by the `id` of what picks it."""

    of: int | tuple[int, int]


@dataclass(frozen=True)
class _Last:
    """A loop's last iteration, by the loop's `id`, or a partial write's last
    writer, by its phi's."""

    of: int


@dataclass(frozen=True)
class _Read:
    """Reads at *binder*'s index of lists the invariant keeps from
    corresponding with it (digit-bound.md, "Element reads")."""

    binder: Key


@dataclass(frozen=True)
class _Apart:
    """An index set enumerated by a binder nested with the one enumerating it
    first: the *n*-th such, apart from the rest."""

    of: '_Range | _Elts'
    n: int


_IndexKey: TypeAlias = _Range | _Elts | _Witness | _Last | _Read | _Apart
"""An index set a copy is taken over."""


@dataclass
class _IndexSet:
    """One copy of some variables at another index: a part, an element read,
    or a witness."""

    tag: str
    binders: tuple[Key, ...] = ()
    """The binders the copy varies with."""
    elements: set[Key] = field(default_factory=set)
    """The spaces the copy moves over."""
    subst: dict[int, Term] = field(default_factory=dict)
    lits: dict[int, int] = field(default_factory=dict)
    """The literals renamed onto it that no variable owns."""
    done: set[int] = field(default_factory=set)
    """The constraints replayed onto it."""
    minted: int = 0
    """How many of :attr:`subst`'s copies have their index spaces."""


@dataclass
class Terms:
    """What a value names in the store.

    ``msb`` and ``lsb`` bracket it from above and below; their difference is
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

    The digit-bound analogue of :attr:`FunctionFormat.arg_fmts`, and it
    cannot take that shape: a relation *between* two arguments -- ``n`` is
    ``xs``'s greatest exponent less twelve -- lives in the store and reduces
    to no per-argument bound, so the terms travel with the system that gives
    them meaning.
    """

    store: DigitBoundStore
    args: tuple[Terms, ...]
    assume: frozenset[int] = frozenset()
    """The literals every call reaching the callee proved, which hold
    wherever its body runs."""
    outer: tuple[Key, ...] = ()
    """The caller's binders around the call, which the callee varies with."""
    elements: tuple[tuple[Key, ...], ...] = ()
    """Each list argument's element spaces, which its parameter's are."""


@dataclass
class DigitBoundAnalysis:
    """Result of digit-bound inference: the constraint system, and what names it."""

    params: DigitBoundParams
    """The binding this function was walked under, so the same one can be
    replayed on a specialized copy of it."""
    by_expr: dict[Expr, Terms] = field(default_factory=dict)
    by_def: dict[Definition, Terms] = field(default_factory=dict)
    by_call: dict[Call, 'DigitBoundAnalysis'] = field(default_factory=dict)
    ret_elements: tuple[Key, ...] = ()
    """The element spaces of a list it returns."""
    ret: Terms = field(default_factory=Terms)
    assume_at: Callable[[Expr], frozenset[int]] = lambda _e: frozenset()
    """The guard literals that hold wherever an expression is evaluated."""

    @property
    def store(self) -> DigitBoundStore:
        return self.params.store

    def query_at(self, e: Expr) -> frozenset[int]:
        """The literals a query about *e* assumes: what holds wherever it is
        evaluated, and that the instance asked about is finite and non-zero
        (digit-bound.md, section 5)."""
        terms = self.by_expr.get(e)
        own = () if terms is None or terms.msb is None else (
            *self.store.high(terms.msb), *self.store.not_neg_inf(terms.msb))
        return self.assume_at(e) | frozenset(own)

    def bounds(self, e: Expr) -> Bounds | None:
        """*e*'s precision, digit position and magnitude.

        ``None`` whenever the store leaves any of the three open.
        """
        terms = self.by_expr.get(e)
        if terms is None or terms.msb is None or terms.lsb is None:
            return None
        # One open value discards all three, so each is asked only once the
        # one before it came back closed -- every query costs a solve.
        at = self.query_at(e)
        prec = self.store.prec(terms.msb, terms.lsb, at)
        if not isinstance(prec, int):
            return None
        exp = -self.store.maximum(-terms.lsb, at)
        if not isinstance(exp, int):
            return None
        mag = self.store.maximum(terms.msb, at)
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

    scopes: dict[ContextScope, Context | PartialContext]
    """Each context scope's resolved context, concrete or partial."""

    def has_finite(self, of: Expr | Definition) -> bool:
        """Does *of*'s format hold any finite value?"""
        ...

    def has_non_finite(self, of: Expr | Definition) -> bool:
        """Can *of* be an infinity or NaN?"""
        ...

    def has_digits(self, of: Expr | Definition) -> bool:
        """Does *of*'s format hold a finite non-zero value?"""
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


_FLOAT_CTXS = (EFloatContext, MPFloatContext, MPBFloatContext, MPSFloatContext)
"""Float formats: a rounding to one keeps its operand's binade or carries."""


_PROPAGATING = (Add, Sub, Mul, Neg, Abs, Round, Cast, Hypot, Fma, Trunc, Floor, Ceil, RoundInt, NearbyInt)
"""Operations whose result is not finite where an operand is not, and is
finite but for an overflow where every operand is."""

_NEAREST = (RoundingMode.RNE, RoundingMode.RNA)
"""The modes rounding to nearest: a value below half the least subnormal
rounds to zero, so none carries further than one binade."""


def _round_will_carry(rm: RoundingMode | None) -> bool:
    """Whether rounding under *rm* can carry out of the top binade.

    Only ``RTZ`` never increases a magnitude; every other mode carries on at
    least one sign, and this domain does not track sign.  An unresolved mode
    has to be assumed to carry.  ``RTO`` is not excepted; see
    ``TestRoundCarry``.
    """
    return rm is not RoundingMode.RTZ


def _placed(block: StmtBlock, loops: tuple[Stmt, ...] = ()) -> Iterator[tuple[Stmt, tuple[Stmt, ...]]]:
    """Every statement in *block*, in program order, with the loops around it."""
    for s in block.stmts:
        yield s, loops
        inner = (*loops, s) if isinstance(s, WhileStmt | ForStmt) else loops
        for _, sub in subblocks(s):
            yield from _placed(sub, inner)


def _stmts(block: StmtBlock) -> Iterator[Stmt]:
    """Every statement in *block*, nested ones included."""
    return (s for s, _ in _placed(block))


def _exprs(node: Stmt | Expr) -> Iterator[Expr]:
    """Every expression in *node*, not in a block it holds."""
    for _, _, e in subexprs(node):
        yield e
        yield from _exprs(e)


def _written_params(fn: FuncDef) -> frozenset[int]:
    """The parameters whose lists *fn* may store into."""
    def_use = DefineUse.analyze(fn)
    alias = Alias.analyze(fn, def_use=def_use)
    return frozenset(i for i, r in alias.param_regions(fn, def_use).items() if alias.may_write(r))


def _is_empty_alloc(d: Definition) -> bool:
    """Was *d* defined by `fp.empty(...)` -- holding no element to describe?"""
    return isinstance(d.site, Assign) and isinstance(d.site.expr, Empty)


def _encloses(outer: tuple[Stmt, ...] | None, inner: tuple[Stmt, ...]) -> bool:
    """Whether the loops *outer* are the outermost of *inner*."""
    return outer is not None and inner[:len(outer)] == outer


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
    the same constraints.  A loop-carried value joins the pre-loop value with a
    copy of the body's at the iteration before (digit-bound.md, "Control").
    """

    func: FuncDef
    view: FormatView
    store: DigitBoundStore
    type_info: TypeAnalysis
    def_use: DefineUseAnalysis
    ctx_use: ContextUseAnalysis
    array_size: ArraySizeAnalysis
    scopes: dict[ContextScope, Context | PartialContext]
    out: DigitBoundAnalysis
    params: DigitBoundParams
    """The caller's binding of this function."""
    outer: Callable[[], frozenset[int]]
    """The literals holding wherever this body runs: its call site's."""
    _loop_index: dict[Definition, int]
    _trip: tuple[Definition, _Operand] | None
    """The innermost loop's index and trip count, where it counts one: only
    its `a + s * k` is element `k` of an index set, as only the innermost
    `range`'s target is for :attr:`_gather`."""
    _builders: dict[Region, tuple[ForStmt, Definition, PhiDef]]
    """Each index function a loop builds: the loop, and the `empty` it fills
    (digit-bound.md, "Lists")."""
    _built: dict[Region, Terms]
    """Each built list's element terms, which every definition of it shares."""
    _in_build: set[Definition]
    """The definitions of built lists inside their builders' bodies, which
    have terms of their own: the list along one path of one iteration."""
    _built_stores: dict[Region, list[IndexedAssign]]
    _built_fields: dict[Region, tuple[Terms | None, ...]]
    """A built list of tuples' element, field by field."""
    _built_expr: dict[Region, Expr]
    """A built list's element, where one store writes it."""
    _lsb_vars: set[int]
    _index_sets: dict[_IndexKey, list[_IndexSet]]
    """Each index set's copies, one per set of classes moved (digit-bound.md, section
    6)."""
    _enumerators: dict[_Range | _Elts, list[Key]]
    """The binders enumerating each index set, one per index set apart."""
    _gather: tuple[_Range, Definition] | None
    _returns: list[tuple[Expr, Terms]]
    _classes: ValueClassAnalysis
    _unvalued_regions: set[Region]
    """The regions whose lists are region summaries; see :meth:`_unvalued`."""
    _written_args: list[Expr]
    """The arguments of calls whose callee may store into them."""
    _read_phis: set[PhiDef]
    """The phis something reads, directly or through another phi."""
    _summary_reads: list[tuple[Terms, Expr | Definition, Region | None]]
    """Each read of a list that is not an index function: its terms, what
    they are of, and the region of the elements it reads."""
    _guards: dict[Definition, int]
    """The literal "this definition is finite", for each one a guard names."""
    _elt_guards: dict[Region, int]
    """The literal "every element of this list is finite", for each list whose
    elements a guard names."""
    _loops: tuple[Stmt, ...]
    """The loops enclosing the walk, outermost first."""
    _binders: tuple[Key, ...]
    """The binders enclosing the walk, outermost first; see :mod:`.spaces`."""
    _binders_of: dict[Expr | Stmt, tuple[Key, ...]]
    _list_keys: set[Key]
    """The element keys opened so far."""
    _arms: tuple[int, ...]
    """The literals of the `if` arms around the walk."""
    _arms_of: dict[Expr | Stmt, tuple[int, ...]]
    _dropped: bool
    """Whether a return producing no finite value was left out."""
    _index_binder: dict[Definition, Key]
    """The binder whose index each loop target is."""
    _callees: list['_DigitBoundInferInstance']
    """The walks of the calls this one made, whose copies it replays last."""
    _walk: int
    """This walk's identity in its keys."""
    _loops_of_stmt: dict[Stmt, tuple[Stmt, ...]]
    _loops_of_expr: dict[Expr, tuple[Stmt, ...]]
    _scope_of: dict[ContextStmt, ContextScope]
    _established: frozenset[tuple[int, int]]
    """What every `with` dominating the walk's position makes true: each
    variable of its position finite, with the literals saying so."""
    _established_at: dict[Expr, frozenset[tuple[int, int]]]
    _summaries: set[int] | None
    """:meth:`_summary_vars`, once the walk is over."""

    def __init__(
        self,
        func: FuncDef,
        view: FormatView,
        params: DigitBoundParams,
        classes: ValueClassAnalysis | None = None,
        outer: Callable[[], frozenset[int]] | None = None,
    ) -> None:
        self.func = func
        self.params = params
        self.outer = outer if outer is not None else lambda: params.assume
        self.view = view
        self.store = params.store
        self.type_info = view.type_info
        self.def_use = view.type_info.def_use
        self.ctx_use = view.ctx_use
        self.array_size = view.array_size
        self.scopes = view.scopes
        self.out = DigitBoundAnalysis(params)
        self._loop_index = {}
        self._trip = None
        self._builders = {}
        self._built = {}
        self._in_build = set()
        self._built_stores = {}
        self._built_fields = {}
        self._built_expr = {}
        self._lsb_vars = set()
        self._index_sets = {}
        self._enumerators = {}
        self._gather = None
        self._returns = []
        self._classes = classes if classes is not None else ValueClassInfer.analyze(
            func, def_use=self.def_use, type_info=self.type_info, ctx_use=self.ctx_use)
        self._unvalued_regions = set()
        self._written_args = []
        self._read_phis = set()
        self._summary_reads = []
        self._guards = {}
        self._elt_guards = {}
        self._loops = ()
        # a callee varies with the caller's binders around the call
        self._binders = params.outer
        self._binders_of = {}
        self._list_keys = set()
        self._arms = ()
        self._arms_of = {}
        self._dropped = False
        self._index_binder = {}
        self._walk = self.store.spaces.walk()
        self._callees = []
        self._loops_of_stmt = {}
        self._loops_of_expr = {}
        self._scope_of = {s.site: s for s in self.ctx_use.scopes if isinstance(s.site, ContextStmt)}
        self._established = frozenset()
        self._established_at = {}
        self._summaries = None

    def analyze(self) -> DigitBoundAnalysis:
        # one term per parameter, or none: `zip` would hide a mismatch
        args = self.params.args
        if args and len(args) != len(self.func.args):
            raise RuntimeError(
                f'digit-bound params for `{self.func.name}` have {len(args)} '
                f'terms for {len(self.func.args)} parameters'
            )
        for arg, src in zip(self.func.args, args):
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                self._share(d, src)
        # a list parameter is the argument's list, element for element
        for arg, keys in zip(self.func.args, self.params.elements):
            if isinstance(arg.name, NamedId):
                d = self.def_use.find_def_from_site(arg.name, arg)
                for k, key in enumerate(keys):
                    self.store.spaces.unify(self._elements(d, k), key)
        self._unvalued_regions = self._unvalued()    # and which loops build index functions
        self._visit_block(self.func.body, None)
        self._bound_summaries()
        # a fact stated after a part was read, such as a `max` over the whole
        # list, still holds of the part
        self._replay_all()
        self.out.ret = self._merge_returns()
        self.out.ret_elements = self._return_elements()
        self._summaries = self._summary_vars()
        self.out.assume_at = self._assumed
        return self.out

    # -- terms ---------------------------------------------------------

    def _var(self, name: str) -> Term:
        """A fresh variable, varying with the binders around the walk."""
        t = self.store.var(name)
        for v, _ in t.coeffs:
            self.store.spaces.mint(v.index, self._binders)
        return t

    def _mark_elt(self, of: Expr | Definition, *ts: Term | None) -> None:
        """Record *ts* as varying with the list *of*'s index, whoever minted
        them: a list's terms describe an element wherever it came from.  Its
        lsb is not among them, being already at or below every element's --
        unless what bounds it varies (`_grid_ge`)."""
        if not isinstance(self._type_of(of), ListType):
            return
        keys = self._all_elements(of)
        for t in ts:
            for v, _ in (t.coeffs if t is not None else ()):
                self.store.spaces.bound(v.index, keys)

    def _binder(self, node: Stmt | Expr, clause: int = 0) -> Key:
        """A loop's, or a comprehension clause's, key: this walk's own, so two
        calls to one function keep theirs apart."""
        return Binder(self._walk, id(node), clause)

    def _elements(self, of: Expr | Definition | FuncDef, depth: int = 0) -> Key:
        """The key of the list *of*'s element space at index *depth*, opened
        inside the binders the list varies with and its outer indices, so that
        no equation puts it with one of them.

        A list that only names another -- a use, a parameter, a row -- has
        no binders of its own: its class takes them from what it names.  Every
        definition of a built list has the `empty`'s."""
        if isinstance(of, Definition) and (r := self._built_of(of)) is not None:
            of = self._builders[r][1]
        key = Elements(self._walk, id(of), depth=depth)
        if key not in self._list_keys:
            self._list_keys.add(key)
            around = self._list_binders_of(of) or ()
            outer = (self._elements(of, depth - 1),) if depth else ()
            self.store.spaces.open(key, (*around, *outer))
        return key

    def _depth(self, of: Expr | Definition) -> int:
        """How many indices the list *of* has: 0 for what is not a list."""
        t, n = self._type_of(of), 0
        while isinstance(t, ListType):
            t, n = t.elt, n + 1
        return n

    def _all_elements(self, of: Expr | Definition) -> tuple[Key, ...]:
        """The keys of every index of the list *of*, outermost first."""
        return tuple(self._elements(of, k) for k in range(self._depth(of)))

    def _list_binders_of(self, of: Expr | Definition | FuncDef) -> tuple[Key, ...] | None:
        """The binders the list *of* is made under, or `None` where it names
        another list."""
        if isinstance(of, FuncDef):
            return self.params.outer
        if isinstance(of, Definition):
            site = of.site
            if isinstance(site, Argument | FuncDef):
                return None
            if isinstance(site, Assign) and isinstance(site.expr, Var):
                return None
            return self._def_binders(of)
        if isinstance(of, Var):
            return None
        return self._binders_of.get(of, self._binders)

    def _unify_elements(self, a: Expr | Definition, b: Expr | Definition, shift: int = 0) -> None:
        """*a*'s index `k` is *b*'s index `k + shift`, where they are lists --
        one list, at `0`; a row of *b*, at `1` -- where the invariant lets the
        two be one index."""
        n = self._depth(b)
        for k in range(self._depth(a)):
            if 0 <= k + shift < n:
                self.store.spaces.correspond(self._elements(a, k), self._elements(b, k + shift))

    def _index_of(self, idx: Expr) -> Key | None:
        """The binder whose index *idx* is, if any."""
        return self._index_binder.get(self.def_use.find_def_from_use(idx)) if isinstance(idx, Var) else None

    def _reads_summary(self, d_lst: Definition, binder: Key | None) -> bool:
        """Make *binder* correspond with *d_lst*'s elements where the invariant
        allows; whether the read is the representative."""
        return binder is None or self.store.spaces.correspond(binder, self._elements(d_lst))

    def _read_apart(self, t: Terms, d_lst: Definition, binder: Key) -> Terms:
        """*t*, of the list *d_lst*'s element, at *binder*'s index: a copy
        every list read apart there shares."""
        around = (*self.store.spaces.around(binder), binder)
        return self._copy(t, (self._elements(d_lst),), _Read(binder), around)

    def _element(self, d_lst: Definition, binder: Key | None) -> Terms:
        """The list *d_lst*'s element at *binder*'s index: its representative, or
        where the binder cannot correspond with it, a copy."""
        src = self._def(d_lst)
        if self._reads_summary(d_lst, binder):
            return src
        assert binder is not None
        return self._read_apart(src, d_lst, binder)

    def _fresh_interval(self, terms: Terms, of: Expr | Definition, tag: str) -> tuple[Term, Term]:
        """Give *terms* a bracketing pair, seeded from *of*'s inferred range.

        ``logb`` gets no lower seed: it bounds a magnitude from above, and a
        zero has no magnitude to bound.  Pinning it above the format's least
        non-zero value would make a zero's own reading contradictory.
        """
        lo, hi = self.view.logb_range(of)
        terms.msb = self._var(f'msb{tag}')
        terms.lsb = self._var(f'lsb{tag}')
        self._lsb_vars.update(v.index for v, _ in terms.lsb.coeffs)
        self._mark_elt(of, terms.msb)
        self._seed(terms.msb, of, None, hi)
        self._seed(terms.lsb, of, lo, None)
        return terms.msb, terms.lsb

    def _seed(
        self, t: Term, of: Expr | Definition, lo: int | None, hi: int | None,
        low: tuple[int, ...] = (),
    ) -> None:
        """Bound *t* by *of*'s format range, the one place a format bounds a
        term (digit-bound.md, "Pointwise facts"): its finite values -- above
        under `high` where *of* may not be finite, below under *low* -- and
        only where the arms around *of* run."""
        arms = self._arms_at(of)
        if hi is not None:
            self.store.le(t, hi, guard=(*self._high_if(of, t), *arms))
        if lo is not None:
            self.store.ge(t, lo, guard=(*low, *arms))

    def _high_if(self, of: Expr | Definition, t: Term) -> tuple[int, ...]:
        """*t*'s `high` literals, where *of* may be non-finite."""
        return self.store.high(t) if self.view.has_non_finite(of) else ()

    def _arms_at(self, of: Expr | Definition) -> tuple[int, ...]:
        """The literals of the `if` arms around *of*."""
        site = of if isinstance(of, Expr) else of.site
        return self._arms_of.get(site, ()) if isinstance(site, Expr | Stmt) else ()

    def _def(self, d: Definition) -> Terms:
        """*d*'s terms, minted on first reference.

        A definition, not its defining expression, is what uses share: a
        parameter has no defining expression, and two reads of one have to land
        on the same variable or nothing cancels.
        """
        terms = self.out.by_def.get(d)
        if terms is None and (r := self._built_region(d)) is not None:
            terms = self.out.by_def[d] = self._built_terms(r)
        if terms is None:
            terms = self.out.by_def[d] = Terms()
            # minted where it is first read, but varying as its definition does
            binders, self._binders = self._binders, self._def_binders(d)
            self._fresh_interval(terms, d, f'd{len(self.out.by_def)}')
            self._binders = binders
        return terms

    def _built_of(self, d: Definition) -> Region | None:
        """The region of the built list *d* is a definition of, if it is one."""
        if not self._builders or not isinstance(self._type_of(d), ListType):
            return None
        r = self._classes.alias.region_of(d)
        return r if r in self._builders else None

    def _built_region(self, d: Definition) -> Region | None:
        """The region of the built list *d* is a definition of, outside its
        builder's body: the list as a whole, which shares the built terms."""
        return None if d in self._in_build else self._built_of(d)

    def _mid_build(self, e: Var | ListRef) -> bool:
        """Does *e* read a built list inside the loop building it, anywhere but
        at the loop's own index?  Its builder states its terms as the loop
        ends, so a read before that names none of them."""
        lst = e if isinstance(e, Var) else e.value
        assert isinstance(lst, Var)
        r = self._built_of(self.def_use.find_def_from_use(lst))
        if r is None or (loop := self._builders[r][0]) not in self._loops:
            return False
        return not (isinstance(e, ListRef) and self._index_of(e.index) == self._binder(loop))

    def _built_terms(self, r: Region) -> Terms:
        """The element terms of the list built into region *r*: one value per
        index wherever a read sees it, so every definition shares them.  Seeded
        by the list as its loop leaves it; bounded by its stores there."""
        terms = self._built.get(r)
        if terms is None:
            _, alloc, done = self._builders[r]
            terms = self._built[r] = Terms()
            binders, self._binders = self._binders, self._def_binders(alloc)
            self._fresh_interval(terms, done, f'B{len(self._built)}')
            self._binders = binders
        return terms

    def _def_binders(self, d: Definition) -> tuple[Key, ...]:
        """The binders *d* varies over: those around its site, and the site's
        own where it binds per iteration -- a loop's target and phis, a
        comprehension's targets."""
        site = d.site
        match site:
            case Argument() | FuncDef():
                return self.params.outer
            case ListComp():
                outer = self._binders_of.get(site, self._binders)
                return (*outer, *(self._binder(site, k) for k in range(len(site.targets))))
            case ForStmt() | WhileStmt():
                return (*self._binders_of.get(site, self._binders), self._binder(site))
            case _:
                return self._binders_of.get(site, self._binders)

    def _share(self, d: Definition, src: Terms) -> None:
        """Adopt *src*'s terms for *d*, field by field, where *d* has none.

        States nothing.  The two end up naming the *same* store variables, so
        every constraint on either binds both -- which is what lets a callee's
        rounding cancel against a position its caller built, and is a stronger
        claim than `_bracket`'s one-directional one.

        A field *d* already has is kept, so the lookup mints nothing.
        """
        dst = self.out.by_def.setdefault(d, Terms())
        dst.msb = dst.msb if dst.msb is not None else src.msb
        dst.lsb = dst.lsb if dst.lsb is not None else self._elt_lsb(d, src)
        dst.value = dst.value if dst.value is not None else src.value
        # a list's terms vary with its own elements, a row's as well as its
        # matrix's
        self._mark_elt(d, dst.msb, dst.value)

    def _bracket(
        self, terms: Terms, d: Expr | Definition, srcs: list[Terms], *, finite: bool = False,
    ) -> None:
        """Bound *terms*, *d*'s, by every one of *srcs*."""
        srcs = self._aligned(d, srcs)
        msbs = [s.msb for s in srcs if s.msb is not None]
        lsbs = [t for s in srcs if (t := self._elt_lsb(d, s)) is not None]
        if terms.msb is not None and len(msbs) == len(srcs):
            guard = self.store.high(terms.msb) if finite else ()
            self.store.le_max(terms.msb, msbs, guard=guard)
            if not finite:
                # one of them, so finite wherever every one is
                self._implied_high(terms.msb, msbs)
        if terms.lsb is not None and len(lsbs) == len(srcs):
            self._grid_ge(terms.lsb, *lsbs)

    def _aligned(
        self, d: Expr | Definition | FuncDef, srcs: list[Terms], keys: tuple[Key, ...] | None = None,
    ) -> list[Terms]:
        """*srcs* as sources of the list *d*'s element.  One varying with
        another element space is read at a witness element of it, varying
        with *d*'s: a constraint over two element spaces is read for every
        pair (digit-bound.md, "The abstract domain"), and an element is one
        source's at some index of it, not every."""
        if keys is None:
            if isinstance(d, FuncDef) or not isinstance(self._type_of(d), ListType):
                return srcs
            keys = self._all_elements(d)
        spaces = self.store.spaces
        own = {spaces.find(k) for k in keys}
        out = []
        for i, src in enumerate(srcs):
            moved = {
                k for t in (src.msb, src.lsb, src.value) if t is not None
                for k in self._varies(t) if isinstance(k, Elements) and spaces.find(k) not in own
            }
            if moved:
                src = self._copy(src, moved, _Witness((id(d), i)), (*self._binders, *keys))
            out.append(src)
        return out

    def _one_space(self, *of: Expr | Definition) -> bool:
        """Are the lists *of* one index, element for element?  Trivially, for
        what is not a list."""
        find = self.store.spaces.find
        return len({tuple(find(k) for k in self._all_elements(x)) for x in of}) == 1

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

        All-or-nothing, as `_bracket` is: a bound over a strict subset of the
        operands is one the dropped operand never had to satisfy.  No
        operands at all is a list of none, which the store reads as no claim.
        """
        return _all(self._msb_of(e) for e in es)

    def _lsbs(self, *es: Expr) -> list[Term] | None:
        """Every operand's grid, or `None` where any lacks one; see
        :meth:`_msbs`."""
        return _all(self._lsb_of(e) for e in es)

    # -- emission ------------------------------------------------------

    def _visit_expr(self, e: Expr, ctx: Any) -> None:
        self._binders_of[e] = self._binders
        self._arms_of[e] = self._arms
        self._loops_of_expr[e] = self._loops
        self._established_at[e] = self._established
        super()._visit_expr(e, ctx)
        if not isinstance(self.type_info.by_expr.get(e), RealType | ListType):
            return
        terms = self.out.by_expr.setdefault(e, Terms())
        if terms.msb is not None:
            return

        # A variable reads what its definition wrote, and `xs[i]` over the
        # whole list *is* an element, so both share terms rather than
        # getting their own.
        match e:
            case ListRef(value=Var() as lst) if self._depth(e):
                # a row: its indices are the list's inner ones
                self._unify_elements(e, self.def_use.find_def_from_use(lst), 1)
            case ListExpr() | ListComp() if self._depth(e) > 1:
                for el in (e.elts if isinstance(e, ListExpr) else (e.elt,)):
                    self._unify_elements(el, e, 1)
        src: Terms | None
        match e:
            case Var() | ListRef(value=Var()) if self._mid_build(e):
                src = None      # what other iterations wrote, or nothing yet: seeded alone
            case Var() if self._is_unvalued(self.def_use.find_def_from_use(e)):
                src = None      # a region summary: bounded by what enters it, once the walk is over
                self._summary_reads.append((terms, e, self._inner_region(e)))
            case Var():
                d = self.def_use.find_def_from_use(e)
                src = self._def(d)
                self._unify_elements(e, d)
            case ListRef(value=Var() as lst) if self._is_unvalued(self.def_use.find_def_from_use(lst)):
                src = None
                self._summary_reads.append((terms, e, self._inner_region(lst)))
            case ListRef(value=Var() as lst, index=idx) if self._covers(
                idx, self._len_of(lst)
            ):
                src = self._element(self.def_use.find_def_from_use(lst), self._index_of(idx))
            case ListRef(value=Var() as lst, index=idx) if (
                inst := self._at_index(lst, idx)
            ) is not None:
                # an element of a *part* of the list: the value comes too,
                # since what a gathered exponent goes on to build cancels
                # against it
                terms.msb, terms.lsb, terms.value = inst.msb, inst.lsb, inst.value
                return
            case ListComp() if len(e.targets) == 1 and self.store.spaces.correspond(
                self._elements(e), self._binder(e),
            ):
                src = self.out.by_expr.get(e.elt)
            case _:
                src = None
        if src is not None:
            terms.msb, terms.lsb = src.msb, self._elt_lsb(e, src)
            return

        logb, grid = self._fresh_interval(terms, e, f'e{len(self.out.by_expr)}')
        if self._summary_reads and self._summary_reads[-1][0] is terms:
            # a summary read's grid is bounded per read, at a witness of its own
            for v, _ in grid.coeffs:
                self.store.spaces.bound(v.index, self._binders)
        if isinstance(e, IfExpr) and isinstance(self._type_of(e), ListType):
            # one of the arm lists, element for element -- where it can be
            for arm in self._live_arms(e):
                self._unify_elements(e, arm)
            if not self._one_space(e, *self._live_arms(e)):
                return
        if isinstance(e, ListComp):
            # several clauses, or one whose index is not the list's: element
            # `k` is built at a point of them, which is no clause's index
            if (src := self.out.by_expr.get(e.elt)) is not None:
                over = [self._binder(e, k) for k in range(len(e.targets))]
                w = self._copy(src, over, _Witness(id(e)), (*self._binders, *self._all_elements(e)))
                self._bracket(terms, e, [w])
            return
        exact, exact_grid = self._active(e, logb, grid)
        if exact is logb and (guard := self._finite_guard(e, logb)):
            # an exact rule bounds the result only where it is finite
            exact = self._var(f'X{len(self.out.by_expr)}')
            self.store.le(logb, exact, guard=guard)
        self._emit_logb(e, exact)
        self._emit_grid(e, exact_grid)
        self._emit_finite(e, logb)

    def _grid_ge(self, grid: Term, *ts: Term, guard: tuple[int, ...] = ()) -> None:
        """Bound *grid* below by the least of *ts*, noting what that costs.

        Every such bound is true of the value in hand.  It stays true of a
        whole *list* only while nothing it was built from moves with that
        list's index, so *grid* inherits what its bounds vary over: see
        :meth:`_elt_lsb`.
        """
        self.store.ge_min(grid, ts, guard=guard)
        keys = {k for t in ts for k in self._varies(t)}
        for v, _ in grid.coeffs:
            self.store.spaces.bound(v.index, keys)

    def _varies(self, t: Term) -> set[Key]:
        """What *t* varies over, entering a bound: where its variables were
        minted and what bounds them -- except that an lsb of its own is already
        at or below every element's, so only what bounds it counts.  Negated it
        is a ceiling, not a floor, and that exception lapses."""
        return {k for v, c in t.coeffs for k in self.store.spaces.keys(
            v.index, minted=not (c > 0 and v.index in self._lsb_vars))}

    def _elt_lsb(self, of: Expr | Definition, src: Terms) -> Term | None:
        """*src*'s grid, as *of*'s -- dropped where *of* is a list and the
        grid moves with its index.

        A list's terms are read as uniform: at or below *every* element's
        grid.  One that moves with the index is at or below one element's,
        and reading it uniformly would claim the smallest element sits as
        high as the widest.  No grid at all is the weaker, true reading.
        """
        if src.lsb is None or not isinstance(self._type_of(of), ListType):
            return src.lsb
        spaces = self.store.spaces
        here = {spaces.find(b) for b in self._binders}
        if any(spaces.find(k) not in here for k in self._varies(src.lsb)):
            return None
        return src.lsb

    def _active(self, e: Expr, logb: Term, grid: Term) -> tuple[Term, Term]:
        """Account for the context *e* is evaluated under: the terms the exact
        rules bound, for its `logb` and its grid.

        The rules below bound the *exact* result, but an expression denotes
        it rounded at the active context (digit-bound.md, "Pointwise facts").  A
        carrying mode reaches past the exact binade, so the rules bound a
        fresh term the result reaches past; the context floors the grid,
        which is what pays for the carry.  A context this does not model
        leaves the result unrelated to the exact one: the rules bound fresh
        terms nothing reads.

        `Round`/`Cast` model their context themselves; a selection, a
        projection or a call hands back a value it did not round, as does a
        `sum` of one element or unknown length.
        """
        if isinstance(e, Round | Cast | Min | Max | AMin | AMax | Fst | Snd | Call):
            return logb, grid
        if e not in self.ctx_use.use_to_scope:
            return logb, grid
        # a fold's length, `None` for a single rounding
        n = self._len_of(e.arg) if isinstance(e, Sum) else None
        if isinstance(e, Sum) and (n is None or n < 2):
            # one element comes back unrounded, and an unknown length may be one
            return logb, grid
        tag = len(self.out.by_expr)
        found = self._rounding(e)
        ctx = self._active_ctx(e)
        if found is not None:
            pos, rm = found
            self._grid_ge(grid, pos + 1)
            if not _round_will_carry(rm):
                return logb, grid
            exact = self._var(f'X{tag}')
            # each of a fold's `n - 1` roundings moves it by under a quantum
            steps = 0 if n is None else math.ceil(math.log2(n))
            self.store.le_max(logb, [exact + 1, pos + 1 + steps], guard=self._finite_guard(e, logb))
            return exact, grid
        if isinstance(ctx, RealContext):
            return logb, grid
        if isinstance(ctx, _FLOAT_CTXS) and isinstance(ctx.pmax, int):
            p = ctx.pmax
            self._grid_ge(grid, logb - (p - 1))
            rm = None if ctx.is_stochastic() else ctx.rm
            if not _round_will_carry(rm):
                return logb, grid
            expmin = getattr(ctx, 'expmin', None)
            exact = self._var(f'X{tag}')
            if n is None:
                # a tiny result rounds up to the least subnormal, unless to
                # nearest, which reaches it from at most one binade below
                floor = [] if expmin is None or rm in _NEAREST else [expmin]
                self.store.le_max(logb, [exact + 1, *floor], guard=self._finite_guard(e, logb))
                return exact, grid
            # A fold's `n - 1` roundings grow it by at most a factor
            # `(1 + 2 ** (1 - p)) ** (n - 1)`, under 2 while `n - 1 <= 2 ** (p - 2)`,
            # and each by under the least subnormal besides.
            if n - 1 > 2 ** (p - 2):
                return exact, self._var(f'Y{tag}')
            if expmin is None:
                self.store.le(logb, exact + 1, guard=self._finite_guard(e, logb))
            else:
                self.store.le_max(logb, [exact + 2, expmin + 1 + math.ceil(math.log2(n))],
                                  guard=self._finite_guard(e, logb))
            return exact, grid
        if isinstance(ctx, MPBFixedContext):
            # wraps or saturates: neither keeps the exact result's binade or grid
            self._grid_ge(grid, Term((), ctx.nmin + 1))
        return self._var(f'X{tag}'), self._var(f'Y{tag}')

    def _exact_int(self, e: Expr) -> bool:
        """Does *e*'s context represent its integer result exactly?  A
        rounded result outside what the context represents was rounded from
        outside it too, so its range answers for the exact one."""
        if e not in self.ctx_use.use_to_scope:
            return True
        ctx = self._active_ctx(e)
        if isinstance(ctx, RealContext):
            return True
        if isinstance(ctx, MPFixedContext):
            return ctx.nmin < 0
        rng = self.view.int_range(e)
        if rng is None:
            return False
        mag = max(abs(rng[0]), abs(rng[1]))
        if isinstance(ctx, MPBFixedContext):
            return ctx.nmin < 0 and ctx.overflow is not OverflowMode.WRAP and mag < ctx.maxval()
        if isinstance(ctx, _FLOAT_CTXS) and isinstance(ctx.pmax, int):
            return mag < 2 ** ctx.pmax and (not isinstance(ctx, SizedContext) or mag < ctx.maxval())
        return False

    def _active_ctx(self, e: Expr) -> object:
        """The context *e* is evaluated under, resolved, or `None`."""
        if e not in self.ctx_use.use_to_scope:
            return None
        scope = self.ctx_use.find_scope_from_use(e)
        return self.scopes.get(scope, scope.ctx)

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
                  | Min() | Max()
                  | Mod() | Fmod() | Remainder()):
                if (ts := self._msbs(*self._operands(e))) is not None:
                    store.le_max(t, ts)
            case AMin() | AMax():
                # one element, and not every element: the worst one
                if (w := self._witness(e)) is not None:
                    store.le(t, w)
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
                # `n` terms below `2 ** (L + 1)` sum below `2 ** (L + 1 + ceil(log2 n))`,
                # `L` the worst term's
                w, n = self._witness(e), self._len_of(e.arg)
                if w is not None and n is not None and n > 0:
                    store.le(t, w + math.ceil(math.log2(n)))
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
                elif src is not None and isinstance(ctx := self._active_ctx(e), _FLOAT_CTXS):
                    # a float format keeps the binade, or carries out of it --
                    # a tiny value as far as the least subnormal; an overflow
                    # is an infinity, or the bound, which is smaller
                    rm = None if ctx.is_stochastic() else ctx.rm
                    if not _round_will_carry(rm):
                        store.le(t, src)
                    elif rm not in _NEAREST and (expmin := getattr(ctx, 'expmin', None)) is not None:
                        store.le_max(t, [src + 1, expmin])
                    else:
                        store.le(t, src + 1)
            case Exp2() | Pow() if (k := self._exp2_arg(e)) is not None:
                store.le(t, k)
            case _:
                pass    # no rule: the term keeps whatever its seed gave it

    def _emit_finite(self, e: Expr, logb: Term) -> None:
        """Relate *e*'s finiteness to its operands': *e* finite implies each
        strict operand is; under a context that never overflows, finite total
        operands imply *e* is."""
        match e:
            case Sum() | AMin() | AMax() if (every := self._every_finite(e.arg)) is not None:
                for h in self.store.high(logb):
                    if isinstance(e, Sum):
                        self.store.imply(h, every)
                    if self._never_overflows(e):
                        self.store.imply(every, h)
            case Exp2() | Pow() if (k := self._exp2_arg(e)) is not None:
                # `2 ** inf` is `inf`, and `2 ** k` finite for finite `k`
                self._implied_high(k, [logb])
                if self._never_overflows(e):
                    self._implied_high(logb, [k])
        for op in self._strict_operands(e):
            if (t := self._msb_of(op)) is not None:
                self._implied_high(t, [logb])
        total = self._total_operands(e)
        if total and self._never_overflows(e) and (msbs := self._msbs(*total)) is not None:
            self._implied_high(logb, msbs)

    @staticmethod
    def _strict_operands(e: Expr) -> tuple[Expr, ...]:
        """The operands of *e* that make it not finite where they are not."""
        if isinstance(e, (*_PROPAGATING, Sqrt)):
            return tuple(e.args)
        match e:
            case Div() | Copysign():
                return (e.first,)
        return ()

    def _total_operands(self, e: Expr) -> tuple[Expr, ...] | None:
        """The operands that, finite, make *e* finite but for an overflow;
        ``None`` for an operation that can fail on finite ones (a division, a
        root, a remainder) or has no rule here."""
        if isinstance(e, (*_PROPAGATING, Min, Max, Fdim)):
            return tuple(e.args)
        match e:
            case Copysign():
                return (e.first,)
            case IfExpr():
                return tuple(self._live_arms(e))
            case ListExpr():
                return tuple(e.elts)
        return None

    def _finite_guard(self, e: Expr, logb: Term) -> tuple[int, ...]:
        """What *e*'s exact rules hold under: nothing, where the result is
        finite or is not only because an operand is not -- then its `logb` is
        `+inf` and so is the rule's bound; else *e*'s `high`."""
        if self._total_operands(e) is not None and self._never_overflows(e):
            return ()
        return self._high_if(e, logb)

    def _never_overflows(self, e: Expr) -> bool:
        """Does *e*'s context leave a finite result finite?"""
        if e not in self.ctx_use.use_to_scope:
            return True
        ctx = self._active_ctx(e)
        if isinstance(ctx, PartialContext):
            return ctx.cls is MPFixedContext
        if isinstance(ctx, MPBFixedContext):
            return ctx.overflow in (OverflowMode.WRAP, OverflowMode.SATURATE)
        return isinstance(ctx, RealContext | MPFixedContext | MPFloatContext | MPSFloatContext)

    def _implied_high(self, of: Term, by: Sequence[Term]) -> None:
        """Every one of *by* finite implies *of* is."""
        premises = [lit for t in by for lit in self.store.high(t)]
        for b in self.store.high(of):
            self.store.imply(premises, b)

    def _every_finite(self, lst: Expr) -> int | None:
        """A literal for "every element of *lst* is finite"."""
        match lst:
            case Var():
                d = self.def_use.find_def_from_use(lst)
                elt, keys = self._def(d), self._list_binders(d, lst)
            case ListComp() if (found := self.out.by_expr.get(lst)) is not None:
                elt, keys = found, self._binders_of.get(lst, self._binders)
            case _:
                return None
        lit = self.store.literal(keys)
        self._every(lit, elt)
        return lit

    def _emit_grid(self, e: Expr, g: Term) -> None:
        """Bound *e*'s least significant digit from its operands'."""
        match e:
            # `Add`/`Sub` take the finer of the two, which aligned summands share
            case (Neg() | Abs() | Copysign() | IfExpr() | ListExpr()
                  | Add() | Sub() | Sum()):
                if (gs := self._lsbs(*self._operands(e))) is not None:
                    self._grid_ge(g, *gs)
            case Mul():
                lhs, rhs = self._lsb_of(e.first), self._lsb_of(e.second)
                if lhs is not None and rhs is not None:
                    self._grid_ge(g, lhs + rhs)
            case Round() | Cast():
                # the result is a multiple of the quantum it rounded at
                found = self._rounding(e)
                if found is not None:
                    self._grid_ge(g, found[0] + 1)
            case Exp2() | Pow() if (k := self._exp2_arg(e)) is not None:
                self._grid_ge(g, k)   # one digit, at position `k`
            case _:
                pass    # no rule: the grid keeps whatever its seed gave it

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
            # under *e*'s own binders, not the asker's: a value is memoized,
            # and one built outside a loop is shared by uses inside it
            outer = self._binders
            self._binders = self._binders_of.get(e, outer)
            terms.value = self._value_uncached(e)
            self._binders = outer
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
            case Logb() | Neg() | Add() | Sub() if not self._exact_int(e):
                # rounded: no longer the exact integer the rules below give
                return self._fresh_value(e) if self.view.int_range(e) else None
            case Logb():
                t = self._msb_of(e.arg)
                of: Expr | Definition = e.arg
                if isinstance(e.arg, Var):
                    d = self.def_use.find_def_from_use(e.arg)
                    of = d
                    t = t if t is not None else self._def(d).msb
                # What anchors the term absolutely -- where the argument is
                # non-zero, since `logb(0)` is `-inf`.  A `with` at a position
                # built from it establishes so (`_visit_context`).
                if t is not None:
                    self._seed(t, of, self.view.logb_range(of)[0], None, low=store.not_neg_inf(t))
                return t
            case Neg():
                inner = self.value_of(e.arg)
                return None if inner is None else -inner
            case Add() | Sub():
                lhs, rhs = self.value_of(e.first), self.value_of(e.second)
                if lhs is None or rhs is None:
                    return None
                return lhs + rhs if isinstance(e, Add) else lhs - rhs
            case IfExpr() if self._one_space(e, *self._live_arms(e)):
                m = self._merge_arms(
                    f'V{len(self.out.by_expr)}', e.cond,
                    self.value_of(e.ift), self.value_of(e.iff),
                )
                self._mark_elt(e, m)
                return m
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
                return self.value_of(e.elt) if len(e.targets) == 1 else None
            case Var() if self._is_unvalued(self.def_use.find_def_from_use(e)):
                return self._fresh_value(e) if self.view.int_range(e) else None
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
        if vacuous is not None:
            self.store.ge_min(m, [iff, *vacuous])
        # Each arm is reached only where its facts hold, so `m` is at least the
        # lesser arm floor under them.  A constant, which carries none of the
        # `else` arm's facts onto the `then` path.
        facts = self._else_facts(cond)
        if facts:
            lo = min(self._floor(ift), self._floor(iff, facts))
            if isinstance(lo, int):
                self.store.ge(m, lo)
        return m

    def _floor(self, t: Term, assuming: frozenset[int] = frozenset()) -> int | float:
        """The least value *t* can take; `-inf` where nothing floors it."""
        return -self.store.maximum(-t, assuming)

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

    def _else_facts(self, cond: Expr) -> frozenset[int]:
        """The literals *cond* failing proves: each value some path zeroes
        alone is non-zero.  Not one tested of every element -- `all(...)`
        failing names no element."""
        paths = self._zero_paths(cond, exact=True)
        if not paths:
            return frozenset()
        every = self._universal_zeros(cond)
        summaries = self._summary_vars()
        return frozenset(
            lit for atoms in paths if len(atoms) == 1
            for t in atoms if t not in every
            for (v, _), lit in zip(t.coeffs, self.store.not_neg_inf(t))
            if v.index not in summaries
        )

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
                # over a comprehension, and over the list a loop filled.  Only
                # one path is every element's: `all(a or b)` is not `all(a)`.
                paths = self._zero_paths(cond)
                return paths[0] if paths is not None and len(paths) == 1 else set()
            case Var():
                return self._universal_zeros_of(
                    self.def_use.find_def_from_use(cond)
                )
            case _:
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

    def _zero_paths(self, cond: Expr, exact: bool = False) -> list[set[Term]] | None:
        """One set of zeroed `logb`s per path making *cond* true.

        ``None`` where some path zeroes nothing.  Lowering leaves `a or b` as a
        phi over two conditions and `all(xs)` as a reduce over a list an inner
        loop filled, so both are read through.

        With *exact*, each path must also *make* *cond* true, which is what a
        failed *cond* can be read back from: a phi is not read, since it may be
        an `and`.
        """
        match cond:
            case Compare(ops=(CompareOp.EQ,), args=(lhs, rhs)):
                for value, other in ((rhs, lhs), (lhs, rhs)):
                    if self.view.int_value(value) == 0:
                        t = self._msb_of(other)
                        return None if t is None else [{t}]
                return None
            case Or():
                # `And` is the dual and is not handled
                return self._zero_paths_all(cond.args, exact)
            case AllOf(arg=ListComp() as comp):
                # every element true, so whatever the element tests is zero
                return self._every_element(
                    comp, self._zero_paths(comp.elt, exact),
                    [self._binder(comp, k) for k in range(len(comp.targets))])
            case AllOf(arg=Var() as xs):
                # ... and the same after the comprehension became a loop
                d = self.def_use.find_def_from_use(xs)
                r = self._built_region(d)
                elt = self._built_expr.get(r) if r is not None else None
                return None if elt is None else self._every_element(
                    xs, self._zero_paths(elt, exact), [self._elements(d)])
            case Var():
                return self._zero_paths_of(self.def_use.find_def_from_use(cond), exact)
            case _:
                return None

    def _every_element(
        self, lst: Expr, paths: list[set[Term]] | None, over: list[Key],
    ) -> list[set[Term]] | None:
        """*paths*, an element's, as `all` over the list *lst* reads them.  An
        empty list makes `all` true and zeroes nothing, so where *lst* may be
        empty a path keeps only values varying with its element (*over*), which
        an empty list has none of; one keeping nothing leaves no paths."""
        n = self._len_of(lst)
        if paths is None or (n is not None and n > 0):
            return paths
        spaces = self.store.spaces
        classes = {spaces.find(k) for k in over}
        out = []
        for atoms in paths:
            kept = {t for t in atoms if any(spaces.find(k) in classes for k in self._varies(t))}
            if not kept:
                return None
            out.append(kept)
        return out

    def _zero_paths_of(self, d: Definition, exact: bool = False) -> list[set[Term]] | None:
        """*d*'s paths, taking a merge as the paths that reach it."""
        if isinstance(d, PhiDef):
            if d.is_loop or exact:
                return None     # carried, or maybe an `and`, not the `a or b` this reads
            return self._zero_paths_all(
                [self.def_use.defs[i] for i in (d.lhs, d.rhs)]
            )
        return self._zero_paths(d.site.expr, exact) if isinstance(d.site, Assign) else None

    def _zero_paths_all(
        self, branches: Sequence[Expr] | Sequence[Definition], exact: bool = False,
    ) -> list[set[Term]] | None:
        """Every branch's paths together, or `None` if any branch has none."""
        found: list[set[Term]] = []
        for b in branches:
            part = (
                self._zero_paths(b, exact) if isinstance(b, Expr)
                else self._zero_paths_of(b, exact)
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
        r = self._built_region(d)
        known = self._built_terms(r) if r is not None else self.out.by_def.get(d)
        if known is not None and known.value is not None:
            return known.value
        return self.value_of(d.site.expr) if isinstance(d.site, Assign) else None

    def _visit_branches(self, stmt: If1Stmt | IfStmt, cond: Expr) -> None:
        """Merge what the two paths through *stmt* leave behind."""
        phis = self.def_use.phis[stmt]
        then_lits = self._arm_non_finite(stmt, 0) if phis else []
        else_lits = self._arm_non_finite(stmt, 1) if phis else []
        then_fin = self._arm_finite(stmt, 0) if phis else frozenset()
        else_fin = self._arm_finite(stmt, 1) if phis else frozenset()
        for phi in phis:
            # `lhs` is the `then` arm of an `if`/`else`, but the *untaken*
            # path of a one-armed `if`, where the body is `rhs`.
            ift, iff = self.def_use.defs[phi.rhs], self.def_use.defs[phi.lhs]
            if isinstance(stmt, IfStmt):
                ift, iff = iff, ift
            self._unify_elements(phi, ift)
            self._unify_elements(phi, iff)
            self._finite_backward(phi, (then_fin | self._implied_by(ift)) & (else_fin | self._implied_by(iff)))
            # An arm with no digits -- no finite non-zero value -- states
            # nothing, so it is dropped rather than joined, as
            # `_visit_return` drops a path returning only infinities.
            # So is an arm leaving a built element unwritten: it is uninitialized.
            unwritten = {d for d in (ift, iff) if phi in self._in_build and d not in self._in_build}
            live = [
                self.out.by_def.get(d)
                for d in (ift, iff)
                if self.view.has_digits(d) and d not in unwritten
            ]
            if live and all(t is not None for t in live):
                # where an arm left out is not finite, so is the phi
                dropped = any(self.view.has_non_finite(d) for d in (ift, iff)
                              if not self.view.has_digits(d))
                self._bracket(self._def(phi), phi, live, finite=dropped)  # type: ignore[arg-type]
            if unwritten:
                value = next((self._def_value(d) for d in (ift, iff) if d not in unwritten), None)
            else:
                value = self._merge_arms(
                    f'V{len(self.out.by_def)}', cond,
                    self._def_value(ift), self._def_value(iff),
                ) if self._one_space(phi, ift, iff) else None
            if value is not None:
                self.out.by_def.setdefault(phi, Terms()).value = value
                self._mark_elt(phi, value)
            for lit in then_lits:
                self._untaken(phi, iff, (lit,))
            for lit in else_lits:
                self._untaken(phi, ift, (lit,))

    def _arm_finite(self, stmt: If1Stmt | IfStmt, arm: int) -> frozenset[int]:
        """The literals "*d* is finite" for each *d* that arm *arm* of *stmt*
        runs only where finite, and that is one value wherever *stmt* runs."""
        non_finite = ValueClass.NAN | ValueClass.INF
        return frozenset(
            self._lit(d) for d, cls in self._classes.arm_facts.get(stmt, ((), ()))[arm]
            if not cls & non_finite and _encloses(self._loops_of_def(d), self._loops)
        )

    def _implied_by(self, d: Definition) -> frozenset[int]:
        """What *d* being finite implies."""
        msb = self.out.by_def[d].msb if d in self.out.by_def else None
        if msb is None or len(msb.coeffs) != 1:
            return frozenset()
        return self.store.closed(frozenset(self.store.high(msb)))

    def _finite_backward(self, phi: Definition, implied: frozenset[int]) -> None:
        """*phi* is finite only where the arm it took is, so it implies
        *implied*: what both arms, finite, would."""
        msb = self._def(phi).msb if isinstance(self._type_of(phi), RealType) else None
        for h in self.store.high(msb) if msb is not None else ():
            for lit in implied:
                self.store.imply(h, lit)

    def _arm_non_finite(self, stmt: If1Stmt | IfStmt, arm: int) -> list[int]:
        """The literals "*d* is finite" for each *d* that arm *arm* (0 `then`, 1
        `else`) of *stmt* runs only where non-finite: where one holds, that
        arm is untaken."""
        out: list[int] = []
        for d, cls in self._classes.arm_facts.get(stmt, ((), ()))[arm]:
            if not cls & (ValueClass.ZERO | ValueClass.FINITE):
                for lit in self._finite_lits(d):
                    if lit not in out:
                        out.append(lit)
        return out

    def _finite_lits(self, d: Definition) -> list[int]:
        """Literals each implying "*d* is finite": its own, and where *d*
        reads an element over the whole of a list nothing changes, "every
        element is finite", at whatever index and in whatever iteration."""
        out = [self._lit(d)]
        if isinstance(d, AssignDef) and isinstance(d.site, Assign) and (
            isinstance(ref := d.site.expr, ListRef)
            and isinstance(lst := ref.value, Var)
            and isinstance(self._type_of(ref), RealType)
            and self._covers(ref.index, self._len_of(lst))
        ):
            d_lst = self.def_use.find_def_from_use(lst)
            region = self._classes.alias.region_of(d_lst)
            if region is not None and not self._classes.alias.may_change(d_lst):
                lit = self._elt_guards.get(region)
                if lit is None:
                    lit = self._elt_guards[region] = self.store.literal(self._list_binders(d_lst, ref))
                self._every(lit, self._def(d_lst))
                out.append(lit)
        return out

    def _lit(self, d: Definition) -> int:
        """The literal "*d* is finite": its `msb`'s `high`, which a parameter
        shares with its argument along with the variable."""
        lit = self._guards.get(d)
        if lit is None:
            msb = self._def(d).msb
            if msb is not None and len(msb.coeffs) == 1:
                (lit,) = self.store.high(msb)
            else:
                lit = self.store.literal(self._def_binders(d))
            self._guards[d] = lit
        return lit

    def _list_binders(self, d: Definition, at: Expr) -> tuple[Key, ...]:
        """The binders the list *d*, read at *at*, is one value for: those
        around the read but its own elements', which a literal about every
        element does not vary with."""
        spaces = self.store.spaces
        own = {spaces.find(k) for k in self._all_elements(d)}
        return tuple(k for k in self._binders_of.get(at, self._binders) if spaces.find(k) not in own)

    def _every(self, lit: int, elt: Terms) -> None:
        """*lit* says every element is finite, so each one's `high`."""
        for h in self.store.high(elt.msb) if elt.msb is not None else ():
            self.store.imply(lit, h)

    def _untaken(self, phi: Definition, other: Definition, guard: tuple[int, ...]) -> None:
        """*phi* is *other* wherever *guard* holds, the arm untaken."""
        src = self.out.by_def.get(other)
        if src is not None and self._one_space(phi, other):
            self._one_of(self._def(phi), [replace(src, value=self._def_value(other))], guard)

    def _unvalued(self) -> set[Region]:
        """The regions whose lists are region summaries (digit-bound.md,
        "Lists"): stored into while another name's definition may be live, or
        through a row; stored into by a loop iterating them; not index
        functions (:meth:`_not_index_functions`); or handed to a callee that
        may store into them.  Drops those regions' builders, and records the
        definitions inside a builder."""
        alias = self._classes.alias
        placed = dict(_placed(self.func.body))
        order = {s: k for k, s in enumerate(placed)}

        def later(d: Definition, w: Stmt) -> bool:
            # defined only past *w*, in no loop around it: dead wherever *w* runs
            site = d.site
            return (isinstance(site, Stmt) and site in order and order[site] > order[w]
                    and not set(placed[site]) & set(placed[w]))

        out: set[Region] = set()
        for stmt in placed:
            match stmt:
                case IndexedAssign():
                    r = alias.region_of(self.def_use.find_def_from_use(stmt), len(stmt.indices) - 1)
                    if r is not None and any(
                        len(stmt.indices) > 1 or (d.name != stmt.var and not later(d, stmt))
                        for d in alias.defs_in(r)
                    ):
                        out.add(r)
                case ForStmt(iterable=Var() as xs):
                    r = alias.region_of(self.def_use.find_def_from_use(xs))
                    if r is not None and any(
                        isinstance(s, IndexedAssign)
                        and alias.region_of(self.def_use.find_def_from_use(s)) == r
                        for s in _stmts(stmt.body)
                    ):
                        out.add(r)
        out |= self._not_index_functions(placed)
        calls = (e for stmt in placed for e in _exprs(stmt) if isinstance(e, Call))
        writes: dict[FuncDef, frozenset[int]] = {}
        for e in calls:
            if isinstance(e.fn, Function):
                if e.fn.ast not in writes:
                    writes[e.fn.ast] = _written_params(e.fn.ast)
                for i in writes[e.fn.ast]:
                    self._written_args.append(e.args[i])
                    r = alias.region_of_expr(e.args[i])
                    while r is not None:
                        out.add(r)
                        r = alias.region_at(r)
        self._builders = {
            r: b for r, b in self._builders.items()
            if not any(alias.region_of(b[1], k) in out for k in range(self._depth(b[1])))
        }
        for d in self.def_use.defs:
            r = alias.region_of(d) if isinstance(self._type_of(d), ListType) else None
            site = d.site
            if r in self._builders and isinstance(site, Stmt) and self._builders[r][0] in placed.get(site, ()):
                self._in_build.add(d)
        return out

    def _not_index_functions(self, placed: dict[Stmt, tuple[Stmt, ...]]) -> set[Region]:
        """The regions whose lists are not index functions (digit-bound.md,
        "Lists"): stored into other than by a :meth:`_builder`, or carried by
        another loop.  Records each builder, and the phis that are read."""
        alias = self._classes.alias
        stores: dict[Region, list[IndexedAssign]] = {}
        for stmt in placed:
            if isinstance(stmt, IndexedAssign):
                r = alias.region_of(self.def_use.find_def_from_use(stmt))
                if r is not None:
                    stores.setdefault(r, []).append(stmt)
        # A phi is read where something uses it, or a phi that is read takes it:
        # an iteration then sees what an earlier one left.  A list built anew
        # in each iteration has a phi nothing reads.
        read = {d for d, us in self.def_use.uses.items() if isinstance(d, PhiDef) and us}
        todo = list(read)
        while todo:
            p = todo.pop()
            for q in (self.def_use.defs[p.lhs], self.def_use.defs[p.rhs]):
                if isinstance(q, PhiDef) and q not in read:
                    read.add(q)
                    todo.append(q)
        self._read_phis = read
        out: set[Region] = set()
        for d in self.def_use.defs:
            r = alias.region_of(d) if isinstance(self._type_of(d), ListType) else None
            if r is None or r in out:
                continue
            built = self._builder(r, stores.get(r, []), placed)
            loop = built[0] if built is not None else None
            carried = isinstance(d, PhiDef) and d.is_loop and d.site is not loop and d in read
            if carried or (stores.get(r) and built is None):
                out.add(r)
            elif built is not None:
                self._builders[r] = built
        return out

    def _builder(
        self, r: Region, stores: list[IndexedAssign], placed: dict[Stmt, tuple[Stmt, ...]],
    ) -> tuple[ForStmt, Definition, PhiDef] | None:
        """The loop building the list of region *r*, the `empty` it fills and
        the loop's phi of it,
        where every store into it writes element `k` at the `k`-th iteration
        of that loop, the innermost around it, for every `k`, and the region
        holds no other object."""
        loops = {placed[stmt][-1] if placed[stmt] else None for stmt in stores}
        if len(loops) != 1 or not isinstance(loop := loops.pop(), ForStmt):
            return None
        for stmt in stores:
            if len(stmt.indices) != 1:
                return None
            idx = stmt.indices[0]
            # the loop's own index, or a count of its iterations (a gather)
            target = isinstance(idx, Var) and isinstance(loop.iterable, Range1) and isinstance(loop.target, NamedId) and (
                self.def_use.find_def_from_use(idx) == self.def_use.find_def_from_site(loop.target, loop))
            if not target and not self._counts(loop, idx):
                return None
        alias = self._classes.alias
        phis = [p for p in self.def_use.phis.get(loop, ()) if alias.region_of(p) == r]
        if len(phis) != 1 or not _is_empty_alloc(alloc := self.def_use.defs[phis[0].lhs]):
            return None
        n = _size(self.array_size.by_def.get(alloc))
        if n is None or self._len_of(loop.iterable) != n:
            return None
        # the `empty` is the region's one object: another put in it later is not built
        for d in alias.defs_in(r):
            site = d.site
            if site is not alloc.site and (isinstance(site, Argument) or (
                    isinstance(site, Assign) and not isinstance(site.expr, Var | ListRef | IfExpr))):
                return None
        return loop, alloc, phis[0]

    def _inner_region(self, of: Expr | Definition) -> Region | None:
        """The region holding the innermost elements of the list *of*."""
        alias = self._classes.alias
        r = alias.region_of(of) if isinstance(of, Definition) else alias.region_of_expr(of)
        for _ in range(self._depth(of) - 1):
            r = alias.region_at(r) if r is not None else None
        return r

    def _bound_summaries(self) -> None:
        """Each read of a list that is not an index function is one of the
        values entering its region, at some index and iteration (digit-bound.md,
        "Lists")."""
        entries, made = self._region_entries()
        spaces = self.store.spaces
        for terms, of, r in self._summary_reads:
            if r is None or not (srcs := entries.get(r)) or terms.msb is None:
                continue
            # what a read is one of is at its own iteration of every loop the
            # region's objects are all made in, and at some point of the rest
            here = (*self.params.outer, *made.get(r, ()))
            kept = {spaces.find(k) for k in here}
            at = self._binders_of.get(of, self._binders) if isinstance(of, Expr) else self._def_binders(of)
            anywhere = []
            for i, src in enumerate(srcs):
                moved = {k for t in (src.msb, src.lsb, src.value) if t is not None
                         for k in self._varies(t) if spaces.find(k) not in kept}
                # each read is its own value: a witness of its own, varying with it
                anywhere.append(self._copy(src, moved, _Witness((id(terms), i)), at) if moved else src)
            self._bracket(terms, of, anywhere)

    def _region_entries(self) -> tuple[dict[Region, list[Terms]], dict[Region, tuple[Key, ...]]]:
        """What enters each region of innermost list elements: a parameter's
        or an allocation's elements, and every value stored.  A region with
        an entry nothing bounds -- a callee's store, an unmodeled list -- has
        none: nothing bounds what it holds.  And the binders every object in
        the region is made inside, where no phi carries one past an
        iteration: a read there sees that iteration's objects alone."""
        out: dict[Region, list[Terms]] = {}
        unknown: set[Region] = set()
        made: dict[Region, set[Key]] = {}
        carried: dict[Region, set[Key]] = {}
        for d in self.def_use.defs:
            if not self._depth(d) or (r := self._inner_region(d)) is None:
                continue
            match d.site:
                case Argument():
                    made[r] = set()
                    t = self.out.by_def.get(d)
                case IndexedAssign() as w:
                    t = self.out.by_expr.get(w.expr)
                case Assign(expr=Var() | ListRef() | IfExpr()):
                    continue    # names a list in the region
                case Assign(expr=x) as site:
                    around = set(self._binders_of.get(site, ())) - set(self.params.outer)
                    made[r] = around if r not in made else made[r] & around
                    if isinstance(x, Empty):
                        continue    # holds nothing yet
                    t = self.out.by_expr.get(x)
                case ForStmt() | WhileStmt() as loop if d in self._read_phis:
                    carried.setdefault(r, set()).add(self._binder(loop))
                    continue
                case _:
                    continue    # a phi or a loop target: names a list in the region
            if t is None or t.msb is None:
                unknown.add(r)
            else:
                out.setdefault(r, []).append(t)
        unknown.update(r for a in self._written_args if (r := self._inner_region(a)) is not None)
        fixed = {r: tuple(made.get(r, set()) - carried.get(r, set())) for r in out}
        return {r: ts for r, ts in out.items() if r not in unknown}, fixed

    def _is_unvalued(self, d: Definition) -> bool:
        """Is *d* a region summary at any depth (:meth:`_unvalued`)?  A list
        holding such rows is."""
        alias = self._classes.alias
        return any(alias.region_of(d, k) in self._unvalued_regions for k in range(self._depth(d)))

    def _assumed(self, e: Expr) -> frozenset[int]:
        """The guard literals *e* may assume: each definition
        `ValueClassInfer` proves finite where *e* is evaluated, and only where
        that definition is one value for every evaluation of *e* -- a
        definition in a loop *e* is outside of is one per iteration.

        And what every `with` dominating *e* makes true; see
        :meth:`_visit_context`."""
        outer = self.outer()
        outer |= frozenset(self._arms_of.get(e, ()))
        summaries = self._summaries if self._summaries is not None else self._summary_vars()
        outer |= frozenset(
            lit for v, lit in self._established_at.get(e, ()) if v not in summaries
        )
        at = self._loops_of_expr.get(e)
        if at is None:
            return outer
        non_finite = ValueClass.NAN | ValueClass.INF
        return outer | frozenset(
            lit for d, lit in self._guards.items()
            if _encloses(self._loops_of_def(d), at)
            and not self._classes.class_at(d, e) & non_finite
        ) | frozenset(
            lit for region, lit in self._elt_guards.items()
            if not self._classes.elements_at(region, e) & non_finite
        )

    def _loops_of_def(self, d: Definition) -> tuple[Stmt, ...] | None:
        """The loops *d* takes a value in, or ``None`` where not known."""
        site = d.site
        if isinstance(site, Argument):
            return ()
        loops = self._loops_of_stmt.get(site) if isinstance(site, Stmt) else None
        if loops is None:
            return None
        # a loop's own target and phis take a value per iteration of it
        return (*loops, site) if isinstance(site, ForStmt | WhileStmt) else loops

    def _visit_context(self, stmt: ContextStmt, ctx: Any) -> None:
        """Constructing a fixed-point context needs its position finite, so
        whatever the `with` dominates may assume it."""
        self._visit_expr(stmt.ctx, ctx)
        scope = self._scope_of.get(stmt)
        found = self._position(scope) if scope is not None else None
        if found is not None:
            pos = found[0]
            for lits in (self.store.not_neg_inf(pos), self.store.high(pos)):
                self._established |= {(v.index, lit) for (v, _), lit in zip(pos.coeffs, lits, strict=True)}
        self._visit_block(stmt.body, ctx)

    def _summary_vars(self) -> set[int]:
        """The variables some list's terms name.  Each stands for every element,
        so a fact about one element -- a failed `xs[i] == 0`, or a `with` at
        `logb(xs[i]) - k` -- is no fact about it."""
        out: set[int] = set()
        for of, terms in (*self.out.by_expr.items(), *self.out.by_def.items()):
            if isinstance(self._type_of(of), ListType):
                for t in (terms.msb, terms.lsb, terms.value):
                    if t is not None:
                        out.update(v.index for v, _ in t.coeffs)
        return out

    def _visit_block(self, block: StmtBlock, ctx: Any) -> None:
        # a `with` dominates the rest of its block, and nothing past it
        established = self._established
        super()._visit_block(block, ctx)
        self._established = established

    def _visit_statement(self, stmt: Stmt, ctx: Any) -> Any:
        self._arms_of[stmt] = self._arms
        self._loops_of_stmt[stmt] = self._loops
        self._binders_of[stmt] = self._binders
        return super()._visit_statement(stmt, ctx)

    def _visit_while(self, stmt: WhileStmt, ctx: Any) -> None:
        # A binder no copy moves and no equation names, so its phis are not
        # joined: what the body computes varies with its iterations, and past
        # it a phi has terms of its own (digit-bound.md, "Control").
        w = self._binder(stmt)
        self.store.spaces.open(w, self._binders)
        loops, self._loops = self._loops, (*self._loops, stmt)
        binders, self._binders = self._binders, (*self._binders, w)
        super()._visit_while(stmt, ctx)
        self._binders, self._loops = binders, loops
        for phi in self.def_use.phis[stmt]:
            if isinstance(self._type_of(phi), RealType | ListType):
                terms = Terms()
                self._fresh_interval(terms, phi, f'x{len(self.out.by_def)}')
                self.out.by_def[phi] = terms

    def _visit_if1(self, stmt: If1Stmt, ctx: Any) -> None:
        self._visit_expr(stmt.cond, ctx)
        self._visit_arm(stmt.body, ctx)
        self._visit_branches(stmt, stmt.cond)

    def _visit_if(self, stmt: IfStmt, ctx: Any) -> None:
        self._visit_expr(stmt.cond, ctx)
        self._visit_arm(stmt.ift, ctx)
        self._visit_arm(stmt.iff, ctx)
        self._visit_branches(stmt, stmt.cond)

    def _visit_arm(self, block: StmtBlock, ctx: Any) -> None:
        """*block* under a literal "this arm runs", which guards what holds
        only there and every expression in it assumes."""
        arms, self._arms = self._arms, (*self._arms, self.store.literal(self._binders))
        self._visit_block(block, ctx)
        self._arms = arms

    def _fresh_value(self, e: Expr) -> Term:
        """A value variable for *e*, bounded by the range its format gives."""
        v = self._var(f'V{len(self.out.by_expr)}')
        rng = self.view.int_range(e)
        if rng is not None:
            # `min(logb(0), 5)` is not finite
            low = self.store.not_neg_inf(v) if self.view.has_non_finite(e) else ()
            self._seed(v, e, *rng, low=low)
        return v

    def _exp2_arg(self, e: Expr) -> Term | None:
        """*e*'s exponent when it is a power of two, directly or through a name
        bound to one.  Only base two: a general power is irrational, and
        nothing states it exactly."""
        match self.def_use.defining_expr(e):
            case Exp2() as p:
                return self.value_of(p.arg)
            case Pow() as p if self.view.int_value(p.first) == 2:
                return self.value_of(p.second)
            case _:
                return None

    def _rounding(self, e: Expr) -> tuple[Term, RoundingMode | None] | None:
        """The digit position *e* rounds at, and the mode it rounds with.

        A *concrete* fixed-point context counts too: it is unbounded above, so
        what limits the result is the operand's own reach.  A rescaled program
        is entirely of that shape.

        ``None`` for an expression that uses no context at all -- a bare
        literal, say -- which `_active` asks about as readily as a rounding.
        """
        if e not in self.ctx_use.use_to_scope:
            return None
        return self._position(self.ctx_use.find_scope_from_use(e))

    def _position(self, scope: ContextScope) -> tuple[Term, RoundingMode | None] | None:
        """The digit position *scope* rounds at, and the mode it rounds with."""
        ctx = self.scopes.get(scope, scope.ctx)
        pos: object
        rm: object
        if isinstance(ctx, MPFixedContext) or (
            # a bounded one, where an overflow does not wrap or saturate
            isinstance(ctx, MPBFixedContext)
            and ctx.overflow in (OverflowMode.OVERFLOW, OverflowMode.ASSERT)
        ):
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
    # A list has one set of terms, its *representative*, describing an
    # arbitrary element.  A part gets a copy, with every fact replayed onto it
    # (`DigitBoundStore.instance`).
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

    def _range_key(self, e: Expr) -> _Range | None:
        """An identity for a range, so two loops over the same one land on
        the same index set."""
        match e:
            case Range2():
                step: _Operand | None = 1
            case Range3():
                step = self._part_key(e.third)
            case _:
                return None
        start, stop = self._part_key(e.first), self._part_key(e.second)
        if start is None or stop is None or step is None:
            return None
        return _Range(start, stop, step)

    def _part_key(self, e: Expr) -> _Operand | None:
        """*e* as an :data:`_Operand`, or `None`."""
        const = self.view.int_value(e)
        if const is not None:
            return const
        if isinstance(e, Var):
            return self.def_use.find_def_from_use(e)
        return None

    def _elt_key(self, d: Definition) -> _Elts | None:
        """The index set of *d* if it is `a + s * k` for `k` over
        `range(n)`."""
        if not isinstance(d, AssignDef) or not isinstance(d.site, Assign):
            return None
        return self._elt_key_of(d.site.expr)

    def _elt_key_of(self, e: Expr) -> _Elts | None:
        """The index set of *e* if it is `a + s * k` for `k` over
        `range(n)`."""
        if not isinstance(e, Add):
            return None
        for start, step in ((e.first, e.second), (e.second, e.first)):
            pairs = [(step.first, step.second), (step.second, step.first)] \
                if isinstance(step, Mul) else [(None, step)]
            a = self._part_key(start)
            for s, k in pairs:
                if not isinstance(k, Var) or a is None or self._trip is None:
                    continue
                if self.def_use.find_def_from_use(k) is not self._trip[0]:
                    continue
                b = 1 if s is None else self._part_key(s)
                if b is not None:
                    return _Elts(a, b, self._trip[1])
        return None

    def _at_index(self, lst: Var, idx: Expr) -> Terms | None:
        """*lst*'s element terms at *idx*: an index set (:meth:`_at_index_set`),
        a constant `k`, the set ``range(k, k + 1)``, or `a + s * k` written
        inline."""
        if isinstance(idx, Var):
            return self._at_index_set(lst, idx)
        k = self.view.int_value(idx)
        if k is not None:
            return self._instance(lst, _Range(k, k + 1))
        key = self._elt_key_of(idx)
        return None if key is None else self._instance(lst, key, self._trip_binder())

    def _at_index_set(self, lst: Var, idx: Var) -> Terms | None:
        """*lst*'s element terms restricted to the index set *idx* covers:
        the enclosing loop's range, or `a + s * k` over a trip count
        (:meth:`_elt_key`); ``None`` where there is no such set.

        A bound the part builds -- a ``max`` over it -- says nothing about
        the rest, so the part gets variables of its own and the store replays
        onto them every fact that holds at every index.  Two parts are
        related exactly when their loops run over the same range, which is
        what the key is for: one renaming per index set, so ``es`` and
        ``prods`` gathered over the evens keep their pairing.
        """
        d_idx = self.def_use.find_def_from_use(idx)
        key: _Range | _Elts | None = self._elt_key(d_idx)
        if key is None:
            if self._gather is None or d_idx is not self._gather[1]:
                return None
            key = self._gather[0]
            return self._instance(lst, key, self._index_binder.get(d_idx))
        return self._instance(lst, key, self._trip_binder())

    def _trip_binder(self) -> Key | None:
        """The binder of the trip count an `a + s * k` index counts by."""
        return None if self._trip is None else self._index_binder.get(self._trip[0])

    def _enumerated(self, key: _Range | _Elts, binder: Key | None) -> _Range | _Elts | _Apart:
        """The index set *key*, as *binder* enumerates it: one with the binders
        that enumerate it already, where *binder* corresponds with theirs --
        both run over it in order, so their indices are one -- else one apart."""
        if binder is None:
            return key
        firsts = self._enumerators.setdefault(key, [])
        n = next((n for n, b in enumerate(firsts) if b == binder or self.store.spaces.correspond(binder, b)), None)
        if n is None:
            n = len(firsts)
            firsts.append(binder)
        return key if n == 0 else _Apart(key, n)

    def _instance(self, lst: Var, key: _Range | _Elts, binder: Key | None = None) -> Terms:
        """*lst*'s element terms renamed for the index set *key*, as *binder*
        enumerates it; see :meth:`_at_index_set`."""
        d_lst = self.def_use.find_def_from_use(lst)
        src = self._def(d_lst)
        if src.value is None:
            # the part is what a rounding position is built from, and the
            # whole list's value is minted lazily, so force it here
            src.value = self._def_value(d_lst)
        return self._copy(src, (self._elements(d_lst),), self._enumerated(key, binder))

    def _witness(self, e: AMax | AMin | Sum) -> Term | None:
        """The `msb` of the worst element of the list *e* reduces: a copy of
        its element at an index of *e*'s own, so two aggregates over one list
        need not share their worst element."""
        src = self.out.by_expr.get(e.arg)
        if src is None or src.msb is None:
            return None
        return self._copy(src, (self._elements(e.arg),), _Witness(id(e))).msb

    def _copy(
        self,
        src: Terms,
        elements: Iterable[Key],
        key: _IndexKey,
        binders: tuple[Key, ...] | None = None,
    ) -> Terms:
        """*src* renamed onto the index set *key*: at another index of
        *elements*' spaces, varying with *binders* (by default, the walk's)."""
        fields = (src.msb, src.lsb, src.value)
        spaces = self.store.spaces
        elements = set(elements)
        classes = {spaces.find(k) for k in elements}
        insts = self._index_sets.setdefault(key, [])
        inst = next((i for i in insts if {spaces.find(k) for k in i.elements} == classes), None)
        if inst is None:
            inst = _IndexSet(
                f'@{sum(map(len, self._index_sets.values()))}',
                self._binders if binders is None else binders)
            insts.append(inst)
        inst.elements.update(elements)
        # a variable that does not vary with the index is the part's too
        for t in fields:
            for v, _ in (t.coeffs if t is not None else ()):
                if spaces.signature(v.index) & classes and v.index not in inst.subst:
                    inst.subst[v.index] = self.store.var(v.name + inst.tag)
        self._replay(inst)
        return Terms(*(None if t is None else t.rename(inst.subst) for t in fields))

    def _replay_all(self) -> None:
        """Every index set's facts stated since, its callees' included: a
        caller states some after the call returns."""
        for insts in self._index_sets.values():
            for inst in insts:
                self._replay(inst)
        for walk in self._callees:
            walk._replay_all()

    def _replay(self, inst: _IndexSet) -> None:
        """Every fact stated since *inst* was last replayed, renamed onto it."""
        spaces = self.store.spaces
        classes = {spaces.find(k) for k in inst.elements}
        self.store.instance(classes, inst.subst, inst.lits, inst.done, inst.tag)
        for old, new in list(inst.subst.items())[inst.minted:]:
            for v, _ in new.coeffs:
                spaces.copy(v.index, old, classes)
                spaces.mint(v.index, inst.binders)
        inst.minted = len(inst.subst)

    def _counts(self, loop: ForStmt, idx: Expr) -> bool:
        """Does *idx* count *loop*'s iterations: a phi of it entering at 0 and
        stepping by 1?"""
        d = self.def_use.find_def_from_use(idx) if isinstance(idx, Var) else None
        if not isinstance(d, PhiDef) or d not in self.def_use.phis.get(loop, ()):
            return False
        if self.view.int_value(self.def_use.defs[d.lhs]) != 0:
            return False
        # the step, through any names it is copied by
        step = self.def_use.defs[d.rhs].site
        while isinstance(step, Assign) and isinstance(step.expr, Var):
            step = self.def_use.find_def_from_use(step.expr).site
        if not isinstance(step, Assign) or not isinstance(step.expr, Add):
            return False
        add = step.expr
        return any(isinstance(a, Var) and self.def_use.find_def_from_use(a) is d and self.view.int_value(b) == 1
                   for a, b in ((add.first, add.second), (add.second, add.first)))

    def _bind_elt(self, target: NamedId, site: ListComp | ForStmt, source: Var, binder: Key) -> None:
        d_src = self.def_use.find_def_from_use(source)
        d = self.def_use.find_def_from_site(target, site)
        self._unify_elements(d, d_src, 1)    # a row: the list's inner indices
        if self._is_unvalued(d_src):
            t = self._def(d)
            for v, _ in (t.lsb.coeffs if t.lsb is not None else ()):
                self.store.spaces.bound(v.index, self._def_binders(d))
            self._summary_reads.append((t, d, self._inner_region(source)))
            return
        self._share(d, self._element(d_src, binder))

    def _bind_iter(self, target: Id | TupleBinding, iterable: Expr, site: ListComp | ForStmt, binder: Key) -> None:
        """What iterating *iterable* as *target* binds, at *site*: shared by a
        comprehension and the loop it lowers to."""
        if not isinstance(target, NamedId):
            return
        d = self.def_use.find_def_from_site(target, site)
        match iterable:
            case Var():
                # iterating a list directly binds the target to its element
                self._bind_elt(target, site, iterable, binder)
            case Range1():
                # ... while `range(len(xs))` makes the target an index.  Only
                # `Range1`: a range with a start or a step does not visit
                # every index from zero, and covering is the claim that it
                # does ...
                self._index_binder[d] = binder
                if (n := self._len_of(iterable)) is not None:
                    self._loop_index[d] = n
                # ... and a trip count, whose `a + s * k` the body may index by
                if (trip := self._part_key(iterable.arg)) is not None:
                    self._trip = (d, trip)
            case Range2() | Range3():
                # ... it visits a *part*, which `_at_index_set` gives
                # variables of its own.
                self._index_binder[d] = binder
                if (key := self._range_key(iterable)) is not None:
                    self._gather = (key, d)
            case _:
                pass    # anything else binds the target to nothing

    def _visit_list_comp(self, e: ListComp, ctx: Any) -> None:
        outer = self._gather, self._trip
        self._trip = None
        clauses = tuple(self._binder(e, k) for k in range(len(e.targets)))
        for k, c in enumerate(clauses):
            self.store.spaces.open(c, (*self._binders, *clauses[:k]))
        for target, iterable, binder in zip(e.targets, e.iterables, clauses):
            self._visit_expr(iterable, ctx)
            # a `zip` has no counterpart in the lowered loop, so it stays here
            if isinstance(target, TupleBinding) and isinstance(iterable, Zip):
                for sub, part in zip(target.elts, iterable.args):
                    if isinstance(sub, NamedId) and isinstance(part, Var):
                        self._bind_elt(sub, e, part, binder)
            else:
                self._bind_iter(target, iterable, e, binder)
        binders, self._binders = self._binders, (*self._binders, *clauses)
        self._visit_expr(e.elt, ctx)
        self._binders = binders
        self._gather, self._trip = outer

    def _visit_for(self, stmt: ForStmt, ctx: Any) -> None:
        self._visit_expr(stmt.iterable, ctx)
        outer = self._gather, self._trip
        self._trip = None
        b = self._binder(stmt)
        self.store.spaces.open(b, self._binders)
        self._bind_iter(stmt.target, stmt.iterable, stmt, b)
        # a built list's element is the iteration that wrote it
        built = {r: self.store.spaces.correspond(b, self._elements(alloc))
                 for r, (loop, alloc, _) in self._builders.items() if loop is stmt}
        loops, self._loops = self._loops, (*self._loops, stmt)
        binders, self._binders = self._binders, (*self._binders, b)
        self._visit_block(stmt.body, ctx)
        self._binders = binders
        self._loops = loops
        self._gather, self._trip = outer
        for r, paired in built.items():
            self._build(r, stmt, paired)
        # A carried value starts an iteration as the pre-loop value or the
        # iteration before's, and leaves the loop as the last iteration's (or
        # the pre-loop one, where it never runs).  A list's phi states nothing:
        # a built list's shares its terms, and any other is a region summary.
        for phi in self.def_use.phis[stmt]:
            if isinstance(self._type_of(phi), ListType):
                continue
            body = self.out.by_def.get(self.def_use.defs[phi.rhs])
            pre = self.out.by_def.get(self.def_use.defs[phi.lhs])
            terms = Terms()
            self._fresh_interval(terms, phi, f'x{len(self.out.by_def)}')
            if body is not None and pre is not None:
                self._bracket(self._def(phi), phi, [pre, self._copy(body, (b,), _Witness(id(stmt)), (*self._binders, b))])
                self._bracket(terms, phi, [pre, self._copy(body, (b,), _Last(id(stmt)))])
            self.out.by_def[phi] = terms

    def _build(self, r: Region, loop: ForStmt, paired: bool) -> None:
        """The list *loop* builds into region *r*: element `k` is what
        iteration `k` leaves it -- or, where the invariant refuses the loop the
        list's index, what some iteration does."""
        stores = self._built_stores.get(r, [])
        if len(stores) == 1:
            # a list of booleans has no terms, but zero paths read its element
            self._built_expr[r] = stores[0].expr
            if paired and isinstance(stores[0].expr, TupleExpr):
                # a list of tuples has no element term of its own: its fields
                self._built_fields[r] = tuple(self.out.by_expr.get(el) for el in stores[0].expr.elts)
        terms = self._built_terms(r)
        phi = self._builders[r][2]
        left = self.def_use.defs[phi.rhs]
        src = self.out.by_def.get(left) if left in self._in_build else None
        if src is None or src.msb is None:
            return      # no path writes, or nothing bounds what one does
        src = replace(src, value=self._def_value(left))
        alloc = self._builders[r][1]
        if not paired:
            # each element is what some iteration wrote: a witness of its own
            src = self._copy(src, (self._binder(loop),), _Witness((id(loop), id(alloc))),
                             (*self._binders, *self._all_elements(alloc)))
        if src.value is not None:
            terms.value = self._var(f'VB{len(self._built)}')
            self._mark_elt(alloc, terms.value)
        assert src.msb is not None and terms.msb is not None
        self._one_of(terms, [src], ())
        self._implied_high(terms.msb, [src.msb])
        if paired:
            self.store.ge(terms.msb, src.msb)
            self._implied_high(src.msb, [terms.msb])

    def _visit_assign(self, stmt: Assign, ctx: Any) -> None:
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
        if isinstance(stmt.expr, Zip):
            # a `zip`'s element `k` holds each operand's element `k`
            for a in stmt.expr.args:
                if isinstance(a, Var):
                    self._unify_elements(d, self.def_use.find_def_from_use(a))
        match stmt.expr:
            case ListRef(value=Var() as lst, index=Var() as idx) if self._covers(
                idx, self._len_of(lst)
            ) and not self._is_unvalued(self.def_use.find_def_from_use(lst)) and not self._mid_build(stmt.expr):
                # a region summary's read, or one mid-build, is bounded as an expression
                d_src = self.def_use.find_def_from_use(lst)
                if isinstance(d_src.site, Assign) and isinstance(d_src.site.expr, Range1):
                    # `range(n)[i] == i`, so indexing a materialized range with
                    # a full index yields another one.  Comprehension
                    # elimination emits exactly this for a `zip`.
                    d_idx = self.def_use.find_def_from_use(idx)
                    self._loop_index[d] = self._loop_index[d_idx]
                    self._index_binder[d] = self._index_binder[d_idx]
                else:
                    self._share(d, self._element(d_src, self._index_of(idx)))
            case _:
                src = self.out.by_expr.get(stmt.expr)
                if src is None:
                    return
                terms = self._def(d)
                self._unify_elements(d, stmt.expr)
                (src,) = self._aligned(d, [src])
                # One-directional: `_def` seeds *d* from its path-refined
                # format, and an equality would export that bound to every
                # use of the expression.
                if terms.msb is not None and src.msb is not None:
                    self.store.le(terms.msb, src.msb)
                    self._implied_high(src.msb, [terms.msb])
                    self._implied_high(terms.msb, [src.msb])
                if terms.lsb is not None and src.lsb is not None:
                    self._grid_ge(terms.lsb, src.lsb)
                terms.value = terms.value if terms.value is not None else src.value

    def _zip_fields(self, d: Definition) -> tuple[Terms | None, ...] | None:
        """A `zip`'s element field by field, each an element of one operand.

        The same list reaches the walk both as a `zip` and, once comprehension
        elimination has run, as one materialized by a loop.
        """
        if not isinstance(d.site, Assign) or not isinstance(d.site.expr, Zip):
            return None
        # a field is its operand's element only where the two are one index
        own = self.store.spaces.find(self._elements(d))
        return tuple(
            self._def(d_a) if isinstance(a, Var) and not self._is_unvalued(
                d_a := self.def_use.find_def_from_use(a))
            and self.store.spaces.find(self._elements(d_a)) == own else None
            for a in d.site.expr.args
        )

    def _unpack_elt(self, stmt: Assign, target: TupleBinding) -> None:
        match stmt.expr:
            case ListRef(value=Var() as lst, index=Var() as idx) if self._covers(
                idx, self._len_of(lst)
            ):
                d_lst = self.def_use.find_def_from_use(lst)
                binder = self._index_of(idx)
            case _:
                return
        if self._is_unvalued(d_lst):
            return      # a region summary: its fields are seeded alone
        r = self._built_region(d_lst)
        fields = (self._built_fields.get(r) if r is not None else None) or self._zip_fields(d_lst)
        if fields is None:
            return
        here = self._reads_summary(d_lst, binder)
        for sub, part in zip(target.elts, fields):
            if isinstance(sub, NamedId) and part is not None:
                if not here:
                    assert binder is not None
                    part = self._read_apart(part, d_lst, binder)
                self._share(self.def_use.find_def_from_site(sub, stmt), part)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx: Any) -> None:
        for s in stmt.indices:
            self._visit_expr(s, ctx)
        self._visit_expr(stmt.expr, ctx)
        r = self._built_of(self.def_use.find_def_from_use(stmt))
        if r is None:
            return      # a region summary's store: an entry of its region (`_region_entries`)
        d = self.def_use.find_def_from_site(stmt.var, stmt)
        self._unify_elements(stmt.expr, d, 1)    # a row stored whole
        # at the builder's index, the list is the value stored
        if (src := self.out.by_expr.get(stmt.expr)) is not None:
            self._share(d, Terms(src.msb, src.lsb, self.value_of(stmt.expr)))
        self._built_stores.setdefault(r, []).append(stmt)

    def _visit_return(self, stmt: ReturnStmt, ctx: Any) -> None:
        self._visit_expr(stmt.expr, ctx)
        # A caller can use the result as a rounding position, and `value_of`
        # is lazy, so the value has to be named before the walk leaves.
        self.value_of(stmt.expr)
        # A path returning nothing but infinities and NaNs states no
        # magnitude: `logb` bounds the finite values, and this one has none.
        if not self.view.has_finite(stmt.expr):
            self._dropped = True
        else:
            t = self.out.by_expr.get(stmt.expr) or Terms()
            # the result is this return's value at the iterations it ran at,
            # which do not vary with the binders it ran in
            inner = self._binders[len(self.params.outer):]
            if inner:
                t = self._copy(t, inner, _Witness(id(stmt)), self.params.outer)
            self._returns.append((stmt.expr, t))

    def _merge_returns(self) -> Terms:
        """The result is one of the returns: as far as the widest reaches, no
        finer than the coarsest, and valued as one of them.  A return reached
        only where some definition is non-finite is not taken where it is
        finite, so the rest bound the result there too."""
        if len(self._returns) <= 1 and not self._dropped:
            return replace(self._returns[0][1]) if self._returns else Terms()
        tag = len(self.out.by_expr)
        terms = [t for _, t in self._returns]
        if any(isinstance(self._type_of(e), ListType) for e, _ in self._returns):
            # several lists, at one index only where they are one
            terms = self._aligned(self.func, terms, self._merged_elements())
        merged = Terms()
        msbs, lsbs, values = (
            _all(getattr(t, f) for t in terms) for f in ('msb', 'lsb', 'value'))
        # where a return left out ran, the result is not finite
        if msbs:
            merged.msb = self._var(f'msbR{tag}')
            high = self.store.high(merged.msb) if self._dropped else ()
            self.store.le_max(merged.msb, msbs, guard=high)
            if not self._dropped:
                self._implied_high(merged.msb, msbs)
            elif len(msbs) == 1:
                # finite only where the one return that can be ran
                self._implied_high(msbs[0], [merged.msb])
        if lsbs:
            merged.lsb = self._var(f'lsbR{tag}')
            self._lsb_vars.update(v.index for v, _ in merged.lsb.coeffs)
            self._grid_ge(merged.lsb, *lsbs)
        if values:
            merged.value = self._var(f'VR{tag}')
            guard = (*self.store.high(merged.value), *self.store.not_neg_inf(merged.value)) if self._dropped else ()
            self.store.le_max(merged.value, values, guard=guard)
            self.store.ge_min(merged.value, values, guard=guard)
        for d in self._non_finite_defs():
            taken = [t for e, t in self._returns
                     if self._classes.class_at(d, e) & (ValueClass.ZERO | ValueClass.FINITE)]
            if taken and len(taken) < len(terms):
                for lit in self._finite_lits(d):
                    self._one_of(merged, taken, (lit,))
        return merged

    def _merged_elements(self) -> tuple[Key, ...]:
        """The element spaces of the list several returns merge into, each one
        with each return's where it can be."""
        keys = tuple(self._elements(self.func, k) for k in range(self._depth(self._returns[0][0])))
        for e, _ in self._returns:
            for k, key in enumerate(keys):
                self.store.spaces.correspond(key, self._elements(e, k))
        return keys

    def _return_elements(self) -> tuple[Key, ...]:
        """The element spaces of the list the function returns: one return's,
        or, for several, their merge's."""
        lists = [e for e, _ in self._returns if isinstance(self._type_of(e), ListType)]
        if not lists:
            return ()
        if len(self._returns) == 1 and not self._dropped:
            return self._all_elements(lists[0])
        keys = self._merged_elements()
        ret = self.out.ret
        for t in (ret.msb, ret.value):
            for v, _ in (t.coeffs if t is not None else ()):
                self.store.spaces.bound(v.index, keys)
        return keys

    def _non_finite_defs(self) -> list[Definition]:
        """Every definition some arm is reached only where it is non-finite."""
        return list(dict.fromkeys(
            d for arms in self._classes.arm_facts.values()
            for facts in arms for d, cls in facts
            if not cls & (ValueClass.ZERO | ValueClass.FINITE)
        ))

    def _one_of(self, dst: Terms, srcs: list[Terms], guard: tuple[int, ...]) -> None:
        """*dst* is one of *srcs* wherever *guard* holds."""
        msbs, lsbs, values = (
            _all(getattr(t, f) for t in srcs) for f in ('msb', 'lsb', 'value'))
        if dst.msb is not None and msbs is not None:
            self.store.le_max(dst.msb, msbs, guard=guard)
        if dst.lsb is not None and lsbs is not None:
            self._grid_ge(dst.lsb, *lsbs, guard=guard)
        if dst.value is not None and values is not None:
            self.store.le_max(dst.value, values, guard=guard)
            self.store.ge_min(dst.value, values, guard=guard)

    def _visit_call(self, e: Call, ctx: Any) -> None:
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
        elements = tuple(self._all_elements(a) for a in e.args)
        walk = _DigitBoundInferInstance(
            e.fn.ast, view, DigitBoundParams(self.store, args, outer=self._binders, elements=elements),
            outer=lambda: self._assumed(e),
        )
        sub = walk.analyze()
        self._callees.append(walk)
        self.out.by_call[e] = sub
        if sub.ret.msb is not None or sub.ret.value is not None:
            # The result's terms are the callee's, so a bound this site knows
            # and the callee does not -- a branch refined away here -- is
            # stated on them.  Only on a term the callee *minted*, though: a
            # callee that returns an argument verbatim hands back the
            # caller's own variable, shared by every use of that name, and
            # this site's bound may hold only on the path reaching the call.
            passed = {a.msb for a in args if a.msb is not None}
            if sub.ret.msb is not None and sub.ret.msb not in passed:
                self._seed(sub.ret.msb, e, None, self.view.logb_range(e)[1])
            self.out.by_expr[e] = Terms(sub.ret.msb, sub.ret.lsb, sub.ret.value)
            for k, key in enumerate(sub.ret_elements):
                # the result is the returned list, element for element
                self.store.spaces.unify(self._elements(e, k), key)


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
        classes: ValueClassAnalysis | None = None,
    ) -> DigitBoundAnalysis:
        """Infer digit-bound relations for *func*, seeded from *view*.

        *params* continues a caller's constraint system instead of starting
        a fresh one; see :class:`DigitBoundParams`.  *classes* is *func*'s
        value classes, where the caller has them with escape summaries --
        without, a list handed to a call has no element facts.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f'Expected \'FuncDef\', got {type(func)} for {func}')
        if params is None:
            params = DigitBoundParams(DigitBoundStore(), ())
        return _DigitBoundInferInstance(func, view, params, classes).analyze()
