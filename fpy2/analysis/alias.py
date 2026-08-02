"""
Alias analysis: what else may refer to a list.

FPy lists are objects with identity — assignment aliases, ``xs[i] = e`` mutates in
place, and passing, returning or projecting carries the identity along.  This
analysis answers, for each list a function creates or receives, *what else may
refer to it*.

Formulation
-----------

A *cell* is a place a list reference can sit: one per list-carrying
:class:`~fpy2.analysis.Definition`, one per list-carrying expression, and — created
lazily — one per *part* of a cell.  A part is either the elements of a list (index
not tracked, since it is rarely a constant) or field *i* of a tuple (arity is
static, so fields are kept apart).  Parts are what make nesting work: the rows of
a ``list[list[Real]]`` are cells in their own right.

Aliasing routes generate *equality* constraints between cells, solved with
union-find (``elts(c)`` is the elements part, ``fld(c, i)`` field *i*):

=========================  ==========================================
``ys = xs``                ``cell(ys) ≡ cell(xs)``
``row = xss[i]``           ``cell(row) ≡ elts(xss)``
``xss[i] = e``             ``elts(xss) ≡ cell(e)``
``[a, b]``                 ``elts(result) ≡ cell(a) ≡ cell(b)``
``(a, b)``                 ``fld(result, 0) ≡ cell(a)``, likewise 1
``a, b = t``               ``cell(a) ≡ fld(t, 0)``, likewise 1
``fst(t)`` / ``t[0]``      ``cell(result) ≡ fld(t, 0)``
``xs[i:j]``                ``elts(result) ≡ elts(xs)`` — a slice is *shallow*
``for row in xss``         ``cell(row) ≡ elts(xss)``
``xs if c else ys``        ``cell(result) ≡ cell(xs) ≡ cell(ys)``
``enumerate``/``zip``      ``elts(result) ≡ elts(arg)``
=========================  ==========================================

Equality rather than inclusion (unification rather than a subset-based solver) is
deliberate: it over-approximates aliasing, which is the safe direction for every
consumer, and it is adequate here — measured over the test corpus, only three
merges anywhere lose precision relative to a subset-based solver.

An expression kind not in the table is handled conservatively: it gets its own
allocation site, and every list-carrying variable inside it is marked shared
outward, so an unmodelled route cannot make a list *look* uniquely owned.

Two routes leave the function, and they mean opposite things for ownership.
*Shared outward* — handed to a call, or to an unmodelled operation — means
something else may hold the list while this function still does.  *Returned* is a
transfer; see :meth:`AliasAnalysis.transfers_ownership`.  Collapsing the two would
forgo every returned value: 37 of the corpus's 299 allocation sites are fresh
values that are returned and nothing else.

Limitations
-----------

Flow-insensitive: ``ys = xs`` marks ``xs`` shared even if ``ys`` is dead
immediately afterwards, and a container part counts as a referrer even if the
container itself is never read.  Intraprocedural: a list handed to a call is shared
outward, without asking whether the callee retains it.  A list index is not
tracked, so ``xs[0]`` and ``xs[1]`` are the same part.  All three cost precision
rather than soundness.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from ..ast import (
    Argument,
    Assign,
    Call,
    DefaultVisitor,
    Empty,
    Enumerate,
    Expr,
    ForStmt,
    Fst,
    FuncDef,
    IfExpr,
    IndexedAssign,
    Integer,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NamedId,
    Range1,
    Range2,
    Range3,
    ReturnStmt,
    Snd,
    TupleBinding,
    TupleExpr,
    Var,
    Zip,
)
from ..function import Function
from ..types import ListType, TupleType, Type
from ..utils import Unionfind
from .define_use import DefineUse, DefineUseAnalysis
from .reaching_defs import AssignDef, Definition, PhiDef
from .type_infer import TypeAnalysis, TypeInfer

if TYPE_CHECKING:
    # `escape` builds on this module; the dependency only goes the other way
    # at type-check time
    from .escape import EscapeSummary

ELTS = None
"""Part key for the elements of a list.

A single key for the whole list: an index is rarely a constant, so one part per
index would buy almost nothing.
"""

_PartKey = int | None

_Missing = Literal['raise', 'create', 'none']
"""What :meth:`_Cells.part` should do when the part does not exist yet."""


def _carries_list(ty: Type | None) -> bool:
    """Whether a value of type *ty* can hold a list reference anywhere inside it.

    Values that cannot are not given cells at all — nothing about them is
    observable through aliasing.
    """
    match ty:
        case ListType():
            return True
        case TupleType():
            return any(_carries_list(elt) for elt in ty.elts)
        case _:
            return False


class AllocSite:
    """A place a list value comes into existence.

    ``kind`` is one of ``literal``, ``comprehension``, ``slice``, ``builtin``,
    ``call``, or ``param``.  A ``param`` site stands for a list the *caller*
    allocated, and a parameter gets one site per list nested in its type, since
    each is a distinct object the caller also owns; ``depth`` is how many list
    levels down it sits.

    Identity-hashed: two sites are the same only if they are the same object.
    """

    __slots__ = ('depth', 'kind', 'node')

    def __init__(self, kind: str, node: Expr | Argument, depth: int = 0):
        self.kind = kind
        self.node = node
        self.depth = depth

    def __repr__(self) -> str:
        # Not `@default_repr`: it would inline the whole AST node, and a site is
        # usually printed as part of a list.
        at = getattr(self.node, 'name', None) or type(self.node).__name__
        return f'<{self.kind} {at}{f"[{self.depth}]" if self.depth else ""}>'


class _Cell:
    """A place a list reference can sit: a name, an expression, or a part.

    An opaque token — the equivalence relation and per-class payload live in
    :class:`_Cells`.
    """

    __slots__ = ('_kind',)

    def __init__(self, kind: str):
        self._kind = kind

    def __repr__(self) -> str:
        return f'<cell {self._kind}>'


Region = _Cell
"""What may be the same list.

An opaque, hashable identity for one equivalence class.  A consumer that has to
give a single answer to everything that may alias — a representation, say — keys
on this rather than on individual definitions, so that two names the analysis
unified cannot be answered differently.
"""


class _Cells:
    """Cells, the equivalence relation over them, and per-class payload.

    Payload lives in dicts keyed by the class representative, the same
    arrangement ``storage_infer`` uses for storage classes.
    """

    def __init__(self):
        self._uf: Unionfind[_Cell] = Unionfind()
        self._sites: dict[_Cell, set[AllocSite]] = {}
        # A class's *referrers*: distinct source names plus container slots,
        # not cells.  A variable's SSA definitions are one place however many
        # times it is redefined; a slot is not a name but is how `[xs]` shares
        # `xs`; an allocation is neither, so `xs = [x, x]` has one referrer.
        self._names: dict[_Cell, set[NamedId]] = {}
        self._slots: dict[_Cell, int] = {}
        # Two ways out of the function, distinguished because they mean opposite
        # things for ownership -- see `AliasAnalysis.transfers_ownership`.
        self._shared_out: set[_Cell] = set()
        self._returned: set[_Cell] = set()
        self._parts: dict[_Cell, dict[_PartKey, _Cell]] = {}

    @property
    def _flags(self) -> tuple[set[_Cell], ...]:
        """Every downward-propagating class flag, so each is handled alike."""
        return (self._shared_out, self._returned)

    def new(
        self, kind: str, *, name: NamedId | None = None, slot: bool = False,
    ) -> _Cell:
        """A fresh cell.  *name* is the source variable it stands for, if any;
        *slot* marks a cell that is a place inside a container."""
        cell = _Cell(kind)
        self._uf.add(cell)
        self._sites[cell] = set()
        self._names[cell] = {name} if name is not None else set()
        self._slots[cell] = 1 if slot else 0
        return cell

    def find(self, c: _Cell) -> _Cell:
        return self._uf.find(c)

    @overload
    def part(
        self, c: _Cell, key: _PartKey = ...,
        *, missing: Literal['raise', 'create'] = ...,
    ) -> _Cell:
        ...

    @overload
    def part(
        self, c: _Cell, key: _PartKey = ..., *, missing: Literal['none'],
    ) -> _Cell | None:
        ...

    def part(
        self, c: _Cell, key: _PartKey = ELTS,
        *, missing: _Missing = 'raise',
    ) -> _Cell | None:
        """The cell standing for part *key* of *c*: its elements, or a field.

        *missing* says what to do when that part does not exist yet, and has no
        default that silently does either thing:

        - ``'raise'`` — a caller that expects a part has a bug if there is none,
          and quietly making one would hide it.
        - ``'create'`` — make it; what building constraints wants.
        - ``'none'`` — return ``None``; what a *query* wants.  A query must not
          extend the structure it reads, and "nothing ever looked inside this
          list" is itself an answer.
        """
        root = self.find(c)
        parts = self._parts.get(root)
        if parts is None or key not in parts:
            if missing == 'none':
                return None
            if missing == 'raise':
                raise KeyError(f'{c!r} has no part {key!r}')
            parts = self._parts.setdefault(root, {})
            parts[key] = self.new('part', slot=True)
            for flag in self._flags:
                # created after the container left -- see `_spread`
                if root in flag:
                    self._spread(flag, parts[key])
        return self.find(parts[key])

    def merge(self, a: _Cell, b: _Cell) -> _Cell:
        """Make *a* and *b* the same class, and what lives inside them likewise.

        Merging two classes forces their matching parts to merge, which cascades.
        A worklist rather than recursion, so the bound is structural: every
        iteration performs a real union, and there are finitely many cells, so the
        loop runs at most once per cell.  Nothing here depends on types being
        non-recursive -- that only bounds how *deep* the parts ever go.
        """
        pending = [(a, b)]
        while pending:
            x, y = pending.pop()
            ra, rb = self.find(x), self.find(y)
            if ra is rb:
                continue
            root = self._uf.union(ra, rb)
            other = rb if root is ra else ra
            self._sites[root] = (
                self._sites.pop(root, set()) | self._sites.pop(other, set())
            )
            self._names[root] = (
                self._names.pop(root, set()) | self._names.pop(other, set())
            )
            self._slots[root] = (
                self._slots.pop(root, 0) + self._slots.pop(other, 0)
            )
            carry = [f for f in self._flags if root in f or other in f]
            for flag in self._flags:
                flag.discard(other)
            mine = self._parts.pop(root, {})
            theirs = self._parts.pop(other, {})
            self._parts[root] = mine
            for key, cell in theirs.items():
                if key in mine:
                    pending.append((mine[key], cell))
                else:
                    mine[key] = cell
            for flag in carry:
                # a class merged into one that left the function left too, parts
                # and all
                self._spread(flag, root)
        return self.find(a)

    def add_site(self, c: _Cell, site: AllocSite) -> None:
        self._sites.setdefault(self.find(c), set()).add(site)

    def sites_at(self, c: _Cell) -> frozenset[AllocSite]:
        return frozenset(self._sites.get(self.find(c), ()))

    def referrers(self, c: _Cell) -> int:
        root = self.find(c)
        return len(self._names.get(root, ())) + self._slots.get(root, 0)

    def _spread(self, flag: set[_Cell], c: _Cell) -> None:
        """Add *c* to *flag*, and everything reachable inside it.

        Leaving the function is transitive downward: handing out a
        ``list[list[Real]]`` hands out its rows too.  Parts created *afterwards*
        inherit the flag in :meth:`part`, so the two directions together are
        order-independent.
        """
        pending, seen = [self.find(c)], set()
        while pending:
            root = self.find(pending.pop())
            if root in seen:
                continue
            seen.add(root)
            flag.add(root)
            pending.extend(self._parts.get(root, {}).values())

    def mark_shared_out(self, c: _Cell) -> None:
        """*c* may be held by something outside while this function holds it."""
        self._spread(self._shared_out, c)

    def mark_returned(self, c: _Cell) -> None:
        """*c* is handed to the caller by a ``return``."""
        self._spread(self._returned, c)

    def is_shared_out(self, c: _Cell) -> bool:
        return self.find(c) in self._shared_out

    def is_returned(self, c: _Cell) -> bool:
        return self.find(c) in self._returned

    def has_param_site(self, c: _Cell) -> bool:
        """Whether the caller already holds something in *c*'s class."""
        return any(s.kind == 'param' for s in self.sites_at(c))


@dataclass
class AliasAnalysis:
    """Result of :class:`Alias`.

    ``sites`` lists every list allocation the function performs, plus one per
    list nested in each parameter's type.
    """

    sites: list[AllocSite]
    _cells: _Cells
    _cell_of: dict[Definition, _Cell]
    _cell_of_expr: dict[Expr, _Cell]
    _site_cell: dict[AllocSite, _Cell]

    def region_of(self, d: Definition, depth: int = 0) -> Region | None:
        """What may be the same list as *d*, *depth* levels in.

        ``depth=0`` is *d* itself; ``depth=1`` is the lists held in its
        elements, and so on.  ``None`` when nothing is known about that level.
        """
        return self._walk(self._cell_of.get(d), depth)

    def region_of_expr(self, e: Expr, depth: int = 0) -> Region | None:
        """As :meth:`region_of`, for an expression.

        An expression that names a definition shares its cell, and so its
        region.
        """
        return self._walk(self._cell_of_expr.get(e), depth)

    def _walk(self, cell: _Cell | None, depth: int) -> _Cell | None:
        for _ in range(depth):
            if cell is None:
                return None
            cell = self._cells.part(cell, missing='none')
        return None if cell is None else self._cells.find(cell)

    def regions_in_a_tuple(self) -> frozenset[Region]:
        """Every region reachable through a tuple field, and everything inside
        those in turn."""
        pending = [
            inner for parts in self._cells._parts.values()
            for key, inner in parts.items() if key is not ELTS
        ]
        seen: set[Region] = set()
        while pending:
            root = self._cells.find(pending.pop())
            if root in seen:
                continue
            seen.add(root)
            pending.extend(self._cells._parts.get(root, {}).values())
        return frozenset(seen)

    def all_defs(self) -> frozenset[Definition]:
        """Every definition the analysis gave a cell to."""
        return frozenset(self._cell_of)

    def region_at(self, region: Region, depth: int = 1) -> Region | None:
        """The region *depth* levels inside *region*."""
        return self._walk(region, depth)

    def defs_in(self, region: Region) -> frozenset[Definition]:
        """Every definition whose value may live in *region*."""
        out = {
            d for d, c in self._cell_of.items()
            if self._cells.find(c) is region
        }
        return frozenset(out)

    def referrers(self, region: Region) -> int:
        """How many places may hold a reference into *region*: distinct source
        names plus container slots."""
        return self._cells.referrers(region)

    def region_of_site(self, site: AllocSite) -> Region | None:
        """Which region *site*'s allocation lives in."""
        cell = self._site_cell.get(site)
        return None if cell is None else self._cells.find(cell)

    def sites_at(self, region: Region | None) -> frozenset[AllocSite]:
        """The allocations that may live in *region*."""
        return frozenset() if region is None else self._cells.sites_at(region)

    def sites_of(self, d: Definition, depth: int = 0) -> frozenset[AllocSite]:
        """The allocations *d* may refer to, *depth* list levels in.

        An empty result means *no information*, not *nothing there*: the level
        may simply never have been looked inside.  Read it conservatively.
        """
        return self.sites_at(self.region_of(d, depth))

    def is_shared(self, site: AllocSite) -> bool:
        """Whether more than one place may refer to *site*."""
        return self._cells.referrers(self._site_cell[site]) > 1

    def escapes(self, site: AllocSite) -> bool:
        """Whether *site*, or a container holding it, leaves the function at all
        — handed to a call, or returned.

        Says nothing about *which*; :meth:`transfers_ownership` separates them.
        """
        cell = self._site_cell[site]
        return self._cells.is_shared_out(cell) or self._cells.is_returned(cell)

    def is_returned(self, site: AllocSite) -> bool:
        """Whether *site*, or a container holding it, is handed to the caller by a
        ``return``."""
        return self._cells.is_returned(self._site_cell[site])

    def transfers_ownership(self, site: AllocSite) -> bool:
        """Whether returning *site* hands out *sole* ownership of it.

        A ``return`` is a transfer, not sharing: the value moves to the caller
        and this function keeps nothing, so a copy at the boundary is
        unobservable.  Unless the caller already holds it — ``return xs`` on a
        parameter leaves two handles to one list — so a ``param`` site anywhere
        in the class blocks the transfer.  A class's site set is the union over
        everything merged into it, so this catches the indirect routes too.
        """
        cell = self._site_cell[site]
        return (
            self._cells.is_returned(cell)
            and not self._cells.has_param_site(cell)
            and not self._cells.is_shared_out(cell)
            and not self.is_shared(site)
        )

    def is_uniquely_owned(self, site: AllocSite) -> bool:
        """Whether nothing else may observe *site* — the condition under which
        copying, or representing it by value, is unobservable.

        One referrer, never shared outward, and if it is returned then the return
        transfers ownership.
        """
        if self.is_shared(site) or self._cells.is_shared_out(self._site_cell[site]):
            return False
        return not self.is_returned(site) or self.transfers_ownership(site)

    def may_alias(self, a: Definition, b: Definition) -> bool:
        """Whether *a* and *b* may refer to the same list."""
        ca, cb = self._cell_of.get(a), self._cell_of.get(b)
        if ca is None or cb is None:
            return False
        return self._cells.find(ca) is self._cells.find(cb)


class _EscapeVars(DefaultVisitor):
    """Marks every list-carrying variable inside an expression as shared outward.

    Used for expression kinds :meth:`_Builder._build_cell` does not model: if an
    unmodelled operation might retain one of its operands, the sound answer is
    that the operand is no longer uniquely owned.  A visitor rather than an
    attribute walk so an unfamiliar node's children are still reached.
    """

    def __init__(self, builder: '_Builder'):
        self.builder = builder

    def _visit_var(self, e: Var, ctx):
        cell = self.builder._cell_for(e)
        if cell is not None:
            self.builder.cells.mark_shared_out(cell)


class _Builder(DefaultVisitor):
    """Generates the constraints for one function.

    A :class:`DefaultVisitor` so traversal is the framework's job: a node this
    analysis should care about cannot be silently skipped by a hand-written
    walk.  Only :meth:`_cell_for` is explicit recursion, because it returns a
    value; it is memoized so reaching an expression twice does not allocate
    twice.
    """

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis,
                 type_info: TypeAnalysis,
                 summaries: 'dict[FuncDef, EscapeSummary]'):
        self.func = func
        self.def_use = def_use
        self.types = type_info
        self.summaries = summaries
        self.cells = _Cells()
        self.sites: list[AllocSite] = []
        self.cell_of: dict[Definition, _Cell] = {}
        self.site_cell: dict[AllocSite, _Cell] = {}
        self._by_expr: dict[Expr, _Cell] = {}

    def run(self) -> AliasAnalysis:
        self._merge_redefinitions()
        self._seed_params()
        self._visit_function(self.func, None)
        return AliasAnalysis(
            self.sites, self.cells, self.cell_of, self._by_expr,
            self.site_cell,
        )

    # -- cells --------------------------------------------------------------

    def _part(self, c: _Cell, key: _PartKey = ELTS) -> _Cell:
        """A part of *c*, created on first mention."""
        return self.cells.part(c, key, missing='create')

    def _cell(self, d: Definition) -> _Cell:
        if d not in self.cell_of:
            self.cell_of[d] = self.cells.new('name', name=d.name)
        return self.cells.find(self.cell_of[d])

    def _site(self, kind: str, node: Expr | Argument, cell: _Cell,
              depth: int = 0) -> None:
        site = AllocSite(kind, node, depth)
        self.sites.append(site)
        self.cells.add_site(cell, site)
        self.site_cell[site] = self.cells.find(cell)

    def _alloc(self, kind: str, node: Expr) -> _Cell:
        # transient: an allocation is not itself a place a reference is held
        cell = self.cells.new(kind)
        self._site(kind, node, cell)
        return cell

    def _merge_redefinitions(self) -> None:
        """Unify the definitions of a variable that are the *same* list.

        Two of SSA's fresh definitions allocate nothing: ``xs[i] = e`` mutates
        the list that was already there, and a branch merge names whichever of
        its operands arrived.  A plain rebinding is *not* included: ``ys = zs``
        has a ``prev`` too, and it is a different list.

        Sound to unify because referrers are counted by source name, so a
        variable's definitions do not read as several places.
        """
        for d in self.def_use.defs:
            if not _carries_list(self.types.by_def.get(d)):
                continue
            match d:
                case AssignDef(site=IndexedAssign()) if d.prev is not None:
                    prevs = [d.prev]
                case PhiDef():
                    prevs = [d.lhs, d.rhs]
                case _:
                    continue
            for i in prevs:
                self.cells.merge(self._cell(d), self._cell(self.def_use.defs[i]))

    def _seed_params(self) -> None:
        for arg in self.func.args:
            if not isinstance(arg.name, NamedId):
                continue
            d = self.def_use.find_def_from_site(arg.name, arg)
            ty = self.types.by_def.get(d)
            if _carries_list(ty):
                self._seed(self._cell(d), ty, arg, 0)

    def _seed(self, cell: _Cell, ty: Type | None, arg: Argument,
              depth: int) -> None:
        """One site per list inside a parameter's type: the caller owns the outer
        list *and* every list nested in it."""
        match ty:
            case ListType():
                self._site('param', arg, cell, depth)
                if _carries_list(ty.elt):
                    self._seed(self._part(cell), ty.elt, arg, depth + 1)
            case TupleType():
                for i, elt in enumerate(ty.elts):
                    if _carries_list(elt):
                        self._seed(self._part(cell, i), elt, arg, depth)

    # -- the value-returning half ------------------------------------------

    def _cell_for(self, e: Expr) -> _Cell | None:
        """The cell *e* denotes, or ``None`` if *e* carries no list."""
        if not _carries_list(self.types.by_expr.get(e)):
            return None
        if e not in self._by_expr:
            cell = self._build_cell(e)
            if cell is None:
                return None
            self._by_expr[e] = cell
        return self.cells.find(self._by_expr[e])

    def _build_cell(self, e: Expr) -> _Cell | None:
        match e:
            case Var():
                return self._cell(self.def_use.find_def_from_use(e))
            case ListRef():
                return self._project(e)
            case Fst() | Snd():
                base = self._cell_for(e.args[0])
                field = 0 if isinstance(e, Fst) else 1
                return None if base is None else self._part(base, field)
            case IfExpr():
                # the result *is* one branch or the other, so it aliases both
                cell = self.cells.new('branch')
                for branch in (e.ift, e.iff):
                    bc = self._cell_for(branch)
                    if bc is not None:
                        cell = self.cells.merge(cell, bc)
                return cell
            case ListSlice():
                # a fresh outer list over the *same* elements
                cell = self._alloc('slice', e)
                base = self._cell_for(e.value)
                if base is not None:
                    self.cells.merge(
                        self._part(cell), self._part(base),
                    )
                return cell
            case ListExpr():
                cell = self._alloc('literal', e)
                for x in e.elts:
                    xc = self._cell_for(x)
                    if xc is not None:
                        self.cells.merge(self._part(cell), xc)
                return cell
            case TupleExpr():
                cell = self._alloc('literal', e)
                for i, x in enumerate(e.elts):
                    xc = self._cell_for(x)
                    if xc is not None:
                        self.cells.merge(self._part(cell, i), xc)
                return cell
            case ListComp():
                cell = self._alloc('comprehension', e)
                self._bind_comp_targets(e)
                xc = self._cell_for(e.elt)
                if xc is not None:
                    self.cells.merge(self._part(cell), xc)
                return cell
            case Enumerate() | Zip():
                # Each element is a *tuple* over the sources' elements, so the
                # constraint is on a field of the element and not on the element
                # itself: `enumerate` pairs an index with the source element,
                # `zip` takes field i from argument i.  Merging the element parts
                # directly instead would conflate the tuple with what it holds,
                # putting a loop variable one level too deep -- conservative for
                # the site verdicts, but wrong for `may_alias`.
                cell = self._alloc('builtin', e)
                elts = self._part(cell)
                for i, a in enumerate(e.args):
                    ac = self._cell_for(a)
                    if ac is not None:
                        field = 1 if isinstance(e, Enumerate) else i
                        self.cells.merge(
                            self._part(elts, field), self._part(ac),
                        )
                return cell
            case Call():
                cell = self._alloc('call', e)
                self._escape_args(e)
                return cell
            case Range1() | Range2() | Range3() | Empty():
                # A fresh sequence over *integer* arguments: there is nothing
                # list-shaped for it to retain, so unlike the catch-all below it
                # must not escape what its operands mention.  `range(len(xs))`
                # is a common enough shape that treating it conservatively
                # would box most parameters for no reason.
                return self._alloc('builtin', e)
            case _:
                # An unmodelled expression: assume it allocates, and that it may
                # retain anything named inside it.
                cell = self._alloc('builtin', e)
                _EscapeVars(self)._visit_expr(e, None)
                return cell

    def _project(self, e: ListRef) -> _Cell | None:
        """``xs[i]`` — the elements part of a list, or a field of a tuple."""
        base = self._cell_for(e.value)
        if base is None:
            return None
        base_ty = self.types.by_expr.get(e.value)
        if not isinstance(base_ty, TupleType):
            return self._part(base)
        if isinstance(e.index, Integer):
            return self._part(base, e.index.val)
        # a tuple index that is not a literal: could be any field
        cell = self._part(base, 0)
        for i in range(1, len(base_ty.elts)):
            cell = self.cells.merge(cell, self._part(base, i))
        return cell

    def _bind_comp_targets(self, e: ListComp) -> None:
        """Bind each comprehension variable to the elements it iterates."""
        for target, iterable in zip(e.targets, e.iterables):
            it = self._cell_for(iterable)
            if it is not None:
                self._bind(target, self._part(it), e)

    def _bind(self, target, cell: _Cell, site) -> None:
        """Bind a name or a tuple pattern to *cell*, field by field."""
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                if _carries_list(self.types.by_def.get(d)):
                    self.cells.merge(self._cell(d), cell)
            case TupleBinding():
                for i, elt in enumerate(target.elts):
                    self._bind(elt, self._part(cell, i), site)

    def _escape_args(self, e: Call) -> None:
        """A list handed to a call may be kept by the callee.

        Unless its summary says otherwise: a callee that only reads or writes
        its argument holds nothing once it returns, so the caller's list is no
        more shared than before the call.  No summary — a foreign function, or
        one not yet analyzed — means the conservative answer.
        """
        summary = None
        if isinstance(e.fn, Function):
            summary = self.summaries.get(e.fn.ast)
        for i, a in enumerate(e.args):
            ac = self._cell_for(a)
            if ac is None:
                continue
            if summary is None or summary.retains(i):
                self.cells.mark_shared_out(ac)

    # -- constraint-generating hooks ---------------------------------------
    #
    # Each does its own work then delegates, so the framework keeps traversing.

    def _visit_expr(self, e: Expr, ctx):
        # The net that makes the analysis total: every list-carrying expression
        # gets a cell, even in a position no hook below intercepts.  Memoized, so
        # an expression a hook already handled is not built twice.
        self._cell_for(e)
        return super()._visit_expr(e, ctx)

    def _visit_call(self, e: Call, ctx):
        """A list handed to a call escapes: whether the callee retains it is not
        decidable here."""
        self._escape_args(e)
        super()._visit_call(e, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        rhs = self._cell_for(stmt.expr)
        if rhs is not None:
            self._bind(stmt.target, rhs, stmt)
        super()._visit_assign(stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        if isinstance(stmt.var, NamedId):
            d = self.def_use.find_def_from_site(stmt.var, stmt)
            cur = self._cell(d)
            rhs = self._cell_for(stmt.expr)
            if rhs is not None:
                for _ in stmt.indices[:-1]:
                    cur = self._part(cur)
                self.cells.merge(self._part(cur), rhs)
        super()._visit_indexed_assign(stmt, ctx)

    def _visit_for(self, stmt: ForStmt, ctx):
        it = self._cell_for(stmt.iterable)
        if it is not None:
            self._bind(stmt.target, self._part(it), stmt)
        super()._visit_for(stmt, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx):
        rhs = self._cell_for(stmt.expr)
        if rhs is not None:
            self.cells.mark_returned(rhs)
        super()._visit_return(stmt, ctx)


class Alias:
    """Alias analysis for FPy programs."""

    @staticmethod
    def analyze(
        func: FuncDef,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
        summaries: 'dict[FuncDef, EscapeSummary] | None' = None,
    ) -> AliasAnalysis:
        """Compute what may refer to each list in *func*.

        Args:
            func: the function to analyze.
            def_use: reuse an existing def-use analysis.
            type_info: reuse an existing type analysis (needed to tell which
                expressions carry lists).
            summaries: escape summaries for the callees, so an argument a callee
                does not retain is not treated as shared.  A callee absent from
                it is assumed to retain everything.
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {func}")
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        return _Builder(func, def_use, type_info, summaries or {}).run()
