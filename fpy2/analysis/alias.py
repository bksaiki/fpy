"""
Alias analysis: what else may refer to a list.

An FPy list is a core list of *references* — one cell per element, and mutation
goes through a cell.  This analysis is an *abstract store* over those cells: it
answers, for each list a function creates or receives, *what else may refer to
it*.

Formulation
-----------

A *region* is an abstract location: the set of run-time locations a place may
hold.  There is one per list-valued
:class:`~fpy2.analysis.Definition`, one per list-valued expression, and —
created lazily — one per *part* of a region: either the elements of a list (one
for all of them, since the semantics gives a cell per element and collapsing them
only over-approximates) or field *i* of a tuple, which is not a cell at all — a
tuple holds values, so a field matters only because the value in it may be a
list.  Parts make nesting work: the rows of a ``list[list[Real]]`` are regions in
their own right.

Three maps make up the abstract store: ``_by_def`` / ``_by_expr`` are the
environment, which region a name or expression denotes; ``_parts`` maps a region
to the region its contents live in; ``_sites`` records which allocation may have
created it.  It abstracts the store twice over: a region is a *set* of locations,
and the map runs region-to-region rather than location-to-value, so only
reachability is tracked, not values.  :meth:`_Regions.merge` still cascades into
matching parts, because identifying two locations identifies their contents.

Each route below is a core rule read backwards: binding is **E-Assign**, which
copies nothing; projection is **E-Index** then **E-Deref**, naming the element
cell's contents; an element store is **E-Update** *through* that cell;
construction is **E-List** over an **E-Ref** per element, so the cell is fresh
but holds the operand's value; tuples are **E-Tuple** / **M-Tuple**.  Routes
generate *equality* constraints, solved with union-find (``elts(c)`` is the
elements part, ``fld(c, i)`` field *i*):

=========================  ==========================================
``ys = xs``                ``reg(ys) ≡ reg(xs)``
``row = xss[i]``           ``reg(row) ≡ elts(xss)``
``xss[i] = e``             ``elts(xss) ≡ reg(e)``
``[a, b]``                 ``elts(result) ≡ reg(a) ≡ reg(b)``
``(a, b)``                 ``fld(result, 0) ≡ reg(a)``, likewise 1
``a, b = t``               ``reg(a) ≡ fld(t, 0)``, likewise 1
``fst(t)`` / ``t[0]``      ``reg(result) ≡ fld(t, 0)``
``xs[i:j]``                ``elts(result) ≡ elts(xs)`` — fresh spine, same rows
``for row in xss``         ``reg(row) ≡ elts(xss)``
``xs if c else ys``        ``reg(result) ≡ reg(xs) ≡ reg(ys)``
``enumerate``/``zip``      ``elts(result) ≡ elts(arg)``
=========================  ==========================================

Equality rather than inclusion is deliberate.  Each place denotes exactly one
region and distinct regions are disjoint — Steensgaard-style, where a
subset-based solver would give each place an overlapping points-to *set* — so
different regions never alias and the same region may.  That over-approximates,
the safe direction for every consumer, and it is adequate here: measured over the
test corpus, only three merges anywhere lose precision against a subset-based
solver.

An expression kind not in the table is handled conservatively: it gets its own
allocation site, and every list-carrying variable inside it is marked shared
outward, so an unmodelled route cannot make a list *look* uniquely owned.  Note
what that does *not* buy: the fresh region is never merged with the operands, so
if such an operation does return its argument, two regions hold one location.
The ownership queries stay sound because shared-outward poisons them;
:meth:`AliasAnalysis.may_alias` is sound only for the routes above.

Two routes leave the function, and they mean opposite things for ownership.
*Shared outward* — handed to a call, or to an unmodelled operation — means
something else may hold the list while this function still does: **E-App** keeps
no store of its own, so a callee writes the caller's cells.  *Returned* is
**E-Ret** handing a location out, which is a transfer; see
:meth:`AliasAnalysis.transfers_ownership`.  Collapsing the two would forgo every
returned value: 37 of the corpus's 299 allocation sites are fresh values that are
returned and nothing else.

Limitations
-----------

One *summary* store for the whole function, not one per program point, so it is
flow-insensitive: ``ys = xs`` marks ``xs`` shared even if ``ys`` is dead
immediately afterwards, and a container part counts as a referrer even if the
container itself is never read.  Intraprocedural: a list handed to a call is
shared outward, without asking whether the callee retains it.  Index-insensitive:
``xs[0]`` and ``xs[1]`` are one part.  All three cost precision, not soundness.
"""

from dataclasses import dataclass, field
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
from .reaching_defs import AssignDef, Definition, same_object_defs
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
"""What :meth:`_Regions.part` should do when the part does not exist yet."""


def _carries_list(ty: Type | None) -> bool:
    """Whether a value of type *ty* can hold a list reference anywhere inside it.

    Values that cannot are not given regions at all — nothing about them is
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


class Region:
    """What may be the same list: an abstract location.

    Created one per place a list reference can sit — a name, an expression, or a
    part.  An opaque, hashable token; the equivalence relation and per-class
    payload live in :class:`_Regions`.  A consumer that has to give a single
    answer to everything that may alias — a representation, say — keys on this
    rather than on individual definitions, so that two names the analysis unified
    cannot be answered differently.
    """

    __slots__ = ('_kind',)

    def __init__(self, kind: str):
        self._kind = kind

    def __repr__(self) -> str:
        return f'<region {self._kind}>'


class _Regions:
    """Cells, the equivalence relation over them, and per-class payload.

    Payload lives in dicts keyed by the class representative, the same
    arrangement ``storage_infer`` uses for storage classes.
    """

    def __init__(self):
        self._uf: Unionfind[Region] = Unionfind()
        self._sites: dict[Region, set[AllocSite]] = {}
        # A class's *referrers*: distinct source names plus container slots,
        # not regions.  A variable's SSA definitions are one place however many
        # times it is redefined; a slot is not a name but is how `[xs]` shares
        # `xs`; an allocation is neither, so `xs = [x, x]` has one referrer.
        self._names: dict[Region, set[NamedId]] = {}
        self._slots: dict[Region, int] = {}
        # Two ways out of the function, distinguished because they mean opposite
        # things for ownership -- see `AliasAnalysis.transfers_ownership`.
        self._shared_out: set[Region] = set()
        self._returned: set[Region] = set()
        self._parts: dict[Region, dict[_PartKey, Region]] = {}

    @property
    def _flags(self) -> tuple[set[Region], ...]:
        """Every downward-propagating class flag, so each is handled alike."""
        return (self._shared_out, self._returned)

    def new(
        self, kind: str, *, name: NamedId | None = None, slot: bool = False,
    ) -> Region:
        """A fresh region.  *name* is the source variable it stands for, if any;
        *slot* marks a region that is a place inside a container."""
        region = Region(kind)
        self._uf.add(region)
        self._sites[region] = set()
        self._names[region] = {name} if name is not None else set()
        self._slots[region] = 1 if slot else 0
        return region

    def find(self, c: Region) -> Region:
        return self._uf.find(c)

    @overload
    def part(
        self, c: Region, key: _PartKey = ...,
        *, missing: Literal['raise', 'create'] = ...,
    ) -> Region:
        ...

    @overload
    def part(
        self, c: Region, key: _PartKey = ..., *, missing: Literal['none'],
    ) -> Region | None:
        ...

    def part(
        self, c: Region, key: _PartKey = ELTS,
        *, missing: _Missing = 'raise',
    ) -> Region | None:
        """The region standing for part *key* of *c*: its elements, or a field.

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

    def merge(self, a: Region, b: Region) -> Region:
        """Make *a* and *b* the same class, and what lives inside them likewise.

        Merging two classes forces their matching parts to merge, which cascades.
        A worklist rather than recursion, so the bound is structural: every
        iteration performs a real union, and there are finitely many regions, so the
        loop runs at most once per region.  Nothing here depends on types being
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
            for key, region in theirs.items():
                if key in mine:
                    pending.append((mine[key], region))
                else:
                    mine[key] = region
            for flag in carry:
                # a class merged into one that left the function left too, parts
                # and all
                self._spread(flag, root)
        return self.find(a)

    def add_site(self, c: Region, site: AllocSite) -> None:
        self._sites.setdefault(self.find(c), set()).add(site)

    def sites_at(self, c: Region) -> frozenset[AllocSite]:
        return frozenset(self._sites.get(self.find(c), ()))

    def referrers(self, c: Region) -> int:
        root = self.find(c)
        return len(self._names.get(root, ())) + self._slots.get(root, 0)

    def _spread(self, flag: set[Region], c: Region) -> None:
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

    def mark_shared_out(self, c: Region) -> None:
        """*c* may be held by something outside while this function holds it."""
        self._spread(self._shared_out, c)

    def mark_returned(self, c: Region) -> None:
        """*c* is handed to the caller by a ``return``."""
        self._spread(self._returned, c)

    def is_shared_out(self, c: Region) -> bool:
        return self.find(c) in self._shared_out

    def is_returned(self, c: Region) -> bool:
        return self.find(c) in self._returned

    def has_param_site(self, c: Region) -> bool:
        """Whether the caller already holds something in *c*'s class."""
        return any(s.kind == 'param' for s in self.sites_at(c))


@dataclass
class AliasAnalysis:
    """Result of :class:`Alias`.

    ``written_regions``, ``slot_replaced`` and ``returned_levels`` are syntactic
    facts keyed by region, for a consumer deciding how a list is held: whether a
    reference may be ``const``, whether it may be a reference at all, and what
    the ``return``s hand back together.

    Attributes:
        sites:            every list allocation, plus one per nested parameter list.
        written_regions:  regions an ``xs[i] = e`` here stores into.
        slot_replaced:    element regions an ``xss[i] = <list>`` replaces.
        returned_levels:  regions the ``return``s hand back, by depth.
    """

    sites: list[AllocSite]
    _regions: _Regions
    _by_def: dict[Definition, Region]
    _by_expr: dict[Expr, Region]
    _site_region: dict[AllocSite, Region]
    written_regions: set[Region] = field(default_factory=set)
    slot_replaced: set[Region] = field(default_factory=set)
    returned_levels: list[set[Region]] = field(default_factory=list)

    def region_of(self, d: Definition, depth: int = 0) -> Region | None:
        """What may be the same list as *d*, *depth* levels in.

        ``depth=0`` is *d* itself; ``depth=1`` is the lists held in its
        elements, and so on.  ``None`` when nothing is known about that level.
        """
        return self._walk(self._by_def.get(d), depth)

    def region_of_expr(self, e: Expr, depth: int = 0) -> Region | None:
        """As :meth:`region_of`, for an expression.

        An expression that names a definition shares its region, and so its
        region.
        """
        return self._walk(self._by_expr.get(e), depth)

    def _walk(self, region: Region | None, depth: int) -> Region | None:
        for _ in range(depth):
            if region is None:
                return None
            region = self._regions.part(region, missing='none')
        return None if region is None else self._regions.find(region)

    def all_defs(self) -> frozenset[Definition]:
        """Every definition the analysis gave a region to."""
        return frozenset(self._by_def)

    def region_at(self, region: Region, depth: int = 1) -> Region | None:
        """The region *depth* list levels inside *region*."""
        return self._walk(region, depth)

    def region_field(self, region: Region | None, i: int) -> Region | None:
        """The region held by field *i* of *region*'s tuple."""
        if region is None:
            return None
        return self._regions.part(region, i, missing='none')

    def defs_in(self, region: Region) -> frozenset[Definition]:
        """Every definition whose value may live in *region*."""
        out = {
            d for d, c in self._by_def.items()
            if self._regions.find(c) is region
        }
        return frozenset(out)

    def referrers(self, region: Region) -> int:
        """How many places may hold a reference into *region*: distinct source
        names plus container slots."""
        return self._regions.referrers(region)

    def region_of_site(self, site: AllocSite) -> Region | None:
        """Which region *site*'s allocation lives in."""
        region = self._site_region.get(site)
        return None if region is None else self._regions.find(region)

    def escapes_at(self, region: Region) -> bool:
        """Whether *region* is held by something outside while this function
        still holds it — handed to a call, or to an unmodelled operation."""
        return self._regions.is_shared_out(region)

    def returned_at(self, region: Region) -> bool:
        """Whether *region* is handed to the caller by a ``return``."""
        return self._regions.is_returned(region)

    def sites_at(self, region: Region | None) -> frozenset[AllocSite]:
        """The allocations that may live in *region*."""
        return frozenset() if region is None else self._regions.sites_at(region)

    def is_shared(self, site: AllocSite) -> bool:
        """Whether more than one place may refer to *site*."""
        return self._regions.referrers(self._site_region[site]) > 1

    def escapes(self, site: AllocSite) -> bool:
        """Whether *site*, or a container holding it, leaves the function at all
        — handed to a call, or returned.

        Says nothing about *which*; :meth:`transfers_ownership` separates them.
        """
        region = self._site_region[site]
        return self._regions.is_shared_out(region) or self._regions.is_returned(region)

    def is_returned(self, site: AllocSite) -> bool:
        """Whether *site*, or a container holding it, is handed to the caller by a
        ``return``."""
        return self._regions.is_returned(self._site_region[site])

    def transfers_ownership(self, site: AllocSite) -> bool:
        """Whether returning *site* hands out *sole* ownership of it.

        A ``return`` is a transfer, not sharing: the value moves to the caller
        and this function keeps nothing, so a copy at the boundary is
        unobservable.  Unless the caller already holds it — ``return xs`` on a
        parameter leaves two handles to one list — so a ``param`` site anywhere
        in the class blocks the transfer.  A class's site set is the union over
        everything merged into it, so this catches the indirect routes too.
        """
        region = self._site_region[site]
        return (
            self._regions.is_returned(region)
            and not self._regions.has_param_site(region)
            and not self._regions.is_shared_out(region)
            and not self.is_shared(site)
        )

    def is_uniquely_owned(self, site: AllocSite) -> bool:
        """Whether nothing else may observe *site* — the condition under which
        copying, or representing it by value, is unobservable.

        The analysis-level answer, which counts every name.  A consumer that
        knows its own bindings can refine it: the C++ backend discounts a name
        it will bind by reference, so it unboxes some sites this calls shared.
        Refining in that direction is safe; the reverse would not be.

        One referrer, never shared outward, and if it is returned then the return
        transfers ownership.
        """
        if self.is_shared(site) or self._regions.is_shared_out(self._site_region[site]):
            return False
        return not self.is_returned(site) or self.transfers_ownership(site)

    def may_alias(self, a: Definition, b: Definition) -> bool:
        """Whether *a* and *b* may refer to the same list."""
        ca, cb = self._by_def.get(a), self._by_def.get(b)
        if ca is None or cb is None:
            return False
        return self._regions.find(ca) is self._regions.find(cb)


class _EscapeVars(DefaultVisitor):
    """Marks every list-carrying variable inside an expression as shared outward.

    Used for expression kinds :meth:`_Builder._build_region` does not model: if an
    unmodelled operation might retain one of its operands, the sound answer is
    that the operand is no longer uniquely owned.  A visitor rather than an
    attribute walk so an unfamiliar node's children are still reached.
    """

    def __init__(self, builder: '_Builder'):
        self.builder = builder

    def _visit_var(self, e: Var, ctx):
        region = self.builder._region_for(e)
        if region is not None:
            self.builder.regions.mark_shared_out(region)


class _Builder(DefaultVisitor):
    """Generates the constraints for one function.

    A :class:`DefaultVisitor` so traversal is the framework's job: a node this
    analysis should care about cannot be silently skipped by a hand-written
    walk.  Only :meth:`_region_for` is explicit recursion, because it returns a
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
        self.regions = _Regions()
        self.sites: list[AllocSite] = []
        self.by_def: dict[Definition, Region] = {}
        self.site_region: dict[AllocSite, Region] = {}
        self._by_expr: dict[Expr, Region] = {}

    def run(self) -> AliasAnalysis:
        self._merge_redefinitions()
        self._seed_params()
        self._visit_function(self.func, None)
        return AliasAnalysis(
            self.sites, self.regions, self.by_def, self._by_expr,
            self.site_region,
        )

    # -- regions --------------------------------------------------------------

    def _part(self, c: Region, key: _PartKey = ELTS) -> Region:
        """A part of *c*, created on first mention."""
        return self.regions.part(c, key, missing='create')

    def _reg(self, d: Definition) -> Region:
        if d not in self.by_def:
            self.by_def[d] = self.regions.new('name', name=d.name)
        return self.regions.find(self.by_def[d])

    def _site(self, kind: str, node: Expr | Argument, region: Region,
              depth: int = 0) -> None:
        site = AllocSite(kind, node, depth)
        self.sites.append(site)
        self.regions.add_site(region, site)
        self.site_region[site] = self.regions.find(region)

    def _alloc(self, kind: str, node: Expr, *, deep: bool = False) -> Region:
        """A region for a value *node* brings into existence.

        *deep* also records a site for each nested level, which an expression
        needs when nothing else describes its elements — ``empty(r, c)`` or a
        call result.  Without one, that level has no site at all and a consumer
        cannot tell "nothing owns this" from "nothing is known about it".

        Off by default, because a literal or a comprehension *does* describe its
        elements: those are expressions of their own, and each allocates its own
        region.  Seeding on top of them would give one place two parts, which
        merge into two referrers and read as shared.
        """
        # transient: an allocation is not itself a place a reference is held
        region = self.regions.new(kind)
        self._site(kind, node, region)
        if not deep:
            return region
        ty = self.types.by_expr.get(node)
        cur, depth = region, 0
        while isinstance(ty, ListType) and isinstance(ty.elt, ListType):
            cur, ty, depth = self._part(cur), ty.elt, depth + 1
            self._site(kind, node, cur, depth)
        return region

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
            for i in same_object_defs(d):
                self.regions.merge(self._reg(d), self._reg(self.def_use.defs[i]))

    def _seed_params(self) -> None:
        for arg in self.func.args:
            if not isinstance(arg.name, NamedId):
                continue
            d = self.def_use.find_def_from_site(arg.name, arg)
            ty = self.types.by_def.get(d)
            if _carries_list(ty):
                self._seed(self._reg(d), ty, arg, 0)

    def _seed(self, region: Region, ty: Type | None, arg: Argument,
              depth: int) -> None:
        """One site per list inside a parameter's type: the caller owns the outer
        list *and* every list nested in it."""
        match ty:
            case ListType():
                self._site('param', arg, region, depth)
                if _carries_list(ty.elt):
                    self._seed(self._part(region), ty.elt, arg, depth + 1)
            case TupleType():
                for i, elt in enumerate(ty.elts):
                    if _carries_list(elt):
                        self._seed(self._part(region, i), elt, arg, depth)

    # -- the value-returning half ------------------------------------------

    def _region_for(self, e: Expr) -> Region | None:
        """The region *e* denotes, or ``None`` if *e* carries no list."""
        if not _carries_list(self.types.by_expr.get(e)):
            return None
        if e not in self._by_expr:
            region = self._build_region(e)
            if region is None:
                return None
            self._by_expr[e] = region
        return self.regions.find(self._by_expr[e])

    def _build_region(self, e: Expr) -> Region | None:
        match e:
            case Var():
                return self._reg(self.def_use.find_def_from_use(e))
            case ListRef():
                return self._project(e)
            case Fst() | Snd():
                base = self._region_for(e.args[0])
                field = 0 if isinstance(e, Fst) else 1
                return None if base is None else self._part(base, field)
            case IfExpr():
                # the result *is* one branch or the other, so it aliases both
                region = self.regions.new('branch')
                for branch in (e.ift, e.iff):
                    bc = self._region_for(branch)
                    if bc is not None:
                        region = self.regions.merge(region, bc)
                return region
            case ListSlice():
                # a fresh outer list over the *same* elements
                region = self._alloc('slice', e)
                base = self._region_for(e.value)
                if base is not None:
                    self.regions.merge(
                        self._part(region), self._part(base),
                    )
                return region
            case ListExpr():
                region = self._alloc('literal', e)
                for x in e.elts:
                    xc = self._region_for(x)
                    if xc is not None:
                        self.regions.merge(self._part(region), xc)
                return region
            case TupleExpr():
                region = self._alloc('literal', e)
                for i, x in enumerate(e.elts):
                    xc = self._region_for(x)
                    if xc is not None:
                        self.regions.merge(self._part(region, i), xc)
                return region
            case ListComp():
                region = self._alloc('comprehension', e)
                self._bind_comp_targets(e)
                xc = self._region_for(e.elt)
                if xc is not None:
                    self.regions.merge(self._part(region), xc)
                return region
            case Enumerate() | Zip():
                # Each element is a *tuple* over the sources' elements, so the
                # constraint is on a field of the element and not on the element
                # itself: `enumerate` pairs an index with the source element,
                # `zip` takes field i from argument i.  Merging the element parts
                # directly instead would conflate the tuple with what it holds,
                # putting a loop variable one level too deep -- conservative for
                # the site verdicts, but wrong for `may_alias`.
                region = self._alloc('builtin', e)
                elts = self._part(region)
                for i, a in enumerate(e.args):
                    ac = self._region_for(a)
                    if ac is not None:
                        field = 1 if isinstance(e, Enumerate) else i
                        self.regions.merge(
                            self._part(elts, field), self._part(ac),
                        )
                return region
            case Call():
                region = self._alloc('call', e, deep=True)
                self._escape_args(e)
                return region
            case Range1() | Range2() | Range3() | Empty():
                # A fresh sequence over *integer* arguments: there is nothing
                # list-shaped for it to retain, so unlike the catch-all below it
                # must not escape what its operands mention.  `range(len(xs))`
                # is a common enough shape that treating it conservatively
                # would box most parameters for no reason.
                return self._alloc('builtin', e, deep=True)
            case _:
                # An unmodelled expression: assume it allocates, and that it may
                # retain anything named inside it.
                region = self._alloc('builtin', e, deep=True)
                _EscapeVars(self)._visit_expr(e, None)
                return region

    def _project(self, e: ListRef) -> Region | None:
        """``xs[i]`` — the elements part of a list, or a field of a tuple."""
        base = self._region_for(e.value)
        if base is None:
            return None
        base_ty = self.types.by_expr.get(e.value)
        if not isinstance(base_ty, TupleType):
            return self._part(base)
        if isinstance(e.index, Integer):
            return self._part(base, e.index.val)
        # a tuple index that is not a literal: could be any field
        region = self._part(base, 0)
        for i in range(1, len(base_ty.elts)):
            region = self.regions.merge(region, self._part(base, i))
        return region

    def _bind_comp_targets(self, e: ListComp) -> None:
        """Bind each comprehension variable to the elements it iterates."""
        for target, iterable in zip(e.targets, e.iterables):
            it = self._region_for(iterable)
            if it is not None:
                self._bind(target, self._part(it), e)

    def _bind(self, target, region: Region, site) -> None:
        """Bind a name or a tuple pattern to *region*, field by field."""
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                if _carries_list(self.types.by_def.get(d)):
                    self.regions.merge(self._reg(d), region)
            case TupleBinding():
                for i, elt in enumerate(target.elts):
                    self._bind(elt, self._part(region, i), site)

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
            ac = self._region_for(a)
            if ac is None:
                continue
            if summary is None or summary.retains(i):
                self.regions.mark_shared_out(ac)

    # -- constraint-generating hooks ---------------------------------------
    #
    # Each does its own work then delegates, so the framework keeps traversing.

    def _visit_expr(self, e: Expr, ctx):
        # The net that makes the analysis total: every list-carrying expression
        # gets a region, even in a position no hook below intercepts.  Memoized, so
        # an expression a hook already handled is not built twice.
        self._region_for(e)
        return super()._visit_expr(e, ctx)

    def _visit_call(self, e: Call, ctx):
        """A list handed to a call escapes: whether the callee retains it is not
        decidable here."""
        self._escape_args(e)
        super()._visit_call(e, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        rhs = self._region_for(stmt.expr)
        if rhs is not None:
            self._bind(stmt.target, rhs, stmt)
        super()._visit_assign(stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        if isinstance(stmt.var, NamedId):
            d = self.def_use.find_def_from_site(stmt.var, stmt)
            cur = self._reg(d)
            rhs = self._region_for(stmt.expr)
            if rhs is not None:
                for _ in stmt.indices[:-1]:
                    cur = self._part(cur)
                self.regions.merge(self._part(cur), rhs)
        super()._visit_indexed_assign(stmt, ctx)

    def _visit_for(self, stmt: ForStmt, ctx):
        it = self._region_for(stmt.iterable)
        if it is not None:
            self._bind(stmt.target, self._part(it), stmt)
        super()._visit_for(stmt, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx):
        rhs = self._region_for(stmt.expr)
        if rhs is not None:
            self.regions.mark_returned(rhs)
        super()._visit_return(stmt, ctx)


class _RegionFacts(DefaultVisitor):
    """Syntactic facts about regions, read off a *finished* region graph.

    A second walk rather than part of :class:`_Builder`: each fact asks
    ``region_of`` for the region a place ends up in, which is only settled once
    every merge has run.  A bare traversal is under 2% of the builder's cost.
    """

    def __init__(self, alias: 'AliasAnalysis', def_use: DefineUseAnalysis):
        self.alias = alias
        self.def_use = def_use
        self.written: set[Region] = set()
        self.slot_replaced: set[Region] = set()
        self.returned_levels: list[set[Region]] = []
        for d in alias.all_defs():
            if isinstance(d, AssignDef) and isinstance(d.site, IndexedAssign):
                if (r := alias.region_of(d)) is not None:
                    self.written.add(r)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        # Only a store of a *list* replaces a slot: a scalar write goes through
        # the cell, leaving whatever region the element held intact.
        if (
            isinstance(stmt.var, NamedId)
            and self.alias.region_of_expr(stmt.expr) is not None
        ):
            d = self.def_use.find_def_from_site(stmt.var, stmt)
            if (r := self.alias.region_of(d, len(stmt.indices))) is not None:
                self.slot_replaced.add(r)
        super()._visit_indexed_assign(stmt, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx):
        depth = 0
        while (r := self.alias.region_of_expr(stmt.expr, depth)) is not None:
            while len(self.returned_levels) <= depth:
                self.returned_levels.append(set())
            self.returned_levels[depth].add(r)
            depth += 1
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
        alias = _Builder(func, def_use, type_info, summaries or {}).run()
        facts = _RegionFacts(alias, def_use)
        facts._visit_function(func, None)
        alias.written_regions = facts.written
        alias.slot_replaced = facts.slot_replaced
        alias.returned_levels = facts.returned_levels
        return alias
