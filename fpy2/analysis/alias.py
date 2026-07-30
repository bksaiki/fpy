"""
Alias analysis: what else may refer to a list.

FPy lists are objects with identity — assignment aliases, ``xs[i] = e`` mutates in
place, and passing, returning or projecting carries the identity along.  This
analysis answers, for each list a function creates or receives, *what else may
refer to it*.

The question is backend-independent: it is a fact about the FPy program, not about
any target.  A backend whose native sequence is a value type (``std::vector``, an
FPCore tensor) needs it to decide where a copy would be observable; the
interpreter needs no such decision, but the fact is the same either way.

Formulation
-----------

A *cell* is a place a list reference can sit: one per list-typed
:class:`~fpy2.analysis.Definition`, plus a lazily created *slot* cell standing for
"whatever lives inside" a cell's list.  Slot cells are what make nesting work —
the outer list of a ``list[list[Real]]`` holds references, so its elements are
cells in their own right.

Aliasing routes generate *equality* constraints between cells, solved with
union-find:

======================  =============================================
``ys = xs``             ``cell(ys) ≡ cell(xs)``
``row = xss[i]``        ``cell(row) ≡ slot(xss)``
``xss[i] = e``          ``slot(xss) ≡ cell(e)``
``[a, b]``              ``slot(result) ≡ cell(a) ≡ cell(b)``
``xs[i:j]``             ``slot(result) ≡ slot(xs)``  — a slice is *shallow*
``for row in xss``      ``cell(row) ≡ slot(xss)``
``enumerate``/``zip``   ``slot(result) ≡ slot(arg)``
======================  =============================================

Equality rather than inclusion (unification rather than a subset-based solver) is
deliberate.  Unification over-approximates aliasing, which is the safe direction
for every consumer: one that wrongly believes a list is shared merely forgoes an
optimization.  It is also adequate here — measured over the test corpus, only
three merges anywhere lose precision relative to a subset-based solver, because
FPy's one form of indirection is "a list slot holds a list reference", and list
types cannot recurse, so the slot chain terminates at a statically known depth.

Limitations
-----------

Flow-insensitive: ``ys = xs`` marks ``xs`` shared even if ``ys`` is dead
immediately afterwards.  Intraprocedural: a list handed to a call is treated as
escaping, without asking whether the callee retains it.  Tuple components are not
modelled, so a list placed in a tuple is conservatively shared.  All three cost
precision rather than soundness.
"""

from dataclasses import dataclass

from ..ast import (
    Argument,
    Assign,
    Call,
    DefaultVisitor,
    Enumerate,
    Expr,
    ForStmt,
    FuncDef,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListRef,
    ListSlice,
    NamedId,
    ReturnStmt,
    TupleBinding,
    TupleExpr,
    Var,
    Zip,
)
from ..types import ListType
from ..utils import Unionfind
from .define_use import DefineUse, DefineUseAnalysis
from .reaching_defs import Definition
from .type_infer import TypeAnalysis, TypeInfer


class AllocSite:
    """A place a list value comes into existence.

    ``kind`` is one of ``literal``, ``comprehension``, ``slice``, ``builtin``,
    ``call``, ``param``, or ``tuple-component``.  A ``param`` site stands for a
    list the *caller* allocated, and gets one site per level of the parameter's
    list type, since each level is a distinct object the caller also owns.

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
    """A place a list reference can sit: a name, or a container slot.

    An opaque token — the equivalence relation and per-class payload live in
    :class:`_Cells`.
    """

    __slots__ = ('_kind',)

    def __init__(self, kind: str):
        self._kind = kind

    def __repr__(self) -> str:
        return f'<cell {self._kind}>'


class _Cells:
    """Cells, the equivalence relation over them, and per-class payload.

    Payload lives in dicts keyed by the class representative, the same
    arrangement ``storage_infer`` uses for storage classes.
    """

    def __init__(self):
        self._uf: Unionfind[_Cell] = Unionfind()
        self._sites: dict[_Cell, set[AllocSite]] = {}
        # How many *referrers* a class has: somewhere a reference can still be
        # held, i.e. a name or a container slot.  Slots have to count -- that is
        # exactly how `[xs]` shares `xs`, and it is not a name.  The transient
        # cell for an allocation does not, so `xs = [x, x]` has one referrer
        # rather than two.
        self._refs: dict[_Cell, int] = {}
        self._escaping: set[_Cell] = set()
        self._slots: dict[_Cell, _Cell] = {}

    def new(self, kind: str, *, is_ref: bool = True) -> _Cell:
        cell = _Cell(kind)
        self._uf.add(cell)
        self._sites[cell] = set()
        self._refs[cell] = 1 if is_ref else 0
        return cell

    def find(self, c: _Cell) -> _Cell:
        return self._uf.find(c)

    def merge(self, a: _Cell, b: _Cell) -> _Cell:
        ra, rb = self.find(a), self.find(b)
        if ra is rb:
            return ra
        root = self._uf.union(ra, rb)
        other = rb if root is ra else ra
        self._sites[root] = (
            self._sites.pop(root, set()) | self._sites.pop(other, set())
        )
        self._refs[root] = self._refs.pop(root, 0) + self._refs.pop(other, 0)
        if other in self._escaping:
            self._escaping.discard(other)
            self._escaping.add(root)
        # merging two classes merges what lives inside them; this is the
        # recursive part, and it terminates because list types cannot recurse
        inner_root = self._slots.pop(root, None)
        inner_other = self._slots.pop(other, None)
        if inner_root is not None and inner_other is not None:
            self._slots[root] = inner_root
            self.merge(inner_root, inner_other)
        elif (inner := inner_root or inner_other) is not None:
            self._slots[root] = inner
        return root

    def slot(self, c: _Cell) -> _Cell:
        """The cell standing for what lives inside *c*'s list."""
        root = self.find(c)
        if root not in self._slots:
            self._slots[root] = self.new('slot')
        return self.find(self._slots[root])

    def add_site(self, c: _Cell, site: AllocSite) -> None:
        self._sites.setdefault(self.find(c), set()).add(site)

    def sites_at(self, c: _Cell) -> frozenset[AllocSite]:
        return frozenset(self._sites.get(self.find(c), ()))

    def referrers(self, c: _Cell) -> int:
        return self._refs.get(self.find(c), 0)

    def add_referrer(self, c: _Cell) -> None:
        root = self.find(c)
        self._refs[root] = self._refs.get(root, 0) + 1

    def mark_escaping(self, c: _Cell) -> None:
        self._escaping.add(self.find(c))

    def escapes(self, c: _Cell) -> bool:
        return self.find(c) in self._escaping


@dataclass
class AliasAnalysis:
    """Result of :class:`Alias`.

    ``sites`` lists every list allocation the function performs, plus one per
    level of each list-typed parameter.
    """

    sites: list[AllocSite]
    _cells: _Cells
    _cell_of: dict[Definition, _Cell]
    _site_cell: dict[AllocSite, _Cell]

    def sites_of(self, d: Definition) -> frozenset[AllocSite]:
        """The allocations *d* may refer to.  Empty if *d* is not a list."""
        cell = self._cell_of.get(d)
        return frozenset() if cell is None else self._cells.sites_at(cell)

    def is_shared(self, site: AllocSite) -> bool:
        """Whether more than one place may refer to *site*.

        ``True`` means a consumer cannot treat the list as uniquely owned: a copy
        of it would be observable through the other referrer.
        """
        return self._cells.referrers(self._site_cell[site]) > 1

    def escapes(self, site: AllocSite) -> bool:
        """Whether *site* is returned, or handed to a call.

        Either way something outside this function may hold it, so what becomes
        of it is not decidable here.
        """
        return self._cells.escapes(self._site_cell[site])

    def is_uniquely_owned(self, site: AllocSite) -> bool:
        """Whether *site* has exactly one referrer and does not escape — the
        condition under which copying the list is unobservable."""
        return not self.is_shared(site) and not self.escapes(site)

    def may_alias(self, a: Definition, b: Definition) -> bool:
        """Whether *a* and *b* may refer to the same list."""
        ca, cb = self._cell_of.get(a), self._cell_of.get(b)
        if ca is None or cb is None:
            return False
        return self._cells.find(ca) is self._cells.find(cb)


class _Builder(DefaultVisitor):
    """Generates the constraints for one function.

    A :class:`DefaultVisitor` so traversal is the framework's job: a node this
    analysis does not care about is still descended into, and one it *should*
    care about cannot be silently skipped by a hand-written walk.  Only the
    value-returning half is explicit recursion — :meth:`_cell_for` maps a
    list-producing expression to the cell it denotes, over a closed set of kinds.
    """

    def __init__(self, func: FuncDef, def_use: DefineUseAnalysis,
                 type_info: TypeAnalysis):
        self.func = func
        self.def_use = def_use
        self.types = type_info
        self.cells = _Cells()
        self.sites: list[AllocSite] = []
        self.cell_of: dict[Definition, _Cell] = {}
        self.site_cell: dict[AllocSite, _Cell] = {}

    def run(self) -> AliasAnalysis:
        self._seed_params()
        self._visit_function(self.func, None)
        return AliasAnalysis(
            self.sites, self.cells, self.cell_of, self.site_cell,
        )

    # -- cells --------------------------------------------------------------

    def _is_list(self, e: Expr) -> bool:
        return isinstance(self.types.by_expr.get(e), ListType)

    def _cell(self, d: Definition) -> _Cell:
        if d not in self.cell_of:
            self.cell_of[d] = self.cells.new('name')
        return self.cells.find(self.cell_of[d])

    def _site(self, kind: str, node: Expr | Argument, cell: _Cell,
              depth: int = 0) -> None:
        site = AllocSite(kind, node, depth)
        self.sites.append(site)
        self.cells.add_site(cell, site)
        self.site_cell[site] = self.cells.find(cell)

    def _alloc(self, kind: str, node: Expr) -> _Cell:
        # transient: an allocation is not itself a place a reference is held
        cell = self.cells.new(kind, is_ref=False)
        self._site(kind, node, cell)
        return cell

    def _seed_params(self) -> None:
        """One site per level of each list-typed parameter: the caller owns the
        outer list *and* every list nested inside it."""
        for arg in self.func.args:
            if not isinstance(arg.name, NamedId):
                continue
            d = self.def_use.find_def_from_site(arg.name, arg)
            level: object = self.types.by_def.get(d)
            if not isinstance(level, ListType):
                continue
            cur, depth = self._cell(d), 0
            while isinstance(level, ListType):
                self._site('param', arg, cur, depth)
                cur, level, depth = self.cells.slot(cur), level.elt, depth + 1

    # -- the value-returning half ------------------------------------------

    def _cell_for(self, e: Expr) -> _Cell | None:
        """The cell *e* denotes, or ``None`` if *e* is not a list.

        Explicit recursion rather than a visitor hook because it returns a value
        and the set of list-producing expressions is closed.  Traversal of
        everything *else* is left to :class:`DefaultVisitor`.
        """
        if not self._is_list(e):
            return None
        match e:
            case Var():
                return self._cell(self.def_use.find_def_from_use(e))
            case ListRef():
                base = self._cell_for(e.value)
                return self.cells.slot(base) if base is not None else None
            case ListSlice():
                # a fresh outer list over the *same* elements
                cell = self._alloc('slice', e)
                base = self._cell_for(e.value)
                if base is not None:
                    self.cells.merge(
                        self.cells.slot(cell), self.cells.slot(base),
                    )
                return cell
            case ListExpr() | TupleExpr():
                cell = self._alloc('literal', e)
                for x in e.elts:
                    xc = self._cell_for(x)
                    if xc is not None:
                        self.cells.merge(self.cells.slot(cell), xc)
                return cell
            case ListComp():
                cell = self._alloc('comprehension', e)
                self._bind_comp_targets(e)
                xc = self._cell_for(e.elt)
                if xc is not None:
                    self.cells.merge(self.cells.slot(cell), xc)
                return cell
            case Enumerate() | Zip():
                # the lowering copies each element into a tuple, so the result's
                # elements are the sources' elements
                cell = self._alloc('builtin', e)
                for a in e.args:
                    ac = self._cell_for(a)
                    if ac is not None:
                        self.cells.merge(
                            self.cells.slot(cell), self.cells.slot(ac),
                        )
                return cell
            case Call():
                return self._alloc('call', e)
            case _:
                # some other list-producing expression: assume it allocates
                return self._alloc('builtin', e)

    def _bind_comp_targets(self, e: ListComp) -> None:
        """Bind each comprehension variable to the elements it iterates."""
        for target, iterable in zip(e.targets, e.iterables):
            it = self._cell_for(iterable)
            if it is None:
                continue
            if isinstance(target, NamedId):
                d = self.def_use.find_def_from_site(target, e)
                self.cells.merge(self._cell(d), self.cells.slot(it))
            else:
                # a tuple-binding target destructures something not modelled
                self.cells.mark_escaping(it)

    def _share_and_escape(self, cell: _Cell) -> None:
        """Treat *cell* as reachable from somewhere this analysis cannot see."""
        self.cells.add_referrer(cell)
        self.cells.mark_escaping(cell)

    # -- constraint-generating hooks ---------------------------------------
    #
    # Each does its own work then delegates, so the framework keeps traversing.

    def _visit_call(self, e: Call, ctx):
        """A list handed to a call escapes: whether the callee retains it is not
        decidable here."""
        for a in e.args:
            ac = self._cell_for(a)
            if ac is not None:
                self.cells.mark_escaping(ac)
        super()._visit_call(e, ctx)

    def _visit_tuple_expr(self, e: TupleExpr, ctx):
        """A tuple is not list-typed, so a list placed in one would otherwise be
        invisible.  Components are not modelled, so treat it as shared."""
        if not self._is_list(e):
            for x in e.elts:
                xc = self._cell_for(x)
                if xc is not None:
                    self._share_and_escape(xc)
        super()._visit_tuple_expr(e, ctx)

    def _visit_assign(self, stmt: Assign, ctx):
        match stmt.target:
            case NamedId() as name:
                rhs = self._cell_for(stmt.expr)
                if rhs is not None:
                    d = self.def_use.find_def_from_site(name, stmt)
                    self.cells.merge(self._cell(d), rhs)
            case TupleBinding() as binding:
                # a list destructured out of a tuple; see `_visit_tuple_expr`
                for name in binding.names():
                    d = self.def_use.find_def_from_site(name, stmt)
                    if isinstance(self.types.by_def.get(d), ListType):
                        cell = self._cell(d)
                        self._site('tuple-component', stmt.expr, cell)
                        self._share_and_escape(cell)
        super()._visit_assign(stmt, ctx)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        if isinstance(stmt.var, NamedId):
            rhs = self._cell_for(stmt.expr)
            if rhs is not None:
                cur = self._cell(
                    self.def_use.find_def_from_site(stmt.var, stmt),
                )
                for _ in stmt.indices[:-1]:
                    cur = self.cells.slot(cur)
                self.cells.merge(self.cells.slot(cur), rhs)
        super()._visit_indexed_assign(stmt, ctx)

    def _visit_for(self, stmt: ForStmt, ctx):
        it = self._cell_for(stmt.iterable)
        if it is not None:
            if isinstance(stmt.target, NamedId):
                d = self.def_use.find_def_from_site(stmt.target, stmt)
                self.cells.merge(self._cell(d), self.cells.slot(it))
            else:
                self.cells.mark_escaping(it)
        super()._visit_for(stmt, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx):
        rhs = self._cell_for(stmt.expr)
        if rhs is not None:
            self.cells.mark_escaping(rhs)
        super()._visit_return(stmt, ctx)


class Alias:
    """Alias analysis for FPy programs."""

    @staticmethod
    def analyze(
        func: FuncDef,
        def_use: DefineUseAnalysis | None = None,
        type_info: TypeAnalysis | None = None,
    ) -> AliasAnalysis:
        """Compute what may refer to each list in *func*.

        Args:
            func: the function to analyze.
            def_use: reuse an existing def-use analysis.
            type_info: reuse an existing type analysis (needed to tell which
                expressions are lists).
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {func}")
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        return _Builder(func, def_use, type_info).run()
