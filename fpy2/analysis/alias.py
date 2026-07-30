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

A *cell* is a place a list reference can sit: one per list-carrying
:class:`~fpy2.analysis.Definition`, one per list-carrying expression, and — created
lazily — one per *part* of a cell.  A part is either the elements of a list (index
not tracked, since it is rarely a constant) or field *i* of a tuple (arity is
static, so fields are kept apart).  Parts are what make nesting work: the outer
list of a ``list[list[Real]]`` holds references, so its elements are cells in their
own right, and so is the list inside a ``tuple[list[Real], Real]``.

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
deliberate.  Unification over-approximates aliasing, which is the safe direction
for every consumer: one that wrongly believes a list is shared merely forgoes an
optimization.  It is also adequate here — measured over the test corpus, only
three merges anywhere lose precision relative to a subset-based solver, because
FPy's one form of indirection is "a container part holds a reference", and neither
list nor tuple types can recurse, so the part chain terminates at a statically
known depth.

An expression kind not in the table above is handled conservatively rather than
optimistically: it gets its own allocation site, and every list-carrying variable
inside it is marked escaping, so an unmodelled route cannot make a list *look*
uniquely owned.

Limitations
-----------

Flow-insensitive: ``ys = xs`` marks ``xs`` shared even if ``ys`` is dead
immediately afterwards, and a container part counts as a referrer even if the
container itself is never read.  Intraprocedural: a list handed to a call is
treated as escaping, without asking whether the callee retains it.  A list index is
not tracked, so ``xs[0]`` and ``xs[1]`` are the same part.  All three cost
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
    ReturnStmt,
    Snd,
    TupleBinding,
    TupleExpr,
    Var,
    Zip,
)
from ..types import ListType, TupleType, Type
from ..utils import Unionfind
from .define_use import DefineUse, DefineUseAnalysis
from .reaching_defs import Definition
from .type_infer import TypeAnalysis, TypeInfer

ELTS = None
"""Part key for the elements of a list.

A single key for the whole list: an index is rarely a constant, so tracking one
part per index would buy almost nothing.  Tuple fields use their integer index,
because a tuple's arity is static.
"""

_PartKey = int | None


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


class _Cells:
    """Cells, the equivalence relation over them, and per-class payload.

    Payload lives in dicts keyed by the class representative, the same
    arrangement ``storage_infer`` uses for storage classes.
    """

    def __init__(self):
        self._uf: Unionfind[_Cell] = Unionfind()
        self._sites: dict[_Cell, set[AllocSite]] = {}
        # How many *referrers* a class has: somewhere a reference can still be
        # held, i.e. a name or a container part.  Parts have to count -- that is
        # exactly how `[xs]` shares `xs`, and it is not a name.  The transient
        # cell for an allocation does not, so `xs = [x, x]` has one referrer
        # rather than two.
        self._refs: dict[_Cell, int] = {}
        self._escaping: set[_Cell] = set()
        self._parts: dict[_Cell, dict[_PartKey, _Cell]] = {}

    def new(self, kind: str, *, is_ref: bool = True) -> _Cell:
        cell = _Cell(kind)
        self._uf.add(cell)
        self._sites[cell] = set()
        self._refs[cell] = 1 if is_ref else 0
        return cell

    def find(self, c: _Cell) -> _Cell:
        return self._uf.find(c)

    def part(self, c: _Cell, key: _PartKey = ELTS) -> _Cell:
        """The cell standing for part *key* of *c*: its elements, or a field."""
        root = self.find(c)
        parts = self._parts.setdefault(root, {})
        if key not in parts:
            parts[key] = self.new('part')
            if root in self._escaping:
                # created after the container escaped -- see `mark_escaping`
                self.mark_escaping(parts[key])
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
            self._refs[root] = self._refs.pop(root, 0) + self._refs.pop(other, 0)
            escaping = root in self._escaping or other in self._escaping
            self._escaping.discard(other)
            mine = self._parts.pop(root, {})
            theirs = self._parts.pop(other, {})
            self._parts[root] = mine
            for key, cell in theirs.items():
                if key in mine:
                    pending.append((mine[key], cell))
                else:
                    mine[key] = cell
            if escaping:
                # a class merged into an escaping one escapes, parts and all
                self.mark_escaping(root)
        return self.find(a)

    def add_site(self, c: _Cell, site: AllocSite) -> None:
        self._sites.setdefault(self.find(c), set()).add(site)

    def sites_at(self, c: _Cell) -> frozenset[AllocSite]:
        return frozenset(self._sites.get(self.find(c), ()))

    def referrers(self, c: _Cell) -> int:
        return self._refs.get(self.find(c), 0)

    def mark_escaping(self, c: _Cell) -> None:
        """Mark *c* escaping, and everything reachable inside it.

        Escape is transitive downward: handing out a ``list[list[Real]]`` hands
        out its rows too.  Parts created *afterwards* inherit the flag in
        :meth:`part`, so the two directions together are order-independent.
        """
        pending, seen = [self.find(c)], set()
        while pending:
            root = self.find(pending.pop())
            if root in seen:
                continue
            seen.add(root)
            self._escaping.add(root)
            pending.extend(self._parts.get(root, {}).values())

    def escapes(self, c: _Cell) -> bool:
        return self.find(c) in self._escaping


@dataclass
class AliasAnalysis:
    """Result of :class:`Alias`.

    ``sites`` lists every list allocation the function performs, plus one per
    list nested in each parameter's type.
    """

    sites: list[AllocSite]
    _cells: _Cells
    _cell_of: dict[Definition, _Cell]
    _site_cell: dict[AllocSite, _Cell]

    def sites_of(self, d: Definition) -> frozenset[AllocSite]:
        """The allocations *d* may refer to.  Empty if *d* carries no list."""
        cell = self._cell_of.get(d)
        return frozenset() if cell is None else self._cells.sites_at(cell)

    def is_shared(self, site: AllocSite) -> bool:
        """Whether more than one place may refer to *site*.

        ``True`` means a consumer cannot treat the list as uniquely owned: a copy
        of it would be observable through the other referrer.
        """
        return self._cells.referrers(self._site_cell[site]) > 1

    def escapes(self, site: AllocSite) -> bool:
        """Whether *site*, or a container holding it, is returned or handed to a
        call.

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


class _EscapeVars(DefaultVisitor):
    """Marks every list-carrying variable inside an expression as escaping.

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
            self.builder.cells.mark_escaping(cell)


class _Builder(DefaultVisitor):
    """Generates the constraints for one function.

    A :class:`DefaultVisitor` so traversal is the framework's job: a node this
    analysis does not care about is still descended into, and one it *should*
    care about cannot be silently skipped by a hand-written walk.  Only the
    value-returning half is explicit recursion — :meth:`_cell_for` maps an
    expression to the cell it denotes, memoized so that reaching the same
    expression twice (once through a hook, once through traversal) does not
    allocate twice.
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
        self._by_expr: dict[Expr, _Cell] = {}

    def run(self) -> AliasAnalysis:
        self._seed_params()
        self._visit_function(self.func, None)
        return AliasAnalysis(
            self.sites, self.cells, self.cell_of, self.site_cell,
        )

    # -- cells --------------------------------------------------------------

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
                    self._seed(self.cells.part(cell), ty.elt, arg, depth + 1)
            case TupleType():
                for i, elt in enumerate(ty.elts):
                    if _carries_list(elt):
                        self._seed(self.cells.part(cell, i), elt, arg, depth)

    # -- the value-returning half ------------------------------------------

    def _cell_for(self, e: Expr) -> _Cell | None:
        """The cell *e* denotes, or ``None`` if *e* carries no list.

        Explicit recursion rather than a visitor hook because it returns a value.
        Traversal of everything *else* is left to :class:`DefaultVisitor`.
        """
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
                return None if base is None else self.cells.part(base, field)
            case IfExpr():
                # the result *is* one branch or the other, so it aliases both
                cell = self.cells.new('branch', is_ref=False)
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
                        self.cells.part(cell), self.cells.part(base),
                    )
                return cell
            case ListExpr():
                cell = self._alloc('literal', e)
                for x in e.elts:
                    xc = self._cell_for(x)
                    if xc is not None:
                        self.cells.merge(self.cells.part(cell), xc)
                return cell
            case TupleExpr():
                cell = self._alloc('literal', e)
                for i, x in enumerate(e.elts):
                    xc = self._cell_for(x)
                    if xc is not None:
                        self.cells.merge(self.cells.part(cell, i), xc)
                return cell
            case ListComp():
                cell = self._alloc('comprehension', e)
                self._bind_comp_targets(e)
                xc = self._cell_for(e.elt)
                if xc is not None:
                    self.cells.merge(self.cells.part(cell), xc)
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
                elts = self.cells.part(cell)
                for i, a in enumerate(e.args):
                    ac = self._cell_for(a)
                    if ac is not None:
                        field = 1 if isinstance(e, Enumerate) else i
                        self.cells.merge(
                            self.cells.part(elts, field), self.cells.part(ac),
                        )
                return cell
            case Call():
                cell = self._alloc('call', e)
                self._escape_args(e)
                return cell
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
            return self.cells.part(base)
        if isinstance(e.index, Integer):
            return self.cells.part(base, e.index.val)
        # a tuple index that is not a literal: could be any field
        cell = self.cells.part(base, 0)
        for i in range(1, len(base_ty.elts)):
            cell = self.cells.merge(cell, self.cells.part(base, i))
        return cell

    def _bind_comp_targets(self, e: ListComp) -> None:
        """Bind each comprehension variable to the elements it iterates."""
        for target, iterable in zip(e.targets, e.iterables):
            it = self._cell_for(iterable)
            if it is not None:
                self._bind(target, self.cells.part(it), e)

    def _bind(self, target, cell: _Cell, site) -> None:
        """Bind a name or a tuple pattern to *cell*, field by field."""
        match target:
            case NamedId():
                d = self.def_use.find_def_from_site(target, site)
                if _carries_list(self.types.by_def.get(d)):
                    self.cells.merge(self._cell(d), cell)
            case TupleBinding():
                for i, elt in enumerate(target.elts):
                    self._bind(elt, self.cells.part(cell, i), site)

    def _escape_args(self, e: Call) -> None:
        for a in e.args:
            ac = self._cell_for(a)
            if ac is not None:
                self.cells.mark_escaping(ac)

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
            rhs = self._cell_for(stmt.expr)
            if rhs is not None:
                cur = self._cell(
                    self.def_use.find_def_from_site(stmt.var, stmt),
                )
                for _ in stmt.indices[:-1]:
                    cur = self.cells.part(cur)
                self.cells.merge(self.cells.part(cur), rhs)
        super()._visit_indexed_assign(stmt, ctx)

    def _visit_for(self, stmt: ForStmt, ctx):
        it = self._cell_for(stmt.iterable)
        if it is not None:
            self._bind(stmt.target, self.cells.part(it), stmt)
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
                expressions carry lists).
        """
        if not isinstance(func, FuncDef):
            raise TypeError(f"expected a 'FuncDef', got {func}")
        if def_use is None:
            def_use = DefineUse.analyze(func)
        if type_info is None:
            type_info = TypeInfer.check(func, def_use=def_use)
        return _Builder(func, def_use, type_info).run()
