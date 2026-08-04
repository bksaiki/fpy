"""
cpp backend: storage-type inference.

Assigns each SSA def to a C++ variable: the variable's identifier and
its storage type.  The emitter consumes the result directly — every
``Var``/``Assign`` is just a lookup.

The partition is computed via union-find over two kinds of coalescing
edges:

- **Phi edges.**  A phi merge means both incoming defs must write to
  the same C++ variable, so they share storage.
- **In-place mutation edges.**  An ``IndexedAssign`` (``xs[i] = e``)
  produces an SSA-fresh def of ``xs`` so value-tracking analyses can
  reason about it, but physically the same vector is mutated — the
  new def is unioned with its ``prev`` so they share a single C++
  name and the emitter produces a direct subscript-store.

Anything *not* connected by either edge is free to rename, so a
sequential rebind of a name without a phi merge gets its *own*
variable with its *own* (possibly narrower) storage type.

Storage per class is chosen by aggregating every member's
:class:`FormatBound` through :func:`aggregate_storage`.  Only members
of the same class need to fit in a common type; cross-class storage
is independent.

Naming per class: function-argument and free-variable defs anchor a
class to the bare source name (the C++ signature already declares it
under that name).  Other classes for the same source name pick up
numeric suffixes (``x_1``, ``x_2``, …).
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce

from ...analysis import Definition
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.format_infer import FormatBound
from ...analysis.format_infer.analysis import (
    ListFormat,
    TupleFormat,
    _join_bounds,
)
from ...analysis.reaching_defs import AssignDef, PhiDef
from ...ast.fpyast import (
    Argument,
    Assign,
    Expr,
    ForStmt,
    FuncDef,
    IfExpr,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListRef,
    NamedId,
    Stmt,
    TupleBinding,
    TupleExpr,
    Var,
)
from ...ast.visitor import DefaultVisitor
from ...utils import Unionfind
from .storage import StorageSelectionError, aggregate_storage
from .types import CppList, CppTuple, CppType


@dataclass
class StorageAnalysis:
    """
    Result of :class:`StorageInfer`.

    Each SSA def is assigned to a *class* (the union-find equivalence
    class over phi and in-place mutation edges), and each class is
    assigned a single C++ identifier and storage type.  The class id
    is the union-find representative — the canonical
    :class:`Definition` standing in for the whole class.  Storage
    classes split into two emission shapes:

    - ``declare_at_assign``: the lowest-index writer in the class is
      its declaration site.  The emitter folds the declaration into
      that assign, e.g. ``double t = (a + b);``,
      ``for (int64_t i = 0; …)``, or ``double y = x;`` immediately
      followed by reassignments inside an ``if1`` body or loop.
    - ``hoists_before``: a class has writers in disjoint branches of
      an ``if/else`` and the variable did not exist before the
      ``if`` (the merge phi has ``is_intro=True``).  In that case
      no single AssignDef dominates the others, so the emitter
      hoists ``T name{};`` *just before* the responsible ``IfStmt``.
      Each ``AssignDef`` in the class then reassigns into that
      variable.

    External classes (containing a function arg or free variable)
    don't appear in either set: the C++ signature / surrounding scope
    already declares them.

    Attributes:
        def_class:          maps each def to its class id (the canonical
                            representative def of the class).
        class_members:      maps each class id to its member defs.
        class_storage:      the C++ storage chosen for each class.
        def_to_name:        the C++ identifier each def reads/writes
                            through.
        hoists_before:      maps each anchor statement (an ``IfStmt``)
                            to the storage classes whose declarations
                            the emitter must emit *just before* that
                            statement.
        declare_at_assign:  AssignDefs whose statement should declare
                            *and* assign in one go (the canonical
                            declaration site of their class).
    """
    def_class: dict[Definition, Definition]
    class_members: dict[Definition, list[Definition]]
    class_storage: dict[Definition, CppType]
    def_to_name: dict[Definition, str]
    hoists_before: dict[Stmt, list[Definition]]
    declare_at_assign: set[AssignDef]

    def storage_of(self, d: Definition) -> CppType:
        """Convenience: the C++ storage type chosen for *d*'s class."""
        return self.class_storage[self.def_class[d]]

    def is_single_def(self, d: Definition) -> bool:
        """True iff *d*'s class has exactly one member — i.e. the C++
        variable is written exactly once (never reassigned via a phi edge
        nor mutated in place).  Such a binding can be a ``const`` reference
        to its initializer instead of an owning copy."""
        return len(self.class_members[self.def_class[d]]) == 1


def is_rebound(storage: 'StorageAnalysis', d: Definition) -> bool:
    """Is the name *d* introduces ever bound to a different value?

    ``xs[i] = e`` is not a rebind — it mutates the list the name already refers
    to, and that def stays in the same class.  Only an ``Assign`` to the same
    name is, and *d*'s own defining assignment does not count.
    """
    cls = storage.def_class[d]
    return any(
        m is not d and isinstance(m, AssignDef) and isinstance(m.site, Assign)
        for m in storage.class_members[cls]
    )


def binds_by_reference(
    storage: 'StorageAnalysis',
    def_use: DefineUseAnalysis,
    d: Definition,
    *,
    allow_projection: bool = False,
) -> bool:
    """Whether the emitter binds *d*'s name as a reference to storage that
    already exists, rather than giving it a place of its own.

    One definition for a question two modules ask: the emitter, to choose
    between a reference and a copy, and :mod:`.unbox`, to decide whether a name
    is a second *place*.  They must agree — discounting a name the emitter then
    copies is a miscompilation — so they share this rather than mirror it.

    All three emitter sites require the name never be rebound, since a ``const``
    reference cannot be.

    *allow_projection* enables ``row = xss[i]``, which a reference can bind only
    where nothing replaces that slot — a caller establishes that from
    :meth:`UnboxAnalysis.may_reference_projection` and passes the answer, so the
    rule stays in one place while the fact it needs comes from the alias
    analysis.
    """
    if not isinstance(storage.storage_of(d), (CppList, CppTuple)):
        return False
    if is_rebound(storage, d):
        return False
    if not _binds_the_whole_value(d):
        return False
    match d.site:
        case Argument() | ForStmt() | ListComp():
            return True
        case Assign(expr=Var() as src):
            return (
                d in storage.declare_at_assign
                and not is_rebound(storage, def_use.find_def_from_use(src))
            )
        case Assign(expr=ListRef() as ref) if allow_projection:
            root = _root_var(ref)
            return (
                root is not None
                and d in storage.declare_at_assign
                and not is_rebound(storage, def_use.find_def_from_use(root))
            )
        case _:
            return False


def _binds_the_whole_value(d: Definition) -> bool:
    """Whether *d*'s binding hands it the whole value.

    A name destructured out of a tuple does not qualify however the tuple
    arrived: the emitter reads it with ``std::get``, which copies.  Harmless
    while the component is a handle — the copy still shares — but a copy of a
    value is a second place, and treating it as a reference would lose writes.
    """
    match d.site:
        case Assign(target=target) | ForStmt(target=target):
            return isinstance(target, NamedId)
        case ListComp(targets=targets):
            return any(t is d.name for t in targets)
        case _:
            return True


def _root_var(e: Expr) -> Var | None:
    """The variable a chain of subscripts is rooted at.

    A slice is not one: it materializes a new list, so a reference into it
    would outlive nothing.
    """
    while isinstance(e, ListRef):
        e = e.value
    return e if isinstance(e, Var) else None


def _is_external(members: list[Definition]) -> bool:
    for d in members:
        if isinstance(d, AssignDef) and (
            isinstance(d.site, Argument) or d.is_free
        ):
            return True
    return False


class _PlaceFloors(DefaultVisitor):
    """Per-def lower bounds from the places each def's value reaches.

    One place admits one C++ type.  :meth:`emitter._emit_at` builds a
    *constructor* at the place's type, but a variable cannot be rebuilt that
    way: changing a list's element type means a new buffer, and a shared list
    cannot survive one — which is the refusal in ``_convert_storage``.  So raise
    the variable's own storage instead, and there is nothing left to convert.

    Lists only.  A scalar converts free at the point of use, so widening its
    declaration would change a signature to no purpose.

    The shape mirrors ``_emit_at`` deliberately: the same set of expressions
    contribute to a place, and the two must agree about which.
    """

    def __init__(
        self,
        def_use: DefineUseAnalysis,
        by_def: dict[Definition, FormatBound],
        by_expr: dict[Expr, FormatBound],
        ret_fmt: FormatBound,
        base: 'StorageAnalysis',
        is_called: bool,
    ):
        self.def_use = def_use
        self.by_def = by_def
        self.by_expr = by_expr
        self.ret_fmt = ret_fmt
        self.base = base
        self.is_called = is_called
        self.floors: dict[Definition, FormatBound] = {}
        self.assigns: list[Assign] = []
        self.indexed: list[IndexedAssign] = []
        self.loops: list[ForStmt] = []
        self.collected = False
        self.changed = False

    def _pinned(self, d: Definition) -> bool:
        """Whether *d*'s storage *class* has anything that cannot be raised.

        Asked about the class, not the definition: :meth:`StorageInfer.infer`
        joins a floor into the whole class's bound, so a floor landing on one
        member raises every member.  Checking only the def the floor arrived at
        pins nothing — an ``xs[i] = e`` def shares its class with its ``prev``
        by the in-place-mutation edge and is itself unpinnable, so it would
        raise the very parameter or reference binding the check refused.
        """
        cls = self.base.def_class[d]
        return any(
            self._unraisable(m) for m in self.base.class_members[cls]
        )

    def _unraisable(self, d: Definition) -> bool:
        """Whether *d* alone has no storage of its own to raise.

        The line is whether the emitter *spells* the type or lets C++ deduce it,
        because raising only means anything when something reads ``storage_of``:

        - **Spelled, so raisable.**  A parameter (``_arg_decl``), a loop target
          and a comprehension target (``_foreach_decl``) all print
          ``storage_of``, so raising one changes the declaration and the
          reference follows.
        - **Deduced, so not.**  ``ys = xs`` and ``row = xss[i]`` are emitted as
          ``const auto&``, naming storage that already exists.  Raising one only
          makes ``storage_of`` describe a type the reference does not have, and
          the ``auto`` hides it until the mismatch surfaces somewhere else.

        One exception: a parameter of a function compiled code *calls*.  Its
        caller's argument type is already fixed, so raising it underneath emits a
        call that does not compile — the rule ``unbox`` states for
        representations, that a called function keeps its signature on both
        sides.  A native caller is not bound by it.
        """
        if isinstance(d.site, Argument):
            return self.is_called
        if isinstance(d.site, ForStmt | ListComp):
            return False
        return binds_by_reference(
            self.base, self.def_use, d, allow_projection=True,
        )

    def _raise(self, d: Definition, fmt: FormatBound) -> None:
        if self._pinned(d):
            return
        cur = self.floors.get(d)
        raised = fmt if cur is None else _join_bounds(cur, fmt)
        if raised != cur:
            self.floors[d] = raised
            self.changed = True

    def _push(self, e: Expr, fmt: FormatBound) -> None:
        if not isinstance(fmt, ListFormat):
            return
        match e:
            case Var():
                self._raise(self.def_use.find_def_from_use(e), fmt)
            case ListExpr():
                for elt in e.elts:
                    self._push(elt, fmt.elt)
            case ListComp():
                self._push(e.elt, fmt.elt)
            case IfExpr():
                self._push(e.ift, fmt)
                self._push(e.iff, fmt)

    def _push_tuple(self, e: Expr, fmt: FormatBound) -> None:
        """A tuple's fields are separate places, each with its own type."""
        if isinstance(e, TupleExpr) and isinstance(fmt, TupleFormat):
            if len(e.elts) == len(fmt.elts):
                for elt, sub in zip(e.elts, fmt.elts):
                    self._push_tuple(elt, sub)
                    self._push(elt, sub)

    def _visit_return(self, stmt, ctx):
        super()._visit_return(stmt, ctx)
        self._push(stmt.expr, self.ret_fmt)
        self._push_tuple(stmt.expr, self.ret_fmt)

    def _visit_list_expr(self, e: ListExpr, ctx):
        super()._visit_list_expr(e, ctx)
        fmt = self.by_expr.get(e)
        if isinstance(fmt, ListFormat):
            for elt in e.elts:
                self._push(elt, fmt.elt)
                self._push_tuple(elt, fmt.elt)

    def _visit_tuple_expr(self, e: TupleExpr, ctx):
        super()._visit_tuple_expr(e, ctx)
        self._push_tuple(e, self.by_expr.get(e))

    def _visit_if_expr(self, e: IfExpr, ctx):
        super()._visit_if_expr(e, ctx)
        fmt = self.by_expr.get(e)
        self._push(e, fmt)
        self._push_tuple(e, fmt)

    def _visit_assign(self, stmt: Assign, ctx):
        super()._visit_assign(stmt, ctx)
        # Once: the body is walked once per round, and re-appending would grow
        # the list every time — harmless for the result, since raising a floor
        # twice is idempotent, but it would make the iteration bound a lie.
        if not self.collected:
            self.assigns.append(stmt)

    def _visit_indexed_assign(self, stmt: IndexedAssign, ctx):
        super()._visit_indexed_assign(stmt, ctx)
        if not self.collected:
            self.indexed.append(stmt)

    def _visit_for(self, stmt: ForStmt, ctx):
        super()._visit_for(stmt, ctx)
        if not self.collected:
            self.loops.append(stmt)

    # -- the other direction ------------------------------------------------

    def _class_bound(self, d: Definition) -> FormatBound:
        """The bound *d*'s C++ variable will actually have.

        Over the whole storage class, because that is the unit
        :meth:`StorageInfer.infer` joins: a floor on any member decides them
        all.  Reading one def's own floor is how ``L6`` came out ``double`` while
        the tuple it was destructured from stayed ``uint8_t`` -- the return
        raised the ``L6[i] = e`` def, and the destructuring def next to it in the
        same class had no floor of its own to find.
        """
        cls = self.base.def_class[d]
        out: FormatBound = None
        for m in self.base.class_members[cls]:
            for b in (self.by_def.get(m), self.floors.get(m)):
                if b is None:
                    continue
                try:
                    out = b if out is None else _join_bounds(out, b)
                except RuntimeError:
                    return None       # incompatible kinds: not a widening question
        return out

    def _effective(self, e: Expr) -> FormatBound:
        """*e*'s bound with the floors its leaves have picked up.

        Raising a variable raises every container built from it: once ``base``
        is a ``vector<double>``, ``scratch = [base]`` has to be a
        ``vector<vector<double>>`` or the container's element type and the
        variable's own declaration disagree — which is the same conflict one
        level out.
        """
        match e:
            case Var():
                return self._class_bound(self.def_use.find_def_from_use(e))
            case ListExpr() if e.elts:
                elts = [self._effective(x) for x in e.elts]
                return ListFormat(reduce(_join_bounds, elts))
            case TupleExpr():
                return TupleFormat(tuple(self._effective(x) for x in e.elts))
            case ListComp():
                return ListFormat(self._effective(e.elt))
            case IfExpr():
                return _join_bounds(
                    self._effective(e.ift), self._effective(e.iff),
                )
        return self.by_expr.get(e)

    def _propagate_up(self) -> None:
        """Carry each floor to the other places that must agree with it."""
        for stmt in self.assigns:
            try:
                eff = self._effective(stmt.expr)
            except RuntimeError:
                continue          # incompatible kinds: not a widening question
            match stmt.target:
                case NamedId():
                    if isinstance(eff, ListFormat | TupleFormat):
                        d = self.def_use.find_def_from_site(stmt.target, stmt)
                        self._raise(d, eff)
                case TupleBinding():
                    self._bind_destructured(stmt, eff)
        for ix in self.indexed:
            self._raise_container(ix)
        for loop in self.loops:
            self._bind_loop_target(loop)

    def _bind_loop_target(self, stmt: ForStmt) -> None:
        """``for row in xss`` reads an element, so the two must agree.

        The target's type is spelled from ``storage_of``, so raising the
        container alone leaves ``for (const fpy::list<float>& row :
        std::vector<fpy::list<double>>)``.  Both directions, as for a
        destructuring: neither side can be left behind.
        """
        if not isinstance(stmt.target, NamedId):
            return          # a destructuring target rides on the tuple instead
        d = self.def_use.find_def_from_site(stmt.target, stmt)
        try:
            eff = self._effective(stmt.iterable)
        except RuntimeError:
            return
        if isinstance(eff, ListFormat) and isinstance(
            eff.elt, ListFormat | TupleFormat,
        ):
            self._raise(d, eff.elt)
        if not isinstance(stmt.iterable, Var):
            return
        mine = self._class_bound(d)
        if isinstance(mine, ListFormat | TupleFormat):
            self._raise(
                self.def_use.find_def_from_use(stmt.iterable), ListFormat(mine),
            )

    def _bind_destructured(self, stmt: Assign, eff: FormatBound) -> None:
        """``a, b = t`` reads each name with ``std::get``, so its type is the
        tuple's field type, not one it can choose.

        A name here has no storage of its own to raise -- but it cannot be
        pinned either, because the *tuple* may be raised and then the field it
        reads has moved.  So the floor goes down from the tuple's fields to the
        names, which is the direction the emitter cannot fix.
        """
        if not isinstance(eff, TupleFormat):
            return
        target = stmt.target
        if not isinstance(target, TupleBinding):
            return
        if len(target.elts) != len(eff.elts):
            return
        for sub, field in zip(target.elts, eff.elts):
            if isinstance(sub, NamedId) and isinstance(
                field, ListFormat | TupleFormat,
            ):
                self._raise(self.def_use.find_def_from_site(sub, stmt), field)
        # ...and back up.  A name raised from somewhere else -- returned wide,
        # say -- has to drag the field it reads with it, or the two disagree the
        # other way round.
        if not isinstance(stmt.expr, Var):
            return
        fields: list[FormatBound] = []
        for sub, field in zip(target.elts, eff.elts):
            if not isinstance(sub, NamedId):
                fields.append(field)
                continue
            d = self.def_use.find_def_from_site(sub, stmt)
            mine = self._class_bound(d)
            fields.append(field if mine is None else _join_bounds(field, mine))
        self._raise(
            self.def_use.find_def_from_use(stmt.expr), TupleFormat(tuple(fields)),
        )

    def _raise_container(self, stmt: IndexedAssign) -> None:
        """``xss[i] = e`` puts *e* in a slot, so the container's element level
        has to hold it.

        The emitter stores straight into the slot and never converts there, so
        a floor on *e* alone would leave the two disagreeing.  One ``ListFormat``
        per index peeled.
        """
        try:
            eff = self._effective(stmt.expr)
        except RuntimeError:
            return
        if not isinstance(eff, ListFormat | TupleFormat):
            return
        for _ in stmt.indices:
            eff = ListFormat(eff)
        self._raise(self.def_use.find_def_from_site(stmt.var, stmt), eff)


def place_floors(
    ast: FuncDef,
    def_use: DefineUseAnalysis,
    by_def: dict[Definition, FormatBound],
    by_expr: dict[Expr, FormatBound],
    ret_fmt: FormatBound,
    base: 'StorageAnalysis',
    is_called: bool = False,
) -> dict[Definition, FormatBound]:
    """See :class:`_PlaceFloors`.  *base* is the unraised storage analysis, used
    only to tell which names the emitter binds by reference.

    The two directions feed each other — a place raises a variable, a raised
    variable raises the container built from it — so iterate.  One round per
    statement suffices: a place's own bound is fixed, so every floor originates
    in the first round and later rounds only carry it along one edge.  The
    corpus settles in one or two.

    Asserts rather than breaking: leaving the loop with work outstanding would
    return a *partial* answer, and an under-raised definition resurfaces as a
    type disagreement the emitter has to refuse.
    """
    v = _PlaceFloors(def_use, by_def, by_expr, ret_fmt, base, is_called)
    v._visit_function(ast, None)          # collects the assignments to revisit
    v.collected = True
    for _ in range(len(v.assigns) + len(v.indexed) + len(v.loops) + 2):
        v.changed = False
        v._visit_function(ast, None)      # places -> the defs reaching them
        v._propagate_up()                 # defs -> containers and aliases
        if not v.changed:
            break
    assert not v.changed, (
        f'place_floors did not settle in '
        f'{len(v.assigns) + len(v.indexed) + len(v.loops) + 2} rounds over '
        f'{len(v.assigns)} assignments: a floor is rising without bound, so '
        f'either a join is not monotone or a cycle is feeding itself'
    )
    return v.floors


def _is_in_place_assign(d: AssignDef) -> bool:
    """Does *d* come from an in-place ``IndexedAssign`` (``xs[i] = e``)?

    The FPy interpreter mutates the underlying list in place
    (``interpret/byte.py:_visit_indexed_assign``).  SSA gives the
    post-mutation name its own AssignDef anyway so value-tracking
    analyses can reason about it, but for the cpp backend the new
    def must share storage with its ``prev`` — no copy, no widening,
    no rename.
    """
    return isinstance(d.site, IndexedAssign)


class StorageInfer:
    """
    Storage-type inference for the cpp emitter.

    Assigns one C++ variable (identifier + storage type) per SSA def,
    coalescing only across phi edges.  See module docstring for the
    full contract.
    """

    @staticmethod
    def infer(
        def_use: DefineUseAnalysis,
        def_to_bound: dict[Definition, FormatBound],
        floors: dict[Definition, FormatBound] | None = None,
    ) -> StorageAnalysis:
        """
        Build a :class:`StorageAnalysis` from def-use info and per-def
        format bounds.

        Args:
            def_use:      def-use analysis result for the function.
            def_to_bound: format bound for each SSA def (typically
                          ``format_info.by_def``).
            floors:       per-def lower bounds from the places each def
                          reaches, from :func:`place_floors`.  A storage
                          decision, not a format one: the bounds themselves
                          stay exactly as the analysis reported them.

        Returns:
            A :class:`StorageAnalysis` carrying the per-def C++ name and
            per-class storage.

        Raises:
            StorageSelectionError: if no ladder entry covers the
            aggregated bound of some phi class.
        """
        defs = def_use.defs

        # ---- 1. union-find over coalescing edges ----
        # Two kinds of edges force defs into the same storage class:
        #   * Phi edges: a phi merge is exactly "both incoming defs
        #     write to the same C++ variable."
        #   * In-place mutation edges: SSA gives ``xs[i] = e`` a
        #     fresh def of ``xs`` so value-tracking analyses can
        #     reason about it, but the FPy interpreter mutates the
        #     existing list in place and C++ does the same — so the
        #     new def is unioned with its ``prev``.  Underlying
        #     vector is the same, so storage cannot widen and the
        #     C++ name must be reused.  This mirrors the comment in
        #     ``reaching_defs`` that physical-property analyses treat
        #     IndexedAssign-sited defs as sharing storage with prev.
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            if isinstance(d, PhiDef):
                uf.union(d, defs[d.lhs])
                uf.union(d, defs[d.rhs])
            elif (
                isinstance(d, AssignDef)
                and d.prev is not None
                and _is_in_place_assign(d)
            ):
                uf.union(d, defs[d.prev])

        def_class: dict[Definition, Definition] = {d: uf.find(d) for d in defs}
        class_members: dict[Definition, list[Definition]] = defaultdict(list)
        for d, c in def_class.items():
            class_members[c].append(d)

        # ---- 2. storage per class ----
        class_storage: dict[Definition, CppType] = {}
        for c, members in class_members.items():
            bounds = [def_to_bound[d] for d in members if d in def_to_bound]
            assert bounds, f'no format bounds for class {c} members={members}'
            # A class is one C++ variable, so a floor on any member raises all
            # of them -- which is what makes the widening reach an alias.
            if floors:
                bounds += [floors[d] for d in members if d in floors]
            try:
                class_storage[c] = aggregate_storage(bounds)
            except StorageSelectionError as e:
                name = members[0].name
                raise StorageSelectionError(
                    f'cannot pick storage for `{name}` (class {c}): {e}'
                ) from e

        # ---- 3. naming per class ----
        # External classes (those containing an arg or free-variable
        # def) are tied to the bare source name and cannot be renamed.
        # Other classes for the same source name pick up a numeric
        # suffix.
        all_source_names = {str(d.name) for d in def_class}
        class_to_name: dict[Definition, str] = {}
        claimed: set[str] = set()
        external_classes: set[Definition] = set()

        # Pass 1: external classes claim the bare source name.  By
        # construction, args and free variables introduce one def each
        # per name, so every external class for a given source name is
        # unique.
        for c, members in class_members.items():
            if _is_external(members):
                src = str(members[0].name)
                class_to_name[c] = src
                claimed.add(src)
                external_classes.add(c)

        # Pass 2: non-external classes.  Process in deterministic
        # min-def-index order so generated names are stable across runs.
        remaining = [c for c in class_members if c not in class_to_name]
        remaining.sort(
            key=lambda c: min(def_use.def_to_idx[d] for d in class_members[c])
        )
        for c in remaining:
            src = str(class_members[c][0].name)
            if src not in claimed:
                class_to_name[c] = src
                claimed.add(src)
                continue
            # Pick the first ``src_N`` that's not already claimed and
            # that isn't itself an existing source name in this function.
            i = 1
            while True:
                cand = f'{src}_{i}'
                if cand not in claimed and cand not in all_source_names:
                    break
                i += 1
            class_to_name[c] = cand
            claimed.add(cand)

        def_to_name = {d: class_to_name[c] for d, c in def_class.items()}

        # Decide each non-external class's emission shape.
        #
        # The only case where we *must* hoist is a phi merge that
        # introduces a name fresh in both branches — i.e., a PhiDef
        # with ``is_intro=True``.  In every other case, FPy
        # well-formedness guarantees the lowest-index AssignDef
        # dominates the rest of the class (it's either a single writer,
        # or a pre-loop / pre-if writer that the body then rebinds via
        # phi).  So that AssignDef can declare-on-assign and any other
        # AssignDefs become plain reassignments.
        #
        # When we *do* hoist, we don't go all the way to the function
        # top — we anchor at the outermost responsible ``IfStmt`` and
        # emit the declaration just before it, narrowing the variable's
        # scope to exactly what its writers need.  For nested if/else
        # introductions the outermost is_intro phi is the one with the
        # highest def index, since phis are appended after their
        # branches finish in pre-order traversal.
        hoists_before: dict[Stmt, list[Definition]] = defaultdict(list)
        declare_at_assign: set[AssignDef] = set()
        for c, members in class_members.items():
            if c in external_classes:
                continue
            intro_phis = [d for d in members
                          if isinstance(d, PhiDef) and d.is_intro]
            if intro_phis:
                anchor_phi = max(intro_phis, key=lambda d: def_use.def_to_idx[d])
                hoists_before[anchor_phi.site].append(c)
                continue
            assigns = [d for d in members if isinstance(d, AssignDef)]
            assert assigns, (
                f'non-external class {c} has no AssignDef members '
                f'(members={members})'
            )
            first_assign = min(assigns, key=lambda d: def_use.def_to_idx[d])
            declare_at_assign.add(first_assign)
        # Stable order per anchor for deterministic output.
        for cs in hoists_before.values():
            cs.sort(key=lambda c: class_to_name[c])

        return StorageAnalysis(
            def_class=def_class,
            class_members=dict(class_members),
            class_storage=class_storage,
            def_to_name=def_to_name,
            hoists_before=dict(hoists_before),
            declare_at_assign=declare_at_assign,
        )
