"""
cpp backend: storage-type inference.

Assigns each SSA def to a C++ variable: the variable's identifier and
its storage type.  The emitter consumes the result directly — every
``Var``/``Assign`` is just a lookup.

The partition is a union-find over ``reaching_defs.same_object_defs`` -- the
defs that denote one runtime object must be one C++ variable.  Anything not
connected that way is free to rename, so a sequential rebind without a phi
merge gets its own variable with its own, possibly narrower, storage.

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

from ...analysis import Definition
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.format_infer import FormatBound
from ...analysis.reaching_defs import AssignDef, PhiDef, same_object_defs
from ...ast.fpyast import (
    Argument,
    Assign,
    Expr,
    ForStmt,
    ListComp,
    ListRef,
    NamedId,
    Stmt,
    Var,
)
from ...utils import Unionfind
from .storage import StorageSelectionError, aggregate_storage
from .types import CppList, CppTuple, CppType


@dataclass
class StorageAnalysis:
    """
    Result of :class:`StorageInfer`.

    Each SSA def belongs to a *class* -- the defs that denote one runtime
    object, per ``same_object_defs`` -- and each class gets one C++ identifier
    and storage type.  The class id is its canonical member.  Two emission
    shapes:

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
        def_class:          each def's class id (the canonical member).
        class_members:      each class id's member defs.
        class_storage:      the C++ storage chosen per class.
        def_to_name:        the identifier each def reads/writes through.
        hoists_before:      anchor ``IfStmt`` -> classes to declare before it.
        declare_at_assign:  AssignDefs that declare and assign in one go.
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
    """Whether the emitter binds *d*'s name as a reference to storage that already
    exists, rather than giving it a place of its own.

    One definition for a question two modules ask -- the emitter, choosing
    reference or copy, and :mod:`.unbox`, deciding whether a name is a second
    *place*.  They must agree: discounting a name the emitter then copies is a
    miscompilation.

    *allow_projection* enables ``row = xss[i]``, valid only where nothing
    replaces that slot -- the caller supplies that fact from the alias analysis.
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
    ) -> StorageAnalysis:
        """Build a :class:`StorageAnalysis` from def-use info and per-def bounds.

        Raises :class:`StorageSelectionError` when no ladder entry covers some
        class's aggregated bound.
        """
        defs = def_use.defs

        # ---- 1. union-find over coalescing edges ----
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            for i in same_object_defs(d):
                uf.union(d, defs[i])

        def_class: dict[Definition, Definition] = {d: uf.find(d) for d in defs}
        class_members: dict[Definition, list[Definition]] = defaultdict(list)
        for d, c in def_class.items():
            class_members[c].append(d)

        # ---- 2. storage per class ----
        class_storage: dict[Definition, CppType] = {}
        for c, members in class_members.items():
            bounds = [def_to_bound[d] for d in members if d in def_to_bound]
            assert bounds, f'no format bounds for class {c} members={members}'
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

        # Hoisting is required only for a phi that introduces a name fresh in
        # both branches (`is_intro`); otherwise FPy well-formedness guarantees
        # the lowest-index AssignDef dominates the class, so it declares on
        # assign and the rest reassign.
        #
        # A hoist anchors at the outermost responsible IfStmt rather than the
        # function top, keeping the variable's scope to what its writers need.
        # That phi is the one with the highest def index, since phis are
        # appended after their branches finish in pre-order.
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
