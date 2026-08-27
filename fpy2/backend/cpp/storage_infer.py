"""
cpp backend: storage-type inference.

Partitions the SSA definitions into classes -- the defs denoting one runtime
object -- and gives each class one storage type.

The partition is a union-find over ``reaching_defs.same_object_defs`` -- the
defs that denote one runtime object must be one C++ variable.  Anything not
connected that way is free to rename, so a sequential rebind without a phi
merge gets its own variable with its own, possibly narrower, storage.

Storage per class is chosen by aggregating every member's
:class:`FormatBound` through :func:`aggregate_storage`.  Only members
of the same class need to fit in a common type; cross-class storage
is independent.

Naming and declaration placement are :mod:`.variables`.
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
from .storage import StorageSelectionError, aggregate_storage, choose_storage
from .types import CppList, CppTuple, CppType


@dataclass
class StorageAnalysis:
    """Result of :class:`StorageInfer`.

    Each SSA def belongs to a *class* -- the defs denoting one runtime object,
    per ``same_object_defs`` -- and each class gets one identifier and storage.

    Naming and declaration placement are :mod:`.variables`, not this: they are
    what a target with variables does with the classes, and this analysis is
    what a backend-independent one would compute.

    Attributes:
        def_class:      each def's class id (the canonical member).
        class_members:  each class id's member defs.
        class_storage:  the C++ storage chosen per class.
    """
    def_class: dict[Definition, Definition]
    class_members: dict[Definition, list[Definition]]
    class_storage: dict[Definition, CppType]
    expr_bound: dict[Expr, FormatBound]
    def_use: DefineUseAnalysis

    def of_expr(self, e: Expr) -> CppType | None:
        """The storage *e*'s value is held in, or ``None`` where no member of
        the domain covers it.

        Definitions take precedence over bounds.  A ``Var`` reads as its
        *class's* storage rather than its own bound: a class is the join over
        its members, so the bound names a type the value is not actually held
        in.  A ``ListRef`` peels the container's element for the same reason --
        its bound can be narrower than what the container declares.  Everything
        else is its own bound.

        ``None`` rather than an error: a caller adjusting a representation has
        nothing to repair when the format has no member, and one that must have
        an answer asks for the bound directly.
        """
        if isinstance(e, Var):
            return self.storage_of(self.def_use.find_def_from_use(e))
        if isinstance(e, ListRef):
            base = self.of_expr(e.value)
            if isinstance(base, CppList):
                return base.elt
        try:
            return choose_storage(self.expr_bound.get(e))
        except StorageSelectionError:
            return None

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

    ``xs[i] = e`` is not a rebind — **E-Update** writes *through* an element
    cell, so the name still denotes the same list and that def stays in the same
    class.  Only an ``Assign`` to the same name is, and *d*'s own defining
    assignment does not count.
    """
    cls = storage.def_class[d]
    return any(
        m is not d and isinstance(m, AssignDef) and isinstance(m.site, Assign)
        for m in storage.class_members[cls]
    )


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
        expr_to_bound: dict[Expr, FormatBound],
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
            # every member, not those that happen to have a bound: one
            # without contributes no constraint, so skipping it can leave the
            # class too narrow to hold its own values
            missing = [d for d in members if d not in def_to_bound]
            assert not missing, (
                f'class {c} has members with no format bound: {missing}'
            )
            bounds = [def_to_bound[d] for d in members]
            try:
                class_storage[c] = aggregate_storage(bounds)
            except StorageSelectionError as e:
                name = members[0].name
                raise StorageSelectionError(
                    f'cannot pick storage for `{name}` (class {c}): {e}'
                ) from e

        return StorageAnalysis(
            def_class=def_class,
            class_members=dict(class_members),
            class_storage=class_storage,
            expr_bound=expr_to_bound,
            def_use=def_use,
        )
