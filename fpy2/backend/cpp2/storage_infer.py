"""
cpp2 backend: storage-type inference.

Assigns each SSA def to a C++ variable: the variable's identifier and
its storage type.  The emitter consumes the result directly — every
``Var``/``Assign`` is just a lookup.

The underlying partition is the SSA "phi web": two defs share a C++
variable iff they are connected by phi edges (computed via union-find
over the phi nodes).  Anything else is free to rename, so a sequential
rebind of a name without a phi merge gets its *own* variable with its
*own* (possibly narrower) storage type.

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

from ...ast.fpyast import Argument, IndexedAssign, Stmt
from ...analysis import Definition
from ...analysis.format_infer import FormatBound
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.reaching_defs import AssignDef, PhiDef
from ...utils import Unionfind

from .storage import aggregate_storage, StorageSelectionError
from .types import CppType


@dataclass
class StorageAnalysis:
    """
    Result of :class:`StorageInfer`.

    Each SSA def is assigned to a *class* (the phi-web equivalence
    class), and each class is assigned a single C++ identifier and
    storage type.  The class id is the union-find representative —
    the canonical :class:`Definition` standing in for the whole class.
    Storage classes split into two emission shapes:

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



def _is_external(members: list[Definition]) -> bool:
    for d in members:
        if isinstance(d, AssignDef) and (
            isinstance(d.site, Argument) or d.is_free
        ):
            return True
    return False


def _is_in_place_assign(d: AssignDef) -> bool:
    """Does *d* come from an in-place ``IndexedAssign`` (``xs[i] = e``)?

    The FPy interpreter mutates the underlying list in place
    (``interpret/byte.py:_visit_indexed_assign``).  SSA gives the
    post-mutation name its own AssignDef anyway so value-tracking
    analyses can reason about it, but for the cpp2 backend the new
    def must share storage with its ``prev`` — no copy, no widening,
    no rename.
    """
    return isinstance(d.site, IndexedAssign)


class StorageInfer:
    """
    Storage-type inference for the cpp2 emitter.

    Assigns one C++ variable (identifier + storage type) per SSA def,
    coalescing only across phi edges.  See module docstring for the
    full contract.
    """

    @staticmethod
    def infer(
        def_use: DefineUseAnalysis,
        def_to_bound: dict[Definition, FormatBound],
    ) -> StorageAnalysis:
        """
        Build a :class:`StorageAnalysis` from def-use info and per-def
        format bounds.

        Args:
            def_use:      def-use analysis result for the function.
            def_to_bound: format bound for each SSA def (typically
                          ``format_info.by_def``).

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
        #   * In-place mutation edges: ``xs[i] = e`` is in-place per
        #     the FPy interpreter, but ``FuncUpdate`` has rewritten it
        #     to ``xs = ListSet(xs, [i], e)`` and the SSA pass
        #     introduced a fresh def for ``xs``.  Detect that
        #     canonical shape and union with ``prev`` — the underlying
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
