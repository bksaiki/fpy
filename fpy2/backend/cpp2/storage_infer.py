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

from ...ast.fpyast import Argument
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
    The emitter looks up ``def_to_name`` for every use/assign, and
    walks ``decl_classes`` to emit hoisted declarations at the top of
    the function body.

    Attributes:
        def_class:       maps each def to its class id (the canonical
                         representative def of the class).
        class_members:   maps each class id to its member defs.
        class_storage:   the C++ storage chosen for each class.
        def_to_name:     the C++ identifier each def reads/writes through.
        decl_classes:    class ids whose declaration the emitter must
                         hoist (everything except classes anchored to
                         function args or free variables).
    """
    def_class: dict[Definition, Definition]
    class_members: dict[Definition, list[Definition]]
    class_storage: dict[Definition, CppType]
    def_to_name: dict[Definition, str]
    decl_classes: list[Definition]



def _is_external(members: list[Definition]) -> bool:
    for d in members:
        if isinstance(d, AssignDef) and (
            isinstance(d.site, Argument) or d.is_free
        ):
            return True
    return False


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

        # ---- 1. union-find over phi edges ----
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            if isinstance(d, PhiDef):
                uf.union(d, defs[d.lhs])
                uf.union(d, defs[d.rhs])

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

        # Declaration set: everything we emit a hoisted ``T name{};`` for.
        decl_classes = [c for c in class_members if c not in external_classes]
        # Order declarations by their chosen C++ name for stable output.
        decl_classes.sort(key=lambda c: class_to_name[c])

        return StorageAnalysis(
            def_class=def_class,
            class_members=dict(class_members),
            class_storage=class_storage,
            def_to_name=def_to_name,
            decl_classes=decl_classes,
        )
