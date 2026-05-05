"""
cpp2 backend: phi-web equivalence classes.

The emitter is free to give every SSA def its own C++ variable.  The
*one* constraint is that defs joined by a phi node must share storage
— a phi merge in source SSA materializes as "both incoming branches
write to the same C++ variable".  This module computes the
phi-equivalence classes (SSA "phi webs") via union-find: two defs are
in the same class iff there's a path of phi edges between them.

Each class then gets:
  - an *aggregated* storage type that subsumes every member's
    inferred bound (only true within a class, not across classes);
  - a unique C++ identifier — the bare source name when possible,
    otherwise the source name with a numeric suffix.

Function-argument and free-variable defs anchor a class to the bare
source name (the C++ signature already declares them under that
name), so any other classes for that source name pick up a suffix.
"""

from collections import defaultdict
from dataclasses import dataclass

from ...ast.fpyast import Argument
from ...analysis import Definition
from ...analysis.format_infer import FormatBound
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.reaching_defs import AssignDef, PhiDef

from .storage import aggregate_storage, StorageSelectionError
from .types import CppType


@dataclass
class PhiWeb:
    """
    Phi-web partition of the SSA defs in a function.

    Attributes:
        def_class:       maps each def to its class id (an opaque int).
        class_members:   maps each class id to its member defs.
        class_storage:   the C++ storage chosen for each class.
        def_to_name:     the C++ identifier each def reads/writes through.
        decl_classes:    class ids whose declaration the emitter must
                         hoist (everything except classes anchored to
                         function args or free variables).
    """
    def_class: dict[Definition, int]
    class_members: dict[int, list[Definition]]
    class_storage: dict[int, CppType]
    def_to_name: dict[Definition, str]
    decl_classes: list[int]


def compute_phi_web(
    def_use: DefineUseAnalysis,
    def_to_bound: dict[Definition, FormatBound],
) -> PhiWeb:
    """
    Build a :class:`PhiWeb` from def-use info and per-def format bounds.

    The result tells the emitter, for each SSA def, which C++ variable
    to read/write through and what type that variable should have.
    """
    defs = def_use.defs

    # ---- 1. union-find over phi edges ----
    parent = list(range(len(defs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, d in enumerate(defs):
        if isinstance(d, PhiDef):
            union(i, d.lhs)
            union(i, d.rhs)

    def_class: dict[Definition, int] = {d: find(i) for i, d in enumerate(defs)}
    class_members: dict[int, list[Definition]] = defaultdict(list)
    for d, c in def_class.items():
        class_members[c].append(d)

    # ---- 2. storage per class ----
    class_storage: dict[int, CppType] = {}
    for c, members in class_members.items():
        bounds = [def_to_bound[d] for d in members if d in def_to_bound]
        assert bounds, f'no format bounds for class {c} members={members}'
        try:
            class_storage[c] = aggregate_storage(bounds)
        except StorageSelectionError as e:
            name = members[0].name
            raise StorageSelectionError(
                f'cannot pick storage for `{name}` (phi-class {c}): {e}'
            ) from e

    # ---- 3. naming per class ----
    # External classes (those containing an arg or free-variable def)
    # are tied to the bare source name and cannot be renamed.  Other
    # classes for the same source name pick up a numeric suffix.
    all_source_names = {str(d.name) for d in def_class}

    def is_external(members: list[Definition]) -> bool:
        for d in members:
            if isinstance(d, AssignDef) and (
                isinstance(d.site, Argument) or d.is_free
            ):
                return True
        return False

    class_to_name: dict[int, str] = {}
    claimed: set[str] = set()
    external_classes: set[int] = set()

    # Pass 1: external classes claim the bare source name.  By
    # construction, args and free variables introduce one def each per
    # name, so every external class for a given source name is unique.
    for c, members in class_members.items():
        if is_external(members):
            src = str(members[0].name)
            class_to_name[c] = src
            claimed.add(src)
            external_classes.add(c)

    # Pass 2: non-external classes.  Process in deterministic
    # min-def-index order so generated names are stable across runs.
    remaining = [c for c in class_members if c not in class_to_name]
    remaining.sort(key=lambda c: min(def_use.def_to_idx[d] for d in class_members[c]))
    for c in remaining:
        src = str(class_members[c][0].name)
        if src not in claimed:
            class_to_name[c] = src
            claimed.add(src)
            continue
        # Pick the first ``src_N`` that's not already claimed and that
        # isn't itself an existing source name in this function.
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
    decl_classes = [
        c for c in class_members if c not in external_classes
    ]
    # Order declarations by their chosen C++ name for stable output.
    decl_classes.sort(key=lambda c: class_to_name[c])

    return PhiWeb(
        def_class=def_class,
        class_members=dict(class_members),
        class_storage=class_storage,
        def_to_name=def_to_name,
        decl_classes=decl_classes,
    )
