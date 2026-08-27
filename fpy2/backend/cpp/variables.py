"""
cpp backend: variable materialization.

Storage inference says which definitions share one runtime object and what type
holds it (:mod:`.storage_infer`).  This module answers the questions that follow
for a target with *variables*: what each class is called, where it is declared,
and whether a name binds a reference to storage that already exists.

None of it is storage.  It is here because it is keyed by the same classes, and
it stays in the backend when the assignment itself becomes a generic analysis: a
target without declarations or block scope has no use for any of it.
"""

from collections import defaultdict
from dataclasses import dataclass

from ...analysis import Definition
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.reaching_defs import AssignDef, PhiDef
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
from .storage import CppStorage
from .storage_infer import is_rebound
from .types import CppType


@dataclass
class VariableAnalysis:
    """Where each storage class lives in the emitted source.

    A class declares at its lowest-index writer (``declare_at_assign``), except
    where writers sit in disjoint ``if``/``else`` branches and the name did not
    exist before: no writer dominates, so the declaration hoists just before that
    ``IfStmt`` (``hoists_before``) and every writer reassigns.  External classes
    -- an argument or free variable -- appear in neither; the signature or
    enclosing scope already declares them.

    Attributes:
        def_to_name:        the identifier each def reads/writes through.
        hoists_before:      anchor ``IfStmt`` -> classes to declare before it.
        declare_at_assign:  AssignDefs that declare and assign in one go.
    """

    def_to_name: dict[Definition, str]
    hoists_before: dict[Stmt, list[Definition]]
    declare_at_assign: set[AssignDef]


class VariableAlloc:
    """Names and declaration sites for the classes :class:`StorageInfer` found.

    Naming per class: function-argument and free-variable defs anchor a class to
    the bare source name (the signature already declares it under that name).
    Other classes for the same source name pick up numeric suffixes (``x_1``,
    ``x_2``, ...).
    """

    @staticmethod
    def assign(
        def_use: DefineUseAnalysis, storage: CppStorage,
    ) -> VariableAnalysis:
        """Name and place every class of *storage*."""
        def_class = storage.def_class
        class_members = storage.class_members
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
        return VariableAnalysis(
            def_to_name=def_to_name,
            hoists_before=dict(hoists_before),
            declare_at_assign=declare_at_assign,
        )


def _is_external(members: list[Definition]) -> bool:
    for d in members:
        if isinstance(d, AssignDef) and (
            isinstance(d.site, Argument) or d.is_free
        ):
            return True
    return False


def binds_by_reference(
    storage: CppStorage,
    variables: VariableAnalysis,
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

    A binding to another *definition* additionally requires the two storages to
    agree.  A reference and a shared storage class are the same claim: if they
    disagree the name has the type C++ deduced from the initializer rather than
    the one chosen for it, and every consumer of ``storage_of`` has to remember
    to compensate.  Requiring agreement makes the divergence impossible instead.
    """
    if not storage.is_aggregate(storage.storage_of(d)):
        return False
    if is_rebound(storage.analysis, d):
        return False
    if not _binds_the_whole_value(d):
        return False
    match d.site:
        case Argument() | ForStmt() | ListComp():
            return True
        case Assign(expr=Var() as src):
            src_def = def_use.find_def_from_use(src)
            return (
                d in variables.declare_at_assign
                and not is_rebound(storage.analysis, src_def)
                and storage.storage_of(d) == storage.storage_of(src_def)
            )
        case Assign(expr=ListRef() as ref) if allow_projection:
            root = _root_var(ref)
            return (
                root is not None
                and d in variables.declare_at_assign
                and not is_rebound(storage.analysis, def_use.find_def_from_use(root))
            )
        case _:
            return False


def _binds_the_whole_value(d: Definition) -> bool:
    """Whether *d*'s binding hands it the whole value.

    A name destructured out of a tuple does not qualify however the tuple
    arrived: the emitter reads it with ``std::get``, which copies.  Harmless
    while the component is a handle -- the copy still shares -- but a copy of a
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
