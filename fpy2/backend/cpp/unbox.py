"""
cpp backend: which lists may drop the handle.

A list compiles to ``fpy::list<T>`` — a shared handle — because FPy lists alias.
Where :mod:`fpy2.analysis.alias` proves nothing else can observe a list, the
indirection buys nothing and it can be a plain ``std::vector<T>`` instead: no
allocation, no refcount, and a type a native caller already holds.

The decision is per **alias region** — the analysis's own equivalence class —
because that is the only unit at which one answer satisfies everything that
reads it.

One answer per storage class
----------------------------

:mod:`.storage_infer` gives one C++ variable per class, coalescing on phi and
in-place-mutation edges; :func:`~fpy2.analysis.alias.Alias` unions on exactly
those two.  So a class maps to a single region per level, and :func:`_regions`
asserts it rather than taking a conjunction that would silently repair a
violation.

Callers must agree
------------------

A signature's representation is part of the contract, so a function that compiled
code calls keeps its handles on *both* sides of it.  Measured over the corpus this
costs 5 of the 50 functions with a list parameter — the kernels worth unboxing are
entry points a *native* caller invokes, and a native caller is not bound by this.

Refusal means keeping the handle, never failing a compile; reasons are recorded in
:attr:`UnboxAnalysis.boxed_because`.  A list inside a tuple is always refused:
:func:`_read` walks only the ``CppList`` spine and cannot reach one.
"""

from dataclasses import dataclass, field

from ...analysis import Definition
from ...analysis.alias import AliasAnalysis, Region
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.reaching_defs import AssignDef
from ...ast.fpyast import (
    Argument,
    Assign,
    Expr,
    ForStmt,
    FuncDef,
    IndexedAssign,
    ListComp,
    NamedId,
    ReturnStmt,
    Var,
)
from ...ast.visitor import DefaultVisitor
from .storage_infer import (
    StorageAnalysis,
    binds_by_reference,
    is_rebound,
)
from .types import CppList, CppType


@dataclass
class UnboxAnalysis:
    """Result of :class:`Unbox`.

    ``storage`` gives the representation-annotated type for every list storage
    class; :meth:`annotate` gives the same for an arbitrary expression, reading
    the same per-region table so the two cannot disagree.
    """

    alias: AliasAnalysis
    boxed: dict[Region, bool] = field(default_factory=dict)
    storage: dict[Definition, CppType] = field(default_factory=dict)
    ret_regions: list[set[Region]] = field(default_factory=list)
    at_boundary: set[Region] = field(default_factory=set)
    boxed_because: dict[tuple[Definition, int], str] = field(
        default_factory=dict,
    )

    def annotate(self, e: Expr, ty: CppType) -> CppType:
        """*ty* with each list level's representation as decided for *e*.

        A level with no region — nothing the analysis tracked — keeps its
        handle.
        """
        def at(depth: int) -> set[Region]:
            r = self.alias.region_of_expr(e, depth)
            return set() if r is None else {r}

        return self._stamp(ty, at, 0)

    def annotate_return(self, ty: CppType) -> CppType:
        """*ty* with the representation every ``return`` in the function agrees
        on — the return type is one more place that admits a single answer."""
        def at(depth: int) -> set[Region]:
            if depth < len(self.ret_regions):
                return self.ret_regions[depth]
            return set()

        return self._stamp(ty, at, 0)

    def _stamp(self, ty: CppType, regions_at, depth: int) -> CppType:
        if not isinstance(ty, CppList):
            return ty
        elt = self._stamp(ty.elt, regions_at, depth + 1)
        regions = regions_at(depth)
        boxed = not regions or any(self.boxed.get(r, True) for r in regions)
        return CppList(elt, boxed=boxed)


class Unbox:
    """Decide the representation of every list."""

    @staticmethod
    def decide(
        ast: FuncDef,
        storage: StorageAnalysis,
        alias: AliasAnalysis,
        def_use: DefineUseAnalysis,
        *,
        is_called: bool = False,
    ) -> UnboxAnalysis:
        """Which lists may drop the handle.

        Args:
            ast: the function, for its ``return`` statements.
            storage: the storage classes to decide.
            alias: what may refer to what.
            def_use: to tell a binding that copies from one that references.
            is_called: whether compiled code calls this function.  If so its
                parameters keep their handles, since a caller passing a list has
                marked it shared outward and will hold a handle.
        """
        out = UnboxAnalysis(alias)

        # 1. each region on its own evidence
        for site in alias.sites:
            region = alias.region_of_site(site)
            if region is None:
                continue
            escapes = alias.escapes(site) and not alias.transfers_ownership(site)
            shared = escapes or _shares_storage(region, alias, storage, def_use)
            out.boxed[region] = out.boxed.get(region, False) or shared

        classes = [
            (cls, _regions(cls, storage, alias))
            for cls, ty in storage.class_storage.items()
            if isinstance(ty, CppList)
        ]
        out.ret_regions = _return_regions(ast, alias)

        # 2. A list inside a tuple keeps its handle: `_read` walks only the
        #     `CppList` spine, and undecided is not the same as boxed -- an
        #     expression of bare list type would still be stamped `std::vector`.
        for region in alias.regions_in_a_tuple():
            out.at_boundary.add(region)

        # 3. Both sides of a compiled-to-compiled boundary keep their handles,
        #    because the other side of it does.  For a callee that is its whole
        #    signature; for a caller it is what a call hands back.
        for site in alias.sites:
            if site.kind == 'call':
                r = alias.region_of_site(site)
                if r is not None:
                    out.at_boundary.add(r)
        if is_called:
            for regions in out.ret_regions:
                out.at_boundary |= regions
            for cls, per_depth in classes:
                if _has_parameter(storage.class_members[cls]):
                    out.at_boundary |= {r for r in per_depth if r is not None}
        for r in out.at_boundary:
            out.boxed[r] = True

        # 4. A function has one return type but may have several ``return``
        #    statements, and unlike a storage class nothing unifies their
        #    regions.  So this group really can hold more than one, and the
        #    conjunction has to be written back or `annotate_return` and
        #    `annotate` disagree: `if c: return xs else: return [y, y]` would
        #    declare `fpy::list` and hand back a `std::vector`.
        changed = True
        while changed:
            changed = False
            for regions in out.ret_regions:
                if len(regions) < 2:
                    continue
                if any(out.boxed.get(r, True) for r in regions):
                    for r in regions:
                        if not out.boxed.get(r, True):
                            out.boxed[r] = True
                            changed = True

        # 5. read the decision back out per class
        for cls, per_depth in classes:
            ty = storage.class_storage[cls]
            out.storage[cls] = _read(ty, per_depth, cls, out, 0)
        return out


def _regions(
    cls: Definition, storage: StorageAnalysis, alias: AliasAnalysis,
) -> list[Region | None]:
    """The alias region each level of *cls*'s storage holds, by depth.

    At most one per level, and the assertion is the point: ``storage_infer``
    coalesces on phi and in-place-mutation edges, and
    ``alias._merge_redefinitions`` unions on exactly those two, so a class
    cannot span regions that could answer differently.  Taking a conjunction
    over several would silently repair a violation of that; this reports it.
    """
    ty = storage.class_storage[cls]
    per_depth: list[Region | None] = []
    depth = 0
    while isinstance(ty, CppList):
        found = {
            r for d in storage.class_members[cls]
            if (r := alias.region_of(d, depth)) is not None
        }
        assert len(found) <= 1, (
            f'storage class `{cls.name}` spans {len(found)} alias regions at '
            f'depth {depth}: the backend has a coalescing edge the alias '
            f'analysis does not mirror'
        )
        per_depth.append(found.pop() if found else None)
        ty, depth = ty.elt, depth + 1
    return per_depth


def _read(
    ty: CppType,
    per_depth: list[Region | None],
    cls: Definition,
    out: UnboxAnalysis,
    depth: int,
) -> CppType:
    if not isinstance(ty, CppList):
        return ty
    elt = _read(ty.elt, per_depth, cls, out, depth + 1)
    region = per_depth[depth]
    if region is None:
        out.boxed_because[(cls, depth)] = 'no alias information'
    elif region in out.at_boundary:
        out.boxed_because[(cls, depth)] = 'reached across a boundary'
    elif out.boxed.get(region, True):
        out.boxed_because[(cls, depth)] = 'shared'
    else:
        return CppList(elt, boxed=False)
    return CppList(elt, boxed=True)


def _has_parameter(members: list[Definition]) -> bool:
    """Whether this storage class is a function parameter.

    Asks every member: which def represents a class is an artifact of union
    order, so the argument-sited one need not be it.
    """
    return any(
        isinstance(d, AssignDef) and isinstance(d.site, Argument)
        for d in members
    )


def _return_regions(ast: FuncDef, alias: AliasAnalysis) -> list[set[Region]]:
    """The regions every ``return`` in *ast* may hand back, by depth."""
    returned: list[Expr] = []

    class _Returns(DefaultVisitor):
        def _visit_return(self, stmt: ReturnStmt, ctx):
            returned.append(stmt.expr)
            super()._visit_return(stmt, ctx)

    _Returns()._visit_function(ast, None)

    out: list[set[Region]] = []
    depth = 0
    while True:
        found = {
            r for e in returned
            if (r := alias.region_of_expr(e, depth)) is not None
        }
        if not found:
            return out
        out.append(found)
        depth += 1


def _shares_storage(
    region: Region,
    alias: AliasAnalysis,
    storage: StorageAnalysis,
    def_use: DefineUseAnalysis,
) -> bool:
    """Whether more than one place in this function holds *region* separately.

    Not ``AliasAnalysis.is_shared``, which counts every name.  What decides a
    representation is whether a second name gets its own *storage*: a name the
    emitter binds by reference copies nothing.  ``for row in xss`` and the
    ``_src = xs`` aliases ``ZipElim`` introduces are both of that kind, and both
    are common enough that counting them would box most idiomatic programs.

    Mirrors the emitter's binding rules exactly rather than approximating them:
    discounting a name the emitter then *copies* would be a miscompilation.
    """
    for d in alias.defs_in(region):
        if (
            isinstance(d, AssignDef)
            and isinstance(d.site, Argument)
            and is_rebound(storage, d)
        ):
            # `_arg_decl` passes a *rebound* parameter by value.  Boxed, the
            # copy is of the handle and writes still reach the caller; unboxed
            # it copies the sequence and they do not.
            return True

    by_name: dict[NamedId, list[AssignDef]] = {}
    for d in alias.defs_in(region):
        # Only definitions that *bind* a name count.  A phi and an
        # `xs[i] = e` are redefinitions of one already there: SSA gives them
        # their own def, but neither introduces a place.
        if isinstance(d, AssignDef) and not isinstance(d.site, IndexedAssign):
            by_name.setdefault(d.name, []).append(d)
    slots = alias.referrers(region) - len(by_name)
    owned_separately = 0
    for ds in by_name.values():
        if not all(_binds_by_reference(d, storage, def_use) for d in ds):
            owned_separately += 1
    return slots + owned_separately > 1


def _binds_by_reference(
    d: AssignDef, storage: StorageAnalysis, def_use: DefineUseAnalysis,
) -> bool:
    """Whether *d* is a reference to storage that exists already, *and* that
    storage is inside this function.

    The one deliberate difference from the emitter: a parameter binds by
    reference too, but to the **caller's** storage, which is a place of its own.
    Discounting it would make ``zss = [xs]`` look unshared.  Spelled as one
    extra term so the divergence stays visible rather than becoming a second
    copy of the rule to keep in sync.
    """
    return (
        binds_by_reference(storage, def_use, d)
        and not isinstance(d.site, Argument)
    )
