"""
cpp backend: which lists may drop the handle.

A list compiles to ``fpy::list<T>`` -- a shared handle -- because FPy lists
alias.  Where :mod:`fpy2.analysis.alias` proves nothing else can observe one, the
indirection buys nothing and it becomes a plain ``std::vector<T>``: no
allocation, no refcount, and a type a native caller already holds.

Decided per **alias region**, the only unit at which one answer satisfies every
reader.  That works because :mod:`.storage_infer` coalesces on exactly the edges
:func:`~fpy2.analysis.alias.Alias` unions on, so a storage class maps to one
region per level -- :func:`_regions` asserts this rather than taking a
conjunction that would silently paper over a violation.

A signature's representation is part of its contract, so a function that
compiled code calls keeps its handles on *both* sides.  Measured: 5 of the 50
corpus functions with a list parameter.  The kernels worth unboxing are entry
points a *native* caller invokes, and a native caller is not bound by this.

Refusing means keeping the handle, never failing a compile; reasons land in
:attr:`UnboxAnalysis.boxed_because`.

:func:`_stamp` is the *only* place any of this is decided -- for an expression,
a return type, and a storage class's declaration alike.  A second traversal for
the declaration once skipped tuples, so ``t = [y, y], 1.0; return t`` declared
``std::tuple<fpy::list<T>, ...>`` and returned ``std::tuple<std::vector<T>,
...>``, which does not compile.  Keep it one traversal.
"""

from dataclasses import dataclass, field

from ...analysis import Definition
from ...analysis.alias import AliasAnalysis, Region
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.escape import EscapeSummary
from ...analysis.format_infer import FormatBound
from ...analysis.reaching_defs import AssignDef
from ...ast.fpyast import (
    Argument,
    Assign,
    Call,
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
from ...function import Function
from .storage import choose_storage
from .storage_infer import (
    StorageAnalysis,
    binds_by_reference,
    is_rebound,
)
from .types import CppList, CppTuple, CppType


@dataclass(frozen=True)
class ParamAbi:
    """What a caller must match about one parameter of a compiled callee."""

    ty: CppType
    written: bool
    """Whether the callee stores into its elements, which makes a ``const``
    reference at the *caller* reject the call."""


@dataclass(frozen=True)
class CalleeAbi:
    """A compiled function's signature, as its callers must see it."""

    params: list[ParamAbi]
    ret: CppType


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
    written: set[Region] = field(default_factory=set)
    at_boundary: set[Region] = field(default_factory=set)
    boxed_because: dict[tuple[Definition, int], str] = field(
        default_factory=dict,
    )

    def may_reference_projection(self, d: Definition) -> bool:
        """Whether ``row = xss[i]`` may bind a reference rather than copy.

        Whenever the projection has a region at all.  A C++ reference follows the
        slot, and so does FPy: a store *overwrites* the slot rather than putting a
        different list in it, so there is no way to detach one from its contents.
        The guard this used to need went away with that rule.
        """
        return self.alias.region_of(d) is not None

    def writes_through(self, region: Region | None, ty: CppType) -> bool:
        """Whether a ``const`` reference here would reject a write FPy allows.

        ``const fpy::list<T>&`` does not: ``const`` qualifies the handle, and
        ``xs[i] = e`` through one is FPy parameter semantics.  An unboxed list has no
        such indirection, so ``const`` reaches its elements -- and through a value
        container, so a write to a row needs the whole thing non-const.  A boxed
        level stops that.
        """
        depth = 0
        while isinstance(ty, CppList) and not ty.boxed:
            if region is None:
                return True
            if region in self.written:
                return True
            region = self.alias.region_at(region)
            ty, depth = ty.elt, depth + 1
        return False

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

    def _stamp(
        self, ty: CppType, regions_at, depth: int,
        cls: Definition | None = None,
    ) -> CppType:
        """*ty* with a representation per list level, descending into tuple
        fields too.  The single place a representation is chosen -- see the
        module docstring for why that has to stay true.

        *cls* records why a level kept its handle, and only down the list
        spine: ``boxed_because`` is keyed by depth, so two list fields of one
        tuple would collide, and nothing consumes a reason for a tuple field.
        """
        if isinstance(ty, CppTuple):
            return CppTuple(
                self._stamp(
                    e, lambda d, i=i: _fields(self.alias, regions_at(d), i),
                    depth,
                )
                for i, e in enumerate(ty.elts)
            )
        if not isinstance(ty, CppList):
            return ty
        elt = self._stamp(ty.elt, regions_at, depth + 1, cls)
        regions = regions_at(depth)
        boxed = not regions or any(self.boxed.get(r, True) for r in regions)
        if boxed and cls is not None:
            self.boxed_because[(cls, depth)] = self._reason(regions)
        return CppList(elt, boxed=boxed)

    def _reason(self, regions: set[Region]) -> str:
        """Why a level kept its handle."""
        if not regions:
            return 'no alias information'
        if any(r in self.at_boundary for r in regions):
            return 'reached across a boundary'
        return 'shared'


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
        summary: 'EscapeSummary | None' = None,
        callees: 'dict[FuncDef, CalleeAbi] | None' = None,
    ) -> UnboxAnalysis:
        """Which lists may drop the handle.

        A parameter the function does not retain (per *summary*) can drop its handle
        even when called, because the caller reads that same summary and stops
        treating the argument as shared; absent means it retains everything.
        *callees* carries the emitted signatures, so an argument and a result get
        the representation each declares.
        """
        out = UnboxAnalysis(alias)
        scan = _Scan(alias, def_use, callees or {})
        scan._visit_function(ast, None)
        out.ret_regions = scan.returned
        out.written = scan.written
        out.at_boundary = scan.at_boundary

        # 1. each region on its own evidence
        for site in alias.sites:
            region = alias.region_of_site(site)
            if region is None:
                continue
            escapes = alias.escapes(site) and not alias.transfers_ownership(site)
            shared = escapes or _shares_storage(
                region, alias, storage, def_use,
            )
            out.boxed[region] = out.boxed.get(region, False) or shared

        # 2. a called function keeps a parameter's handle only where it
        #    *retains* it; its callers read the same summary, so both ends
        #    reach the same answer.
        # A tuple is here too: it may hold a list, and that list's
        # representation has to be decided or the declaration and the return
        # type disagree about it.
        classes = [
            (cls, _regions(cls, storage, alias))
            for cls, ty in storage.class_storage.items()
            if isinstance(ty, CppList | CppTuple)
        ]
        if is_called:
            for cls, per_depth in classes:
                i = _parameter_index(ast, storage.class_members[cls])
                if i is not None and (summary is None or summary.retains(i)):
                    out.at_boundary |= {r for r in per_depth if r is not None}
        for r in out.at_boundary:
            out.boxed[r] = True

        # 3. A function has one return type but may have several ``return``
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

        # 5. read the decision back out per class -- through the same
        #    traversal `annotate` and `annotate_return` use, so a variable's
        #    declaration cannot disagree with what is handed through it.
        for cls, per_depth in classes:
            ty = storage.class_storage[cls]
            out.storage[cls] = out._stamp(ty, _at_depth(per_depth), 0, cls)
        return out


def _regions(
    cls: Definition, storage: StorageAnalysis, alias: AliasAnalysis,
) -> list[Region | None]:
    """The alias region each level of *cls*'s storage holds, by depth.

    At most one per level, and the assertion is the point: ``storage_infer`` and
    ``alias`` coalesce on the same edges, so a class cannot span regions that
    would answer differently.  A conjunction over several would silently repair
    that violation; this reports it.

    A tuple contributes its region and stops -- :func:`_stamp` descends its
    fields with :func:`_fields`, which needs that region, not a per-depth entry.
    """
    ty = storage.class_storage[cls]
    per_depth: list[Region | None] = []
    depth = 0
    while isinstance(ty, CppList | CppTuple):
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
        if isinstance(ty, CppTuple):
            break
        ty, depth = ty.elt, depth + 1
    return per_depth


def _fields(alias: AliasAnalysis, regions: set[Region], i: int) -> set[Region]:
    """Field *i* of each of *regions*."""
    return {
        f for r in regions if (f := alias.region_field(r, i)) is not None
    }


def _at_depth(per_depth: list[Region | None]):
    """A ``regions_at`` callback over a storage class's per-depth regions.

    :func:`_stamp`'s other callers key off an expression or the return group;
    this is the same interface over a class, so all three share one traversal.
    """
    def at(depth: int) -> set[Region]:
        r = per_depth[depth] if depth < len(per_depth) else None
        return set() if r is None else {r}
    return at


def _parameter_index(ast: FuncDef, members: list[Definition]) -> int | None:
    """Which parameter this storage class is, if any.

    Asks every member: which def represents a class is an artifact of union
    order, so the argument-sited one need not be it.
    """
    for d in members:
        if isinstance(d, AssignDef) and isinstance(d.site, Argument):
            for i, arg in enumerate(ast.args):
                if arg is d.site:
                    return i
    return None


class _Scan(DefaultVisitor):
    """One walk collecting the three facts ``decide`` reads off the syntax.

    ``written`` regions stored into here or in a callee, which a ``const``
    reference would reject.  ``at_boundary`` regions crossing a call that
    declared a handle; an unknown callee counts as declaring one everywhere.
    ``returned`` what each ``return`` hands back, by depth.
    """

    def __init__(
        self,
        alias: AliasAnalysis,
        def_use: DefineUseAnalysis,
        callees: 'dict[FuncDef, CalleeAbi]',
    ):
        self.alias = alias
        self.def_use = def_use
        self.callees = callees
        self.written: set[Region] = set()
        self.at_boundary: set[Region] = set()
        self.returned: list[set[Region]] = []
        for d in alias.all_defs():
            if isinstance(d, AssignDef) and isinstance(d.site, IndexedAssign):
                if (r := alias.region_of(d)) is not None:
                    self.written.add(r)

    def _levels(self, e: Expr) -> set[Region]:
        out, depth = set(), 0
        while (r := self.alias.region_of_expr(e, depth)) is not None:
            out.add(r)
            depth += 1
        return out

    def _visit_call(self, e: Call, ctx):
        abi = self.callees.get(e.fn.ast) if isinstance(e.fn, Function) else None
        for i, a in enumerate(e.args):
            param = abi.params[i] if abi and i < len(abi.params) else None
            if param is None or not _unboxed(param.ty):
                self.at_boundary |= self._levels(a)
            if param is not None and param.written:
                if (r := self.alias.region_of_expr(a)) is not None:
                    self.written.add(r)
        if abi is None or not _unboxed(abi.ret):
            self.at_boundary |= self._levels(e)
        super()._visit_call(e, ctx)

    def _visit_return(self, stmt: ReturnStmt, ctx):
        depth = 0
        while (r := self.alias.region_of_expr(stmt.expr, depth)) is not None:
            while len(self.returned) <= depth:
                self.returned.append(set())
            self.returned[depth].add(r)
            depth += 1
        super()._visit_return(stmt, ctx)


def _unboxed(ty: CppType | None) -> bool:
    return isinstance(ty, CppList) and not ty.boxed


def _shares_storage(
    region: Region,
    alias: AliasAnalysis,
    storage: StorageAnalysis,
    def_use: DefineUseAnalysis,
) -> bool:
    """Whether more than one place holds *region* separately.

    Not ``is_shared``, which counts every name: what decides a representation is
    whether a second name gets its own *storage*.  ``for row in xss`` and
    ``ZipElim``'s ``_src = xs`` do not, and are common enough that counting them
    would box most idiomatic programs.  Mirrors the binding rules exactly --
    discounting a name the emitter then copies would be a miscompilation.
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
        if not all(
            _binds_by_reference(d, storage, def_use, alias) for d in ds
        ):
            owned_separately += 1
    return slots + owned_separately > 1


def _binds_by_reference(
    d: AssignDef,
    storage: StorageAnalysis,
    def_use: DefineUseAnalysis,
    alias: AliasAnalysis,
) -> bool:
    """Whether *d* references storage that already exists *inside this function*.

    The one deliberate difference from the emitter: a parameter also binds by
    reference, but to the caller's storage, a place of its own -- discounting it
    would make ``zss = [xs]`` look unshared.  One extra term, so the divergence
    stays visible instead of becoming a second copy of the rule.
    """
    return (
        binds_by_reference(
            storage, def_use, d,
            allow_projection=alias.region_of(d) is not None,
        )
        and not isinstance(d.site, Argument)
    )


def return_storage(
    ret_fmt: FormatBound, unbox: UnboxAnalysis | None,
) -> CppType:
    """The storage a function's return value takes.

    ``ret_fmt`` joins every ``ReturnStmt``, so a multiple-return program gets a
    class wide enough for every path.  A ``None`` bound is not a missing return
    but format inference's convention for a non-numeric result.
    """
    ty = choose_storage(ret_fmt)
    return ty if unbox is None else unbox.annotate_return(ty)
