"""
cpp backend: which lists may drop the handle.

An FPy list is a core list of *references* -- one cell per element -- so two
names can hold the same cells and a write through either is visible to both.
A shared handle -- ``std::shared_ptr<std::vector<T>>`` -- is how that identity
survives compilation.  Where :mod:`fpy2.analysis.alias` proves nothing else can
observe the cells, the indirection buys nothing and the list becomes a plain
``std::vector<T>``: no allocation, no refcount, and a type a native caller
already holds.

Decided per **alias region**, the only unit at which one answer satisfies every
reader.  That works because :class:`~fpy2.analysis.StorageInfer` coalesces on
exactly the edges :func:`~fpy2.analysis.alias.Alias` unions on, so a storage
class maps to one region per level -- :func:`_regions` asserts this rather than
taking a conjunction that would silently paper over a violation.

A dropped handle can go one step further: where
:class:`~fpy2.analysis.array_size.ArraySizeInfer` proves every value a region
ever holds has one length, the value becomes ``std::array<T, K>``
(:func:`_region_sizes` builds the table, poisoning on any doubt).  Sizes are
region-keyed and stamped in the same traversal as boxedness, so the two axes
cannot disagree.  They cross call edges by *specialization*
(:class:`~fpy2.transform.specialize.Specialize` keys specs on argument
lengths), not by conversion, so both ends of a call agree by construction.

A signature's representation is part of its contract, so a function that
compiled code calls keeps its handles on *both* sides.  Measured: 5 of the 50
corpus functions with a list parameter.  The kernels worth unboxing are entry
points a *native* caller invokes, and a native caller is not bound by this.

The analysis itself never fails: refusing means keeping the handle, with
reasons in :attr:`UnboxAnalysis.boxed_because`.  Under :class:`UnboxMode.STRICT`
the compiler then turns each retained handle into an error --
:func:`check_strict` for anything a storage class or the return type names,
:meth:`UnboxAnalysis.annotate` for expression temporaries -- rather than let a
``std::shared_ptr`` into the output.

:func:`_stamp` is the *only* place any of this is decided -- for an expression,
a return type, and a storage class's declaration alike.  A second traversal for
the declaration once skipped tuples, so ``t = [y, y], 1.0; return t`` declared
a boxed tuple field where the return said ``std::tuple<std::vector<T>,
...>``, which does not compile.  Keep it one traversal.
"""

import enum
from dataclasses import dataclass, field

from ...analysis import Definition
from ...analysis.alias import AliasAnalysis, Region
from ...analysis.array_size import (
    ArraySizeAnalysis,
    ArraySizeBound,
    ListSize,
    TupleSize,
    concrete_size,
)
from ...analysis.define_use import DefineUseAnalysis
from ...analysis.escape import EscapeSummary
from ...analysis.format_infer import FormatBound
from ...analysis.reaching_defs import AssignDef
from ...ast.fpyast import (
    Argument,
    Call,
    Empty,
    Enumerate,
    Expr,
    FuncDef,
    IndexedAssign,
    ListComp,
    ListExpr,
    ListSlice,
    NamedId,
    Range1,
    Range2,
    Range3,
    Var,
    Zip,
)
from ...ast.visitor import DefaultVisitor
from ...function import Function
from ...utils import enum_repr
from .storage import CppStorage, choose_storage
from .types import CppList, CppTuple, CppType
from .variables import VariableAnalysis, binds_by_reference


@enum_repr
class UnboxMode(enum.Enum):
    """How aggressively the compiler drops the ``std::shared_ptr`` handle.

    - ``NEVER``: every list keeps its handle -- correct, but slower at a
      native boundary.  A fixed length refines the *value* representation, so
      ``NEVER`` also means no ``std::array`` anywhere, whatever the compiler's
      ``arrays`` flag says.
    - ``ALLOW``: drop the handle where the alias analysis proves nothing
      observes the difference; keep it everywhere else.
    - ``STRICT``: like ``ALLOW``, but a list that must keep its handle is a
      compile error rather than a ``std::shared_ptr`` in the output.
    """

    NEVER = 0
    ALLOW = 1
    STRICT = 2


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
    slot_replaced: set[Region] = field(default_factory=set)
    """Element regions some ``xss[i] = <list>`` puts a *different* list into.

    ``row = xss[i]`` is **E-Index** then **E-Deref**: it reads through the cell
    *once* and binds what was in it.  A C++ reference re-reads the slot on every
    use, so the two diverge exactly where an **E-Update** replaces the cell's
    contents.  ``_regression_replaced_slot`` is that program.
    """
    at_boundary: set[Region] = field(default_factory=set)
    boxed_because: dict[tuple[Definition, int], str] = field(
        default_factory=dict,
    )
    strict: bool = False
    """Set by the compiler under :class:`UnboxMode.STRICT`: :meth:`annotate`
    then refuses a boxed level instead of returning it, catching the
    expression temporaries :func:`check_strict` has no storage class for."""
    sizes: dict[Region, int | None] = field(default_factory=dict)
    """One proven length per region, from :func:`_region_sizes`.

    ``int`` means every value the region ever holds has that length, so its
    unboxed storage may be ``std::array``; ``None`` or absent means unknown.
    Region-keyed like ``boxed``, and sound for the same reason: FPy lists never
    change length after construction, so a region whose every contributor
    agrees on one length genuinely has one.
    """

    def may_reference_projection(self, d: Definition) -> bool:
        """Whether ``row = xss[i]`` may bind a reference rather than copy.

        Only when nothing replaces that slot: a reference re-reads it, where
        **E-Deref** read it once at the binding.
        """
        region = self.alias.region_of(d)
        return region is not None and region not in self.slot_replaced

    def writes_through(self, region: Region | None, ty: CppType) -> bool:
        """Whether a ``const`` reference here would reject a write FPy allows.

        **E-App** keeps no store of its own, so a callee's ``xs[i] = e`` writes
        the caller's cells -- and a ``const`` reference to the handle permits
        exactly that, since ``const`` qualifies the handle and not the cells.  An
        unboxed list has no such indirection, so ``const`` reaches its elements
        -- and through a value container, so a write to a row needs the whole
        thing non-const.  A boxed level stops that.
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
        handle.  Under ``strict`` a boxed level raises instead: the handle
        would surface as a ``std::shared_ptr`` the mode promised away.
        """
        def at(depth: int) -> set[Region]:
            r = self.alias.region_of_expr(e, depth)
            return set() if r is None else {r}

        ty = self._stamp(ty, at, 0)
        if self.strict and contains_boxed(ty):
            raise StrictUnboxError(
                'an expression must keep its handle (a shared temporary, or a '
                'list level the alias analysis did not track); use '
                'unbox=CppCompiler.UnboxMode.ALLOW to permit handles'
            )
        return ty

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
        size = None if boxed else self._size_of(regions)
        return CppList(elt, boxed=boxed, size=size)

    def _size_of(self, regions: set[Region]) -> int | None:
        """The one length every region in *regions* agrees on, if any.

        Reverse polarity from boxedness: no region is no evidence, hence no
        size, and several regions (only the return group has them) must be
        unanimous.
        """
        sizes = {self.sizes.get(r) for r in regions}
        return sizes.pop() if len(sizes) == 1 else None

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
        storage: CppStorage,
        variables: VariableAnalysis,
        alias: AliasAnalysis,
        def_use: DefineUseAnalysis,
        *,
        is_called: bool = False,
        summary: 'EscapeSummary | None' = None,
        callees: 'dict[FuncDef, CalleeAbi] | None' = None,
        array_size: 'ArraySizeAnalysis | None' = None,
    ) -> UnboxAnalysis:
        """Which lists may drop the handle.

        A parameter the function does not retain (per *summary*) can drop its handle
        even when called, because the caller reads that same summary and stops
        treating the argument as shared; absent means it retains everything.
        *callees* carries the emitted signatures, so an argument and a result get
        the representation each declares.

        *array_size* enables fixed-length storage: an unboxed level whose region
        has one proven length becomes ``std::array``.  ``None`` -- the
        compiler's ``arrays=False`` -- stamps no sizes at all.
        """
        out = UnboxAnalysis(alias)
        if array_size is not None:
            out.sizes = _region_sizes(alias, array_size)
        scan = _Scan(alias, callees or {})
        scan._visit_function(ast, None)
        out.slot_replaced = alias.slot_replaced
        out.ret_regions = alias.returned_levels
        out.written = scan.written
        out.at_boundary = scan.at_boundary

        # 1. each region on its own evidence
        for site in alias.sites:
            region = alias.region_of_site(site)
            if region is None:
                continue
            escapes = alias.escapes(site) and not alias.transfers_ownership(site)
            shared = escapes or _shares_storage(
                region, alias, storage, variables, def_use,
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
        #    declare a handle and hand back a `std::vector`.
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
    cls: Definition, storage: CppStorage, alias: AliasAnalysis,
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


_ALLOC_EXPRS = (
    ListExpr, ListComp, Empty, ListSlice, Range1, Range2, Range3, Zip,
    Enumerate, Call,
)
"""Expression forms that produce a list of their own.

These seed :func:`_region_sizes` in addition to the defs: a returned literal has
a region but no def, and only its ``by_expr`` bound can size it.  Plain reads
(``Var``, ``ListRef``) are left out -- they only mirror what a def contributed,
and a lossy read-side bound must not poison a region a definition proved.
"""


def _region_sizes(
    alias: AliasAnalysis, array_size: ArraySizeAnalysis,
) -> dict[Region, int | None]:
    """One proven length per region, by meeting every contribution.

    A region's storage must hold every value it is ever bound to, so the meet
    poisons on any doubt: a def with no bound, a level the size analysis
    answered ``None`` for, a symbolic size (a per-run variable, never a length
    ``std::array`` can spell), or two differing lengths all force ``None``.
    Contributions walk each bound structurally against the region graph,
    mirroring :meth:`UnboxAnalysis._stamp`, so the table is keyed exactly where
    ``_stamp`` reads it.
    """
    sizes: dict[Region, int | None] = {}

    def contribute(region: Region, k: int | None) -> None:
        if region in sizes and sizes[region] != k:
            sizes[region] = None
        else:
            sizes[region] = k

    def seed(bound: ArraySizeBound, region: Region | None) -> None:
        if region is None:
            return
        match bound:
            case ListSize():
                contribute(region, concrete_size(bound.size))
                seed(bound.elt, alias.region_at(region))
            case TupleSize():
                for i, b in enumerate(bound.elts):
                    seed(b, alias.region_field(region, i))
            case None:
                contribute(region, None)

    for d in alias.all_defs():
        seed(array_size.by_def.get(d), alias.region_of(d))
    for e, bound in array_size.by_expr.items():
        if isinstance(e, _ALLOC_EXPRS):
            seed(bound, alias.region_of_expr(e))
    return sizes


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
    """The two facts ``decide`` cannot get from :class:`AliasAnalysis`.

    Both read a callee's ABI, so both are about a representation this target
    chose: ``at_boundary`` is the regions crossing a call that declared a
    handle -- an unknown callee counts as declaring one everywhere -- and
    ``written`` adds the arguments a callee stores through to the ones
    ``alias.written_regions`` already found here.
    """

    def __init__(
        self,
        alias: AliasAnalysis,
        callees: 'dict[FuncDef, CalleeAbi]',
    ):
        self.alias = alias
        self.callees = callees
        self.at_boundary: set[Region] = set()
        self.written: set[Region] = set(alias.written_regions)

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


def _unboxed(ty: CppType | None) -> bool:
    return isinstance(ty, CppList) and not ty.boxed


def _shares_storage(
    region: Region,
    alias: AliasAnalysis,
    storage: CppStorage,
    variables: VariableAnalysis,
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
            and storage.analysis.is_rebound(d)
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
            _binds_by_reference(d, storage, variables, def_use, alias)
            for d in ds
        ):
            owned_separately += 1
    return slots + owned_separately > 1


def _binds_by_reference(
    d: AssignDef,
    storage: CppStorage,
    variables: VariableAnalysis,
    def_use: DefineUseAnalysis,
    alias: AliasAnalysis,
) -> bool:
    """Whether *d* references storage that already exists *inside this function*.

    The one deliberate difference from the emitter: a parameter also binds by
    reference, but to the caller's storage, a place of its own -- discounting it
    would make ``zss = [xs]`` look unshared.  One extra term, so the divergence
    stays visible instead of becoming a second copy of the rule.
    """
    region = alias.region_of(d)
    return (
        binds_by_reference(
            storage, variables, def_use, d,
            allow_projection=(
                region is not None and region not in alias.slot_replaced
            ),
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


class StrictUnboxError(Exception):
    """Raised by :func:`check_strict` when a list must keep its handle."""


def check_strict(
    unbox: UnboxAnalysis,
    storage: CppStorage,
    variables: VariableAnalysis,
    ret_ty: CppType,
) -> None:
    """Refuse every list that kept its handle.

    ``UnboxMode.STRICT``'s contract is that the emitted unit holds no
    ``std::shared_ptr``.  This walks the artifacts the emitter reads its
    types from -- each storage class's stamped type and the stamped return
    type -- and reports every boxed level with the reason
    :meth:`Unbox.decide` recorded, so the error names the list to fix rather
    than the fact of failure.  Reported together: fixing one shared list only
    to be told about the next is a bad loop to put a user in.  One entry per
    list, too: a returned parameter is one list, so a ``<return>`` level
    whose regions a named class already reported is skipped.

    Expression temporaries have no storage class, so they are out of reach
    here; :meth:`UnboxAnalysis.annotate` refuses those at emission.
    """
    offenders: list[str] = []
    covered: set[Region] = set()
    for cls, ty in unbox.storage.items():
        name = variables.def_to_name[cls]
        per_depth = _regions(cls, storage, unbox.alias)
        for depth, on_spine in _boxed_levels(ty):
            reason = (
                unbox.boxed_because.get((cls, depth), 'shared')
                if on_spine else 'shared (tuple field)'
            )
            offenders.append(f'`{name}` (depth {depth}): {reason}')
            if on_spine and depth < len(per_depth):
                if (r := per_depth[depth]) is not None:
                    covered.add(r)
    for depth, on_spine in _boxed_levels(ret_ty):
        regions = (
            unbox.ret_regions[depth]
            if depth < len(unbox.ret_regions) else set()
        )
        if on_spine and regions and regions <= covered:
            continue
        reason = unbox._reason(regions) if on_spine else 'shared (tuple field)'
        offenders.append(f'<return> (depth {depth}): {reason}')
    if offenders:
        raise StrictUnboxError(
            'these lists must keep their shared handle '
            '(`std::shared_ptr`):\n  '
            + '\n  '.join(offenders)
            + '\nuse unbox=CppCompiler.UnboxMode.ALLOW to permit handles'
        )


def contains_boxed(ty: CppType) -> bool:
    """Whether any list level of *ty*, at any depth, keeps its handle."""
    return any(True for _ in _boxed_levels(ty))


def _boxed_levels(ty: CppType, depth: int = 0, on_spine: bool = True):
    """``(depth, on_spine)`` for every boxed list level in *ty*.

    ``on_spine`` mirrors :meth:`UnboxAnalysis._stamp`: a reason is recorded
    only down the list spine, so a tuple field's level has none to look up.
    """
    if isinstance(ty, CppList):
        if ty.boxed:
            yield depth, on_spine
        yield from _boxed_levels(ty.elt, depth + 1, on_spine)
    elif isinstance(ty, CppTuple):
        for e in ty.elts:
            yield from _boxed_levels(e, depth, False)
