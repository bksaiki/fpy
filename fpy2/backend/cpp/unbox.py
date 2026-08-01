"""
cpp backend: which lists may drop the handle.

A list compiles to ``fpy::list<T>`` — a shared handle — because FPy lists alias.
Where :mod:`fpy2.analysis.alias` proves nothing else can observe a list, the
indirection buys nothing and it can be a plain ``std::vector<T>`` instead: no
allocation, no refcount, and a type a native caller already holds.

The decision is per **alias region** — the analysis's own equivalence class —
because that is the only unit at which one answer is guaranteed to satisfy
everything that reads it.  Deciding per definition, or per expression, would let
two things that must share a C++ type get different answers.

Two partitions, and neither refines the other
---------------------------------------------

- :mod:`fpy2.analysis.alias` groups by what may refer to what.
- :mod:`.storage_infer` groups by phi edges and in-place-mutation edges, giving
  one C++ variable per class — reasons the alias analysis knows nothing about.

So a storage class can hold defs from different alias regions::

    if c:  ys = [x, x]      # fresh: nothing else refers to it
    else:  ys = xs          # the caller's list
    ys[0] = 99

``ys`` is one C++ variable spanning a literal (owned) and a parameter (shared).
There is no representation satisfying both, and taking the optimistic one makes
``ys = xs`` a silent conversion that loses the write.

Hence: **a storage class is unboxed only if every region it touches is uniquely
owned, and that verdict is then written back onto all of them.** The write-back
is what keeps expressions honest — without it, the literal above would still
report itself owned, and be emitted as a ``std::vector`` initialising an
``fpy::list``.  Since a class can drag a region to *boxed* and that region may
belong to another class, the write-back is iterated to a fixed point; it
terminates because verdicts only ever move one way.

Callers must agree
------------------

A parameter's representation is part of the signature, so a function called from
compiled code keeps its handles: the caller passes a list it has marked shared
outward, which is boxed.  Measured over the corpus this costs 5 of the 50
functions with a list parameter — the kernels worth unboxing are entry points a
*native* caller invokes.

Refusal means keeping the handle, never failing a compile — unboxing is an
optimization.  Reasons are recorded in :attr:`UnboxAnalysis.boxed_because` so a
missed case is inspectable rather than invisible.

Not decided here: a list inside a tuple.  Storage for a ``CppTuple`` is left
alone, so such a list stays boxed.
"""

from dataclasses import dataclass, field

from ...analysis import Definition
from ...analysis.alias import AliasAnalysis, Region
from ...analysis.reaching_defs import AssignDef
from ...ast.fpyast import Argument, Expr, FuncDef, ReturnStmt
from ...ast.visitor import DefaultVisitor
from .storage_infer import StorageAnalysis
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
    own_boxed: dict[Region, bool] = field(default_factory=dict)
    storage: dict[Definition, CppType] = field(default_factory=dict)
    ret_regions: list[set[Region]] = field(default_factory=list)
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
        *,
        is_called: bool = False,
    ) -> UnboxAnalysis:
        """Which lists may drop the handle.

        Args:
            ast: the function, for its ``return`` statements.
            storage: the storage classes to decide.
            alias: what may refer to what.
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
            owned = alias.is_uniquely_owned(site)
            out.boxed[region] = out.boxed.get(region, False) or not owned
        out.own_boxed = dict(out.boxed)

        # 2. a storage class must have one representation per level, so every
        #    region it touches takes the conjunction.  Iterated: dragging a
        #    region to boxed can make another class conservative in turn.
        classes = [
            (cls, _regions(cls, storage, alias))
            for cls, ty in storage.class_storage.items()
            if isinstance(ty, CppList)
        ]
        out.ret_regions = _return_regions(ast, alias)
        groups = [p for _c, per in classes for p in per] + out.ret_regions
        changed = True
        while changed:
            changed = False
            for regions in groups:
                if not regions:
                    continue
                boxed = any(out.boxed.get(r, True) for r in regions)
                for r in regions:
                    if out.boxed.get(r) != boxed:
                        out.boxed[r] = boxed
                        changed = True

        # 3. read the decision back out per class
        for cls, per_depth in classes:
            ty = storage.class_storage[cls]
            out.storage[cls] = _read(ty, per_depth, cls, out, 0, is_called)
        return out


def _regions(
    cls: Definition, storage: StorageAnalysis, alias: AliasAnalysis,
) -> list[set[Region]]:
    """The alias regions each level of *cls*'s storage may hold, by depth."""
    ty = storage.class_storage[cls]
    per_depth: list[set[Region]] = []
    depth = 0
    while isinstance(ty, CppList):
        found = {
            r for d in storage.class_members[cls]
            if (r := alias.region_of(d, depth)) is not None
        }
        per_depth.append(found)
        ty, depth = ty.elt, depth + 1
    return per_depth


def _read(
    ty: CppType,
    per_depth: list[set[Region]],
    cls: Definition,
    out: UnboxAnalysis,
    depth: int,
    is_called: bool,
) -> CppType:
    if not isinstance(ty, CppList):
        return ty
    elt = _read(ty.elt, per_depth, cls, out, depth + 1, is_called)
    regions = per_depth[depth]
    if not regions:
        out.boxed_because[(cls, depth)] = 'no alias information'
        return CppList(elt, boxed=True)
    if is_called and _is_parameter(cls):
        for r in regions:
            out.boxed[r] = True
        out.boxed_because[(cls, depth)] = 'a caller holds a handle'
        return CppList(elt, boxed=True)
    if any(out.boxed.get(r, True) for r in regions):
        # distinguish "this list really is shared" from "one C++ variable spans
        # regions that answered differently, and it admits a single answer"
        own = [out.own_boxed.get(r, True) for r in regions]
        out.boxed_because[(cls, depth)] = (
            'shared' if all(own) else 'sites disagree'
        )
        return CppList(elt, boxed=True)
    return CppList(elt, boxed=False)


def _is_parameter(cls: Definition) -> bool:
    return isinstance(cls, AssignDef) and isinstance(cls.site, Argument)


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
