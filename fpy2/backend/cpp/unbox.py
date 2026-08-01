"""
cpp backend: which lists may drop the handle.

A list compiles to ``fpy::list<T>`` — a shared handle — because FPy lists alias.
Where :mod:`fpy2.analysis.alias` proves nothing else can observe a list, the
indirection buys nothing and it can be a plain ``std::vector<T>`` instead: no
allocation, no refcount, and a type a native caller already holds.

This module decides that, per storage class.  It is the join between two
partitions of the program that do not refine each other:

- :mod:`fpy2.analysis.alias` gives a verdict per **allocation site**, grouping by
  what may refer to what.
- :mod:`.storage_infer` gives one C++ variable per **storage class**, grouping by
  phi edges and in-place-mutation edges — reasons the alias analysis knows nothing
  about.

So a storage class can hold defs whose sites disagree::

    if c:  ys = [x, x]      # fresh: nothing else refers to it
    else:  ys = xs          # the caller's list
    ys[0] = 99

``ys`` is one C++ variable, and its class holds a literal (owned) and a parameter
(shared).  There is no representation satisfying both: unboxing ``ys`` while
``xs`` stays boxed makes ``ys = xs`` a silent conversion, and the write is lost.

Hence the rule: **a class is unboxed only if every site of every member is
uniquely owned.**  Disagreement, or no information at all, means boxed.  Since
unboxing is an optimization, "refuse" means keep the handle, never fail to
compile — but the reason is recorded in :attr:`UnboxAnalysis.boxed_because` so a
missed case is inspectable rather than invisible.

Why the conjunction is enough
-----------------------------

It might look as though a second pass is needed to check that two classes
connected by an assignment agree — otherwise ``ys = xs`` could still convert.  It
is not.  ``ys = xs`` makes the alias analysis merge ``cell(ys)`` with
``cell(xs)``, so both defs report the *same* site set.  A storage class only ever
adds *more* alias classes to the conjunction, never fewer, so it cannot come out
unboxed while a class it is assigned from comes out boxed.  Unification does that
work already.

Levels are independent
----------------------

A nested list gets one decision per level, from the sites at that depth: the rows
of a ``list[list[Real]]`` are distinct objects with their own ownership.  All four
combinations are expressible (``std::vector<fpy::list<T>>`` and
``fpy::list<std::vector<T>>`` included), so no level constrains another.

Not yet decided here: a list inside a tuple.  Storage for a ``CppTuple`` is left
alone, so such a list stays boxed.
"""

from dataclasses import dataclass, field

from ...analysis import Definition
from ...analysis.alias import AliasAnalysis
from .storage_infer import StorageAnalysis
from .types import CppList, CppType


@dataclass
class UnboxAnalysis:
    """Result of :class:`Unbox`.

    ``storage`` gives the representation-annotated type for every storage class
    the decision touched; a class absent from it keeps whatever
    :mod:`.storage_infer` chose.
    """

    storage: dict[Definition, CppType] = field(default_factory=dict)
    boxed_because: dict[tuple[Definition, int], str] = field(
        default_factory=dict,
    )

    def storage_of(self, cls: Definition, fallback: CppType) -> CppType:
        return self.storage.get(cls, fallback)

    def unboxed_levels(self) -> int:
        """How many list levels came out unboxed — for reporting and tests."""
        return sum(_count_unboxed(ty) for ty in self.storage.values())


def _count_unboxed(ty: CppType) -> int:
    if not isinstance(ty, CppList):
        return 0
    return (0 if ty.boxed else 1) + _count_unboxed(ty.elt)


class Unbox:
    """Decide the representation of every list storage class."""

    @staticmethod
    def decide(
        storage: StorageAnalysis,
        alias: AliasAnalysis,
    ) -> UnboxAnalysis:
        """Which storage classes may drop the handle, and at which levels."""
        out = UnboxAnalysis()
        for cls, ty in storage.class_storage.items():
            if not isinstance(ty, CppList):
                continue
            members = storage.class_members[cls]
            out.storage[cls] = _decide(ty, cls, members, alias, out, 0)
        return out


def _decide(
    ty: CppType,
    cls: Definition,
    members: list[Definition],
    alias: AliasAnalysis,
    out: UnboxAnalysis,
    depth: int,
) -> CppType:
    """*ty* with a representation chosen for each of its list levels."""
    if not isinstance(ty, CppList):
        return ty
    elt = _decide(ty.elt, cls, members, alias, out, depth + 1)

    sites = frozenset().union(*(alias.sites_of(d, depth) for d in members))
    if not sites:
        # No site reaches this level: either the alias analysis never looked
        # inside it, or the defs are emitter-internal.  Absence is not evidence.
        out.boxed_because[(cls, depth)] = 'no alias information'
        return CppList(elt, boxed=True)

    owned = {s for s in sites if alias.is_uniquely_owned(s)}
    if owned != sites:
        out.boxed_because[(cls, depth)] = (
            'sites disagree' if owned else 'shared'
        )
        return CppList(elt, boxed=True)
    return CppList(elt, boxed=False)
