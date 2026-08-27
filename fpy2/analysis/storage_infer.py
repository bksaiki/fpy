"""
Storage inference: one storage format per runtime object.

Format inference bounds an expression by the smallest format its value can take;
storage inference picks, from a distinguished set the target can spell, one
format that *contains* that bound:

    e : real   fmt(e) = F   F <= S
    ------------------------------
              store(e) = S

The definitions denoting one runtime object form a class -- a union-find over
``reaching_defs.same_object_defs``, which unions on phi edges and in-place
updates and nothing else -- and each class stores at the join of its members'
bounds.  A plain rebind starts a *new* class with its own, possibly narrower,
storage: ``store`` is per object, not per name.

The domain is :class:`FormatBound` itself, so a backend supplies an ordered
sequence of formats and a fallback hook (:class:`StorageDomain`), nothing more.
Spelling a format in the target's own types is the backend's, as is
*representation* -- whether a list is a handle or a value is no property of a
format.

The join gives *containment*: every member of a class fits its class's storage.
It says nothing about *realizability* -- whether a value can be got into a place
whose storage was fixed elsewhere, such as a parameter or a callee's result --
which is where every refusal a consumer raises comes from.  For a scalar that is
a question about formats; for a list it is one about the heap, since changing an
element type means a new buffer and so a new object.

Termination rests on ``FormatBound`` being finite-depth: a cycle would need
``xs[0] = xs``, which fails to unify.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..ast.fpyast import Assign, Expr, ListRef, Var
from ..number.context.real import REAL_FORMAT
from ..utils import Unionfind
from .define_use import DefineUseAnalysis
from .format_infer import (
    AbstractableFormat,
    AbstractFormat,
    FormatBound,
    ListFormat,
    SetFormat,
    TupleFormat,
    VarFormat,
    is_bottom,
)
from .format_infer.analysis import _to_abstract
from .reaching_defs import AssignDef, Definition, same_object_defs


class StorageSelectionError(Exception):
    """Raised when no storage in the domain contains an inferred format."""


class StorageDomain(Protocol):
    """What a backend contributes to storage assignment: its formats.

    One input and one hook is enough.  A storage *is* a format the target can
    spell, so containment over formats answers the rest: ``of_bound`` is the
    first member containing a bound, ``join`` the first containing several, and
    losslessness *is* containment, so no conversion relation is needed.  Nor a
    map over structure -- ``ListFormat`` and ``TupleFormat`` are already bounds.
    """

    @property
    def sigma(self) -> Sequence[AbstractableFormat]:
        """The storage formats, smallest first.

        A **sequence**, not a set: containment over formats is not a
        join-semilattice -- ``{s8, u16}`` has two incomparable minimal upper
        bounds and no least one -- so the order is the tie-break, and a
        different order changes which programs are storable.
        """
        ...

    def fallback(self, bound: FormatBound) -> AbstractableFormat | None:
        """A storage for a bound no member of :attr:`sigma` contains, or
        ``None`` to refuse.

        The one thing the sequence cannot supply: a target may accept a bound on
        terms of its own -- the cpp backend stores an unbounded integer in
        ``int64_t`` and treats overflow as the user's problem, which no
        containment test would allow.
        """
        ...


@dataclass
class StorageAnalysis:
    """Result of :class:`StorageInfer`.

    Each SSA def belongs to a *class* -- the defs denoting one runtime object,
    per ``same_object_defs``.  What a target with variables then does with a
    class -- name it, place its declaration -- is the backend's.

    Attributes:
        def_class:      each def's class id (the canonical member).
        class_members:  each class id's member defs.
        class_storage:  the storage format chosen per class.
    """
    def_class: dict[Definition, Definition]
    class_members: dict[Definition, list[Definition]]
    class_storage: dict[Definition, FormatBound]
    expr_bound: dict[Expr, FormatBound]
    def_use: DefineUseAnalysis
    domain: StorageDomain

    def of_expr(self, e: Expr) -> FormatBound:
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
            if isinstance(base, ListFormat):
                return base.elt
        try:
            return of_bound(self.domain, self.expr_bound.get(e))
        except StorageSelectionError:
            return None

    def is_rebound(self, d: Definition) -> bool:
        """Is the name *d* introduces ever bound to a different value?

        ``xs[i] = e`` is not a rebind -- it writes *through* an element cell, so
        the name still denotes the same list and that def stays in the same
        class.  Only an ``Assign`` to the same name is, and *d*'s own defining
        assignment does not count.
        """
        cls = self.def_class[d]
        return any(
            m is not d and isinstance(m, AssignDef) and isinstance(m.site, Assign)
            for m in self.class_members[cls]
        )

    def storage_of(self, d: Definition) -> FormatBound:
        """The storage chosen for *d*'s class."""
        return self.class_storage[self.def_class[d]]


def _lift(bound: FormatBound) -> AbstractFormat | None:
    """*bound* as an :class:`AbstractFormat`, which is what carries ``<=``, or
    ``None`` for a bound with no scalar reading."""
    if isinstance(bound, AbstractableFormat | SetFormat):
        return _to_abstract(bound)
    return None


def of_bound(domain: StorageDomain, bound: FormatBound) -> FormatBound:
    """The smallest storage in *domain* containing *bound*.

    Structural: a list's storage is a list of its element's storage, a tuple's
    is field-wise.  ``None`` -- a non-numeric bound, or a kind nothing resolved
    -- has no numeric storage and stays ``None``; the backend spells it however
    it spells a boolean.

    A bottom bound holds no value, so every member contains it vacuously and the
    first wins.  Where no member contains the bound the domain gets one chance to
    accept it anyway (:meth:`StorageDomain.fallback`) before this refuses.
    """
    if bound is None or isinstance(bound, VarFormat):
        return None
    if isinstance(bound, TupleFormat):
        return TupleFormat(tuple(of_bound(domain, b) for b in bound.elts))
    if isinstance(bound, ListFormat):
        return ListFormat(of_bound(domain, bound.elt))
    if is_bottom(bound):
        return domain.sigma[0]
    if bound == REAL_FORMAT:
        raise StorageSelectionError(
            'cannot store an unconstrained real value in any storage format; '
            'is the active rounding context symbolic?  Try monomorphizing the '
            'function with a concrete context.'
        )
    af = _lift(bound)
    if af is None:
        raise StorageSelectionError(
            f'cannot compare {bound!r} against the storage formats; '
            'storage selection requires a dyadic format'
        )
    for sigma in domain.sigma:
        if af <= AbstractFormat.from_format(sigma):
            return sigma
    chosen = domain.fallback(bound)
    if chosen is not None:
        return chosen
    raise StorageSelectionError(f'no storage format contains {bound!r}')


def join(domain: StorageDomain, storages: list[FormatBound]) -> FormatBound:
    """The smallest storage in *domain* containing every input.

    **N-ary, and it must not be folded.**  Containment is not a
    join-semilattice, so a pairwise fold is both less precise and less total
    than one search over the whole collection: over the cpp ladder,
    ``join{s8, u16, f32}`` is ``float`` where folding gives ``double``, and
    ``join{s8, u32, f32}`` succeeds where folding fails outright.

    Inputs share a structural shape -- the type checker upstream guarantees it --
    so a mismatch is an analysis bug rather than a program this cannot store.
    """
    head, *rest = storages
    if not rest:
        return head
    if head is None:
        assert all(s is None for s in rest), (
            f'cannot join a non-numeric storage with a numeric one: {storages!r}'
        )
        return None
    if isinstance(head, ListFormat):
        elts = []
        for s in storages:
            assert isinstance(s, ListFormat), storages
            elts.append(s.elt)
        return ListFormat(join(domain, elts))
    if isinstance(head, TupleFormat):
        tuples = []
        for s in storages:
            assert isinstance(s, TupleFormat), storages
            assert len(s.elts) == len(head.elts), storages
            tuples.append(s)
        return TupleFormat(tuple(
            join(domain, [t.elts[i] for t in tuples])
            for i in range(len(head.elts))
        ))
    afs = []
    for s in storages:
        af = _lift(s)
        assert af is not None, (
            f'cannot join a non-numeric storage with a numeric one: {storages!r}'
        )
        afs.append(af)
    for sigma in domain.sigma:
        rung = AbstractFormat.from_format(sigma)
        if all(af <= rung for af in afs):
            return sigma
    raise StorageSelectionError(f'no storage format subsumes {storages!r}')


def _aggregate(domain: StorageDomain, bounds: list[FormatBound]) -> FormatBound:
    """One storage containing every bound in *bounds*.

    A bottom bound -- a fresh ``empty(...)`` -- holds no value, so it constrains
    nothing and is dropped whenever another member does constrain: keeping it
    would widen for nothing, since its storage is the domain's first member and
    joining that with a signed one costs a rung.

    The join is over *storages*, not bounds, which is where the residual
    imprecision lives: each member is rounded up to its own smallest containing
    format first, so a class of ``{0}`` and ``s8`` joins at ``s16``, and a bound
    only *partly* bottom still contributes the first member at its empty slots.
    Joining the bounds and choosing storage once would be strictly tighter.
    """
    assert bounds, 'a class has at least one member'
    constraining = [b for b in bounds if not is_bottom(b)]
    return join(domain, [of_bound(domain, b) for b in (constraining or bounds)])


class StorageInfer:
    """Storage assignment; see the module docstring for the contract."""

    @staticmethod
    def infer(
        def_use: DefineUseAnalysis,
        def_to_bound: dict[Definition, FormatBound],
        expr_to_bound: dict[Expr, FormatBound],
        domain: StorageDomain,
    ) -> StorageAnalysis:
        """Build a :class:`StorageAnalysis` from def-use info and per-def bounds.

        Raises :class:`StorageSelectionError` when no member of *domain* covers
        some class's joined bound.
        """
        defs = def_use.defs

        # classes: union-find over coalescing edges
        uf: Unionfind[Definition] = Unionfind(defs)
        for d in defs:
            for i in same_object_defs(d):
                uf.union(d, defs[i])

        def_class: dict[Definition, Definition] = {d: uf.find(d) for d in defs}
        class_members: dict[Definition, list[Definition]] = defaultdict(list)
        for d, c in def_class.items():
            class_members[c].append(d)

        # one storage per class
        class_storage: dict[Definition, FormatBound] = {}
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
                class_storage[c] = _aggregate(domain, bounds)
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
            domain=domain,
        )
