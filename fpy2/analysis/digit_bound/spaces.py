"""
Index spaces: which loops and lists each store variable is a summary over.

A variable minted inside a binder -- a loop, a comprehension clause -- stands
for any of its iterations, and a list's element terms for any element.  A
binder *corresponds* with a list it reads or builds at its own index, and the
classes of that correspondence are the index spaces; a variable's
*signature* is the set of them it varies over.  See
``docs/todos/digit-bound-inference.md``, Part I, "Layer 2".
"""

from collections.abc import Hashable, Iterable

from ...utils import Unionfind

Key = Hashable
"""A binder or a list's element space, as the walk names it."""


class IndexSpaces:
    """Correspondence between :data:`Key`\\ s, and per variable the keys it
    varies over: the binders it was minted under, and what bounds it."""

    _classes: Unionfind[Key]
    _minted: dict[int, set[Key]]
    _bounded: dict[int, set[Key]]

    def __init__(self) -> None:
        self._classes = Unionfind()
        self._minted = {}
        self._bounded = {}

    def unify(self, a: Key, b: Key) -> None:
        """*a* and *b* correspond."""
        self._classes.union(self._classes.add(a), self._classes.add(b))

    def find(self, k: Key) -> Key:
        """*k*'s class, as its representative key."""
        return self._classes.get(k, k)

    def mint(self, var: int, keys: Iterable[Key]) -> None:
        """*var* was minted under the binders *keys*."""
        self._minted.setdefault(var, set()).update(keys)

    def bound(self, var: int, keys: Iterable[Key]) -> None:
        """*var* is bounded by something varying over *keys*."""
        self._bounded.setdefault(var, set()).update(keys)

    def keys(self, var: int, *, minted: bool = True) -> set[Key]:
        """The keys *var* varies over; without *minted*, only what bounds it."""
        out = set(self._bounded.get(var, ()))
        if minted:
            out |= self._minted.get(var, set())
        return out

    def signature(self, var: int) -> frozenset[Key]:
        """The index spaces *var* varies over."""
        return frozenset(self.find(k) for k in self.keys(var))
