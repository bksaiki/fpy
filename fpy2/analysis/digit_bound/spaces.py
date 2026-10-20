"""
Index spaces: which loops and lists each store variable is a summary over.

A variable minted inside a binder -- a loop, a comprehension clause -- stands
for any of its iterations, and a list's element terms for any element.  A
binder *corresponds* with a list it reads or builds at its own index, and the
classes of that correspondence are the index spaces; a variable's
*signature* is the set of them it varies over.  See
``docs/todos/digit-bound-inference.md``, Part I, "Layer 2".
"""

from collections.abc import Container, Iterable
from dataclasses import dataclass
from typing import TypeAlias

from ...utils import Unionfind


@dataclass(frozen=True)
class Binder:
    """A loop, or a comprehension's clause, in one walk."""

    walk: int
    node: int
    """The loop's or comprehension's `id`."""
    clause: int = 0


@dataclass(frozen=True)
class Elements:
    """A list's element space, in one walk."""

    walk: int
    of: int
    """The list expression's or definition's `id`."""


Key: TypeAlias = Binder | Elements
"""A binder or a list's element space."""


class IndexSpaces:
    """Correspondence between :data:`Key`\\ s, and per variable the keys it
    varies over: the binders it was minted under, and what bounds it."""

    _classes: Unionfind[Key]
    _minted: dict[int, set[Key]]
    _bounded: dict[int, set[Key]]
    _around: dict[Key, tuple[Key, ...]]
    _walks: int

    def __init__(self) -> None:
        self._classes = Unionfind()
        self._minted = {}
        self._bounded = {}
        self._around = {}
        self._walks = 0

    def walk(self) -> int:
        """A fresh walk's identity, for its keys: an `id` is reused once its
        object is freed, and a callee's walk is."""
        self._walks += 1
        return self._walks

    def unify(self, a: Key, b: Key) -> None:
        """*a* and *b* correspond."""
        self._classes.union(self._classes.add(a), self._classes.add(b))

    def open(self, binder: Key, around: tuple[Key, ...]) -> None:
        """*binder* runs inside the binders *around*, outermost first."""
        self._around[binder] = around

    def around(self, binder: Key) -> tuple[Key, ...]:
        """The binders *binder* runs inside, outermost first."""
        return self._around.get(binder, ())

    def correspond(self, a: Key, b: Key) -> bool:
        """Unify *a* and *b*, unless that puts two nested binders in one class:
        each has its own index at a point inside both, and a class is one
        index.  Whether it did."""
        ca = self._classes.component(self._classes.add(a))
        cb = self._classes.component(self._classes.add(b))
        if any(x in self.around(y) or y in self.around(x) for x in ca for y in cb):
            return False
        self.unify(a, b)
        return True

    def find(self, k: Key) -> Key:
        """*k*'s class, as its representative key."""
        return self._classes.get(k, k)

    def mint(self, var: int, keys: Iterable[Key]) -> None:
        """*var* was minted under the binders *keys*."""
        self._minted.setdefault(var, set()).update(keys)

    def bound(self, var: int, keys: Iterable[Key]) -> None:
        """*var* is bounded by something varying over *keys*."""
        self._bounded.setdefault(var, set()).update(keys)

    def copy(self, var: int, src: int, moved: Container[Key]) -> None:
        """*var* is *src* at another index of the classes *moved*: it varies
        over the rest of what *src* does, as *src* does."""
        for table in (self._minted, self._bounded):
            keys = {k for k in table.get(src, ()) if self.find(k) not in moved}
            table.setdefault(var, set()).update(keys)

    def keys(self, var: int, *, minted: bool = True) -> set[Key]:
        """The keys *var* varies over; without *minted*, only what bounds it."""
        out = set(self._bounded.get(var, ()))
        if minted:
            out |= self._minted.get(var, set())
        return out

    def signature(self, var: int) -> frozenset[Key]:
        """The index spaces *var* varies over."""
        return frozenset(self.find(k) for k in self.keys(var))
