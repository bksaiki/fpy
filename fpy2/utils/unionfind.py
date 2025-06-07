"""Unionfind data structure implementation."""

from typing import Generic, TypeVar

T = TypeVar("T")

class Unionfind(Generic[T]):
    """
    The classic union-find data structure.

    A union-find extends the usual set data structure by
    grouping elements into disjoint (sub-)sets.
    The union-find supports three main operations:
    - `find`: determine which set a particular element is in.
    - `union`: join two sets together.
    """

    _parent: dict[T, T]

    def __init__(self, *args):
        self._parent = {elt: elt for elt in args}

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self._parent!r})'

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, x: T) -> bool:
        return x in self._parent

    def add(self, x: T) -> None:
        """
        Add an element `x` to the union-find.
        If `x` is already present, this is a no-op.
        """
        if x not in self._parent:
            self._parent[x] = x

    def _find(self, x: T) -> T:
        if self._parent[x] != x:
            self._parent[x] = self._find(self._parent[x])  # Path compression
        return self._parent[x]

    def find(self, x: T) -> T:
        """
        Finds the representative of the set containing `x`.
        Uses path compression for efficiency.
        """
        if x not in self._parent:
            raise KeyError(x)
        return self._find(x)

    def union(self, x: T, y: T) -> T:
        """
        Union the sets containing `x` and `y`.
        The leader of `x` is the representative of the union.
        Returns the representative.
        """
        if x not in self._parent:
            raise KeyError(x)
        if y not in self._parent:
            raise KeyError(y)

        root_x = self._find(x)
        root_y = self._find(y)
        self._parent[root_y] = root_x
        return root_x
