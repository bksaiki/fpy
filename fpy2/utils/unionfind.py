"""Unionfind data structure implementation."""

from typing import Generic, TypeVar, Iterable

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

    def __init__(self, xs: Iterable | None = None):
        self._parent = {} if xs is None else {x: x for x in xs}

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self._parent!r})'

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, x: T) -> bool:
        return x in self._parent

    def __iter__(self):
        return iter(self._parent)

    def add(self, x: T) -> T:
        """
        Add an element `x` to the union-find.
        Returns the representative of the set containing `x`.
        """
        if x not in self._parent:
            self._parent[x] = x
            return x
        else:
            return self._find(x)

    def _find(self, x: T) -> T:
        # use path compression
        parent = self._parent[x]
        while x != parent:
            # path compression
            gparent = self._parent[parent]
            self._parent[x] = gparent
            # advance twice
            x = gparent
            parent = self._parent[x]
        return x

    def find(self, x: T) -> T:
        """
        Finds the representative of the set containing `x`.
        Uses path compression for efficiency.
        """
        if x not in self._parent:
            raise KeyError(x)
        return self._find(x)

    def get(self, x: T, default=None):
        """
        Finds the representative of the set containing `x`,
        or `default` if a representative is not found.
        """
        if x in self._parent:
            return self._find(x)
        else:
            return default

    def union(self, x: T, y: T) -> T:
        """
        Union the sets containing `x` and `y`.
        The leader of `x` is the representative of the union.
        Returns the representative.
        """
        if x not in self._parent:
            raise KeyError(x)
        if y not in self._parent:
            raise KeyError(y, self._parent)

        root_x = self._find(x)
        root_y = self._find(y)
        self._parent[root_y] = root_x
        return root_x

    def items(self):
        """
        Returns an iterator over the items in the union-find.
        """
        return iter(self._parent.items())

    def representatives(self) -> set[T]:
        """
        Returns the set of all representatives (root elements) in the union-find.
        Each representative corresponds to a distinct disjoint set.
        """
        return {self._find(x) for x in self._parent}
