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
    _sets: dict[T, set[T]]

    def __init__(self, xs: Iterable[T] | None = None):
        if xs is None:
            self._parent = {}
            self._sets = {}
        else:
            self._parent = { x: x for x in xs}
            self._sets = { x: {x} for x in xs }

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'({self._sets!r})'

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, x: T) -> bool:
        return x in self._parent

    def __iter__(self):
        return iter(self._sets)

    def add(self, x: T) -> T:
        """
        Add an element `x` to the union-find.
        Returns the representative of the set containing `x`.
        """
        if x not in self._parent:
            self._parent[x] = x
            self._sets[x] = {x}
            return x
        else:
            return self._find(x)

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
        return self._union(x, y)

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

    def _find(self, x: T) -> T:
        """
        Finds the representative of the set containing `x`.
        """
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

    def _union(self, x: T, y: T) -> T:
        """
        Unifies the sets containing `x` and `y`.
        The representative of the union is the representative of
        the set containing `x`.
        """
        # unifies the sets containing x and y
        root_x = self._find(x)
        root_y = self._find(y)
        if root_x != root_y:
            self._parent[root_y] = root_x
            self._sets[root_x].update(self._sets[root_y])
            del self._sets[root_y]
        return root_x
