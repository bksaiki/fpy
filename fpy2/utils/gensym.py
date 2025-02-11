"""
This module defines the `Gensym` object that generates unique identifiers.
"""

from .identifier import Id, NamedId, UnderscoreId

class Gensym(object):
    """
    Unique identifier generator.

    The identifier is guaranteed to be unique among names that it
    has generated or reserved.
    """
    _idents: set[NamedId]
    _counter: int

    def __init__(self, *idents: NamedId):
        self._idents = set(idents)
        self._counter = len(idents)

    def reserve(self, *idents: NamedId):
        """Reserves a set of identifiers"""
        for ident in idents:
            if not isinstance(ident, NamedId):
                raise TypeError('must be a list of identifiers', idents)
            if ident in self._idents:
                raise RuntimeError(f'identifier `{ident}` already reserved')
            self._idents.add(ident)

    def refresh(self, ident: NamedId):
        """Generates a unique identifier for an existing identifier."""
        ident = NamedId(ident.base, ident.count)
        while ident in self._idents:
            ident.count = self._counter
            self._counter += 1

        self._idents.add(ident)
        return ident

    def fresh(self, prefix: str = 't'):
        """Generates a unique identifier with a given prefix."""
        return self.refresh(NamedId(prefix))

    def __contains__(self, name: NamedId):
        return name in self._idents

    def __len__(self):
        return len(self._idents)

    @property
    def names(self):
        return set(self._idents)
