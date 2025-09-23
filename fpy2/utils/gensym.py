"""
This module defines the `Gensym` object that generates unique identifiers.
"""

from typing import Callable, Collection

from .identifier import *

class Gensym(object):
    """
    Unique identifier generator.

    The identifier is guaranteed to be unique among names that it
    has generated or reserved.
    """
    _idents: set[NamedId]
    _generated: set[NamedId]
    _counter: int
    _rename_hook: Callable[[str], str] | None

    def __init__(
        self,
        reserved: Collection[NamedId] | None = None,
        rename_hook: Callable[[str], str] | None = None
    ):
        if reserved is None:
            self._idents = set()
            self._counter = 0
        else:
            self._idents = set(reserved)
            self._counter = len(reserved)

        self._generated = set()
        self._rename_hook = rename_hook

    def _copy_id(self, id: NamedId) -> NamedId:
        match id:
            case SourceId():
                return SourceId(id.base, id.loc, id.count)
            case NamedId():
                return NamedId(id.base, id.count)

    def reserve(self, *idents: NamedId):
        """Reserves a set of identifiers. Does not add to `self.generated`."""
        for ident in idents:
            if not isinstance(ident, NamedId):
                raise TypeError('must be a list of identifiers', idents)
            if ident in self._idents:
                raise RuntimeError(f'identifier `{ident}` already reserved')
            self._idents.add(ident)

    def refresh(self, ident: NamedId):
        """Generates a unique identifier for an existing identifier."""
        ident = self._copy_id(ident)
        while ident in self._idents:
            ident.count = self._counter
            self._counter += 1

        self._idents.add(ident)
        self._generated.add(ident)
        return ident

    def fresh(self, prefix: str = 't'):
        """
        Generates a unique identifier with a given prefix.

        If this generator was initialized with `rename_hook=...`,
        then the identifier is potentially transformed.
        """
        if self._rename_hook:
            prefix = self._rename_hook(prefix)
        return self.refresh(NamedId(prefix))

    def __contains__(self, name: NamedId):
        return name in self._idents

    def __len__(self):
        return len(self._idents)

    @property
    def names(self):
        return set(self._idents)

    @property
    def generated(self):
        """Returns the set of generated identifiers."""
        return set(self._generated)
