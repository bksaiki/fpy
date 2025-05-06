"""
This module defines a substitution for FPy IR.
"""

from ..ast import NamedId, Expr
from ..utils import default_repr


@default_repr
class Subst:
    """Mapping between pattern variable and IR instances."""

    env: dict[NamedId, Expr]
    """mapping from pattern variable to expressions"""

    def __init__(self):
        self.env = {}

    def __getitem__(self, key: str | NamedId) -> Expr:
        """Returns the value of the key in the substitution."""
        if isinstance(key, str):
            key = NamedId(key)
        return self.env[key]

    def __setitem__(self, key: str | NamedId, value: Expr):
        """Sets the value of the key in the substitution."""
        if isinstance(key, str):
            key = NamedId(key)
        self.env[key] = value

    def __contains__(self, key: str | NamedId) -> bool:
        """Returns True if the key is in the substitution."""
        if isinstance(key, str):
            key = NamedId(key)
        return key in self.env

    def __delitem__(self, key: str | NamedId):
        """Deletes the key from the substitution."""
        if isinstance(key, str):
            key = NamedId(key)
        del self.env[key]

    def __iter__(self):
        """Returns an iterator over the keys of the substitution."""
        return iter(self.env)

    def __or__(self, other: 'Subst'):
        """Returns the union of two substitutions."""
        if not isinstance(other, Subst):
            raise TypeError(f'Cannot union {type(self)} with {type(other)}')
        new_subst = Subst()
        new_subst.env = self.env
        for k, v in other.env.items():
            if k in new_subst.env and new_subst.env[k] != v:
                raise ValueError(f'unioning substitutions with different values for {k}: {new_subst.env[k]} != {v}')
            new_subst.env[k] = v
        return new_subst

    def vars(self):
        """Returns the domain of the substitution."""
        return set(self.env.keys())
