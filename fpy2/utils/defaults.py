"""
Decorators implementing some default behavior.
"""

from .ordering import Ordering

###########################################################
# Default methods

def __default_repr__(x: object):
    return f'{x.__class__.__name__}({", ".join(f"{k}={v!r}" for k, v in x.__dict__.items())})'

def __eq__(self, other):
    ord = self.compare(other)
    return ord is not None and ord == Ordering.EQUAL

def __lt__(self, other):
    ord = self.compare(other)
    return ord is not None and ord == Ordering.LESS

def __le__(self, other):
    ord = self.compare(other)
    return ord is not None and ord != Ordering.GREATER

def __gt__(self, other):
    ord = self.compare(other)
    return ord is not None and ord == Ordering.GREATER

def __ge__(self, other):
    ord = self.compare(other)
    return ord is not None and ord != Ordering.LESS

###########################################################
# Decorators

def default_repr(cls):
    """Default __repr__ implementation for a class."""
    cls.__repr__ = __default_repr__
    return cls

def partial_ord(cls):
    """Implement `__eq__`, `__lt__`, `__le__`, `__gt__`, and `__ge__`. """
    if not hasattr(cls, 'compare'):
        raise TypeError(f"{cls.__name__} must implement an abstract method 'compare'")

    cls.__eq__ = __eq__
    cls.__lt__ = __lt__
    cls.__le__ = __le__
    cls.__gt__ = __gt__
    cls.__ge__ = __ge__
    return cls
