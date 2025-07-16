"""
Decorators implementing some default behavior.
"""

from enum import Enum

###########################################################
# Default __repr__ decorator

def _get_slots(cls):
    """
    Get all slots from the class and its base classes.
    """
    slots = set()
    for c in cls.__mro__:
        if hasattr(c, '__slots__'):
            slots.update(c.__slots__)
    return slots

def __default_repr__(x: object):
    # get attributes from __dict__ if available
    items: list[str] = []
    if hasattr(x, '__dict__'):
        for k, v in x.__dict__.items():
            if not k.startswith('_'):
                items.append(f'{k}={v!r}')
    # get attributes from __slots__ if available, including inherited slots
    for slot in _get_slots(type(x)):
        if not slot.startswith('_') and hasattr(x, slot):
            value = getattr(x, slot)
            items.append(f'{slot}={value!r}')

    return f'{x.__class__.__name__}({" ".join(items)})'

def default_repr(cls):
    """Default __repr__ implementation for a class."""
    cls.__repr__ = __default_repr__
    return cls

###########################################################
# Comparison reversal

def __req__(self, other, cls, __eq__):
    if isinstance(other, cls):
        # reverse the comparison
        return other == self
    else:
        # normal order
        return __eq__(self, other)

def __rlt__(self, other, cls, __lt__):
    if isinstance(other, cls):
        # reverse the comparison
        return other > self
    else:
        # normal order
        return __lt__(self, other)

def __rle__(self, other, cls, __le__):
    if isinstance(other, cls):
        # reverse the comparison
        return other >= self
    else:
        # normal order
        return __le__(self, other)

def __rgt__(self, other, cls, __gt__):
    if isinstance(other, cls):
        # reverse the comparison
        return other < self
    else:
        # normal order
        return __gt__(self, other)

def __rge__(self, other, cls, __ge__):
    if isinstance(other, cls):
        # reverse the comparison
        return other <= self
    else:
        # normal order
        return __ge__(self, other)


def rcomparable(cls):
    """
    Implement `__eq__`, `__lt__`, `__le__`, `__gt__`, and `__ge__`
    between this class and an object of type `cls`

    Use this decorate to reverse the order of comparison,
    that is, if `cls` does not support comparison against this class,
    but this class may be compared against `cls`, extend
    comparison functionality by reversing the comparison.
    """
    def wrap(this_cls):
        old_eq = cls.__eq__
        old_lt = cls.__lt__
        old_le = cls.__le__
        old_gt = cls.__gt__
        old_ge = cls.__ge__

        cls.__eq__ = lambda self, other: __req__(self, other, this_cls, old_eq)
        cls.__lt__ = lambda self, other: __rlt__(self, other, this_cls, old_lt)
        cls.__le__ = lambda self, other: __rle__(self, other, this_cls, old_le)
        cls.__gt__ = lambda self, other: __rgt__(self, other, this_cls, old_gt)
        cls.__ge__ = lambda self, other: __rge__(self, other, this_cls, old_ge)

        return this_cls

    return wrap

############################################################
# Default __repr__ for enum values

def __default_enum_repr__(x: Enum):
    """
    Default __repr__ implementation for enum values.
    """
    return f'{x.__class__.__name__}.{x.name}'

def enum_repr(cls):
    """
    Default __repr__ implementation for enum values.
    """
    cls.__repr__ = __default_enum_repr__
    return cls
