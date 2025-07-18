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

    return f'{x.__class__.__name__}({", ".join(items)})'

def default_repr(cls):
    """Default __repr__ implementation for a class."""
    cls.__repr__ = __default_repr__
    return cls

###########################################################
# Comparison reversal

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
        __eq__ = cls.__eq__
        __lt__ = cls.__lt__
        __le__ = cls.__le__
        __gt__ = cls.__gt__
        __ge__ = cls.__ge__

        def __req__(self, other):
            if isinstance(other, this_cls):
                # reverse the comparison
                return other == self
            else:
                # normal order
                return __eq__(self, other)

        def __rlt__(self, other):
            if isinstance(other, this_cls):
                # reverse the comparison
                return other > self
            else:
                # normal order
                return __lt__(self, other)

        def __rle__(self, other):
            if isinstance(other, this_cls):
                # reverse the comparison
                return other >= self
            else:
                # normal order
                return __le__(self, other)

        def __rgt__(self, other):
            if isinstance(other, this_cls):
                # reverse the comparison
                return other < self
            else:
                # normal order
                return __gt__(self, other)

        def __rge__(self, other):
            if isinstance(other, this_cls):
                # reverse the comparison
                return other <= self
            else:
                # normal order
                return __ge__(self, other)

        cls.__eq__ = __req__
        cls.__lt__ = __rlt__
        cls.__le__ = __rle__
        cls.__gt__ = __rgt__
        cls.__ge__ = __rge__

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
