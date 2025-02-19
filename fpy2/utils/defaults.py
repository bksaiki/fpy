"""
Decorators implementing some default behavior.
"""

def __default_repr__(x: object):
    return f'{x.__class__.__name__}({", ".join(f"{k}={v!r}" for k, v in x.__dict__.items())})'

def default_repr(cls):
    """Default __repr__ implementation for a class."""
    cls.__repr__ = __default_repr__
    return cls
