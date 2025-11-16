"""
Engine interface for round-to-odd arithmetic implementations.
"""

from .engine import Engine, ENGINES, register_engine
from .gmp import MPFREngine

__all__ = [
    'Engine',
    'MPFREngine',
    'ENGINES',
    'register_engine',
]

# register default engine
register_engine(MPFREngine.instance(), priority=0)
