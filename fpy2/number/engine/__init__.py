"""
Engine interface for round-to-odd arithmetic implementations.
"""

from .engine import Engine, ENGINES, register_engine
from .gmp import MPFREngine
from .real import RealEngine

__all__ = [
    'Engine',
    'MPFREngine',
    'RealEngine',
    'ENGINES',
    'register_engine',
]

# register default engines
register_engine(MPFREngine.instance(), priority=1)
register_engine(RealEngine.instance(), priority=0) # lower priority than MPFR
