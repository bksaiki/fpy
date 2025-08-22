"""
FPy backend abstraction.
"""

from abc import ABC, abstractmethod

from ..function import Function
from ..number import Context

class CompileError(Exception):
    """
    Base class for compilation errors.
    """
    pass

class Backend(ABC):
    """
    Abstract base class for FPy backends.
    """

    @abstractmethod
    def compile(self, func: Function, ctx: Context | None = None):
        """Compiles `func` to the backend's target language."""
        ...
