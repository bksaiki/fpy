"""
FPy backend abstraction.
"""

from abc import ABC, abstractmethod

from ..ir import FuncDef


class Backend(ABC):
    """
    Abstract base class for FPy backends.
    """

    @abstractmethod
    def compile(self, func: FuncDef):
        """Compiles `func` to the backend's target language."""
        ...
