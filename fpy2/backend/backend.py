"""
FPy backend abstraction.
"""

from abc import ABC, abstractmethod

from ..runtime import Function


class Backend(ABC):
    """
    Abstract base class for FPy backends.
    """

    @abstractmethod
    def compile(self, func: Function):
        """Compiles `func` to the backend's target language."""
        raise NotImplementedError('virtual method')
