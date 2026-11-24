"""
Uninitalized value.
"""

from typing import Self


class _Uninit:
    """Singleton type for representing uninitialized values."""

    _instance = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNINIT"

    def __str__(self) -> str:
        return "UNINIT"

    def __bool__(self) -> bool:
        return False


# Singleton instance
UNINIT = _Uninit()
