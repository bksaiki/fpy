"""
Uninitalized value.
"""

class _Uninit:
    """Singleton type for representing uninitialized values."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "UNINIT"

    def __str__(self):
        return "UNINIT"

    def __bool__(self):
        return False


# Singleton instance
UNINIT = _Uninit()
