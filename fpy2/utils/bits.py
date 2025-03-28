"""
Bitwise operations.
"""

import struct

def bitmask(k: int) -> int:
    """Return a bitmask of `k` bits."""
    return (1 << k) - 1

def float_to_bits(x: float) -> int:
    """Convert a float to its bit representation."""
    if not isinstance(x, float):
        raise TypeError(f'Expected float x={x}')
    s = struct.pack('@d', x)
    return struct.unpack('@q', s)[0]
