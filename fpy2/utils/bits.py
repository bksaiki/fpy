"""
Bitwise operations.
"""

import struct

def bitmask(k: int) -> int:
    """Return a bitmask of `k` bits."""
    return (1 << k) - 1

def float_to_bits(x: float) -> int:
    """Convert a Python float into a bistring."""
    if not isinstance(x, float):
        raise TypeError(f'Expected float x={x}')
    s = struct.pack('@d', x)
    return struct.unpack('@Q', s)[0]

def bits_to_float(i: int) -> float:
    """Convert a bistring into a Python float."""
    if not isinstance(i, int):
        raise TypeError(f'Expected integer i={i}')
    if i < 0 or i >= 2 ** 64:
        raise ValueError(f'Expected i={i} on [0, 2 ** 64)')
    s = struct.pack('@Q', i)
    return struct.unpack('@d', s)[0]

