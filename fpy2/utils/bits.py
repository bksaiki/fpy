"""
Bitwise operations.
"""

import struct

def bitmask(k: int) -> int:
    """Return a bitmask of `k` bits."""
    return (1 << k) - 1

def is_power_of_two(k: int) -> bool:
    """
    Returns if `k` is a power of two.
    Must be `k >= 0`.
    """
    if k < 0:
        raise ValueError(f'Expected k is non-negative: k={k}')
    return (k & (k - 1)) == 0

def float_to_bits(x: float) -> int:
    """Convert a Python float into a bistring."""
    if not isinstance(x, float):
        raise TypeError(f'Expected float x={x}')
    return int.from_bytes(struct.pack('@d', x), byteorder='little')

def bits_to_float(i: int) -> float:
    """Convert a bistring into a Python float."""
    if not isinstance(i, int):
        raise TypeError(f'Expected integer i={i}')
    if i < 0 or i >= 2 ** 64:
        raise ValueError(f'Expected i={i} on [0, 2 ** 64)')
    return struct.unpack('@d', i.to_bytes(8, byteorder='little'))[0]

