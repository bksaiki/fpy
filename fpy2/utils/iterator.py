"""
This module defines additional iterators.
"""

import collections
import itertools
from collections.abc import Iterable, Iterator


def sliding_window[T](iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) → ABCD BCDE CDEF DEFG
    iterator = iter(iterable)
    window = collections.deque(itertools.islice(iterator, n - 1), maxlen=n)
    for x in iterator:
        window.append(x)
        yield tuple(window)
