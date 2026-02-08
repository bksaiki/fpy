"""
Example functions for each operation.
"""

import fpy2 as fp

@fp.fpy
def test_logb(x: fp.Real) -> fp.Real:
    """Example function for `logb`."""
    return fp.logb(x)

@fp.fpy
def test_declcontext(x: fp.Real) -> list[fp.Real]:
    """Example function for `declcontext`."""
    with fp.declcontext(x):
        y = fp.round(x)
    return [x, y]

@fp.fpy
def test_empty1(n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(n)
    for i in range(n):
        arr[i] = i
    return arr

@fp.fpy
def test_empty2(m: fp.Real, n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(m, n)
    for i in range(m):
        for j in range(n):
            arr[i][j] = i * n + j
    return arr

@fp.fpy
def test_empty3(k: fp.Real, m: fp.Real, n: fp.Real):
    """Example function for `empty`."""
    arr = fp.empty(k, m, n)
    for i in range(k):
        for j in range(m):
            for l in range(n):
                arr[i][j][l] = (i * m + j) * n + l
    return arr
