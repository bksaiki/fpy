"""
Tests for dead code elimination.
"""

import fpy2 as fp

from .defs import tests, examples

def _test_unit():
    for test in tests + examples:
        fn = fp.transform.DeadCodeEliminate.apply(test.ast)
        print('dead_code', test.name)
        print(fn.format())

def test_dead_code():
    _test_unit()

if __name__ == '__main__':
    test_dead_code()
