"""
Tests for dead code elimination.
"""

import fpy2 as fp

from .unit_tests import tests, examples

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]


def _test_unit():
    for test in tests + examples:
        print('dead_code', test.name)
        fn = fp.transform.DeadCodeEliminate.apply(test.ast)
        print(fn.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                fn = fp.transform.DeadCodeEliminate.apply(obj.ast)
                print('dead_code', obj.name)
                print(fn.format())

def test_dead_code():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_dead_code()
