"""
Tests for while unrolling.
"""

import fpy2 as fp

from ..examples import all_tests

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]


def _test_unit():
    for test in all_tests():
        print('for_unroll', test.name)
        print('unroll (1x)')
        fn = fp.transform.ForUnroll.apply(test.ast, times=0)
        print(fn.format())
        print('unroll (2x)')
        fn = fp.transform.ForUnroll.apply(test.ast, times=1)
        print(fn.format())
        print('unroll (4x)')
        fn = fp.transform.ForUnroll.apply(test.ast, times=3)
        print(fn.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                print('for_unroll', obj.name)
                print('unroll (1x)')
                fn = fp.transform.ForUnroll.apply(obj.ast, times=0)
                print(fn.format())
                print('unroll (2x)')
                fn = fp.transform.ForUnroll.apply(obj.ast, times=1)
                print(fn.format())
                print('unroll (4x)')
                fn = fp.transform.ForUnroll.apply(obj.ast, times=3)
                print(fn.format())


def test_for_unroll():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_for_unroll()

