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
        print('while_unroll', test.name)
        print('unroll (0x)')
        fn = fp.transform.WhileUnroll.apply(test.ast, times=0)
        print(fn.format())
        print('unroll (1x)')
        fn = fp.transform.WhileUnroll.apply(test.ast, times=1)
        print(fn.format())
        print('unroll (2x)')
        fn = fp.transform.WhileUnroll.apply(test.ast, times=2)
        print(fn.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                print('while_unroll', obj.name)
                print('unroll (1x)')
                fn = fp.transform.WhileUnroll.apply(obj.ast, times=0)
                print(fn.format())
                print('unroll (2x)')
                fn = fp.transform.WhileUnroll.apply(obj.ast, times=1)
                print(fn.format())
                print('unroll (3x)')
                fn = fp.transform.WhileUnroll.apply(obj.ast, times=2)
                print(fn.format())

def test_while_unroll():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_while_unroll()

