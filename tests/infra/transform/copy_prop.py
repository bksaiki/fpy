"""
Tests for copy propagation.
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
        print('copy_prop', test.name)
        fn = fp.transform.CopyPropagate.apply(test.ast)
        print(fn.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                fn = fp.transform.CopyPropagate.apply(obj.ast)
                print('copy_prop', obj.name)
                print(fn.format())

def test_copy_prop():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_copy_prop()
