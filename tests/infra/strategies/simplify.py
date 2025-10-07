"""
Tests for `simplify` strategy.
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
        print('simplify', test.name)
        test = fp.strategies.simplify(test)
        print(test.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                print('simplify', obj.name)
                obj = fp.strategies.simplify(obj)
                print(obj.format())

def test_simplify():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_simplify()
