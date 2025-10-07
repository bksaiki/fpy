"""
Tests for split loop.
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
        print('split_loop', test.name)
        fn = fp.transform.SplitLoop.apply(test.ast, fp.ast.Integer(4, None))
        print(fn.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                print('split_loop', obj.name)
                fn = fp.transform.SplitLoop.apply(obj.ast, fp.ast.Integer(4, None))
                print(fn.format())

def test_split_loop():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_split_loop()

