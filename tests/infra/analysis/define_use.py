"""
Tests for define-use analysis
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
        print('define_use', test.name)
        defs = fp.analysis.DefineUse.analyze(test.ast)
        print(defs.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                defs = fp.analysis.DefineUse.analyze(obj.ast)
                print('define_use', obj.name)
                print(defs.format())

def test_define_use():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_define_use()
