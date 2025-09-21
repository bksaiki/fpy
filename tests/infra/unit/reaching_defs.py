"""
Tests for reaching definitions.
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
        defs = fp.analysis.ReachingDefs.analyze(test.ast)
        print(defs.format())

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                defs = fp.analysis.ReachingDefs.analyze(obj.ast)
                print('dead_code', obj.name)
                print(defs.format())

def test_reaching_defs():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_reaching_defs()
