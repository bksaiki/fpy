"""
Tests for definition analysis
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
        print('dead_code', test.name)
        defs = fp.analysis.DefAnalysis.analyze(test.ast)
        print(defs)

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                print('dead_code', obj.name)
                defs = fp.analysis.DefAnalysis.analyze(obj.ast)
                print(defs)

def test_defs():
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_defs()
