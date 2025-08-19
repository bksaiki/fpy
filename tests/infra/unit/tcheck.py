"""
Type checking tests
"""

import fpy2 as fp

from fpy2.analysis import TypeCheck

from .defs import tests, examples

_modules = [
    fp.libraries.core
]

def _test_tcheck_unit():
    for core in tests + examples:
        assert isinstance(core, fp.Function)
        info = TypeCheck.check(core.ast)
        print(core.name, info.fn_type)

def _test_tcheck_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                info = TypeCheck.check(obj.ast)
                print(obj.name, info.fn_type)

def test_tcheck():
    _test_tcheck_unit()
    _test_tcheck_library()

if __name__ == '__main__':
    test_tcheck()
