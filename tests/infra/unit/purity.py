"""
Tests for function purity analysis.
"""

import fpy2 as fp

from .unit_tests import tests, examples

_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]


@fp.fpy
def f(x: list[fp.Real]):
    x[0] = 1.0
    return x

@fp.fpy
def g(x: list[fp.Real]) -> fp.Real:
    i = f(x)
    return i[0]


def _test_example():
    is_pure = fp.analysis.Purity.analyze(f.ast)
    assert not is_pure, f'Expected impure, got {is_pure} for {f}'
    is_pure = fp.analysis.Purity.analyze(g.ast)
    assert not is_pure, f'Expected impure, got {is_pure} for {g}'


def _test_unit():
    for core in tests + examples:
        is_pure = fp.analysis.Purity.analyze(core.ast)
        print('purity', core.name, is_pure)

def _test_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                is_pure = fp.analysis.Purity.analyze(obj.ast)
                print('purity', obj.name, is_pure)


def test_purity():
    _test_example()
    _test_unit()
    _test_library()

if __name__ == '__main__':
    test_purity()
