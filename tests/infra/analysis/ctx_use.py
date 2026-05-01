"""
Tests for context-use analysis
"""

import fpy2 as fp

from fpy2.analysis import CtxUse

from ..examples import all_tests


_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

def _print_analysis(name: str, func: fp.ast.FuncDef):
    info = CtxUse.analyze(func)
    print(name)
    for scope in info.scopes:
        site = type(scope.site).__name__
        print(f' scope at {site}: ctx={scope.ctx}, uses={len(info.uses[scope])}')

def _test_analysis_unit():
    for core in all_tests():
        assert isinstance(core, fp.Function)
        _print_analysis(core.name, core.ast)

def _test_analysis_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                _print_analysis(obj.name, obj.ast)

def test_ctx_use():
    _test_analysis_unit()
    _test_analysis_library()

if __name__ == '__main__':
    test_ctx_use()
