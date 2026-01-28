"""
Partial evaluation tests
"""

import fpy2 as fp

from fpy2.analysis import PartialEval

from ..examples import all_tests


_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

_unit_ignore = [
    'test_context_expr1',
    'test_context_expr2',
    'test_context8',
    'example_static_context1',
    'example_static_context2',
    'keep_p_1'
]

def _test_analysis_unit():
    for core in all_tests():
        assert isinstance(core, fp.Function)
        if core.name in _unit_ignore:
            continue

        info = PartialEval.apply(core.ast)
        print(core.name)
        for e, v in info.by_expr.items():
            print(f' E[{e.format()}] = {v}')
        for d, v in info.by_def.items():
            if not isinstance(v, fp.Function | fp.Primitive):
                print(f' {d.name} = {v}')

def _test_analysis_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            match obj:
                case fp.Function():
                    info = PartialEval.apply(obj.ast)
                    print(obj.name)
                    for e, v in info.by_expr.items():
                        print(f' E[{e.format()}] = {v}')
                    for d, v in info.by_def.items():
                        if not isinstance(v, fp.Function | fp.Primitive):
                            print(f' {d.name} = {v}')

def test_partial_eval():
    _test_analysis_unit()
    _test_analysis_library()

if __name__ == '__main__':
    test_partial_eval()

