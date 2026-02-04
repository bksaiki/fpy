"""
Array size inference tests
"""

import fpy2 as fp


from ..examples import all_tests


_modules = [
    fp.libraries.core,
    fp.libraries.eft,
    fp.libraries.vector,
    fp.libraries.matrix
]

def _test_tcheck_unit():
    for core in all_tests():
        assert isinstance(core, fp.Function)
        print(core.name)
        info = fp.analysis.ArraySizeInfer.infer(core.ast)
        for e, v in info.by_expr.items():
            if isinstance(v, fp.analysis.array_size._Array):
                print(f'  {e.format()} : {v}')

def _test_tcheck_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            if isinstance(obj, fp.Function):
                print(obj.name)
                info = fp.analysis.ArraySizeInfer.infer(obj.ast)
                for e, v in info.by_expr.items():
                    if isinstance(v, fp.analysis.array_size._Array):
                        print(f'  {e.format()} : {v}')

def test_array_size_infer():
    _test_tcheck_unit()
    _test_tcheck_library()

if __name__ == '__main__':
    test_array_size_infer()
