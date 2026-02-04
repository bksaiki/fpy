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
        info = fp.analysis.ArraySizeInfer.infer(core.ast)

def _test_tcheck_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            match obj:
                case fp.Function():
                    info = fp.analysis.ArraySizeInfer.infer(obj.ast)

def test_context_infer():
    _test_tcheck_unit()
    _test_tcheck_library()

if __name__ == '__main__':
    test_context_infer()
