"""
Format analysis integration tests.
"""

import fpy2 as fp

from fpy2.analysis import FormatInfer
from fpy2.transform import ConstFold

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


def _test_format_infer_unit():
    for core in all_tests():
        assert isinstance(core, fp.Function)
        if core.name in _unit_ignore:
            continue

        print(core.name)
        ast = ConstFold.apply(core.ast, enable_op=True)
        info = FormatInfer.analyze(ast)
        for d, fmt in info.by_def.items():
            print(f'  {d.name} -> {fmt}')


def _test_format_infer_library():
    for mod in _modules:
        for obj in mod.__dict__.values():
            match obj:
                case fp.Function():
                    ast = ConstFold.apply(obj.ast, enable_op=True)
                    print(obj.name)
                    info = FormatInfer.analyze(ast)
                    for d, fmt in info.by_def.items():
                        print(f'  {d.name} -> {fmt}')


def test_format_infer():
    _test_format_infer_unit()
    _test_format_infer_library()


if __name__ == '__main__':
    test_format_infer()
