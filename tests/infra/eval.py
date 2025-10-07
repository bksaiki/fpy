import fpy2 as fp

from .examples import all_tests

def _gen_arg(ty: fp.ast.TypeAnn, length=3):
    match ty:
        case None | fp.ast.AnyTypeAnn() | fp.ast.RealTypeAnn():
            # assume real for `None` and `Any`
            return 1.0
        case fp.ast.ListTypeAnn():
            return [_gen_arg(ty.elt, length) for _ in range(length)]
        case fp.ast.TupleTypeAnn():
            return tuple(_gen_arg(a, length) for a in ty.elts)
        case _:
            raise NotImplementedError(f'Unsupported type annotation: {ty}')


def test_eval():
    for core in all_tests():
        fn = core
        args = [_gen_arg(a.type) for a in core.args]
        print(core.name)
        print('=>', fn(*args))

if __name__ == '__main__':
    test_eval()
