"""
Tensor-argument size annotations round-tripping through FPCore.

The FPCore parser maps a dimensioned argument to a nested, sized
``ListTypeAnn`` (outermost dimension first), and the FPCore compiler reads
those sizes back out as tensor dimensions.
"""

import titanfp.fpbench.fpcparser as fpcparser

from fpy2 import Function, FPCoreCompiler
from fpy2.ast.fpyast import ListTypeAnn
from fpy2.analysis import DefineUse, TypeInfer
from fpy2.backend.fpc import _FPCoreCompileInstance
from fpy2.utils import NamedId


def _parse(src: str) -> Function:
    return Function.from_fpcore(fpcparser.compile1(src), ignore_unknown=True)


class TestFPCoreTensorSizes:
    def test_fixed_dims_parse_to_nested_sized_list(self):
        """``(A 3 4)`` -> ``list[list[any][4]][3]`` (outermost dim first)."""
        fun = _parse('(FPCore foo ((A 3 4)) :precision binary64 1.0)')
        ann = fun.ast.args[0].type
        assert isinstance(ann, ListTypeAnn) and ann.length == 3
        assert isinstance(ann.elt, ListTypeAnn) and ann.elt.length == 4

    def test_named_dims_parse_to_symbolic_lengths(self):
        """Named dims become symbolic lengths on the nested list."""
        fun = _parse('(FPCore foo ((B m n)) :precision binary64 1.0)')
        ann = fun.ast.args[0].type
        assert isinstance(ann, ListTypeAnn) and isinstance(ann.length, NamedId)
        assert ann.length.base == 'm'
        assert isinstance(ann.elt, ListTypeAnn) and isinstance(ann.elt.length, NamedId)
        assert ann.elt.length.base == 'n'

    def test_fixed_dims_round_trip(self):
        """FPCore -> FPy -> FPCore preserves fixed tensor dimensions."""
        fun = _parse('(FPCore foo ((A 3 4)) :precision binary64 1.0)')
        core = FPCoreCompiler().compile(fun)
        assert core.inputs == [('A', {}, [3, 4])]

    def test_named_dims_compile_to_dim_names(self):
        """The compiler emits named dimensions back as FPCore dim names.

        (The full named-dim body round-trip is blocked by an unrelated
        FPCore-backend limitation, so the argument compilation is checked
        directly.)"""
        fun = _parse('(FPCore foo ((A 3 4) (B m n)) :precision binary64 1.0)')
        inst = _FPCoreCompileInstance(fun.ast, DefineUse.analyze(fun.ast), TypeInfer.check(fun.ast))
        name, _, dims = inst._compile_arg(fun.ast.args[1])
        assert name == 'B' and dims == ['m', 'n']
