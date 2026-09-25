"""`StorageInfer` against the Triton domain, with no emitter.

Given the ladder, the real analyses decide what holds each value, and the
answers are checkable without generating a line of Triton.

The program is the backend's running example: a batched dot product.
"""

import pytest

import fpy2 as fp
from fpy2.analysis import (
    Alias,
    ArraySizeInfer,
    ContextUse,
    DefineUse,
    FormatInfer,
    ValueClassInfer,
)
from fpy2.analysis.format_infer import ListFormat
from fpy2.analysis.storage_infer import StorageInfer, StorageSelectionError
from fpy2.backend.triton.storage import TritonStorageDomain, to_triton
from fpy2.backend.triton.types import TritonScalar as T
from fpy2.module import Module
from fpy2.transform import FreeVarElim, Specialize
from fpy2.types import ListType, RealType, Type

K = 4


@fp.fpy(ctx=fp.REAL)
def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        p = x * y                 # REAL: exact
        with fp.FP32:
            acc = acc + p         # rounded to FP32
    return acc


def _storage(
    func: fp.Function, arg_types: list[Type], ctx: fp.Context = fp.FP32,
) -> dict[str, set[T]]:
    """The Triton storages of each scalar variable's definitions.

    The pipeline up to `StorageInfer`, with the Triton domain swapped in for
    the cpp one -- which is the whole of what a `StorageDomain` contributes.
    """
    m = Module()
    m.add(func, ctx=ctx, arg_types=arg_types)
    m = m.map(lambda _m, fd: FreeVarElim.apply(fd))
    spec = Specialize.apply(m, size_key=True)
    ast = list(spec.call_graph().order)[-1].ast

    def_use = DefineUse.analyze(ast)
    ctx_use = ContextUse.analyze(ast, def_use=def_use)
    array_size = ArraySizeInfer.analyze(ast)
    fmt = FormatInfer.analyze(
        ast, def_use=def_use, ctx_use=ctx_use, array_size=array_size,
    )
    alias = Alias.analyze(ast, def_use=def_use)
    cls = ValueClassInfer.analyze(
        ast, def_use=def_use, type_info=fmt.type_info,
        ctx_use=ctx_use, alias=alias,
    )
    du = fmt.type_info.def_use
    chosen = StorageInfer.infer(
        du, fmt.by_def, fmt.by_expr, TritonStorageDomain(),
        {d: cls.bound_of(d) for d in du.defs},
    )
    by_name: dict[str, set[T]] = {}
    for d in du.defs:
        if not isinstance(storage := chosen.storage_of(d), ListFormat):
            by_name.setdefault(str(d.name), set()).add(to_triton(storage))
    return by_name


class TestTheRunningExample:
    def test_fp16_arguments_stay_fp16(self):
        """The point of having an fp16 rung at all: the cpp backend widens
        these parameters to `float` at the boundary because it cannot spell
        fp16, so its kernel cannot take an fp16 buffer."""
        got = _storage(dot, [ListType(RealType(fp.FP16), K)] * 2)
        assert got['x'] == got['y'] == {T.F16}

    def test_the_exact_product_gets_fp32(self):
        """FP16 carries prec 11, so a product needs 22 bits against fp32's 24.
        The analysis must place it a rung up -- and exactly one rung up, since
        fp64 would be correct but wasteful."""
        got = _storage(dot, [ListType(RealType(fp.FP16), K)] * 2)
        assert got['p'] == {T.F32}
        assert T.F64 not in set().union(*got.values())

    def test_fp32_arguments_force_fp64_for_the_product(self):
        """The contrast: a product of two fp32 values needs prec 48, which no
        fp32 holds."""
        got = _storage(dot, [ListType(RealType(fp.FP32), K)] * 2, ctx=fp.FP64)
        assert got['p'] == {T.F64}


class TestListsAreRefused:
    def test_a_list_storage_has_no_spelling(self):
        """Triton has no list type.  Refusing names the reason; spelling one as
        a tile would silently change what the program means."""
        with pytest.raises(StorageSelectionError, match='no list storage'):
            to_triton(ListFormat(fp.FP32.format()))
