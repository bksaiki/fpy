"""`StorageInfer` against the Triton domain, with no emitter.

This is what the target description is *for*, and the roadmap's claim that §2
is testable on its own: given the ladder, the real analyses decide what holds
each value, and the answers are checkable without generating a line of Triton.

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
from fpy2.analysis.storage_infer import StorageInfer
from fpy2.backend.triton.storage import TritonStorageDomain, to_triton
from fpy2.backend.triton.types import TritonScalar as T
from fpy2.module import Module
from fpy2.transform import FreeVarElim, Specialize
from fpy2.types import ListType, RealType

K = 4


@fp.fpy(ctx=fp.REAL)
def dot(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        p = x * y                 # REAL: exact
        with fp.FP32:
            acc = acc + p         # rounded to FP32
    return acc


def _storage(func, arg_types, ctx=fp.FP32):
    """Every expression's Triton storage, keyed by source text.

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
    return chosen


def _scalars(chosen):
    """The scalar storages chosen, as a multiset of dtype names."""
    out = []
    for fmt in chosen.class_storage.values():
        try:
            ty = to_triton(fmt)
        except Exception:
            continue
        if isinstance(ty, T):
            out.append(ty)
    return out


class TestTheRunningExample:
    def test_fp16_arguments_stay_fp16(self):
        """The point of having an fp16 rung at all: the cpp backend widens
        these parameters to `float` at the boundary because it cannot spell
        fp16, so its kernel cannot take an fp16 buffer."""
        chosen = _storage(
            dot, [ListType(RealType(fp.FP16), K)] * 2,
        )
        assert T.F16 in _scalars(chosen)

    def test_the_exact_product_gets_fp32(self):
        """FP16 carries prec 11, so a product needs 22 bits against fp32's 24.
        The analysis must place it a rung up -- and exactly one rung up, since
        fp64 would be correct but wasteful."""
        chosen = _storage(dot, [ListType(RealType(fp.FP16), K)] * 2)
        got = _scalars(chosen)
        assert T.F32 in got
        assert T.F64 not in got

    def test_fp32_arguments_force_fp64_for_the_product(self):
        """The contrast that shows the previous test measured something: a
        product of two fp32 values needs prec 48, which no fp32 holds."""
        chosen = _storage(
            dot, [ListType(RealType(fp.FP32), K)] * 2, ctx=fp.FP64,
        )
        assert T.F64 in _scalars(chosen)


class TestListsAreRefused:
    def test_a_list_storage_has_no_spelling(self):
        """Triton has no list type.  Refusing names the reason; spelling one as
        a tile would silently change what the program means."""
        from fpy2.analysis.format_infer import ListFormat
        from fpy2.analysis.storage_infer import StorageSelectionError
        with pytest.raises(StorageSelectionError, match='no list storage'):
            to_triton(ListFormat(fp.FP32.format()))
