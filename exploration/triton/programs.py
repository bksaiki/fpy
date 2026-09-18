"""The two FPy programs the Triton spike rests on.

Both are dot products.  They differ in one respect, and that respect is the
whole point: whether the *product* is exact at the storage the pipeline picks
for it.

- :func:`dot_exact_product` — FP16 arguments, products under ``REAL``,
  accumulation under FP32.  FP16 carries prec 11, so a product needs 22 bits
  against fp32's 24: it is exact, and ``StorageInfer`` puts it in fp32.
- :func:`dot_rounded_product` — FP32 arguments and FP32 arithmetic throughout.
  A product needs prec 48 against fp32's 24, so it rounds.

The pair discriminates fused multiply-add.  Contracting ``acc + x * y`` into
an ``fma`` computes the product exactly and rounds once; the unfused form
rounds the product and then rounds the sum.  Where the product is already
exact those are the same operation, so contraction is unobservable — and where
it is not, they differ, and contraction changes the answer FPy specifies.

That predicate is not a new analysis.  It is ``scalar_fits_in(product_format,
product_storage)``, which the pipeline already computes.  See *What is implemented* in
``docs/todos/backend-triton.md``.
"""

import fpy2 as fp
from fpy2.types import ListType, RealType

__all__ = [
    'K',
    'arg_types_fp16',
    'arg_types_fp32',
    'dot_exact_product',
    'dot_exact_product_fma',
    'dot_rounded_product',
    'dot_rounded_product_fma',
]

K = 8
"""Vector length.  Constant so ``Specialize(size_key=True)`` proves it, which
is what makes it a ``tl.constexpr`` on the Triton side."""


@fp.fpy(ctx=fp.REAL)
def dot_exact_product(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    """FP16 in, exact products, FP32 accumulation.

    The `with` block is the only rounding: `p` is exact, so the emitted
    product must be computed at a width that holds it.  A Triton emitter that
    spells this `x * y` on two fp16 operands computes it *in fp16* and rounds
    it -- see `kernels.dot_trap`.
    """
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        p = x * y                 # REAL: exact
        with fp.FP32:
            acc = acc + p         # rounded to FP32
    return acc


@fp.fpy(ctx=fp.FP32)
def dot_rounded_product(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    """FP32 throughout: the product rounds, so fma contraction is observable."""
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        acc = acc + x * y
    return acc


@fp.fpy(ctx=fp.REAL)
def dot_exact_product_fma(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    """`dot_exact_product` with the multiply-add contracted.

    Expected to agree with it on **every** input: the product is exact, so
    "round the product, then round the sum" and "round the sum of the exact
    product" are the same operation.  A disagreement would falsify the premise
    that `enable_fp_fusion` is safe for this program.
    """
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        with fp.FP32:
            acc = fp.fma(x, y, acc)
    return acc


@fp.fpy(ctx=fp.FP32)
def dot_rounded_product_fma(xs: list[fp.Real], ys: list[fp.Real]) -> fp.Real:
    """`dot_rounded_product` contracted.  Expected to *disagree* on some
    inputs -- which is what makes `enable_fp_fusion=False` load-bearing."""
    acc = fp.round(0)
    for x, y in zip(xs, ys):
        acc = fp.fma(x, y, acc)
    return acc


arg_types_fp16 = [ListType(RealType(fp.FP16), K), ListType(RealType(fp.FP16), K)]
arg_types_fp32 = [ListType(RealType(fp.FP32), K), ListType(RealType(fp.FP32), K)]
