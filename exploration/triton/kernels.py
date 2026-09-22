"""Hand-written Triton kernels pinning the numerics flags.

Findings are recorded under *What is implemented* in
`docs/todos/backend-triton.md`.

These are what the backend would have to emit for
:mod:`exploration.triton.programs`.  Writing them by hand first is the point of
§1: the flags get pinned against a real GPU before an emitter exists, and the
target text is known before anything has to generate it.

**Batch-lifted, not tensorized.**  One lane per dot product, N of them across a
tile; the fold over `K` stays sequential *within* a lane.  That is the shape
the roadmap argues for -- it reaches full parallelism while preserving FPy's
left-fold order exactly, where a tile reduction over `K` would reassociate and
change the answer of an inexact accumulation.

`triton` is imported lazily so the module is importable, and the programs
inspectable, on a machine with no GPU.
"""

from typing import Any

__all__ = ['BLOCK', 'available', 'kernels', 'why_unavailable']

BLOCK = 128
"""Lanes per program instance -- a power of two, as Triton requires.  The
backend would emit this as a `tl.constexpr` and leave the value to
`triton.autotune`; see the roadmap's note on `BLOCK`."""

_ERR: str | None = None
try:
    import triton
    import triton.language as tl
    available = True
except Exception as e:                        # pragma: no cover - env-dependent
    available = False
    _ERR = f'{type(e).__name__}: {e}'


def why_unavailable() -> str | None:
    """Why `triton` did not import, or `None` if it did."""
    return _ERR


kernels: dict[str, Any] = {}

if available:

    @triton.jit
    def dot_exact(xs_ptr, ys_ptr, out_ptr, n_rows,
                  K: tl.constexpr, BLOCK: tl.constexpr):
        """`programs.dot_exact_product`, compiled the way it must be.

        The two `.to(tl.float32)` casts are **mandatory and non-obvious**.  Both
        operands are fp16, and Triton's rule for `fp16 op fp16` is fp16 -- so
        `x * y` would compute the product *in fp16* and round it, when the
        program says the product is exact.  Nothing in Triton asks for the
        cast; it comes from `StorageInfer` giving the product fp32 storage,
        exactly as the C++ backend already emits `static_cast<double>`.
        """
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = row < n_rows
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(K):
            off = row * K + k
            x = tl.load(xs_ptr + off, mask=mask, other=0.0)
            y = tl.load(ys_ptr + off, mask=mask, other=0.0)
            acc = acc + x.to(tl.float32) * y.to(tl.float32)
        tl.store(out_ptr + row, acc, mask=mask)

    @triton.jit
    def dot_trap(xs_ptr, ys_ptr, out_ptr, n_rows,
                 K: tl.constexpr, BLOCK: tl.constexpr):
        """The same program with the cast in the wrong place.

        `(x * y).to(tl.float32)` multiplies in fp16 and widens the *rounded*
        product.  Kept as an executable witness that the trap is real rather
        than theoretical: if this agrees with `dot_exact` on the sampled
        inputs, the test data is too weak to distinguish them.
        """
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = row < n_rows
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(K):
            off = row * K + k
            x = tl.load(xs_ptr + off, mask=mask, other=0.0)
            y = tl.load(ys_ptr + off, mask=mask, other=0.0)
            acc = acc + (x * y).to(tl.float32)
        tl.store(out_ptr + row, acc, mask=mask)

    @triton.jit
    def dot_fp32(xs_ptr, ys_ptr, out_ptr, n_rows,
                 K: tl.constexpr, BLOCK: tl.constexpr):
        """`programs.dot_rounded_product`: fp32 throughout, so the product
        rounds and fma contraction is observable.  Run under both settings of
        `enable_fp_fusion`."""
        row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = row < n_rows
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(K):
            off = row * K + k
            x = tl.load(xs_ptr + off, mask=mask, other=0.0)
            y = tl.load(ys_ptr + off, mask=mask, other=0.0)
            acc = acc + x * y
        tl.store(out_ptr + row, acc, mask=mask)

    kernels = {
        'dot_exact': dot_exact,
        'dot_trap': dot_trap,
        'dot_fp32': dot_fp32,
    }
