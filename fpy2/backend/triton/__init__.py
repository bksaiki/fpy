"""
Triton backend: compiles FPy to a ``@triton.jit`` kernel on torch tensors.

If compilation succeeds, the kernel behaves as the interpreter does;
otherwise it refuses.
"""

from .compiler import TritonCompiler
from .emitter import (
    KernelSource,
    TritonEmitError,
    emit_block,
    emit_expr,
    emit_kernel,
)
from .launcher import launch, load_kernel, unavailable
from .normalize import TritonNormalizeError, normalize, normalize_module
from .storage import TritonStorageDomain, choose_storage_scalar
from .target import ScalarOpTable, TritonOp, TritonOpStyle, is_native_ctx, make_op_table
from .types import TritonScalar
from .vectorize import TileResult, tile_loops, why_not_tileable

__all__ = [
    'KernelSource',
    'ScalarOpTable',
    'TileResult',
    'TritonCompiler',
    'TritonEmitError',
    'TritonNormalizeError',
    'TritonOp',
    'TritonOpStyle',
    'TritonScalar',
    'TritonStorageDomain',
    'choose_storage_scalar',
    'emit_block',
    'emit_expr',
    'emit_kernel',
    'is_native_ctx',
    'launch',
    'load_kernel',
    'make_op_table',
    'normalize',
    'normalize_module',
    'tile_loops',
    'unavailable',
    'why_not_tileable',
]
