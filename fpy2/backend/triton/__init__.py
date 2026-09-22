"""
Triton backend: target description.

Compiles FPy to a ``@triton.jit`` kernel callable on torch tensors.  This
package currently holds the normal form and the target description; there is
no emitter yet.

The contract is the cpp backend's, held on a target that does not normally
hold it: if compilation succeeds, the emitted kernel must behave as the FPy
interpreter does.  A refusal is always acceptable; a different answer is not.
"""

from .emitter import TritonEmitError, emit_block, emit_expr
from .normalize import TritonNormalizeError, normalize, normalize_module
from .storage import TritonStorageDomain, choose_storage, choose_storage_scalar
from .target import ScalarOpTable, TritonOp, TritonOpStyle, is_native_ctx, make_op_table
from .types import TritonScalar, TritonTuple, TritonType
from .vectorize import TileResult, tile_loops, why_not_tileable

__all__ = [
    'ScalarOpTable',
    'TileResult',
    'TritonEmitError',
    'TritonNormalizeError',
    'TritonOp',
    'TritonOpStyle',
    'TritonScalar',
    'TritonStorageDomain',
    'TritonTuple',
    'TritonType',
    'choose_storage',
    'choose_storage_scalar',
    'emit_block',
    'emit_expr',
    'is_native_ctx',
    'make_op_table',
    'normalize',
    'normalize_module',
    'tile_loops',
    'why_not_tileable',
]
