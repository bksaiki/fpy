"""
Triton backend: target description.

Compiles FPy to a ``@triton.jit`` kernel callable on torch tensors.  See
``docs/todos/backend-triton.md`` for the design and its staging; this package
currently holds §2, the target description, and no emitter.

The contract is the cpp backend's, held on a target that does not normally
hold it: if compilation succeeds, the emitted kernel must behave as the FPy
interpreter does.  A refusal is always acceptable; a different answer is not.
"""

from .storage import TritonStorageDomain, choose_storage, choose_storage_scalar
from .target import ScalarOpTable, TritonOp, TritonOpStyle, is_native_ctx, make_op_table
from .types import TritonScalar, TritonTuple, TritonType

__all__ = [
    'ScalarOpTable',
    'TritonOp',
    'TritonOpStyle',
    'TritonScalar',
    'TritonStorageDomain',
    'TritonTuple',
    'TritonType',
    'choose_storage',
    'choose_storage_scalar',
    'is_native_ctx',
    'make_op_table',
]
