"""
Scheduling language constructs for FPy programs.

A strategy with sites takes a `where`: an index into its candidates, or a cursor.
A cursor or region selects every candidate *at or beneath* the program point it
names, so the statement an earlier rewrite left behind names the site now nested
inside it; one from an earlier program is forwarded to this one first.
:func:`sites` lists what a `where` may name.
"""

from ..transform import (
    BlockCursor,
    BlockPath,
    Cursor,
    ExprCursor,
    ExprPath,
    FuncBody,
    StmtCursor,
    StmtPath,
    SubBlock,
    TransformDeclined,
    TransformError,
    TransformReferenceError,
)
from .anf import to_anf
from .comp_lower import comp_to_loop
from .context_lift import lift_context
from .fixed_rescale import rescale_fixed
from .float_lower import float_to_fixed
from .free_var import close
from .func_inline import inline
from .hoistable import to_hoistable
from .iter_elim import elim_iter
from .loop_split import split
from .loop_unroll import unroll_for, unroll_while
from .mono import monomorphize
from .neg_zero_unfold import unfold_neg_zero
from .overflow_unfold import unfold_overflow
from .reduce_fusion import fuse
from .round_elim import elim_round
from .round_insert import insert_round
from .round_split import split_round
from .simple import simplify
from .sites import refusals, sites
from .special_unfold import unfold_special

__all__ = [
    'BlockCursor',
    'BlockPath',
    'Cursor',
    'ExprCursor',
    'ExprPath',
    'FuncBody',
    'StmtCursor',
    'StmtPath',
    'SubBlock',
    'TransformDeclined',
    'TransformError',
    'TransformReferenceError',
    'close',
    'comp_to_loop',
    'elim_iter',
    'elim_round',
    'float_to_fixed',
    'fuse',
    'inline',
    'insert_round',
    'lift_context',
    'monomorphize',
    'refusals',
    'rescale_fixed',
    'simplify',
    'sites',
    'split',
    'split_round',
    'to_anf',
    'to_hoistable',
    'unfold_neg_zero',
    'unfold_overflow',
    'unfold_special',
    'unroll_for',
    'unroll_while',
]
