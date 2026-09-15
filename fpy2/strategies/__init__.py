"""
Scheduling language constructs for FPy programs.

A strategy with sites takes a `where`: an index into its candidates, or a cursor.
A cursor or region selects every candidate *at or beneath* the program point it
names, so the statement an earlier rewrite left behind names the site now nested
inside it; one from an earlier program is forwarded to this one first.
:func:`sites` lists what a `where` may name.

**What a rounding site is.**  The rounding rewrites are aimed at the rounding
itself, not at the block it sits in: a ``fp.round`` -- or a ``fp.cast``, where
the rewrite takes one -- is a site wherever the context active there is one that
rewrite can restate.  Under a ``with``, under the function's own annotation, or
beside statements rounding to something else: all the same to it.  An operand
that is not already a name is bound to one first, in the scope it was written
in, which is the scope that already rounded it.  A position with no statement
slot for what the rewrite emits -- a ternary arm, a comprehension element -- is
refused, and :func:`to_hoistable` gives it one.

**Pinning a point across a sequence.**  These rewrites emit *into* the block
they found the rounding in rather than replacing it, so a ``with`` left holding
no rounding goes when :func:`simplify` runs.  To aim a whole sequence at one
program point, pin the *statement* holding the rounding rather than the
expression :func:`sites` reports: each rewrite consumes the rounding it acts on,
so that expression names nothing afterwards, while the statement survives with
what replaced it beneath.
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
from .iter_unfold import unfold_enumerate, unfold_zip
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
    'unfold_enumerate',
    'unfold_neg_zero',
    'unfold_overflow',
    'unfold_special',
    'unfold_zip',
    'unroll_for',
    'unroll_while',
]
