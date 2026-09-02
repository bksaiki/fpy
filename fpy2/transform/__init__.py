"""
This module defines compiler transforms over FPy IR.
"""

from .anf import ANF
from .comp_to_loop import CompToLoop
from .const_fold import ConstFold
from .copy_propagate import CopyPropagate
from .cursor import (
    BlockCursor,
    Cursor,
    Edit,
    EditLog,
    ExprCursor,
    StmtCursor,
    contains,
)
from .dead_code import DeadCodeEliminate
from .enumerate_elim import EnumerateElim
from .error import TransformDeclined, TransformError, TransformReferenceError
from .float_to_fixed import FloatToFixed
from .for_bundling import ForBundling
from .for_unpack import ForUnpack
from .for_unroll import ForUnroll, ForUnrollStrategy
from .free_var_elim import FreeVarElim
from .func_inline import FuncInline
from .hoistable import Hoistable
from .if_bundling import IfBundling
from .lift_context import LiftContext
from .monomorphize import Monomorphize
from .path import (
    BlockPath,
    ExprPath,
    FuncBody,
    StmtPath,
    SubBlock,
    walk_blocks,
    walk_exprs,
)
from .reduce_fusion import ReduceFusion
from .rename_target import RenameTarget
from .rescale_fixed import RescaleFixed
from .round_elim import RoundElim
from .round_insert import RoundInsert
from .simplify import Simplify
from .simplify_if import SimplifyIf
from .specialize import Specialize
from .split_loop import SplitLoop, SplitLoopStrategy
from .split_round import SplitRound
from .subst_var import SubstVar
from .unfold_neg_zero import UnfoldNegZero
from .unfold_overflow import UnfoldOverflow
from .unfold_special import UnfoldSpecial
from .utils import SiteRewriter, check_where, clone
from .while_bundling import WhileBundling
from .while_unroll import WhileUnroll
from .zip_elim import ZipElim
