"""
This module defines compiler transforms over FPy IR.
"""

from .copy_propagate import CopyPropagate
from .const_fold import ConstFold
from .const_prop import ConstPropagate
from .dead_code import DeadCodeEliminate
from .for_bundling import ForBundling
from .for_unpack import ForUnpack
from .for_unroll import ForUnroll, ForUnrollStrategy
from .func_inline import FuncInline
from .func_update import FuncUpdate
from .if_bundling import IfBundling
from .monomorphize import Monomorphize
from .rename_target import RenameTarget
from .simplify_if import SimplifyIf
from .split_loop import SplitLoop, SplitLoopStrategy
from .subst_var import SubstVar
from .while_bundling import WhileBundling
from .while_unroll import WhileUnroll
