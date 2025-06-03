"""
This module defines compiler transforms over FPy IR.
"""

from .copy_propagate import CopyPropagate
from .context_inline import ContextInline
from .for_bundling import ForBundling
from .for_unpack import ForUnpack
from .func_update import FuncUpdate
from .if_bundling import IfBundling
from .rename_target import RenameTarget
from .simplify_if import SimplifyIf
from .while_bundling import WhileBundling
