"""
This module defines compiler analyses over FPy IR.
"""

from .define_use import DefineUse
from .live_vars import LiveVars
from .reaching_defs import ReachingDefs
from .verify import VerifyIR
