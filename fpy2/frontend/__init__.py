"""
Frontend for the FPy language.

This module contains a parser, syntax checker, and type checking
for the FPy language.
"""

from .codegen import IRCodegen
from .decorator import fpy
from .fpc import fpcore_to_fpy
from .syntax_check import SyntaxCheck
