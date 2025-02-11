"""
Frontend for the FPy language.

This module contains a parser, syntax checker, and type checking
for the FPy language.
"""

from . import fpyast as fpyast

from .decorator import fpy
from .formatter import Formatter

fpyast.set_default_formatter(Formatter())
