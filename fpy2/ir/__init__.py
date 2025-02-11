"""
This module contains the intermediate representation (IR)
and the visitor of the FPy compiler.
"""

from . import ir

from .formatter import Formatter
from .ir import *
from .types import *
from .visitor import *

ir.set_default_formatter(Formatter())
