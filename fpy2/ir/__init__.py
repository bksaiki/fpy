"""
This module contains the intermediate representation (IR)
and the visitor of the FPy compiler.
"""

from .formatter import Formatter, set_default_formatter
from .ir import *
from .types import *
from .visitor import *

set_default_formatter(Formatter())
