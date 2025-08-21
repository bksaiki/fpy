"""
FPy standard library.

- base: FPy builtins 
- core: core numerical operations and utilities
- eft: error-free transformations
- vector: vector operation
"""

###########################################################
# Sublibraries

# base library (auto-imported into the `fpy2` module)
from . import base

# core operations and utilities
from . import core

# error-free transformations
from . import eft

# vectors
from . import vector

# matrices
from . import matrix
