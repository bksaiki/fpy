"""
Typing hints for FPy programs.

FPy programs are re-parsed and analyzing with the `@fpy` decorator.
Thus, functions in FPy programs don't actually exist in the runtime.

To help static analyzers like Pylance and mypy,
this module provides a stub file with typing hints
for FPy functions.

Use the hints:

>>  > from fpy2.typing import *

The names in this module must be imported directly into
the importing namespace.
"""

class Real(object):
    """Virtual object representing any real FPy term."""
