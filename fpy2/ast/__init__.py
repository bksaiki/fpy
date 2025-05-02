"""
Abstract Syntax Tree (AST) for the FPy language.
"""

from .formatter import Formatter
from .fpyast import set_default_formatter
from .visitor import AstVisitor

set_default_formatter(Formatter())
