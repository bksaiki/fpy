"""
Abstract Syntax Tree (AST) for the FPy language.
"""

from .fpyast import *
from .formatter import Formatter
from .fpyast import set_default_formatter
from .visitor import AstVisitor, DefaultAstVisitor

set_default_formatter(Formatter())
