"""
Abstract Syntax Tree (AST) for the FPy language.
"""

from .fpyast import *
from .formatter import Formatter, BaseFormatter
from .visitor import AstVisitor, DefaultAstVisitor, DefaultAstTransformVisitor

set_default_formatter(Formatter())
