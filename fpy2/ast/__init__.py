"""
Abstract Syntax Tree (AST) for the FPy language.
"""

from .formatter import BaseFormatter, Formatter
from .fpyast import *
from .visitor import DefaultTransformVisitor, DefaultVisitor, Visitor

set_default_formatter(Formatter())
