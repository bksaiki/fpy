"""
Abstract Syntax Tree (AST) for the FPy language.
"""

from .fpyast import *
from .formatter import Formatter
from .fpyast import set_default_formatter
from .visitor import AstVisitor, DefaultAstVisitor, DefaultAstTransformVisitor

from .context_inline import ContextInline
from .define_use import DefineUse
from .live_vars import LiveVars
from .syntax_check import SyntaxCheck

set_default_formatter(Formatter())
