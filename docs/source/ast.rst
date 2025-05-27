Abstract Syntax Tree (AST)
================================

Abstract Syntax Tree
---------------------------

The Abstract Syntax Tree (AST) is a representation of an FPy program.

.. automodule:: fpy2.ast.fpyast
   :members:
   :show-inheritance:

Visitors
---------------------------

.. automodule:: fpy2.ast.AstVisitor
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.DefaultAstVisitor
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.DefaultAstTransformVisitor
   :members:
   :show-inheritance:


Formatter
---------------------------

The AST can be formatted to a string representation using the :py:class:`fpy2.ast.Formatter` class.

.. automodule:: fpy2.ast.BaseFormatter
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.Formatter
   :members:
   :show-inheritance:

Analyses
---------------------------

.. autoclass:: fpy2.ast.DefineUse
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.LiveVars
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.SyntaxCheck
   :members:
   :show-inheritance:

Tranformations
---------------------------

.. autoclass:: fpy2.ast.ContextInline
   :members:
   :show-inheritance:
