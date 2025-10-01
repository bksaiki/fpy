Abstract Syntax Tree (AST)
================================

Abstract Syntax Tree
---------------------------

The Abstract Syntax Tree (AST) is a representation of an FPy program.

.. automodule:: fpy2.ast.fpyast
   :members:
   :show-inheritance:
   :exclude-members: Context

Visitors
---------------------------

.. automodule:: fpy2.ast.Visitor
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.DefaultVisitor
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ast.DefaultTransformVisitor
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

.. autoclass:: fpy2.analysis.ContextInfer
   :members:
   :show-inheritance:

.. autoclass:: fpy2.analysis.DefineUse
   :members:
   :show-inheritance:

.. autoclass:: fpy2.analysis.LiveVars
   :members:
   :show-inheritance:

.. autoclass:: fpy2.analysis.Purity
   :members:
   :show-inheritance:

.. autoclass:: fpy2.analysis.Reachability
   :members:
   :show-inheritance:

.. autoclass:: fpy2.analysis.SyntaxCheck
   :members:
   :show-inheritance:

.. autoclass:: fpy2.analysis.TypeInfer
   :members:
   :show-inheritance:

Tranformations
---------------------------

.. automodule:: fpy2.transform.ConstFold
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.ConstPropagate
   :members:
   :show-inheritance:

.. automodule:: fpy2.transform.CopyPropagate
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.DeadCodeEliminate
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.ForBundling
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.ForUnpack
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.FuncUpdate
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.IfBundling
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.RenameTarget
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.SimplifyIf
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.SubstVar
   :members:
   :show-inheritance:

.. autoclass:: fpy2.transform.WhileBundling
   :members:
   :show-inheritance:
