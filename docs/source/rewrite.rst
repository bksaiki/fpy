Rewriter
====================

Patterns
--------------------

FPy provides a pattern-based rewriter for transforming FPy programs.
Patterns are specified using the :py:deco:`fpy2.pattern` decorator

.. autofunction:: fpy2.pattern

.. autoclass:: fpy2.rewrite.Pattern
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.ExprPattern
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.StmtPattern
   :members:
   :show-inheritance:

Matcher
--------------------

Given a :py:class:`fpy2.Pattern` instance,
the :py:class:`fpy2.rewrite.Matcher` class the locations within
an FPy program where the pattern matches.

.. autoclass:: fpy2.rewrite.Matcher
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.LocatedMatch
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.ExprMatch
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.StmtMatch
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.Subst
   :members:
   :show-inheritance:

Applier
--------------------

Given a :py:class:`fpy2.Pattern` instance,
the :py:class:`fpy2.rewrite.Applier` class applies a substitution,
a mapping from pattern variable to syntax, to produce a new FPy program.

.. autoclass:: fpy2.rewrite.Applier
   :members:
   :show-inheritance:

Rewrite
--------------------

The :py:class:`fpy2.rewrite.Rewrite` class combines the matcher and applier
to perform a rewrite replacing `l` with `r` where the substitution
produced by matches of `l` are applied to `r`.

.. autoclass:: fpy2.rewrite.Rewrite
   :members:
   :show-inheritance:
