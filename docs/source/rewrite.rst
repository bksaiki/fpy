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

Given a :py:class:`fpy2.Pattern` instance, the
:py:class:`fpy2.rewrite.Matcher` class finds the places within an FPy program
where the pattern matches. Each match carries a *cursor*, so the place it names
survives the rewrites around it — see :doc:`strategies`.

.. autoclass:: fpy2.rewrite.Matcher
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.Match
   :members:
   :show-inheritance:

.. autoclass:: fpy2.rewrite.Subst
   :members:
   :show-inheritance:

Finding
--------------------

A pattern is the other way to name a location: not "the *n*\ th candidate" but
"the place that looks like this". :func:`fpy2.rewrite.find` insists on one
match; :func:`fpy2.rewrite.find_all` lists them all.

.. autofunction:: fpy2.rewrite.find

.. autofunction:: fpy2.rewrite.find_all

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

A rewrite is aimed like any other strategy — by index, by cursor, or everywhere
— and reports what it replaced, so a cursor crosses it and a schedule can put a
user rule between two built-in operators. It is *not* verified: nothing checks
that `l` and `r` compute the same thing.

.. autoclass:: fpy2.rewrite.Rewrite
   :members:
   :show-inheritance:
