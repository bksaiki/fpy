Scheduling
===================

FPy's scheduling language allows users to transform FPy functions using
various small and reusable strategies. These strategies can be combined
to create more complex transformations.

The available strategies are found in the :mod:`fpy2.strategies` module:

.. autofunction:: fpy2.strategies.simplify

.. autofunction:: fpy2.strategies.split

.. autofunction:: fpy2.strategies.unroll_for

.. autofunction:: fpy2.strategies.unroll_while
