Scheduling
===================

FPy's scheduling language allows users to transform FPy functions using
various small and reusable strategies. These strategies can be combined
to create more complex transformations.

The available strategies are found in the :mod:`fpy2.strategies` module:

.. autofunction:: fpy2.strategies.close

.. autofunction:: fpy2.strategies.elim_iter

.. autofunction:: fpy2.strategies.elim_round

.. autofunction:: fpy2.strategies.fuse

.. autofunction:: fpy2.strategies.inline

.. autofunction:: fpy2.strategies.lift_context

.. autofunction:: fpy2.strategies.monomorphize

.. autofunction:: fpy2.strategies.rescale_fixed

.. autofunction:: fpy2.strategies.simplify

.. autofunction:: fpy2.strategies.split

.. autofunction:: fpy2.strategies.unroll_for

.. autofunction:: fpy2.strategies.unroll_while
