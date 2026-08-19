Scheduling
===================

FPy's scheduling language allows users to transform FPy functions using
various small and reusable strategies. These strategies can be combined
to create more complex transformations.

An explicit ``where`` that names a site a strategy refuses raises
:class:`~fpy2.strategies.TransformDeclined` with the reason, and one that
names no site raises :class:`~fpy2.strategies.TransformReferenceError`;
``where=None`` rewrites every site the strategy can and skips the rest.

.. autoexception:: fpy2.strategies.TransformError

.. autoexception:: fpy2.strategies.TransformDeclined

.. autoexception:: fpy2.strategies.TransformReferenceError

The available strategies are found in the :mod:`fpy2.strategies` module:

.. autofunction:: fpy2.strategies.close

.. autofunction:: fpy2.strategies.elim_iter

.. autofunction:: fpy2.strategies.elim_round

.. autofunction:: fpy2.strategies.float_to_fixed

.. autofunction:: fpy2.strategies.fuse

.. autofunction:: fpy2.strategies.inline

.. autofunction:: fpy2.strategies.lift_context

.. autofunction:: fpy2.strategies.monomorphize

.. autofunction:: fpy2.strategies.rescale_fixed

.. autofunction:: fpy2.strategies.simplify

.. autofunction:: fpy2.strategies.split

.. autofunction:: fpy2.strategies.unfold_neg_zero

.. autofunction:: fpy2.strategies.unfold_overflow

.. autofunction:: fpy2.strategies.unfold_special

.. autofunction:: fpy2.strategies.unroll_for

.. autofunction:: fpy2.strategies.unroll_while
