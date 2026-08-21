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

Cursors
-------

A ``where`` may also be a *cursor*: a location that survives the rewrites
around it, so a schedule pins a program point once and aims a whole sequence
of strategies at it. A cursor from an earlier program is forwarded to the
current one on arrival, and one that no longer names anything raises
:class:`~fpy2.strategies.TransformReferenceError` rather than landing
somewhere else.

.. autoclass:: fpy2.strategies.StmtCursor
   :members:

.. autoclass:: fpy2.strategies.BlockCursor
   :members:

.. autoclass:: fpy2.strategies.ExprCursor
   :members:

:func:`~fpy2.strategies.sites` is how a cursor is obtained:

.. autofunction:: fpy2.strategies.sites

A pattern is the other way: :func:`fpy2.rewrite.find` names a location by what
it looks like rather than by counting candidates. See :doc:`rewrite`.

Strategies
----------

The available strategies are found in the :mod:`fpy2.strategies` module:

.. autofunction:: fpy2.strategies.close

.. autofunction:: fpy2.strategies.elim_iter

.. autofunction:: fpy2.strategies.elim_round

.. autofunction:: fpy2.strategies.float_to_fixed

.. autofunction:: fpy2.strategies.fuse

.. autofunction:: fpy2.strategies.inline

.. autofunction:: fpy2.strategies.insert_round

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
