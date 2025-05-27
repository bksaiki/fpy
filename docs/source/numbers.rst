Number Systems
==================

FPy supports a variety of number systems including
multi-precision floating-point and fixed-point numbers.

Philosophy
------------------

This library embraces design principles found in the FPCore standard.
Most importantly, numerical programs need only be specified by
(i) real-number mathematical operations and (ii) rounding.
Number formats, e.g., `double` or `float`, are not first-class, as in most programming languages.
Rather, formats are just *side effects* of rounding and should be de-emphasized or eliminated entirely.
Furthermore, all values within a numerical program should be viewed as (extended) real numbers.

Rounding Contexts
^^^^^^^^^^^^^^^^^^

FPy provides a hierarchy of abstract classes describing rounding contexts.

.. autoclass:: fpy2.Context
   :members:
   :show-inheritance:

.. autoclass:: fpy2.OrdinalContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.SizedContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.EncodableContext
   :members:
   :show-inheritance:

FPy provides a number of concrete rounding contexts.
Each context implements a particular flavor of rounding.

.. autoclass:: fpy2.RealContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.MPContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.MPSContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.MPBContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.IEEEContext
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ExtContext
   :members:
   :show-inheritance:

Common Rounding Contexts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FPy provides a number of aliases for common rounding contexts.

.. autodata:: fpy2.FP256

Number Types
^^^^^^^^^^^^^^^^^^

FPy only provides two number types.

.. autoclass:: fpy2.RealFloat
   :members:
   :show-inheritance:

.. autoclass:: fpy2.Float
   :members:
   :show-inheritance:

Rounding Utilities
^^^^^^^^^^^^^^^^^^

Additional types for specifying rounding operations.

.. autoclass:: fpy2.RoundingMode
   :members:
   :show-inheritance:

.. autoclass:: fpy2.RoundingDirection
   :members:
   :show-inheritance:

.. autoclass:: fpy2.ExtNanKind
   :members:
   :show-inheritance:
