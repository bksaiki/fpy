Metrics
==================

.. module:: fpy2.libraries.metrics

The metrics library provides common error metrics and condition numbers for floating-point analysis.

Error Metrics
-------------

.. py:function:: absolute_error(x, y)
   :module: fpy2.libraries.metrics

   Computes the absolute error between ``x`` and ``y``, i.e., ``|x - y|``, rounding under the current context.

   :param x: First value
   :type x: Real
   :param y: Second value
   :type y: Real
   :return: Absolute error ``|x - y|``
   :rtype: Real

.. py:function:: relative_error(x, y)
   :module: fpy2.libraries.metrics

   Computes the relative error between ``x`` and ``y``, i.e., ``|x - y| / |y|``, rounding under the current context.

   :param x: First value (approximate)
   :type x: Real
   :param y: Second value (reference)
   :type y: Real
   :return: Relative error ``|x - y| / |y|``
   :rtype: Real

.. py:function:: scaled_error(x, y, s)
   :module: fpy2.libraries.metrics

   Computes the scaled error between ``x`` and ``y``, scaled by ``s``, i.e., ``|x - y| / |s|``, rounding under the current context.

   :param x: First value
   :type x: Real
   :param y: Second value
   :type y: Real
   :param s: Scaling factor
   :type s: Real
   :return: Scaled error ``|x - y| / |s|``
   :rtype: Real

   **Note:** When ``s = y``, this is equivalent to ``relative_error(x, y)``.

.. py:function:: ordinal_error(x, y)
   :module: fpy2.libraries.metrics

   Computes the ordinal error between ``x`` and ``y``, i.e., the number of floating-point numbers between ``x`` and ``y``.

   This is equivalent to ``|int(x) - int(y)|``, where ``int`` is the conversion from ``Float`` to ``int``.

   :param x: First value
   :type x: Float
   :param y: Second value
   :type y: Float
   :return: Number of floating-point numbers between x and y
   :rtype: Fraction
   :raises TypeError: if ``x`` or ``y`` is not a Float or Fraction, or if the context is not an OrdinalContext

   **Primitive**: This is an FPy primitive that requires an OrdinalContext.

