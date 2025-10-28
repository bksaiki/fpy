Core Library
==================

.. module:: fpy2.libraries.core

The core library provides essential numerical functions including splitting operations, predicates, exponent manipulation, and context queries.

Splitting Functions
-------------------

.. py:function:: split(x, n)
   :module: fpy2.libraries.core

   Splits ``x`` into two parts:

   - all digits of ``x`` that are above the ``n`` th digit
   - all digits of ``x`` that are at or below the ``n`` th digit

   The operation is performed exactly.

   :param x: Value to split
   :type x: Float
   :param n: Digit position (must be an integer)
   :type n: Float
   :return: Tuple of (high_part, low_part)
   :rtype: tuple[Float, Float]
   :raises ValueError: if ``n`` is not an integer

   **Special cases:**
   
   - if ``x`` is NaN, the result is ``(NaN, NaN)``
   - if ``x`` is infinite, the result is ``(x, x)``

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context ('R', 'R').

.. py:function:: modf(x)
   :module: fpy2.libraries.core

   Decomposes ``x`` into its integral and fractional parts.
   The operation is performed exactly.

   :param x: Value to decompose
   :type x: Float
   :return: Tuple of (fractional_part, integral_part)
   :rtype: tuple[Float, Float]

   **Special cases** (mirroring C/C++ ``modf``):
   
   - if ``x`` is ``+/-0``, the result is ``(+/-0, +/-0)``
   - if ``x`` is ``+/-Inf``, the result is ``(+/-0, +/-Inf)``
   - if ``x`` is NaN, the result is ``(NaN, NaN)``

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context ('R', 'R').

.. py:function:: frexp(x)
   :module: fpy2.libraries.core

   Decomposes ``x`` into its mantissa and exponent.
   The computation is performed exactly.

   :param x: Value to decompose
   :type x: Float
   :return: Tuple of (mantissa, exponent)
   :rtype: tuple[Float, Float]

   **Special cases** (mirroring C/C++ ``frexp``):
   
   - if ``x`` is NaN, the result is ``(NaN, NaN)``
   - if ``x`` is infinity, the result is ``(x, NaN)``
   - if ``x`` is zero, the result is ``(x, 0)``

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context ('R', 'R').

Predicates
----------

.. py:function:: isinteger(x)
   :module: fpy2.libraries.core

   Checks if ``x`` is an integer.

   :param x: Value to check
   :type x: Real
   :return: True if ``x`` is an integer, False otherwise
   :rtype: bool

.. py:function:: isnar(x)
   :module: fpy2.libraries.core

   Checks if ``x`` is either NaN or infinity (Not-a-Real).

   :param x: Value to check
   :type x: Real
   :return: True if ``x`` is NaN or infinity, False otherwise
   :rtype: bool

Exponent Functions
------------------

.. py:function:: logb(x)
   :module: fpy2.libraries.core

   Returns the normalized exponent of ``x``.

   :param x: Input value
   :type x: Float
   :return: Normalized exponent
   :rtype: Float

   **Special cases:**
   
   - If ``x == 0``, the result is ``-INFINITY``
   - If ``x`` is NaN, the result is NaN
   - If ``x`` is infinite, the result is ``INFINITY``

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context 'R'.

.. py:function:: ldexp(x, n)
   :module: fpy2.libraries.core

   Computes ``x * 2**n`` with correct rounding.

   :param x: Base value
   :type x: Float
   :param n: Exponent (must be an integer)
   :type n: Float
   :return: Result of ``x * 2**n``
   :rtype: Float
   :raises ValueError: if ``n`` is not an integer

   **Special cases:**
   
   - If ``x`` is NaN, the result is NaN
   - If ``x`` is infinite, the result is infinite

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context 'R'.

.. py:function:: max_e(xs)
   :module: fpy2.libraries.core

   Computes the largest (normalized) exponent of the subset of finite, non-zero elements of ``xs``.

   :param xs: List of values
   :type xs: list[Real]
   :return: Tuple of (largest_exponent, exists_non_zero)
   :rtype: tuple[Real, bool]

   Returns the largest exponent and whether any such element exists.
   If all elements are zero, infinite, or NaN, the exponent is ``0``.

   **Function context**: Uses INTEGER context.

Context Operations
------------------

.. py:function:: max_p()
   :module: fpy2.libraries.core

   Returns the maximum precision of the current context.
   This is a no-op for the ``RealContext``.

   :return: Maximum precision
   :rtype: Float
   :raises ValueError: if the context does not have a maximum precision

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context 'R'.

.. py:function:: min_n()
   :module: fpy2.libraries.core

   Returns the least absolute digit of the current context.
   This is the position of the most significant digit that can never be represented.

   :return: Least absolute digit position
   :rtype: Float
   :raises ValueError: if the context does not have a least absolute digit

   **Primitive**: This is an FPy primitive with context parameter 'R' and return context 'R'.

