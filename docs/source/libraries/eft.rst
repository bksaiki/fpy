Error-Free Transformations (EFT)
==================================

.. module:: fpy2.libraries.eft

The EFT library provides error-free transformations for basic arithmetic operations. These algorithms decompose floating-point operations into a primary result and an error term that exactly represent the mathematical result.

Splitting
---------

.. py:function:: veltkamp_split(x, s)
   :module: fpy2.libraries.eft

   Splits a floating-point number into a high and low part such that the high part is representable in ``prec(x) - s`` digits and the low part is representable in ``s`` digits.
   
   This algorithm is due to Veltkamp.

   :param x: Value to split
   :type x: Real
   :param s: Number of low-order digits
   :type s: Real
   :return: Tuple of (high_part, low_part)
   :rtype: tuple[Real, Real]

Addition
--------

.. py:function:: ideal_2sum(a, b)
   :module: fpy2.libraries.eft

   Error-free transformation of the sum of two floating-point numbers.

   :param a: First operand
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (s, t) where ``s`` is the floating-point sum and ``t`` is the error term such that ``s + t = a + b``
   :rtype: tuple[Real, Real]

   This implementation is "ideal" since the error term may not be representable in the caller's rounding context.

.. py:function:: fast_2sum(a, b)
   :module: fpy2.libraries.eft

   Error-free transformation of the sum of two floating-point numbers.
   This algorithm is due to Dekker (1971).

   :param a: First operand (must satisfy |a| >= |b|)
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (s, t) where ``s`` is the floating-point sum and ``t`` is the error term such that ``s + t = a + b``
   :rtype: tuple[Real, Real]

   **Assumes:**
   
   - ``|a| >= |b|``
   - the rounding context is floating point
   - the rounding mode is round-nearest

.. py:function:: classic_2sum(a, b)
   :module: fpy2.libraries.eft

   Computes the sum of two floating-point numbers with error-free transformation.
   This algorithm is due to Knuth and Moller.

   :param a: First operand
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (s, t) where ``s`` is the floating-point sum and ``t`` is the error term such that ``s + t = a + b``
   :rtype: tuple[Real, Real]

   **Assumes:**
   
   - the rounding context is floating point
   - the rounding mode is round-nearest

.. py:function:: priest_2sum(a, b)
   :module: fpy2.libraries.eft

   Computes the sum of two floating-point numbers with error-free transformation.
   This algorithm is due to Priest.

   :param a: First operand
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (s, t) where ``s`` is the faithfully-rounded sum and ``t`` is the error term such that ``s + t = a + b``
   :rtype: tuple[Real, Real]

   **Assumes:**
   
   - the rounding context is floating point

Multiplication
--------------

.. py:function:: ideal_2mul(a, b)
   :module: fpy2.libraries.eft

   Error-free transformation of the product of two floating-point numbers.

   :param a: First operand
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (p, t) where ``p`` is the floating-point product and ``t`` is the error term such that ``p + t = a * b``
   :rtype: tuple[Real, Real]

   This implementation is "ideal" since the error term may not be representable in the caller's rounding context.

.. py:function:: classic_2mul(a, b)
   :module: fpy2.libraries.eft

   Computes the product of two floating-point numbers with error-free transformation.
   This algorithm is due to Dekker.

   :param a: First operand
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (p, t) where ``p`` is the floating-point product and ``t`` is the error term such that ``p + t = a * b``
   :rtype: tuple[Real, Real]

   **Assumes:**
   
   - the rounding context is floating point
   - the rounding mode is round-nearest

.. py:function:: fast_2mul(a, b)
   :module: fpy2.libraries.eft

   Error-free transformation of the product of two floating-point numbers.

   :param a: First operand
   :type a: Real
   :param b: Second operand
   :type b: Real
   :return: Tuple (p, t) where ``p`` is the floating-point product and ``t`` is the error term such that ``p + t = a * b``
   :rtype: tuple[Real, Real]

   **Assumes:**
   
   - the rounding context is floating point

Fused Multiply-Add (FMA)
-------------------------

.. py:function:: ideal_fma(a, b, c)
   :module: fpy2.libraries.eft

   Error-free transformation of the fused multiply-add operation.

   :param a: First multiplicand
   :type a: Real
   :param b: Second multiplicand
   :type b: Real
   :param c: Addend
   :type c: Real
   :return: Tuple (r, t) where ``r`` is the floating-point result and ``t`` is the error term such that ``r + t = a * b + c``
   :rtype: tuple[Real, Real]

   This implementation is "ideal" since the error term may not be representable in the caller's rounding context.

.. py:function:: classic_2fma(a, b, c)
   :module: fpy2.libraries.eft

   Computes the fused multiply-add operation with error-free transformation.
   This algorithm is due to Boldo and Muller.

   :param a: First multiplicand
   :type a: Real
   :param b: Second multiplicand
   :type b: Real
   :param c: Addend
   :type c: Real
   :return: Tuple (r1, r2, r3) where ``a * b + c = r1 + r2 + r3``
   :rtype: tuple[Real, Real, Real]

   **Assumes:**
   
   - the rounding context is floating point
   - the rounding mode is round-nearest

