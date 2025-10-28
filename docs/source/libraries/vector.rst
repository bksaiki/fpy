Vector Operations
==================

.. module:: fpy2.libraries.vector

The vector library provides operations on vectors represented as Python lists of Real numbers.

Construction
------------

.. py:function:: zeros(n)
   :module: fpy2.libraries.vector

   Create a zero vector of length n.

   :param n: Vector length
   :type n: int
   :return: Zero vector
   :rtype: list[Real]

.. py:function:: ones(n)
   :module: fpy2.libraries.vector

   Create a vector of ones of length n.

   :param n: Vector length
   :type n: int
   :return: Vector of ones
   :rtype: list[Real]

Element-wise Operations
------------------------

.. py:function:: add(x, y)
   :module: fpy2.libraries.vector

   Element-wise addition of two vectors.

   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :return: Result vector x + y
   :rtype: list[Real]

.. py:function:: sub(x, y)
   :module: fpy2.libraries.vector

   Element-wise subtraction of two vectors.

   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :return: Result vector x - y
   :rtype: list[Real]

.. py:function:: hadamard(x, y)
   :module: fpy2.libraries.vector

   Element-wise multiplication (Hadamard product) of two vectors.

   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :return: Result vector x ⊙ y
   :rtype: list[Real]

.. py:function:: scale(a, x)
   :module: fpy2.libraries.vector

   Scale a vector by a scalar.

   :param a: Scalar multiplier
   :type a: Real
   :param x: Input vector
   :type x: list[Real]
   :return: Result vector a*x
   :rtype: list[Real]

Products
--------

.. py:function:: dot(x, y)
   :module: fpy2.libraries.vector

   Compute the dot product of two vectors.

   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :return: Dot product of x and y
   :rtype: Real

.. py:function:: dot_add(x, y, c)
   :module: fpy2.libraries.vector

   Compute `xy + c`, dot product with addition.

   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :param c: Scalar to add
   :type c: Real
   :return: Result x·y + c
   :rtype: Real

.. py:function:: cross(x, y)
   :module: fpy2.libraries.vector

   Compute cross product of two 3D vectors.

   :param x: First 3D vector
   :type x: list[Real]
   :param y: Second 3D vector
   :type y: list[Real]
   :return: Cross product x × y
   :rtype: list[Real]

BLAS-like Operations
---------------------

.. py:function:: axpy(a, x, y)
   :module: fpy2.libraries.vector

   Compute a*x + y (AXPY operation).

   :param a: Scalar multiplier
   :type a: Real
   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :return: Result vector a*x + y
   :rtype: list[Real]

Norms
-----

.. py:function:: norm1(x)
   :module: fpy2.libraries.vector

   Compute the L1 norm (Manhattan norm) of a vector.

   :param x: Input vector
   :type x: list[Real]
   :return: L1 norm of x
   :rtype: Real

.. py:function:: norm2(x)
   :module: fpy2.libraries.vector

   Compute the L2 norm (Euclidean norm) of a vector.

   :param x: Input vector
   :type x: list[Real]
   :return: L2 norm of x
   :rtype: Real

.. py:function:: norm_inf(x)
   :module: fpy2.libraries.vector

   Compute the infinity norm (maximum norm) of a vector.

   :param x: Input vector
   :type x: list[Real]
   :return: Infinity norm of x
   :rtype: Real

.. py:function:: norm_p(x, p)
   :module: fpy2.libraries.vector

   Compute the p-norm of a vector.

   :param x: Input vector
   :type x: list[Real]
   :param p: Norm parameter (p >= 1)
   :type p: Real
   :return: p-norm of x
   :rtype: Real

Normalization
-------------

.. py:function:: normalize(x)
   :module: fpy2.libraries.vector

   Normalize a vector to unit length (L2 norm).

   :param x: Input vector
   :type x: list[Real]
   :return: Unit vector in direction of x
   :rtype: list[Real]

.. py:function:: normalize_p(x, p)
   :module: fpy2.libraries.vector

   Normalize a vector using p-norm.

   :param x: Input vector
   :type x: list[Real]
   :param p: Norm parameter
   :type p: Real
   :return: Vector normalized by p-norm
   :rtype: list[Real]

Similarity and Distance
-----------------------

.. py:function:: cosine_similarity(x, y)
   :module: fpy2.libraries.vector

   Compute cosine similarity between two vectors.

   :param x: First vector
   :type x: list[Real]
   :param y: Second vector
   :type y: list[Real]
   :return: Cosine similarity x·y / (||x|| ||y||)
   :rtype: Real

Statistics
----------

.. py:function:: mean(x)
   :module: fpy2.libraries.vector

   Compute the mean of vector elements.

   :param x: Input vector
   :type x: list[Real]
   :return: Mean of elements
   :rtype: Real

.. py:function:: min_element(x)
   :module: fpy2.libraries.vector

   Find minimum element in vector.

   :param x: Input vector
   :type x: list[Real]
   :return: Minimum element
   :rtype: Real

.. py:function:: max_element(x)
   :module: fpy2.libraries.vector

   Find maximum element in vector.

   :param x: Input vector
   :type x: list[Real]
   :return: Maximum element
   :rtype: Real

