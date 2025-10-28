Matrix Operations
==================

.. module:: fpy2.libraries.matrix

The matrix library provides operations on matrices represented as lists of lists (2D arrays) of Real numbers.

Construction
------------

.. py:function:: zeros(rows, cols)
   :module: fpy2.libraries.matrix

   Create a zero matrix of size rows x cols.

   :param rows: Number of rows
   :type rows: int
   :param cols: Number of columns
   :type cols: int
   :return: Zero matrix
   :rtype: list[list[Real]]

.. py:function:: ones(rows, cols)
   :module: fpy2.libraries.matrix

   Create a matrix of ones of size rows x cols.

   :param rows: Number of rows
   :type rows: int
   :param cols: Number of columns
   :type cols: int
   :return: Matrix of ones
   :rtype: list[list[Real]]

.. py:function:: identity(n)
   :module: fpy2.libraries.matrix

   Create an n x n identity matrix.

   :param n: Matrix size
   :type n: int
   :return: Identity matrix
   :rtype: list[list[Real]]

.. py:function:: diagonal(values)
   :module: fpy2.libraries.matrix

   Create a diagonal matrix from a list of values.

   :param values: Diagonal values
   :type values: list[Real]
   :return: Diagonal matrix
   :rtype: list[list[Real]]

.. py:function:: vander(x, n)
   :module: fpy2.libraries.matrix

   Generate Vandermonde matrix.

   :param x: Input vector
   :type x: list[Real]
   :param n: Number of columns
   :type n: int
   :return: Vandermonde matrix
   :rtype: list[list[Real]]

Predicates
----------

.. py:function:: is_square(A)
   :module: fpy2.libraries.matrix

   Check if matrix is square.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: True if square, False otherwise
   :rtype: bool

.. py:function:: is_symmetric(A)
   :module: fpy2.libraries.matrix

   Check if a matrix is symmetric.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: True if matrix is symmetric, False otherwise
   :rtype: bool

.. py:function:: is_diagonal(A)
   :module: fpy2.libraries.matrix

   Check if matrix is diagonal.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: True if diagonal, False otherwise
   :rtype: bool

.. py:function:: is_upper_triangular(A)
   :module: fpy2.libraries.matrix

   Check if matrix is upper triangular.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: True if upper triangular, False otherwise
   :rtype: bool

.. py:function:: is_lower_triangular(A)
   :module: fpy2.libraries.matrix

   Check if matrix is lower triangular.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: True if lower triangular, False otherwise
   :rtype: bool

Element-wise Operations
------------------------

.. py:function:: add(A, B)
   :module: fpy2.libraries.matrix

   Element-wise addition of two matrices.

   :param A: First matrix
   :type A: list[list[Real]]
   :param B: Second matrix
   :type B: list[list[Real]]
   :return: Result matrix A + B
   :rtype: list[list[Real]]

.. py:function:: sub(A, B)
   :module: fpy2.libraries.matrix

   Element-wise subtraction of two matrices.

   :param A: First matrix
   :type A: list[list[Real]]
   :param B: Second matrix
   :type B: list[list[Real]]
   :return: Result matrix A - B
   :rtype: list[list[Real]]

.. py:function:: hadamard(A, B)
   :module: fpy2.libraries.matrix

   Element-wise multiplication (Hadamard product) of two matrices.

   :param A: First matrix
   :type A: list[list[Real]]
   :param B: Second matrix
   :type B: list[list[Real]]
   :return: Result matrix A ⊙ B
   :rtype: list[list[Real]]

.. py:function:: scale(scalar, A)
   :module: fpy2.libraries.matrix

   Scale a matrix by a scalar.

   :param scalar: Scalar multiplier
   :type scalar: Real
   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Result matrix scalar * A
   :rtype: list[list[Real]]

Matrix Multiplication
---------------------

.. py:function:: matmul(A, B)
   :module: fpy2.libraries.matrix

   Matrix multiplication A * B.

   :param A: First matrix (m x n)
   :type A: list[list[Real]]
   :param B: Second matrix (n x p)
   :type B: list[list[Real]]
   :return: Result matrix (m x p)
   :rtype: list[list[Real]]

.. py:function:: matvec(A, x)
   :module: fpy2.libraries.matrix

   Multiply a matrix by a vector: A * x.

   :param A: Matrix (m x n)
   :type A: list[list[Real]]
   :param x: Vector (length n)
   :type x: list[Real]
   :return: Result vector (length m)
   :rtype: list[Real]

.. py:function:: outer_product(x, y)
   :module: fpy2.libraries.matrix

   Compute outer product of two vectors: x ⊗ y.

   :param x: First vector (length m)
   :type x: list[Real]
   :param y: Second vector (length n)
   :type y: list[Real]
   :return: Result matrix (m x n)
   :rtype: list[list[Real]]

Matrix Transformations
----------------------

.. py:function:: transpose(A)
   :module: fpy2.libraries.matrix

   Transpose a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Transposed matrix A^T
   :rtype: list[list[Real]]

Row and Column Operations
--------------------------

.. py:function:: get_row(A, i)
   :module: fpy2.libraries.matrix

   Extract a row from a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :param i: Row index
   :type i: int
   :return: Row vector
   :rtype: list[Real]

.. py:function:: get_column(A, j)
   :module: fpy2.libraries.matrix

   Extract a column from a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :param j: Column index
   :type j: int
   :return: Column vector
   :rtype: list[Real]

.. py:function:: set_row(A, i, row)
   :module: fpy2.libraries.matrix

   Set a row in a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :param i: Row index
   :type i: int
   :param row: New row values
   :type row: list[Real]
   :return: Matrix with updated row
   :rtype: list[list[Real]]

.. py:function:: set_column(A, j, col)
   :module: fpy2.libraries.matrix

   Set a column in a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :param j: Column index
   :type j: int
   :param col: New column values
   :type col: list[Real]
   :return: Matrix with updated column
   :rtype: list[list[Real]]

Matrix Properties
-----------------

.. py:function:: trace(A)
   :module: fpy2.libraries.matrix

   Compute the trace (sum of diagonal elements) of a square matrix.

   :param A: Input square matrix
   :type A: list[list[Real]]
   :return: Trace of A
   :rtype: Real

.. py:function:: determinant_2x2(A)
   :module: fpy2.libraries.matrix

   Compute determinant of a 2x2 matrix.

   :param A: 2x2 matrix
   :type A: list[list[Real]]
   :return: Determinant of A
   :rtype: Real

.. py:function:: determinant_3x3(A)
   :module: fpy2.libraries.matrix

   Compute determinant of a 3x3 matrix using cofactor expansion.

   :param A: 3x3 matrix
   :type A: list[list[Real]]
   :return: Determinant of A
   :rtype: Real

Norms
-----

.. py:function:: frobenius_norm(A)
   :module: fpy2.libraries.matrix

   Compute the Frobenius norm of a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Frobenius norm ||A||_F
   :rtype: Real

.. py:function:: norm_1(A)
   :module: fpy2.libraries.matrix

   Compute 1-norm (maximum absolute column sum) of matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: 1-norm of matrix
   :rtype: Real

.. py:function:: norm_inf(A)
   :module: fpy2.libraries.matrix

   Compute infinity-norm (maximum absolute row sum) of matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Infinity-norm of matrix
   :rtype: Real

Element Statistics
-------------------

.. py:function:: max_element(A)
   :module: fpy2.libraries.matrix

   Find the maximum element in a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Maximum element
   :rtype: Real

.. py:function:: min_element(A)
   :module: fpy2.libraries.matrix

   Find the minimum element in a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Minimum element
   :rtype: Real

.. py:function:: sum_elements(A)
   :module: fpy2.libraries.matrix

   Sum all elements in a matrix.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Sum of all elements
   :rtype: Real

.. py:function:: mean_elements(A)
   :module: fpy2.libraries.matrix

   Compute mean of all matrix elements.

   :param A: Input matrix
   :type A: list[list[Real]]
   :return: Mean of all elements
   :rtype: Real

