"""
Matrix operations.
"""

from . import base as fp


@fp.fpy
def zeros(rows: int, cols: int) -> list[list[fp.Real]]:
    """
    Create a zero matrix of size rows x cols.

    :param rows: Number of rows.
    :param cols: Number of columns.
    :return: Zero matrix.
    """
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

@fp.fpy
def ones(rows: int, cols: int) -> list[list[fp.Real]]:
    """
    Create a matrix of ones of size rows x cols.

    :param rows: Number of rows.
    :param cols: Number of columns.
    :return: Matrix of ones.
    """
    return [[1.0 for _ in range(cols)] for _ in range(rows)]

@fp.fpy
def identity(n: int) -> list[list[fp.Real]]:
    """
    Create an n x n identity matrix.

    :param n: Matrix size.
    :return: Identity matrix.
    """
    result = zeros(n, n)
    for i in range(n):
        result[i][i] = 1.0
    return result

@fp.fpy
def is_square(A: list[list[fp.Real]]) -> bool:
    """
    Check if matrix is square.

    :param A: Input matrix.
    :return: True if square, False otherwise.
    """
    rows = len(A)
    cols = len(A[0])
    return rows == cols

@fp.fpy
def is_symmetric(A: list[list[fp.Real]]) -> bool:
    """
    Check if a matrix is symmetric.

    :param A: Input matrix.
    :return: True if matrix is symmetric, False otherwise.
    """
    if is_square(A):
        n = len(A)
        result = True
        for i in range(n):
            for j in range(n):
                if A[i][j] != A[j][i]:
                    result = False
    else:
        result = False

    return result

@fp.fpy
def is_diagonal(A: list[list[fp.Real]]) -> bool:
    """
    Check if matrix is diagonal.

    :param A: Input matrix.
    :return: True if diagonal, False otherwise.
    """
    # assert message: "Matrix must be square"
    if is_square(A):
        n = len(A)
        result = True
        for i in range(n):
            for j in range(n):
                if i != j and A[i][j] != 0.0:
                    result = False
    else:
        result = False

    return result

@fp.fpy
def is_upper_triangular(A: list[list[fp.Real]]) -> bool:
    """
    Check if matrix is upper triangular.

    :param A: Input matrix.
    :return: True if upper triangular, False otherwise.
    """
    if is_square(A):
        n = len(A)
        result = True
        for i in range(n):
            for j in range(i):
                if A[i][j] != 0.0:
                    result = False
    else:
        result = False

    return result

@fp.fpy
def is_lower_triangular(A: list[list[fp.Real]]) -> bool:
    """
    Check if matrix is lower triangular.

    :param A: Input matrix.
    :return: True if lower triangular, False otherwise.
    """
    if is_square(A):
        n = len(A)
        result = True
        for i in range(n):
            for j in range(n):
                # TODO: range does not have a start
                if j > i:
                    if A[i][j] != 0.0:
                        result = False
    else:
        result = False

    return result

@fp.fpy
def diagonal(values: list[fp.Real]) -> list[list[fp.Real]]:
    """
    Create a diagonal matrix from a list of values.

    :param values: Diagonal values.
    :return: Diagonal matrix.
    """
    n = len(values)
    result = zeros(n, n)
    for i in range(n):
        result[i][i] = values[i]
    return result

@fp.fpy
def add(A: list[list[fp.Real]], B: list[list[fp.Real]]) -> list[list[fp.Real]]:
    """
    Element-wise addition of two matrices.

    :param A: First matrix.
    :param B: Second matrix.
    :return: Result matrix A + B.
    """
    rows, cols = len(A), len(A[0])
    # assert message: "Matrix dimensions must match for addition"
    assert len(B) == rows and len(B[0]) == cols

    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] + B[i][j]
    return result

@fp.fpy
def sub(A: list[list[fp.Real]], B: list[list[fp.Real]]) -> list[list[fp.Real]]:
    """
    Element-wise subtraction of two matrices.

    :param A: First matrix.
    :param B: Second matrix.
    :return: Result matrix A - B.
    """
    rows, cols = len(A), len(A[0])
    # assert message: "Matrix dimensions must match for subtraction"
    assert len(B) == rows and len(B[0]) == cols

    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] - B[i][j]
    return result

@fp.fpy
def hadamard(A: list[list[fp.Real]], B: list[list[fp.Real]]) -> list[list[fp.Real]]:
    """
    Element-wise multiplication (Hadamard product) of two matrices.

    :param A: First matrix.
    :param B: Second matrix.
    :return: Result matrix A ⊙ B.
    """
    rows, cols = len(A), len(A[0])
    # assert message: "Matrix dimensions must match for Hadamard product"
    assert len(B) == rows and len(B[0]) == cols
    
    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = A[i][j] * B[i][j]
    return result

@fp.fpy
def scale(scalar: fp.Real, A: list[list[fp.Real]]) -> list[list[fp.Real]]:
    """
    Scale a matrix by a scalar.

    :param scalar: Scalar multiplier.
    :param A: Input matrix.
    :return: Result matrix scalar * A.
    """
    rows, cols = len(A), len(A[0])
    result = zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            result[i][j] = scalar * A[i][j]
    return result

@fp.fpy
def matmul(A: list[list[fp.Real]], B: list[list[fp.Real]]) -> list[list[fp.Real]]:
    """
    Matrix multiplication A * B.

    :param A: First matrix (m x n).
    :param B: Second matrix (n x p).
    :return: Result matrix (m x p).
    """
    m, n = len(A), len(A[0])
    n_b, p = len(B), len(B[0])
    # assert message: "Matrix dimensions incompatible for multiplication"
    assert n == n_b
    
    result = zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] = result[i][j] + A[i][k] * B[k][j]
    return result

@fp.fpy
def transpose(A: list[list[fp.Real]]) -> list[list[fp.Real]]:
    """
    Transpose a matrix.

    :param A: Input matrix.
    :return: Transposed matrix A^T.
    """
    rows, cols = len(A), len(A[0])
    result = zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            result[j][i] = A[i][j]
    return result

@fp.fpy
def trace(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute the trace (sum of diagonal elements) of a square matrix.

    :param A: Input square matrix.
    :return: Trace of A.
    """
    assert is_square(A) # Matrix must be square
    
    result = 0.0
    for i in range(len(A)):
        result = result + A[i][i]
    return result

@fp.fpy
def frobenius_norm(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute the Frobenius norm of a matrix.

    :param A: Input matrix.
    :return: Frobenius norm ||A||_F.
    """
    rows, cols = len(A), len(A[0])
    sum_squares = 0.0
    for i in range(rows):
        for j in range(cols):
            sum_squares = sum_squares + A[i][j] * A[i][j]
    return fp.sqrt(sum_squares)

@fp.fpy
def determinant_2x2(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute determinant of a 2x2 matrix.

    :param A: 2x2 matrix.
    :return: Determinant of A.
    """
    # assert message: "Matrix must be 2x2"
    assert len(A) == 2 and len(A[0]) == 2
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

@fp.fpy
def determinant_3x3(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute determinant of a 3x3 matrix using cofactor expansion.

    :param A: 3x3 matrix.
    :return: Determinant of A.
    """
    # assert message: "Matrix must be 3x3"
    assert len(A) == 3 and len(A[0]) == 3

    # Cofactor expansion along first row
    det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
    det = det - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
    det = det + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    return det

@fp.fpy
def matvec(A: list[list[fp.Real]], x: list[fp.Real]):
    """
    Multiply a matrix by a vector: A * x.

    :param A: Matrix (m x n).
    :param x: Vector (length n).
    :return: Result vector (length m).
    """
    m, n = len(A), len(A[0])
    # assert message: "Vector length must match matrix columns"
    assert len(x) == n

    result = [0.0 for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i] = result[i] + A[i][j] * x[j]
    return result

@fp.fpy
def outer_product(x: list[fp.Real], y: list[fp.Real]):
    """
    Compute outer product of two vectors: x ⊗ y.

    :param x: First vector (length m).
    :param y: Second vector (length n).
    :return: Result matrix (m x n).
    """
    m, n = len(x), len(y)
    result = zeros(m, n)
    for i in range(m):
        for j in range(n):
            result[i][j] = x[i] * y[j]
    return result

@fp.fpy
def get_row(A: list[list[fp.Real]], i: int):
    """
    Extract a row from a matrix.

    :param A: Input matrix.
    :param i: Row index.
    :return: Row vector.
    """
    return A[i][:]

@fp.fpy
def get_column(A: list[list[fp.Real]], j: int):
    """
    Extract a column from a matrix.

    :param A: Input matrix.
    :param j: Column index.
    :return: Column vector.
    """
    return [A[i][j] for i in range(len(A))]

@fp.fpy
def set_row(A: list[list[fp.Real]], i: int, row: list[fp.Real]) -> list[list[fp.Real]]:
    """
    Set a row in a matrix.

    :param A: Input matrix.
    :param i: Row index.
    :param row: New row values.
    :return: Matrix with updated row.
    """
    result = [row_data[:] for row_data in A]  # Deep copy
    result[i] = row[:]
    return result

@fp.fpy
def set_column(A: list[list[fp.Real]], j: int, col: list[fp.Real]) -> list[list[fp.Real]]:
    """
    Set a column in a matrix.

    :param A: Input matrix.
    :param j: Column index.
    :param col: New column values.
    :return: Matrix with updated column.
    """
    rows = len(A)
    # assert message: "Column length must match matrix rows"
    assert len(col) == rows

    result = [row_data[:] for row_data in A]  # Deep copy
    for i in range(rows):
        result[i][j] = col[i]
    return result

@fp.fpy
def max_element(A: list[list[fp.Real]]) -> fp.Real:
    """
    Find the maximum element in a matrix.

    :param A: Input matrix.
    :return: Maximum element.
    """
    rows, cols = len(A), len(A[0])
    max_val = A[0][0]
    for i in range(rows):
        for j in range(cols):
            if A[i][j] > max_val:
                max_val = A[i][j]
    return max_val

@fp.fpy
def min_element(A: list[list[fp.Real]]) -> fp.Real:
    """
    Find the minimum element in a matrix.

    :param A: Input matrix.
    :return: Minimum element.
    """
    rows, cols = len(A), len(A[0])
    min_val = A[0][0]
    for i in range(rows):
        for j in range(cols):
            if A[i][j] < min_val:
                min_val = A[i][j]
    return min_val

@fp.fpy
def sum_elements(A: list[list[fp.Real]]) -> fp.Real:
    """
    Sum all elements in a matrix.
    
    :param A: Input matrix.
    :return: Sum of all elements.
    """
    rows, cols = len(A), len(A[0])
    result = 0.0
    for i in range(rows):
        for j in range(cols):
            result = result + A[i][j]
    return result

@fp.fpy
def mean_elements(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute mean of all matrix elements.
    
    :param A: Input matrix.
    :return: Mean of all elements.
    """
    rows, cols = len(A), len(A[0])
    total = sum_elements(A)
    return total / (rows * cols)

@fp.fpy
def norm_1(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute 1-norm (maximum absolute column sum) of matrix.

    :param A: Input matrix.
    :return: 1-norm of matrix.
    """
    rows, cols = len(A), len(A[0])
    max_col_sum = 0.0

    for j in range(cols):
        col_sum = 0.0
        for i in range(rows):
            col_sum = col_sum + abs(A[i][j])
        if col_sum > max_col_sum:
            max_col_sum = col_sum

    return max_col_sum

@fp.fpy
def norm_inf(A: list[list[fp.Real]]) -> fp.Real:
    """
    Compute infinity-norm (maximum absolute row sum) of matrix.

    :param A: Input matrix.
    :return: Infinity-norm of matrix.
    """
    rows, cols = len(A), len(A[0])
    max_row_sum = 0.0

    for i in range(rows):
        row_sum = 0.0
        for j in range(cols):
            row_sum = row_sum + abs(A[i][j])
        if row_sum > max_row_sum:
            max_row_sum = row_sum

    return max_row_sum

@fp.fpy
def vander(x: list[fp.Real], n: int) -> list[list[fp.Real]]:
    """
    Generate Vandermonde matrix.

    :param x: Input vector.
    :param n: Number of columns.
    :return: Vandermonde matrix.
    """
    m = len(x)
    result = zeros(m, n)

    for i in range(m):
        for j in range(n):
            result[i][j] = fp.pow(x[i], j)

    return result
