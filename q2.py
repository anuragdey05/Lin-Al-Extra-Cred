from q1 import CN, Mat

# Q2 a) Functions to check matrix properties

def is_zero(matrix: Mat):
    # Checks if the matrix is a zero matrix.
    return all(all(value == 0 for value in row) for row in matrix.data)

def is_square(matrix: Mat):
    # Checks if the matrix is square (n x n).
    return matrix.n == matrix.m

def is_symmetric(matrix: Mat):
    # Checks if the matrix is symmetric (A == A^T).
    if not is_square(matrix):
        return False
    return all(
        matrix.data[i][j] == matrix.data[j][i]
        for i in range(matrix.n) for j in range(matrix.m)
    )

def is_hermitian(matrix: Mat):
    # Checks if the matrix is Hermitian (A == A^H).
    if not is_square(matrix):
        return False
    return all(
        matrix.data[i][j] == matrix.data[j][i].cc() if isinstance(matrix.data[i][j], CN) else False
        for i in range(matrix.n) for j in range(matrix.m)
    )

def is_identity(matrix: Mat):
    # Checks if the matrix is an identity matrix.
    if not is_square(matrix):
        return False
    return all(
        (matrix.data[i][j] == 1 if i == j else matrix.data[i][j] == 0)
        for i in range(matrix.n) for j in range(matrix.m)
    )

def is_scalar(matrix: Mat):
    # Checks if the matrix is a scalar matrix (diagonal with all equal diagonal values).
    if not is_square(matrix):
        return False
    scalar_value = matrix.data[0][0]
    return all(
        (matrix.data[i][j] == scalar_value if i == j else matrix.data[i][j] == 0)
        for i in range(matrix.n) for j in range(matrix.m)
    )

def is_singular(matrix: Mat):
    # Checks if the matrix is singular (det(A) == 0).
    if not is_square(matrix):
        return False
    return determinant(matrix) == 0

def is_invertible(matrix: Mat):
    # Checks if the matrix is invertible (non-singular).
    if not is_square(matrix):
        return False
    return determinant(matrix) != 0

def is_positive_definite(matrix: Mat):
    # Checks if the matrix is positive definite.
    if not is_square(matrix):
        return False
    for i in range(1, matrix.n + 1):
        submatrix = get_submatrix(matrix, i)  # Extract leading principal submatrices
        if determinant(submatrix) <= 0:
            return False
    return True

def is_nilpotent(matrix: Mat, k=10):
    # Checks if the matrix is nilpotent (A^k = 0 for some k).
    if not is_square(matrix):
        return False
    current = matrix
    for _ in range(k):
        current = current * matrix
        if is_zero(current):
            return True
    return False

def is_diagonalizable(matrix: Mat):
    # Checks if the matrix is diagonalizable (A = PDP^(-1)).
    if not is_square(matrix):
        return False
    # Assuming eigenvalues are distinct for simplicity (full implementation requires more checks)
    eigenvalues = characteristic_polynomial(matrix)
    return len(set(eigenvalues)) == len(eigenvalues)

def is_orthogonal(matrix: Mat):
    # Checks if the matrix is orthogonal (A^T * A = I).
    if not is_square(matrix):
        return False
    return matrix.transpose() * matrix == is_identity(matrix)

def is_unitary(matrix: Mat):
    # Checks if the matrix is unitary (A^H * A = I).
    if not is_square(matrix):
        return False
    return matrix.conj_transpose() * matrix == is_identity(matrix)

def is_LU(matrix: Mat):
    # Checks if the matrix has an LU decomposition without using external libraries.
    if not is_square(matrix):
        return False

    n = matrix.n
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for k in range(i, n):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix.data[i][k] - sum

        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                if U[i][i] == 0:
                    return False
                L[k][i] = (matrix.data[k][i] - sum) / U[i][i]

    return True

# Helper Functions
def determinant(matrix: Mat):
    # Computes the determinant of a square matrix recursively.
    if matrix.n != matrix.m:
        raise ValueError("Determinant can only be computed for square matrices.")
    if matrix.n == 1:
        return matrix.data[0][0]
    if matrix.n == 2:
        a, b, c, d = matrix.data[0][0], matrix.data[0][1], matrix.data[1][0], matrix.data[1][1]
        return a * d - b * c
    det = 0
    for j in range(matrix.m):
        submatrix = get_submatrix(matrix, exclude_row=0, exclude_col=j)
        det += ((-1) ** j) * matrix.data[0][j] * determinant(submatrix)
    return det

def get_submatrix(matrix: Mat, exclude_row=None, exclude_col=None, size=None):
    # Returns a submatrix with specified rows/columns excluded.
    data = [
        [matrix.data[i][j] for j in range(matrix.m) if j != exclude_col]
        for i in range(matrix.n) if i != exclude_row
    ]
    return Mat(matrix.field, len(data), len(data[0]), [item for row in data for item in row])

def characteristic_polynomial(matrix: Mat):
    # Returns the characteristic polynomial of the matrix.
    if not is_square(matrix):
        raise ValueError("Characteristic polynomial can only be computed for square matrices.")
    
    def matrix_minor(matrix, i, j):
        # Return minor of matrix after removing i-th row and j-th column.
        return [
            [matrix[r][c] for c in range(len(matrix)) if c != j]
            for r in range(len(matrix)) if r != i
        ]

    def determinant_recursive(matrix):
        # Recursively calculate the determinant of a matrix.
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for c in range(len(matrix)):
            det += ((-1) ** c) * matrix[0][c] * determinant_recursive(matrix_minor(matrix, 0, c))
        return det

    def identity_matrix(size):
        # Create an identity matrix of given size.
        return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    def subtract_matrices(A, B):
        # Subtract matrix B from matrix A.
        return [[A[i][j] - B[i][j] for j in range(len(A))] for i in range(len(A))]

    n = matrix.n
    I = identity_matrix(n)
    coefficients = []
    
    for k in range(n + 1):
        lambda_I = [[I[i][j] * k for j in range(n)] for i in range(n)]
        subtracted_matrix = subtract_matrices(matrix.data, lambda_I)
        coefficients.append(determinant_recursive(subtracted_matrix))
    
    return coefficients

