#Extra Credit Assignment

# Q1 a) Class for Complex Numbers with required operations
class CN:
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        if not isinstance(other, CN):
            raise TypeError("Addition requires another CN.")
        return CN(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        if not isinstance(other, CN):
            raise TypeError("Multiplication requires another CN.")
        real_part = self.real * other.real - self.imag * other.imag
        imag_part = self.real * other.imag + self.imag * other.real
        return CN(real_part, imag_part)

    def __truediv__(self, other):
        if not isinstance(other, CN):
            raise TypeError("Division requires another CN.")
        if other.real == 0 and other.imag == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        denominator = other.real ** 2 + other.imag ** 2
        real_part = (self.real * other.real + self.imag * other.imag) / denominator
        imag_part = (self.imag * other.real - self.real * other.imag) / denominator
        return CN(real_part, imag_part)

    def abs(self):
        return (self.real ** 2 + self.imag ** 2) ** 0.5

    def cc(self):
        return CN(self.real, -self.imag)

    def __repr__(self):
        return f"{self.real} + {self.imag}i"

# Q1 b) Class for Vectors with real or complex fields
class Vec:
    def __init__(self, field: str, length: int, values: list):
        if field not in ["real", "complex"]:
            raise ValueError("Field must be 'real' or 'complex'.")
        if len(values) != length:
            raise ValueError(f"Expected {length} values, got {len(values)}.")
        self.field = field
        self.values = self._validate_values(values)

    def _validate_values(self, values):
        if self.field == "real" and not all(isinstance(x, (int, float)) for x in values):
            raise ValueError("Values must be real numbers.")
        if self.field == "complex" and not all(isinstance(x, CN) for x in values):
            raise ValueError("Values must be CN instances.")
        return values

    def __repr__(self):
        return f"Vec({self.values})"

# Q1 c) & d) Class for Matrices with real or complex fields
class Mat:
    def __init__(self, field: str, n: int, m: int, values=None, vectors=None):
        self.field = field
        self.n = n
        self.m = m
        if vectors:
            self._init_from_vectors(vectors)  # d) Initialize using vectors
        elif values:
            self._init_from_values(values)  # c) Initialize using values
        else:
            raise ValueError("Either values or vectors must be provided.")

    def _init_from_values(self, values):
        if len(values) != self.n * self.m:
            raise ValueError(f"Expected {self.n * self.m} values, got {len(values)}.")
        self.data = [values[i:i + self.m] for i in range(0, len(values), self.m)]

    def _init_from_vectors(self, vectors):
        if len(vectors) != self.m:
            raise ValueError(f"Expected {self.m} vectors, got {len(vectors)}.")
        if any(len(v.values) != self.n for v in vectors):
            raise ValueError("All vectors must have length equal to n.")
        self.data = [[v.values[i] for v in vectors] for i in range(self.n)]

    # Q1 e) Overloaded addition operator for matrices
    def __add__(self, other):
        if not isinstance(other, Mat) or self.n != other.n or self.m != other.m:
            raise ValueError("Matrices must have the same dimensions for addition.")
        return Mat(
            self.field,
            self.n,
            self.m,
            [self.data[i][j] + other.data[i][j] for i in range(self.n) for j in range(self.m)],
        )

    # Q1 e) Overloaded multiplication operator for matrices
    def __mul__(self, other):
        if isinstance(other, Mat):
            if self.m != other.n:
                raise ValueError("Invalid dimensions for matrix multiplication.")
            result = [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.m))
                    for j in range(other.m)
                ]
                for i in range(self.n)
            ]
            return Mat(self.field, self.n, other.m, [item for row in result for item in row])
        else:
            raise TypeError("Can only multiply by another Mat.")

    # Q1 f) Get specific rows or columns
    def get_row(self, index: int):
        if index < 0 or index >= self.n:
            raise IndexError("Row index out of bounds.")
        return self.data[index]

    def get_column(self, index: int):
        if index < 0 or index >= self.m:
            raise IndexError("Column index out of bounds.")
        return [self.data[i][index] for i in range(self.n)]

    # Q1 g) Transpose of a matrix
    def transpose(self):
        transposed = [[self.data[j][i] for j in range(self.n)] for i in range(self.m)]
        return Mat(self.field, self.m, self.n, [item for row in transposed for item in row])

    # Q1 g) Conjugate of a matrix
    def conj(self):
        conjugated = [[value.cc() if isinstance(value, CN) else value for value in row] for row in self.data]
        return Mat(self.field, self.n, self.m, [item for row in conjugated for item in row])

    # Q1 g) Transpose-Conjugate of a matrix
    def conj_transpose(self):
        return self.transpose().conj()

    def __repr__(self):
        return f"Mat({self.data})"
