class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
    
    def shape(self):
        return (self.rows, self.cols)
    
    def transpose(self):
        transposed = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)] # swap ij to ji
        return Matrix(transposed)

    def matmul(self, other):
        assert self.cols == other.rows, "The shapes of the matrices are not compatible. # cols of matrix 1 should equal # rows of matrix 2."
        res = [[0.0 for _ in range(other.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    res[i][j] += self.data[i][k] * other.data[k][j]
        return Matrix(res)

    def inverse(self): # self must be square
        assert self.rows == self.cols, "Only square matrices are invertible."
        n = self.rows
        
        #Build augmented matrix
        augmented_matrix_all_zeros = [[0.0 for _ in range(2*n)] for _ in range(n)]
        augmented_matrix = Matrix(augmented_matrix_all_zeros)
        for i in range(n):
            for j in range(n):
                augmented_matrix.data[i][j] = self.data[i][j]
            for j in range(n, 2*n):
                if j - n == i:
                    augmented_matrix.data[i][j] = 1
                else:
                    augmented_matrix.data[i][j] = 0
        
        for i in range(n):
            pivot = augmented_matrix.data[i][i]
            if pivot == 0:
                raise ValueError("Matrix is singular and therefore cannot be inverted.")
            for j in range(2*n):
                augmented_matrix.data[i][j] = augmented_matrix.data[i][j] / pivot
            
            for k in range(n):
                if k != i:
                    factor = augmented_matrix.data[k][i]
                    for j in range(2*n):
                        augmented_matrix.data[k][j] = augmented_matrix.data[k][j] - (factor * augmented_matrix.data[i][j])
        
        A_inverse_all_zeros = [[0.0 for _ in range(self.rows)] for _ in range(self.cols)]
        A_inverse = Matrix(A_inverse_all_zeros)
        for i in range(n):
            for j in range(n):
                A_inverse.data[i][j] = augmented_matrix.data[i][j+n]
        return A_inverse

    def __repr__(self):
        return "\n".join(str(row) for row in self.data)
