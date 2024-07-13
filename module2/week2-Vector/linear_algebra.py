import numpy as np

class LinearAlgebra:
    @staticmethod
    def compute_vector_length(vector):
        return np.linalg.norm(vector)

    @staticmethod
    def compute_dot_product(vector1, vector2):
        return np.dot(vector1, vector2)

    @staticmethod
    def matrix_multi_vector(matrix, vector):
        return np.dot(matrix, vector)

    @staticmethod
    def matrix_multi_matrix(matrix1, matrix2):
        return np.dot(matrix1, matrix2)

    @staticmethod
    def inverse_matrix(matrix):
        return np.linalg.inv(matrix)

    @staticmethod
    def compute_eigenvalues_eigenvectors(matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

