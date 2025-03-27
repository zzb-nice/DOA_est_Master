import numpy as np
from scipy.linalg import toeplitz


# matrix: complex value matrix, shape M*M
# matrix_concat: real value matrix, shape 2*M*M
# vec: real value vector, shape M*(M-1),实部在前虚部在后
# others: according vec or Phy and amp , not employed now

# transform toeplitz vector to matrix
def vec_2_toeplitzM(vec: np.ndarray):
    assert vec.ndim == 1, 'error input size'
    v_lens = len(vec) // 2
    vec = vec[:, v_lens] + 1j * vec[v_lens:]

    return toeplitz(vec.conj())


def get_toeplitzM_vec(toeplitzM: np.ndarray):
    assert toeplitzM.ndim == 2, 'error input size'
    toeplitz_vec = toeplitzM[0, :]
    vec_real = toeplitz_vec.real
    vec_imag = toeplitz_vec.imag

    return np.concatenate([vec_real, vec_imag], axis=0)


def matrix_2_matrix_concat(matrix: np.ndarray):
    assert matrix.ndim == 2, 'error input size'
    matrix_real = matrix.real
    matrix_imag = matrix.imag

    return np.stack([matrix_real, matrix_imag], axis=0)


def concat_matrix_2_matrix(concat_matrix: np.ndarray):
    assert concat_matrix.ndim == 3, 'error input size'

    return concat_matrix[0] + 1j * concat_matrix[1]


def matrix_2_vec(matrix: np.ndarray):
    assert matrix.ndim == 2, 'error input size'
    upper_index = np.triu_indices_from(matrix, k=1)

    vec = matrix[upper_index]
    return np.concatenate([vec.real, vec.imag], axis=0)


def matrix_2_enhance(matrix: np.ndarray):
    assert matrix.ndim == 2, 'error input size'
    matrix_real, matrix_imag, matrix_angle = matrix.real, matrix.imag, np.angle(matrix)
    return np.stack([matrix_real, matrix_imag, matrix_angle], axis=0)


# def vec_2_matrix(vec: np.ndarray):
#     assert vec.ndim == 1, 'error input size'
#     v_lens = len(vec) // 2
#     vec = vec[:v_lens] + 1j*vec[v_lens:]
