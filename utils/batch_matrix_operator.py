import numpy as np
from scipy.linalg import toeplitz
import copy


#  batch operate version of "matrix_operator.py"
#  be compatible with (batch,*) and (*)


# transform toeplitz vector to matrix
def vec_2_toeplitzM(vec: np.ndarray):
    v_lens = vec.shape[-1] // 2
    vec = vec[:, :v_lens] + 1j * vec[:, v_lens:]
    vec = vec.conj()
    # operation on the last axis
    toeplitzM = np.apply_along_axis(toeplitz, -1, vec)
    return toeplitzM


def get_toeplitzM_vec(toeplitzM: np.ndarray):
    toeplitz_vec = toeplitzM[..., 0, :]
    vec_real = toeplitz_vec.real
    vec_imag = toeplitz_vec.imag

    return np.concatenate([vec_real, vec_imag], axis=-1)


# b,h,w and h,w input
def matrix_2_matrix_concat(matrix: np.ndarray):
    matrix_real = matrix.real
    matrix_imag = matrix.imag

    return np.stack([matrix_real, matrix_imag], axis=-3)


def concat_matrix_2_matrix(concat_matrix: np.ndarray):
    return concat_matrix[..., 0, :, :] + 1j * concat_matrix[..., 1, :, :]


def matrix_2_vec(matrix: np.ndarray):
    m, n = matrix.shape[-2], matrix.shape[-1]
    upper_index = np.triu_indices(m, 1, n)

    vec = matrix[..., upper_index[0], upper_index[1]]
    return np.concatenate([vec.real, vec.imag], axis=-1)


def matrix_2_vec2(matrix: np.ndarray):
    m, n = matrix.shape[-2], matrix.shape[-1]
    upper_index = np.triu_indices(m, 0, n)

    vec = matrix[..., upper_index[0], upper_index[1]]
    return np.concatenate([vec.real, vec.imag], axis=-1)


def vec_2_matrix(vec: np.ndarray, reshape_size):
    v_lens = vec.shape[-1] // 2
    vec = vec[..., :v_lens] + 1j * vec[..., v_lens:]
    matrix = np.zeros(reshape_size, dtype=np.complex64)
    upper_index = np.triu_indices(reshape_size[-1], 0)
    matrix[..., upper_index[0], upper_index[1]] = vec
    # upper_triangular + upper_triangular.transpose()
    i = np.arange(reshape_size[-1])
    matrix_lt = copy.deepcopy(matrix)
    # matrix_lt[..., i, i] = 0
    matrix_lt = matrix_lt.swapaxes(-1,-2).conj()
    return matrix_lt+matrix


def matrix_2_enhance(matrix: np.ndarray):
    matrix_real, matrix_imag, matrix_angle = matrix.real, matrix.imag, np.angle(matrix)

    return np.stack([matrix_real, matrix_imag, matrix_angle], axis=-3)


# operation to generate a batch of diagonal matrices from a batch of diagonal elements
def batch_diag_matrices(batch_diags):
    assert batch_diags.ndim == 2, 'error input size'
    batch_size, diag_length = batch_diags.shape

    # Initialize a 3D array to hold the diagonal matrices
    batch_diag_matrices = np.zeros((batch_size, diag_length, diag_length))

    i = np.arange(diag_length)
    batch_diag_matrices[:, i, i] = batch_diags

    return batch_diag_matrices


# transform num to db and transform db to num
def to_db(num, factor=10):
    return factor * np.log10(num)


def to_num(num, factor=10):
    return 10 ** (num / factor)
