import numpy as np


class Unity_ESPRIT:
    def __init__(self, theta_fromz, M=8, displacement=1):
        """
        :param theta_fromz: function to get doas from z
        :param M: total number of sensors
        :param displacement: control the displacement of two subarrays
        """
        self.z_to_theta = theta_fromz
        self.displacement = displacement or None

        idx = np.arange(M)
        if self.displacement is None:
            self.sub_arr1 = np.array([0, 2, 4, 6])
            self.sub_arr2 = np.array([1, 3, 5, 7])
        else:
            self.sub_arr1 = idx[:-displacement]
            self.sub_arr2 = idx[displacement:]

    # receive y_t to calculate doas
    def tls_estimate(self, y_t, k, return_z=False):
        M, L = y_t.shape
        # create enhanced sample data
        Ph1 = generate_antidiagonal_matrix(M)
        Ph2 = generate_antidiagonal_matrix(L)
        Y_t = np.concatenate([y_t, Ph1 @ y_t.conj() @ Ph2], axis=-1)

        # calculate signal subspace
        y_t_real, Qm, Q_2l = Realization_Realize(Y_t, M, 2 * L)
        U, S, VT = np.linalg.svd(y_t_real)
        Es = Qm @ U[:, :k]

        # solve tls problem
        Ph3 = generate_antidiagonal_matrix(k)
        E12 = np.concatenate([Es[self.sub_arr1, :], Es[self.sub_arr2, :] @ Ph3], axis=-1)
        T12, Q_sub, Qk = Realization_Realize(E12, E12.shape[0], 2 * k)
        Ut, St, VTt = np.linalg.svd(T12)
        V11, V12 = VTt.T[:k, k:], VTt.T[k:, k:]

        real_tls = -np.linalg.solve(V12.T, V11.T).T
        # z probably transform to complex value
        z = np.linalg.eigvals(real_tls)
        z = -(z - 1j) / (z + 1j)
        if return_z:
            return True, self.z_to_theta(z), z
        else:
            return True, self.z_to_theta(z)


def Realization_Realize(ComplexM, m, n):
    Qm = create_Q(m)
    Qn = create_Q(n)
    RealM = Qm.T.conj() @ ComplexM @ Qn
    return RealM.real, Qm, Qn


def create_Q(m):
    In = np.eye(m // 2)
    PIn = generate_antidiagonal_matrix(m // 2)
    zero = np.zeros((m // 2, 1))
    if m % 2 == 0:
        Qm = np.block([[In, 1j * In], [PIn, -1j * PIn]]) / np.sqrt(2)
    else:
        Qm = np.block([[In, zero, 1j * In], [zero.T, np.sqrt(2), zero.T], [PIn, zero, -1j * PIn]]) / np.sqrt(2)
    return Qm


def generate_antidiagonal_matrix(size):
    matrix = np.zeros((size, size))

    idx1 = np.arange(size)
    idx2 = size - 1 - idx1
    matrix[idx1, idx2] = 1

    return matrix


if __name__ == '__main__':
    unity_esprit = Unity_ESPRIT(1, 8, 1)
    yt = np.random.randn(8, 256)
    unity_esprit.tls_estimation(yt, 3)
