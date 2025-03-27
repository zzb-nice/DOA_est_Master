import numpy as np


class ESPRIT:
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

    def tls_estimate(self, R, k, return_z=False):
        # get signal subspace of two subarray
        Us = get_signal_subspace(R, k)
        Us1 = Us[self.sub_arr1, :]
        Us2 = Us[self.sub_arr2, :]

        Us12 = np.concatenate([Us1, Us2], axis=1)
        Gn = Us12.transpose().conj() @ Us12

        # calculate tls results
        _, E = np.linalg.eigh(Gn)
        E = np.fliplr(E)  # Now in descending order
        E1 = E[:k, k:]
        E2 = E[k:, k:]
        # Phi = - E1 * np.linalg.inv(E2)
        Phi = -np.linalg.solve(E2.T, E1.T).T  # this is better

        z = np.linalg.eigvals(Phi)
        if return_z:
            return True, self.z_to_theta(z), z
        else:
            return True, self.z_to_theta(z)


def get_signal_subspace(R, k):
    """
    Gets the signal eigenvectors.

    Args:
        R: Covariance matrix.
        k: Number of sources.
    """
    eigenvalue, U = np.linalg.eigh(R)
    U_s = U[:, -k:]
    U_n = U[:, :-k]
    # Note: eigenvalues are sorted in ascending order.
    return U_s
