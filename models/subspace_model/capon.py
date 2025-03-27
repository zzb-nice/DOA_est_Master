from music import peak_finder
import numpy as np


def capon_doa(X, theta_grid, steering_vector):
    """
    Capon DOA estimation.

    :param X: MxN array, received signal data where M is the number of sensors and N is the number of snapshots.
    :param theta_grid: 1D array of possible angles (in degrees) to search for DOA.
    :param steering_vector: function that generates the steering vector a(θ) for a given angle θ.

    :return: Estimated DOA angles.
    """
    M, N = X.shape
    R = (X @ X.conj().T) / N  # Sample covariance matrix
    R_inv = np.linalg.inv(R)  # Inverse of covariance matrix

    # Calculate Capon spatial spectrum
    P_capon = np.zeros_like(theta_grid, dtype=float)
    for i, theta in enumerate(theta_grid):
        a_theta = steering_vector(theta)
        P_capon[i] = 1 / np.real(a_theta.conj().T @ R_inv @ a_theta)

    # Normalize and convert to dB
    P_capon_db = 10 * np.log10(P_capon / np.max(P_capon))

    # Find peaks (local maxima) as DOA estimates
    doa_estimates = theta_grid[np.where(P_capon_db == np.max(P_capon_db))]

    return doa_estimates, P_capon_db

class Capon:
    def __init__(self, get_steer_vector, start=-90, end=90, step=0.1):
        self._search_grid = np.arange(start, end + 0.001, step)
        self._peak_finder = peak_finder

        # steering matrix of array : A
        self._A = get_steer_vector(self._search_grid, None)

    def estimate(self, R, k, include_endpoint=False, return_sp=False):
        R_inv = np.linalg.inv(R)  # Inverse of covariance matrix

        # sp 计算有待优化
        spectrum = np.diagonal((self._A.transpose(-1, -2).conj()@R_inv@self._A.transpose(-1, -2)).astype(np.float32))
        # 获取谱峰
        peak_indices = self._peak_finder(spectrum, include_endpoint)

        # sp and grid -> k DOAs
        n_peaks = len(peak_indices)
        if n_peaks < k:
            peak = self._search_grid[peak_indices]
            peak = np.concatenate([peak, np.full((k - len(peak)), np.nan)])
            if return_sp:
                return False, peak, spectrum
            else:
                return False, peak

        else:
            peak_values = spectrum[peak_indices]
            # Identify the k largest peaks.
            top_indices = np.argsort(peak_values)[-k:]
            # Filter out the peak indices of the k largest peaks.
            peak_indices = [peak_indices[i] for i in top_indices]
            peak_indices.sort()

            peak = self._search_grid[peak_indices]

            if return_sp:
                return True, peak, spectrum
            else:
                return True, peak