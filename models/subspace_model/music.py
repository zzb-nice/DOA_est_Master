import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import maximum_filter


# TODO：添加对batch的操作，然后改到torch上，可以调用train()函数
# 获得噪声子空间 -> 空间谱 -> theta
# frequency and k should be known ahead to implement Music and Root Music algorithm
class Music:
    def __init__(self, get_steer_vector, start=-90, end=90, step=0.1):
        self._search_grid = np.arange(start, end + 0.001, step)
        self._peak_finder = peak_finder

        # steering matrix of array : A
        self._A = get_steer_vector(self._search_grid, None)

    def estimate(self, R, k, include_endpoint=False, return_sp=False):
        U_n = get_noise_subspace(R, k)

        # sp:np.float64/32 ？
        spectrum = f_music(self._A, U_n).astype(np.float32)
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


class Music_2D:
    def __init__(self, get_steer_vector, start_theta=-180, end_theta=180, start_phi=0, end_phi=60, step=1):
        self._search_grid_theta = np.arange(start_theta, end_theta + 0.001, step)
        self._search_grid_phi = np.arange(start_phi, end_phi + 0.001, step)
        grid = np.meshgrid(self._search_grid_theta, self._search_grid_phi)
        self.grid = np.stack([grid[0], grid[1]], axis=0)

        # steering matrix of array : A
        self._A = np.swapaxes(get_steer_vector(self.grid, None), 0, 1)  # N_phi,M,N_theta -> M,N_phi,N_theta

    def estimate(self, R, k, include_endpoint=False, return_sp=False):
        U_n = get_noise_subspace(R, k)

        # sp:np.float64/32 ？
        spectrum = f_music_2d(self._A, U_n).astype(np.float32)
        local_max = maximum_filter(spectrum, 3)
        peak_indices = np.argwhere(local_max == spectrum)   # phi,theta

        # sp and grid -> k DOAs
        n_peaks = len(peak_indices)
        if n_peaks < k:
            peak = self.grid[:, peak_indices[:, 0], peak_indices[:, 1]]
            peak = np.concatenate([peak, np.full((2, k - peak.shape[-1]), np.nan)],axis=-1)
            if return_sp:
                return False, peak, spectrum
            else:
                return False, peak

        else:
            peak_values = spectrum[peak_indices[:, 0], peak_indices[:, 1]]
            # Identify the k largest peaks.
            top_indices = np.argsort(peak_values)[-k:]
            # Filter out the peak indices of the k largest peaks.
            peak_indices = np.array([peak_indices[i, :] for i in top_indices])
            # 按第2列排序,按theta排序
            idx = np.argsort(peak_indices[:, 1])
            peak_indices = peak_indices[idx, :]

            peak = self.grid[:, peak_indices[:, 0], peak_indices[:, 1]]  # (2,k)

            if return_sp:
                return True, peak, spectrum
            else:
                return True, peak


class Root_Music:
    def __init__(self, theta_fromz):
        self.z_to_theta = theta_fromz

    # 绘制z在复平面的图
    def plot_z(self, z):
        pass

    def estimate(self, R, k, return_z=False):
        M = R.shape[0]
        U_n = get_noise_subspace(R, k)
        Gn = U_n @ U_n.T.conj()
        # =========
        # TODO: test the algorithm and test if Gn is Hermitian matrix or not
        # Gn = 1/2*(Gn+Gn.conj())
        # coef = np.zeros((M - 1,), dtype=np.complex_)
        # for i in range(1, M):
        #     coef[i - 1] += np.sum(np.diag(Gn, i))
        # coef = np.hstack((coef[::-1].conj(), np.sum(np.diag(Gn)), coef))
        # =========

        coef = np.zeros((2 * M - 1), dtype=np.complex64)
        for i in range(-(M - 1), M - 1 + 1):  # 共2M-1个系数,2M-2个根
            # 多项式首项是coef[0]
            coef[i + M - 1] = np.sum(np.diag(Gn, -i))

        z = np.roots(coef)
        # TODO： 为啥舍弃圆外点
        z = z[np.abs(z) <= 1]
        if len(z) < k:
            theta = np.concatenate([self.z_to_theta(z), np.full((k - len(z)), np.nan)])
            if return_z:
                return False, theta, z
            else:
                return False, theta
        else:
            sorted_indices = np.argsort(np.abs(1 - np.abs(z)))
            z = z[sorted_indices[:k]]
            if return_z:
                # z -> 目标角度
                return True, self.z_to_theta(z), z
            else:
                return True, self.z_to_theta(z)


def f_music(A, U_n):
    r"""Computes the classical MUSIC spectrum

    This is a vectorized implementation of the spectrum function:

    .. math::
        P_{\mathrm{MUSIC}}(\theta)
        = \frac{1}{\mathbf{a}^H(\theta) \mathbf{E}_\mathrm{n}
                   \mathbf{E}_\mathrm{n}^H \mathbf{a}(\theta)}

    Args:
        A: m x k steering matrix of candidate direction-of-arrivals, where
            m is the number of sensors and k is the number of candidate
            direction-of-arrivals.
        U_n: m x d matrix of noise eigenvectors, where d is the dimension of the
            noise subspace.
    """
    v = U_n.T.conj() @ A
    return np.reciprocal(np.sum(v * v.conj(), axis=0).real)


def f_music_2d(A, U_n):
    v = np.einsum('mn,mjk->njk', U_n.conj(), A)
    return np.reciprocal(np.sum(v * v.conj(), axis=0).real)


def get_noise_subspace(R, k):
    """
    Gets the noise eigenvectors.

    Args:
        R: Covariance matrix.
        k: Number of sources.
    """
    eigenvalue, U = np.linalg.eigh(R)
    U_s = U[:, -k:]
    U_n = U[:, :-k]
    # Note: eigenvalues are sorted in ascending order.
    return U_n


def peak_finder(sp, include_endpoint=False):
    peak_indices = find_peaks(sp)[0]
    if include_endpoint and sp[0] >= sp[1]:
        peak_indices = np.concatenate([np.array([0]), peak_indices], axis=0)
    if include_endpoint and sp[-1] >= sp[-2]:
        peak_indices = np.concatenate([peak_indices, np.array([len(sp) - 1])], axis=0)

    return peak_indices


def numpy_topk(x, k, largest=True):
    """
    实现类似于 PyTorch 的 topk 函数
    :param x: 输入的 numpy 数组
    :param k: 返回前 k 个最大值或最小值
    :param largest: 如果为 True，则返回最大值；如果为 False，则返回最小值
    :return: top k 个值和它们的索引
    """
    if largest:
        # 取前 k 个最大值
        indices = np.argpartition(-x, k)[:k]  # 获取前 k 个元素的索引
        topk_values = x[indices]
        sorted_indices = np.argsort(-topk_values)  # 对前 k 个元素排序
        return topk_values[sorted_indices], indices[sorted_indices]
    else:
        # 取前 k 个最小值
        indices = np.argpartition(x, k)[:k]  # 获取前 k 个元素的索引
        topk_values = x[indices]
        sorted_indices = np.argsort(topk_values)  # 对前 k 个元素排序
        return topk_values[sorted_indices], indices[sorted_indices]
