import copy
import os
import numpy as np
import scipy
import random
import torch

from utils.batch_matrix_operator import *
from data_creater.norm import l2_norm

from data_save.plot.plot_confusion_matrix import plot_confusion_matrix


# author zbb
# create all data needed for model train and test
class ULA_dataset:
    def __init__(self,
                 M=8,
                 st_angle=-90, ed_angle=90, step=1,
                 rho=0):
        """
        :param M: number of array elements
        :param st_angle/ed_angle/step: create the grid for model fitting
        :param rho: rho = 0 -> 1 / degree of array imperfection
        """
        self.M = M
        self.max_k = 3 * M  # max source_number to estimate
        self.d = 0.1
        self.array_pos = np.linspace(0, (self.M - 1) * self.d, M)
        self.signal_grid = np.arange(st_angle, ed_angle + 0.001, step)  # self.st_angle = self.signal_grid[0]

        # 矩阵A的阵列误差\array imperfection of steering matrix A
        # set rho = 0.0, if it doesn't add array imperfection
        self.MC_mtx, self.AP_mtx, self.pos_para = self.retrieve_array_imperfection_m(rho)

        # saved dataset
        # base data
        self.y_t = []
        self.ori_scm = []
        self.scm = []
        self.scm_vec = []

        self.tau_cat_scm = []
        # processed data
        self.ori_truth_scm = []
        self.truth_scm = []
        self.enhance_truth_scm = []

        self.enhance_scm = []
        self.FFT_scm = []
        # target
        self.doa = []
        self.subspace = []
        self.spatial_sp = []  # spatial_spectrum
        self.num_k = []
        self.sep_k = []
        self.sep_k_spatial = []

    def clear(self):
        # saved dataset
        # base data
        self.y_t = []
        self.ori_scm = []
        self.scm = []
        self.scm_vec = []
        self.tau_cat_scm = []
        # processed data
        self.ori_truth_scm = []
        self.truth_scm = []
        self.enhance_truth_scm = []

        self.enhance_scm = []
        self.FFT_scm = []
        # target
        self.doa = []
        self.subspace = []
        self.spatial_sp = []  # spatial_spectrum
        self.num_k = []
        self.sep_k = []
        self.sep_k_spatial = []

    # batch operation
    # TODO: snap 改成可变的，需要时修改
    def Create_DOA_data(self, k: int,
                        doas: np.ndarray,
                        snrs_db: np.ndarray,
                        s_t_type,
                        in_f=None,
                        snap=256,
                        snr_set=1,
                        **kwargs):
        """
        Args:
            k: 入射信号数量 / number of input signals
            doas: 所有入射信号对应的角度 / 2d np.ndarray, direction af arrival of signals
            snrs_db: 所有入射信号对应的信噪比(db) / 2d np.ndarray, snr of signals(db)
            s_t_type: 入射信号s(t)的形式 'f_i_input' or 'random_phase_input' or 'gauss_input' or 'zero_input'
                1.同频(窄带)信号输入 2.随机相位信号 3.高斯入射信号(最简单) 4.全0输入
            in_f: 所有入射信号对应的入射频率 / frequency of signals
            snap: 阵列的采样次数 / snap
            snr_set: 0/1,if 0,设置信号的功率为1，否则设置噪声功率为1
            **kwargs: other parameters

        Returns: 在dataset中保存模型训练要用到的input和output

        # x(t)=A*s(t)+n(t),先分别生成A,s(t),n(t),然后计算需要的参数
        """
        # TODO: 改create_dataset
        assert doas.shape[-1] == k and doas.shape == snrs_db.shape, 'error signal setting'
        batch_size = doas.shape[0]

        # calculate best f/lambda of incident signals
        c = 3 * 10 ** 8
        f = in_f or c / (2 * self.d)  # lambda = c/f = 2d
        lamda = c / f

        # build matrix A
        steer_vector = -1j * 2 * np.pi * (self.array_pos + self.pos_para) / lamda  # 生成-(2pi*j*d_i/lamda)
        horizon_vec = np.sin(doas / 180 * np.pi)
        A = np.exp(steer_vector[:, np.newaxis] * horizon_vec[:, np.newaxis, :])

        A = self.MC_mtx @ self.AP_mtx @ A  # influence of array imperfection

        # power of signal and noise
        if snr_set == 1:
            signal_power = to_num(snrs_db, 20)  # transform db to digit， snrs of amplitude, not power!!
            noise_power = np.ones((batch_size, self.M))
        elif snr_set == 0:
            one_sig = np.min(snrs_db, axis=-1, keepdims=True)  # 功率最小的入射信号功率设为1
            signal_power = to_num(snrs_db - one_sig, 20)
            noise_power = to_num(-one_sig, 20)  # boardcast

        # s(t) setting
        if s_t_type == 'gauss_input':
            signal = signal_power[..., np.newaxis] * (np.random.randn(*[batch_size, k, snap])
                                                      + 1j * np.random.randn(*[batch_size, k, snap])) / np.sqrt(2)
        elif s_t_type == 'zero_input':
            signal = np.zeros((batch_size, k, snap))
        else:
            raise ValueError('undefined input s(t) type')

        noise = noise_power[..., np.newaxis] * (np.random.randn(*[batch_size, self.M, snap]) + 1j * np.random.randn(
            *[batch_size, self.M, snap])) / np.sqrt(2)
        # calculate through complex64/float32
        A, signal, noise = A.astype(np.complex64), signal.astype(np.complex64), noise.astype(np.complex64)
        y_t = A @ signal + noise

        # calculate truth_scm
        ori_truth_scm = A @ batch_diag_matrices(signal_power ** 2) @ A.swapaxes(-1, -2).conj() + np.eye(self.M)
        ori_truth_scm = 1 / 2 * (ori_truth_scm + ori_truth_scm.swapaxes(-1, -2).conj()) \
            .astype(np.complex64)  # transform matrix to Hermitian matrix
        truth_scm = l2_norm(matrix_2_matrix_concat(ori_truth_scm), axes=(-1, -2)).astype(np.complex64)

        self.ori_truth_scm.extend(split_to_list(ori_truth_scm))
        self.truth_scm.extend(split_to_list(truth_scm))
        self.enhance_truth_scm.extend(split_to_list(l2_norm(matrix_2_enhance(ori_truth_scm), axes=(-1, -2))))

        # calculate received ori_scm for classic algorithm, and normalized scm, vec for training
        ori_scm = self.calculate_scm(y_t)
        scm = self.get_concat_scm(ori_scm)
        scm_vec = self.get_scm_vec(ori_scm)

        # add tau_cat_scm
        tau_cat_scm = np.concatenate([scm[:, 0], scm[:, 1]], axis=1)
        self.tau_cat_scm.extend(split_to_list(tau_cat_scm[:, None, ...]))

        self.y_t.extend(split_to_list(y_t))
        self.ori_scm.extend(split_to_list(ori_scm))
        self.scm.extend(split_to_list(scm))
        self.scm_vec.extend(split_to_list(scm_vec))

        # developed scheme requires enhance_scm, FFT_scm and so on
        enhance_scm = l2_norm(matrix_2_enhance(ori_scm), axes=(-1, -2))
        self.enhance_scm.extend(split_to_list(enhance_scm))
        self.FFT_scm.extend(self.get_FFT_scm(ori_scm))

        # get target
        subspace = self.get_sub_space(A, 'matrix')
        spatial_sp = self.get_spatial_sp(doas)

        self.subspace.extend(split_to_list(subspace))
        self.doa.extend(split_to_list(doas))
        self.spatial_sp.extend(split_to_list(spatial_sp))

        # estimate the number of source
        k_target = np.concatenate([np.ones((doas.shape[0], k)), np.zeros((doas.shape[0], self.max_k - k))], axis=-1)
        self.num_k.extend(split_to_list(k_target))
        sep_k = np.diff(doas, n=1, axis=-1)  # used in ASL algorithm
        self.sep_k.extend(split_to_list(sep_k))
        sep_k_spatial = self.get_spatial_sp(sep_k, self.signal_grid - self.signal_grid[0])  # grid of separation
        self.sep_k_spatial.extend(split_to_list(sep_k_spatial))

        return 0

    # get a steering matrix of the input; not batch operation
    # TODO： A 可以写成解析式，会更方便吗？
    def get_A(self, doas, in_f=None, array_imperfection=False):
        assert doas.ndim == 1, 'error input DOAs'

        # calculate best f/lambda of incident signals
        c = 3 * 10 ** 8
        f = in_f or c / (2 * self.d)  # lambda = c/f = 2d
        lamda = c / f

        # build matrix A
        if array_imperfection is True:
            steer_vector = -1j * 2 * np.pi * (self.array_pos + self.pos_para) / lamda  # 生成-(2pi*j*d_i/lamda)
        else:
            steer_vector = -1j * 2 * np.pi * self.array_pos / lamda
        horizon_vec = np.sin(doas / 180 * np.pi)
        A = np.exp(steer_vector[:, np.newaxis] * horizon_vec[np.newaxis, :])

        if array_imperfection is True:
            A = self.MC_mtx @ self.AP_mtx @ A

        return A.astype(np.complex64)
        # return A.astype(np.complex256)

    def __len__(self):
        return len(self.ori_scm)

    # root-music, esprit get target doa value through z
    # "Displacement" represents the input phase is phase difference across n array elements.
    def get_theta_fromz(self, z, displacement=1, in_f=None):
        c = 3 * 10 ** 8
        f = in_f or 1 / 2 * c / self.d

        ang_z = np.angle(z)
        sin_value = -ang_z / (2 * np.pi * f * self.d) * c / displacement

        sin_value.sort()
        deg_value = np.arcsin(sin_value) * 180 / np.pi
        return deg_value

    def theta_to_z(self, theta, displacement=1, in_f=None):
        # transform to radian value
        theta = theta / 180 * np.pi
        # sort theta in descendant order, then z will in increasing order
        theta.sort()
        theta = theta[::-1]
        c = 3 * 10 ** 8
        f = in_f or 1 / 2 * c / self.d

        z = np.exp(-1j * 2 * np.pi * f * self.d * np.sin(theta) * displacement / c)
        return z

    # simulate array imperfection matrix for ULA
    def retrieve_array_imperfection_m(self, rho):
        M = self.M

        mc_flag = True
        ap_flag = True
        pos_flag = True

        # create amp,phase,pos parameters for various M,
        # 设置不同阵元数M时的幅度,相位,位置偏差
        # amp_bias = np.array([0.0, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2] * (M // 8 + 1))
        # phase_bias = np.array([0.0, -30, -30, -30, -30, 30, 30, 30] * (M // 8 + 1))
        # pos_bias = np.array([0.0, -1, -1, -1, -1, 1, 1, 1] * (M // 8 + 1)) * 0.2
        amp_bias = np.array([0.0, 0.2, 0.2, 0.2, 0.2, -0.2, -0.2, -0.2] * (M // 8 + 1))
        phase_bias = np.array([0.0, -30, -30, -30, -30, 30, 30, 30] * (M // 8 + 1))
        pos_bias = np.array([0.0, -1, -1, -1, -1, 1, 1, 1] * (M // 8 + 1)) * 0.2

        # 幅相的偏移量,需要在阵元数M改变时改变
        amp_coef = rho * amp_bias[:M]
        phase_coef = rho * phase_bias[:M]
        # 位置误差偏移量
        pos_coef = rho * pos_bias[:M] * self.d

        # mutual coupling matrix
        if mc_flag:
            mc_para = rho * 0.3 * np.exp(1j * 60 / 180 * np.pi)
            MC_coef = mc_para ** np.array(np.arange(M))
            # 重要important
            # MC_mtx = scipy.linalg.toeplitz(MC_coef)
            MC_mtx = scipy.linalg.toeplitz(MC_coef, MC_coef)
        else:
            MC_mtx = np.identity(M)
        # amplitude & phase error
        if ap_flag:
            AP_coef = [(1 + amp_coef[idx]) * np.exp(1j * phase_coef[idx] / 180 * np.pi) for idx in range(M)]
            AP_mtx = np.diag(AP_coef)
        else:
            AP_mtx = np.identity(M)
        # sensor position error
        if pos_flag:
            pos_para = pos_coef
        else:
            pos_para = np.zeros([M])

        return MC_mtx, AP_mtx, pos_para

    # batch operator
    @staticmethod
    def calculate_scm(y_t):
        scm = y_t @ (y_t.swapaxes(-1, -2).conj())
        snap = y_t.shape[-1]
        scm = scm / snap
        scm = 1 / 2 * (scm + scm.swapaxes(-1, -2).conj())

        return scm

    # batch operator
    @staticmethod
    def get_concat_scm(ori_scm, set_zero=False):
        ori_scm = copy.copy(ori_scm)
        if set_zero is True:
            index = np.triu_indices(ori_scm.shape(-1))
            ori_scm[..., index] = 0  # TODO: test this function

        ori_scm = l2_norm(ori_scm, (-1, -2))
        scm = matrix_2_matrix_concat(ori_scm)

        return scm

    # batch operator
    @staticmethod
    def get_scm_vec(ori_scm):
        ori_scm = l2_norm(ori_scm, (-1, -2))
        scm_vec = matrix_2_vec(ori_scm)
        return scm_vec

    # batch operator
    @staticmethod
    def get_FFT_scm(ori_scm):
        FFT_scm = np.fft.fftn(ori_scm, axes=(-1, -2))
        FFT_scm = matrix_2_matrix_concat(l2_norm(FFT_scm, axes=(-1, -2)))

        return FFT_scm

    @staticmethod
    # reconstruct for signal subspace
    def get_sub_space(A: np.ndarray, mode='vector'):
        if mode == 'vector':
            # Hermitian+Toeplitz property of matrix, 重建一行就行
            sub_vec = A[..., 0:1, :] @ (A.swapaxes(-1, -2).conj())
            sub_vec = sub_vec[..., 0, :]
            sub_vec = np.concatenate([sub_vec.real(), sub_vec.imag()], axis=-1)
            # the angle of A contains all the information, 恢复A的相位信息
            # reconstruct_vector = np.angle(sub_vec)
            return sub_vec
        elif mode == 'matrix':
            sub_space = A @ (A.swapaxes(-1, -2).conj())
            sub_space = 1 / 2 * (sub_space + sub_space.swapaxes(-1, -2).conj())
            sub_space = matrix_2_matrix_concat(sub_space)

            return sub_space

    # TODO: test
    def get_spatial_sp(self, doas, grid=None):
        if grid is None:
            grid = self.signal_grid
        # grid Left-closed, right-open interval./左闭右开区间
        place = np.searchsorted(grid, doas)
        step = grid[1] - grid[0]
        score_left = (step - (doas % step)) / step
        score_right = (doas % step) / step

        # initial the spatial spectrum
        spatial_sp = np.zeros((doas.shape[0], len(grid)))
        # broadcast of index
        spatial_sp[np.arange(doas.shape[0])[:, None], place - 1] += score_left
        spatial_sp[np.arange(doas.shape[0])[:, None], place] += score_right

        return spatial_sp.astype(np.float32)


class model_fit_type:
    def __init__(self, element1, element2):
        self._source = element1
        self._target = element2

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def set_elements(self, element1, element2):
        self._source = element1
        self._target = element2


class array_Dataloader:
    def __init__(self, dataset, batch_size=8, shuffle=True, load_style='np', input_type='scm', output_type='doa'):
        """
        Args:
            dataset: 加载的ULA_DOA_dataset
            batch_size: 批处理大小
            shuffle: 数据装载时是否随机打乱次序
            load_style: 'np' or 'torch',以np.ndarray或者torch形式装载
            input_type: 控制data_loader中x载入的数据
            output_type: 控制data_loader中y载入的数据
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_data = len(dataset)
        self.index = list(range(self.num_data))
        self.data_count = 0
        if self.shuffle:
            random.shuffle(self.index)

        self.load_style = load_style
        # set the type of x,y of dataloader
        self.data_type = model_fit_type(input_type, output_type)
        # TODO: dataloader can be implemented in a much simply way
        self.data = tuple(
            zip(self.dataset.__dict__[self.data_type.source], self.dataset.__dict__[self.data_type.target]))

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_count >= self.num_data:
            self.data_count = 0
            if self.shuffle:
                random.shuffle(self.index)
            raise StopIteration
        else:
            batch_index = self.index[self.data_count:self.data_count + self.batch_size]
            data_batch = [self.data[i] for i in batch_index]
            # 两个返回值时self.dataset[i] 是 tuple 类型
            self.data_count += self.batch_size
            # print(type(data_batch))

            # 添加对batch的处理
            # 返回都是元组形式
            data_batch = tuple(zip(*data_batch))
            input_data, labels = data_batch
            if self.load_style == 'np':
                input_data = np.array(input_data)
                labels = np.array(labels)
            elif self.load_style == 'torch':
                input_data = torch.as_tensor(np.array(input_data))
                labels = torch.as_tensor(np.array(labels))

            return input_data, labels


def split_to_list(array):
    """
    Split a NumPy array along the first dimension into a list of arrays.

    Parameters:
    array (np.ndarray): The input array to be split.

    Returns:
    list: A list where each element is an array corresponding to a slice of the original array along the first dimension.
    """
    return [row for row in array]


if __name__ == '__main__':
    from data_creater.Create_k_source_dataset import Create_random_k_input_theta, Create_datasets

    dataset = ULA_dataset(8, -90, 90, 1, 2)
    snap = 100
    snr = 10
    k = 3
    theta_set = Create_random_k_input_theta(k, -60, 60, 100)
    Create_datasets(dataset, k, theta_set, 100, snap, snr)

    # plot covariance matrix
    # path = '/home/xd/zbb_Code/研二code/DOA_deep_learn/results/SCM/snap_100_snr_10_rho_2'
    # for i, scm in enumerate(dataset.ori_scm):
    #     plot_confusion_matrix(scm, os.path.join(path, f'scm_{i}.png'))

    dataloader = array_Dataloader(dataset, 32, True, load_style='torch', input_type='scm', output_type='spatial_sp')
    for step, data_batch in enumerate(dataloader):
        scm, spatial_sp = data_batch
        pass
