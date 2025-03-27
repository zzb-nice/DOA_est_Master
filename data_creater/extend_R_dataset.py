import copy

import numpy as np
import scipy
import random
import torch

from utils.batch_matrix_operator import *
from data_creater.norm import l2_norm


# author zbb
# extend covariance matrix R to extend matrix R+
class extend_R:
    def __init__(self,
                 M=8,
                 M_extend=64,
                 rho=0):
        """
        :param M: number of array elements
        :param M: number of extended array elements
        :param st_angle/ed_angle/step: create the grid for model fitting
        :param rho: rho = 0 -> 1 / degree of array imperfection of original array
        """
        self.M = M
        self.M_extend = M_extend
        self.d = 0.01
        self.array_pos = np.linspace(0, (self.M - 1) * self.d, M)
        self.array_pos_ext = np.linspace(0, (self.M_extend - 1) * self.d, M_extend)

        # set rho = 0.0, if it doesn't add array imperfection
        self.MC_mtx, self.AP_mtx, self.pos_para = self.retrieve_array_imperfection_m(rho)

        # original scm
        self.ori_scm = []
        self.scm = []
        self.scm_vec = []
        self.scm_vec2 = []
        # extend scm
        self.ori_ext_scm = []
        self.ext_scm = []
        self.ext_scm_vec = []
        self.ext_scm_vec2 = []
        # truth scm
        self.ori_truth_scm = []
        self.truth_scm = []
        self.truth_scm_vec = []
        # truth extend scm
        self.ori_truth_ext_scm = []
        self.truth_ext_scm = []
        self.truth_ext_scm_vec = []

    # batch operation
    def Create_DOA_data(self, num_signals: int,
                        doas: np.ndarray,
                        snrs_db: np.ndarray,
                        s_t_type,
                        in_f=None,
                        snap=256,
                        **kwargs):
        """
        Args:
            num_signals: 入射信号数量 / number of input signals
            doas: 所有入射信号对应的角度 / 2d np.ndarray, direction af arrival of signals
            snrs_db: 所有入射信号对应的信噪比(db) / 2d np.ndarray, snr of signals(db)
            s_t_type: 入射信号s(t)的形式 'f_i_input' or 'random_phase_input' or 'gauss_input' or 'zero_input'
                1.同频(窄带)信号输入 2.随机相位信号 3.高斯入射信号(最简单) 4.全0输入
            in_f: 所有入射信号对应的入射频率 / frequency of signals
            snap: 阵列的采样次数 / snap
            **kwargs: other parameters

        Returns: 在dataset中保存模型训练要用到的input和output

        # x(t)=A*s(t)+n(t),先分别生成A,s(t),n(t),然后计算需要的参数
        """
        # TODO: 改create_dataset
        assert doas.shape[-1] == num_signals and doas.shape == snrs_db.shape, 'error signal setting'
        batch_size = doas.shape[0]

        # calculate best f/lambda of incident signals
        c = 3 * 10 ** 8
        f = in_f or c / (2 * self.d)  # lambda = c/f = 2d
        lamda = c / f

        # build matrix A
        steer_vector = -1j * 2 * np.pi * (self.array_pos + self.pos_para) / lamda  # 生成-(2pi*j*d_i/lamda)
        horizon_vec = np.sin(doas / 180 * np.pi)
        A = np.exp(steer_vector[:, np.newaxis] * horizon_vec[:, np.newaxis, :])
        # build A_extend, which not be influenced by array imperfection
        steer_vector = -1j * 2 * np.pi * self.array_pos_ext / lamda
        A_ext = np.exp(steer_vector[:, np.newaxis] * horizon_vec[:, np.newaxis, :])

        A = self.MC_mtx @ self.AP_mtx @ A

        # s(t) setting
        snrs = to_num(snrs_db, 20)  # transform db to digit， snrs of amplitude, not power!!
        if s_t_type == 'gauss_input':
            signal = snrs[..., np.newaxis] * (np.random.randn(*[batch_size, num_signals, snap])
                                              + 1j * np.random.randn(*[batch_size, num_signals, snap])) / np.sqrt(2)
        elif s_t_type == 'zero_input':
            signal = np.zeros((batch_size, num_signals, snap))
        else:
            raise ValueError('undefined input s(t) type')

        noise = (np.random.randn(*[batch_size, self.M, snap]) + 1j * np.random.randn(
            *[batch_size, self.M, snap])) / np.sqrt(2)
        noise_ext = (np.random.randn(*[batch_size, self.M_extend, snap]) + 1j * np.random.randn(
            *[batch_size, self.M_extend, snap])) / np.sqrt(2)
        # calculate through complex64/float32
        A, signal, noise = A.astype(np.complex64), signal.astype(np.complex64), noise.astype(np.complex64)
        y_t = A @ signal + noise
        y_ext = A_ext @ signal + noise_ext
        y_ext, noise_ext = y_ext.astype(np.complex64), noise_ext.astype(np.complex64)

        # calculate truth_scm
        ori_truth_scm = A @ batch_diag_matrices(snrs ** 2) @ A.swapaxes(-1, -2).conj() + np.eye(self.M)
        ori_truth_scm = 1 / 2 * (ori_truth_scm +
                                 ori_truth_scm.swapaxes(-1, -2).conj())  # transform matrix to Hermitian matrix
        truth_scm = l2_norm(matrix_2_matrix_concat(ori_truth_scm), axes=(-1, -2))

        self.ori_truth_scm.extend(split_to_list(ori_truth_scm))
        self.truth_scm.extend(split_to_list(truth_scm))
        self.truth_scm_vec.extend(split_to_list(l2_norm(self.get_scm_vec(ori_truth_scm), axes=(-1, -2))))

        # calculate received ori_scm for classic algorithm, and normalized scm, vec for training
        ori_scm = self.calculate_scm(y_t)
        scm = self.get_concat_scm(ori_scm)
        scm_vec = self.get_scm_vec(ori_scm)
        scm_vec2 = self.get_scm_vec2(ori_scm)

        self.ori_scm.extend(split_to_list(ori_scm))
        self.scm.extend(split_to_list(scm))
        self.scm_vec.extend(split_to_list(scm_vec))
        self.scm_vec2.extend(split_to_list(scm_vec2))

        ori_ext_scm = self.calculate_scm(y_ext)
        ext_scm = self.get_concat_scm(ori_ext_scm)
        ext_scm_vec = self.get_scm_vec(ori_ext_scm)
        ext_scm_vec2 = self.get_scm_vec2(ori_ext_scm)

        self.ori_ext_scm.extend(split_to_list(ori_ext_scm))
        self.ext_scm.extend(split_to_list(ext_scm))
        self.ext_scm_vec.extend(split_to_list(ext_scm_vec))
        self.ext_scm_vec2.extend(split_to_list(ext_scm_vec2))

        # calculate truth_scm
        ori_truth_ext_scm = A_ext @ batch_diag_matrices(snrs ** 2) @ A_ext.swapaxes(-1, -2).conj() \
                            + np.eye(self.M_extend)
        ori_truth_ext_scm = 1 / 2 * (ori_truth_ext_scm +
                                     ori_truth_ext_scm.swapaxes(-1, -2).conj())  # transform matrix to Hermitian matrix
        truth_ext_scm = l2_norm(matrix_2_matrix_concat(ori_truth_ext_scm), axes=(-1, -2))

        self.ori_truth_ext_scm.extend(split_to_list(ori_truth_ext_scm))
        self.truth_ext_scm.extend(split_to_list(truth_ext_scm))
        self.truth_ext_scm_vec.extend(split_to_list(l2_norm(self.get_scm_vec(ori_truth_ext_scm), axes=(-1, -2))))

        return 0

    def get_ext_A(self, doas, in_f=None):
        assert doas.ndim == 1, 'error input DOAs'

        # calculate best f/lambda of incident signals
        c = 3 * 10 ** 8
        f = in_f or c / (2 * self.d)  # lambda = c/f = 2d
        lamda = c / f

        # build matrix A
        horizon_vec = np.sin(doas / 180 * np.pi)
        steer_vector = -1j * 2 * np.pi * self.array_pos_ext / lamda
        A_ext = np.exp(steer_vector[:, np.newaxis] * horizon_vec[np.newaxis, :])

        return A_ext.astype(np.complex64)

    def __len__(self):
        return len(self.ori_scm)

    # batch operator
    @staticmethod
    def calculate_scm(y_t):
        scm = y_t @ (y_t.swapaxes(-1, -2).conj())
        snap = y_t.shape[-1]
        scm = scm / snap

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

    @staticmethod
    def get_scm_vec2(ori_scm):
        ori_scm = l2_norm(ori_scm, (-1, -2))
        scm_vec = matrix_2_vec2(ori_scm)
        return scm_vec

    # simulate array imperfection matrix for ULA
    def retrieve_array_imperfection_m(self, rho):
        M = self.M

        mc_flag = True
        ap_flag = True
        pos_flag = True

        # create amp,phase,pos parameters for various M,
        # 设置不同阵元数M时的幅度,相位,位置偏差
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
            MC_mtx = scipy.linalg.toeplitz(MC_coef)
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


def split_to_list(array):
    """
    Split a NumPy array along the first dimension into a list of arrays.

    Parameters:
    array (np.ndarray): The input array to be split.

    Returns:
    list: A list where each element is an array corresponding to a slice of the original array along the first dimension.
    """
    return [row for row in array]


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