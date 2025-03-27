from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import find_peaks
from models.subspace_model.music import peak_finder


# nn 进行空间谱估计和其他空间谱估计算法的区别是：无法进一步在前向传播中细化空间谱网格
# 不必要继承nn.Module, ABC
class Grid_Based_network(nn.Module):
    def __init__(self, start_angle=-60, end_angle=60, step=1, threshold=0):
        super().__init__()
        # threshold 是寻峰函数中使用的阈值
        # super(Grid_Based_network, self).__init__()
        # ABC.__init__()
        # grid 存在gpu上
        # self._grid = np.arange(start_angle, end_angle + 0.0001, step=step)
        self.register_buffer('_grid', torch.arange(start_angle, end_angle + 0.0001, step=step))
        self._out_dim = self._grid.shape[0]
        self.threshold = threshold

        self._find_peaks = peak_finder

    @property
    def grid(self):
        return self._grid

    @property
    def out_dim(self):
        return self._out_dim

    def grid_to_theta(self, sp_batch: torch.Tensor, k, return_sp=False):
        # sp_batch (batch, grid+1)
        assert sp_batch.ndim == 2, ''
        b = sp_batch.shape[0]
        device = sp_batch.device
        sp_diff = torch.diff(sp_batch, n=1, dim=-1)
        sp_peak = (sp_diff[:, :-1] >= 0) & (sp_diff[:, 1:] <= 0)  # bool (batch, grid-1)
        sp_peak = torch.cat([torch.zeros((b, 1), device=device, dtype=torch.bool), sp_peak,
                             torch.zeros((b, 1), device=device, dtype=torch.bool)], dim=-1)  # bool (batch, grid+1)
        sp_peak = torch.where(sp_peak, sp_batch, torch.zeros_like(sp_batch, device=device) - self.threshold)
        # sp_peak = sp_batch[sp_peak]

        sp_peak, idx = sp_peak.sort(dim=-1, descending=True)
        succ = sp_peak[:, k - 1].bool().to(sp_peak.device)
        theta = self.grid[idx[:, :k]]
        theta, _ = theta.sort(dim=-1)

        if return_sp:
            return succ, theta, sp_batch
        else:
            return succ, theta

    # 对允许对gpu-torch操作
    # 可以梯度回传
    def grid_to_thetas_method1(self, sp_batch: torch.Tensor, k, return_sp=False):
        '''
        Args:
            sp_batch:[batch,sp],  spectrum that model estimates thorough the network
            the __len__ of sp should 对应 self._grid
            k: number of signals input
            return_sp: return spectrum model estimates or not

        Returns:
            thetas:[batch,thetas] location of incident signal
        '''
        device = sp_batch.device
        # 直接输入sp,则增加batch维度
        if sp_batch.ndim == 1:
            sp_batch = sp_batch.unsqueeze_(0)
        batch = sp_batch.shape[0]

        # torch.where -> 得到不等长的tensor
        # sp_over_thre:sp_batch 中满足条件的坐标(索引)
        sp_over_thre = torch.where(sp_batch > self.threshold)
        # tuple of idx_1,2
        split_point_1 = torch.where(torch.diff(sp_over_thre[0], n=1, dim=0) == 1)[0] + 1
        split_point_2 = torch.where(torch.diff(sp_over_thre[1], n=1, dim=0) != 1)[0] + 1
        # 数组的切分点
        split_point = torch.unique(torch.concatenate([split_point_1, split_point_2]))
        # 统计每个数据有多少谱峰
        # torch.histc()
        value_all = sp_batch[sp_over_thre[0], sp_over_thre[1]]
        grid = self.grid.repeat(batch)
        # 一次性取所有grid值
        grid_all = grid[sp_over_thre[1]]
        # 一次计算所有能量值
        # energy_all = value_all*grid_all
        # 计数后续改进成split_point_1,2 求区间总数,后续改进
        _, count_i = torch.unique(sp_over_thre[0][split_point], return_counts=True)
        count_i[0] += 1
        # 区间数量最大值
        len_i = torch.max(count_i)
        # split
        # tensor_split ->index 在cpu上？
        value_split = list(torch.tensor_split(value_all, split_point.cpu()))
        grid_split = list(torch.tensor_split(grid_all, split_point.cpu()))
        # 区间长度补至相同
        value_2d = torch.transpose(torch.nn.utils.rnn.pad_sequence(value_split), 0, 1)
        grid_2d = torch.transpose(torch.nn.utils.rnn.pad_sequence(grid_split), 0, 1)
        # 也可以对2d tensor做split
        # 3d数组（最大区间长度设为100）
        value_3d = torch.zeros((batch, len_i, value_2d.shape[-1]), device=device)
        grid_3d = torch.zeros((batch, len_i, grid_2d.shape[-1]), device=device)
        count = 0
        # 尝试用cuda并行赋值
        # 对batch操作,赋值效率低
        for i in range(batch):
            value_3d[i, 0:count_i[i]] = value_2d[count:count + count_i[i]]
            grid_3d[i, 0:count_i[i]] = grid_2d[count:count + count_i[i]]
            count = count + count_i[i]
        energy = torch.sum(value_3d, dim=-1)
        top_value, top_idx = torch.topk(energy, k=k, dim=-1)
        # 得到batch,k,区间
        # .repeat 不用torch.
        # torch.sort idx 大小排序
        value_choise = value_3d[torch.unsqueeze(torch.arange(0, batch, 1, device=device), 1).repeat(1, k),
                       torch.sort(top_idx, dim=-1)[0], :]
        # 计算权重
        value_choise = value_choise / torch.sum(value_choise, dim=-1, keepdim=True)
        grid_choise = grid_3d[torch.unsqueeze(torch.arange(0, batch, 1, device=device), 1).repeat(1, k),
                      torch.sort(top_idx, dim=-1)[0], :]
        theta = torch.sum(value_choise * grid_choise, -1)

        if return_sp:
            return theta, sp_batch
        else:
            return theta

    # 不能对batch批量操作,极慢
    # 对允许对gpu-torch操作
    # 可以梯度回传
    def grid_to_thetas_method1_old(self, sp_batch: torch.Tensor, k, return_sp=False):
        '''
        Args:
            sp_batch:[batch,sp],  spectrum that model estimates thorough the network
            the __len__ of sp should 对应 self._grid
            k: number of signals input
            return_sp: return spectrum model estimates or not

        Returns:
            thetas:[batch,thetas] location of incident signal
        '''
        # 直接输入sp,则增加batch维度
        if sp_batch.ndim == 1:
            sp_batch = sp_batch.unsqueeze_(0)
        thetas_batch = []
        # tensor 必须等长,torch.where后恢复不了
        # 遍历第0个维度,对每个sp操作
        for sp in sp_batch:
            # 满足条件的index
            sp_index = torch.where(sp > self.threshold)[0]
            sp_diff = torch.diff(sp_index, n=1, dim=-1)
            # 满足条件的index划分成连续区间
            # torch.split 和numpy不一样
            cond_indices = torch.tensor_split(sp_index, tuple(torch.where(sp_diff != 1)[0] + 1), dim=-1)

            # 存储若干个谱峰值
            peaks = []
            peaks_idx = []
            if len(cond_indices) >= k:
                # torch快还是list快？
                # peaks_energy = torch.zeros((0,))
                peaks_energy = []
                for cond_index in cond_indices:
                    peaks.append(sp[cond_index])
                    peaks_idx.append(self._grid[cond_index])
                    # 升维，下文concat
                    peaks_energy.append(torch.sum(sp[cond_index]))
                # torch.tensor 直接把 list 变到cpu上
                # peaks_energy = torch.tensor(peaks_energy)
                # peaks_energy = torch.concatenate(peaks_energy, dim=0)
                peaks_energy = torch.stack(peaks_energy)
                # idx 变成peaks的index
                # 取能量最大的peak进行运算，顺序也是按能量大小排序
                max_peaks_idx = torch.argsort(peaks_energy)[-k:]
                peaks_weight = [peaks[i] / torch.sum(peaks[i]) for i in max_peaks_idx]
                peaks_idx = [peaks_idx[i] for i in max_peaks_idx]
                thetas = []
                for i in range(k):
                    thetas.append(torch.sum(peaks_weight[i] * peaks_idx[i]))
                # torch.Tensor 会自动转到cpu, 用torch.stack
                thetas, _ = torch.sort(torch.stack(thetas))
                # torch.concatenate([thetas_batch, torch.unsqueeze(thetas, 0)], dim=0, out=thetas_batch)
                thetas_batch.append(thetas)
            else:
                print('no enough peaks')
                for cond_index in cond_indices:
                    peaks.append(sp[cond_index])
                    peaks_idx.append(self._grid[cond_index])
                peaks_weight = [peaks[i] / torch.sum(peaks[i]) for i in range(len(cond_indices))]
                thetas = []
                for i in range(len(cond_indices)):
                    thetas.append(torch.sum(peaks_weight[i] * peaks_idx[i]))
                # 谱峰不足先补0,好训练
                thetas = F.pad(torch.tensor(thetas), (0, 3 - len(thetas)), 'constant', 0)
                # thetas,_ = torch.sort(torch.tensor(thetas))
                thetas, _ = torch.sort(thetas)
                thetas_batch.append(thetas)

        if return_sp:
            return torch.stack(thetas_batch), sp_batch
        else:
            return torch.stack(thetas_batch)

    # 调用numpy,scipy 的 寻峰函数,不适用于gpu-torch
    # 梯度回传不了,无法训练,只能测试
    def grid_to_thetas_method2(self, sp_s, k):
        # 传统方法加入对batch的操作
        # 要将torch转化为numpy操作,numpy和torch的平衡
        thetas = torch.zeros(0, k, device=self.device)
        # zip 返回元组
        # for sp in zip(sp_s.detach().cpu().numpy()):
        for sp in sp_s.detach().cpu().numpy():
            theta = self.grid_to_theta_method2(sp, k)
            thetas = torch.vstack([thetas, theta])
            # torch.vstack([thetas, theta], out=thetas) 报错,可恶啊
            # torch.vstack([thetas, theta.from_numpy()], out=thetas)
        return thetas

    # 训练时，未训练好的网络用grid_to_theta 时很容易出现谱峰数不足，导致报错的情况
    # 不能针对batch操作，在测试时使用
    # 只能对numpy处理
    def grid_to_theta_method2(self, sp, k):
        '''
        Args:
            sp:  spectrum that model estimates thorough the network
            k: number of signals input

        Returns:
            theta: location of incident signal
        '''
        # peak_finder 不能直接调用scipy
        peak_indices = self._find_peaks(sp)

        n_peaks = len(peak_indices)
        if n_peaks < k:
            theta = self._grid[peak_indices]
            raise ValueError(f'get theta: {theta} <= {k}, not enough peaks')
        else:
            peak_values = sp[peak_indices]
            top_indices = np.argsort(peak_values)[-k:]
            peak_indices = [peak_indices[i] for i in top_indices]
            # ?
            # peak_indices = peak_values[top_indices]
            peak_indices.sort()
            theta = self._grid[peak_indices]
        return theta


def list_split(list, sep_sequence):
    i = 0
    return_list = []
    for sep in sep_sequence:
        return_list.append(list[i:i + sep])
        i = i + sep
    return return_list


if __name__ == '__main__':
    grid_net = Grid_Based_network(-60, 60, 1)
    grid_net.to('cuda')
    sp = torch.randn(10, 121).to('cuda')
    thetas = grid_net.grid_to_theta(sp, 3)
    print(thetas)
