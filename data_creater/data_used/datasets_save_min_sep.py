import os
import numpy as np

from data_creater.signal_datasets import ULA_dataset
from data_creater.Create_classic_test_dataset import Create_equal_separate_theta
from data_creater.Create_k_source_dataset import Create_datasets

if __name__ == '__main__':
    # 基本参数设置
    snap = 10
    snr = 0
    k = 3
    M = 8
    signal_range = (-60, 60)
    rho = 1

    # 设置信号间隔
    # intervals = [[[0.5, 0.5]], [[1, 1]], [[2, 2]], [[4, 4]], [[6, 6]],
    #              [[8, 8]], [[10, 10]], [[12, 12]], [[14, 14]]]
    intervals = [[[4, 4]], [[6, 6]], [[8, 8]], [[10, 10]], [[12, 12]], [[14, 14]]]  # min_sep > 2
    # intervals = [[[10, 10]], [[20, 20]], [[30, 30]], [[40, 40]], [[50, 50]]]

    dir = f"/home/xd/DOA_code/open_source_code/data/ULA_data/test/M_{M}_k_{k}_snap_{snap}_snr_{snr}_rho{rho}_min_sep"
    if not os.path.exists(dir):
        os.makedirs(dir)

    step = 0.5  # 步长
    # step = 0.1  # 步长
    for sep in intervals:
        # 创建等间隔的信号
        theta_set = Create_equal_separate_theta(k, signal_range[0], signal_range[1], delta_theta_list=sep, step=step)

        # 创建数据集
        dataset = ULA_dataset(M, -60, 60, 1, rho)
        Create_datasets(dataset, k, theta_set, batch_size=5, snap=snap, snr=snr)

        # 保存数据集
        dataset.save_all_data(os.path.join(dir, f"test_sep_{sep}"))
