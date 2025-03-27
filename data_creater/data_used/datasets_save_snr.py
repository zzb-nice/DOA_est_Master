import os

import numpy as np

from data_save.plot.plot_confusion_matrix import plot_confusion_matrix
from data_creater.signal_datasets import ULA_dataset, array_Dataloader

from data_creater.Create_k_source_dataset import Create_random_k_input_theta, Create_datasets
from data_creater.Create_classic_test_dataset import Create_monte_carlo_theta

if __name__ == '__main__':
    snap = 10
    snrs = [-20, -15, -10, -5, 0, 5]
    k = 3
    M = 8
    signal_range = [-60, 60]
    grid_size = 1
    rho = 0.5
    dir = f"/home/xd/DOA_code/open_source_code/data/ULA_data/test/M_{M}_k_{k}_test1_rho{rho}"

    if not os.path.exists(dir):
        os.makedirs(dir)

    # # 训练集初始化
    # train_theta_set = Create_random_k_input_theta(k, -60, 60, 50000, min_delta_theta=1)
    # for snr in snrs:
    #     train_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(train_dataset, k, train_theta_set, 100, snap, snr, snr_set=0)
    #     train_dataset.save_all_data(os.path.join(dir, f"train_dataset_snr_{snr}"))
    #
    # # 验证集初始化
    # val_theta_set = Create_random_k_input_theta(k, -60, 60, 20000, min_delta_theta=3)
    # for snr in snrs:
    #     val_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(val_dataset, k, val_theta_set, 100, snap, snr, snr_set=0)
    #     val_dataset.save_all_data(os.path.join(dir, f"val_dataset_snr_{snr}"))

    # random_input 测试集
    test_theta_set = Create_random_k_input_theta(k, -60, 60, 5000, min_delta_theta=3)
    for snr in snrs:
        test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
        Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
        test_dataset.save_all_data(os.path.join(dir, f"test_random_input_snr_{snr}"))

    # monte_carlo 测试集
    test_theta_set = Create_monte_carlo_theta([np.array([10, 13, 16])], repeat_num=5000)
    for snr in snrs:
        test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
        Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
        test_dataset.save_all_data(os.path.join(dir, f"test_monte_carlo[10,13,16]_snr_{snr}"))

    test_theta_set = Create_monte_carlo_theta([np.array([10, 20, 30])], repeat_num=5000)
    for snr in snrs:
        test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
        Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
        test_dataset.save_all_data(os.path.join(dir, f"test_monte_carlo[10,20,30]_snr_{snr}"))

    # # 更多入射信号时, linspace 测试集
    # test_theta_set = Create_monte_carlo_theta([np.linspace(-36, 36, 7)], repeat_num=5000)
    # for snr in snrs:
    #     test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
    #     test_dataset.save_all_data(os.path.join(dir, f"test_linspace(-36, 36, 7)_snr_{snr}"))
    #
    # test_theta_set = Create_monte_carlo_theta([np.linspace(-48, 48, 7)], repeat_num=5000)
    # for snr in snrs:
    #     test_dataset = ULA_dataset(M, signal_range[0], signal_range[1], grid_size, rho)
    #     Create_datasets(test_dataset, k, test_theta_set, 100, snap, snr, snr_set=0)
    #     test_dataset.save_all_data(os.path.join(dir, f"test_linspace(-48, 48, 7)_snr_{snr}"))
