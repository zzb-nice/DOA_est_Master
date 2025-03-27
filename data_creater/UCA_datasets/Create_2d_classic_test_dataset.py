import numpy as np
import itertools
import time


def Create_monte_carlo_theta(all_theta=None, repeat_num=1000):
    theta_set = []
    if all_theta is None:
        all_theta = [np.array([[10.1, 13.3, 15], [13, 16, 20]])]
    for theta in all_theta:
        for i in range(repeat_num):
            theta_set.append(theta)

    return np.array(theta_set).astype(np.float32)


# def Create_equal_separate_theta(k=3,
#                                 start_angle=-60, end_angle=60,
#                                 delta_theta_list=None,
#                                 step=0.1
#                                 ):
#     if delta_theta_list is None:
#         delta_theta_list = [[5, 5], [10, 10], [15, 15]]
#     start, end = start_angle, end_angle
#
#     print(f'add k={k} angles signal data...')
#
#     theta_set = []
#     if delta_theta_list is not None:
#         for delta_theta in delta_theta_list:
#             assert len(delta_theta) == k - 1, f'dimensions of delta_theta_set don\'t match k={k}'
#
#     for delta_theta in delta_theta_list:
#         # list 转化为 np.array
#         delta_theta = np.array(delta_theta)
#
#         delta_theta = np.cumsum(np.array(delta_theta))
#         for theta in np.arange(start, end - delta_theta[-1], step=step):
#             theta = np.concatenate([theta[np.newaxis], theta + delta_theta], axis=0)
#             theta_set.append(theta)
#
#     return np.array(theta_set).astype(np.float32)
