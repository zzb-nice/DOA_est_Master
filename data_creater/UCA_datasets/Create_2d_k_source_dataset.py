import warnings

import numpy as np
import itertools
import math
import time

delta_thetas_1 = [1, 2, 3, 4, 8, 12, 16, 20, 24, 30, 45, 80]


def Create_random_k_input_theta(k=3,
                                start_theta=-180, end_theta=180,
                                start_phi=0, end_phi=60,
                                theta_num=50000,
                                min_delta_theta=0.3):
    print(f'add k={k} angles signal data...')

    theta_set1 = []  # theta
    theta_set2 = []  # phi
    not_satisfied_num = 0

    if isinstance(start_theta, int):
        print(f'total {math.comb((end_theta - start_theta), k)} inputs if seperation == 1')
        for i in range(theta_num):  # create theta
            # !防止类型转换超过60度
            theta = np.random.uniform(start_theta, end_theta - np.finfo(np.float32).resolution, size=(k,))

            theta.sort()
            theta_diff_bool = np.diff(theta) > min_delta_theta
            # np.True is True : False
            while not np.all(theta_diff_bool):
                not_satisfied_num += 1

                theta = np.random.uniform(start_theta, end_theta - np.finfo(np.float32).resolution, size=(k,))
                theta.sort()
                theta_diff_bool = np.diff(theta) > min_delta_theta

            theta_set1.append(theta)

        for i in range(theta_num):  # create phi
            theta = np.random.uniform(start_phi, end_phi - np.finfo(np.float32).resolution, size=(k,))

            theta.sort()
            theta_diff_bool = np.diff(theta) > min_delta_theta
            # np.True is True : False
            while not np.all(theta_diff_bool):

                theta = np.random.uniform(start_phi, end_phi - np.finfo(np.float32).resolution, size=(k,))
                theta.sort()
                theta_diff_bool = np.diff(theta) > min_delta_theta

            theta_set2.append(np.random.permutation(theta))
    elif isinstance(start_theta, tuple):
        for i in range(theta_num):
            theta = np.random.uniform(start_theta, end_theta - np.finfo(np.float32).resolution, size=(k,))
            theta.sort()

            theta_diff_bool = np.diff(theta) > min_delta_theta
            # np.True is True : False
            while not np.all(theta_diff_bool):
                not_satisfied_num += 1

                theta = np.random.uniform(start_theta, end_theta - np.finfo(np.float32).resolution, size=(k,))
                theta.sort()
                theta_diff_bool = np.diff(theta) > min_delta_theta

            theta_set1.append(theta)

        for i in range(theta_num):
            theta = np.random.uniform(start_phi, end_phi - np.finfo(np.float32).resolution, size=(k,))
            theta.sort()

            theta_diff_bool = np.diff(theta) > min_delta_theta
            # np.True is True : False
            while not np.all(theta_diff_bool):

                theta = np.random.uniform(start_phi, end_phi - np.finfo(np.float32).resolution, size=(k,))
                theta.sort()
                theta_diff_bool = np.diff(theta) > min_delta_theta

            theta_set2.append(np.random.permutation(theta))

    print(f'{not_satisfied_num} samples not satisfied the limitation of sep')
    theta_set1 = np.array(theta_set1)
    theta_set2 = np.array(theta_set2)
    theta_set = np.stack([theta_set1, theta_set2], axis=1).astype(np.float32)
    return theta_set


# TODO： 暂时每个batch的snap一样，需要时修改
# Create dataset through input thetas
def Create_datasets(dataset, k, theta_set, batch_size, snap, snr, snr_set=1, **kwargs):
    # snap and snr can be set to int or tuple
    num_sample = theta_set.shape[0]
    theta_set = np.array_split(theta_set, theta_set.shape[0] // batch_size, 0)
    time_point_1 = time.time()
    if isinstance(snap, tuple) and len(snap) == 2:
        if isinstance(snr, tuple) and len(snr) == 2:
            for theta in theta_set:
                snr = np.random.uniform(snr[0], snr[1], size=(theta.shape[0], k))
                snap = int(np.random.randint(snap[0], snap[1] + 1, size=(1,))[0])
                dataset.Create_DOA_data(k, theta, snr, s_t_type='gauss_input', snap=snap, snr_set=snr_set, **kwargs)
        else:
            for theta in theta_set:
                snap = int(np.random.randint(snap[0], snap[1] + 1, size=(1,))[0])
                dataset.Create_DOA_data(k, theta, snr * np.ones((theta.shape[0], k)),
                                        s_t_type='gauss_input', snap=snap, snr_set=snr_set, **kwargs)
    else:
        if isinstance(snr, tuple) and len(snr) == 2:
            for theta in theta_set:
                snr = np.random.uniform(snr[0], snr[1], size=(theta.shape[0], k))
                dataset.Create_DOA_data(k, theta, snr, s_t_type='gauss_input', snap=snap, snr_set=snr_set, **kwargs)
        else:
            for theta in theta_set:
                dataset.Create_DOA_data(k, theta, snr * np.ones((theta.shape[0], k)),
                                        s_t_type='gauss_input', snap=snap, snr_set=snr_set, **kwargs)

    time_point_2 = time.time()
    print(f'time Consume:{time_point_2 - time_point_1}', end='  ,')
    print(f'{num_sample} data has been created')
    return 0
