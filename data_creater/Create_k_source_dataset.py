import warnings

import numpy as np
import itertools
import math
import time

delta_thetas_1 = [1, 2, 3, 4, 8, 12, 16, 20, 24, 30, 45, 80]


def Create_determined_sep_doas(k=3,
                               start_angle=-60,
                               end_angle=60,
                               delta_thetas=None,
                               repeats=10,
                               theta_random=True,
                               step=0.1
                               ):
    """
    :param k: num of incident signals
    :param start_angle end_angle: the range of generated DOA values
    :param delta_thetas: determined intervals
                         'None': the default intervals
                         False: return all combination of possible angles
    :param repeats: multiple samples are generated for each DOA value,
                    Ensure an adequate number of samples for model training.
    :param theta_random: The randomness or not in generating DOAs
                         while theta_random is set to True,
                         the 'step' should be divisible by end_angle - start_angle and delta_thetas
    :param step: step for generate DOAs

    Return batch*k DOA values with determined intervals
    """
    if delta_thetas is None:
        delta_thetas = [1, 2, 3, 4, 8, 12, 16, 20, 24, 30, 45, 80]
    if theta_random is True and (end_angle - start_angle) % step != 0:
        warnings.warn('The step should be divisible by the end-start')

    print(f'add k={k} angles signal data...')

    # generate DOA values
    theta_set = []
    if delta_thetas is not False:
        # k-1 intervals for k DOAs of incident signals
        delta_thetas = [delta_thetas] * (k - 1)
        for delta_theta in itertools.product(*delta_thetas):
            delta_theta = np.array(delta_theta)
            cumsum_delta = np.cumsum(np.array(delta_theta))

            for theta in np.arange(start_angle, end_angle - cumsum_delta[-1], step=step):
                for i in range(repeats):
                    thetas = np.concatenate([theta[np.newaxis], theta + cumsum_delta], axis=0)
                    if theta_random:
                        thetas = thetas + step * np.random.rand(k)
                    theta_set.append(thetas)
    else:
        # all combinations of possible DOAs, the end is excluded
        theta_s = itertools.combinations(np.arange(start_angle, end_angle, step), k)
        for i in range(repeats):
            for theta in theta_s:
                theta_set.append(np.array(theta))

    count = len(theta_set)
    theta_set = np.array(theta_set).astype(np.float32)
    # Remove values that exceed the range due to truncation errors. /去掉因为截断误差而超出范围的值
    theta_set = np.minimum(theta_set, end_angle - np.finfo(np.float32).resolution)

    print(f'{count // repeats}*{repeats}={count}  data has been created')
    return theta_set


def Create_random_k_input_theta(k=3,
                                start_angle=-60, end_angle=60,
                                theta_num=50000,
                                min_delta_theta=0.3):
    print(f'add k={k} angles signal data...')

    theta_set = []
    not_satisfied_num = 0

    if isinstance(start_angle, int):
        # print(f'total {math.comb((end_angle - start_angle), k)} inputs if seperation == 1')
        for i in range(theta_num):
            # !防止类型转换超过60度
            theta = np.random.uniform(start_angle, end_angle - np.finfo(np.float32).resolution, size=(k,))

            theta.sort()
            theta_diff_bool = np.diff(theta) > min_delta_theta
            # np.True is True : False
            while not np.all(theta_diff_bool):
                not_satisfied_num += 1

                theta = np.random.uniform(start_angle, end_angle - np.finfo(np.float32).resolution, size=(k,))
                theta.sort()
                theta_diff_bool = np.diff(theta) > min_delta_theta

            theta_set.append(theta)
    elif isinstance(start_angle, tuple):
        for i in range(theta_num):
            theta = np.random.uniform(start_angle, end_angle - np.finfo(np.float32).resolution, size=(k,))
            theta.sort()

            theta_diff_bool = np.diff(theta) > min_delta_theta
            # np.True is True : False
            while not np.all(theta_diff_bool):
                not_satisfied_num += 1

                theta = np.random.uniform(start_angle, end_angle - np.finfo(np.float32).resolution, size=(k,))
                theta.sort()
                theta_diff_bool = np.diff(theta) > min_delta_theta

            theta_set.append(theta)

    print(f'{not_satisfied_num} samples not satisfied the limitation of sep')
    return np.array(theta_set).astype(np.float32)


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
