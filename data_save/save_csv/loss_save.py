import numpy as np
import pandas as pd
import os


def save_array(ori_array: np.ndarray, dir, header, index):
    # assert dims of ori_array==2
    if ori_array.ndim == 1:
        ori_array = np.expand_dims(ori_array, 0)
    pd_data = pd.DataFrame(ori_array, index=index, columns=header)
    pd_data.to_csv(dir, header=True, index=True)

    return 0
