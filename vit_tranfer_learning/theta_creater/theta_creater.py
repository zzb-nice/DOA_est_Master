import numpy as np
from PIL import ImageFilter

import torch
from data_creater.signal_datasets import ULA_dataset
from data_creater.UCA_datasets.UCA_datasets import UCA_dataset


class same_data_Creater:
    def __init__(self, base_array=ULA_dataset(), data_type='scm'):
        # bound to judge theta is similar or discrepancy, set bound to min(theta2-theta1)/2 is better
        self.base_array = base_array
        self.data_type = data_type

    def __call__(self, theta, snap, snr):
        self.base_array.clear()
        # B*N维入射角输入
        assert theta.ndim == 2, 'error input'
        theta = theta.detach().cpu().numpy()

        self.base_array.Create_DOA_data(theta.shape[-1], theta, snrs_db=snr * np.ones_like(theta),
                                        s_t_type='gauss_input', snap=snap)

        retrieve_data = np.array(self.base_array.__dict__[self.data_type])
        retrieve_data = torch.tensor(retrieve_data)

        return retrieve_data


class same_data_Creater_2d:
    def __init__(self, base_array=UCA_dataset(), data_type='scm'):
        # bound to judge theta is similar or discrepancy, set bound to min(theta2-theta1)/2 is better
        self.base_array = base_array
        self.data_type = data_type

    def __call__(self, theta, snap, snr):
        self.base_array.clear()
        theta = theta.detach().cpu().numpy()

        self.base_array.Create_DOA_data(theta.shape[-1], theta, snrs_db=snr * np.ones((theta.shape[0],theta.shape[-1])),
                                        s_t_type='gauss_input', snap=snap)

        retrieve_data = np.array(self.base_array.__dict__[self.data_type])
        retrieve_data = torch.tensor(retrieve_data)

        return retrieve_data


class same_data_Creater_set_2:
    def __init__(self, base_array=ULA_dataset(), data_type='scm'):
        # bound to judge theta is similar or discrepancy, set bound to min(theta2-theta1)/2 is better
        self.base_array = base_array
        self.data_type = data_type

    def __call__(self, theta, snap, snr, **kwargs):
        self.base_array.clear()
        # theta: B*N
        assert theta.ndim == 2, 'error input'
        theta = theta.detach().cpu().numpy()

        self.base_array.Create_DOA_data(theta.shape[-1], theta, snrs_db=snr * np.ones_like(theta),
                                        s_t_type='gauss_input', snap=snap, **kwargs)

        retrieve_data = np.array(self.base_array.__dict__[self.data_type])
        retrieve_data = torch.tensor(retrieve_data)

        return retrieve_data
