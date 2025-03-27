import torch
import torch.nn as nn


class Grid_Based_network(nn.Module):
    def __init__(self, start_angle=-60, end_angle=60, step=0.1, threshold=0):
        nn.Module.__init__(self)
        # threshold 是寻峰函数中使用的阈值
        # super(Grid_Based_network, self).__init__()
        # ABC.__init__()
        # grid 存在gpu上
        # self._grid = np.arange(start_angle, end_angle + 0.0001, step=step)
        self.register_buffer('_grid', torch.arange(start_angle, end_angle + 0.0001, step=step))
        self._out_dim = self._grid.shape[0]
        self.threshold = threshold

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
        succ = sp_peak[:, k - 1].bool()
        theta = self.grid[idx[:, :k]]
        theta, _ = theta.sort(dim=-1)

        if return_sp:
            return succ, theta, sp_batch
        else:
            return succ, theta


class MLP(Grid_Based_network):
    def __init__(self, model_size, drop_out_ratio=0, sp_mode=False, **kwargs):
        out_dims = model_size[-1]
        if sp_mode:
            Grid_Based_network.__init__(self, **kwargs)
            assert self._grid.shape[0] == out_dims, 'error grid_size or model output dims'
            self.sp_to_doa = self.grid_to_theta  # 重命名 空间谱估计->角度的函数
        else:
            # super(VisionTransformer, self).__init__()
            nn.Module.__init__(self)

        layers = []

        for input_size, output_size in zip(model_size[:-2], model_size[1:-1]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(drop_out_ratio))
        layers.append(nn.Linear(model_size[-2], model_size[-1]))  # output layer 不需要bn,relu
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
