import torch.nn as nn

from models.dl_model.grid_based_network import Grid_Based_network


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
