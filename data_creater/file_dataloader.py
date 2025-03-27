import numpy as np
import scipy
import random
import torch


# 拓展处理多个target的情况
class model_fit_type:
    def __init__(self, element1, element2):
        self._source = element1
        self._target = [element2] if isinstance(element2, str) else element2

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    def set_elements(self, element1, element2):
        self._source = element1
        self._target = element2


class file_array_Dataloader:
    def __init__(self, file_path, batch_size=8, shuffle=True, load_style='np', input_type='scm', output_type='doa'):
        """
        Args:
            file_path: 加载文件的路径
            batch_size: 批处理大小
            shuffle: 数据装载时是否随机打乱次序
            load_style: 'np' or 'torch',以np.ndarray或者torch形式装载
            input_type: 控制data_loader中x载入的数据
            output_type: 控制data_loader中y载入的数据
        """
        self.file_path = file_path
        self.all_data = np.load(file_path)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_data = self.all_data['doa'].shape[0]
        self.index = list(range(self.num_data))
        self.data_count = 0
        if self.shuffle:
            random.shuffle(self.index)

        self.load_style = load_style
        # set the type of x,y of dataloader
        self.data_type = model_fit_type(input_type, output_type)
        target_data = [self.all_data[target] for target in self.data_type.target]
        self.data = tuple(
            zip(self.all_data[self.data_type.source], *target_data))

    def __iter__(self):
        return self

    def __next__(self):
        if self.data_count >= self.num_data:
            self.data_count = 0
            if self.shuffle:
                random.shuffle(self.index)
            raise StopIteration
        else:
            batch_index = self.index[self.data_count:self.data_count + self.batch_size]
            data_batch = [self.data[i] for i in batch_index]
            # 两个返回值时self.dataset[i] 是 tuple 类型
            self.data_count += self.batch_size
            # print(type(data_batch))

            # 添加对batch的处理
            # 返回都是元组形式
            data_batch = tuple(zip(*data_batch))
            if len(data_batch) == 2:  # 单个输入输出
                input_data, labels = data_batch
                if self.load_style == 'np':
                    input_data = np.array(input_data)
                    labels = np.array(labels)
                elif self.load_style == 'torch':
                    input_data = torch.as_tensor(np.array(input_data))
                    labels = torch.as_tensor(np.array(labels))

                return input_data, labels
            else:
                input_data = data_batch[0]
                labels = data_batch[1:]
                if self.load_style == 'np':
                    input_data = np.array(input_data)
                    labels = [np.array(label) for label in labels]
                elif self.load_style == 'torch':
                    input_data = torch.as_tensor(np.array(input_data))
                    labels = [torch.as_tensor(np.array(label)) for label in labels]

                return input_data, *labels


def DictionaryToAttributes(cls, data):
    """
    Initialize the object and dynamically assign dictionary key-value pairs as attributes.

    Args:
        data (dict): A dictionary where keys become attribute names and values become their values.
    """
    # if not isinstance(data, dict):
    #     raise TypeError("Input must be a dictionary")

    for key, value in data.items():
        setattr(cls, key, value)


if __name__ == "__main__":
    file_path = "/home/xd/DOA_code/DOA_deep_learning/data/ULA_data/M_8_k_3_v_snr/train_dataset_snr_0.npz"

    dataloader = file_array_Dataloader(file_path, load_style='torch')
    for step, data_batch in enumerate(dataloader):
        scm, doa = data_batch
        pass



