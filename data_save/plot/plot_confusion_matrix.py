import os.path

import matplotlib.pyplot as plt
import numpy as np
import itertools

from prettytable import PrettyTable


# 给协方差矩阵画图    plt.show()
# 协方差矩阵x,y轴数量相同,用同一个indices
def plot_confusion_matrix(matrix: np.ndarray, save_dir=None):
    plt.figure()
    # 根据每个协方差的模值来决定颜色深浅
    weight = np.abs(matrix).astype(np.float32)
    # weight = (matrix * matrix.conj()).astype(np.float32)
    plt.imshow(weight.transpose(), cmap=plt.cm.Blues)  # ?

    # indices = range(len(matrix))
    # labels = [f'阵元{i}' for i in indices]
    # labels = [f'Array {i}' for i in indices]
    # 设置x轴坐标label
    # plt.xticks(indices, labels, rotation=45)
    # 设置y轴坐标label
    # plt.yticks(indices, labels)

    # 显示colorbar
    # plt.colorbar()
    # plt.title('Covariance matrix')

    # # 根据weight[x,y]确定字体,打印matrix[x,y]
    # thresh = np.max(weight) / 2
    # # 限制打印精度
    # np.set_printoptions(precision=3)
    # for x in indices:
    #     for y in indices:
    #         # num = f'{matrix[x, y]:.1f}'
    #         # 先乘10打印,后续再调整
    #         num = f'{10 * matrix[x, y]:.0f}'
    #         plt.text(y, x, num,
    #                  verticalalignment='center',
    #                  horizontalalignment='center',
    #                  color='white' if weight[x, y] > thresh else 'black'
    #                  )

    plt.tight_layout()
    # 不显示网格线
    plt.grid(False)
    if save_dir is None:
        plt.show()
        input()
    else:
        plt.savefig(save_dir, dpi=300, bbox_inches='tight', transparent=True)
        # plt.clf()
        plt.close()
    return 0


def plot_contrast(matrix: np.ndarray,save_dir=None):
    # to_show = np.arange(25).reshape(5, 5)  打的居然是反的
    # 根据每个协方差的模值来决定颜色深浅
    # weight = (matrix * matrix.conj()).real.astype(np.float32)
    weight = matrix.astype(np.float32)
    plt.imshow(weight, cmap=plt.cm.Blues)  # ,cmap=plt.cm.Blues

    indices_1, indices_2 = range(matrix.shape[0]), range(matrix.shape[1])

    # labels = [f'阵元{i}' for i in indices]
    labels_1 = [f'snap_{i}' for i in indices_1]
    labels_2 = [f'snr_{i}' for i in indices_2]
    # 设置y轴坐标label
    plt.yticks(indices_1, labels_1)
    # 设置x轴坐标label
    plt.xticks(indices_2, labels_2, rotation=45)

    # 显示colorbar
    plt.colorbar()
    plt.title('data contrast')

    # 根据weight[x,y]确定字体,打印matrix[x,y]
    thresh = np.max(weight) / 2
    # 限制打印精度
    np.set_printoptions(precision=3)
    for x in indices_1:  # snap
        for y in indices_2:  # snr
            # num = f'{matrix[x, y]:.1f}'
            # 先乘10打印,后续再调整
            # num = f'{10 * matrix[x, y]:.0f}'
            num = f'{matrix[x, y]:.3f}'
            plt.text(y, x, num,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color='white' if weight[x, y] > thresh else 'black'
                     )

    # plt.text(0, 1, '***',verticalalignment='center',horizontalalignment='center',color='white')
    plt.tight_layout()
    # 不显示网格线
    plt.grid(False)
    if save_dir is None:
        plt.show()
        input()
    else:
        plt.savefig(save_dir)
    return 0


# 面对对象的api不知道是哪些
def plot_confusion_matrix_(matrix: np.ndarray):
    fig = plt.figure()
    ax = plt.axes()

    # 根据每个协方差的模值来决定颜色深浅
    weight = (matrix * matrix.conj()).astype(np.float32)
    ax.imshow(weight)  # ,cmap=plt.cm.Blues

    indices = range(len(matrix))
    labels = [f'阵元{i}' for i in indices]
    # 设置x轴坐标label
    ax.set_xticks(indices, labels, rotation=45)
    # 设置y轴坐标label
    ax.set_yticks(indices, labels)

    # 显示colorbar
    # ax.colorbar()
    ax.set_title('Covariance matrix')

    # 根据weight[x,y]确定字体,打印matrix[x,y]
    thresh = np.max(weight) / 2
    for x in indices:
        for y in indices:
            num = matrix[x, y]
            ax.text(x, y, num,
                    verticalalignment='center',
                    horizontalalignment='center',
                    color='white' if weight[x, y] > thresh else 'black'
                    )

    # ax.tight_layout()
    fig.show()

    input()
