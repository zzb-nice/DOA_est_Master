import copy

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_half_radar(theta, data1, data2, labels=("SubNet+MVDR", "MVDR"), title="narrowband & coherent, $M=3$"):
    """
    绘制半圆雷达图（-90°到90°），对比两组数据。

    Parameters:
    - theta: list or array, 数据的角度 (rad)。
    - data1: list or array, 第一组数据的幅值。
    - data2: list or array, 第二组数据的幅值。
    - labels: tuple, 数据的标签，用于图例。
    - title: str, 图表标题。

    Returns:
    - None
    """
    # 检查输入数据长度是否一致
    if len(theta) != len(data1) or len(theta) != len(data2):
        raise ValueError("Input theta, data1, and data2 must have the same length.")

    # 转换为弧度并限制范围到 [-90°, 90°]
    theta_deg = np.rad2deg(theta)  # 转换为角度，便于设置刻度
    theta_rad = np.deg2rad(theta_deg)  # 确保数据格式正确

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})

    # 设置半圆范围（-90°到90°）
    ax.set_theta_zero_location('N')  # 0°在顶部
    ax.set_theta_direction(-1)  # 顺时针方向
    ax.set_thetalim(-np.pi / 2, np.pi / 2)  # 设置显示角度范围 [-90°, 90°]

    # 绘制两组数据
    ax.plot(theta_rad, data1, label=labels[0], color='purple', linestyle='-', linewidth=2)
    ax.plot(theta_rad, data2, label=labels[1], color='orange', linestyle='-', linewidth=2)

    # 增加箭头标记（可选）
    for angle, value in zip(theta_rad, data1):
        ax.annotate("", xy=(angle, value), xytext=(angle, 0),
                    arrowprops=dict(arrowstyle="->", color='purple', lw=1))

    # 设置刻度
    ax.set_xticks(np.linspace(-np.pi / 2, np.pi / 2, 7))  # 等间距设置角度刻度
    ax.set_xticklabels([' -90°', '-60°', '-30°', '0°', '30°', '60°', '90°'])

    # 图例和标题
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.set_title(title, va='bottom', fontsize=12)

    # 美化图表
    ax.grid(True)

    # 显示图表
    plt.show()


def plot_radar_fig(gt, pred, contrast_results, dir):
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    gt = copy.deepcopy(gt)
    pred = copy.deepcopy(pred)
    contrast_results = copy.deepcopy(contrast_results)

    gt[:, 0, :] = gt[:, 0, :]/180*np.pi
    pred[:, 0, :] = pred[:, 0, :]/180*np.pi
    for model_name, contrast_result in contrast_results.items():
        contrast_results[model_name][:, 0, :] = contrast_results[model_name][:, 0, :]/180*np.pi
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # 设置半圆范围（-90°到90°）
    ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°', '0°'])

    for model_name, contrast_result in contrast_results.items():
        parm_cycle = plt.rcParams["axes.prop_cycle"]()
        for k in range(gt.shape[-1]):
            color = next(parm_cycle)['color']
            if k == 0:
                ax.plot(contrast_result[:, 0, k], contrast_result[:, 1, k], linestyle='', marker='.', color=color,
                        label=model_name, alpha=0.3)
                ax.plot(gt[:, 0, k], gt[:, 1, k], linestyle='', marker='+', markersize=15, color=color,
                        label='Positive Marker')
            else:
                ax.plot(contrast_result[:, 0, k], contrast_result[:, 1, k], linestyle='', marker='.', color=color, alpha=0.3)
                ax.plot(gt[:, 0, k], gt[:, 1, k], linestyle='', marker='+', markersize=15, color=color)

    # 图例和标题
    ax.legend(loc="upper right")
    ax.set_title('2D DOA estimation', fontsize=16)

    # 美化图表
    ax.grid(True)

    # 显示图表
    # plt.show()
    fig.savefig(dir.split('.')[0]+'_music.png', dpi=300, bbox_inches='tight')
    fig.clf()

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

    # 设置半圆范围（-90°到90°）
    ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
    ax.set_xticklabels(['0°', '45°', '90°', '135°', '±180°', '-135°', '-90°', '-45°', '0°'])

    parm_cycle = plt.rcParams["axes.prop_cycle"]()
    for k in range(gt.shape[-1]):
        color = next(parm_cycle)['color']
        if k == 0:  # set one label
            ax.plot(pred[:, 0, k], pred[:, 1, k], linestyle='', marker='.', color=color, label='Ours', alpha=0.3)
            ax.plot(gt[:, 0, k], gt[:, 1, k], linestyle='', marker='+', markersize=15, color=color, label='Positive Marker')
        else:
            ax.plot(pred[:, 0, k], pred[:, 1, k], linestyle='', marker='.', color=color, alpha=0.3)
            ax.plot(gt[:, 0, k], gt[:, 1, k], linestyle='', marker='+', markersize=15, color=color)
    # # 增加箭头标记（可选）
    # for angle, value in zip(theta_rad, data1):
    #     ax.annotate("", xy=(angle, value), xytext=(angle, 0),
    #                 arrowprops=dict(arrowstyle="->", color='purple', lw=1))

    # 图例和标题
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.legend(loc="upper right")
    ax.set_title('2D DOA estimation', fontsize=16)

    # 美化图表
    ax.grid(True)

    # 显示图表
    # plt.show()
    fig.savefig(dir, dpi=300, bbox_inches='tight')

    return 0


# def plot_radar_fig(gt, pred, contrast_results, dir):
#     plt.rcParams['font.family'] = 'DeJavu Serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
#
#     # 创建图形
#     fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
#
#     # 设置半圆范围（-90°到90°）
#     ax.set_xticks(np.linspace(0, 2 * np.pi, 9))
#     ax.set_xticklabels(['-90°', '-45°', '0°', '45°', '90°', '135°', '±180°', '-135°', '-90°'])
#
#     for model_name, contrast_result in contrast_results.items():
#         parm_cycle = plt.rcParams["axes.prop_cycle"]()
#         for k in range(gt.shape[-1]):
#             color = next(parm_cycle)['color']
#             if k == 0:
#                 ax.plot(contrast_result[:, 0, k], contrast_result[:, 1, k], linestyle='', marker='.', color=color,
#                         label=model_name)
#             else:
#                 ax.plot(contrast_result[:, 0, k], contrast_result[:, 1, k], linestyle='', marker='.', color=color)
#
#     parm_cycle = plt.rcParams["axes.prop_cycle"]()
#     for k in range(gt.shape[-1]):
#         color = next(parm_cycle)['color']
#         if k == 0:  # set one label
#             ax.plot(pred[:, 0, k], pred[:, 1, k], linestyle='', marker='.', color=color, label='Ours')
#             ax.plot(gt[:, 0, k], gt[:, 1, k], linestyle='', marker='+', markersize=15, color=color, label='Positive Marker')
#         else:
#             ax.plot(pred[:, 0, k], pred[:, 1, k], linestyle='', marker='.', color=color)
#             ax.plot(gt[:, 0, k], gt[:, 1, k], linestyle='', marker='+', markersize=15, color=color)
#     # # 增加箭头标记（可选）
#     # for angle, value in zip(theta_rad, data1):
#     #     ax.annotate("", xy=(angle, value), xytext=(angle, 0),
#     #                 arrowprops=dict(arrowstyle="->", color='purple', lw=1))
#
#     # 图例和标题
#     ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
#     ax.set_title('Radar Fig', fontsize=16)
#
#     # 美化图表
#     ax.grid(True)
#
#     # 显示图表
#     # plt.show()
#     fig.savefig(dir, dpi=300, bbox_inches='tight')
#
#     return 0


if __name__ == '__main__':
    # 示例数据
    theta_example = np.linspace(-np.pi / 2, np.pi / 2, 100)  # 从 -90° 到 90°
    data1_example = np.abs(np.sin(theta_example)) * 0.8  # 示例曲线1
    data2_example = np.abs(np.cos(theta_example)) * 0.9  # 示例曲线2

    # 绘制
    plot_half_radar(theta_example, data1_example, data2_example)
