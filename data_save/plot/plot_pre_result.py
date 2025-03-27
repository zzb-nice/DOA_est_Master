import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import os

import pandas as pd


def plot_predict_result(gt, pred, dir):
    """
    plot predict results for multiple targets DOA estimation
    :param gt: shape:(num_samples, num_target)
    :param pred: shape:(num_samples, num_target)
    :param dir: the path to save the figure
    """
    # be compatible with list of np.ndarray
    gt = np.array(gt)
    assert gt.ndim == 2, 'error input size'
    assert gt.shape == pred.shape, 'error input size'

    # build x_axis : num_samples, num_target
    x_axis = np.arange(gt.shape[0])

    fig, ax = plt.subplots(figsize=(14, 6))

    # 同一个目标角度用同一个颜色表示
    parm_cycle = plt.rcParams["axes.prop_cycle"]()
    for k in range(gt.shape[1]):
        color = next(parm_cycle)['color']
        ax.plot(x_axis, np.array(gt)[:, k], color=color, linewidth=4.0)
        ax.plot(x_axis, pred[:, k], 'o', color=color)

    ax.grid()

    ax.set(xlim=(x_axis[0], x_axis[-1]),
           ylim=(-75, 75),
           xlabel='items',
           ylabel='predict result',
           title='model predict results'
           )

    plt.tight_layout()
    fig.savefig(dir, dpi=300, bbox_inches='tight')
    fig.clf()

    return 0


def plot_error(gt, pred, dir):
    """
    plot predict results for multiple targets DOA estimation
    :param gt: shape:(num_samples, num_target)
    :param pred: shape:(num_samples, num_target)
    :param dir: the path to save the figure
    """

    # be compatible with list of np.ndarray
    gt = np.array(gt)
    assert gt.ndim == 2, 'error input size'
    assert gt.shape == pred.shape, 'error input size'

    # build x_axis : num_samples, num_target
    x_axis = np.arange(gt.shape[0])

    fig, ax = plt.subplots(figsize=(14, 6))

    paint = pred - gt
    ax.plot(x_axis, np.zeros_like(x_axis), 'k', linewidth=2.0)
    # 同一个目标角度用同一个颜色表示
    for k in range(gt.shape[1]):
        ax.plot(x_axis, paint[:, k], 'o')

    ax.grid()
    # axes[j].legend()

    ax.set(xlim=(x_axis[0], x_axis[-1]),
           ylim=(-75, 75),
           xlabel='items',
           ylabel='predict error',
           title='model predict errors'
           )

    plt.tight_layout()
    fig.savefig(dir, dpi=300, bbox_inches='tight')
    fig.clf()

    return 0


# plot the results of different algorithms using multiple subplots
def plot_v_predict_result(gt, pred, contrast_results, dir):
    """
    plot predict results for multiple targets DOA estimation
    :param gt: shape:(num_samples, num_target)
    :param pred: shape:(num_samples, num_target)
    contrast_results: a dict of predict results, the key is the name of the model
    :param dir: the path to save the figure
    """
    x_axis = range(len(gt))
    num_plots = len(contrast_results) + 1
    max_cols = 5  # 每行最多5个子图
    num_rows = (num_plots - 1) // max_cols + 1
    
    # 计算最后一行的子图数量
    last_row_cols = num_plots - (num_rows - 1) * max_cols
    
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(max_cols * 6, num_rows * 6), 
                            sharex=True, sharey=True)
    
    # 确保axes是二维数组
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    j = 0
    parm_cycle = plt.rcParams["axes.prop_cycle"]()
    
    # 绘制我们的方法
    for k in range(gt.shape[1]):
        color = next(parm_cycle)['color']
        axes[0][0].plot(x_axis, np.array(gt)[:, k], color=color, linewidth=4.0)
        axes[0][0].plot(x_axis, pred[:, k], 'o', color=color, alpha=0.3)
    axes[0][0].set_title('Ours',fontsize=20)
    axes[0][0].grid()
    
    # 绘制对比方法
    row = 0
    col = 1
    for model_name, contrast_result in contrast_results.items():
        parm_cycle = plt.rcParams["axes.prop_cycle"]()
        for k in range(gt.shape[1]):
            color = next(parm_cycle)['color']
            axes[row][col].plot(x_axis, np.array(gt)[:, k], color=color, linewidth=4.0)
            axes[row][col].plot(x_axis, contrast_result[:, k], 'o', color=color, alpha=0.3)
        axes[row][col].set_title(model_name,fontsize=20)
        axes[row][col].grid()
        
        col += 1
        if col >= max_cols:
            col = 0
            row += 1
    
    # 删除多余的子图
    if last_row_cols < max_cols:
        for col in range(last_row_cols, max_cols):
            axes[num_rows-1][col].remove()
    
    axes[0][0].set(xlim=(x_axis[0], x_axis[-1]),
                   ylim=(-75, 75),
                   # xlabel='items',
                   ylabel='predict result')
    
    plt.tight_layout()
    fig.savefig(dir)
    fig.clf()
    
    return 0


def plot_v_error(gt, pred, contrast_results, dir):
    """
    plot predict results for multiple targets DOA estimation
    :param gt: shape:(num_samples, num_target)
    :param pred: shape:(num_samples, num_target)
    contrast_results: a dict of predict results, the key is the name of the model
    :param dir: the path to save the figure
    """

    # be compatible with list of np.ndarray
    gt = np.array(gt)
    assert gt.ndim == 2, 'error input size'
    assert gt.shape == pred.shape, 'error input size'

    # build x_axis : num_samples, num_target
    x_axis = np.arange(gt.shape[0])

    # num_plots : 需要绘制的情景数
    num_plots = len(contrast_results) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6), sharex=True, sharey=True)

    j = 0
    paint = pred - gt

    axes[j].plot(x_axis, np.zeros_like(x_axis), 'k', linewidth=2.0)

    for k in range(gt.shape[1]):
        axes[j].plot(x_axis, paint[:, k], 'o')
    axes[j].set_title('Ours')
    axes[j].grid()

    for model_name, contrast_result in contrast_results.items():
        j += 1
        paint = contrast_result - gt

        axes[j].plot(x_axis, np.zeros_like(x_axis), 'k', linewidth=2.0)
        for k in range(gt.shape[1]):
            axes[j].plot(x_axis, paint[:, k], 'o')

        axes[j].set_title(model_name)
        axes[j].grid()

    axes[0].set(xlim=(x_axis[0], x_axis[-1]),
                ylim=(-75, 75),
                xlabel='items',
                ylabel='predict errors'
                )
    plt.tight_layout()
    fig.savefig(dir)
    fig.clf()

    return 0


# 气泡图
def plot_doa_error(gt, pred, contrast_results, dir):
    """
    plot predict results for multiple targets DOA estimation
    :param gt: shape:(num_samples, num_target)
    :param pred: shape:(num_samples, num_target)
    contrast_results: a dict of predict results, the key is the name of the model
    :param dir: the path to save the figure
    """
    # be compatible with list of np.ndarray
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    gt = np.array(gt)
    assert gt.ndim == 2, 'error input size'
    assert gt.shape == pred.shape, 'error input size'

    # build x_axis : num_samples, num_target
    sample_indices = np.arange(gt.shape[0])
    k = gt.shape[-1]

    # num_plots : 需要绘制的情景数
    num_plots = len(contrast_results) + 1

    for theta_i in range(k):
        # 设置图表大小
        fig, axes = plt.subplots(num_plots, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [3, 1]})
        fig.subplots_adjust(hspace=0.5)  # 设置行间距

        axes[0, 0].scatter(sample_indices, gt[:, theta_i], label='True', facecolors='none', edgecolor='blue', s=30,
                           marker='o', linewidth=2, alpha=0.3)
        axes[0, 0].scatter(sample_indices, pred[:, theta_i], label='Proposed', color='#900C3F', s=10, marker='.')
        error = np.abs(np.array(gt[:, theta_i]) - np.array(pred[:, theta_i]))
        axes[0, 0].set_ylabel("DOA(°)")
        axes[0, 0].set_xlabel("Sample index")
        axes[0, 0].set_xlim([0, gt.shape[0]])

        axes[0, 1].plot(sample_indices, error, color='#900C3F', marker='.', linestyle='-', markersize=4)
        axes[0, 1].set_ylabel("Error(°)")
        axes[0, 1].set_xlabel("Sample index")
        axes[0, 1].set_xlim([0, gt.shape[0]])
        handles = [
            plt.Line2D([], [], color='#900C3F', marker='o', markersize=5, label='True', markerfacecolor='white',
                       markeredgewidth=1, markeredgecolor='blue', linestyle='None', linewidth=2, alpha=0.3),
            plt.Line2D([], [], color='#900C3F', marker='.', markersize=5, label='Proposed'),
        ]
        j = 0
        parm_cycle = plt.rcParams["axes.prop_cycle"]()
        for model_name, contrast_result in contrast_results.items():
            j += 1
            color = next(parm_cycle)['color']
            axes[j, 0].scatter(sample_indices, gt[:, theta_i], label='True', facecolors='none', edgecolor='blue', s=30,
                               marker='o', linewidth=2, alpha=0.3)
            axes[j, 0].scatter(sample_indices, contrast_result[:, theta_i], label=model_name, color=color,
                               s=10, marker='.')

            axes[j, 0].set_ylabel("DOA(°)")
            axes[j, 0].set_xlabel("Sample index")
            axes[j, 0].set_xlim([0, gt.shape[0]])
            if j == 0:
                axes[j, 0].legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.3), fontsize=8)

            # 计算误差并绘制Error图
            error = np.abs(np.array(gt[:, theta_i]) - np.array(contrast_result[:, theta_i]))
            axes[j, 1].plot(sample_indices, error, color=color, marker='.', linestyle='-', markersize=4)
            axes[j, 1].set_ylabel("Error(°)")
            axes[j, 1].set_xlabel("Sample index")
            axes[j, 1].set_xlim([0, gt.shape[0]])
            # 添加上方的图例
            handles.append(plt.Line2D([], [], color=color, marker='.', markersize=5, label=model_name))

        fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.95),
                   title="Methods")
        # fig.legend(handles=handles, loc='upper center', ncol=len(handles), fontsize=10, frameon=True,
        #            bbox_to_anchor=(0.5, 0.95), title="Methods", columnspacing=1,
        #            borderpad=1, labelspacing=1)
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        name, ext = os.path.splitext(dir)
        fig.savefig(os.path.join(dir, f'{name}_{theta_i}{ext}'), dpi=300, bbox_inches='tight')
        fig.clf()


# def plot_with_invalid():
#
#
#
# def scatter_with_invalid():
