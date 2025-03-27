import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re
import os

import pandas as pd

import random  # 定义随机生成颜色函数


def loss_1d_plot(loss_list, tag_list, x_axis, x_axis_label, logy: bool, dir):
    """
    Paint loss figure 根据多组target_loss 和 对比模型的 contrast loss

    Args:
        loss_list: 绘制和模型相关的loss信息,np.ndarray or list of np.ndarray
        tag_list: loss信息对应的图例
        x_axis: x轴数字
        x_axis_label: x轴标签，x_axis_label = 'snr' or 'snap'
        logy: set y as log axis or not
        dir: direction to save the figure

    Returns:
        保留模型和对比模型的结果
    """
    # 设置字体
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=(14, 6))
    # 字体等图像设置
    plt.rcParams.update({'font.size': 14})

    # 兼容 np.ndarray or list of np.ndarray
    if isinstance(loss_list, list):
        for i, loss in enumerate(loss_list):
            ax.plot(x_axis, loss, '-o', label=tag_list[i], linewidth=2)
    else:
        ax.plot(x_axis, loss_list, '-o', label=tag_list, linewidth=2)

    # set y as log axis
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(x_axis_label, fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(0, 45)

    ax.xaxis.set_major_locator(ticker.FixedLocator(x_axis))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5 / 3))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    ax.grid()
    ax.legend()

    fig.savefig(dir, dpi=300, bbox_inches='tight')
    fig.clf()

    return 0


def loss_1d_v_plot(loss_list, tag_list, x_axis, x_axis_label, contrast_model_loss, logy: bool, dir):
    """
    Paint loss figure 根据多组target_loss 和 对比模型的 contrast loss

    Args:
        loss_list: 绘制和模型相关的loss信息,np.ndarray or list of np.ndarray
        tag_list: loss信息对应的图例
        x_axis: x轴数字
        x_axis_label: x轴标签,x_axis_label = 'snr' or 'snap'
        contrast_model_loss: dict of contrast model loss
        logy: set y as log axis or not
        dir: direction to save the figure

    Returns:
        保留模型和对比模型的结果
    """
    # 设置字体
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 设置颜色方案（对比算法的颜色）
    contrast_colors = [
        '#bcbd22',  # 黄绿色
        '#2ca02c',  # 绿色
        '#ff7f0e',  # 橙色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#1f77b4',  # 蓝色
        '#17becf',  # 青色
        '#aec7e8',  # 浅蓝色
        '#98df8a',  # 浅绿色
        '#ff9896',  # 浅红色
    ]
    linestyles = '-.'
    markers = ['>', 's', 'P', 'd', '<', 'D', 'x', 'o']
    markeredgecolor = 'black'

    # fig, ax = plt.subplots(figsize=(12, 6))
    # fig, ax = plt.subplots(figsize=(12, 7))
    fig, ax = plt.subplots(figsize=(9, 5))
    # 字体等图像设置
    plt.rcParams.update({'font.size': 14})

    # 先画对比模型的结果
    for i, (model_name, contrast_loss) in enumerate(contrast_model_loss.items()):
        ax.plot(x_axis, contrast_loss, linestyle='-', marker=markers[i % len(contrast_colors)], label=model_name,
                color=contrast_colors[i % len(contrast_colors)],
                linewidth=2, markersize=6, markeredgewidth=1, markeredgecolor='black', alpha=0.7)

    # 最后画我们的模型（红色）
    if isinstance(loss_list, list):
        for i, loss in enumerate(loss_list):
            ax.plot(x_axis, loss, linestyle='-', marker='o', label=tag_list[i],
                    color='red', linewidth=2, markersize=6, markeredgewidth=1, markeredgecolor='black')
    else:
        ax.plot(x_axis, loss_list, linestyle='-', marker='o', label=tag_list,
                color='red', linewidth=2, markersize=6, markeredgewidth=1, markeredgecolor='black')

    # set y as log axis
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(x_axis_label, fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(0, 45)
    # ax.set_title("Algorithm Performance with Randomly Generated Incident Angles", fontsize=14)

    ax.xaxis.set_major_locator(ticker.FixedLocator(x_axis))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5 / 3))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

    fig.savefig(dir, dpi=300, bbox_inches='tight')
    fig.clf()

    return 0


def succ_1d_v_plot(succ_list, tag_list, x_axis, x_axis_label, contrast_model_succ, logy: bool, dir):
    """
    Paint loss figure 根据多组target_loss 和 对比模型的 contrast loss

    Args:
        succ_list: 绘制和模型相关的loss信息,np.ndarray or list of np.ndarray
        tag_list: loss信息对应的图例
        x_axis: x轴数字
        x_axis_label: x轴标签,x_axis_label = 'snr' or 'snap'
        contrast_model_succ: dict of contrast model loss
        logy: set y as log axis or not
        dir: direction to save the figure

    Returns:
        保留模型和对比模型的结果
    """
    # 设置字体
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 设置颜色方案（对比算法的颜色）
    contrast_colors = [
        '#bcbd22',  # 黄绿色
        '#2ca02c',  # 绿色
        '#ff7f0e',  # 橙色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉色
        '#1f77b4',  # 蓝色
        '#17becf',  # 青色
        '#aec7e8',  # 浅蓝色
        '#98df8a',  # 浅绿色
        '#ff9896',  # 浅红色
    ]

    # fig, ax = plt.subplots(figsize=(12, 6))
    # fig, ax = plt.subplots(figsize=(14, 6))
    fig, ax = plt.subplots(figsize=(9, 5))
    # 字体等图像设置
    plt.rcParams.update({'font.size': 14})

    # 先画对比模型的结果
    for i, (model_name, contrast_succ) in enumerate(contrast_model_succ.items()):
        ax.plot(x_axis, contrast_succ, '-o', label=model_name,
                color=contrast_colors[i % len(contrast_colors)],
                linewidth=1, markersize=6)

    # 最后画我们的模型（红色）
    if isinstance(succ_list, list):
        for i, loss in enumerate(succ_list):
            ax.plot(x_axis, loss, '-o', label=tag_list[i],
                    color='red', linewidth=1, markersize=8)
    else:
        ax.plot(x_axis, succ_list, '-o', label=tag_list,
                color='red', linewidth=1, markersize=8)

    # set y as log axis
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(x_axis_label, fontsize=14)
    ax.set_ylabel('Success Ratio', fontsize=14)
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(0, 1.02)

    ax.xaxis.set_major_locator(ticker.FixedLocator(x_axis))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(2.5 / 3))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)

    fig.savefig(dir, dpi=300, bbox_inches='tight')
    fig.clf()

    return 0


def loss_1d_plot_std(loss_mean, loss_std, tag_list, x_axis, x_axis_label, logy: bool, dir):
    """
    Paint loss figure with mean and standard deviation

    Args:
        loss_mean: 均值，np.ndarray or list of np.ndarray
        loss_std: 标准差，np.ndarray or list of np.ndarray
        tag_list: loss信息对应的图例
        x_axis: x轴数字
        x_axis_label: x轴标签，x_axis_label = 'snr' or 'snap'
        logy: set y as log axis or not
        dir: direction to save the figure

    Returns:
        保留模型和对比模型的结果
    """
    # 设置字体
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=(14, 6))
    # 字体等图像设置
    plt.rcParams.update({'font.size': 14})

    # 兼容 np.ndarray or list of np.ndarray
    if isinstance(loss_mean, list):
        for i, (mean, std) in enumerate(zip(loss_mean, loss_std)):
            line = ax.plot(x_axis, mean, '-o', label=tag_list[i], linewidth=2)
            color = line[0].get_color()
            # 添加标准差区域
            ax.fill_between(x_axis,
                            mean - std,
                            mean + std,
                            alpha=0.2,  # 透明度
                            color=color)
    else:
        line = ax.plot(x_axis, loss_mean, '-o', label=tag_list, linewidth=1)
        color = line[0].get_color()
        # 添加标准差区域
        ax.fill_between(x_axis,
                        loss_mean - loss_std,
                        loss_mean + loss_std,
                        alpha=0.2,
                        color='r')

    # set y as log axis
    if logy:
        ax.set_yscale('log')

    ax.set_xlabel(x_axis_label, fontsize=14)
    ax.set_ylabel('RMSE', fontsize=14)
    ax.set_xlim(x_axis[0], x_axis[-1])
    ax.set_ylim(0, 45)

    ax.xaxis.set_major_locator(ticker.FixedLocator(x_axis))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)

    fig.savefig(dir, dpi=300, bbox_inches='tight')
    fig.clf()

    return 0
