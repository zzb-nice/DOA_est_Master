import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# flatten 然后直接画所有角度的误差
def plot_cdf(gt, pred, dir, num_points=1000):
    paint = np.abs(gt - pred).flatten
    # 设定均匀的 x 轴（根据数据的最小值和最大值来设定范围）
    x_values = np.linspace(np.min(paint), np.max(paint), num_points)

    # 计算每个 x 值对应的 CDF
    cdf_values = np.array([np.sum(paint <= x) / len(paint) for x in x_values])

    # 绘制 CDF 图像
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, cdf_values, marker='.', linestyle='-', color='blue')

    # 添加标签和标题
    plt.title('CDF of MAE')
    plt.xlabel('MAE')
    plt.ylabel('CDF')

    plt.grid(True)
    plt.savefig(dir)

    return 0


def plot_v_cdf(gt, pred, contrast_results, dir):
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # 设置颜色方案（与 plot_loss.py 保持一致）
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

    plt.figure(figsize=(8, 6))

    # 先画对比模型的结果
    for i, (model_name, contrast_result) in enumerate(contrast_results.items()):
        paint_i = np.abs(gt - contrast_result).flatten()
        paint_i = paint_i[~np.isnan(paint_i)]  # 去掉nan

        x_values = np.linspace(0, 60, 3001)
        cdf_values = np.array([np.sum(paint_i <= x) / len(paint_i) for x in x_values])

        plt.plot(x_values, cdf_values,
                 linestyle='-',
                 linewidth=1.5,
                 color=contrast_colors[i % len(contrast_colors)],
                 label=model_name)

    # 最后画我们的方法（红色）
    paint_i = np.abs(gt - pred).flatten()
    paint_i = paint_i[~np.isnan(paint_i)]  # 去掉nan
    cdf_values = np.array([np.sum(paint_i <= x) / len(paint_i) for x in x_values])
    plt.plot(x_values, cdf_values,
             linestyle='-',
             linewidth=2.0,
             color='red',
             label='Ours')

    # 添加标签和标题
    # plt.title('CDF of Mean Absolute Error (MAE)', fontsize=14)
    # plt.xlabel(r'Mean Absolute Error ($^\circ$)', fontsize=12)
    # 小图换大字体
    plt.xlabel('Absolute Error (°)', fontsize=18)
    plt.ylabel('CDF', fontsize=18)
    plt.legend(fontsize=18)
    plt.xlim([0, 60])
    plt.ylim([0, 1])

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(dir, dpi=300, bbox_inches='tight')
    plt.close()

    return 0


def calculate_cdf_and_quantiles(gt, pred, contrast_results, quantiles=None, cal_percent=None):
    if quantiles is None:
        quantiles = np.array([10, 50, 90])
    if cal_percent is None:
        cal_percent = np.array([1, 5, 10])
    # Calculate absolute errors for the main model
    absolute_errors = np.abs(gt - pred).flatten()
    absolute_errors = absolute_errors[~np.isnan(absolute_errors)]  # 去除预测的nan

    # Define x-axis range for the CDF plot
    x_values = np.linspace(0, 60, 3001)

    # Calculate the CDF for the main model
    cdf_values = np.array([np.sum(absolute_errors <= x) / len(absolute_errors) for x in x_values])

    # tore quantiles for the models
    model_quantile = []
    model_cal_percent = []
    contrast_quantile = {}
    contrast_cal_percent = {}

    # Calculate quantiles for the main model
    for quantile in quantiles:
        model_quantile.append(np.percentile(absolute_errors, quantile))
    for precent in cal_percent:
        model_cal_percent.append(cdf_values[np.where(x_values == precent)[0]])

    # Calculate for the contrast models
    for model_name, contrast_result in contrast_results.items():
        absolute_errors_contrast = np.abs(gt - contrast_result).flatten()
        absolute_errors_contrast = absolute_errors_contrast[~np.isnan(absolute_errors_contrast)]  # 去除预测的nan

        cdf_values_contrast = np.array(
            [np.sum(absolute_errors_contrast <= x) / len(absolute_errors_contrast) for x in x_values])

        contrast_quantile[model_name] = []
        contrast_cal_percent[model_name] = []
        # Calculate quantiles for each contrast model
        for quantile in quantiles:
            contrast_quantile[model_name].append(np.percentile(absolute_errors_contrast, quantile))
        for precent in cal_percent:
            contrast_cal_percent[model_name].append(cdf_values_contrast[np.where(x_values == precent)[0]])

    return model_quantile, contrast_quantile, model_cal_percent, contrast_cal_percent
