import os
import pandas as pd
import matplotlib.pyplot as plt


def read_csv_results(snap, snr_list, csv_path):
    data = pd.read_csv(csv_path, header=0, index_col=0)

    # Find the row corresponding to the snap
    snap_row = data.loc[f'snap_{snap}']

    if not snap_row.empty:
        snr_columns = [f"snr_{snr}" for snr in snr_list]
        result_data = snap_row[snr_columns].values
        return result_data
    else:
        raise ValueError(f"Snap {snap} not found in the dataset.")


def load_all_test_result(root, snap, snr_list):
    result_data = {}  # model_name: np.ndarray 形式
    files = os.listdir(root)
    for file in files:
        if file.startswith('RMSE_mean') and file.endswith('.csv'):
            result_data['Ours'] = read_csv_results(snap, snr_list, os.path.join(root, file))  # load the model RMSE

    contrast_model_root = os.path.join(root, 'contrast_model')   # load the contrast model RMSE
    files = os.listdir(contrast_model_root)
    for file in files:
        if file.endswith('.csv'):
            model_name = file.split('_rmse_')[0]
            data = read_csv_results(snap, snr_list, os.path.join(contrast_model_root, file))
            result_data[model_name] = data
    return result_data


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
root1 = '/home/xd/DOA_code/open_source_code/results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/file_test1_random_input_rho1'
root2 = '/home/xd/DOA_code/open_source_code/results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample50/file_test1_random_input_rho1'


snap = 10
# 横轴数据 (SNR)
snr_values = [-20, -15, -10, -5, 0, 5]
scene1_data = load_all_test_result(root1, snap, snr_values)
scene2_data = load_all_test_result(root2, snap, snr_values)
scene1_data['DOA-ViT'] = scene1_data.pop('Ours')
scene2_data['DOA-ViT'] = scene2_data.pop('Ours')

# 定义样式
linestyles = {"scene1": "-", "scene2": "--"}
markers = ['>', 's', 'P', 'd', '<', 'D', 'x', 'o']
colors = [
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
# 绘图
plt.figure(figsize=(12, 7))
# plt.figure(figsize=(9, 5))
order = ['MUSIC', 'Unity-ESPRIT', r'$\ell_1$-SVD', 'SPE-CNN', 'ASL-2', 'SubspaceNet', r'Learning-SPICE', 'DOA-ViT']
scene1_data = {k: scene1_data[k] for k in order}
scene2_data = {k: scene2_data[k] for k in order}

for idx, algorithm in enumerate(scene1_data.keys()):
    # 从两组数据中提取
    scene1_results = scene1_data[algorithm]
    scene2_results = scene2_data[algorithm]

    # 获取样式
    marker = markers[idx % len(markers)]
    color = colors[idx % len(colors)]

    if algorithm == 'DOA-ViT':
        plt.plot(snr_values, scene2_results, linestyle=linestyles["scene2"], marker='o', color='red',
                 label=f"{algorithm}  num_sample=50", linewidth=2, markersize=6, markeredgewidth=1, markeredgecolor='black')
        plt.plot(snr_values, scene1_results, linestyle=linestyles["scene1"], marker='o', color='red',
                 label=f"{algorithm}  num_sample=500", linewidth=2, markersize=6, markeredgewidth=1, markeredgecolor='black')
    else:
        plt.plot(snr_values, scene1_results, linestyle=linestyles["scene1"], marker=marker, color=color,
                 label=f"{algorithm}",linewidth=2, markersize=6, markeredgewidth=1, markeredgecolor='black', alpha=0.7)
    # plt.plot(snap_values, scene1_results, linestyle=linestyles["scene1"], marker=marker, color=color,
    #          label=f"{algorithm} (scene1)")
    # # label=f"{algorithm} [10$^\circ$ 13$^\circ$ 16$^\circ$]"
    # plt.plot(snap_values, scene2_results, linestyle=linestyles["scene2"], marker=marker, color=color,
    #          label=f"{algorithm} (scene2)")
    # label=f"{algorithm} [10$^\circ$ 20$^\circ$ 30$^\circ$]
# 设置坐标轴刻度字体大小
plt.tick_params(axis="both", which="major", labelsize=14)  # 调整x和y轴刻度的字体大小
# 设置图例、标题和标签
plt.xlabel("SNR (dB)", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
plt.xlim([-20, 5])
plt.ylim([0, 40])
plt.title("Algorithm Performance Using Transfer Learning", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=14)
plt.tight_layout()

# 保存或显示图表
plt.savefig("/home/xd/DOA_code/open_source_code/results/vit/vit_M_8_k_3_transfer_learning_rho_1_mse+angle_sample500/algorithm_performance3.png", dpi=300, bbox_inches='tight')
plt.show()
