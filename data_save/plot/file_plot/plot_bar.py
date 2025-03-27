import matplotlib.pyplot as plt
import numpy as np
import os

from utils.util import load_all_test_result,load_all_test_result_snap

root = '/home/xd/DOA_code/open_source_code/data_save/vit/file_test1_random_input_M_16_k_3'
snap = 10
# 横轴数据 (SNR)
snr_values = [-15, -10, -5, 0, 5]
data = load_all_test_result(root, snap, snr_values)

order = ['DOA-ViT', 'Unity-ESPRIT', r'$\ell_1$-SVD', 'MUSIC', 'SPE-CNN', r'Learning-SPICE']
data = {k: data[k] for k in order}

# Colors for the bars
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
colors = ['#0072B2', '#E69F00', '#56B4E9', '#F0E442', '#009E73', '#CC79A7', '#D55E00']
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#AF7AA1', '#17becf']
# colors = ['#AEC6CF', '#FFB347', '#77DD77', '#FF6961', '#CFCFC4', '#F49AC2', '#B39EB5']
# colors = [
#     'r',
#     '#bcbd22',  # 黄绿色
#     '#2ca02c',  # 绿色
#     '#ff7f0e',  # 橙色
#     '#9467bd',  # 紫色
#     '#8c564b',  # 棕色
#     '#e377c2',  # 粉色
#     '#1f77b4',  # 蓝色
#     '#17becf',  # 青色
#     '#aec7e8',  # 浅蓝色
#     '#98df8a',  # 浅绿色
#     '#ff9896',  # 浅红色
# ]


plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
width = 0.14  # Width of each bar
x = np.arange(len(snr_values))  # Positions for groups of bars

# Plot each method's data
for i, (method, values) in enumerate(data.items()):
    ax.bar(
        x + i * width,
        values,
        width,
        label=method,
        color=colors[i % len(colors)],
        edgecolor='black',  # Add black borders to bars
        zorder=3
    )

# Add labels, title, and legend
ax.set_xlabel("SNR (dB)", fontsize=14)
ax.set_ylabel("RMSE", fontsize=14)
ax.set_title("Comparison of RMSE Across Methods", fontsize=14)
ax.set_xticks(x + width * (len(data) - 1) / 2)  # Center x-ticks
ax.set_xticklabels(snr_values)
ax.legend(loc="upper right", fontsize=10)

# Add gridlines for readability
ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(root,'bar_results.png'), dpi=300, bbox_inches='tight')
plt.show()
