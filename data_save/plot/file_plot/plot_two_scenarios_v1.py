import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.util import load_all_test_result, load_all_test_result_snap

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# root1 = '/home/xd/DOA_code/DOA_deep_learning/data_save/vit/v_snap/file_test1_monte_carlo[5.1, 30.3, 54.5]_mode2'
# root2 = '/home/xd/DOA_code/DOA_deep_learning/data_save/vit/v_snap/file_test1_monte_carlo[-9.9, 15.3, 39.5]_mode2'
root1 = '/home/xd/DOA_code/open_source_code/data_save/vit/file_test1_monte_carlo[10,13,16]_rho0'
root2 = '/home/xd/DOA_code/open_source_code/data_save/vit/file_test1_monte_carlo[10,20,30]_rho0'

snap = 10
# 横轴数据 (SNR)
snr_values = [-20, -15, -10, -5, 0, 5]
scene1_data = load_all_test_result(root1, snap, snr_values)
scene2_data = load_all_test_result(root2, snap, snr_values)
order = ['DOA-ViT', 'MUSIC', r'Learning-SPICE', 'Unity-ESPRIT', r'$\ell_1$-SVD']
scene1_data = {k: scene1_data[k] for k in order}
scene2_data = {k: scene2_data[k] for k in order}
# snr = -10
# snap_values = [1, 5, 10, 30, 50, 100]
# scene1_data = load_all_test_result_snap(root1, snr, snap_values)
# scene2_data = load_all_test_result_snap(root2, snr, snap_values)

# 定义样式
linestyles = {"scene1": "--", "scene2": "-"}
markers = ["s", "o", "^", "d", "x", "*", "v", "<", ">"]
colors = ["r", "g", "b", "c", "m", "y", "k"]

# 绘图
plt.figure(figsize=(12, 7))
# plt.figure(figsize=(9, 5))

for idx, algorithm in enumerate(scene1_data.keys()):
    # 从两组数据中提取
    scene1_results = scene1_data[algorithm]
    scene2_results = scene2_data[algorithm]
    
    # 获取样式
    marker = markers[idx % len(markers)]
    color = colors[idx % len(colors)]
    
    # 绘制场景 1 和场景 2
    if algorithm == 'Ours' or algorithm == 'DOA-ViT':  # ours 用红色,单独绘制
        plt.plot(snr_values, scene1_results, linestyle=linestyles["scene1"], marker=marker, color=color, linewidth=2.0,
                 label=f"{algorithm} (Scene 1)", zorder=10)
        plt.plot(snr_values, scene2_results, linestyle=linestyles["scene2"], marker=marker, color=color, linewidth=2.0,
                 label=f"{algorithm} (Scene 2)", zorder=10)
    else:
        plt.plot(snr_values, scene1_results, linestyle=linestyles["scene1"], marker=marker, color=color,
                 label=f"{algorithm} (Scene 1)", alpha=0.7)
        plt.plot(snr_values, scene2_results, linestyle=linestyles["scene2"], marker=marker, color=color,
                 label=f"{algorithm} (Scene 2)", alpha=0.7)

plt.xlabel("SNR (dB)", fontsize=18)
# plt.xlabel("Sanps", fontsize=12)
plt.ylabel("RMSE", fontsize=18)
plt.xlim([-20, 5])
plt.ylim([0, 45])
# plt.xticks([1, 5, 10, 30, 50, 100])
plt.xticks([-20, -15, -10, -5, 0, 5])
plt.title("Algorithm Performance Across Two Scenes", fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

plt.savefig(os.path.join("/home/xd/DOA_code/open_source_code/data_save/vit/", "algorithm_performance4.png"), dpi=300)
plt.show()
