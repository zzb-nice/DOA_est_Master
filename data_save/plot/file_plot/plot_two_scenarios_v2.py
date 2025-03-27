import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from utils.util import load_all_test_result, load_all_test_result_snap

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

root1 = '/home/xd/DOA_code/open_source_code/data_save/vit/v_snap/file_test1_monte_carlo[5.1,30.3,54.5]_origin'
root2 = '/home/xd/DOA_code/open_source_code/data_save/vit/v_snap/file_test1_monte_carlo[-9.9,15.3,39.5]_origin'

# snap = 10
# # 横轴数据 (SNR)
# snr_values = [-20, -15, -10, -5, 0, 5]
# scene1_data = load_all_test_result(root1, snap, snr_values)
# scene2_data = load_all_test_result(root2, snap, snr_values)

snr = -10
snap_values = [1, 5, 10, 30, 50, 100]
scene1_data = load_all_test_result_snap(root1, snr, snap_values)
scene2_data = load_all_test_result_snap(root2, snr, snap_values)
order = ['DOA-ViT', 'MUSIC', r'Learning-SPICE', 'Unity-ESPRIT', r'$\ell_1$-SVD']
scene1_data = {k: scene1_data[k] for k in order}
scene2_data = {k: scene2_data[k] for k in order}
# 定义样式
linestyles = {"scene1": "--", "scene2": "-"}
markers = ["s", "o", "^", "d", "x", "*", "v", "<", ">"]
colors = ["r", "b", "g", "c", "m", "y", "k"]

# 绘图
fig = plt.figure(figsize=(12, 7))
ax = plt.gca()

for idx, algorithm in enumerate(scene1_data.keys()):
    # 从两组数据中提取
    scene1_results = scene1_data[algorithm]
    scene2_results = scene2_data[algorithm]
    
    # 获取样式
    marker = markers[idx % len(markers)]
    color = colors[idx % len(colors)]
    
    # 绘制场景 1 和场景 2
    if algorithm == 'Ours' or algorithm == 'DOA-ViT':  # ours 用红色,单独绘制
        plt.plot(snap_values, scene1_results, linestyle=linestyles["scene1"], marker=marker, color=color, linewidth=2.0,
                 label=f"{algorithm} (Scene 1)", zorder=10)
        plt.plot(snap_values, scene2_results, linestyle=linestyles["scene2"], marker=marker, color=color, linewidth=2.0,
                 label=f"{algorithm} (Scene 2)", zorder=10)
    else:
        plt.plot(snap_values, scene1_results, linestyle=linestyles["scene1"], marker=marker, color=color,
                 label=f"{algorithm} (Scene 1)")
        plt.plot(snap_values, scene2_results, linestyle=linestyles["scene2"], marker=marker, color=color,
                 label=f"{algorithm} (Scene 2)")

# plt.xlabel("SNR [dB]", fontsize=12)
plt.xlabel("Snaps", fontsize=18)
plt.ylabel("RMSE", fontsize=18)
plt.xlim([1, 100])
plt.ylim([0, 45])
plt.xticks([1, 5, 10, 30, 50, 100])
plt.title("Algorithm Performance Across Two Scenes", fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=10)

# 添加放大区域（在所有绘图完成后添加）
axins = inset_axes(ax, width="40%", height="40%", loc="upper center")

# 复制主图内容到放大区域
for line in ax.get_lines():
    axins.plot(line.get_xdata(), line.get_ydata(), 
               linestyle=line.get_linestyle(),
               marker=line.get_marker(),
               color=line.get_color(),
               linewidth=line.get_linewidth(),
               zorder=line.get_zorder())

# 设置放大区域
axins.set_xlim(28, 52)
axins.set_ylim(6, 20)
axins.grid(True, linestyle="--", alpha=0.7)

# 修改连接框的样式，添加linestyle参数设置虚线
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="black", alpha=0.5, lw=2, ls="-.")


plt.tight_layout()
plt.savefig(os.path.join("/home/xd/DOA_code/open_source_code/data_save/vit/v_snap", "algorithm_performance3.png"), dpi=300)
plt.show()
