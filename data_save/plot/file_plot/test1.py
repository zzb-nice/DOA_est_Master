import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 创建示例数据
x = range(0, 100)
y = [i**2 for i in x]

# 绘制主图
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Main Plot')

# 创建插图坐标轴（局部放大）
inset_ax = inset_axes(ax, width="30%", height="30%", loc="upper right")

# 绘制局部区域放大图
inset_ax.plot(x, y)
inset_ax.set_xlim(10, 30)  # 放大 x 轴范围
inset_ax.set_ylim(100, 900)  # 放大 y 轴范围

# 在插图中添加矩形框（可选，指示被放大的区域）
from matplotlib.patches import Rectangle
ax.indicate_inset_zoom(inset_ax)

# 显示图形
plt.show()
