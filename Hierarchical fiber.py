import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# 整体螺旋（大尺度）参数
R = 3.6     # 整体螺旋的半径
P = 3.512      # 整体螺旋的螺距

# 局部螺旋（小尺度）参数
r = 1.78       # 局部螺旋的半径

# 计算整体螺旋的角频率
L_g = np.sqrt((2 * np.pi * R)**2 + P**2)   # 整体螺旋每圈的长度
K = 2 * np.pi / L_g                        # 整体螺旋的角频率

# 定义局部螺旋每个整体螺旋圈数
local_turns_per_overall_turn = 20  # 设置每个整体螺旋包含的局部螺旋圈数

# 计算局部螺旋的角频率
k = local_turns_per_overall_turn * K

# 定义弧长参数 s
num_turns = 10                 # 绘制整体螺旋的圈数
s_max = L_g * num_turns        # 总弧长
num_points = 1500            # 链上点的数量
s = np.linspace(0, s_max, num_points)

# 计算位置向量
x = (R + r * np.cos(k * s)) * np.cos(K * s)
y = (R + r * np.cos(k * s)) * np.sin(K * s)
z = (P / (2 * np.pi)) * K * s + r * np.sin(k * s)

# 保存拟合的构象
posi = np.vstack([x, y, z]).T
np.savetxt("structure_fitting.xyz", posi)

# 创建图形，包含三个子图
fig = plt.figure(figsize=(18, 6))

# 绘制3D曲线
ax = fig.add_subplot(131, projection='3d')
ax.plot(x, y, z, lw=0.5, color='blue')
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.set_title('Hierarchical Fiber Structure', fontsize=15)
ax.set_box_aspect([np.ptp(i) for i in [x, y, z]])
ax.view_init(elev=30, azim=45)

# 计算所有点对之间的距离
coords = np.vstack((x, y, z)).T  # 将坐标组合成 (N, 3) 形式
dist_matrix = pdist(coords)      # 计算距离的一维数组

# 构建直方图
bins = 20000                       # 直方图的箱数
hist_range = (0, np.max(dist_matrix))
hist, bin_edges = np.histogram(dist_matrix, bins=bins, range=hist_range, density=False)

# 归一化处理
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
dr = bin_edges[1] - bin_edges[0]  # bin宽度
volume = (4/3) * np.pi * ((np.max(z) - np.min(z))/2)**3  # 近似链的体积
number_density = num_points / volume                     # 近似链的密度
ideal_g = 4 * np.pi * bin_centers**2 * dr * number_density  # 理想气体的对分布函数

# 避免除零错误
ideal_g[ideal_g == 0] = 1e-10

# 归一化实际的 g(r)
g_r = hist / ideal_g
g_r = g_r / np.max(g_r)  # 将 g(r) 归一化到最大值为1

# 绘制完整的 g(r)
ax2 = fig.add_subplot(132)
ax2.plot(bin_centers, g_r, color='red')
ax2.set_xlabel('Distance r', fontsize=12)
ax2.set_ylabel('Pair Distribution Function g(r)', fontsize=12)
ax2.set_title('Pair Distribution Function', fontsize=15)
ax2.grid(True)

# 绘制小 r 区域的 g(r)
ax3 = fig.add_subplot(133)
max_r = 5.0  # 设置小 r 的最大值，根据需要调整
indices = bin_centers <= max_r  # 获取对应于小 r 的数据索引
ax3.plot(bin_centers[indices], g_r[indices], color='green')
ax3.set_xlabel('Distance r', fontsize=12)
ax3.set_ylabel('Pair Distribution Function g(r)', fontsize=12)
ax3.set_title('g(r) at Small r', fontsize=15)
ax3.grid(True)

plt.tight_layout()
plt.show()
