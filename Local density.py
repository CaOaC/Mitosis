import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# 定义读取文件数据的函数，并加入偏移量
def load_data_with_offset(filenames, offsets):
    all_data = []
    for i, file in enumerate(filenames):
        data = np.loadtxt(file, usecols=(2, 3, 4))  # 读取2,3,4列为粒子位置
        data[:, 0] += offsets[i]  # 只在x方向进行平移
        all_data.append(data)
    return np.vstack(all_data)  # 将所有数据拼接在一起

# 定义局部密度计算函数
def calculate_local_density(positions, radius):
    tree = cKDTree(positions)  # 使用KDTree加速邻近搜索
    densities = []
    volume = (4/3) * np.pi * (radius**3)  #小球体积
    for pos in positions:
        indices = tree.query_ball_point(pos, r=radius)  # 找到半径r内的粒子
        num_particles = len(indices)  # 邻近粒子的数量
        density = num_particles / volume  # 数密度
        densities.append(density)
    return np.array(densities)

# 文件名列表
filenames = [
    'kappa/traj_0.0_0.0.txt',
    'kappa/traj_5.0_0.0.txt',
    'kappa/traj_0.0_5.0.txt',
    'kappa/traj_5.0_5.0.txt'
]

# 对应每个文件的偏移量，使它们沿X轴并列放置
offsets = [0, 200, 350, 480]  # 可根据需要调整偏移量

# 读取所有文件中的数据，并进行偏移
data = load_data_with_offset(filenames, offsets)

# 计算局部密度，设定球形半径
radius = 5.0  # 可以根据需要调整
densities = calculate_local_density(data, radius)

# 绘制3D散点图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图，使用局部密度作为颜色，数据点为不同文件的数据
sc = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=densities, cmap='jet', marker='o')

# 添加颜色条
cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
cbar.set_label('Local Number Density', fontsize=25)
cbar.ax.tick_params(labelsize=20)

# 移除坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# 设置轴标签（可以选择保留或者注释掉）
ax.set_xlabel('X', fontsize=25)
ax.set_ylabel('Y', fontsize=25)
ax.set_zlabel('Z', fontsize=25)

plt.show()
